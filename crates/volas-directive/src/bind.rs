//! Bind a parsed directive ([`Cst`]) to its executable form ([`Ast`]).
//!
//! This is the single place a directive is *validated*, and it happens **once**
//! (at parse / cache time), not on every execution. For each command it walks the
//! spec argument list one time and, per position, either parses-and-checks the
//! supplied token or substitutes the spec default — validation, defaulting, and
//! type conversion are the same single pass (see [`crate::spec::ArgBound::bind`]),
//! so there is no separate "validate then re-read" stage and the default value
//! lives only in the spec. The resulting [`Ast`] carries canonical names, ready typed
//! [`ArgValue`]s and each command's [`CommandSpec`], so `exec` / `lookback` /
//! `stringify` never re-parse, re-default, re-validate or look the line up again.
//!
//! Binding has three forms: [`bind_node`] (frame-blind: a bare name stays a name),
//! [`bind_frame`] (against a frame's data columns: a bare name becomes the column or the
//! command it names, with directive-shaped data references kept as data) and [`bind_form`] (lenient on an absent
//! argument or operand, for `directive_stringify`).

use crate::spec::{
    canon_sub, command_spec, is_command, normalize, CommandSpec, HV_SESSION_MINUTES,
    HV_TRADING_DAYS,
};
use crate::types::{ArgValue, Ast, Bound, BoundNode, Command, Cst, Form, FormArgs, Node};
use volas_core::{Result, VolasError};
use volas_time::Span;
use std::borrow::Cow;

/// Lower a concrete syntax tree to a bound AST. A bare [`Node::Name`] is left
/// unbound — without a frame it cannot be told apart from a column reference; see
/// [`bind_frame`].
pub(crate) fn bind_node(cst: &Cst) -> Result<BoundNode> {
    Ok(match cst {
        Node::Data(s) => Node::Data(s.clone()),
        Node::Name(s) => Node::Name(s.clone()),
        Node::Scalar(v) => Node::Scalar(*v),
        Node::Command(c) => {
            Node::Command(bind_command_node(&c.name, c.sub.as_deref(), &c.args, &c.series)?)
        }
        Node::Unary { op, operand } => Node::Unary {
            op: *op,
            operand: Box::new(bind_node(operand)?),
        },
        Node::Binary { left, op, right } => Node::Binary {
            left: Box::new(bind_node(left)?),
            op: *op,
            right: Box::new(bind_node(right)?),
        },
    })
}

/// The spec of `name` / `sub` (canonical), or the error naming why there is none.
fn spec_of(name: &str, sub: Option<&str>) -> Result<CommandSpec> {
    match command_spec(name, sub) {
        Some(spec) => Ok(spec),
        None if is_command(name) => Err(VolasError::Value(match sub {
            Some(s) => format!("command \"{name}\" has no sub-command \"{s}\""),
            None => format!("command \"{name}\" requires a sub-command"),
        })),
        None => Err(VolasError::Value(format!("unknown command \"{name}\""))),
    }
}

/// `Err` when `name` is given more arguments or operands than its spec has slots.
fn check_counts(name: &str, spec: &CommandSpec, args: usize, series: usize) -> Result<()> {
    if args > spec.args.len() {
        return Err(VolasError::Value(format!(
            "command \"{name}\" accepts at most {} argument(s), got {args}",
            spec.args.len()
        )));
    }
    if series > spec.series.len() {
        return Err(VolasError::Value(format!(
            "command \"{name}\" accepts at most {} series operand(s), got {series}",
            spec.series.len()
        )));
    }
    Ok(())
}

/// Bind the supplied token `s` of slot `i` against its bound.
fn bind_arg(name: &str, spec: &CommandSpec, i: usize, s: &str) -> Result<ArgValue> {
    spec.args[i]
        .bound
        .bind(s)
        .map_err(|why| VolasError::Value(format!("{name}: argument #{i} {why}, got '{s}'")))
}

/// Whether an operand slot is filled: an absent slot or an empty name leaves it to the
/// command's default column.
fn filled<A>(series: &[Node<A>], i: usize) -> bool {
    series.get(i).is_some_and(|n| !matches!(n, Node::Name(s) if s.is_empty()))
}

/// Validate and resolve one command into its bound form: canonical name / sub,
/// arguments parsed-checked-defaulted in a single pass, operands lowered, the line's
/// spec attached. The one boundary where a bad configuration errors loudly (a zero
/// window, an out-of-domain seed, a NaN multiplier, inverted `mavp` limits, a missing
/// required argument or operand, more operands than the line reads, an unknown
/// command / sub) instead of becoming a valid-shaped column with no signal (V17).
pub(crate) fn bind_command_node(
    name: &str,
    sub: Option<&str>,
    args: &[Option<String>],
    series: &[Cst],
) -> Result<Command<Bound>> {
    let name = normalize(name);
    let name = name.as_ref();
    let sub = canon_sub(name, sub);
    let sub = sub.as_deref();
    let spec = spec_of(name, sub)?;
    check_counts(name, &spec, args.len(), series.len())?;

    // One pass over the spec: parse-and-check a supplied token, or take the
    // (valid-by-construction) default; a missing required argument errors.
    let mut bound = Vec::with_capacity(spec.args.len());
    for (i, arg) in spec.args.iter().enumerate() {
        let value = match args.get(i).and_then(|o| o.as_deref()) {
            Some(s) => bind_arg(name, &spec, i, s)?,
            None => arg
                .default
                .value()
                .ok_or_else(|| VolasError::Value(format!("{name}: argument #{i} is required")))?,
        };
        bound.push(value);
    }
    bind_cross_args(name, &bound)?;
    if let Some(i) = (0..spec.series.len()).find(|&i| spec.series[i].is_empty() && !filled(series, i)) {
        return Err(VolasError::Value(format!("{name}: series argument #{i} is required")));
    }

    let series = series.iter().map(bind_node).collect::<Result<Vec<_>>>()?;
    Ok(Command {
        name: name.to_string(),
        sub: sub.map(str::to_string),
        args: Bound { values: bound.into_boxed_slice(), spec, bare: None },
        series,
    })
}

/// Bind a command's raw tokens and operands into an opaque executable directive.
pub fn bind_command(
    name: &str,
    sub: Option<&str>,
    args: &[Option<String>],
    series: &[Cst],
) -> Result<Ast> {
    bind_command_node(name, sub, args, series).map(|cmd| finish(Node::Command(cmd)))
}

/// The command a bare `name` runs as: every argument at its default, the spelling kept.
pub(crate) fn bind_bare(name: &str) -> Result<Command<Bound>> {
    let mut cmd = bind_command_node(name, None, &[], &[])?;
    cmd.args.bare = Some(name.into());
    Ok(cmd)
}

/// Whether `name` written bare (no colon, no arguments) is a built-in command it can
/// run as: every argument has a default and every input a default column (`repeat`,
/// whose input is a required `@` operand, cannot run bare).
pub fn is_bare_command(name: &str) -> bool {
    command_spec(&normalize(name), None).is_some_and(|spec| spec.usable_bare())
}

/// Canonical directive spelling of a physical data-column label. Invalid directive
/// labels retain their ordinary exact-name semantics at the caller's lookup boundary.
/// Unchanged identifiers borrow the schema label; only a changed spelling or an
/// expression's rendered identity needs owned storage.
#[inline]
pub fn canonical_data_name(text: &str) -> Result<Cow<'_, str>> {
    // Ordinary schema labels need neither a parsed tree nor its continuation facts.
    // Keep the renderer's bare-name rule: only commands usable bare normalize.
    let mut bytes = text.bytes();
    if bytes.next().is_some_and(crate::parser::is_ident_start)
        && bytes.all(crate::parser::is_ident_char)
    {
        // Lowercase identifiers already render identically, whether or not they
        // are commands. Consult the spec only if normalization changes a label.
        let normalized = normalize(text);
        return Ok(if normalized != text && is_bare_command(&normalized) {
            normalized
        } else {
            Cow::Borrowed(text)
        });
    }
    crate::parse(text).map(|node| Cow::Owned(crate::stringify::stringify_node(node.node())))
}

/// Bind to data columns named by `is_column`. A data column owns every canonical
/// spelling of its directive, including references nested inside expressions.
pub fn bind_frame(node: &Ast, is_column: &impl Fn(&str) -> bool, directive: &str) -> Result<Ast> {
    bind_frame_resolved(node, &|name| is_column(name).then(|| name.to_string()), directive)
}

/// Bind canonical directive references to their physical data-column names. All
/// expression levels use this one resolver; a resolved data node never becomes a
/// command again when continuation or lookback facts are selected.
pub fn bind_frame_resolved(node: &Ast, resolve: &impl Fn(&str) -> Option<String>, _directive: &str) -> Result<Ast> {
    bind_frame_node(node.node(), resolve).map(finish)
}

/// Seal a validated tree together with its continuation decision.
pub(crate) fn finish(node: BoundNode) -> Ast { Ast::bound(node) }

fn bind_frame_node(node: &BoundNode, resolve: &impl Fn(&str) -> Option<String>) -> Result<BoundNode> {
    let canonical = crate::stringify::stringify_node(node);
    if !canonical.is_empty() && !matches!(node, Node::Scalar(_)) {
        if let Some(name) = resolve(&canonical) { return Ok(Node::Data(name)); }
    }
    Ok(match node {
        Node::Name(name) if name.is_empty() => node.clone(),
        Node::Name(name) if is_command(&normalize(name)) => Node::Command(bind_bare(name)?),
        Node::Data(_) | Node::Name(_) | Node::Scalar(_) => node.clone(),
        Node::Command(c) => Node::Command(Command {
            name: c.name.clone(),
            sub: c.sub.clone(),
            args: c.args.clone(),
            series: c
                .series
                .iter()
                .map(|s| bind_frame_node(s, resolve))
                .collect::<Result<_>>()?,
        }),
        Node::Unary { op, operand } => Node::Unary {
            op: *op,
            operand: Box::new(bind_frame_node(operand, resolve)?),
        },
        Node::Binary { left, op, right } => Node::Binary {
            left: Box::new(bind_frame_node(left, resolve)?),
            op: *op,
            right: Box::new(bind_frame_node(right, resolve)?),
        },
    })
}

/// Whether the frame-bound `node` uses a built-in command through a bare name
/// whose canonical command name is `name`. Command spelling does not affect its
/// identity; the prospective data column's `name` stays case-sensitive.
pub fn uses_bare_command(node: &Ast, name: &str) -> bool {
    uses_bare_node(node.node(), name)
}

fn uses_bare_node(node: &BoundNode, name: &str) -> bool {
    match node {
        Node::Data(_) | Node::Name(_) | Node::Scalar(_) => false,
        Node::Command(c) => {
            (c.args.bare.is_some() && c.name == name) || c.series.iter().any(|s| uses_bare_node(s, name))
        }
        Node::Unary { operand, .. } => uses_bare_node(operand, name),
        Node::Binary { left, right, .. } => uses_bare_node(left, name) || uses_bare_node(right, name),
    }
}

/// Whether adding data under canonical `name` would shadow a computation inside an
/// already-bound cached directive. Its stored Data references remain ordinary reads;
/// command and formula references retain their computation until explicitly dropped.
pub fn uses_command_shadow(node: &Ast, name: &str) -> bool {
    fn reads(node: &BoundNode, name: &str) -> bool {
        match node {
            Node::Data(_) | Node::Name(_) | Node::Scalar(_) => false,
            Node::Command(c) => crate::stringify::stringify_node(node) == name || c.series.iter().any(|s| reads(s, name)),
            Node::Unary { operand, .. } => crate::stringify::stringify_node(node) == name || reads(operand, name),
            Node::Binary { left, right, .. } => crate::stringify::stringify_node(node) == name || reads(left, name) || reads(right, name),
        }
    }
    reads(node.node(), name)
}

/// The bare-name rule for `directive_lookback`: without a frame a bare name is taken
/// as a command, so a known command must bind exactly as written — one with no sub-less
/// form (`kdj`) or a missing required argument (`ma`) is rejected, as `df[d]` rejects it
/// at execute, instead of passing as a 0-lookback column. An unknown bare name is an
/// unverifiable column reference and passes.
fn check_bare_command(name: &str) -> Result<()> {
    if is_command(&normalize(name)) {
        bind_command_node(name, None, &[], &[])?;
    }
    Ok(())
}

/// Walk a bound [`Ast`] applying only the bare-name rule — every `Command` was
/// already validated when `bind` produced the tree. Used by `directive_lookback`.
pub fn check_bare(node: &Ast) -> Result<()> { check_bare_node(node.node()) }

pub(crate) fn check_bare_node(node: &BoundNode) -> Result<()> {
    match node {
        Node::Data(_) => Ok(()),
        Node::Name(name) => check_bare_command(name),
        Node::Scalar(_) => Ok(()),
        Node::Command(c) => c.series.iter().try_for_each(check_bare_node),
        Node::Unary { operand, .. } => check_bare_node(operand),
        Node::Binary { left, right, .. } => {
            check_bare_node(left)?;
            check_bare_node(right)
        }
    }
}

/// Bind a raw [`Cst`] as a directive *form*, for `directive_stringify`. Like `bind` it
/// rejects an unknown command / sub-command, too many arguments or operands, an
/// out-of-domain *supplied* argument, a bad cross-argument rule, and a bare
/// sub-requiring name — but it is lenient on an *absent required* argument or operand,
/// since a form such as `donchian.upper` names a real indicator whose period is
/// supplied only at use. Every supplied argument is bound, so the form renders exactly
/// as the column the directive caches under.
pub fn bind_form(cst: &Cst) -> Result<Form> {
    Ok(match cst {
        Node::Data(name) => Node::Data(name.clone()),
        Node::Name(name) => {
            let canonical = normalize(name);
            if is_command(&canonical) && command_spec(&canonical, None).is_none() {
                return Err(VolasError::Value(format!("command \"{canonical}\" requires a sub-command")));
            }
            Node::Name(name.clone())
        }
        Node::Scalar(v) => Node::Scalar(*v),
        Node::Command(c) => {
            let name = normalize(&c.name);
            let sub = canon_sub(&name, c.sub.as_deref());
            let spec = spec_of(&name, sub.as_deref())?;
            check_counts(&name, &spec, c.args.len(), c.series.len())?;
            let values = (0..spec.args.len())
                .map(|i| match c.args.get(i).and_then(|o| o.as_deref()) {
                    Some(s) => bind_arg(&name, &spec, i, s).map(Some),
                    None => Ok(None),
                })
                .collect::<Result<Vec<_>>>()?;
            check_form_cross(&name, &spec, &values)?;
            Node::Command(Command {
                name: name.to_string(),
                sub: sub.map(|s| s.into_owned()),
                args: FormArgs { values, spec },
                series: c.series.iter().map(bind_form).collect::<Result<_>>()?,
            })
        }
        Node::Unary { op, operand } => Node::Unary { op: *op, operand: Box::new(bind_form(operand)?) },
        Node::Binary { left, op, right } => Node::Binary {
            left: Box::new(bind_form(left)?),
            op: *op,
            right: Box::new(bind_form(right)?),
        },
    })
}

/// Cross-argument rules over a form's arguments: an absent slot takes its default, and
/// a rule an absent required slot leaves unknown is skipped.
fn check_form_cross(name: &str, spec: &CommandSpec, values: &[Option<ArgValue>]) -> Result<()> {
    let slot = |i: usize| values[i].clone().or_else(|| spec.args[i].default.value());
    match name {
        // hv's time frame / trading days / session all have defaults: the rule runs on
        // every form
        "hv" => {
            let known = |i| slot(i).expect("hv's time frame, trading days and session have defaults");
            check_hv_args(&known(1), &known(2), &known(3))
        }
        "mavp" => match (slot(0), slot(1)) {
            (Some(lo), Some(hi)) => check_mavp_limits(&lo, &hi),
            _ => Ok(()),
        },
        _ => Ok(()),
    }
}

/// Cross-argument domain rules — a constraint *between* two resolved arguments
/// that no single-argument bound can express.
fn bind_cross_args(name: &str, args: &[ArgValue]) -> Result<()> {
    match name {
        "hv" => check_hv_args(&args[1], &args[2], &args[3]),
        "mavp" => check_mavp_limits(&args[0], &args[1]),
        _ => Ok(()),
    }
}

/// `mavp`'s limits: each per-bar period is clamped into `[min_period, max_period]`;
/// inverted limits are meaningless (and panic `i64::clamp`). Both are bound `>= 1`.
fn check_mavp_limits(min_p: &ArgValue, max_p: &ArgValue) -> Result<()> {
    let (min_p, max_p) = (min_p.as_usize(), max_p.as_usize());
    if min_p > max_p {
        return Err(VolasError::Value(format!(
            "mavp: min_period ({min_p}) must be <= max_period ({max_p})"
        )));
    }
    Ok(())
}

/// `hv`'s rules between its time frame and its annualization inputs (the formula is
/// `exec::hv_annualization`): a sub-day bar must fit in the trading session, and an
/// argument the frame does not use — `session_minutes` beyond sub-day bars,
/// `trading_days` beyond day bars — must keep its default rather than be silently
/// ignored.
fn check_hv_args(time_frame: &ArgValue, trading_days: &ArgValue, session: &ArgValue) -> Result<()> {
    let tf = time_frame.as_time_frame();
    let (days, session) = (trading_days.as_i64(), session.as_usize());
    let label = tf.label();
    let why = match tf.span() {
        Span::Seconds(s) if s > session as i64 * 60 => {
            format!("a {label} bar is longer than the {session}-minute session")
        }
        Span::Seconds(_) => return Ok(()),
        _ if session != HV_SESSION_MINUTES => format!(
            "session_minutes applies to sub-day time frames only, got {session} with {label}"
        ),
        Span::Days(_) => return Ok(()),
        _ if days != HV_TRADING_DAYS => format!(
            "trading_days applies to sub-day and day time frames only, got {days} with {label}"
        ),
        _ => return Ok(()),
    };
    Err(VolasError::Value(format!("hv: {why}")))
}

#[cfg(test)]
mod tests;
