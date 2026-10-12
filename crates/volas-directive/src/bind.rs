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
//! command it names, or is rejected as ambiguous) and [`bind_form`] (lenient on an absent
//! argument or operand, for `directive_stringify`).

use crate::spec::{
    canon_sub, command_spec, is_command, normalize, CommandSpec, HV_SESSION_MINUTES,
    HV_TRADING_DAYS,
};
use crate::types::{ArgValue, Ast, Bound, Command, Cst, Form, FormArgs, Node};
use volas_core::{Result, VolasError};
use volas_time::Span;

/// Lower a concrete syntax tree to a bound AST. A bare [`Node::Name`] is left
/// unbound — without a frame it cannot be told apart from a column reference; see
/// [`bind_frame`].
pub fn bind_node(cst: &Cst) -> Result<Ast> {
    Ok(match cst {
        Node::Name(s) => Node::Name(s.clone()),
        Node::Scalar(v) => Node::Scalar(*v),
        Node::Command(c) => {
            Node::Command(bind_command(&c.name, c.sub.as_deref(), &c.args, &c.series)?)
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
pub fn bind_command(
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

/// The command a bare `name` runs as: every argument at its default, the spelling kept.
pub(crate) fn bind_bare(name: &str) -> Result<Command<Bound>> {
    let mut cmd = bind_command(name, None, &[], &[])?;
    cmd.args.bare = Some(name.into());
    Ok(cmd)
}

/// Whether `name` written bare (no colon, no arguments) is a built-in command it can
/// run as: every argument has a default and every input a default column (`repeat`,
/// whose input is a required `@` operand, cannot run bare).
pub fn is_bare_command(name: &str) -> bool {
    command_spec(&normalize(name), None).is_some_and(|spec| spec.usable_bare())
}

/// The R17 error for a bare `name` in `directive` that is both a data column and a
/// built-in command usable bare.
fn ambiguous(name: &str, directive: &str) -> VolasError {
    VolasError::Value(format!(
        "ambiguous name '{name}' in '{directive}': it is both a column of this DataFrame \
         and the built-in command '{command}'. Write '{name}:' to use the command, or \
         rename the column — df.rename(columns={{'{name}': '{name}_pct'}}) — to use the \
         data.",
        command = normalize(name)
    ))
}

/// Bind a directive against a frame whose data columns `is_column` names: each bare
/// name becomes what it reads — a data column stays a name; a built-in command becomes
/// that command (all defaults, its spelling kept, so it renders as written); anything
/// else stays a name (an absent column, an error at execution). A name that is both a
/// data column and a command usable bare is ambiguous, and `directive` (as written) is
/// rejected, naming both ways out. The result reads no bare name as a command, so it
/// executes the same on every later frame.
pub fn bind_frame(node: &Ast, is_column: &impl Fn(&str) -> bool, directive: &str) -> Result<Ast> {
    Ok(match node {
        Node::Name(name) if name.is_empty() => node.clone(),
        Node::Name(name) if is_column(name) => {
            if is_bare_command(name) {
                return Err(ambiguous(name, directive));
            }
            node.clone()
        }
        Node::Name(name) if is_command(&normalize(name)) => Node::Command(bind_bare(name)?),
        Node::Name(_) | Node::Scalar(_) => node.clone(),
        Node::Command(c) => Node::Command(Command {
            name: c.name.clone(),
            sub: c.sub.clone(),
            args: c.args.clone(),
            series: c
                .series
                .iter()
                .map(|s| bind_frame(s, is_column, directive))
                .collect::<Result<_>>()?,
        }),
        Node::Unary { op, operand } => Node::Unary {
            op: *op,
            operand: Box::new(bind_frame(operand, is_column, directive)?),
        },
        Node::Binary { left, op, right } => Node::Binary {
            left: Box::new(bind_frame(left, is_column, directive)?),
            op: *op,
            right: Box::new(bind_frame(right, is_column, directive)?),
        },
    })
}

/// Whether the frame-bound `node` reads the built-in command spelled exactly `name`
/// through a bare name — so a data column added under that name would have changed
/// what it reads.
pub fn uses_bare_command(node: &Ast, name: &str) -> bool {
    match node {
        Node::Name(_) | Node::Scalar(_) => false,
        Node::Command(c) => {
            c.args.bare.as_deref() == Some(name) || c.series.iter().any(|s| uses_bare_command(s, name))
        }
        Node::Unary { operand, .. } => uses_bare_command(operand, name),
        Node::Binary { left, right, .. } => uses_bare_command(left, name) || uses_bare_command(right, name),
    }
}

/// The bare-name rule for `directive_lookback`: without a frame a bare name is taken
/// as a command, so a known command must bind exactly as written — one with no sub-less
/// form (`kdj`) or a missing required argument (`ma`) is rejected, as `df[d]` rejects it
/// at execute, instead of passing as a 0-lookback column. An unknown bare name is an
/// unverifiable column reference and passes.
fn check_bare_command(name: &str) -> Result<()> {
    if is_command(&normalize(name)) {
        bind_command(name, None, &[], &[])?;
    }
    Ok(())
}

/// Walk a bound [`Ast`] applying only the bare-name rule — every `Command` was
/// already validated when `bind` produced the tree. Used by `directive_lookback`.
pub fn check_bare<A>(node: &Node<A>) -> Result<()> {
    match node {
        Node::Name(name) => check_bare_command(name),
        Node::Scalar(_) => Ok(()),
        Node::Command(c) => c.series.iter().try_for_each(check_bare),
        Node::Unary { operand, .. } => check_bare(operand),
        Node::Binary { left, right, .. } => {
            check_bare(left)?;
            check_bare(right)
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
