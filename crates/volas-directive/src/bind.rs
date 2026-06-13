//! Bind a parsed directive ([`Cst`]) to its executable form ([`Ast`]).
//!
//! This is the single place a directive is *validated*, and it happens **once**
//! (at parse / cache time), not on every execution. For each command it walks the
//! spec argument list one time and, per position, either parses-and-checks the
//! supplied token or substitutes the spec default — validation, defaulting, and
//! type conversion are the same single pass (see [`crate::spec::ArgBound::bind`]),
//! so there is no separate "validate then re-read" stage and the default value
//! lives only in the spec. The resulting [`Ast`] carries canonical names and
//! ready typed [`ArgValue`]s, so `exec` / `lookback` / `stringify` never re-parse,
//! re-default, or re-validate.

use crate::spec::{canon_sub, command_spec, is_command, normalize};
use crate::types::{ArgValue, Ast, Command, Cst, Node};
use volas_core::{Result, VolasError};

/// Lower a concrete syntax tree to a bound AST. A bare [`Node::Name`] is left
/// unbound — without a frame it cannot be told apart from a column reference, so
/// its command validation is deferred to `exec` (column-first, then [`bind_command`]).
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

/// Validate and resolve one command into its bound form: canonical name / sub,
/// arguments parsed-checked-defaulted in a single pass, series operands lowered.
/// The one boundary where a bad configuration errors loudly (a zero window, an
/// out-of-domain seed, a NaN multiplier, inverted `mavp` limits, a missing
/// required argument, an unknown command / sub) instead of becoming a
/// valid-shaped column with no signal (V17). Also called at `exec` for a bare
/// `Name` that resolves to a command.
pub fn bind_command(
    name: &str,
    sub: Option<&str>,
    args: &[Option<String>],
    series: &[Cst],
) -> Result<Command<Box<[ArgValue]>>> {
    let name = normalize(name);
    let name = name.as_ref();
    let sub = canon_sub(name, sub);
    let sub = sub.as_deref();

    let spec = match command_spec(name, sub) {
        Some(spec) => spec,
        None if is_command(name) => {
            return Err(VolasError::Value(match sub {
                Some(s) => format!("command \"{name}\" has no sub-command \"{s}\""),
                None => format!("command \"{name}\" requires a sub-command"),
            }));
        }
        None => return Err(VolasError::Value(format!("unknown command \"{name}\""))),
    };
    if args.len() > spec.args.len() {
        return Err(VolasError::Value(format!(
            "command \"{name}\" accepts at most {} argument(s), got {}",
            spec.args.len(),
            args.len()
        )));
    }

    // One pass over the spec: parse-and-check a supplied token, or take the
    // (valid-by-construction) default; a missing required argument errors.
    let mut bound = Vec::with_capacity(spec.args.len());
    for (i, arg) in spec.args.iter().enumerate() {
        let value = match args.get(i).and_then(|o| o.as_deref()) {
            Some(s) => arg
                .bound
                .bind(s)
                .map_err(|why| VolasError::Value(format!("{name}: argument #{i} {why}, got '{s}'")))?,
            None => arg
                .default
                .value()
                .ok_or_else(|| VolasError::Value(format!("{name}: argument #{i} is required")))?,
        };
        bound.push(value);
    }

    bind_cross_args(name, &bound)?;

    let series = series.iter().map(bind_node).collect::<Result<Vec<_>>>()?;
    Ok(Command {
        name: name.to_string(),
        sub: sub.map(str::to_string),
        args: bound.into_boxed_slice(),
        series,
    })
}

/// The bare-name rule for the frame-blind APIs: without a frame a bare name is
/// taken as a command, so a known command with no sub-less form — `kdj`, `stoch`,
/// `vortex`, … — is rejected, exactly as `df[d]` rejects it at execute. An unknown
/// bare name is an unverifiable column reference and passes.
fn check_bare_name(name: &str) -> Result<()> {
    let name = normalize(name);
    if is_command(&name) && command_spec(&name, None).is_none() {
        return Err(VolasError::Value(format!(
            "command \"{name}\" requires a sub-command"
        )));
    }
    Ok(())
}

/// Walk a bound [`Ast`] applying only the bare-name rule — every `Command` was
/// already validated when `bind` produced the tree. Used by `directive_lookback`.
pub fn check_bare<A>(node: &Node<A>) -> Result<()> {
    match node {
        Node::Name(name) => check_bare_name(name),
        Node::Scalar(_) => Ok(()),
        Node::Command(c) => c.series.iter().try_for_each(check_bare),
        Node::Unary { operand, .. } => check_bare(operand),
        Node::Binary { left, right, .. } => {
            check_bare(left)?;
            check_bare(right)
        }
    }
}

/// Validate a raw [`Cst`] for the form-level `directive_stringify`. Like `bind` it
/// rejects an unknown command / sub-command, too many arguments, an out-of-domain
/// *supplied* argument, a bad cross-argument rule, and a bare sub-requiring name —
/// but it is lenient on a *missing required* argument, since a form such as
/// `donchian.upper` names a real indicator whose period is supplied only at use. So
/// `directive_stringify` rejects exactly what `df[d]` would except for the absent
/// period of an otherwise-valid form.
pub fn check_form(cst: &Cst) -> Result<()> {
    match cst {
        Node::Name(name) => check_bare_name(name),
        Node::Scalar(_) => Ok(()),
        Node::Command(c) => {
            let name = normalize(&c.name);
            let sub = canon_sub(&name, c.sub.as_deref());
            let spec = match command_spec(&name, sub.as_deref()) {
                Some(spec) => spec,
                None if is_command(&name) => {
                    return Err(VolasError::Value(match sub.as_deref() {
                        Some(s) => format!("command \"{name}\" has no sub-command \"{s}\""),
                        None => format!("command \"{name}\" requires a sub-command"),
                    }));
                }
                None => return Err(VolasError::Value(format!("unknown command \"{name}\""))),
            };
            if c.args.len() > spec.args.len() {
                return Err(VolasError::Value(format!(
                    "command \"{name}\" accepts at most {} argument(s), got {}",
                    spec.args.len(),
                    c.args.len()
                )));
            }
            for (i, arg) in spec.args.iter().enumerate() {
                if let Some(s) = c.args.get(i).and_then(|o| o.as_deref()) {
                    arg.bound.bind(s).map_err(|why| {
                        VolasError::Value(format!("{name}: argument #{i} {why}, got '{s}'"))
                    })?;
                }
            }
            check_form_cross(&name, &c.args)?;
            c.series.iter().try_for_each(check_form)
        }
        Node::Unary { operand, .. } => check_form(operand),
        Node::Binary { left, right, .. } => {
            check_form(left)?;
            check_form(right)
        }
    }
}

/// Cross-argument rules over *supplied* raw tokens (lenient): checked only when both
/// operands are present, mirroring [`bind_cross_args`] without requiring them.
fn check_form_cross(name: &str, args: &[Option<String>]) -> Result<()> {
    if name == "mavp" {
        let at = |i: usize| args.get(i).and_then(|o| o.as_deref()).and_then(|s| s.parse::<usize>().ok());
        if let (Some(min_p), Some(max_p)) = (at(0), at(1)) {
            if min_p > max_p {
                return Err(VolasError::Value(format!(
                    "mavp: min_period ({min_p}) must be <= max_period ({max_p})"
                )));
            }
        }
    }
    Ok(())
}

/// Cross-argument domain rules — a constraint *between* two resolved arguments
/// that no single-argument bound can express.
fn bind_cross_args(name: &str, args: &[ArgValue]) -> Result<()> {
    if name == "mavp" {
        // Each per-bar period is clamped into [min_period, max_period]; inverted
        // limits are meaningless (and panic `i64::clamp`). Both are bound `>= 1`.
        let (min_p, max_p) = (args[0].as_usize(), args[1].as_usize());
        if min_p > max_p {
            return Err(VolasError::Value(format!(
                "mavp: min_period ({min_p}) must be <= max_period ({max_p})"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, parse_cst};

    /// `bind` (via `parse`) is the single validation boundary; an out-of-domain
    /// configuration is rejected, with the diagnostic naming the offending command.
    fn err_msg(d: &str) -> String {
        format!("{:?}", parse(d).unwrap_err())
    }

    #[test]
    fn binds_every_command_error_branch() {
        // unknown command / sub-command existence / count.
        assert!(err_msg("frobnicate:5").contains("unknown command"));
        assert!(err_msg("kdj:9").contains("requires a sub-command"));
        assert!(err_msg("macd.bogus").contains("no sub-command"));
        assert!(err_msg("ema:5,6").contains("at most"));
        // a supplied argument out of its domain, and a missing required slot.
        assert!(err_msg("ma:0").contains(">= 1"));
        assert!(err_msg("ema:").contains("argument #0 is required"));
        // cross-argument rule (mavp's inverted period limits).
        assert!(err_msg("mavp:30,2@close,close").contains("min_period"));
    }

    #[test]
    fn binds_through_operators_and_series() {
        // A valid tree of every node kind binds; a nested invalid command anywhere
        // (unary operand, either binary side, or a series sub-expression) is rejected.
        for ok in ["3.5", "close", "~(close>10)", "ma:5 + close", "repeat:2@(close>10)"] {
            assert!(parse(ok).is_ok(), "{ok} should bind");
        }
        for bad in ["-(ma:0)", "ma:0 + close", "close + rsi:0", "repeat:2@(ma:0>10)"] {
            assert!(parse(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn binds_bare_name_command_on_the_fly() {
        // A bare name resolves its defaults (no-arg `obv`, default-only `boll` -> 20);
        // a name needing a sub or a required argument errors; an unknown name too.
        assert!(bind_command("obv", None, &[], &[]).is_ok());
        let boll = bind_command("boll", None, &[], &[]).unwrap();
        assert_eq!(boll.args[0], ArgValue::Usize(20));
        assert_eq!(boll.name, "boll");
        // `cdl` folds to the canonical `style` namespace; sub canonicalizes.
        let doji = bind_command("CDL", Some("doji"), &[], &[]).unwrap();
        assert_eq!(doji.name, "style");
        assert!(bind_command("ema", None, &[], &[]).is_err()); // required period
        assert!(bind_command("atr", None, &[], &[]).is_err()); // required period
        assert!(bind_command("kdj", None, &[], &[]).is_err()); // needs a sub
        assert!(bind_command("nope", None, &[], &[]).is_err()); // unknown
    }

    #[test]
    fn check_directive_validates_commands_leniently() {
        // Operates on a raw `Cst` (no binding): rejects an unknown command, a bad
        // sub-command, and a bare sub-requiring name — even nested — but is lenient
        // on arguments (`donchian.upper` names a real form, period supplied at use).
        let check = |d: &str| check_form(&parse_cst(d).unwrap());
        assert!(check("close").is_ok()); // column reference (unknown bare name)
        assert!(check("obv").is_ok()); // no-arg command
        assert!(check("3.5").is_ok()); // scalar
        assert!(check("close > 5").is_ok()); // binary with a scalar operand
        assert!(check("ma:5@close").is_ok()); // command with a valid series operand
        assert!(check("rsi").is_ok()); // a required-arg command still has a main form
        assert!(check("donchian.upper").is_ok()); // valid form, period absent (lenient)
        assert!(check("kdj").is_err()); // bare name needs a sub
        assert!(check("~kdj").is_err()); // nested in a unary
        assert!(check("kdj > 0").is_err()); // nested in a binary
        assert!(check("ma:5@vortex").is_err()); // bare sub-requiring name in a series
        assert!(check("ma.foo").is_err()); // unknown sub-command
        assert!(check("kdj:9").is_err()); // a Command whose command needs a sub-command
        assert!(check("frobnicate:5").is_err()); // unknown command
        assert!(check("ema:2,3").is_err()); // too many arguments
        assert!(check("ma:0").is_err()); // out-of-domain supplied argument
        assert!(check("mavp:2,30@close,close").is_ok()); // valid cross-arg (min <= max)
        assert!(check("mavp:30,2@close,close").is_err()); // inverted period limits
        assert!(check("mavp@close,close").is_ok()); // both absent -> cross-arg skipped
    }

    #[test]
    fn check_bare_walks_every_node_kind() {
        // `check_bare` (the bound-`Ast` path for `directive_lookback`) applies only the
        // bare-name rule, but visits every node kind.
        let bare = |d: &str| check_bare(&parse(d).unwrap());
        assert!(bare("close").is_ok()); // Name (column reference)
        assert!(bare("5").is_ok()); // Scalar
        assert!(bare("~(close > 5)").is_ok()); // Unary -> Binary -> Name / Scalar
        assert!(bare("ma:5 + ma:10").is_ok()); // Binary -> Command (-> series)
        assert!(bare("kdj").is_err()); // a bare name needing a sub
    }
}
