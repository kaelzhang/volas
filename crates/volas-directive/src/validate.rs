//! Directive domain validation — the single check shared by every public entry
//! point (`exec`, `directive_lookback`, `directive_stringify`), so all three
//! accept exactly the same set of directives (V17). A directive that `df[d]`
//! rejects can no longer be canonicalized, persisted, or have a warm-up window
//! computed for it.
//!
//! Validates, per command: name / sub-command existence, argument count, each
//! supplied argument's domain (against its `spec` bound), and the cross-argument
//! rules a single-arg bound cannot express. Missing *required* arguments stay
//! `exec`'s concern (`arg_usize`), matching the reviewer's scope.

use crate::spec::{canon_sub, command_spec, is_command};
use crate::types::Node;
use volas_core::{Result, VolasError};

/// Validate every command in a directive AST. A bare `Name` is a known no-arg
/// command (validated) or a column reference (unverifiable without a frame, so
/// accepted); operators and `@`-series sub-expressions recurse.
pub fn validate_node(node: &Node) -> Result<()> {
    match node {
        Node::Scalar(_) => Ok(()),
        Node::Name(name) => {
            if names_a_command(name) {
                validate_command(name, None, &[])
            } else {
                Ok(()) // a column reference — checked against the frame at exec time
            }
        }
        Node::Command(cmd) => {
            validate_command(&cmd.name, cmd.sub.as_deref(), &cmd.args)?;
            cmd.series.iter().try_for_each(validate_node)
        }
        Node::Unary { operand, .. } => validate_node(operand),
        Node::Binary { left, right, .. } => {
            validate_node(left)?;
            validate_node(right)
        }
    }
}

/// Normalize a command name the way `exec` does: case-insensitive (P6), with
/// `cdl` aliased to the `style` candlestick namespace.
fn normalize(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    if lower == "cdl" {
        "style".to_string()
    } else {
        lower
    }
}

/// Whether `name` (normalized) is a known command — so a bare `Name` is told
/// apart from a column reference.
fn names_a_command(name: &str) -> bool {
    is_command(&normalize(name))
}

/// Validate one command's name / sub-command / argument count / argument domains
/// / cross-argument rules against the spec. The single boundary where a bad
/// configuration errors loudly (a zero window, an out-of-domain seed, a NaN
/// multiplier, inverted `mavp` limits) instead of becoming a valid-shaped column
/// with no signal (V17). Shared verbatim by `exec_command`.
pub fn validate_command(name: &str, sub: Option<&str>, args: &[Option<String>]) -> Result<()> {
    let name = normalize(name);
    let name = name.as_str();
    let sub = canon_sub(name, sub);
    let sub = sub.as_deref();

    match command_spec(name, sub) {
        Some(spec) if args.len() > spec.args.len() => {
            return Err(VolasError::Value(format!(
                "command \"{name}\" accepts at most {} argument(s), got {}",
                spec.args.len(),
                args.len()
            )));
        }
        Some(spec) => {
            for (i, arg) in spec.args.iter().enumerate() {
                if let Some(s) = args.get(i).and_then(|o| o.as_deref()) {
                    arg.bound.validate(s).map_err(|why| {
                        VolasError::Value(format!("{name}: argument #{i} {why}, got '{s}'"))
                    })?;
                }
            }
        }
        None if is_command(name) => {
            return Err(VolasError::Value(match sub {
                Some(s) => format!("command \"{name}\" has no sub-command \"{s}\""),
                None => format!("command \"{name}\" requires a sub-command"),
            }));
        }
        None => return Err(VolasError::Value(format!("unknown command \"{name}\""))),
    }

    validate_cross_args(name, args)
}

/// Cross-argument domain rules — a constraint *between* two arguments that no
/// single-argument bound can express.
fn validate_cross_args(name: &str, args: &[Option<String>]) -> Result<()> {
    if name == "mavp" {
        // Each per-bar period is clamped into [min_period, max_period]; inverted
        // limits are meaningless (and panic `i64::clamp`).
        let parse_at = |i: usize| args.get(i).and_then(|o| o.as_deref()).and_then(|s| s.parse::<usize>().ok());
        if let (Some(min_p), Some(max_p)) = (parse_at(0), parse_at(1)) {
            if min_p > max_p {
                return Err(VolasError::Value(format!(
                    "mavp: min_period ({min_p}) must be <= max_period ({max_p})"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn ok(d: &str) {
        validate_node(&parse(d).unwrap()).unwrap_or_else(|e| panic!("{d:?} should pass: {e:?}"));
    }
    fn err(d: &str) {
        assert!(validate_node(&parse(d).unwrap()).is_err(), "{d:?} should fail");
    }

    #[test]
    fn validate_node_covers_every_arm() {
        // Scalar literal + a column-reference Name: both accepted (no command).
        ok("3.5");
        ok("close");
        // a bare known command (Name arm) and one needing a sub.
        ok("tr");
        err("kdj"); // requires a sub-command
        ok("TR"); // case-insensitive (P6)
        // Command arm: domain bound + cross-arg + count + sub + unknown.
        ok("ma:5");
        err("ma:0"); // zero window
        err("ma:5,9"); // matype out of [0, 8]
        err("ema:5,6"); // too many args
        err("macd.bogus"); // unknown sub-command
        err("frobnicate:5"); // unknown command
        err("mavp:30,2@close,close"); // inverted period limits (cross-arg)
        ok("cdl.doji"); // cdl -> style alias
        // Unary + Binary + @-series recursion: an invalid command nested inside.
        ok("-close");
        err("-(ma:0)");
        ok("ma:5 + close");
        err("ma:0 + close"); // left operand invalid
        err("close + rsi:0"); // right operand invalid
        ok("repeat:2@(close>10)");
        err("repeat:2@(ma:0>10)"); // invalid command inside a series sub-expression
    }
}
