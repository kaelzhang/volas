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
    // hv: a bar longer than the session; an argument the frame does not use.
    assert!(err_msg("hv:10,8h,252,240").contains("a 8h bar is longer than the 240-minute"));
    assert!(err_msg("hv:10,1d,252,390").contains("session_minutes applies to sub-day"));
    assert!(err_msg("hv:10,1w,365").contains("trading_days applies to sub-day and day"));
    for ok in ["hv:10,4h,252,240", "hv:10,1d,365", "hv:10,1w,252,1440", "hv:10,1M"] {
        assert!(parse(ok).is_ok(), "{ok}");
    }
}

/// A command takes at most as many `@` operands as it reads, and a required operand
/// must be given — both checked when binding, never ignored or deferred to execution.
#[test]
fn binds_the_operand_slots() {
    assert!(err_msg("ema:2@close,(ma:100)").contains("accepts at most 1 series operand(s), got 2"));
    assert!(err_msg("ma:5@close,nope").contains("at most 1 series"));
    assert!(err_msg("correl:5@close").contains("series argument #1 is required"));
    assert!(err_msg("correl:5@close,").contains("series argument #1 is required"));
    assert!(err_msg("repeat:2").contains("series argument #0 is required"));
    for ok in ["correl:5@close,open", "kdj.j:@,high", "mavp:2,9@,volume", "repeat:2@(close>1)"] {
        assert!(parse(ok).is_ok(), "{ok}");
    }
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
    assert!(bind_command_node("obv", None, &[], &[]).is_ok());
    let boll = bind_bare("boll").unwrap();
    assert_eq!(boll.args[0], ArgValue::Usize(20));
    assert_eq!(boll.name, "boll");
    assert_eq!(boll.args.bare.as_deref(), Some("boll"));
    // `cdl` folds to the canonical `style` namespace; sub canonicalizes.
    let doji = bind_command_node("CDL", Some("doji"), &[], &[]).unwrap();
    assert_eq!(doji.name, "style");
    assert!(bind_command_node("ema", None, &[], &[]).is_err()); // required period
    assert!(bind_command_node("atr", None, &[], &[]).is_err()); // required period
    assert!(bind_command_node("kdj", None, &[], &[]).is_err()); // needs a sub
    assert!(bind_command_node("nope", None, &[], &[]).is_err()); // unknown
}

/// Against a frame, a bare name reads what it names: a data column, a command (kept as
/// written), or — when it is both a data column and a command usable bare — nothing:
/// the directive is ambiguous.
#[test]
fn binds_bare_names_against_a_frame() {
    let columns = |n: &str| matches!(n, "close" | "change" | "repeat");
    let bind = |d: &str| bind_frame(&parse(d).unwrap(), &columns, d);
    let node = bind("close * obv").unwrap();
    assert!(uses_bare_command(&node, "obv") && !uses_bare_command(&node, "close"));
    assert!(matches!(node.node(), Node::Binary { right, .. } if matches!(&**right, Node::Command(_))));
    let err = format!("{:?}", bind("close * change").unwrap_err());
    assert!(err.contains("ambiguous name 'change' in 'close * change'"), "{err}");
    // `repeat` is no bare command (its operand is required): the data column is read
    assert!(bind("accbands@repeat").is_ok());
    assert!(bind("ma:5@(-obv) + nope + 1").is_ok()); // every node kind; an unknown name stays a name
    assert!(bind("change: * 2").is_ok());
    assert!(!uses_bare_command(&parse("~(1 > close)").unwrap(), "close"));
}

#[test]
fn bind_form_validates_commands_leniently() {
    // Rejects an unknown command, a bad sub-command, and a bare sub-requiring name —
    // even nested — but is lenient on absent arguments and operands (`donchian.upper`
    // names a real form, its period supplied at use).
    let check = |d: &str| bind_form(&parse_cst(d).unwrap());
    assert!(check("close").is_ok()); // column reference (unknown bare name)
    assert!(check("obv").is_ok()); // no-arg command
    assert!(check("3.5").is_ok()); // scalar
    assert!(check("close > 5").is_ok()); // binary with a scalar operand
    assert!(check("ma:5@close").is_ok()); // command with a valid series operand
    assert!(check("rsi").is_ok()); // a required-arg command still has a main form
    assert!(check("donchian.upper").is_ok()); // valid form, period absent (lenient)
    assert!(check("correl:5").is_ok()); // a required operand absent (lenient)
    assert!(check("kdj").is_err()); // bare name needs a sub
    assert!(check("~kdj").is_err()); // nested in a unary
    assert!(check("kdj > 0").is_err()); // nested in a binary
    assert!(check("ma:5@vortex").is_err()); // bare sub-requiring name in a series
    assert!(check("ma.foo").is_err()); // unknown sub-command
    assert!(check("kdj:9").is_err()); // a Command whose command needs a sub-command
    assert!(check("frobnicate:5").is_err()); // unknown command
    assert!(check("ema:2,3").is_err()); // too many arguments
    assert!(check("ema:2@close,open").is_err()); // too many operands
    assert!(check("ma:0").is_err()); // out-of-domain supplied argument
    assert!(check("mavp:2,30@close,close").is_ok()); // valid cross-arg (min <= max)
    assert!(check("mavp:30,2@close,close").is_err()); // inverted period limits
    assert!(check("mavp@close,close").is_ok()); // both absent -> cross-arg skipped
    // hv's rule runs on the form too, absent slots taking their defaults.
    assert!(check("hv").is_ok());
    assert!(check("hv:,8h,252,240").is_err()); // the bar outlasts the session
    assert!(check("hv:,1w,365").is_err()); // trading_days unused by weekly bars
    assert!(check("hv:,1d,,390").is_err()); // session_minutes unused by day bars
    assert!(check("hv:,4h,,240").is_ok());
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


/// A bare command's identity is canonical; a prospective data column keeps its case.
#[test]
fn bare_command_collisions_use_canonical_identity() {
    for spelling in ["accbands", "ACCBANDS", "AcCbAnDs"] {
        for text in [
            format!("({spelling})+({spelling}@open)"),
            format!("-{spelling}"),
            format!("ma:3@{spelling}"),
        ] {
            let bound = bind_frame(&parse(&text).unwrap(), &|n| n == "open", &text).unwrap();
            assert!(uses_bare_command(&bound, "accbands"), "{text}");
            assert!(!uses_bare_command(&bound, "ACCBANDS"), "{text}");
            assert!(!uses_bare_command(&bound, "open"), "{text}");
        }
    }
    let explicit = bind_frame(&parse("ACCBANDS: + 1").unwrap(), &|_| false, "").unwrap();
    assert!(!uses_bare_command(&explicit, "accbands"));
}
