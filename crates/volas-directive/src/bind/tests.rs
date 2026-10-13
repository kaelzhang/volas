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
/// written); a data column owns the name even when it also spells a command.
#[test]
fn binds_bare_names_against_a_frame() {
    let columns = |n: &str| matches!(n, "close" | "change" | "repeat");
    let bind = |d: &str| bind_frame(&parse(d).unwrap(), &columns, d);
    let node = bind("close * obv").unwrap();
    assert!(uses_bare_command(&node, "obv") && !uses_bare_command(&node, "close"));
    assert!(matches!(node.node(), Node::Binary { right, .. } if matches!(&**right, Node::Command(_))));
    let data = bind("close * change").unwrap();
    assert!(matches!(data.node(), Node::Binary { right, .. } if matches!(&**right, Node::Data(name) if name == "change")));
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

/// Projected indicator values are ordinary data at every command spelling and at
/// every expression depth, even when the frame no longer has the command's inputs.
#[test]
fn directive_named_data_shadows_commands_in_every_operand() {
    use volas_core::{Column, DataFrame};
    let data = vec![2.0, 3.0, 5.0, 7.0];
    let frame = DataFrame::new(vec!["atr:14".into(), "change".into()],
        vec![Column::f64(data.clone()), Column::f64(data.clone())], None).unwrap();
    for text in ["atr:14", "ATR:14", "atr:014@high,low,close", "atr:14+0", "CHANGE", "change:", "change:*1"] {
        let node = bind_frame(&parse(text).unwrap(), &|n| frame.has_column(n), text).unwrap();
        assert_eq!(crate::execute(&frame, &node).unwrap().to_f64_vec(), data, "{text}");
    }
    let text = "ema:2@(ATR:14)";
    let node = bind_frame(&parse(text).unwrap(), &|n| frame.has_column(n), text).unwrap();
    let got = crate::execute(&frame, &node).unwrap().to_f64_vec();
    let expected = volas_compute::raw::indicators::ema(&data, 2);
    assert!(got.iter().zip(expected).all(|(a,b)| a.to_bits()==b.to_bits() || a.is_nan() && b.is_nan()));
}


#[test]
fn a_resolved_data_name_keeps_its_physical_label_and_window_history() {
    use volas_core::{Column, DataFrame, History};
    let frame = DataFrame::new(vec!["ATR:014".into(), "CHANGE".into()],
        vec![Column::f64(vec![2.0, 3.0]), Column::f64(vec![4.0, 5.0])], None).unwrap();
    let resolve = |name: &str| match name { "atr:14" => Some("ATR:014".to_string()), "change" => Some("CHANGE".to_string()), _ => None };
    for text in ["atr:14+0", "CHANGE", "change:", "ma:2@(change)"] {
        let node = bind_frame_resolved(&parse(text).unwrap(), &resolve, text).unwrap();
        let (lookback, history) = (crate::lookback::lookback(&node), crate::lookback::history(&node));
        assert_eq!(history, if text.starts_with("ma") { History::Accumulator } else { History::Window });
        assert_eq!(lookback, usize::from(text.starts_with("ma")));
        assert!(crate::execute(&frame, &node).is_ok(), "{text}");
    }
    let node = bind_frame_resolved(&parse("CHANGE").unwrap(), &resolve, "CHANGE").unwrap();
    assert!(matches!(node.node(), Node::Data(name) if name == "CHANGE"));
    assert!(matches!(node.continuation(), crate::types::Continuation::None));
}


#[test]
fn a_cached_bound_command_blocks_every_equivalent_data_shadow() {
    let no_data = |_: &str| false;
    for text in ["close-OBV", "close-obv:", "close-(obv:)"] {
        let node = bind_frame(&parse(text).unwrap(), &no_data, text).unwrap();
        assert!(uses_command_shadow(&node, "obv"));
        assert!(!uses_command_shadow(&node, "close"));
    }
    let node = bind_frame(&parse("change:*2").unwrap(), &no_data, "change:*2").unwrap();
    assert!(uses_command_shadow(&node, "change"));
    assert!(!uses_bare_command(&node, "change"));
    let node = bind_frame(&parse("ema:2@(atr:14+0)").unwrap(), &no_data, "").unwrap();
    assert!(uses_command_shadow(&node, "atr:14"));
    assert!(uses_command_shadow(&node, "atr:14+0"));
    let node = bind_frame(&parse("change:*2").unwrap(), &|name| name == "change", "").unwrap();
    assert!(!uses_command_shadow(&node, "change"));
}

#[test]
fn expression_literals_remain_constants_beside_numeric_data_labels() {
    use volas_core::{Column, DataFrame};
    let frame = DataFrame::new(vec!["close".into(), "1".into()],
        vec![Column::f64(vec![2.0, 3.0]), Column::f64(vec![100.0, 200.0])], None).unwrap();
    let node = bind_frame(&parse("close+1").unwrap(), &|name| frame.has_column(name), "").unwrap();
    assert_eq!(crate::execute(&frame, &node).unwrap().to_f64_vec(), vec![3.0, 4.0]);
}

#[test]
fn data_nodes_keep_identity_through_lowering_form_and_bare_validation() {
    let data: Cst = Node::Data("ATR:014".into());
    let bound = bind_node(&data).unwrap();
    assert!(matches!(&bound, Node::Data(name) if name == "ATR:014"));
    assert!(check_bare_node(&bound).is_ok());
    let form = bind_form(&data).unwrap();
    assert!(matches!(&form, Node::Data(name) if name == "ATR:014"));
    assert_eq!(crate::stringify(&form), "ATR:014");
}

#[test]
fn cached_unary_computation_protects_its_expression_and_nested_command() {
    let node = bind_frame(&parse("~(close>ema:2)").unwrap(), &|name| name == "close", "").unwrap();
    assert!(uses_command_shadow(&node, &crate::stringify(&node)));
    assert!(uses_command_shadow(&node, "ema:2"));
    assert!(!uses_command_shadow(&node, "close"));
    assert!(!uses_command_shadow(&node, "1"));
}
