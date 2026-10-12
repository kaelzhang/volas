//! Checkpoint-vs-full matrix: every output line a bounded frame keeps through a
//! checkpoint (`History::Carried`), on every data class, from every cut.
//!
//! A checkpoint captured over the rows before a cut, replayed over a frame that holds
//! only the rows from the cut on (less the rows its operands read before), must give the
//! full computation's values bit for bit — in one replay, or chained through a second
//! checkpoint — and the classification must agree with what the kernels carry: a
//! recursion is `Carried` exactly when it has a carried state.

mod matrix;

use std::collections::BTreeSet;

use volas_core::{Column, DataFrame, History};
use volas_directive::exec::{capture, checkpoint, execute, initial_state, replay};

thread_local! {
    static FROM_STATE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
use volas_directive::lookback::{history, lookback, reach};
use volas_directive::parse;
use volas_directive::types::Ast;

const ROWS: usize = 90;

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn first_diff(got: &[f64], want: &[f64]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("length {} != {}", got.len(), want.len()));
    }
    got.iter().zip(want).position(|(a, b)| !same(*a, *b)).map(|i| format!("row +{i}: {:?} != {:?}", got[i], want[i]))
}

/// Check one carried directive on one data class; returns each failure.
fn check(df: &DataFrame, node: &Ast, operands: usize) -> Vec<String> {
    let mut fails = Vec::new();
    let n = df.height();
    let Ok(whole) = execute(df, node).map(|c| c.to_f64_vec()) else { return fails };
    let lb = lookback(node);
    let cuts: BTreeSet<usize> = [1, 2, lb, lb + 1, lb + 2, n / 2, n - 2]
        .into_iter()
        .filter(|&c| c >= operands.max(1) && c + 1 < n)
        .collect();
    for cut in cuts {
        let Some(cp) = capture(df, node, cut) else {
            fails.push(format!("no checkpoint at {cut}"));
            continue;
        };
        assert_eq!(cp.at, cut - 1);
        // the same history kept from a carried state by a frame holding its last rows only
        if let Some(state) = cp.state.clone() {
            let part = df.slice_data(cut / 2, n);
            if let Some(kept) = checkpoint(&part, node, Some(state), cut - cut / 2, false) {
                FROM_STATE.with(|c| c.set(c.get() + 1));
                match replay(&part, node, &kept, part.height()) {
                    Some((got, _)) => {
                        if let Some(why) = first_diff(&got.to_f64_vec(), &whole[cut..]) {
                            fails.push(format!("replay from a carried state at {cut}: {why}"));
                        }
                    }
                    None => fails.push(format!("replay from a carried state at {cut} declined")),
                }
            }
        }
        // the frame a bounded window keeps: from the rows its operands read before the cut
        let frame = df.slice_data(cut - operands, n);
        let end = frame.height();
        match replay(&frame, node, &cp, end) {
            Some((got, next)) => {
                if let Some(why) = first_diff(&got.to_f64_vec(), &whole[cut..]) {
                    fails.push(format!("replay from {cut}: {why}"));
                }
                assert_eq!(next.at, n - 1);
            }
            None => fails.push(format!("replay from {cut} declined")),
        }
        // chained: up to the middle of the rest, then on from there
        let mid = cut + (n - cut) / 2;
        let Some((a, cp2)) = replay(&frame, node, &cp, mid - (cut - operands)) else {
            fails.push(format!("replay {cut}..{mid} declined"));
            continue;
        };
        let Some((b, _)) = replay(&frame, node, &cp2, end) else {
            fails.push(format!("replay {mid}.. declined"));
            continue;
        };
        let mut chained = a.to_f64_vec();
        chained.extend(b.to_f64_vec());
        if let Some(why) = first_diff(&chained, &whole[cut..]) {
            fails.push(format!("chained {cut}..{mid}..: {why}"));
        }
    }
    fails
}

#[test]
fn a_checkpoint_replays_every_carried_line_bit_exactly() {
    let classes = matrix::classes(ROWS);
    let mut carried = 0usize;
    let mut fails = Vec::new();
    for line in &matrix::lines() {
        for text in matrix::variants(line) {
            let Ok(node) = parse(&text) else { panic!("{text} does not parse") };
            let History::Carried { operands } = reach(&node).history else { continue };
            carried += 1;
            for data in &classes {
                for why in check(&data.frame, &node, operands) {
                    fails.push(format!("{text} [{}]: {why}", data.name));
                }
            }
        }
    }
    assert!(carried > 40, "only {carried} carried lines — the matrix lost its reach");
    assert!(FROM_STATE.with(|c| c.get()) > 100, "checkpoints from a carried state were not exercised");
    assert!(fails.is_empty(), "{} checkpoint replay failure(s):\n{}", fails.len(), fails.join("\n"));
}

/// A recursion is kept through a checkpoint exactly when its command carries a state:
/// `Carried` agrees with the kernels on every line, and a line without a recursion is
/// not `Unkept`.
#[test]
fn the_history_class_agrees_with_the_carried_states() {
    let frame = matrix::classes(ROWS).into_iter().next().unwrap().frame;
    for line in &matrix::lines() {
        for text in matrix::variants(line) {
            let node = parse(&text).unwrap();
            let Ok(col) = execute(&frame, &node) else { continue };
            let state = initial_state(&frame, &node, &col);
            match history(&node) {
                History::Carried { .. } => assert!(state.is_some(), "{text}: Carried without a state"),
                History::Unkept => assert!(state.is_none(), "{text}: a state, but Unkept"),
                _ => {}
            }
        }
    }
}

/// The history class of every command, as the spec owns it (REQUIREMENTS R13): a change
/// to a command's class changes this table, deliberately.
#[test]
fn every_command_has_its_history_class() {
    use volas_directive::spec::{history_class, HistoryClass, Keep, COMMANDS};
    const ACCUMULATORS: &[&str] = &[
        "accbands", "apo", "bbi", "bbw", "beta", "bias", "boll", "cci", "cmf", "cog", "coppock",
        "correl", "dkx", "dma", "dpo", "emv", "hma", "hv", "imi", "kst", "linearreg",
        "linearreg_angle", "linearreg_intercept", "linearreg_slope", "ma", "macdext", "mavp", "mfi",
        "ppo", "relative_vigor", "stddev", "stoch", "sum", "trima", "tsf", "ttm_squeeze", "ultosc",
        "var", "vr", "vwma", "wma", "wvad",
    ];
    const UNKEPT: &[&str] = &["crsi", "kcw", "stoch_momentum"];
    let mut seen = 0;
    for &name in COMMANDS {
        // the default arguments (no MA-type override); the stochastics' %D line
        let class = history_class(name, if name == "stochf" { Some("d") } else { None }, &[]);
        let want = if name == "increase" {
            HistoryClass::Recursion(Keep::Window)
        } else if UNKEPT.contains(&name) {
            HistoryClass::Recursion(Keep::Nothing)
        } else if volas_directive::spec::is_recursive(name, &[]) {
            HistoryClass::Recursion(Keep::State)
        } else if ACCUMULATORS.contains(&name) || name == "stochf" {
            HistoryClass::Accumulator
        } else {
            HistoryClass::Window
        };
        assert_eq!(class, want, "{name}");
        seen += 1;
    }
    assert_eq!(seen, COMMANDS.len());
    assert!(seen > 100);
    // an MA-type argument selecting a recursive average makes a recursion that recomputes
    let ema_type = parse("ma:5,1").unwrap();
    assert_eq!(history(&ema_type), History::Unkept);
}

#[test]
fn a_directive_takes_its_highest_class() {
    for (text, want) in [
        ("ema:5", History::Carried { operands: 0 }),
        ("obv", History::Carried { operands: 0 }),
        ("rsi:5@open", History::Carried { operands: 0 }),
        ("rsi:4@(ma:3)", History::Carried { operands: 2 }),
        ("ema:4@(high + low)", History::Carried { operands: 0 }),
        ("ema:4@(increase:2)", History::Carried { operands: 2 }),
        ("roc:5", History::Window),
        ("close > open", History::Window),
        ("maxindex:5", History::Window),
        ("psy", History::Window),
        ("ma:5", History::Accumulator),
        ("close > ma:3", History::Accumulator),
        ("increase:3", History::Counted),
        ("increase:2 & (close > ma:3)", History::Counted),
        ("ema:3@(rsi:5)", History::Unkept),
        ("ma:3@(ema:2)", History::Unkept),
        ("ema:3 - ema:5", History::Unkept),
        ("close > ema:5", History::Unkept),
        ("kcw", History::Unkept),
        ("crsi", History::Unkept),
        ("stoch_momentum", History::Unkept),
        ("ma:5,1", History::Unkept),
    ] {
        let h = history(&parse(text).unwrap());
        assert_eq!(h, want, "{text}");
        assert_eq!(h.class(), match want { History::Window => 1, History::Accumulator => 2, _ => 3 }, "{text}");
    }
}

fn frame(close: Vec<f64>) -> DataFrame {
    let n = close.len();
    let open: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
    DataFrame::new(vec!["open".into(), "close".into()], vec![Column::f64(open), Column::f64(close)], None)
        .unwrap()
        .with_origin(0)
        .slice_data(0, n)
}

/// Inside the warm-up a checkpoint has no state: it keeps every complete row of the
/// history, and the replay computes the whole history from them.
#[test]
fn a_warm_up_checkpoint_keeps_every_complete_row() {
    let close: Vec<f64> = (0..40).map(|i| 50.0 + (i as f64 * 0.7).sin()).collect();
    let mut gappy = close.clone();
    gappy[1] = f64::NAN;
    let df = frame(gappy);
    let node = parse("ema:10").unwrap();
    let cp = capture(&df, &node, 4).unwrap();
    assert!(cp.state.is_none());
    assert_eq!(cp.inputs, vec![(0, vec![df.column("close").unwrap().get_f64(0), close[2], close[3]])]);
    let (got, next) = replay(&df.slice_data(4, 40), &node, &cp, 36).unwrap();
    let want = execute(&df, &node).unwrap().to_f64_vec();
    assert!(first_diff(&got.to_f64_vec(), &want[4..]).is_none());
    assert!(next.state.is_some());
    // a replay with nothing after the checkpoint keeps it
    let (none, same_cp) = replay(&df.slice_data(4, 40), &node, &cp, 0).unwrap();
    assert_eq!((none.len(), same_cp), (0, cp));
}

/// A replay that cannot run declines: not a recursion, a frame not holding the row
/// after the checkpoint, an input that is not the one recorded, an unreadable input.
#[test]
fn a_replay_that_cannot_run_declines() {
    let df = frame((0..40).map(|i| 50.0 + (i as f64 * 0.7).sin()).collect());
    let ema = parse("ema:5").unwrap();
    let cp = capture(&df, &ema, 10).unwrap();
    assert!(capture(&df, &parse("ma:5").unwrap(), 10).is_none()); // not a recursion
    assert!(capture(&df, &ema, 0).is_none());
    assert!(capture(&df, &ema, 41).is_none());
    assert!(replay(&df, &parse("ma:5").unwrap(), &cp, 40).is_none());
    assert!(replay(&df.slice_data(11, 40), &ema, &cp, 29).is_none()); // starts after the row
    assert!(replay(&df, &ema, &cp, 41).is_none()); // past the frame
    assert!(replay(&df, &ema, &cp, 5).is_none()); // ends before the checkpoint
    let atr = parse("atr:5").unwrap();
    assert!(replay(&df, &atr, &cp, 40).is_none()); // reads high / low: no column
    let mut other = cp.clone();
    other.inputs[0].0 = 3;
    assert!(replay(&df, &ema, &other, 40).is_none()); // another input slot
    assert!(capture(&df, &atr, 10).is_none());
    // from a carried state: a frame without the resume's reach of complete rows before the
    // row cannot vouch for the history, unless it holds all of it
    let state = cp.state.clone();
    assert!(checkpoint(&df.slice_data(9, 40), &ema, state.clone(), 1, false).is_none());
    assert!(checkpoint(&df.slice_data(0, 40), &ema, None, 1, false).is_none());
    assert_eq!(checkpoint(&df, &ema, None, 1, true).unwrap().state, None);
    assert!(checkpoint(&df, &ema, state.clone(), 0, true).is_none());
    assert!(checkpoint(&df, &ema, state.clone(), 41, true).is_none());
    assert!(checkpoint(&df, &parse("ma:5").unwrap(), state.clone(), 10, true).is_none());
    assert!(checkpoint(&df, &atr, state, 10, true).is_none());
}

/// A line without a recursion depends on its lookback's rows alone: computed over just
/// the rows `[k - L, k]` (the frame's positions kept), its value at row `k` is the full
/// computation's — on every line and data class, at every row past the warm-up. This is
/// what lets a bounded window recompute it from the rows it keeps (a lookback that
/// under-states the rows a value reads fails here).
#[test]
fn a_window_line_depends_on_its_lookback_rows_alone() {
    let classes = matrix::classes(ROWS);
    let mut fails = Vec::new();
    for line in &matrix::lines() {
        for text in matrix::variants(line) {
            let node = parse(&text).unwrap();
            if history(&node) != History::Window {
                continue;
            }
            let l = lookback(&node);
            for data in &classes {
                let Ok(whole) = execute(&data.frame, &node) else { continue };
                let whole = whole.to_f64_vec();
                for k in (l..ROWS).step_by(7).chain([ROWS - 1]) {
                    let Ok(part) = execute(&data.frame.slice_data(k - l, k + 1), &node) else { continue };
                    let got = part.to_f64_vec()[l];
                    if !same(got, whole[k]) {
                        fails.push(format!("{text} [{}] row {k}: {got:?} != {:?}", data.name, whole[k]));
                        break;
                    }
                }
            }
        }
    }
    assert!(fails.is_empty(), "{} line(s) read past their lookback:\n{}", fails.len(), fails.join("\n"));
}
