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
use volas_directive::exec::{capture, execute, initial_state, replay};
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
    assert!(fails.is_empty(), "{} checkpoint replay failure(s):\n{}", fails.len(), fails.join("\n"));
}

/// A recursion is kept through a checkpoint exactly when its command carries a state:
/// `Carried` agrees with the kernels on every line, and a line without a recursion is
/// `Window`.
#[test]
fn the_history_class_agrees_with_the_carried_states() {
    let frame = matrix::classes(ROWS).into_iter().next().unwrap().frame;
    for line in &matrix::lines() {
        for text in matrix::variants(line) {
            let node = parse(&text).unwrap();
            let Ok(col) = execute(&frame, &node) else { continue };
            let state = initial_state(&frame, &node, &col);
            match history(&node) {
                History::Window => {}
                History::Carried { .. } => assert!(state.is_some(), "{text}: Carried without a state"),
                History::Unkept => assert!(state.is_none(), "{text}: a state, but Unkept"),
            }
        }
    }
}

#[test]
fn compositions_of_a_recursion_are_unkept() {
    for (text, want) in [
        ("ema:5", History::Carried { operands: 0 }),
        ("obv", History::Carried { operands: 0 }),
        ("rsi:5@open", History::Carried { operands: 0 }),
        ("rsi:4@(ma:3)", History::Carried { operands: 2 }),
        ("ema:4@(high + low)", History::Carried { operands: 0 }),
        ("ma:5", History::Window),
        ("close > ma:3", History::Window),
        ("ema:3@(rsi:5)", History::Unkept),
        ("ma:3@(ema:2)", History::Unkept),
        ("ema:3 - ema:5", History::Unkept),
        ("close > ema:5", History::Unkept),
        ("kcw", History::Unkept),
        ("crsi", History::Unkept),
        ("stoch_momentum", History::Unkept),
        ("ma:5,1", History::Unkept),
    ] {
        assert_eq!(history(&parse(text).unwrap()), want, "{text}");
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
