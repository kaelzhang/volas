//! State chains: a directive whose recursions sit below its top (inside `@` operands, or
//! combined with other values) carries a state per recursive node and continues new rows
//! from them — through the single-row step, the `Vec` resume over several rows, and the
//! computation with its states — bit for bit the full computation, on every data class,
//! and the chain it carries equals the one captured afresh.

#[allow(dead_code)] // the shared matrix module: this test reads its data classes only
mod matrix;

use volas_directive::exec::{execute, execute_resume, execute_resume_step, execute_with_states, initial_state};
use volas_directive::parse;

const CHAINED: &[&str] = &[
    "ema:5@(ema:3)",
    "ema:2@(maxindex:2)",
    "ema:2@(minindex:2)",
    "rsi:2@(maxindex:2)",
    "ma:5@(ema:3)",
    "rsi:5@(ema:4@(close+open))",
    "ema:5@(kama:4)",
    "close-ma:5+obv",
    "close>ema:5",
    "ema:3-ema:5",
    "(ema:3 >< ema:5) & (close > ma:4)",
    "ema:4@(rsi:3)+sar",
    "atr:4@(ema:2@high),low,close",
    "ma:4@(maxindex:5)",
    "stddev:4@(macd.signal:5,9,3)",
    "-obv",
    "ema:3@(ema:2@(ema:2))",
];

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn same_chain(a: Option<&Vec<f64>>, b: Option<&Vec<f64>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(*x, *y)),
        (a, b) => a.is_none() && b.is_none(),
    }
}

fn values(col: &volas_core::Column) -> Vec<f64> {
    match col.as_bool() {
        Some(_) => (0..col.len()).map(|i| col.truth_at(i).map_or(f64::NAN, |t| f64::from(u8::from(t)))).collect(),
        None => col.to_f64_vec(),
    }
}

#[test]
fn a_chain_continues_bit_for_bit() {
    let mut fails = Vec::new();
    let mut chains = 0;
    let n = 120;
    for data in matrix::classes(n) {
        let df = &data.frame;
        for &text in CHAINED {
            let node = parse(text).unwrap();
            let full = execute(df, &node).unwrap();
            let whole = values(&full);
            let (col, after, before) = execute_with_states(df, &node).unwrap();
            if !values(&col).iter().zip(&whole).all(|(a, b)| same(*a, *b)) {
                fails.push(format!("{text} [{}]: values with states differ", data.name));
            }
            if after.is_some() {
                chains += 1;
                if !same_chain(after.as_ref(), initial_state(df, &node, n).as_ref())
                    || !same_chain(before.as_ref(), initial_state(df, &node, n - 1).as_ref())
                {
                    fails.push(format!("{text} [{}]: chains differ from a fresh capture", data.name));
                }
            }
            for from in [40, 75, 119] {
                let Some(mut chain) = initial_state(df, &node, from) else { continue };
                // one row at a time
                for (row, &want) in whole.iter().enumerate().skip(from) {
                    let mut next = chain.clone();
                    match execute_resume_step(df, &node, &chain, (row, 0), &mut next) {
                        Some(v) if same(v, want) => chain = next,
                        other => {
                            fails.push(format!("{text} [{}] step at {row}: {other:?} != {want}", data.name));
                            break;
                        }
                    }
                }
                if !same_chain(Some(&chain), initial_state(df, &node, n).as_ref()) {
                    fails.push(format!("{text} [{}] from {from}: stepped chain != fresh", data.name));
                }
                // several rows at once
                let start = initial_state(df, &node, from).unwrap();
                match execute_resume(df, &node, &start, from, n, 0) {
                    Some((tail, last)) => {
                        if !values(&tail).iter().zip(&whole[from..]).all(|(a, b)| same(*a, *b)) {
                            fails.push(format!("{text} [{}] from {from}: vec resume differs", data.name));
                        }
                        if !same_chain(Some(&last), initial_state(df, &node, n).as_ref()) {
                            fails.push(format!("{text} [{}] from {from}: vec chain != fresh", data.name));
                        }
                    }
                    None => fails.push(format!("{text} [{}] from {from}: the vec resume declined", data.name)),
                }
            }
        }
    }
    assert!(chains > 100, "only {chains} chains formed");
    assert!(fails.is_empty(), "{} failure(s):\n{}", fails.len(), fails[..fails.len().min(40)].join("\n"));
}

/// A chain continues only forwards and only from a chain of its own layout: a range that
/// ends before it starts, or a chain longer than the directive's, declines.
#[test]
fn a_chain_declines_what_does_not_fit_it() {
    let data = matrix::classes(80).into_iter().next().unwrap();
    let df = &data.frame;
    let node = parse("ema:5@(ema:3)").unwrap();
    let chain = initial_state(df, &node, 60).unwrap();
    assert!(execute_resume(df, &node, &chain, 61, 60, 0).is_none());
    let mut longer = chain.clone();
    longer.push(0.0);
    assert!(execute_resume(df, &node, &longer, 60, 80, 0).is_none());
    assert!(execute_resume(df, &node, &chain, 60, 80, 0).is_some());
}

/// CRSI now composes through the ordinary continuation plan.
#[test]
fn a_crsi_operand_carries_its_continuation() {
    let data = matrix::classes(120).remove(0);
    let node = parse("ema:4@(crsi:3,2,5)").unwrap();
    assert!(initial_state(&data.frame, &node, 90).is_some());
    assert_eq!(volas_directive::lookback::history(&node), volas_core::History::Carried { operands: 0 });
}

#[test]
fn chains_replay_from_a_window_checkpoint() {
    use volas_directive::exec::{capture, checkpoint, replay};
    for data in matrix::classes(120) {
        for &text in CHAINED {
            let node = parse(text).unwrap();
            let whole = values(&execute(&data.frame, &node).unwrap());
            for cut in [2, 40, 75, 100] {
                let cp = capture(&data.frame, &node, cut).expect(text);
                let span = match volas_directive::lookback::history(&node) {
                    volas_core::History::Carried { operands } => operands,
                    _ => volas_directive::lookback::lookback(&node) + 1,
                };
                let start = cut.saturating_sub(span);
                let frame = data.frame.slice_data(start, 120);
                let cp = if cp.state.is_some() {
                    checkpoint(&frame, &node, cp.state.map(|state| state.to_vec()), cut - start, false).unwrap()
                } else { cp };
                let (got, _) = replay(&frame, &node, &cp, frame.height()).unwrap_or_else(|| panic!("{text} [{}] {cut} replay declined", data.name));
                assert!(values(&got).iter().zip(&whole[cut..]).all(|(a,b)| same(*a,*b)), "{text} [{}] {cut}", data.name);
            }
        }
    }
}

#[test]
fn rows_only_chain_nodes_decline_scalar_execution() {
    let df = matrix::classes(300).remove(0).frame;
    for text in ["asi > 0", "kdj.k > 50", "ht_dcperiod > 20"] {
        let node = parse(text).unwrap();
        let state = initial_state(&df, &node, 200).unwrap();
        let mut next = state.clone();
        assert!(execute_resume_step(&df, &node, &state, (200, 0), &mut next).is_none());
        let (got, _) = execute_resume(&df, &node, &state, 200, 300, 0).unwrap();
        let full = values(&execute(&df, &node).unwrap());
        assert!(values(&got).iter().zip(&full[200..]).all(|(a,b)| same(*a,*b)), "{text}");
    }
}
