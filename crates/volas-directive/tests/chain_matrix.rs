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
            let truth = full.as_bool().is_some();
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
                if truth {
                    // a step yields a float: a comparison's truth declines it (the
                    // `Vec` resume below continues it)
                    let mut next = chain.clone();
                    if execute_resume_step(df, &node, &chain, (from, 0), &mut next).is_some() {
                        fails.push(format!("{text} [{}] step at {from}: a truth stepped", data.name));
                    }
                }
                // one row at a time
                for (row, &want) in whole.iter().enumerate().skip(from).take_while(|_| !truth) {
                    let mut next = chain.clone();
                    match execute_resume_step(df, &node, &chain, (row, 0), &mut next) {
                        Some(v) if same(v, want) => chain = next,
                        other => {
                            fails.push(format!("{text} [{}] step at {row}: {other:?} != {want}", data.name));
                            break;
                        }
                    }
                }
                if !truth && !same_chain(Some(&chain), initial_state(df, &node, n).as_ref()) {
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
