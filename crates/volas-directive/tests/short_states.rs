//! Every command's short-history publication must decline or continue exactly.
#[allow(dead_code)]
mod matrix;
use volas_core::{Column, DataFrame};
use volas_directive::{parse, exec::{execute, execute_with_states, execute_resume}, lookback::lookback};

fn frame(n: usize) -> DataFrame {
    let names = ["open", "high", "low", "close", "volume", "periods"];
    let cols = names.iter().enumerate().map(|(k, _)| Column::f64((0..n).map(|i| if k == 5 { 3.0 } else { 100.0 + ((i * 7) % 13) as f64 + k as f64 }).collect())).collect();
    DataFrame::new(names.iter().map(|s| s.to_string()).collect(), cols, None).unwrap()
}
fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a,b)| a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan())
}
#[test]
fn every_command_publishes_safe_short_history_states() {
    let mut texts: Vec<_> = matrix::lines().iter().flat_map(matrix::variants).collect();
    texts.extend(["stochrsi.fastk:2,", "stochrsi.d:2,5,8", "stoch.d:2,3,0,8,0", "macdext.signal:2,1,3,1,8,0", "mavp:2,8,1@close,periods", "ema:3+ma:2", "ema:3@(ma:2)"].map(str::to_owned));
    for text in texts {
        let node = parse(&text).unwrap();
        let reach = volas_directive::lookback::reach(&node).resume;
        for n in 0..=reach.max(lookback(&node)) + 1 {
            let head = frame(n);
            if volas_directive::lookback::history(&node).recursive() {
                let mut sparse = frame(2 * (lookback(&node) + 1) + 2 + n);
                let total = sparse.height();
                for name in ["open", "high", "low", "close", "volume"] {
                    let mut values = sparse.column(name).unwrap().to_f64_vec();
                    values[..total - n].fill(f64::NAN);
                    sparse.set_column(name, Column::f64(values)).unwrap();
                }
                let (got, _, _) = execute_with_states(&sparse, &node).unwrap();
                assert!(same(&got.to_f64_vec(), &execute(&sparse, &node).unwrap().to_f64_vec()), "sparse {text} n={n}");
            }
            let (got, after, _) = execute_with_states(&head, &node).unwrap();
            assert!(same(&got.to_f64_vec(), &execute(&head, &node).unwrap().to_f64_vec()), "{text} n={n}");
            if let Some(state) = after {
                let next = frame(n + 1);
                if let Some((tail, _)) = execute_resume(&next, &node, &state, n, n+1, 0) {
                    assert!(same(&tail.to_f64_vec(), &execute(&next, &node).unwrap().to_f64_vec()[n..]), "{text} n={n}");
                }
            }
        }
    }
}
