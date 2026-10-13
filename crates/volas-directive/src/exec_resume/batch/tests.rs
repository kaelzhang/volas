use super::*;
use volas_core::{Column, DataFrame};

fn short_passes(text: &str) {
    let node = crate::parse(text).unwrap();
    let crate::types::Continuation::Own(cmd) = node.continuation() else { panic!("{text}") };
    let cmd = super::super::bound(cmd);
    for n in 0..=cmd.reach().max(cmd.spec.lookback_of(cmd.args)) + 1 {
        let names = ["open", "high", "low", "close", "volume", "periods"];
        let cols = names.iter().map(|_| Column::f64(vec![3.0; n])).collect();
        let df = DataFrame::new(names.iter().map(|s| s.to_string()).collect(), cols, None).unwrap();
        let ops = Operands::new(&df, cmd.series);
        if let Some(pass) = compute(&ops, &cmd) {
            let want = crate::exec::execute(&df, &node).unwrap().to_f64_vec();
            assert_eq!(pass.values.len(), want.len());
            assert!(pass.values.iter().zip(want).all(|(a, b)| *a == b || a.is_nan() && b.is_nan()), "{text} n={n}");
        }
    }
}

#[test]
fn stochrsi_k_declines_an_incomplete_rsi_tail() { short_passes("stochrsi.fastk:2,"); }

#[test]
fn stochrsi_d_declines_an_incomplete_smoothing_tail() { short_passes("stochrsi.d:2,1,8"); }

#[test]
fn averages_decline_an_incomplete_signal_tail() { short_passes("macdext.signal:2,1,3,1,8,0"); }

#[test]
fn stochastic_declines_an_incomplete_second_stage_tail() { short_passes("stoch.d:2,3,0,8,0"); }

#[test]
fn mavp_declines_empty_history_and_out_of_frame_stage_starts() { short_passes("mavp:2,8,1@close,periods"); }

#[test]
fn terminal_tail_publication_requires_the_before_last_reach() {
    for reach in 0..=8 {
        for n in 0..=reach + 1 {
            let values = vec![3.0; n];
            let got = tails(&values, reach);
            assert_eq!(got.is_some(), n > reach);
            if let Some((after, before)) = got {
                assert_eq!(after.len(), reach);
                assert_eq!(before.len(), reach);
            }
        }
    }
}
