use volas_compute::{indicators as ind, kernels};

fn assert_nan_prefix(values: &[f64], len: usize) {
    assert!(values[..len].iter().all(|v| v.is_nan()));
}

fn assert_f64_same(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
            "index {i}: {a:?} != {e:?}"
        );
    }
}

fn ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let close: Vec<f64> = (0..n)
        .map(|i| 100.0 + i as f64 * 0.2 + (i as f64 * 0.31).sin() * 3.0)
        .collect();
    let open: Vec<f64> = close
        .iter()
        .enumerate()
        .map(|(i, v)| v - 0.4 + (i as f64 * 0.13).cos() * 0.2)
        .collect();
    let high: Vec<f64> = open
        .iter()
        .zip(&close)
        .map(|(o, c)| o.max(*c) + 1.5)
        .collect();
    let low: Vec<f64> = open
        .iter()
        .zip(&close)
        .map(|(o, c)| o.min(*c) - 1.5)
        .collect();
    (open, high, low, close)
}

#[test]
fn rolling_max_min_van_herk_fast_path_matches_separate_kernels() {
    let (_, high, low, _) = ohlc(64);
    let (hh, ll) = kernels::rolling_max_min(&high, &low, 17);
    let want_h = kernels::rolling_max(ndarray::ArrayView1::from(&high), 17);
    let want_l = kernels::rolling_min(ndarray::ArrayView1::from(&low), 17);
    assert_f64_same(&hh, want_h.as_slice().unwrap());
    assert_f64_same(&ll, want_l.as_slice().unwrap());
}

#[test]
fn momentum_and_aroon_degenerate_guards_are_publicly_covered() {
    assert_nan_prefix(&ind::mom(&[1.0, 2.0], 2), 2);
    assert_nan_prefix(&ind::roc(&[1.0, 2.0], 2), 2);
    assert_eq!(ind::roc(&[0.0, 3.0], 1)[1], 0.0);
    assert_nan_prefix(&ind::aroon_down(&[1.0, 2.0], &[1.0, 0.5], 2), 2);
}

#[test]
fn stochastic_public_fast_paths_match_generic_components() {
    let (_, high, low, close) = ohlc(80);
    let fk = ind::stoch_fastk(&high, &low, &close, 5);
    assert_nan_prefix(&fk, 4);
    assert!(fk[4].is_finite());

    let generic = ind::stoch_fastk(&high, &low, &close, 20);
    assert_nan_prefix(&generic, 19);
    assert!(generic[20].is_finite());

    let slow_d = ind::stoch_d_default_sma(&high, &low, &close).unwrap();
    assert_nan_prefix(&slow_d, 8);
    assert!(slow_d[8].is_finite());

    let fast_d = ind::stochf_d_default_sma(&high, &low, &close).unwrap();
    assert_nan_prefix(&fast_d, 6);
    assert!(fast_d[6].is_finite());
}

#[test]
fn short_hilbert_and_sar_bootstrap_edges_are_covered() {
    let short = vec![1.0; 12];
    let (inphase, quad) = ind::ht_phasor(&short);
    assert!(inphase.iter().all(|v| v.is_nan()));
    assert!(quad.iter().all(|v| v.is_nan()));
    assert!(ind::ht_phasor_line(&short, false)
        .iter()
        .all(|v| v.is_nan()));
    assert!(ind::ht_trendline(&short).iter().all(|v| v.is_nan()));
    let (mama, fama) = ind::mama(&short, 0.5, 0.05);
    assert!(mama.iter().all(|v| v.is_nan()));
    assert!(fama.iter().all(|v| v.is_nan()));

    let high = vec![10.0, 9.0, 8.0, 8.5, 7.5];
    let low = vec![9.0, 7.0, 6.0, 7.0, 5.0];
    let sar = ind::sar(&high, &low, 0.02, 0.2);
    assert_eq!(sar[1], high[0]);
    let state = ind::sar_final_state(&high, &low, 0.02, 0.2).unwrap();
    assert_eq!(state[0], 0.0);

    let sx = ind::sarext(&high, &low, 0.0, 0.1, 0.5, 0.5, 0.1, 0.5, 0.5, 0.1);
    assert!(sx[1].is_sign_negative());
    let state =
        ind::sarext_final_state(&high, &low, 0.0, 0.0, 0.5, 0.5, 0.1, 0.5, 0.5, 0.1).unwrap();
    assert!(state[0] == 0.0 || state[0] == 1.0);
}

#[test]
fn short_candlestick_patterns_cover_output_guards() {
    let (open, high, low, close) = ohlc(1);
    for out in [
        ind::cdl_engulfing(&open, &high, &low, &close),
        ind::cdl_piercing(&open, &high, &low, &close),
        ind::cdl_darkcloudcover(&open, &high, &low, &close, 0.5),
        ind::cdl_inneck(&open, &high, &low, &close),
        ind::cdl_hangingman(&open, &high, &low, &close),
        ind::cdl_2crows(&open, &high, &low, &close),
        ind::cdl_stalledpattern(&open, &high, &low, &close),
        ind::cdl_tristar(&open, &high, &low, &close),
        ind::cdl_tasukigap(&open, &high, &low, &close),
        ind::cdl_3linestrike(&open, &high, &low, &close),
    ] {
        assert_eq!(out.len(), 1);
        assert!(out[0].is_nan());
    }
}

#[test]
fn hikkake_warmup_confirmation_resets_pending_setup() {
    let open = vec![5.0; 6];
    let high = vec![10.0, 8.0, 7.0, 9.0, 9.0, 9.0];
    let low = vec![0.0, 2.0, 1.0, 1.0, 1.0, 1.0];
    let close = vec![5.0, 5.0, 5.0, 9.0, 5.0, 5.0];
    let out = ind::cdl_hikkake(&open, &high, &low, &close);
    assert!(out[..5].iter().all(|v| v.is_nan()));
    assert_eq!(out[5], 0.0);
}

#[test]
fn stochastic_public_fallbacks_cover_none_and_flat_generic_paths() {
    let (mut open, mut high, low, close) = ohlc(32);
    high[4] = f64::NAN;
    let out = ind::stoch_fastk(&high, &low, &close, 5);
    assert_eq!(out.len(), 32);

    open.fill(4.0);
    let out = ind::stoch_fastk(&open, &open, &open, 20);
    assert_eq!(out[19], 0.0);

    let fk = ind::stochrsi_fastk(&close, 14, 5);
    assert_eq!(fk.len(), close.len());
    assert!(fk.iter().any(|v| v.is_finite()));
    let short_fk = ind::stochrsi_fastk(&close[..20], 14, 20);
    assert!(short_fk.iter().all(|v| v.is_nan()));

    let increasing = (0..80).map(|i| 100.0 + i as f64).collect::<Vec<_>>();
    let out = ind::stochrsi_d_default_sma(&increasing).unwrap();
    assert!(out.contains(&0.0));
}

#[test]
fn atr_resume_one_guards_and_matches_vec_resume() {
    let h = [10.0, 11.0, 12.0];
    let l = [9.0, 9.5, 10.0];
    let c = [9.5, 10.5, 11.5];
    // None guards (row 0, empty state, row out of range) mirror `atr_resume`.
    assert!(ind::atr_resume_one(&h, &l, &c, 14, 0, &[5.0]).is_none());
    assert!(ind::atr_resume_one(&h, &l, &c, 14, 1, &[]).is_none());
    assert!(ind::atr_resume_one(&h, &l, &c, 14, 3, &[5.0]).is_none());
    // The scalar step is bit-identical to the two-`Vec` `atr_resume`'s single step.
    let (tail, _) = ind::atr_resume(&h, &l, &c, 14, 2, &[5.0]).unwrap();
    let one = ind::atr_resume_one(&h, &l, &c, 14, 2, &[5.0]).unwrap();
    assert_eq!(one.to_bits(), tail[0].to_bits());
}
