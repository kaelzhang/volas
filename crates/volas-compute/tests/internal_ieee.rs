//! Input admission must not classify a finite-input indicator's arithmetic as missing.
use volas_compute::raw::indicators as ind;

#[test]
fn cmf_witness_admits_original_inputs_even_when_the_range_has_no_flow() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_dead_beef)] {
        for column in 0..4 {
            let mut inputs = [[2.0; 6], [2.0; 6], [1.5; 6], [10.0; 6]];
            inputs[column][2] = bad;
            let [high, low, close, volume] = inputs;
            let out = ind::cmf(&high, &low, &close, &volume, 2, 0);
            assert!(out[0].is_nan());
            assert_eq!(out[1], 0.0);
            assert!(out[2].is_nan() && out[3].is_nan());
            assert_eq!(out[4], 0.0);
            assert_eq!(out[5], 0.0);
            let pass = volas_compute::pass::KernelPass::run(|s| ind::cmf_pass_with(
                &high[..3], &low[..3], &close[..3], &volume[..3], 2, 0, false, s,
            )).unwrap();
            let mut state = pass.after;
            let mut next = state.clone();
            for (row, expected) in out.iter().enumerate().skip(3) {
                let got = ind::cmf_step(&high, &low, &close, &volume, 2, row, &state, &mut next).unwrap();
                assert_eq!(got.is_nan(), expected.is_nan());
                if !expected.is_nan() { assert_eq!(got.to_bits(), expected.to_bits()); }
                std::mem::swap(&mut state, &mut next);
            }
        }
    }
    // A finite range or location can overflow while the guarded flow stays zero.
    let volume = [1.0; 3];
    for (high, low, close) in [([f64::MAX; 3], [-f64::MAX; 3], [0.0; 3]), ([f64::MAX; 3], [f64::MAX; 3], [-f64::MAX; 3])] {
        for complete in [false, true] {
            let out = ind::cmf_pass_with(&high, &low, &close, &volume, 1, 0, complete, &mut ());
            assert!(out.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
        }
    }
    // A finite-volume sum can also overflow: it must not turn the zero flow missing.
    let out = ind::cmf(&[2.0; 6], &[2.0; 6], &[1.5; 6], &[f64::MAX; 6], 2, 0);
    assert!(out[0].is_nan());
    assert!(out[1..].iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
}

#[test]
fn mfi_missing_moving_inputs_expire_at_the_original_read_window() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_dead_beef)] {
        for column in 0..4 {
            let mut inputs: [[f64; 8]; 4] = [
                std::array::from_fn(|i| i as f64 + 2.0),
                std::array::from_fn(|i| i as f64),
                std::array::from_fn(|i| i as f64 + 1.0),
                [1.0; 8],
            ];
            inputs[column][2] = bad;
            let [high, low, close, volume] = inputs;
            let out = ind::mfi(&high, &low, &close, &volume, 2, 0);
            let recovered = if column == 3 { 4 } else { 5 };
            assert!(out[..recovered].iter().all(|v| v.is_nan()));
            assert!(out[recovered..].iter().all(|v| v.to_bits() == 100.0_f64.to_bits()));
        }
        let prices = [1.0, 2.0, 3.0, 4.0];
        let volume = [bad, 1.0, 1.0, 1.0];
        let out = ind::mfi(&prices, &prices, &prices, &volume, 2, 0);
        assert_eq!(&out[2..], &[100.0, 100.0]);
    }
}

#[test]
fn noncanonical_nan_inputs_are_missing_before_output_publication() {
    for bits in [0x7ff8_0000_dead_beef, 0x7ff8_0000_0000_1234, 0xfff8_0000_0000_5678] {
        let nan = f64::from_bits(bits);
        let close = [1.0, nan, 2.0, 3.0];
        let out = ind::roc(&close, 1);
        assert!(out[..3].iter().all(|v| v.is_nan()));
        assert_eq!(out[3], 50.0);
        let low = [0.0; 4];
        let volume = [100_000_000.0; 4];
        let out = ind::emv(&close, &low, &volume, 1, 0);
        assert!(out[..3].iter().all(|v| v.is_nan()));
        assert_eq!(out[3], 1.5);
        for out in [ind::accbands_upper(&close, &low, 1, 0), ind::accbands_lower(&close, &low, 1, 0)] {
            assert!(out[1].is_nan());
            assert!(out[0].is_finite() && out[2..].iter().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn price_admission_preserves_finite_overflow_and_signed_zero() {
    let x = [f64::MAX, -f64::MAX, -0.0, 1.0];
    for out in [
        ind::avgprice(&x, &x, &x, &x), ind::avgprice_complete(&x, &x, &x, &x),
        ind::medprice(&x, &x), ind::medprice_complete(&x, &x),
        ind::typprice(&x, &x, &x), ind::typprice_complete(&x, &x, &x),
        ind::wclprice(&x, &x, &x), ind::wclprice_complete(&x, &x, &x),
    ] {
        for (actual, expected) in out.into_iter().zip([f64::INFINITY, f64::NEG_INFINITY, -0.0, 1.0]) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    let x = [f64::INFINITY, f64::NAN, f64::NEG_INFINITY, 1.0];
    for out in [ind::avgprice(&x, &x, &x, &x), ind::medprice(&x, &x), ind::typprice(&x, &x, &x), ind::wclprice(&x, &x, &x)] {
        assert!(out[..3].iter().all(|v| v.is_nan()));
        assert_eq!(out[3], 1.0);
    }
}

#[test]
fn tr_fused_admission_does_not_hide_a_missing_prior_close() {
    let high = [f64::MAX, f64::MAX, 1.0, 3.0];
    let low = [-f64::MAX, -f64::MAX, 0.0, 1.0];
    let close = [0.0; 4];
    for out in [ind::tr(&high, &low, &close), ind::tr_complete(&high, &low, &close)] {
        assert!(out[0].is_nan());
        assert_eq!(&out[1..], &[f64::INFINITY, 1.0, 3.0]);
    }
    for missing in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let close = [0.0, missing, 0.0, 0.0];
        let out = ind::tr(&high, &low, &close);
        assert_eq!(out[1], f64::INFINITY);
        assert!(out[2].is_nan());
        assert_eq!(out[3], 3.0);
    }
}

#[test]
fn mfi_unknown_admission_cannot_hide_a_nonfinite_flat_flow() {
    let price = [1.0; 8];
    for missing in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut volume = [1.0; 8];
        volume[4] = missing;
        let out = ind::mfi_pass_with(&price, &price, &price, &volume, 2, 0, false, &mut ());
        for (row, value) in out.into_iter().enumerate() {
            if row < 2 || row == 4 || row == 5 {
                assert!(value.is_nan(), "row {row}, volume {missing}");
            } else {
                assert_eq!(value, 0.0, "row {row}, volume {missing}");
            }
        }
        let prices = [missing; 8];
        assert!(ind::mfi(&prices, &prices, &prices, &price, 2, 0).iter().all(|v| v.is_nan()));
    }
}

#[test]
fn generated_product_infinity_is_preserved_in_vwma_batch_and_step() {
    let c = [1.0, 1.0, f64::MAX, 1.0, 1.0];
    let v = [2.0; 5];
    let out = ind::vwma(&c, &v, 2, 0);
    assert_eq!(out[2], f64::INFINITY);
    assert_eq!(out[3], f64::INFINITY);
    assert_eq!(out[4], 1.0);
    let state = ind::vwma_state(&c[..2], &v[..2], 2, 0).unwrap();
    let mut next = state.clone();
    let value = ind::vwma_step(&c, &v, 2, 2, &state, &mut next).unwrap();
    assert_eq!(value.to_bits(), out[2].to_bits());
}

#[test]
fn generated_typical_price_infinity_is_a_flat_mfi_term() {
    let x = [f64::MAX; 8];
    let v = [1.0; 8];
    for out in [ind::mfi(&x, &x, &x, &v, 2, 0), ind::mfi_complete(&x, &x, &x, &v, 2, 0)] {
        assert!(out[..2].iter().all(|x| x.is_nan()));
        assert!(out[2..].iter().all(|&x| x == 0.0));
    }
    let state = ind::mfi_state(&x[..4], &x[..4], &x[..4], &v[..4], 2, 0).unwrap();
    let mut next = state.clone();
    assert_eq!(ind::mfi_step(&x, &x, &x, &v, 2, 4, &state, &mut next), Some(0.0));
}

#[test]
fn generated_acceleration_edge_infinity_survives_its_internal_mean() {
    let h = [f64::MAX / 3.0, f64::MAX / 3.0, f64::MAX / 3.0, 1.0, 1.0, 1.0];
    let l = [0.0; 6];
    let out = ind::accbands_upper(&h, &l, 2, 0);
    assert!(out[0].is_nan());
    assert!(out[1..4].iter().all(|&x| x == f64::INFINITY));
    assert!(out[4..].iter().all(|&x| x.to_bits() == 5.0f64.to_bits()));
    let state = ind::accbands_band_state(&h[..2], &l[..2], 2, true, 0).unwrap();
    let mut next = state.clone();
    assert_eq!(ind::accbands_band_step(&h, &l, 2, true, 2, &state, &mut next), Some(f64::INFINITY));
}

#[test]
fn generated_kst_zero_base_ratio_infinity_matches_every_checkpoint_step() {
    let mut close = vec![0.0; 40];
    close.extend([1.0; 60]);
    let full = ind::kst(&close, 0);
    assert_eq!(full[54], f64::INFINITY);
    assert_eq!(full[99].to_bits(), 0.0f64.to_bits());
    let mut state = ind::kst_state(&close[..45], 0).unwrap();
    let mut next = state.clone();
    for (row, expected) in full.iter().enumerate().skip(45) {
        let got = ind::kst_step(&close, row, &state, &mut next).unwrap();
        assert_eq!(got.is_nan(), expected.is_nan(), "row {row}");
        if !expected.is_nan() { assert_eq!(got.to_bits(), expected.to_bits(), "row {row}"); }
        std::mem::swap(&mut state, &mut next);
    }
}

#[test]
fn imi_admits_original_prices_and_preserves_generated_down_overflow() {
    let c = [-f64::MAX; 3];
    let o = [f64::MAX; 3];
    let out = ind::imi(&o, &c, 2, 0);
    assert_eq!(out[1].to_bits(), 0.0f64.to_bits());
    let bad = [f64::INFINITY; 3];
    assert!(ind::imi(&bad, &c, 2, 0).iter().all(|x| x.is_nan()));
    let state = ind::imi_state(&o[..2], &c[..2], 2, 0).unwrap();
    let mut next = state.clone();
    assert_eq!(ind::imi_step(&o, &c, 2, 2, &state, &mut next), Some(0.0));
    assert!(ind::imi_step(&bad, &c, 2, 2, &state, &mut next).unwrap().is_nan());
}

#[test]
fn generated_stochastic_infinity_declines_the_fused_sum_and_recovers_at_the_stage() {
    let h = [2.0; 12];
    let l = [0.0; 12];
    let mut c = [1.0; 12];
    c[6] = f64::MAX;
    let k = ind::stoch_fastk(&h, &l, &c, 5);
    assert_eq!(k[6], f64::INFINITY);
    assert!(ind::stochf_d_default_sma(&h, &l, &c, 0).is_none());
    assert!(ind::stoch_d_default_sma(&h, &l, &c, 0).is_none());
    let stage = ind::MaStage::new(3, 0).unwrap();
    let d = stage.pass(&k, 0, &mut ()).unwrap();
    assert!(d[6..9].iter().all(|x| x.is_nan()));
    assert!(d[9..].iter().all(|&x| x.to_bits() == 50.0f64.to_bits()));
}

#[test]
fn emv_preserves_generated_midpoint_infinity_and_recovers() {
    let low = [0.0, f64::MAX / 2.0, 0.0, 0.0];
    let volume = [100_000_000.0; 4];
    let finite = [2.0, f64::MAX, 2.0, 2.0];
    let out = ind::emv(&finite, &low, &volume, 1, 0);
    assert!(out[0].is_nan());
    assert_eq!(out[1], f64::INFINITY);
    assert_eq!(out[2], f64::NEG_INFINITY);
    assert_eq!(out[3].to_bits(), 0.0f64.to_bits());
    let external = [2.0, f64::INFINITY, 2.0, 2.0];
    let out = ind::emv(&external, &low, &volume, 1, 0);
    assert!(out[..3].iter().all(|x| x.is_nan()));
    assert_eq!(out[3].to_bits(), 0.0f64.to_bits());
}

#[test]
fn admitted_momentum_entries_keep_zero_bases_and_finite_input_overflow() {
    let x = [0.0, f64::MAX, -f64::MAX, 0.0];
    let mom = ind::mom_complete(&x, 1);
    assert!(mom[0].is_nan());
    assert_eq!(&mom[1..], &[f64::MAX, f64::NEG_INFINITY, f64::MAX]);
    let rocp = ind::rocp_complete(&x, 1);
    assert!(rocp[0].is_nan());
    assert_eq!(rocp[1].to_bits(), 0.0f64.to_bits());
    assert_eq!(&rocp[2..], &[-2.0, -1.0]);
}

#[test]
fn psy_raw_entry_recovers_after_both_missing_comparisons_leave() {
    let close = [0.0, 1.0, 2.0, f64::INFINITY, 3.0, 4.0, 5.0];
    let out = ind::psy(&close, 2, 0);
    assert!(out[..2].iter().all(|x| x.is_nan()));
    assert_eq!(out[2], 100.0);
    assert!(out[3..6].iter().all(|x| x.is_nan()));
    assert_eq!(out[6], 100.0);
}

#[test]
fn an_overflowed_admission_witness_preserves_finite_momentum_values() {
    for sign in [-1.0, 1.0] {
        let x = [sign * f64::MAX; 2];
        assert_eq!(ind::mom(&x, 1)[1].to_bits(), 0.0f64.to_bits());
        assert_eq!(ind::roc(&x, 1)[1].to_bits(), 0.0f64.to_bits());
        assert_eq!(ind::rocp(&x, 1)[1].to_bits(), 0.0f64.to_bits());
        assert_eq!(ind::rocr(&x, 1)[1], 1.0);
        assert_eq!(ind::rocr100(&x, 1)[1], 100.0);
    }
}

#[test]
fn trix_keeps_zero_base_and_generated_infinity_before_internal_nan() {
    for sign in [-1.0, 1.0] {
        let close = [sign, 0.0, 0.0, sign * f64::MAX, -sign * f64::MAX, sign, sign];
        let full = ind::trix(&close, 1);
        assert!(full[0].is_nan());
        assert_eq!(full[1], -100.0);
        assert!(full[2..4].iter().all(|x| x.to_bits() == 0.0f64.to_bits()));
        assert_eq!(full[4], f64::NEG_INFINITY);
        assert!(full[5..].iter().all(|x| x.is_nan()));
        let state = ind::trix_final_state(&close[..4], 1).unwrap();
        let mut resumed = full[..4].to_vec();
        ind::trix_resume(&close, 1, 4, &state, &mut resumed);
        for (got, expected) in resumed.iter().zip(&full) {
            assert_eq!(got.is_nan(), expected.is_nan());
            if !expected.is_nan() { assert_eq!(got.to_bits(), expected.to_bits()); }
        }
    }
}

#[test]
fn acceleration_fixup_preserves_each_vector_lane_and_finite_overflow() {
    for len in 0..=70 {
        for upper in [false, true] {
            let edge = |h: &[f64], l: &[f64]| if upper {
                ind::accbands_upper(h, l, 1, 0)
            } else {
                ind::accbands_lower(h, l, 1, 0)
            };
            let h = vec![3.0; len];
            let l = vec![1.0; len];
            let clean = edge(&h, &l);
            for lane in 0..len {
                for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    for high_input in [false, true] {
                        let (mut h, mut l) = (h.clone(), l.clone());
                        if high_input { h[lane] = bad; } else { l[lane] = bad; }
                        let got = edge(&h, &l);
                        assert!(got[lane].is_nan());
                        for i in 0..len {
                            if i != lane { assert_eq!(got[i].to_bits(), clean[i].to_bits()); }
                        }
                    }
                }
            }
            if len > 0 {
                let (h, l) = if upper { (vec![f64::MAX / 3.0; len], vec![0.0; len]) }
                    else { (vec![0.0; len], vec![-f64::MAX / 3.0; len]) };
                assert!(edge(&h, &l).iter().all(|x| x.is_infinite()));
            }
        }
    }
}

#[test]
fn mike_specialized_lines_keep_read_sets_and_ieee_order() {
    use ind::MikeLine::{MidR, MidS, StrongR, StrongS, WeakR, WeakS};
    let lines = [WeakR, MidR, StrongR, WeakS, MidS, StrongS];
    let expected = [[9.0, 6.0, 10.0], [12.0, 10.0, 14.0], [15.0, 14.0, 18.0],
        [2.0, 0.0, 2.0], [-2.0, -2.0, -2.0], [-6.0, -4.0, -6.0]];
    let high = [5.0, 8.0, 7.0, 10.0];
    let low = [1.0, 2.0, 2.0, 3.0];
    let close = [3.0, 5.0, 3.0, 5.0];
    for (k, line) in lines.into_iter().enumerate() {
        for complete in [false, true] {
            let got = ind::mike_with(&high, &low, &close, 2, line, complete);
            assert!(got[0].is_nan());
            assert_eq!(&got[1..], &expected[k]);
            let huge = [f64::MAX; 4];
            let got = ind::mike_with(&huge, &huge, &huge, 2, line, complete);
            assert!(got[1..].iter().all(|x| *x == f64::INFINITY));
            let zero = [-0.0; 40];
            let got = ind::mike_with(&zero, &zero, &zero, 2, line, complete);
            let sign: f64 = if k == 0 || k == 3 || k == 4 { -0.0 } else { 0.0 };
            assert!(got[1..].iter().all(|x| x.to_bits() == sign.to_bits()), "line={k} complete={complete} bits={:?}", got.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_dead_beef)] {
            for field in 0..3 {
                let (mut h, mut l, mut c) = (high, low, close);
                match field { 0 => h[2] = bad, 1 => l[2] = bad, _ => c[2] = bad }
                let got = ind::mike(&h, &l, &c, 2, line);
                assert!(got[0].is_nan());
                assert_eq!(got[1], expected[k][0]);
                if field == 2 && (k == 2 || k == 5) { assert_eq!(got[2], expected[k][1]); }
                else { assert!(got[2].is_nan()); }
                if field == 2 || (field == 0 && k == 0) || (field == 1 && k == 3) {
                    assert_eq!(got[3], expected[k][2]);
                } else { assert!(got[3].is_nan()); }
            }
        }
    }
}

#[test]
fn swma_normalizes_missing_nan_payloads_before_publishing_its_output() {
    for n in 0..=12 {
        let clean: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        for at in 0..n {
            let mut expected = clean.clone();
            expected[at] = f64::NAN;
            let expected = ind::swma(&expected);
            for bits in [0x7ff8_0000_dead_beef, 0xfff8_0000_dead_beef, 0x7ff8_0000_0000_1234] {
                let mut input = clean.clone();
                input[at] = f64::from_bits(bits);
                let actual = ind::swma(&input);
                assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            }
        }
    }
    assert_eq!(ind::swma(&[f64::MAX; 4])[3], f64::INFINITY);
}
