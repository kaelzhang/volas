//! Arithmetic NaNs in true range are values of the computation, not missing inputs.

use volas_compute::{indicators as ind, raw::indicators as raw};

fn same(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(), "row {i}: {a} != {b}");
    }
}

#[test]
fn true_range_keeps_the_initial_nan_and_zero_sign() {
    for value in [f64::INFINITY, f64::NEG_INFINITY] {
        let tr = raw::tr(&[3., value, 3.], &[1., value, 1.], &[2.; 3]);
        same(&tr, &[f64::NAN, f64::NAN, 2.]);
    }
    same(&raw::tr(&[0., -0.], &[0., 0.], &[0., 0.]), &[f64::NAN, -0.]);
    // A NaN in a later gap does not replace the initial infinite high-low range.
    same(&raw::tr(&[3., f64::INFINITY], &[1., 1.], &[f64::INFINITY, 2.]),
         &[f64::NAN, f64::INFINITY]);
}

#[test]
fn atr_includes_arithmetic_nan_in_its_fixed_seed() {
    let h = [3., f64::INFINITY, 3., 4., 3.];
    let l = [1., f64::INFINITY, 1., 1., 1.];
    let c = [2.; 5];
    assert!(raw::atr(&h, &l, &c, 2).iter().all(|v| v.is_nan()));
    // The seed exists even when its value is NaN; capture may not postpone it.
    same(&raw::atr_final_state(&h[..3], &l[..3], &c[..3], 2).unwrap(), &[f64::NAN]);
    same(&raw::atr(&h, &l, &c, 1), &raw::tr(&h, &l, &c));
    same(&raw::atr_final_state(&h[..2], &l[..2], &c[..2], 1).unwrap(), &[f64::NAN]);
}

#[test]
fn atr_and_natr_captures_and_both_resumes_match_full_at_every_split() {
    for bad in 0..12 {
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut h = [4.; 14];
            let mut l = [1.; 14];
            let c = [2.; 14];
            h[bad] = value;
            l[bad] = value;
            for period in [1, 2, 7] {
                let full = ind::atr(&h, &l, &c, period);
                let normalized = ind::natr(&h, &l, &c, period);
                for from in 1..14 {
                    let Some(state) = ind::atr_final_state(&h[..from], &l[..from], &c[..from], period) else {
                        continue;
                    };
                    let (tail, end) = ind::atr_resume(&h, &l, &c, period, from, &state).unwrap();
                    same(&tail, &full[from..]);
                    same(&end, &ind::atr_final_state(&h, &l, &c, period).unwrap());
                    let (tail, nend) = ind::natr_resume(&h, &l, &c, period, from, &state).unwrap();
                    same(&tail, &normalized[from..]);
                    same(&nend, &end);
                    let mut live = state;
                    for row in from..14 {
                        let mut next = [0.];
                        let got = ind::atr_resume_one(&h, &l, &c, period, row, &live, &mut next).unwrap();
                        same(&[got], &full[row..=row]);
                        same(&[ind::natr_value(got, c[row])], &normalized[row..=row]);
                        live.copy_from_slice(&next);
                    }
                    same(&live, &end);
                }
            }
        }
    }
}

#[test]
fn smma_one_preserves_input_and_resumes_after_infinity() {
    let x = [-0., f64::INFINITY, 2., f64::NAN, f64::NEG_INFINITY, 3., -0.];
    same(&ind::smma(&x, 1), &x);
    for from in 1..x.len() {
        let state = ind::smma_final_state(&x[..from], 1).unwrap();
        let (tail, end) = ind::smma_resume(&x, 1, from, &state);
        same(&tail, &x[from..]);
        same(&end, &[-0.]);
        let mut live = state;
        for row in from..x.len() {
            let mut next = [0.];
            same(&[ind::smma_resume_one(&x, 1, row, &live, &mut next).unwrap()], &x[row..=row]);
            live.copy_from_slice(&next);
        }
        same(&live, &end);
    }
}
