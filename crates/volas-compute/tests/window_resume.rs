//! The finite-window kernels that accumulate resume bit-exactly: the state captured over
//! any prefix, stepped one row at a time over the rest, reproduces the full computation —
//! on clean data and on data holding missing and infinite rows.

use volas_compute::raw::indicators::{self as ind, BollLine, FiniteMa, LinregLine};

const N: usize = 160;

struct Bars {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

fn bars(gappy: bool) -> Bars {
    let x = |i: usize| i as f64;
    let close: Vec<f64> = (0..N).map(|i| 1000.0 + 3.7 * (0.31 * x(i)).sin() + 0.013 * x(i) * x(i).sqrt()).collect();
    let mut b = Bars {
        open: close.iter().enumerate().map(|(i, c)| c - 0.41 * (0.7 * x(i)).cos()).collect(),
        high: close.iter().enumerate().map(|(i, c)| c + 1.3 + 0.27 * (1.3 * x(i)).sin().abs()).collect(),
        low: close.iter().enumerate().map(|(i, c)| c - 1.1 - 0.31 * (0.9 * x(i)).cos().abs()).collect(),
        volume: (0..N).map(|i| 500.0 + 77.7 * (0.5 * x(i)).sin().abs() * std::f64::consts::E).collect(),
        close,
    };
    if gappy {
        for i in [3, 40, 41, 97] {
            b.close[i] = f64::NAN;
        }
        b.high[60] = f64::NAN;
        b.volume[70] = f64::NAN;
        b.close[120] = f64::INFINITY;
    }
    b
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Resume `step` from `state(k)` (the state over the first `k` rows) for several `k`,
/// checking every row against `full`.
fn check(
    what: &str,
    full: &[f64],
    state: impl Fn(usize) -> Option<Vec<f64>>,
    step: impl Fn(usize, &[f64], &mut [f64]) -> Option<f64>,
) {
    for k in [0, 1, 5, 37, 80, N - 1] {
        let mut prev = state(k).unwrap_or_else(|| panic!("{what}: no state at {k}"));
        let mut next = prev.clone();
        for (row, &want) in full.iter().enumerate().skip(k) {
            let got = step(row, &prev, &mut next).unwrap_or_else(|| panic!("{what}: no step at {row}"));
            assert!(same(got, want), "{what} from {k}: row {row}: {got:?} vs {want:?}");
            std::mem::swap(&mut prev, &mut next);
        }
    }
}

#[test]
fn every_accumulating_kernel_resumes_bit_exactly() {
    for gappy in [false, true] {
        let b = bars(gappy);
        let (o, h, l, c, v) = (&b.open[..], &b.high[..], &b.low[..], &b.close[..], &b.volume[..]);
        let tag = |s: &str| format!("{s} gappy={gappy}");
        for (mt, p, full) in
            [(0, 5, ind::ma(c, 5)), (2, 5, ind::wma(c, 5)), (5, 6, ind::trima(c, 6)), (5, 7, ind::trima(c, 7))]
        {
            check(&tag(&format!("ma mt={mt} p={p}")), &full, |k| ind::ma_state(&c[..k], p, mt), |r, s, n| {
                ind::ma_step(c, p, mt, r, s, n)
            });
        }
        check(&tag("trima"), &ind::trima(c, 8), |k| ind::trima_state(&c[..k], 8), |r, s, n| ind::trima_step(c, 8, r, s, n));
        check(&tag("vwma"), &ind::vwma(c, v, 6), |k| ind::vwma_state(&c[..k], &v[..k], 6), |r, s, n| {
            ind::vwma_step(c, v, 6, r, s, n)
        });
        check(&tag("hma"), &ind::hma(c, 9), |k| ind::hma_state(&c[..k], 9), |r, s, n| ind::hma_step(c, 9, r, s, n));
        check(&tag("bbi"), &ind::bbi(c, 3, 6, 12, 24), |k| ind::bbi_state(&c[..k], 3, 6, 12, 24), |r, s, n| {
            ind::bbi_step(c, 3, 6, 12, 24, r, s, n)
        });
        for (line, full) in [
            (BollLine::Upper, ind::boll_upper(c, 20, 2.0)),
            (BollLine::Lower, ind::boll_lower(c, 20, 2.0)),
            (BollLine::Width, ind::bbw(c, 20)),
        ] {
            check(&tag(&format!("{line:?}")), &full, |k| ind::boll_band_state(&c[..k], 20), |r, s, n| {
                ind::boll_band_step(c, 20, 2.0, line, r, s, n)
            });
        }
        check(&tag("hv"), &ind::hv(c, 10, 60, 252), |k| ind::hv_state(&c[..k], 10), |r, s, n| {
            ind::hv_step(c, 10, 60, 252, r, s, n)
        });
        for upper in [true, false] {
            let full = if upper { ind::accbands_upper(h, l, 20) } else { ind::accbands_lower(h, l, 20) };
            check(&tag("accbands"), &full, |k| ind::accbands_band_state(&h[..k], &l[..k], 20, upper), |r, s, n| {
                ind::accbands_band_step(h, l, 20, upper, r, s, n)
            });
        }
        check(&tag("var"), &ind::var(c, 7), |k| ind::var_state(&c[..k], 7), |r, s, n| ind::var_step(c, 7, None, r, s, n));
        check(&tag("stddev"), &ind::stddev(c, 7, 1.5), |k| ind::var_state(&c[..k], 7), |r, s, n| {
            ind::var_step(c, 7, Some(1.5), r, s, n)
        });
        check(&tag("sum"), &ind::sum(c, 9), |k| ind::sum_state(&c[..k], 9), |r, s, n| ind::sum_step(c, 9, r, s, n));
        for (line, full) in [
            (LinregLine::Value, ind::linearreg(c, 9)),
            (LinregLine::Slope, ind::linearreg_slope(c, 9)),
            (LinregLine::Intercept, ind::linearreg_intercept(c, 9)),
            (LinregLine::Angle, ind::linearreg_angle(c, 9)),
            (LinregLine::Forecast, ind::tsf(c, 9)),
        ] {
            check(&tag(&format!("{line:?}")), &full, |k| ind::linreg_state(&c[..k], 9), |r, s, n| {
                ind::linreg_step(c, 9, line, r, s, n)
            });
        }
        check(&tag("correl"), &ind::correl(c, v, 11), |k| ind::correl_state(&c[..k], &v[..k], 11), |r, s, n| {
            ind::correl_step(c, v, 11, r, s, n)
        });
        check(&tag("beta"), &ind::beta(c, o, 11), |k| ind::beta_state(&c[..k], &o[..k], 11), |r, s, n| {
            ind::beta_step(c, o, 11, r, s, n)
        });
        check(&tag("cci"), &ind::cci(h, l, c, 14), |k| ind::cci_state(&h[..k], &l[..k], &c[..k], 14), |r, s, n| {
            ind::cci_step(h, l, c, 14, r, s, n)
        });
        check(&tag("mfi"), &ind::mfi(h, l, c, v, 14), |k| ind::mfi_state(&h[..k], &l[..k], &c[..k], &v[..k], 14), |r, s, n| {
            ind::mfi_step(h, l, c, v, 14, r, s, n)
        });
        check(&tag("cog"), &ind::cog(c, 10), |k| ind::cog_state(&c[..k], 10), |r, s, n| ind::cog_step(c, 10, r, s, n));
        check(&tag("cmf"), &ind::cmf(h, l, c, v, 20), |k| ind::cmf_state(&h[..k], &l[..k], &c[..k], &v[..k], 20), |r, s, n| {
            ind::cmf_step(h, l, c, v, 20, r, s, n)
        });
        check(&tag("dpo"), &ind::dpo(c, 20), |k| ind::dpo_state(&c[..k], 20), |r, s, n| ind::dpo_step(c, 20, r, s, n));
        check(&tag("emv"), &ind::emv(h, l, v, 14), |k| ind::emv_state(&h[..k], &l[..k], &v[..k], 14), |r, s, n| {
            ind::emv_step(h, l, v, 14, r, s, n)
        });
        check(&tag("kst"), &ind::kst(c), |k| ind::kst_state(&c[..k]), |r, s, n| ind::kst_step(c, r, s, n));
        check(&tag("coppock"), &ind::coppock(c, 10, 14, 11), |k| ind::coppock_state(&c[..k], 10, 14, 11), |r, s, n| {
            ind::coppock_step(c, 10, 14, 11, r, s, n)
        });
        for signal in [false, true] {
            let full = if signal {
                ind::relative_vigor_signal(o, h, l, c, 10)
            } else {
                ind::relative_vigor(o, h, l, c, 10)
            };
            check(
                &tag("rvi"),
                &full,
                |k| ind::relative_vigor_state(&o[..k], &h[..k], &l[..k], &c[..k], 10, signal),
                |r, s, n| ind::relative_vigor_step(o, h, l, c, 10, signal, r, s, n),
            );
        }
        for ma in [None, Some(10)] {
            let full = match ma {
                None => ind::dkx(o, h, l, c),
                Some(m) => ind::dkx_ma(o, h, l, c, m),
            };
            check(&tag("dkx"), &full, |k| ind::dkx_state(&o[..k], &h[..k], &l[..k], &c[..k], ma), |r, s, n| {
                ind::dkx_step(o, h, l, c, ma, r, s, n)
            });
        }
        check(
            &tag("wvad"),
            &ind::wvad(o, h, l, c, v, 24),
            |k| ind::wvad_state(&o[..k], &h[..k], &l[..k], &c[..k], &v[..k], 24),
            |r, s, n| ind::wvad_step(o, h, l, c, v, 24, r, s, n),
        );
        check(&tag("vr"), &ind::vr(c, v, 26), |k| ind::vr_state(&c[..k], &v[..k], 26), |r, s, n| {
            ind::vr_step(c, v, 26, r, s, n)
        });
        let fastk = ind::stoch_fastk(h, l, c, 5);
        for stages in [vec![(3, 0)], vec![(3, 0), (3, 0)], vec![(4, 2), (6, 5)]] {
            let mut full = match stages[0].1 {
                0 => ind::ma(&fastk, stages[0].0),
                _ => ind::wma(&fastk, stages[0].0),
            };
            if let Some(&(p, mt)) = stages.get(1) {
                full = if mt == 0 { ind::ma(&full, p) } else { ind::trima(&full, p) };
            }
            check(&tag("stoch"), &full, |k| ind::stoch_state(&h[..k], &l[..k], &c[..k], 5, &stages), |r, s, n| {
                ind::stoch_step(h, l, c, 5, &stages, r, s, n)
            });
        }
    }
}

#[test]
fn a_malformed_state_or_argument_declines() {
    let b = bars(false);
    let (o, h, l, c, v) = (&b.open[..], &b.high[..], &b.low[..], &b.close[..], &b.volume[..]);
    let short = [0.0; 1];
    let mut n1 = [0.0; 1];
    let mut big = [0.0; 64];
    // a recursive moving-average type is no finite-window stage
    assert!(FiniteMa::new(5, 1).is_none());
    assert!(ind::ma_state(c, 5, 1).is_none());
    assert!(ind::ma_step(c, 5, 0, N, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::ma_step(c, 5, 0, 1, &short, &mut big[..2]).is_none());
    assert!(ind::ma_step(c, 0, 0, 1, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::ma_step(c, 5, 2, 1, &short, &mut big[..4]).is_none());
    assert!(ind::trima_step(c, 6, N, &[0.0; 7], &mut big[..7]).is_none());
    assert!(ind::trima_state(c, 0).is_none());
    let trima = FiniteMa::new(6, 5).unwrap();
    assert_eq!(trima.reach(), 3);
    assert!(trima.step_chained(1, 0.0, &short, &mut n1).is_none());
    assert!(ind::vwma_step(c, v, 0, 1, &[0.0; 3], &mut big[..3]).is_none());
    assert!(ind::hma_state(c, 1).is_none());
    assert!(ind::hma_step(c, 9, 1, &short, &mut n1).is_none());
    assert!(ind::bbi_step(c, 3, 6, 12, 24, N, &[0.0; 8], &mut big[..8]).is_none());
    assert!(ind::boll_band_step(c, 0, 2.0, BollLine::Upper, 1, &[0.0; 3], &mut big[..3]).is_none());
    assert!(ind::hv_step(c, 1, 60, 252, 1, &[0.0; 3], &mut big[..3]).is_none());
    assert!(ind::accbands_band_step(h, l, 20, true, N, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::var_step(c, 0, None, 1, &[0.0; 3], &mut big[..3]).is_none());
    assert!(ind::sum_step(c, 5, N, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::sum_step(c, 0, 1, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::linreg_step(c, 0, LinregLine::Value, 1, &[0.0; 4], &mut big[..4]).is_none());
    assert!(ind::linreg_step(c, 5, LinregLine::Value, 1, &short, &mut big[..4]).is_none());
    assert!(ind::correl_step(c, v, 0, 1, &[0.0; 6], &mut big[..6]).is_none());
    assert!(ind::beta_step(c, v, 0, 1, &[0.0; 5], &mut big[..5]).is_none());
    assert!(ind::cci_step(h, l, c, 0, 1, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::mfi_step(h, l, c, v, 0, 1, &[0.0; 3], &mut big[..3]).is_none());
    assert!(ind::cog_step(c, 0, 1, &[0.0; 6], &mut big[..6]).is_none());
    assert!(ind::cog_step(c, 5, 1, &short, &mut big[..6]).is_none());
    assert!(ind::cmf_step(h, l, c, v, 5, N, &[0.0; 4], &mut big[..4]).is_none());
    assert!(ind::emv_step(h, l, v, 5, N, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::kst_step(c, N, &[0.0; 8], &mut big[..8]).is_none());
    assert!(ind::coppock_step(c, 10, 14, 11, N, &[0.0; 4], &mut big[..4]).is_none());
    assert!(ind::relative_vigor_step(o, h, l, c, 10, false, N, &[0.0; 4], &mut big[..4]).is_none());
    assert!(ind::dkx_step(o, h, l, c, None, N, &[0.0; 4], &mut big[..4]).is_none());
    assert!(ind::wvad_step(o, h, l, c, v, 5, N, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::vr_step(c, v, 5, N, &[0.0; 6], &mut big[..6]).is_none());
    assert!(ind::stoch_state(h, l, c, 5, &[]).is_none());
    assert!(ind::stoch_step(h, l, c, 5, &[(3, 0)], N, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::stoch_step(h, l, c, 5, &[(3, 0)], 1, &short, &mut big[..2]).is_none());
    // a window with no present extreme reads NaN
    let nan = [f64::NAN; 6];
    assert!(ind::stoch_fastk_at(&nan, &nan, c, 3, 4).is_nan());
    // the rule-applying single-row resume declines a `next` of the wrong length on a
    // missing row too
    let mut empty: [f64; 0] = [];
    assert!(volas_compute::indicators::ema_resume_one(&[f64::NAN], 3, 0, &[1.0], &mut empty).is_none());
    // a flat window's standard deviation is zero
    let flat = [5.0; 8];
    let st = ind::var_state(&flat[..7], 3).unwrap();
    assert_eq!(ind::var_step(&flat, 3, Some(2.0), 7, &st, &mut big[..3]), Some(0.0));
}
