//! The finite-window kernels that accumulate resume bit-exactly: the state captured over
//! any prefix, stepped one row at a time over the rest, reproduces the full computation —
//! on clean data and on data holding missing and infinite rows.

use volas_compute::raw::indicators::{self as ind, BollLine, MaStage, LinregLine};

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
    let close: Vec<f64> = (0..N)
        .map(|i| 1000.0 + 3.7 * (0.31 * x(i)).sin() + 0.013 * x(i) * x(i).sqrt())
        .collect();
    let mut b = Bars {
        open: close
            .iter()
            .enumerate()
            .map(|(i, c)| c - 0.41 * (0.7 * x(i)).cos())
            .collect(),
        high: close
            .iter()
            .enumerate()
            .map(|(i, c)| c + 1.3 + 0.27 * (1.3 * x(i)).sin().abs())
            .collect(),
        low: close
            .iter()
            .enumerate()
            .map(|(i, c)| c - 1.1 - 0.31 * (0.9 * x(i)).cos().abs())
            .collect(),
        volume: (0..N)
            .map(|i| 500.0 + 77.7 * (0.5 * x(i)).sin().abs() * std::f64::consts::E)
            .collect(),
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
            let got =
                step(row, &prev, &mut next).unwrap_or_else(|| panic!("{what}: no step at {row}"));
            assert!(
                same(got, want),
                "{what} from {k}: row {row}: {got:?} vs {want:?}"
            );
            std::mem::swap(&mut prev, &mut next);
        }
    }
}

/// Bars that alternate runs of huge and tiny values: once a run of huge values has
/// passed through a running sum, the sum cannot hold every bit of the tiny window that
/// follows, so its value depends on where it was last re-anchored — what makes a resume
/// that anchors anywhere else show.
fn wide() -> Bars {
    let scale = |i: usize, f: f64| if (i / 13).is_multiple_of(2) { 1e14 } else { 1e-8 } * (1.0 + 0.3 * (f * i as f64).sin());
    let close: Vec<f64> = (0..N).map(|i| 1000.0 * scale(i, 1.7)).collect();
    Bars {
        open: close
            .iter()
            .enumerate()
            .map(|(i, c)| c * (1.0 - 0.01 * (0.7 * i as f64).cos()))
            .collect(),
        high: close.iter().map(|c| c * 1.02).collect(),
        low: close.iter().map(|c| c * 0.97).collect(),
        volume: (0..N).map(|i| 500.0 * scale(i, 2.3)).collect(),
        close,
    }
}

#[test]
fn every_accumulating_kernel_resumes_bit_exactly() {
    for (kind, b) in [
        ("clean", bars(false)),
        ("gappy", bars(true)),
        ("wide", wide()),
    ] {
        for origin in [0usize, 1, 7] {
            let (o, h, l, c, v) = (
                &b.open[..],
                &b.high[..],
                &b.low[..],
                &b.close[..],
                &b.volume[..],
            );
            let tag = |s: &str| format!("{s} {kind} origin={origin}");
            for (mt, p, full) in [
                (0, 5, ind::ma(c, 5, origin)),
                (2, 5, ind::wma(c, 5, origin)),
                (5, 6, ind::trima(c, 6, origin)),
                (5, 7, ind::trima(c, 7, origin)),
            ] {
                check(
                    &tag(&format!("ma mt={mt} p={p}")),
                    &full,
                    |k| ind::ma_state(&c[..k], p, mt, origin),
                    |r, s, n| ind::ma_step(c, p, mt, r, s, n),
                );
            }
            check(
                &tag("trima"),
                &ind::trima(c, 8, origin),
                |k| ind::trima_state(&c[..k], 8, origin),
                |r, s, n| ind::trima_step(c, 8, r, s, n),
            );
            check(
                &tag("vwma"),
                &ind::vwma(c, v, 6, origin),
                |k| ind::vwma_state(&c[..k], &v[..k], 6, origin),
                |r, s, n| ind::vwma_step(c, v, 6, r, s, n),
            );
            check(
                &tag("hma"),
                &ind::hma(c, 9, origin),
                |k| ind::hma_state(&c[..k], 9, origin),
                |r, s, n| ind::hma_step(c, 9, r, s, n),
            );
            check(
                &tag("bbi"),
                &ind::bbi(c, 3, 6, 12, 24, origin),
                |k| ind::bbi_state(&c[..k], 3, 6, 12, 24, origin),
                |r, s, n| ind::bbi_step(c, 3, 6, 12, 24, r, s, n),
            );
            for (line, full) in [
                (BollLine::Upper, ind::boll_upper(c, 20, 2.0, origin)),
                (BollLine::Lower, ind::boll_lower(c, 20, 2.0, origin)),
                (BollLine::Width, ind::bbw(c, 20, origin)),
            ] {
                check(
                    &tag(&format!("{line:?}")),
                    &full,
                    |k| ind::boll_band_state(&c[..k], 20, origin),
                    |r, s, n| ind::boll_band_step(c, 20, 2.0, line, r, s, n),
                );
            }
            check(
                &tag("hv"),
                &ind::hv(c, 10, (252.0f64 * 24.0).sqrt(), origin),
                |k| ind::hv_state(&c[..k], 10, origin),
                |r, s, n| ind::hv_step(c, 10, (252.0f64 * 24.0).sqrt(), r, s, n),
            );
            for upper in [true, false] {
                let full = if upper {
                    ind::accbands_upper(h, l, 20, origin)
                } else {
                    ind::accbands_lower(h, l, 20, origin)
                };
                check(
                    &tag("accbands"),
                    &full,
                    |k| ind::accbands_band_state(&h[..k], &l[..k], 20, upper, origin),
                    |r, s, n| ind::accbands_band_step(h, l, 20, upper, r, s, n),
                );
            }
            check(
                &tag("var"),
                &ind::var(c, 7, origin),
                |k| ind::var_state(&c[..k], 7, origin),
                |r, s, n| ind::var_step(c, 7, None, r, s, n),
            );
            check(
                &tag("stddev"),
                &ind::stddev(c, 7, 1.5, origin),
                |k| ind::var_state(&c[..k], 7, origin),
                |r, s, n| ind::var_step(c, 7, Some(1.5), r, s, n),
            );
            check(
                &tag("sum"),
                &ind::sum(c, 9, origin),
                |k| ind::sum_state(&c[..k], 9, origin),
                |r, s, n| ind::sum_step(c, 9, r, s, n),
            );
            for (line, full) in [
                (LinregLine::Value, ind::linearreg(c, 9, origin)),
                (LinregLine::Slope, ind::linearreg_slope(c, 9, origin)),
                (
                    LinregLine::Intercept,
                    ind::linearreg_intercept(c, 9, origin),
                ),
                (LinregLine::Angle, ind::linearreg_angle(c, 9, origin)),
                (LinregLine::Forecast, ind::tsf(c, 9, origin)),
            ] {
                check(
                    &tag(&format!("{line:?}")),
                    &full,
                    |k| ind::linreg_state(&c[..k], 9, origin),
                    |r, s, n| ind::linreg_step(c, 9, line, r, s, n),
                );
            }
            check(
                &tag("correl"),
                &ind::correl(c, v, 11, origin),
                |k| ind::correl_state(&c[..k], &v[..k], 11, origin),
                |r, s, n| ind::correl_step(c, v, 11, r, s, n),
            );
            check(
                &tag("beta"),
                &ind::beta(c, o, 11, origin),
                |k| ind::beta_state(&c[..k], &o[..k], 11, origin),
                |r, s, n| ind::beta_step(c, o, 11, r, s, n),
            );
            check(
                &tag("cci"),
                &ind::cci(h, l, c, 14, origin),
                |k| ind::cci_state(&h[..k], &l[..k], &c[..k], 14, origin),
                |r, s, n| ind::cci_step(h, l, c, 14, r, s, n),
            );
            check(
                &tag("mfi"),
                &ind::mfi(h, l, c, v, 14, origin),
                |k| ind::mfi_state(&h[..k], &l[..k], &c[..k], &v[..k], 14, origin),
                |r, s, n| ind::mfi_step(h, l, c, v, 14, r, s, n),
            );
            check(
                &tag("cog"),
                &ind::cog(c, 10, origin),
                |k| ind::cog_state(&c[..k], 10, origin),
                |r, s, n| ind::cog_step(c, 10, r, s, n),
            );
            check(
                &tag("imi"),
                &ind::imi(o, c, 9, origin),
                |k| ind::imi_state(&o[..k], &c[..k], 9, origin),
                |r, s, n| ind::imi_step(o, c, 9, r, s, n),
            );
            check(
                &tag("ultosc"),
                &ind::ultosc(h, l, c, 4, 7, 11, origin),
                |k| ind::ultosc_state(&h[..k], &l[..k], &c[..k], 4, 7, 11, origin),
                |r, s, n| ind::ultosc_step(h, l, c, 4, 7, 11, r, s, n),
            );
            check(
                &tag("cmf"),
                &ind::cmf(h, l, c, v, 20, origin),
                |k| ind::cmf_state(&h[..k], &l[..k], &c[..k], &v[..k], 20, origin),
                |r, s, n| ind::cmf_step(h, l, c, v, 20, r, s, n),
            );
            check(
                &tag("dpo"),
                &ind::dpo(c, 20, origin),
                |k| ind::dpo_state(&c[..k], 20, origin),
                |r, s, n| ind::dpo_step(c, 20, r, s, n),
            );
            check(
                &tag("emv"),
                &ind::emv(h, l, v, 14, origin),
                |k| ind::emv_state(&h[..k], &l[..k], &v[..k], 14, origin),
                |r, s, n| ind::emv_step(h, l, v, 14, r, s, n),
            );
            check(
                &tag("kst"),
                &ind::kst(c, origin),
                |k| ind::kst_state(&c[..k], origin),
                |r, s, n| ind::kst_step(c, r, s, n),
            );
            check(
                &tag("coppock"),
                &ind::coppock(c, 10, 14, 11, origin),
                |k| ind::coppock_state(&c[..k], 10, 14, 11, origin),
                |r, s, n| ind::coppock_step(c, 10, 14, 11, r, s, n),
            );
            for signal in [false, true] {
                let full = if signal {
                    ind::relative_vigor_signal(o, h, l, c, 10, origin)
                } else {
                    ind::relative_vigor(o, h, l, c, 10, origin)
                };
                check(
                    &tag("rvi"),
                    &full,
                    |k| {
                        ind::relative_vigor_state(
                            &o[..k],
                            &h[..k],
                            &l[..k],
                            &c[..k],
                            10,
                            signal,
                            origin,
                        )
                    },
                    |r, s, n| ind::relative_vigor_step(o, h, l, c, 10, signal, r, s, n),
                );
            }
            for ma in [None, Some(10)] {
                let full = match ma {
                    None => ind::dkx(o, h, l, c, origin),
                    Some(m) => ind::dkx_ma(o, h, l, c, m, origin),
                };
                check(
                    &tag("dkx"),
                    &full,
                    |k| ind::dkx_state(&o[..k], &h[..k], &l[..k], &c[..k], ma, origin),
                    |r, s, n| ind::dkx_step(o, h, l, c, ma, r, s, n),
                );
            }
            check(
                &tag("wvad"),
                &ind::wvad(o, h, l, c, v, 24, origin),
                |k| ind::wvad_state(&o[..k], &h[..k], &l[..k], &c[..k], &v[..k], 24, origin),
                |r, s, n| ind::wvad_step(o, h, l, c, v, 24, r, s, n),
            );
            check(
                &tag("vr"),
                &ind::vr(c, v, 26, origin),
                |k| ind::vr_state(&c[..k], &v[..k], 26, origin),
                |r, s, n| ind::vr_step(c, v, 26, r, s, n),
            );
            let fastk = ind::stoch_fastk(h, l, c, 5);
            for stages in [vec![(3, 0)], vec![(3, 0), (3, 0)], vec![(4, 2), (6, 5)]] {
                let mut full = match stages[0].1 {
                    0 => ind::ma(&fastk, stages[0].0, origin),
                    _ => ind::wma(&fastk, stages[0].0, origin),
                };
                if let Some(&(p, mt)) = stages.get(1) {
                    full = if mt == 0 {
                        ind::ma(&full, p, origin)
                    } else {
                        ind::trima(&full, p, origin)
                    };
                }
                check(
                    &tag("stoch"),
                    &full,
                    |k| ind::stoch_state(&h[..k], &l[..k], &c[..k], 5, &stages, origin),
                    |r, s, n| ind::stoch_step(h, l, c, 5, &stages, r, s, n),
                );
            }
        }
    }
}

#[test]
fn a_malformed_state_or_argument_declines() {
    let b = bars(false);
    let (o, h, l, c, v) = (
        &b.open[..],
        &b.high[..],
        &b.low[..],
        &b.close[..],
        &b.volume[..],
    );
    let short = [0.0; 1];
    let mut n1 = [0.0; 1];
    let mut big = [0.0; 64];
    // a state or `next` of the wrong length, a zero period: no step
    assert!(ind::ultosc_step(h, l, c, 4, 7, 11, 5, &[0.0; 3], &mut big[..3]).is_none());
    assert!(ind::ultosc_step(h, l, c, 0, 7, 11, 5, &[0.0; 3], &mut big[..3]).is_none());
    assert!(ind::imi_step(o, c, 9, 5, &[0.0; 6], &mut big[..2]).is_none());
    assert!(ind::imi_step(o, c, 0, 5, &[0.0; 6], &mut big[..6]).is_none());
    assert!(ind::linreg_step(c, 9, LinregLine::Slope, 5, &[0.0; 3], &mut big[..7]).is_none());
    assert!(ind::stochrsi_final_state(c, 14, 0, false, 3, 0, 0).is_none());
    assert!(ind::stochrsi_final_state(c, 14, 5, true, 0, 0, 0).is_none());
    assert!(ind::ma_step(c, 0, 0, 1, &[0.0; 4], &mut big[..4]).is_none());
    // the shared stage supports recursive types too
    assert!(MaStage::new(5, 1).is_some());
    assert!(ind::ma_state(c, 5, 1, 0).is_some());
    assert!(ind::ma_step(c, 5, 0, N, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::ma_step(c, 5, 0, 1, &short, &mut big[..2]).is_none());
    assert!(ind::ma_step(c, 0, 0, 1, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::ma_step(c, 5, 2, 1, &short, &mut big[..4]).is_none());
    assert!(ind::trima_step(c, 6, N, &[0.0; 7], &mut big[..7]).is_none());
    assert!(ind::trima_state(c, 0, 0).is_none());
    let trima = MaStage::new(6, 5).unwrap();
    assert_eq!(trima.reach(), 3);
    assert!(trima.step_chained(1, 0.0, &short, &mut n1).is_none());
    assert!(ind::vwma_step(c, v, 0, 1, &[0.0; 3], &mut big[..3]).is_none());
    assert!(ind::hma_state(c, 1, 0).is_none());
    assert!(ind::hma_step(c, 9, 1, &short, &mut n1).is_none());
    assert!(ind::bbi_step(c, 3, 6, 12, 24, N, &[0.0; 8], &mut big[..8]).is_none());
    assert!(ind::boll_band_step(c, 0, 2.0, BollLine::Upper, 1, &[0.0; 3], &mut big[..3]).is_none());
    assert!(ind::hv_step(c, 1, 1.0, 1, &[0.0; 3], &mut big[..3]).is_none());
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
    assert!(ind::stoch_state(h, l, c, 5, &[], 0).is_none());
    assert!(ind::stoch_step(h, l, c, 5, &[(3, 0)], N, &[0.0; 2], &mut big[..2]).is_none());
    assert!(ind::stoch_step(h, l, c, 5, &[(3, 0)], 1, &short, &mut big[..2]).is_none());
    // a window with no present extreme reads NaN
    let nan = [f64::NAN; 6];
    assert!(ind::stoch_fastk_at(&nan, &nan, c, 3, 4).is_nan());
    // the rule-applying single-row resume declines a `next` of the wrong length on a
    // missing row too
    let mut empty: [f64; 0] = [];
    assert!(
        volas_compute::indicators::ema_resume_one(&[f64::NAN], 3, 0, &[1.0], &mut empty).is_none()
    );
    // a flat window's standard deviation is zero
    let flat = [5.0; 8];
    let st = ind::var_state(&flat[..7], 3, 0).unwrap();
    assert_eq!(
        ind::var_step(&flat, 3, Some(2.0), 7, &st, &mut big[..st.len()]),
        Some(0.0)
    );
}

/// A state carried into a frame that does not hold the rows it reads — its full window
/// and the row leaving it, here a frame starting just before the step — is declined,
/// never read below the frame's row 0 (a bounded window's resume too close to its head).
#[test]
fn a_state_reading_rows_before_the_frame_declines() {
    let c = &bars(false).close[..];
    let mut next = [0.0; 64];
    // states after 40 rows (each window full)
    let wma = ind::ma_state(&c[..40], 5, 2, 0).unwrap();
    let linreg = ind::linreg_state(&c[..40], 5, 0).unwrap();
    let cog = ind::cog_state(&c[..40], 5, 0).unwrap();
    // the same rows from the frame holding row 40 and the 5 before it: exact
    let tail = &c[35..];
    assert!(ind::ma_step(tail, 5, 2, 5, &wma, &mut next[..wma.len()]).is_some());
    assert!(ind::linreg_step(tail, 5, LinregLine::Value, 5, &linreg, &mut next[..linreg.len()]).is_some());
    assert!(ind::cog_step(tail, 5, 5, &cog, &mut next[..cog.len()]).is_some());
    // one row short: the row leaving the window is before the frame
    let short = &c[36..];
    assert!(ind::ma_step(short, 5, 2, 4, &wma, &mut next[..wma.len()]).is_none());
    assert!(ind::linreg_step(short, 5, LinregLine::Value, 4, &linreg, &mut next[..linreg.len()]).is_none());
    assert!(ind::cog_step(short, 5, 4, &cog, &mut next[..cog.len()]).is_none());
}

/// An extreme the carried tracker points at before the frame's row 0 has left every
/// window the frame holds: the resume rebuilds the extreme instead of placing it at row
/// 0, and the positions equal the full computation's.
#[test]
fn an_extreme_before_the_frame_is_rebuilt() {
    let data = [0.0, 0.0, 100.0, 1.0, 2.0, 0.0, 5.0];
    let low = [100.0, 100.0, 0.0, 3.0, 2.0, 4.0, 1.0];
    // the tracker after rows 0..=4 points at row 2 (100 / 0)
    let hi = ind::maxindex_final_state(&data[..5], 3, 0).unwrap();
    let lo = ind::minindex_final_state(&low[..5], 3, 0).unwrap();
    assert_eq!((hi[0], lo[0]), (2.0, 2.0));
    // a frame holding rows 3.. (origin 3) resumes at its row 2 (absolute row 5)
    let mut got = Vec::new();
    let _ = ind::maxindex_resume(&data[3..], 3, 2, 3, &hi, &mut got).unwrap();
    assert_eq!(got, ind::maxindex(&data, 3, 0)[5..].to_vec());
    let mut got = Vec::new();
    let _ = ind::minindex_resume(&low[3..], 3, 2, 3, &lo, &mut got).unwrap();
    assert_eq!(got, ind::minindex(&low, 3, 0)[5..].to_vec());
}

/// TA_MA resolves period one before validating its selector. Every such stage must
/// carry precisely the SMA state, while an unknown selector at other periods fails.
#[test]
fn stage_selector_boundary_preserves_identity_values_and_state() {
    let data = [3.0, 1.5, -0.0, f64::NAN, f64::INFINITY, 4.25];
    let sma = MaStage::new(1, 0).unwrap();
    let bits = |s: &[f64]| s.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for selector in 0..=9 {
        let stage = MaStage::new(1, selector).unwrap();
        let (mut state, mut reference) = (stage.initial(0), sma.initial(0));
        for row in 0..data.len() {
            let (mut next, mut want) = (state.clone(), reference.clone());
            let got = stage.step(row, |i| data[i], &state, &mut next).unwrap();
            let expected = sma.step(row, |i| data[i], &reference, &mut want).unwrap();
            assert_eq!(got.to_bits(), expected.to_bits(), "selector {selector}, row {row}");
            assert_eq!(bits(&next), bits(&want), "selector {selector}, state at {row}");
            (state, reference) = (next, want);
        }
    }
    assert!(MaStage::new(2, 9).is_none());
    assert!(MaStage::new(5, usize::MAX).is_none());
}
