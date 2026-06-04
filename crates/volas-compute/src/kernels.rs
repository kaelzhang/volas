//! Core rolling / moving-window kernels operating on `f64` arrays.
//!
//! `NaN` denotes a missing value. Sum / std windows are valid only when fully
//! populated; min / max reduce over the values present. All kernels are O(n):
//! sum / std slide a running accumulator, min / max use a monotonic deque —
//! never the O(n·period) per-window re-scan they were ported from.
//!
//! The kernels are deliberately scalar. The EWMA recurrence is division-latency
//! bound, and the two independent chains macd / rsi need ([`dual_ewma`]) are
//! already issued in parallel by the out-of-order core (ILP); a measured `f64x2`
//! SIMD dual-EWMA came out 1.00x, so explicit SIMD buys nothing here.

use ndarray::{Array1, ArrayView1};
use std::collections::VecDeque;

/// Simple moving average (a window is valid only when fully populated).
///
/// O(n) sliding running sum: each step adds the entering value and subtracts
/// the leaving one. A running count of in-window NaNs gates emission so the
/// result is identical to a per-window re-sum (any NaN in a window -> NaN).
#[inline]
pub fn sma(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    if period == 0 || period > n {
        return result;
    }
    // Fast path — a clean sliding sum with no per-element NaN bookkeeping (same
    // accumulation order, so bit-identical for NaN-free data, ~3.6x faster — the
    // common OHLCV case). A NaN poisons the running sum permanently (subtracting
    // finite values never clears it), so a single `sum.is_nan()` check *after* the
    // pass detects any NaN without a separate upfront scan; on the rare NaN case we
    // reset and fall through to the precise slow path.
    if let Some(src) = data.as_slice() {
        let mut sum = 0.0;
        {
            let dst = result.as_slice_mut().expect("from_elem is contiguous");
            for i in 0..n {
                sum += src[i];
                if i >= period {
                    sum -= src[i - period];
                }
                if i + 1 >= period {
                    dst[i] = sum / period as f64;
                }
            }
        }
        if !sum.is_nan() {
            return result;
        }
        result.fill(f64::NAN);
    }
    // Slow path — NaN-aware: a window containing any NaN yields NaN.
    let mut sum = 0.0;
    let mut nan_count = 0usize;
    for i in 0..n {
        let x = data[i];
        if x.is_nan() {
            nan_count += 1;
        } else {
            sum += x;
        }
        if i >= period {
            let leaving = data[i - period];
            if leaving.is_nan() {
                nan_count -= 1;
            } else {
                sum -= leaving;
            }
        }
        if i + 1 >= period && nan_count == 0 {
            result[i] = sum / period as f64;
        }
    }
    result
}

/// One in-place EWMA update step for the new value `cur` — the per-element body
/// shared by [`ewma_com`] and [`dual_ewma`] so the recurrence lives in one place.
#[inline(always)]
fn ewma_step(
    cur: f64,
    wavg: &mut f64,
    old_wt: &mut f64,
    nobs: &mut usize,
    old_wt_factor: f64,
    new_wt: f64,
    adjust: bool,
    ignore_na: bool,
) {
    let is_observation = !cur.is_nan();
    if is_observation {
        *nobs += 1;
    }
    if !wavg.is_nan() {
        if is_observation || !ignore_na {
            *old_wt *= old_wt_factor;
            if is_observation {
                if *wavg != cur {
                    *wavg = (*old_wt * *wavg + new_wt * cur) / (*old_wt + new_wt);
                }
                if adjust {
                    *old_wt += new_wt;
                } else {
                    *old_wt = 1.0;
                }
            }
        }
    } else if is_observation {
        *wavg = cur;
    }
}

/// Exponentially weighted moving average parameterised by center-of-mass,
/// matching pandas' `ewm(com=...)`.
#[inline]
pub fn ewma_com(
    data: ArrayView1<f64>,
    com: f64,
    adjust: bool,
    ignore_na: bool,
    min_periods: usize,
) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    if n == 0 {
        return result;
    }
    let min_periods = min_periods.max(1);
    let alpha = 1.0 / (1.0 + com);
    let old_wt_factor = 1.0 - alpha;
    let new_wt = if adjust { 1.0 } else { alpha };

    let mut wavg = data[0];
    let mut nobs = if wavg.is_nan() { 0 } else { 1 };
    let mut old_wt = 1.0;
    result[0] = if nobs >= min_periods { wavg } else { f64::NAN };

    for i in 1..n {
        ewma_step(
            data[i],
            &mut wavg,
            &mut old_wt,
            &mut nobs,
            old_wt_factor,
            new_wt,
            adjust,
            ignore_na,
        );
        result[i] = if nobs >= min_periods { wavg } else { f64::NAN };
    }
    result
}

/// Two independent EWMAs in a single traversal — bit-identical to two
/// [`ewma_com`] calls but with one pass and shared bounds. macd uses it (one
/// input, two coms); rsi uses it (two inputs, one com).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dual_ewma(
    a: ArrayView1<f64>,
    com_a: f64,
    min_a: usize,
    b: ArrayView1<f64>,
    com_b: f64,
    min_b: usize,
    adjust: bool,
    ignore_na: bool,
) -> (Array1<f64>, Array1<f64>) {
    let n = a.len();
    let mut ra = Array1::from_elem(n, f64::NAN);
    let mut rb = Array1::from_elem(n, f64::NAN);
    if n == 0 {
        return (ra, rb);
    }
    let min_a = min_a.max(1);
    let min_b = min_b.max(1);
    let alpha_a = 1.0 / (1.0 + com_a);
    let alpha_b = 1.0 / (1.0 + com_b);
    let owf_a = 1.0 - alpha_a;
    let owf_b = 1.0 - alpha_b;
    let nw_a = if adjust { 1.0 } else { alpha_a };
    let nw_b = if adjust { 1.0 } else { alpha_b };

    let mut wavg_a = a[0];
    let mut nobs_a = if wavg_a.is_nan() { 0 } else { 1 };
    let mut old_wt_a = 1.0;
    ra[0] = if nobs_a >= min_a { wavg_a } else { f64::NAN };

    let mut wavg_b = b[0];
    let mut nobs_b = if wavg_b.is_nan() { 0 } else { 1 };
    let mut old_wt_b = 1.0;
    rb[0] = if nobs_b >= min_b { wavg_b } else { f64::NAN };

    for i in 1..n {
        ewma_step(
            a[i],
            &mut wavg_a,
            &mut old_wt_a,
            &mut nobs_a,
            owf_a,
            nw_a,
            adjust,
            ignore_na,
        );
        ewma_step(
            b[i],
            &mut wavg_b,
            &mut old_wt_b,
            &mut nobs_b,
            owf_b,
            nw_b,
            adjust,
            ignore_na,
        );
        ra[i] = if nobs_a >= min_a { wavg_a } else { f64::NAN };
        rb[i] = if nobs_b >= min_b { wavg_b } else { f64::NAN };
    }
    (ra, rb)
}

/// Smoothed moving average (EWMA with `alpha = 1/period`).
#[inline]
pub fn smma(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    ewma_com(data, (period - 1) as f64, true, false, period)
}

/// SMA-seeded recursive smoother — the shape TA-Lib uses for EMA / Wilder. The
/// first output, at the `period`-th finite value, is the SMA of the first `period`
/// finite values; thereafter `prev = step(prev, x)` per row. Leading `NaN` in
/// `data` (e.g. `tr[0]`) is skipped for seeding; rows before the seed are warm-up
/// `NaN`. The exact `step` arithmetic is the caller's, to match TA-Lib per kind.
#[inline]
fn sma_seeded(data: ArrayView1<f64>, period: usize, step: impl Fn(f64, f64) -> f64) -> Array1<f64> {
    let n = data.len();
    let mut out = Array1::from_elem(n, f64::NAN);
    if period == 0 || n == 0 {
        return out;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    let mut seed_idx = None;
    for i in 0..n {
        let x = data[i];
        if !x.is_nan() {
            sum += x;
            count += 1;
            if count == period {
                seed_idx = Some(i);
                break;
            }
        }
    }
    let Some(si) = seed_idx else { return out };
    let mut prev = sum / period as f64;
    out[si] = prev;
    for i in (si + 1)..n {
        prev = step(prev, data[i]);
        out[i] = prev;
    }
    out
}

/// Wilder's smoothing (RMA), TA-Lib ATR / RSI / ADX arithmetic:
/// `out[i] = (out[i-1]*(period-1) + x[i]) / period`, SMA-seeded.
#[inline]
pub fn wilder(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let pf = period as f64;
    // `(prev*(period-1) + x) / period` rewritten as `prev*a + x*b` with the two
    // reciprocals precomputed, then fused (`mul_add`). This takes the per-element
    // **division** (~14-cycle latency, on the recurrence's critical path) off the
    // hot loop, leaving a single FMA — a large win for the division-bound Wilder
    // smoother (ATR / SMMA). Wilder smoothing is contractive (factor `a < 1`), so
    // the ~1e-16 reassociation error decays rather than accumulates: well within
    // the 1e-9 TA-Lib parity tolerance.
    let a = (pf - 1.0) / pf;
    let b = 1.0 / pf;
    sma_seeded(data, period, move |prev, x| prev.mul_add(a, x * b))
}

/// Exponential moving average, TA-Lib arithmetic:
/// `out[i] = out[i-1] + k*(x[i] - out[i-1])` with `k = 2/(period+1)`, SMA-seeded
/// (TA-Lib's default EMA seeding).
#[inline]
pub fn ema_seeded(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let k = 2.0 / (period as f64 + 1.0);
    // `(x - prev) * k + prev` via a fused multiply-add: one rounding (slightly more
    // accurate than TA-Lib's two-op form, within the 1e-9 parity tolerance) and a
    // shorter dependency chain — on FMA-capable hardware this cuts the latency-bound
    // EWMA recurrence by ~35%, so ema / macd / macd.signal beat TA-Lib's C loop.
    sma_seeded(data, period, move |prev, x| (x - prev).mul_add(k, prev))
}

/// EWMA seeded with an explicit initial value (used by KDJ).
#[inline]
pub fn ewma_with_init(data: ArrayView1<f64>, period: usize, init: f64) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    if n == 0 {
        return result;
    }
    let alpha = 1.0 / period as f64;
    let base = 1.0 - alpha;
    let mut k = init;
    for i in 0..n {
        k = base * k + alpha * data[i];
        result[i] = k;
    }
    result
}

/// van Herk / Gil-Werman sliding window reduction (min or max) in O(n) for
/// NaN-free data: per-block prefix and suffix extrema, then each length-`period`
/// window's extremum is `reduce(suffix[start], prefix[end])`. ~3 compares per
/// element, fully sequential (cache-friendly), with no per-window rescan — so it
/// has none of the O(n·period) worst case of a track-and-rescan. `reduce` is
/// monomorphised (inlined); `ident` is its identity (`+∞` for min, `−∞` for max).
#[inline]
fn van_herk(
    src: &[f64],
    period: usize,
    out: &mut [f64],
    reduce: impl Fn(f64, f64) -> f64,
    ident: f64,
) {
    let n = src.len();
    let mut prefix = vec![0.0f64; n];
    let mut suffix = vec![0.0f64; n];
    let mut s = 0;
    while s < n {
        let e = (s + period).min(n);
        let mut m = ident;
        for i in s..e {
            m = reduce(m, src[i]);
            prefix[i] = m;
        }
        let mut m = ident;
        for i in (s..e).rev() {
            m = reduce(m, src[i]);
            suffix[i] = m;
        }
        s = e;
    }
    for i in (period - 1)..n {
        out[i] = reduce(suffix[i + 1 - period], prefix[i]);
    }
}

/// Rolling minimum over the values present in each window.
///
/// NaN-free data takes the O(n) van Herk fast path; otherwise an ascending
/// monotonic deque of indices (the front is always the window minimum). NaNs are
/// never enqueued, so a window is `NaN` only when it holds no present value —
/// matching the original per-window scan.
#[inline]
pub fn rolling_min(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    if period == 0 || period > n {
        return result;
    }
    // Fast path — no NaN: van Herk O(n) sliding extremum (no deque, no indirect
    // reads). ~2.3x faster than the deque for typical periods and, unlike a
    // track-and-rescan, never degrades to O(n·period). The common OHLCV case.
    if let Some(src) = data.as_slice() {
        if !src.iter().any(|x| x.is_nan()) {
            let dst = result.as_slice_mut().expect("from_elem is contiguous");
            van_herk(src, period, dst, |a, b| a.min(b), f64::INFINITY);
            return result;
        }
    }
    let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
    for i in 0..n {
        while let Some(&front) = dq.front() {
            if front + period <= i {
                dq.pop_front();
            } else {
                break;
            }
        }
        let x = data[i];
        if !x.is_nan() {
            while let Some(&back) = dq.back() {
                if data[back] >= x {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);
        }
        if i + 1 >= period {
            if let Some(&front) = dq.front() {
                result[i] = data[front];
            }
        }
    }
    result
}

/// Rolling maximum over the values present in each window.
///
/// O(n) via a descending monotonic deque of indices (mirror of [`rolling_min`]).
#[inline]
pub fn rolling_max(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    if period == 0 || period > n {
        return result;
    }
    // Fast path — no NaN: van Herk O(n) sliding max (mirror of `rolling_min`).
    if let Some(src) = data.as_slice() {
        if !src.iter().any(|x| x.is_nan()) {
            let dst = result.as_slice_mut().expect("from_elem is contiguous");
            van_herk(src, period, dst, |a, b| a.max(b), f64::NEG_INFINITY);
            return result;
        }
    }
    let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
    for i in 0..n {
        while let Some(&front) = dq.front() {
            if front + period <= i {
                dq.pop_front();
            } else {
                break;
            }
        }
        let x = data[i];
        if !x.is_nan() {
            while let Some(&back) = dq.back() {
                if data[back] <= x {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);
        }
        if i + 1 >= period {
            if let Some(&front) = dq.front() {
                result[i] = data[front];
            }
        }
    }
    result
}

/// Rolling standard deviation with `ddof` degrees of freedom.
///
/// O(n) sliding sums of `x` and `x²`: `var = (Σx² - (Σx)²/period) / (period -
/// ddof)`, emitted only for fully-populated windows. The variance is clamped at
/// zero before the square root to absorb floating-point cancellation; for the
/// magnitudes seen in OHLCV data this stays well within the parity tolerance of
/// the two-pass form it replaces.
#[inline]
pub fn rolling_std(data: ArrayView1<f64>, period: usize, ddof: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    if period == 0 || period > n || period <= ddof {
        return result;
    }
    let p = period as f64;
    let denom = (period - ddof) as f64;
    // Fast path — contiguous data with no NaN: clean sliding sums of x and x²
    // with no per-element NaN bookkeeping. Same accumulation order, so it is
    // bit-identical to the slow path for NaN-free data while running faster — the
    // common case for real OHLCV.
    if let Some(src) = data.as_slice() {
        if !src.iter().any(|x| x.is_nan()) {
            {
                let dst = result.as_slice_mut().expect("from_elem is contiguous");
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                for i in 0..n {
                    let x = src[i];
                    sum += x;
                    sum_sq += x * x;
                    if i >= period {
                        let leaving = src[i - period];
                        sum -= leaving;
                        sum_sq -= leaving * leaving;
                    }
                    if i + 1 >= period {
                        let variance = (sum_sq - sum * sum / p) / denom;
                        dst[i] = variance.max(0.0).sqrt();
                    }
                }
            }
            return result;
        }
    }
    // Slow path — NaN-aware: a window containing any NaN yields NaN.
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut nan_count = 0usize;
    for i in 0..n {
        let x = data[i];
        if x.is_nan() {
            nan_count += 1;
        } else {
            sum += x;
            sum_sq += x * x;
        }
        if i >= period {
            let leaving = data[i - period];
            if leaving.is_nan() {
                nan_count -= 1;
            } else {
                sum -= leaving;
                sum_sq -= leaving * leaving;
            }
        }
        if i + 1 >= period && nan_count == 0 {
            let variance = (sum_sq - sum * sum / p) / denom;
            result[i] = variance.max(0.0).sqrt();
        }
    }
    result
}

/// Fused rolling **mean and std** in a single pass — one NaN scan, one sliding
/// accumulation of Σx and Σx². Bit-identical to calling [`sma`] and
/// [`rolling_std`] separately (same accumulation order), but Bollinger
/// bands / bandwidth need both, so this halves the rolling work (one buffer
/// init, one scan, one loop instead of two).
#[inline]
pub fn rolling_mean_std(
    data: ArrayView1<f64>,
    period: usize,
    ddof: usize,
) -> (Array1<f64>, Array1<f64>) {
    let n = data.len();
    let mut mean = Array1::from_elem(n, f64::NAN);
    let mut std = Array1::from_elem(n, f64::NAN);
    if period == 0 || period > n || period <= ddof {
        return (mean, std);
    }
    let p = period as f64;
    let denom = (period - ddof) as f64;
    // Fast path — one clean sliding pass emitting both outputs, no per-element NaN
    // bookkeeping. A NaN permanently poisons the running sum, so a single
    // `sum.is_nan()` check after the pass replaces a separate upfront scan; reset
    // and fall through on the rare NaN case.
    if let Some(src) = data.as_slice() {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        {
            let md = mean.as_slice_mut().expect("from_elem is contiguous");
            let sd = std.as_slice_mut().expect("from_elem is contiguous");
            for i in 0..n {
                let x = src[i];
                sum += x;
                sum_sq += x * x;
                if i >= period {
                    let leaving = src[i - period];
                    sum -= leaving;
                    sum_sq -= leaving * leaving;
                }
                if i + 1 >= period {
                    md[i] = sum / p;
                    let variance = (sum_sq - sum * sum / p) / denom;
                    sd[i] = variance.max(0.0).sqrt();
                }
            }
        }
        if !sum.is_nan() {
            return (mean, std);
        }
        mean.fill(f64::NAN);
        std.fill(f64::NAN);
    }
    // Slow path — NaN-aware: a window with any NaN yields NaN in both outputs.
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut nan_count = 0usize;
    for i in 0..n {
        let x = data[i];
        if x.is_nan() {
            nan_count += 1;
        } else {
            sum += x;
            sum_sq += x * x;
        }
        if i >= period {
            let leaving = data[i - period];
            if leaving.is_nan() {
                nan_count -= 1;
            } else {
                sum -= leaving;
                sum_sq -= leaving * leaving;
            }
        }
        if i + 1 >= period && nan_count == 0 {
            mean[i] = sum / p;
            let variance = (sum_sq - sum * sum / p) / denom;
            std[i] = variance.max(0.0).sqrt();
        }
    }
    (mean, std)
}

/// First difference (`data[i] - data[i-1]`).
#[inline]
pub fn diff(data: ArrayView1<f64>) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    for i in 1..n {
        if !data[i].is_nan() && !data[i - 1].is_nan() {
            result[i] = data[i] - data[i - 1];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sma() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(data.view(), 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sma_with_nan() {
        let data = array![f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(data.view(), 3);
        assert!(result[3].is_nan());
        assert!((result[4] - 2.0).abs() < 1e-10);
        assert!((result[6] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min_max() {
        let data = array![3.0, 1.0, 4.0, 1.0, 5.0];
        let mn = rolling_min(data.view(), 3);
        let mx = rolling_max(data.view(), 3);
        assert!((mn[2] - 1.0).abs() < 1e-10);
        assert!((mx[2] - 4.0).abs() < 1e-10);
        assert!((mx[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff() {
        let data = array![1.0, 3.0, 6.0];
        let d = diff(data.view());
        assert!(d[0].is_nan());
        assert!((d[1] - 2.0).abs() < 1e-10);
        assert!((d[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn empty_input_and_invalid_periods_return_nan() {
        let empty = Array1::<f64>::zeros(0);
        assert_eq!(sma(empty.view(), 3).len(), 0);
        assert_eq!(ewma_com(empty.view(), 1.0, true, false, 1).len(), 0);
        assert_eq!(ewma_with_init(empty.view(), 3, 0.0).len(), 0);

        let d = array![1.0, 2.0];
        assert!(rolling_min(d.view(), 5).iter().all(|x| x.is_nan())); // period > n
        assert!(rolling_max(d.view(), 5).iter().all(|x| x.is_nan()));
        assert!(rolling_std(d.view(), 5, 1).iter().all(|x| x.is_nan()));
        assert!(rolling_min(d.view(), 0).iter().all(|x| x.is_nan())); // period == 0
        assert!(rolling_std(d.view(), 2, 2).iter().all(|x| x.is_nan())); // ddof >= period
    }

    #[test]
    fn ewma_handles_nan_and_adjust_variants() {
        let d = array![1.0, f64::NAN, 3.0, 4.0];
        assert_eq!(ewma_com(d.view(), 1.0, true, true, 1).len(), 4); // adjust + ignore_na
        assert_eq!(ewma_com(d.view(), 1.0, false, false, 1).len(), 4); // neither
                                                                       // a leading NaN keeps nobs below min_periods -> result[0] is NaN
        let lead = array![f64::NAN, 2.0, 3.0];
        assert!(ewma_com(lead.view(), 1.0, true, false, 1)[0].is_nan());
    }

    // --- O(n) rewrite safety net -------------------------------------------
    // Independent naive (re-scan-every-window) oracles. The fast sliding /
    // deque kernels MUST match these 1:1 (NaN-aware), including interior NaNs
    // that slide through a window — the case real OHLCV parity data never hits.

    fn av(s: &[f64]) -> ArrayView1<'_, f64> {
        ArrayView1::from(s)
    }

    fn naive_sma(d: &[f64], p: usize) -> Vec<f64> {
        let n = d.len();
        let mut out = vec![f64::NAN; n];
        if p == 0 || p > n {
            return out;
        }
        for i in (p - 1)..n {
            let w = &d[i + 1 - p..=i];
            if w.iter().all(|x| !x.is_nan()) {
                out[i] = w.iter().sum::<f64>() / p as f64;
            }
        }
        out
    }

    fn naive_std(d: &[f64], p: usize, ddof: usize) -> Vec<f64> {
        let n = d.len();
        let mut out = vec![f64::NAN; n];
        if p == 0 || p > n || p <= ddof {
            return out;
        }
        for i in (p - 1)..n {
            let w = &d[i + 1 - p..=i];
            if w.iter().all(|x| !x.is_nan()) {
                let m = w.iter().sum::<f64>() / p as f64;
                let v = w.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (p - ddof) as f64;
                out[i] = v.sqrt();
            }
        }
        out
    }

    fn naive_minmax(d: &[f64], p: usize, max: bool) -> Vec<f64> {
        let n = d.len();
        let mut out = vec![f64::NAN; n];
        if p == 0 || p > n {
            return out;
        }
        for i in (p - 1)..n {
            let mut acc = if max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            let mut any = false;
            for &x in &d[i + 1 - p..=i] {
                if !x.is_nan() {
                    acc = if max { acc.max(x) } else { acc.min(x) };
                    any = true;
                }
            }
            if any {
                out[i] = acc;
            }
        }
        out
    }

    fn approx_eq_nan(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            if x.is_nan() || y.is_nan() {
                assert!(x.is_nan() && y.is_nan(), "idx {i}: {x} vs {y}");
            } else {
                assert!((x - y).abs() <= tol + tol * y.abs(), "idx {i}: {x} vs {y}");
            }
        }
    }

    /// Deterministic Park–Miller series with stock-price-like magnitude.
    fn series(n: usize) -> Vec<f64> {
        let mut x: i64 = 1_234_567;
        let mut s = Vec::with_capacity(n);
        for _ in 0..n {
            x = (x * 16807) % 2_147_483_647;
            s.push(100.0 + (x as f64 / 2_147_483_647.0) * 50.0);
        }
        s
    }

    #[test]
    fn sma_matches_naive_including_interior_nan() {
        let mut d = series(500);
        d[7] = f64::NAN;
        d[123] = f64::NAN; // interior NaNs that slide through windows
        for p in [1usize, 2, 5, 20, 50] {
            approx_eq_nan(&sma(av(&d), p).to_vec(), &naive_sma(&d, p), 1e-9);
        }
    }

    #[test]
    fn rolling_std_matches_naive_within_tolerance() {
        let mut d = series(500);
        d[50] = f64::NAN;
        for p in [2usize, 5, 20] {
            for ddof in [0usize, 1] {
                approx_eq_nan(
                    &rolling_std(av(&d), p, ddof).to_vec(),
                    &naive_std(&d, p, ddof),
                    1e-7,
                );
            }
        }
    }

    #[test]
    fn rolling_min_max_match_naive_with_interior_nan() {
        let mut d = series(500);
        d[10] = f64::NAN;
        d[11] = f64::NAN; // a fully-NaN sub-run
        for p in [1usize, 3, 10, 30] {
            approx_eq_nan(
                &rolling_min(av(&d), p).to_vec(),
                &naive_minmax(&d, p, false),
                0.0,
            );
            approx_eq_nan(
                &rolling_max(av(&d), p).to_vec(),
                &naive_minmax(&d, p, true),
                0.0,
            );
        }
    }

    #[test]
    fn rolling_min_deque_resurfaces_after_min_leaves_window() {
        // when the running min leaves the window, the next-smallest must surface
        let z = [5.0, 1.0, 2.0, 3.0, 4.0, 6.0];
        let m = rolling_min(av(&z), 3).to_vec();
        assert!(m[0].is_nan() && m[1].is_nan());
        assert_eq!(&m[2..], &[1.0, 1.0, 2.0, 3.0]);
        let x = rolling_max(av(&z), 3).to_vec();
        assert_eq!(&x[2..], &[5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn all_nan_window_is_nan_for_minmax() {
        let d = [f64::NAN, f64::NAN, 1.0, f64::NAN, f64::NAN];
        let m = rolling_min(av(&d), 2).to_vec();
        assert!(m[1].is_nan());
        assert_eq!(m[2], 1.0);
        assert_eq!(m[3], 1.0);
        assert!(m[4].is_nan());
    }

    #[test]
    fn dual_ewma_matches_two_ewma_com_bit_exact() {
        let d = series(300);
        let mut g = series(300);
        g[5] = f64::NAN; // a distinct second input with an interior NaN
        let (a, b) = (av(&d), av(&g));
        // one input, two coms (the macd shape)
        let (f, s) = dual_ewma(a, 5.5, 12, a, 12.5, 26, true, false);
        approx_eq_nan(
            &f.to_vec(),
            &ewma_com(a, 5.5, true, false, 12).to_vec(),
            0.0,
        );
        approx_eq_nan(
            &s.to_vec(),
            &ewma_com(a, 12.5, true, false, 26).to_vec(),
            0.0,
        );
        // two inputs, one com (the rsi shape)
        let (ga, gb) = dual_ewma(a, 13.0, 14, b, 13.0, 14, true, false);
        approx_eq_nan(
            &ga.to_vec(),
            &ewma_com(a, 13.0, true, false, 14).to_vec(),
            0.0,
        );
        approx_eq_nan(
            &gb.to_vec(),
            &ewma_com(b, 13.0, true, false, 14).to_vec(),
            0.0,
        );
    }
}
