//! Rolling dispersion: sliding standard deviation, alone and fused with the mean
//! (Bollinger bands / bandwidth need both).

use crate::guard::{NonFiniteRows, RowGuard};
use ndarray::{Array1, ArrayView1};

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
        if src.iter().all(|x| x.is_finite()) {
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
    // Careful path — window-local (see [`crate::guard`]).
    let mut guard = NonFiniteRows::default();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for i in 0..n {
        let x = data[i];
        if guard.enter(x.is_finite()) {
            sum += x;
            sum_sq += x * x;
        }
        if i >= period {
            let leaving = data[i - period];
            if guard.leave(leaving.is_finite()) {
                sum -= leaving;
                sum_sq -= leaving * leaving;
            }
        }
        if i + 1 >= period {
            result[i] = if guard.clean() {
                let variance = (sum_sq - sum * sum / p) / denom;
                variance.max(0.0).sqrt()
            } else {
                window_mean_std(data, i + 1 - period, i, p, denom).1
            };
        }
    }
    result
}

/// Mean and std of the window `data[start..=end]` evaluated directly on its rows —
/// the value of a window holding a non-finite row. The variance is checked for NaN
/// before the cancellation clamp, which would otherwise turn NaN into 0.
fn window_mean_std(data: ArrayView1<f64>, start: usize, end: usize, p: f64, denom: f64) -> (f64, f64) {
    window_mean_std_of(start..=end, |j| data[j], p, denom)
}

/// [`window_mean_std`] over the rows `rows` of the series `term`.
pub(crate) fn window_mean_std_of(
    rows: std::ops::RangeInclusive<usize>,
    term: impl Fn(usize) -> f64,
    p: f64,
    denom: f64,
) -> (f64, f64) {
    let (sum, sum_sq) = rows.fold((0.0, 0.0), |(s, q), j| {
        let x = term(j);
        (s + x, q + x * x)
    });
    let variance = (sum_sq - sum * sum / p) / denom;
    let std = if variance.is_nan() { f64::NAN } else { variance.max(0.0).sqrt() };
    (sum / p, std)
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
        if sum.is_finite() && sum_sq.is_finite() {
            return (mean, std);
        }
        mean.fill(f64::NAN);
        std.fill(f64::NAN);
    }
    // Careful path — window-local (see [`crate::guard`]).
    let mut guard = NonFiniteRows::default();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for i in 0..n {
        let x = data[i];
        if guard.enter(x.is_finite()) {
            sum += x;
            sum_sq += x * x;
        }
        if i >= period {
            let leaving = data[i - period];
            if guard.leave(leaving.is_finite()) {
                sum -= leaving;
                sum_sq -= leaving * leaving;
            }
        }
        if i + 1 >= period {
            if guard.clean() {
                mean[i] = sum / p;
                let variance = (sum_sq - sum * sum / p) / denom;
                std[i] = variance.max(0.0).sqrt();
            } else {
                (mean[i], std[i]) = window_mean_std(data, i + 1 - period, i, p, denom);
            }
        }
    }
    (mean, std)
}
