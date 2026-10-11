//! Rolling dispersion: sliding standard deviation, alone and fused with the mean
//! (Bollinger bands / bandwidth need both).

use super::mean_std_row;
use crate::accum::{Anchors, Csum};
use crate::guard::Slide;
use ndarray::{Array1, ArrayView1};

/// Rolling standard deviation with `ddof` degrees of freedom; row 0 is at absolute
/// position `origin` (see [`crate::accum`]).
///
/// O(n) sliding exact sums of `x` and `x²` ([`Csum`]): `var = (mean(x²) − mean(x)²) ·
/// period / (period − ddof)`, emitted only for fully-populated windows, clamped at zero
/// before the square root. A flat window's deviation is exactly zero.
#[inline]
pub fn rolling_std(data: ArrayView1<f64>, period: usize, ddof: usize, origin: usize) -> Array1<f64> {
    rolling_mean_std(data, period, ddof, origin).1
}

/// The mean and standard deviation of a window from its exact sums `Σx` / `Σx²`.
#[inline(always)]
pub(crate) fn mean_std_of_sums(sum: &Csum, sum_sq: &Csum, p: f64, denom: f64) -> (f64, f64) {
    let mean = sum.mean(p);
    let variance = (sum_sq.mean(p) - mean * mean) * (p / denom);
    (mean, variance.max(0.0).sqrt())
}

/// Mean and std of the window `data[start..=end]` evaluated directly on its rows —
/// the value of a window holding a non-finite row. The variance is checked for NaN
/// before the cancellation clamp, which would otherwise turn NaN into 0.
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

/// Fused rolling **mean and std** in a single pass — one sliding accumulation of Σx
/// and Σx² (row 0 at absolute position `origin`). Bollinger bands / bandwidth need
/// both, so this halves the rolling work.
#[inline]
pub fn rolling_mean_std(
    data: ArrayView1<f64>,
    period: usize,
    ddof: usize,
    origin: usize,
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
    // bookkeeping: the careful slide's arithmetic ([`mean_std_row`]) on clean data. A
    // non-finite value poisons the sums until the next anchor; that is checked at every
    // anchor and at the end, and then the careful path takes over.
    if let Some(src) = data.as_slice() {
        let mut sum = Csum::default();
        let mut sum_sq = Csum::default();
        let mut anchors = Anchors::from(origin, period);
        let mut clean = true;
        {
            let md = mean.as_slice_mut().expect("from_elem is contiguous");
            let sd = std.as_slice_mut().expect("from_elem is contiguous");
            for i in 0..n {
                let x = src[i];
                sum.add(x);
                sum_sq.add(x * x);
                if i >= period {
                    let leaving = src[i - period];
                    sum.sub(leaving);
                    sum_sq.sub(leaving * leaving);
                }
                if anchors.tick() && i + 1 >= period {
                    clean &= sum.value().is_finite() && sum_sq.value().is_finite();
                    let w = &src[i + 1 - period..=i];
                    sum = Csum::of(w.iter().copied());
                    sum_sq = Csum::of(w.iter().map(|x| x * x));
                }
                if i + 1 >= period {
                    (md[i], sd[i]) = mean_std_of_sums(&sum, &sum_sq, p, denom);
                }
            }
        }
        if clean && sum.value().is_finite() && sum_sq.value().is_finite() {
            return (mean, std);
        }
    }
    // Careful path — window-local (see [`crate::guard`]).
    let mut slide = Slide::<2>::at(origin);
    for i in 0..n {
        (mean[i], std[i]) = mean_std_row(&mut slide, period, ddof, i, |j| data[j]);
    }
    (mean, std)
}
