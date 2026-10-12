//! Rolling dispersion: sliding standard deviation, alone and fused with the mean
//! (Bollinger bands / bandwidth need both).

use super::mean_std_row;
use crate::accum::{comoment, square_terms, Anchors, Csum};
use crate::guard::Slide;
use ndarray::{Array1, ArrayView1};

/// Rolling standard deviation with `ddof` degrees of freedom; row 0 is at absolute
/// position `origin` (see [`crate::accum`]).
///
/// O(n) sliding exact sums of `x` and `x²` ([`Csum`], each square split into its
/// rounded value and its rounding error, so `Σx²` is exact): `var = (n·Σx² − (Σx)²) /
/// (n·(n − ddof))` with the numerator formed by [`comoment`] — accurate however the raw
/// moments cancel, exactly zero for a flat window — emitted only for fully-populated
/// windows.
#[inline]
pub fn rolling_std(data: ArrayView1<f64>, period: usize, ddof: usize, origin: usize) -> Array1<f64> {
    rolling_mean_std(data, period, ddof, origin).1
}

/// The mean and standard deviation of a window from its exact sums `[Σx, Σx²]` (the
/// squares' sum split as [`square_terms`] splits each square).
#[inline(always)]
pub(crate) fn mean_std_of_sums([sum, sq_hi, sq_lo]: &[Csum; 3], p: f64, denom: f64) -> (f64, f64) {
    let s = sum.words();
    let variance = comoment(p, s, s, &[*sq_hi, *sq_lo]) / (p * denom);
    (sum.mean(p), variance.max(0.0).sqrt())
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
    // non-finite value poisons the sums (and marks them inexact); that is checked at
    // every re-anchoring and at the end, and then the careful path takes over.
    if let Some(src) = data.as_slice() {
        let mut sums = [Csum::default(); 3];
        let mut anchors = Anchors::from(origin, period);
        let mut clean = true;
        {
            let md = mean.as_slice_mut().expect("from_elem is contiguous");
            let sd = std.as_slice_mut().expect("from_elem is contiguous");
            for i in 0..n {
                for (s, t) in sums.iter_mut().zip(square_terms(src[i])) {
                    s.add(t);
                }
                if i >= period {
                    for (s, t) in sums.iter_mut().zip(square_terms(src[i - period])) {
                        s.sub(t);
                    }
                }
                let anchor = anchors.tick();
                if (anchor || !sums.iter().all(Csum::is_exact)) && i + 1 >= period {
                    clean &= sums.iter().all(|s| s.value().is_finite());
                    if clean {
                        sums = Csum::of_rows(src[i + 1 - period..=i].iter().map(|&x| square_terms(x)));
                    }
                }
                if i + 1 >= period {
                    (md[i], sd[i]) = mean_std_of_sums(&sums, p, denom);
                }
            }
        }
        if clean && sums.iter().all(|s| s.value().is_finite()) {
            return (mean, std);
        }
    }
    // Careful path — window-local (see [`crate::guard`]).
    let mut slide = Slide::<3>::at(origin);
    for i in 0..n {
        (mean[i], std[i]) = mean_std_row(&mut slide, period, ddof, i, |j| data[j]);
    }
    (mean, std)
}
