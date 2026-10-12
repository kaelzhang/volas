//! Rolling dispersion: sliding standard deviation, alone and fused with the mean
//! (Bollinger bands / bandwidth need both).

use super::mean_std_row;
use crate::accum::{spread, Anchors, Csum};
use crate::guard::Slide;
use ndarray::{Array1, ArrayView1};

/// Rolling standard deviation with `ddof` degrees of freedom; row 0 is at absolute
/// position `origin` (see [`crate::accum`]).
///
/// O(n) sliding exact sums of `x` and of the squared deviations `(x − K)²` from the
/// window's shift `K` (the value at its anchor row, see [`crate::accum`]'s second
/// moments): `var = (Σd² − (Σd)²/n) / (n − ddof)` — within `(36n + 2)·2⁻⁵³` of the exact
/// variance however the window's level dwarfs its spread, exactly zero for a flat
/// window — emitted only for fully-populated windows.
#[inline]
pub fn rolling_std(data: ArrayView1<f64>, period: usize, ddof: usize, origin: usize) -> Array1<f64> {
    rolling_mean_std(data, period, ddof, origin).1
}

/// The terms of a window's `[Σx, Σ(x − k)²]` for the row value `x` under the shift `k`.
#[inline(always)]
pub(crate) fn shifted_terms(x: f64, k: f64) -> [f64; 2] {
    let d = x - k;
    [x, d * d]
}

/// The mean and standard deviation of a window of `p` rows from its exact sums
/// `[Σx, Σ(x − k)²]` ([`shifted_terms`]). A window holding a non-finite value takes its
/// IEEE values: the sum's infinity (or NaN) over `p` as mean, NaN as deviation.
#[inline(always)]
pub(crate) fn mean_std_of_sums([sum, sq]: &[Csum; 2], k: f64, p: f64, denom: f64) -> (f64, f64) {
    let (mean, c) = spread(p, sum, k, sq);
    let variance = c / denom;
    let total = sum.value();
    let mean = if total.is_finite() { mean } else { total / p };
    (mean, if variance.is_nan() { f64::NAN } else { variance.max(0.0).sqrt() })
}

/// Mean and std of the window `rows` (shifted by `k`) evaluated on its own rows — the
/// value of a window holding a non-finite row: its exact sums, then
/// [`mean_std_of_sums`].
pub(crate) fn window_mean_std_of(
    rows: std::ops::RangeInclusive<usize>,
    term: impl Fn(usize) -> f64,
    k: f64,
    p: f64,
    denom: f64,
) -> (f64, f64) {
    mean_std_of_sums(&Csum::of_rows(rows.map(|j| shifted_terms(term(j), k))), k, p, denom)
}

/// Fused rolling **mean and std** in a single pass — one sliding accumulation of Σx
/// and Σ(x − K)² (row 0 at absolute position `origin`). Bollinger bands / bandwidth need
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
    // bookkeeping: the careful slide's arithmetic ([`mean_std_row`]) on clean data. The
    // shift is the value at the latest anchor row (the first row before one), and the
    // sums are rebuilt at every anchor. A non-finite value poisons the sums (and marks
    // them inexact); that is checked at every re-anchoring and at the end, and then the
    // careful path takes over.
    if let Some(src) = data.as_slice() {
        let mut sums = [Csum::default(); 2];
        let mut anchors = Anchors::from(origin, period);
        let mut k = src[0];
        let mut clean = true;
        {
            let md = mean.as_slice_mut().expect("from_elem is contiguous");
            let sd = std.as_slice_mut().expect("from_elem is contiguous");
            for i in 0..n {
                let anchor = anchors.tick();
                if anchor {
                    k = src[i];
                }
                for (s, t) in sums.iter_mut().zip(shifted_terms(src[i], k)) {
                    s.add(t);
                }
                if i >= period {
                    for (s, t) in sums.iter_mut().zip(shifted_terms(src[i - period], k)) {
                        s.sub(t);
                    }
                }
                if anchor || (i + 1 >= period && !sums.iter().all(Csum::is_exact)) {
                    clean &= sums.iter().all(|s| s.value().is_finite());
                    if clean {
                        let rows = &src[(i + 1).saturating_sub(period)..=i];
                        sums = Csum::of_rows(rows.iter().map(|&x| shifted_terms(x, k)));
                    }
                }
                if i + 1 >= period {
                    (md[i], sd[i]) = mean_std_of_sums(&sums, k, p, denom);
                }
            }
        }
        if clean && sums.iter().all(|s| s.value().is_finite()) {
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
