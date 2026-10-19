//! pandas-semantics windowed aggregations — the single kernel source for the
//! python `rolling` / `expanding` / `ewm` API surface.
//!
//! Distinct from [`crate::raw::indicators`] (TA-Lib semantics: a NaN poisons its
//! window): here NaN and ±inf mean *missing* and is skipped, and a row emits a value
//! once the window holds at least `min_periods` present values — exactly
//! pandas's window model. `window == usize::MAX` is an expanding window.
//!
//! All kernels take `&[f64]` (the python layer funnels int/bool, and routes the
//! dtype-preserving members `first` / `last` / `count` / `nunique` through the
//! position / count kernels instead).

mod params;
pub use params::{validate_alpha, validate_window, WindowError};
use params::{validate_pair, validate_quantile, validate_rank};

use crate::accum::{comoment, square_terms, Csum, Exact};
use std::collections::HashMap;

/// The trailing window over `i`: `[start, i]`.
#[inline]
fn win_start(i: usize, window: usize) -> usize {
    (i + 1).saturating_sub(window)
}

/// Generic per-window fold over the PRESENT values (reusable buffer): the
/// simple, numerically transparent path used by the order / moment statistics.
fn per_window(
    data: &[f64],
    window: usize,
    min_periods: usize,
    mut f: impl FnMut(&[f64]) -> f64,
) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let mut buf: Vec<f64> = Vec::with_capacity(window.min(n));
    for i in 0..n {
        buf.clear();
        for &x in &data[win_start(i, window)..=i] {
            if !crate::missing::is_missing(x) {
                buf.push(x);
            }
        }
        if buf.len() >= min_periods.max(1) {
            out[i] = f(&buf);
        }
    }
    out
}

// --- count / sum / mean (sliding O(n)) ---------------------------------------

/// Present-value count per window (pandas `count`). pandas implements count
/// as `notna().rolling(...).sum()` over the DENSE 0/1 mask, so `min_periods`
/// gates on the number of rows the window COVERS (NaN rows included), not on
/// the present count — a window full of NaN emits `0`, a not-yet-full head
/// window emits NaN. `center=true` uses the clipped centered bounds.
pub fn count(
    data: &[f64],
    window: usize,
    min_periods: usize,
    center: bool,
) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok({
        let n = data.len();
        let fwd = if center { window - window / 2 - 1 } else { 0 };
        let back = window.saturating_sub(fwd + 1);
        (0..n)
            .map(|i| {
                let end = std::cmp::min(i + fwd, n - 1);
                let start = if window == usize::MAX { 0 } else { i.saturating_sub(back) };
                let covered = end - start + 1;
                if covered < min_periods.max(1) {
                    return f64::NAN;
                }
                data[start..=end].iter().filter(|&&x| !crate::missing::is_missing(x)).count() as f64
            })
            .collect()
    })
}

/// Slide finite present values, skipping NaN and both infinity signs. Read each
/// window with at least `min_periods` present values, otherwise emit NaN. The
/// running sums are the exact ones of [`crate::accum`], re-summed whenever one went
/// inexact, so each window's sums are the correctly rounded sums of its own terms
/// (`math.fsum`), however long the slide — and depend on the window's values alone,
/// not on where the series starts. An expanding window never drops a value: its sums
/// are kept exactly by [`Exact`] expansions once the compensated form can no longer
/// hold them — after an extreme value (its magnitude far from the rest's), every later
/// row is added to the expansion and rounded from it, an `O(expansion words)` step with
/// inline storage for common expansions instead of the compensated add, for the rest of that series.
/// `read(sums, count)`.
fn present_sums<const K: usize>(
    data: &[f64],
    window: usize,
    min_periods: usize,
    terms: impl Fn(f64) -> [f64; K],
    read: impl Fn(&[Csum; K], usize) -> f64,
) -> Vec<f64> {
    present_sums_at(data, window, min_periods, terms, |_, sums, count| read(sums, count))
}

/// The same exact slide, with the window's endpoint available to its reader.
fn present_sums_at<const K: usize>(
    data: &[f64],
    window: usize,
    min_periods: usize,
    terms: impl Fn(f64) -> [f64; K],
    mut read: impl FnMut(usize, &[Csum; K], usize) -> f64,
) -> Vec<f64> {
    let rolling = window != usize::MAX;
    let mut sums = [Csum::default(); K];
    let mut expansion: Option<[Exact; K]> = None;
    let mut count = 0usize;
    let mut out = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        if rolling && i >= window {
            let x = data[i - window];
            if !crate::missing::is_missing(x) {
                count -= 1;
                {
                    for (s, t) in sums.iter_mut().zip(terms(x)) {
                        s.sub(t);
                    }
                }
            }
        }
        let x = data[i];
        if !crate::missing::is_missing(x) {
            count += 1;
            if let Some(e) = &mut expansion {
                for (e, t) in e.iter_mut().zip(terms(x)) {
                    e.add(t);
                }
            } else {
                for (s, t) in sums.iter_mut().zip(terms(x)) {
                    s.add(t);
                }
            }
        }
        if expansion.is_none() && !sums.iter().all(Csum::is_exact) {
            let rows = &data[win_start(i, window)..=i];
            sums = Csum::of_rows(rows.iter().filter(|&&x| !crate::missing::is_missing(x)).map(|&x| terms(x)));
            if !rolling && !sums.iter().all(Csum::is_exact) {
                let mut e: [Exact; K] = std::array::from_fn(|_| Exact::default());
                for &x in rows.iter().filter(|&&x| !crate::missing::is_missing(x)) {
                    for (e, t) in e.iter_mut().zip(terms(x)) {
                        e.add(t);
                    }
                }
                expansion = Some(e);
            }
        }
        if let Some(e) = &expansion {
            sums = std::array::from_fn(|j| e[j].to_csum());
        }
        out.push(if count >= min_periods { read(i, &sums, count) } else { f64::NAN });
    }
    out
}

/// Rolling / expanding sum of the present values (pandas `sum`): the exact sum of the
/// finite ones; missing inputs are skipped.
pub fn sum(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(present_sums(data, window, min_periods, |x| [x], |[s], _| s.value()))
}

/// Rolling / expanding mean of the present values (pandas `mean`): the exact sum over
/// the count ([`Csum::mean`] — a flat window has its value as mean).
pub fn mean(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(present_sums(data, window, min_periods, |x| [x], |[s], n| s.mean(n as f64)))
}

// --- min / max (monotonic deque, O(n)) ---------------------------------------

fn extremum(data: &[f64], window: usize, min_periods: usize, is_min: bool) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let better = |a: f64, b: f64| if is_min { a <= b } else { a >= b };
    let mut nobs = 0usize;
    for i in 0..n {
        if window != usize::MAX && i >= window && !crate::missing::is_missing(data[i - window]) {
            nobs -= 1;
        }
        if let Some(&front) = deque.front() {
            if window != usize::MAX && i >= window && front <= i - window {
                deque.pop_front();
            }
        }
        if !crate::missing::is_missing(data[i]) {
            nobs += 1;
            while let Some(&back) = deque.back() {
                if better(data[i], data[back]) {
                    deque.pop_back();
                } else {
                    break;
                }
            }
            deque.push_back(i);
        }
        if nobs >= min_periods.max(1) {
            if let Some(&front) = deque.front() {
                out[i] = data[front];
            }
        }
    }
    out
}

/// Rolling / expanding minimum (pandas `min`).
pub fn min(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(extremum(data, window, min_periods, true))
}

/// Rolling / expanding maximum (pandas `max`).
pub fn max(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(extremum(data, window, min_periods, false))
}

// --- dispersion / moments -----------------------------------------------------

/// Whether every value in a (non-empty) window is the same — a constant window.
/// `0.0 == -0.0`, so signed zeros are one value.
fn is_constant(b: &[f64]) -> bool {
    b.iter().all(|&x| x == b[0])
}

/// The window mean as the center of every deviation-based statistic: the exact sum of
/// the window ([`Csum::of`]) over its count. A constant window's mean is its value
/// EXACTLY — a rounded `sum / n` would leave a residue (`101.37 × 20 / 20 ≠ 101.37`)
/// that the deviations then mistake for dispersion.
fn window_mean(b: &[f64]) -> f64 {
    Csum::of(b.iter().copied()).mean(b.len() as f64)
}

/// The two-pass variance of a window with `ddof` degrees of freedom: the exact sum of
/// the squared deviations from [`window_mean`] over `n − ddof`.
fn buf_var(b: &[f64], ddof: usize) -> f64 {
    if b.len() <= ddof {
        return f64::NAN;
    }
    let m = window_mean(b);
    Csum::of(b.iter().map(|x| (x - m) * (x - m))).value() / (b.len() - ddof) as f64
}

/// The variance with `ddof` degrees of freedom of each window's present values, from
/// their exact sums `[Σx, Σx²]` ([`square_terms`]): `(n·Σx² − (Σx)²) / (n·(n − ddof))`
/// by [`comoment`] — within an ulp or two of the exact variance however the window's
/// level dwarfs its spread, exactly zero for a constant window, NaN when the window
/// holds an infinity or no more than `ddof` values. A function of the window's values
/// alone (no shift by an absolute position: this API is a stateless snapshot). O(n):
/// one exact sliding pass, no per-window rescan.
fn present_var(data: &[f64], window: usize, min_periods: usize, ddof: usize, read: impl Fn(f64, usize) -> f64) -> Vec<f64> {
    present_sums(data, window, min_periods, square_terms, |sums, n| {
        if n <= ddof {
            return f64::NAN;
        }
        let nf = n as f64;
        let s = sums[0].words();
        read((comoment(nf, s, s, &sums[1..]) / (nf * (n - ddof) as f64)).max(0.0), n)
    })
}

/// Rolling / expanding sample variance (pandas `var`, default ddof=1).
pub fn var(data: &[f64], window: usize, min_periods: usize, ddof: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(present_var(data, window, min_periods, ddof, |v, _| v))
}

/// Rolling / expanding standard deviation (pandas `std`).
pub fn std(data: &[f64], window: usize, min_periods: usize, ddof: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(present_var(data, window, min_periods, ddof, |v, _| v.sqrt()))
}

/// Standard error of the mean (pandas `sem`): `std(ddof) / sqrt(count)`.
pub fn sem(data: &[f64], window: usize, min_periods: usize, ddof: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(present_var(data, window, min_periods, ddof, |v, n| (v / n as f64).sqrt()))
}

/// Mean absolute deviation about the window mean (TradingView `ta.dev`):
/// `mean(|x - mean(x)|)` over the present values — the dispersion measure CCI
/// is built on, and the one statistic missing from var/std/sem.
pub fn dev(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(per_window(data, window, min_periods, |b| {
        let m = window_mean(b);
        b.iter().map(|x| (x - m).abs()).sum::<f64>() / b.len() as f64
    }))
}

/// Rolling mode (TradingView `ta.mode`): the most frequent present value; ties
/// resolve to the SMALLEST value; NA cells are ignored. Values are counted by numeric
/// equality, so `-0.0` and `+0.0` are one value (as pandas counts them), and a zero mode
/// is `+0.0` whichever signed zeros the window holds.
pub fn mode(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(per_sorted_window(data, window, min_periods, |sorted, _| {
        // `sorted` is ascending (total order: the signed zeros sit side by side), so
        // numerically equal values are contiguous: scan runs and keep the value with
        // the longest run; on a tie the earlier (smaller) value is already held,
        // giving "ties -> smallest" for free.
        let (mut best_val, mut best_len) = (sorted[0], 1usize);
        let (mut cur_val, mut cur_len) = (sorted[0], 1usize);
        for &x in &sorted[1..] {
            if x == cur_val {
                cur_len += 1;
            } else {
                cur_val = x;
                cur_len = 1;
            }
            if cur_len > best_len {
                best_len = cur_len;
                best_val = cur_val;
            }
        }
        best_val + 0.0 // a zero mode is +0
    }))
}

/// Borrow finite windows and read their exact mean from the shared sliding sum.
/// Central moments still reduce in the original row order. Missing/non-finite
/// inputs retain the present-value gather and direct exact mean.
fn per_moment(data: &[f64], window: usize, min_periods: usize, mut f: impl FnMut(&[f64], f64) -> f64) -> Vec<f64> {
    if !crate::missing::any_missing(data) {
        present_sums_at(data, window, min_periods, |x| [x], |i, sums, count| {
            f(&data[win_start(i, window)..=i], sums[0].mean(count as f64))
        })
    } else {
        per_window(data, window, min_periods, |b| f(b, window_mean(b)))
    }
}

/// Bias-corrected sample skewness (pandas `skew`; needs >= 3 present values). A
/// constant window is exactly `0` (pandas 3's degenerate-window rule).
pub fn skew(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(per_moment(data, window, min_periods, |b, m| {
        let n = b.len() as f64;
        if b.len() < 3 {
            return f64::NAN;
        }
        if is_constant(b) {
            return 0.0;
        }
        let m2 = b.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let m3 = b.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n;
        if m2 <= 0.0 {
            return f64::NAN;
        }
        (n * (n - 1.0)).sqrt() / (n - 2.0) * m3 / m2.powf(1.5)
    }))
}

/// Bias-corrected excess kurtosis (pandas `kurt`; needs >= 4 present values). A
/// constant window is exactly `-3` (pandas 3's degenerate-window rule).
pub fn kurt(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(per_moment(data, window, min_periods, |b, m| {
        let n = b.len() as f64;
        if b.len() < 4 {
            return f64::NAN;
        }
        if is_constant(b) {
            return -3.0;
        }
        let m2 = b.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let m4 = b.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
        if m2 <= 0.0 {
            return f64::NAN;
        }
        ((n + 1.0) * m4 / (m2 * m2) - 3.0 * (n - 1.0)) * (n - 1.0)
            / ((n - 2.0) * (n - 3.0))
    }))
}

// --- order statistics ----------------------------------------------------------

/// `q`-quantile of a SORTED buffer under a pandas interpolation mode.
fn sorted_quantile(sorted: &[f64], q: f64, interpolation: &str) -> f64 {
    let n = sorted.len();
    let pos = q * (n - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    let frac = pos - lo as f64;
    match interpolation {
        "lower" => sorted[lo],
        "higher" => sorted[hi],
        // a .5 tie rounds to the EVEN index (pandas/numpy 'nearest')
        "nearest" => sorted[if frac > 0.5 || (frac == 0.5 && lo % 2 == 1) { hi } else { lo }],
        "midpoint" => midpoint(sorted[lo], sorted[hi]),
        // "linear" (validated by the caller)
        _ => lerp(sorted[lo], sorted[hi], frac),
    }
}

/// `(a + b) / 2` (pandas's arithmetic) unless the sum alone overflows for finite
/// `a`, `b`, then `a/2 + b/2` (`[1e308, 1e308]` has midpoint `1e308`).
#[inline]
fn midpoint(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if sum.is_finite() || !(a.is_finite() && b.is_finite()) {
        sum / 2.0
    } else {
        a / 2.0 + b / 2.0
    }
}

/// `a + t·(b − a)` for sorted `a <= b`, `t ∈ [0, 1]` — without overflowing when
/// the span `b − a` of two finite values exceeds `f64::MAX` although the result is
/// representable (`[-1e308, 1e308]` has median `0`). The plain form (pandas's
/// arithmetic) is kept whenever the span is finite; the weighted form
/// `(1−t)·a + t·b` takes over only past it.
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let span = b - a;
    if span.is_finite() || !(a.is_finite() && b.is_finite()) {
        a + t * span
    } else {
        (1.0 - t) * a + t * b
    }
}

/// Run `f` over each window's present values kept sorted in IEEE total order — one
/// canonical multiset per window, so the row a value leaves at removes exactly that
/// value (a `-0.0` leaving never takes a `+0.0` with it) and a value never depends on
/// the rows before its window.
fn per_sorted_window(
    data: &[f64],
    window: usize,
    min_periods: usize,
    mut f: impl FnMut(&[f64], usize) -> f64,
) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    // a maintained sorted multiset of the window's present values
    let mut sorted: Vec<f64> = Vec::with_capacity(window.min(n));
    for i in 0..n {
        if window != usize::MAX && i >= window {
            let leaving = data[i - window];
            if !crate::missing::is_missing(leaving) {
                // the very value leaving (total order: a `-0.0` is not a `+0.0`)
                let pos = sorted.partition_point(|v| v.total_cmp(&leaving).is_lt());
                sorted.remove(pos);
            }
        }
        let x = data[i];
        if !crate::missing::is_missing(x) {
            let pos = sorted.partition_point(|v| v.total_cmp(&x).is_lt());
            sorted.insert(pos, x);
        }
        if sorted.len() >= min_periods.max(1) {
            out[i] = f(&sorted, i);
        }
    }
    out
}

/// Rolling / expanding median (pandas `median`).
pub fn median(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok(per_sorted_window(data, window, min_periods, |s, _| {
        sorted_quantile(s, 0.5, "linear")
    }))
}

/// Rolling / expanding quantile (pandas `quantile(q, interpolation)`).
pub fn quantile(
    data: &[f64],
    window: usize,
    min_periods: usize,
    q: f64,
    interpolation: &str,
) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    validate_quantile(q, interpolation)?;
    Ok(per_sorted_window(data, window, min_periods, |s, _| {
        sorted_quantile(s, q, interpolation)
    }))
}

/// Rank of the CURRENT row's value within its trailing window (pandas `rank`):
/// `method` is `average` / `min` / `max`, ties handled accordingly; `pct`
/// scales by the present count; a missing current value stays NaN.
pub fn rank(
    data: &[f64],
    window: usize,
    min_periods: usize,
    method: &str,
    ascending: bool,
    pct: bool,
) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    validate_rank(method)?;
    rank_padded(data, window, min_periods, method, ascending, pct, data.len())
}

/// [`rank`] over data whose rows from `data_len` on are trailing padding (a
/// centered window's forward reach): every window ranks its LAST DATA row — the
/// window's own last value, pandas's rank target — so a window ending in padding
/// ranks row `data_len - 1`, never a padding cell. `data_len = data.len()` is
/// [`rank`].
#[allow(clippy::too_many_arguments)]
pub fn rank_padded(
    data: &[f64],
    window: usize,
    min_periods: usize,
    method: &str,
    ascending: bool,
    pct: bool,
    data_len: usize,
) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    validate_rank(method)?;
    if data_len > data.len() || (data_len == 0 && !data.is_empty()) { return Err(WindowError::Padding); }
    Ok(per_sorted_window(data, window, min_periods, |s, i| {
        let x = data[i.min(data_len.saturating_sub(1))];
        if crate::missing::is_missing(x) {
            return f64::NAN;
        }
        let below = s.partition_point(|&v| v < x);
        let through = s.partition_point(|&v| v <= x);
        let (lo, hi) = if ascending {
            (below + 1, through)
        } else {
            (s.len() - through + 1, s.len() - below)
        };
        let r = match method {
            "min" => lo as f64,
            "max" => hi as f64,
            _ => (lo + hi) as f64 / 2.0, // "average" (validated by the caller)
        };
        if pct {
            r / s.len() as f64
        } else {
            r
        }
    }))
}

/// The hash key of a present value under NUMERIC equality: `0.0` and `-0.0` are
/// one value (their bit patterns differ). NaN never reaches a key (it is missing).
#[inline]
fn value_key(x: f64) -> u64 {
    if x == 0.0 {
        0.0f64.to_bits()
    } else {
        x.to_bits()
    }
}

/// Count of distinct present values per window (pandas `nunique`).
pub fn nunique(data: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    Ok({
        let n = data.len();
        let mut out = vec![f64::NAN; n];
        let mut counts: HashMap<u64, usize> = HashMap::new();
        let mut nobs = 0usize;
        for i in 0..n {
            if window != usize::MAX && i >= window {
                let leaving = data[i - window];
                if !crate::missing::is_missing(leaving) {
                    nobs -= 1;
                    let k = value_key(leaving);
                    let c = counts.get_mut(&k).expect("leaving value was inserted");
                    *c -= 1;
                    if *c == 0 {
                        counts.remove(&k);
                    }
                }
            }
            let x = data[i];
            if !crate::missing::is_missing(x) {
                nobs += 1;
                *counts.entry(value_key(x)).or_insert(0) += 1;
            }
            if nobs >= min_periods {
                out[i] = counts.len() as f64;
            }
        }
        out
    })
}

/// Position of the first / last PRESENT value in each window (`None` below
/// min_periods or for an all-missing window). The caller gathers from the
/// original column, so `first` / `last` preserve the source dtype.
pub fn edge_positions(
    valid: &[bool],
    window: usize,
    min_periods: usize,
    last: bool,
) -> Result<Vec<Option<usize>>, WindowError> {
    validate_window(window, min_periods)?;
    Ok({
        let n = valid.len();
        let mut out = vec![None; n];
        for i in 0..n {
            let start = win_start(i, window);
            let nobs = valid[start..=i].iter().filter(|&&v| v).count();
            if nobs >= min_periods {
                out[i] = if last {
                    (start..=i).rev().find(|&k| valid[k])
                } else {
                    (start..=i).find(|&k| valid[k])
                };
            }
        }
        out
    })
}

// --- two-series ---------------------------------------------------------------

fn pairwise(
    x: &[f64],
    y: &[f64],
    window: usize,
    min_periods: usize,
    mut f: impl FnMut(&[f64], &[f64]) -> f64,
) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    let mut bx: Vec<f64> = Vec::new();
    let mut by: Vec<f64> = Vec::new();
    #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
    for i in 0..n {
        bx.clear();
        by.clear();
        for k in win_start(i, window)..=i {
            // pandas pairwise rule: a row counts only when BOTH sides are present
            if !crate::missing::is_missing(x[k]) && !crate::missing::is_missing(y[k]) {
                bx.push(x[k]);
                by.push(y[k]);
            }
        }
        if bx.len() >= min_periods.max(1) {
            out[i] = f(&bx, &by);
        }
    }
    out
}

/// Rolling / expanding sample covariance of two series (pandas `cov`).
pub fn cov(x: &[f64], y: &[f64], window: usize, min_periods: usize, ddof: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    validate_pair(x, y)?;
    Ok(pairwise(x, y, window, min_periods, |bx, by| {
        if bx.len() <= ddof {
            return f64::NAN;
        }
        let n = bx.len() as f64;
        let (mx, my) = (
            window_mean(bx),
            window_mean(by),
        );
        bx.iter()
            .zip(by)
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>()
            / (n - ddof as f64)
    }))
}

/// Rolling / expanding Pearson correlation of two series (pandas `corr`).
pub fn corr(x: &[f64], y: &[f64], window: usize, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_window(window, min_periods)?;
    validate_pair(x, y)?;
    Ok(pairwise(x, y, window, min_periods, |bx, by| {
        let sx = buf_var(bx, 1).sqrt();
        let sy = buf_var(by, 1).sqrt();
        // A zero *or NaN* std has no correlation; the negated form catches both.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(sx > 0.0) || !(sy > 0.0) {
            return f64::NAN;
        }
        let n = bx.len() as f64;
        let (mx, my) = (
            window_mean(bx),
            window_mean(by),
        );
        let c = bx
            .iter()
            .zip(by)
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>()
            / (n - 1.0);
        c / (sx * sy)
    }))
}

// --- exponentially weighted ----------------------------------------------------

/// pandas `ewm(...).mean()` — both `adjust` modes, both `ignore_na` modes
/// (ports the pandas `ewm` kernel: with `ignore_na=False` the old weight keeps
/// decaying across a missing row; with `True` a gap is invisible).
pub fn ewm_mean(
    data: &[f64],
    alpha: f64,
    adjust: bool,
    ignore_na: bool,
    min_periods: usize,
) -> Result<Vec<f64>, WindowError> {
    validate_alpha(alpha)?;
    Ok({
        let old_wt_factor = 1.0 - alpha;
        let new_wt = if adjust { 1.0 } else { alpha };
        let mut avg = f64::NAN;
        let mut old_wt = 1.0;
        let mut nobs = 0usize;
        data.iter()
            .map(|&x| {
                let is_obs = !crate::missing::is_missing(x);
                nobs += is_obs as usize;
                if !avg.is_nan() {
                    if is_obs || !ignore_na {
                        old_wt *= old_wt_factor;
                        if is_obs {
                            if avg != x {
                                // pandas 3 uses the complement of the decayed weight
                                // for its com=1 recursive mean, including across gaps.
                                let wt = if !adjust && alpha == 0.5 { 1.0 - old_wt } else { new_wt };
                                avg = (old_wt * avg + wt * x) / (old_wt + wt);
                            }
                            if adjust {
                                old_wt += new_wt;
                            } else {
                                old_wt = 1.0;
                            }
                        }
                    }
                } else if is_obs {
                    avg = x;
                }
                if nobs >= min_periods.max(1) {
                    avg
                } else {
                    f64::NAN
                }
            })
            .collect()
    })
}

/// pandas `ewm(...).sum()` — the un-normalized weighted sum
/// (`adjust=True` only; pandas raises for `adjust=False`, and so does the caller).
pub fn ewm_sum(data: &[f64], alpha: f64, ignore_na: bool, min_periods: usize) -> Result<Vec<f64>, WindowError> {
    validate_alpha(alpha)?;
    Ok({
        let old_wt_factor = 1.0 - alpha;
        let mut s = f64::NAN;
        let mut nobs = 0usize;
        data.iter()
            .map(|&x| {
                let is_obs = !crate::missing::is_missing(x);
                nobs += is_obs as usize;
                if !s.is_nan() {
                    if is_obs || !ignore_na {
                        s *= old_wt_factor;
                        if is_obs {
                            s += x;
                        }
                    }
                } else if is_obs {
                    s = x;
                }
                if nobs >= min_periods.max(1) {
                    s
                } else {
                    f64::NAN
                }
            })
            .collect()
    })
}

/// pandas `ewm(...).cov(other)` kernel — also the engine behind `var` / `std`
/// (`y = x`) and `corr`. `bias=false` applies pandas's debiasing factor.
#[allow(clippy::too_many_arguments)]
pub fn ewm_cov(
    x: &[f64],
    y: &[f64],
    alpha: f64,
    adjust: bool,
    ignore_na: bool,
    bias: bool,
    min_periods: usize,
) -> Result<Vec<f64>, WindowError> {
    validate_alpha(alpha)?;
    validate_pair(x, y)?;
    Ok({
        let old_wt_factor = 1.0 - alpha;
        let new_wt = if adjust { 1.0 } else { alpha };
        let (mut mean_x, mut mean_y) = (f64::NAN, f64::NAN);
        let mut cov = 0.0;
        let (mut sum_wt, mut sum_wt2, mut old_wt) = (1.0, 1.0, 1.0);
        let mut nobs = 0usize;
        x.iter()
            .zip(y)
            .map(|(&xi, &yi)| {
                let is_obs = !crate::missing::is_missing(xi) && !crate::missing::is_missing(yi);
                nobs += is_obs as usize;
                if !mean_x.is_nan() {
                    if is_obs || !ignore_na {
                        sum_wt *= old_wt_factor;
                        sum_wt2 *= old_wt_factor * old_wt_factor;
                        old_wt *= old_wt_factor;
                        if is_obs {
                            let (old_mean_x, old_mean_y) = (mean_x, mean_y);
                            if mean_x != xi {
                                mean_x = (old_wt * old_mean_x + new_wt * xi) / (old_wt + new_wt);
                            }
                            if mean_y != yi {
                                mean_y = (old_wt * old_mean_y + new_wt * yi) / (old_wt + new_wt);
                            }
                            cov = (old_wt
                                * (cov + (old_mean_x - mean_x) * (old_mean_y - mean_y))
                                + new_wt * (xi - mean_x) * (yi - mean_y))
                                / (old_wt + new_wt);
                            sum_wt += new_wt;
                            sum_wt2 += new_wt * new_wt;
                            old_wt += new_wt;
                            if !adjust {
                                sum_wt /= old_wt;
                                sum_wt2 /= old_wt * old_wt;
                                old_wt = 1.0;
                            }
                        }
                    }
                } else if is_obs {
                    mean_x = xi;
                    mean_y = yi;
                }
                if nobs >= min_periods.max(1) {
                    if bias {
                        cov
                    } else {
                        let numerator = sum_wt * sum_wt;
                        let denominator = numerator - sum_wt2;
                        if denominator > 0.0 {
                            (numerator / denominator) * cov
                        } else {
                            f64::NAN
                        }
                    }
                } else {
                    f64::NAN
                }
            })
            .collect()
    })
}

/// pandas `ewm(...).var(bias)` — `ewm_cov(x, x)?`.
pub fn ewm_var(
    data: &[f64],
    alpha: f64,
    adjust: bool,
    ignore_na: bool,
    bias: bool,
    min_periods: usize,
) -> Result<Vec<f64>, WindowError> {
    validate_alpha(alpha)?;
    ewm_cov(data, data, alpha, adjust, ignore_na, bias, min_periods)
}

/// pandas `ewm(...).corr(other)`: `cov / sqrt(var_x · var_y)` with the biased
/// (bias=true) building blocks. Divide before multiplying to retain range.
pub fn ewm_corr(
    x: &[f64],
    y: &[f64],
    alpha: f64,
    adjust: bool,
    ignore_na: bool,
    min_periods: usize,
) -> Result<Vec<f64>, WindowError> {
    validate_alpha(alpha)?;
    validate_pair(x, y)?;
    Ok({
        let cv = ewm_cov(x, y, alpha, adjust, ignore_na, true, min_periods)?;
        // pandas masks each side to the PAIRWISE-present rows before the marginal
        // variances, so a row missing in `y` does not perturb `var(x)`.
        let n = x.len();
        let (mut xp, mut yp) = (vec![f64::NAN; n], vec![f64::NAN; n]);
        for i in 0..n {
            if !crate::missing::is_missing(x[i]) && !crate::missing::is_missing(y[i]) {
                xp[i] = x[i];
                yp[i] = y[i];
            }
        }
        let vx = ewm_cov(&xp, &xp, alpha, adjust, ignore_na, true, min_periods)?;
        let vy = ewm_cov(&yp, &yp, alpha, adjust, ignore_na, true, min_periods)?;
        cv.iter()
            .zip(vx.iter().zip(&vy))
            .map(|(&c, (&a, &b))| {
                if a > 0.0 && b > 0.0 {
                    // Divide by standard deviations separately: the variance product
                    // need never be representable. Equal variances preserve c/a exactly.
                    if a == b { c / a } else { (c / a.sqrt()) / b.sqrt() }
                } else {
                    f64::NAN
                }
            })
            .collect()
    })
}

#[cfg(test)]
mod tests;
