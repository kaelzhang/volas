//! pandas-semantics windowed aggregations — the single kernel source for the
//! python `rolling` / `expanding` / `ewm` API surface.
//!
//! Distinct from [`crate::indicators`] (TA-Lib semantics: a NaN poisons its
//! window): here `NaN` means *missing* and is skipped, and a row emits a value
//! once the window holds at least `min_periods` present values — exactly
//! pandas's window model. `window == usize::MAX` is an expanding window.
//!
//! All kernels take `&[f64]` (the python layer funnels int/bool, and routes the
//! dtype-preserving members `first` / `last` / `count` / `nunique` through the
//! position / count kernels instead).

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
            if !x.is_nan() {
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
) -> Vec<f64> {
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
            data[start..=end].iter().filter(|x| !x.is_nan()).count() as f64
        })
        .collect()
}

/// Drive a leaving/entering sliding scan; returns the per-row state snapshots.
fn sliding<T: Copy>(
    data: &[f64],
    window: usize,
    mut update: impl FnMut(f64, bool) -> T,
) -> Vec<T> {
    (0..data.len())
        .map(|i| {
            if window != usize::MAX && i >= window {
                update(data[i - window], false);
            }
            update(data[i], true)
        })
        .collect()
}

/// Rolling / expanding sum of the present values (pandas `sum`).
pub fn sum(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let mut s = 0.0f64;
    let mut nobs = 0usize;
    sliding(data, window, |x, entering| {
        if !x.is_nan() {
            if entering {
                s += x;
                nobs += 1;
            } else {
                s -= x;
                nobs -= 1;
            }
        }
        (s, nobs)
    })
    .into_iter()
    .map(|(s, c)| if c >= min_periods.max(1) { s } else { f64::NAN })
    .collect()
}

/// Rolling / expanding mean of the present values (pandas `mean`).
pub fn mean(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let mut s = 0.0f64;
    let mut nobs = 0usize;
    sliding(data, window, |x, entering| {
        if !x.is_nan() {
            if entering {
                s += x;
                nobs += 1;
            } else {
                s -= x;
                nobs -= 1;
            }
        }
        (s, nobs)
    })
    .into_iter()
    .map(|(s, c)| {
        if c >= min_periods.max(1) {
            s / c as f64
        } else {
            f64::NAN
        }
    })
    .collect()
}

// --- min / max (monotonic deque, O(n)) ---------------------------------------

fn extremum(data: &[f64], window: usize, min_periods: usize, is_min: bool) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let better = |a: f64, b: f64| if is_min { a <= b } else { a >= b };
    let mut nobs = 0usize;
    for i in 0..n {
        if window != usize::MAX && i >= window && !data[i - window].is_nan() {
            nobs -= 1;
        }
        if let Some(&front) = deque.front() {
            if window != usize::MAX && front + window <= i {
                deque.pop_front();
            }
        }
        if !data[i].is_nan() {
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
pub fn min(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    extremum(data, window, min_periods, true)
}

/// Rolling / expanding maximum (pandas `max`).
pub fn max(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    extremum(data, window, min_periods, false)
}

// --- dispersion / moments -----------------------------------------------------

fn buf_var(b: &[f64], ddof: usize) -> f64 {
    if b.len() <= ddof {
        return f64::NAN;
    }
    let m = b.iter().sum::<f64>() / b.len() as f64;
    b.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (b.len() - ddof) as f64
}

/// Rolling / expanding sample variance (pandas `var`, default ddof=1).
/// Per-window two-pass — numerically stable, no sliding cancellation.
pub fn var(data: &[f64], window: usize, min_periods: usize, ddof: usize) -> Vec<f64> {
    per_window(data, window, min_periods, |b| buf_var(b, ddof))
}

/// Rolling / expanding standard deviation (pandas `std`).
pub fn std(data: &[f64], window: usize, min_periods: usize, ddof: usize) -> Vec<f64> {
    per_window(data, window, min_periods, |b| buf_var(b, ddof).sqrt())
}

/// Standard error of the mean (pandas `sem`): `std(ddof) / sqrt(count)`.
pub fn sem(data: &[f64], window: usize, min_periods: usize, ddof: usize) -> Vec<f64> {
    per_window(data, window, min_periods, |b| {
        (buf_var(b, ddof) / b.len() as f64).sqrt()
    })
}

/// Mean absolute deviation about the window mean (TradingView `ta.dev`):
/// `mean(|x - mean(x)|)` over the present values — the dispersion measure CCI
/// is built on, and the one statistic missing from var/std/sem.
pub fn dev(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    per_window(data, window, min_periods, |b| {
        let m = b.iter().sum::<f64>() / b.len() as f64;
        b.iter().map(|x| (x - m).abs()).sum::<f64>() / b.len() as f64
    })
}

/// Rolling mode (TradingView `ta.mode`): the most frequent present value; ties
/// resolve to the SMALLEST value. NA cells are ignored (matching Pine).
pub fn mode(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    per_sorted_window(data, window, min_periods, |sorted, _| {
        // `sorted` is ascending, so equal values are contiguous: scan runs and
        // keep the value with the longest run; on a tie the earlier (smaller)
        // value is already held, giving "ties -> smallest" for free.
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
        best_val
    })
}

/// Bias-corrected sample skewness (pandas `skew`; needs >= 3 present values).
pub fn skew(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    per_window(data, window, min_periods, |b| {
        let n = b.len() as f64;
        if b.len() < 3 {
            return f64::NAN;
        }
        let m = b.iter().sum::<f64>() / n;
        let m2 = b.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let m3 = b.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n;
        if m2 <= 0.0 {
            return f64::NAN;
        }
        (n * (n - 1.0)).sqrt() / (n - 2.0) * m3 / m2.powf(1.5)
    })
}

/// Bias-corrected excess kurtosis (pandas `kurt`; needs >= 4 present values).
pub fn kurt(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    per_window(data, window, min_periods, |b| {
        let n = b.len() as f64;
        if b.len() < 4 {
            return f64::NAN;
        }
        let m = b.iter().sum::<f64>() / n;
        let m2 = b.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let m4 = b.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
        if m2 <= 0.0 {
            return f64::NAN;
        }
        ((n + 1.0) * m4 / (m2 * m2) - 3.0 * (n - 1.0)) * (n - 1.0)
            / ((n - 2.0) * (n - 3.0))
    })
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
        "midpoint" => (sorted[lo] + sorted[hi]) / 2.0,
        // "linear" (validated by the caller)
        _ => sorted[lo] + frac * (sorted[hi] - sorted[lo]),
    }
}

fn per_sorted_window(
    data: &[f64],
    window: usize,
    min_periods: usize,
    mut f: impl FnMut(&[f64], f64) -> f64,
) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    // a maintained sorted multiset of the window's present values
    let mut sorted: Vec<f64> = Vec::with_capacity(window.min(n));
    for i in 0..n {
        if window != usize::MAX && i >= window {
            let leaving = data[i - window];
            if !leaving.is_nan() {
                let pos = sorted.partition_point(|&v| v < leaving);
                sorted.remove(pos);
            }
        }
        let x = data[i];
        if !x.is_nan() {
            let pos = sorted.partition_point(|&v| v < x);
            sorted.insert(pos, x);
        }
        if sorted.len() >= min_periods.max(1) {
            out[i] = f(&sorted, x);
        }
    }
    out
}

/// Rolling / expanding median (pandas `median`).
pub fn median(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    per_sorted_window(data, window, min_periods, |s, _| {
        sorted_quantile(s, 0.5, "linear")
    })
}

/// Rolling / expanding quantile (pandas `quantile(q, interpolation)`).
pub fn quantile(
    data: &[f64],
    window: usize,
    min_periods: usize,
    q: f64,
    interpolation: &str,
) -> Vec<f64> {
    per_sorted_window(data, window, min_periods, |s, _| {
        sorted_quantile(s, q, interpolation)
    })
}

/// Rank of the CURRENT row's value within its window (pandas `rank`):
/// `method` is `average` / `min` / `max`, ties handled accordingly; `pct`
/// scales by the present count; a missing current value stays NaN.
pub fn rank(
    data: &[f64],
    window: usize,
    min_periods: usize,
    method: &str,
    ascending: bool,
    pct: bool,
) -> Vec<f64> {
    per_sorted_window(data, window, min_periods, |s, x| {
        if x.is_nan() {
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
    })
}

/// Count of distinct present values per window (pandas `nunique`).
pub fn nunique(data: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let mut counts: HashMap<u64, usize> = HashMap::new();
    let mut nobs = 0usize;
    for i in 0..n {
        if window != usize::MAX && i >= window {
            let leaving = data[i - window];
            if !leaving.is_nan() {
                nobs -= 1;
                let k = leaving.to_bits();
                let c = counts.get_mut(&k).expect("leaving value was inserted");
                *c -= 1;
                if *c == 0 {
                    counts.remove(&k);
                }
            }
        }
        let x = data[i];
        if !x.is_nan() {
            nobs += 1;
            *counts.entry(x.to_bits()).or_insert(0) += 1;
        }
        if nobs >= min_periods.max(1) {
            out[i] = counts.len() as f64;
        }
    }
    out
}

/// Position of the first / last PRESENT value in each window (`None` below
/// min_periods or for an all-missing window). The caller gathers from the
/// original column, so `first` / `last` preserve the source dtype.
pub fn edge_positions(
    valid: &[bool],
    window: usize,
    min_periods: usize,
    last: bool,
) -> Vec<Option<usize>> {
    let n = valid.len();
    let mut out = vec![None; n];
    for i in 0..n {
        let start = win_start(i, window);
        let nobs = valid[start..=i].iter().filter(|&&v| v).count();
        if nobs >= min_periods.max(1) {
            out[i] = if last {
                (start..=i).rev().find(|&k| valid[k])
            } else {
                (start..=i).find(|&k| valid[k])
            };
        }
    }
    out
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
    for i in 0..n {
        bx.clear();
        by.clear();
        for k in win_start(i, window)..=i {
            // pandas pairwise rule: a row counts only when BOTH sides are present
            if !x[k].is_nan() && !y[k].is_nan() {
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
pub fn cov(x: &[f64], y: &[f64], window: usize, min_periods: usize, ddof: usize) -> Vec<f64> {
    pairwise(x, y, window, min_periods, |bx, by| {
        if bx.len() <= ddof {
            return f64::NAN;
        }
        let n = bx.len() as f64;
        let (mx, my) = (
            bx.iter().sum::<f64>() / n,
            by.iter().sum::<f64>() / n,
        );
        bx.iter()
            .zip(by)
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>()
            / (n - ddof as f64)
    })
}

/// Rolling / expanding Pearson correlation of two series (pandas `corr`).
pub fn corr(x: &[f64], y: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    pairwise(x, y, window, min_periods, |bx, by| {
        let sx = buf_var(bx, 1).sqrt();
        let sy = buf_var(by, 1).sqrt();
        // A zero *or NaN* std has no correlation; the negated form catches both.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(sx > 0.0) || !(sy > 0.0) {
            return f64::NAN;
        }
        let n = bx.len() as f64;
        let (mx, my) = (
            bx.iter().sum::<f64>() / n,
            by.iter().sum::<f64>() / n,
        );
        let c = bx
            .iter()
            .zip(by)
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>()
            / (n - 1.0);
        c / (sx * sy)
    })
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
) -> Vec<f64> {
    let old_wt_factor = 1.0 - alpha;
    let new_wt = if adjust { 1.0 } else { alpha };
    let mut avg = f64::NAN;
    let mut old_wt = 1.0;
    let mut nobs = 0usize;
    data.iter()
        .map(|&x| {
            let is_obs = !x.is_nan();
            nobs += is_obs as usize;
            if !avg.is_nan() {
                if is_obs || !ignore_na {
                    old_wt *= old_wt_factor;
                    if is_obs {
                        if avg != x {
                            avg = (old_wt * avg + new_wt * x) / (old_wt + new_wt);
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
}

/// pandas `ewm(...).sum()` — the un-normalized weighted sum
/// (`adjust=True` only; pandas raises for `adjust=False`, and so does the caller).
pub fn ewm_sum(data: &[f64], alpha: f64, ignore_na: bool, min_periods: usize) -> Vec<f64> {
    let old_wt_factor = 1.0 - alpha;
    let mut s = f64::NAN;
    let mut nobs = 0usize;
    data.iter()
        .map(|&x| {
            let is_obs = !x.is_nan();
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
) -> Vec<f64> {
    let old_wt_factor = 1.0 - alpha;
    let new_wt = if adjust { 1.0 } else { alpha };
    let (mut mean_x, mut mean_y) = (f64::NAN, f64::NAN);
    let mut cov = 0.0;
    let (mut sum_wt, mut sum_wt2, mut old_wt) = (1.0, 1.0, 1.0);
    let mut nobs = 0usize;
    x.iter()
        .zip(y)
        .map(|(&xi, &yi)| {
            let is_obs = !xi.is_nan() && !yi.is_nan();
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
}

/// pandas `ewm(...).var(bias)` — `ewm_cov(x, x)`.
pub fn ewm_var(
    data: &[f64],
    alpha: f64,
    adjust: bool,
    ignore_na: bool,
    bias: bool,
    min_periods: usize,
) -> Vec<f64> {
    ewm_cov(data, data, alpha, adjust, ignore_na, bias, min_periods)
}

/// pandas `ewm(...).corr(other)`: `cov / sqrt(var_x · var_y)` with the biased
/// (bias=true) building blocks, exactly as pandas computes it.
pub fn ewm_corr(
    x: &[f64],
    y: &[f64],
    alpha: f64,
    adjust: bool,
    ignore_na: bool,
    min_periods: usize,
) -> Vec<f64> {
    let cv = ewm_cov(x, y, alpha, adjust, ignore_na, true, min_periods);
    // pandas masks each side to the PAIRWISE-present rows before the marginal
    // variances, so a row missing in `y` does not perturb `var(x)`.
    let n = x.len();
    let (mut xp, mut yp) = (vec![f64::NAN; n], vec![f64::NAN; n]);
    for i in 0..n {
        if !x[i].is_nan() && !y[i].is_nan() {
            xp[i] = x[i];
            yp[i] = y[i];
        }
    }
    let vx = ewm_cov(&xp, &xp, alpha, adjust, ignore_na, true, min_periods);
    let vy = ewm_cov(&yp, &yp, alpha, adjust, ignore_na, true, min_periods);
    cv.iter()
        .zip(vx.iter().zip(&vy))
        .map(|(&c, (&a, &b))| {
            let d = (a * b).sqrt();
            if d > 0.0 {
                c / d
            } else {
                f64::NAN
            }
        })
        .collect()
}
