//! Pure NaN-skipping numeric algorithms over `f64` slices, shared by the Series
//! / DataFrame numeric methods in the binding layer. They live here (not in the
//! pyo3 binding) so they stay unit-tested and coverage-measured.
//!
//! NaN handling mirrors pandas's `skipna=True`: a NaN does not contribute to a
//! running accumulator / reduction, and in the cumulative forms it is kept as
//! NaN in place.
//!
//! The cumulative kernels are generic over [`Numeric`] so they preserve the
//! input dtype (int stays int) and compute natively (no f64 round-trip); the
//! reductions / moment kernels stay f64. A missing element (`NaN`, f64 only —
//! `i64::is_missing` is always false) passes through unchanged.

use std::cmp::Ordering;

use crate::numeric::Numeric;

/// Cumulative sum (pandas `cumsum`, skipna=True), dtype-preserving.
pub fn cumsum<T: Numeric>(v: &[T]) -> Vec<T> {
    let mut acc = T::ZERO;
    v.iter()
        .map(|&x| {
            if x.is_missing() {
                x
            } else {
                acc = acc.wrapping_add(x);
                acc
            }
        })
        .collect()
}

/// Cumulative maximum (pandas `cummax`, skipna=True), dtype-preserving.
pub fn cummax<T: Numeric>(v: &[T]) -> Vec<T> {
    cum_extreme(v, true)
}

/// Cumulative minimum (pandas `cummin`, skipna=True), dtype-preserving.
pub fn cummin<T: Numeric>(v: &[T]) -> Vec<T> {
    cum_extreme(v, false)
}

/// Shared running-extreme for `cummax` (`want_max`) / `cummin`. NaN is excluded
/// before any comparison, so the partial order is total over what we compare.
fn cum_extreme<T: Numeric>(v: &[T], want_max: bool) -> Vec<T> {
    let mut acc: Option<T> = None;
    v.iter()
        .map(|&x| {
            if x.is_missing() {
                return x;
            }
            let next = match acc {
                Some(a) if (x > a) != want_max => a,
                _ => x,
            };
            acc = Some(next);
            next
        })
        .collect()
}

/// Element-wise absolute value (pandas `abs`), dtype-preserving; a missing value
/// passes through. Generic so the wrapping `abs` resolves to the [`Numeric`] impl
/// (matching pandas int64 overflow: `abs(i64::MIN) == i64::MIN`).
pub fn abs<T: Numeric>(v: &[T]) -> Vec<T> {
    v.iter()
        .map(|&x| if x.is_missing() { x } else { x.wrapping_abs() })
        .collect()
}

/// Cumulative product (pandas `cumprod`, skipna=True), dtype-preserving.
pub fn cumprod<T: Numeric>(v: &[T]) -> Vec<T> {
    let mut acc = T::ONE;
    v.iter()
        .map(|&x| {
            if x.is_missing() {
                x
            } else {
                acc = acc.wrapping_mul(x);
                acc
            }
        })
        .collect()
}

/// Missing-skipping sum (`0` when empty / all-missing, matching pandas),
/// dtype-preserving (i64 sums in i64, wrapping like pandas).
pub fn sum<T: Numeric>(v: &[T]) -> T {
    let mut acc = T::ZERO;
    for &x in v {
        if !x.is_missing() {
            acc = acc.wrapping_add(x);
        }
    }
    acc
}

/// Missing-skipping product (`1` when empty / all-missing, matching pandas),
/// dtype-preserving.
pub fn prod<T: Numeric>(v: &[T]) -> T {
    let mut acc = T::ONE;
    for &x in v {
        if !x.is_missing() {
            acc = acc.wrapping_mul(x);
        }
    }
    acc
}

/// Missing-skipping minimum (`want_max = false`) / maximum, dtype-preserving;
/// `None` when empty / all-missing. The `want_max` branch is hoisted out of the
/// loop so each fold is a single tight comparison (as fast as a specialized
/// min/max, and no intermediate allocation).
pub fn extreme<T: Numeric>(v: &[T], want_max: bool) -> Option<T> {
    let mut it = v.iter().copied().filter(|x| !x.is_missing());
    let first = it.next()?;
    Some(if want_max {
        it.fold(first, |a, x| if x > a { x } else { a })
    } else {
        it.fold(first, |a, x| if x < a { x } else { a })
    })
}

/// The finite (non-NaN) values, in order.
fn non_nan(v: &[f64]) -> Vec<f64> {
    v.iter().copied().filter(|x| !x.is_nan()).collect()
}

/// Standard error of the mean (pandas `sem`, ddof=1): `sample_std / sqrt(n)`.
/// NaN with fewer than two finite values.
pub fn sem(v: &[f64]) -> f64 {
    let xs = non_nan(v);
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0);
    (var / nf).sqrt()
}

/// Adjusted Fisher-Pearson sample skewness (pandas `skew`). NaN with fewer than
/// three finite values; 0.0 for a zero-variance sample.
pub fn skew(v: &[f64]) -> f64 {
    let xs = non_nan(v);
    let n = xs.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let m2: f64 = xs.iter().map(|&x| (x - mean).powi(2)).sum();
    let m3: f64 = xs.iter().map(|&x| (x - mean).powi(3)).sum();
    if m2 == 0.0 {
        return 0.0;
    }
    (nf * (nf - 1.0).sqrt() / (nf - 2.0)) * (m3 / m2.powf(1.5))
}

/// Bias-corrected excess kurtosis (pandas `kurt`, Fisher's definition). NaN with
/// fewer than four finite values; 0.0 for a zero-variance sample.
pub fn kurt(v: &[f64]) -> f64 {
    let xs = non_nan(v);
    let n = xs.len();
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let m2: f64 = xs.iter().map(|&x| (x - mean).powi(2)).sum();
    let m4: f64 = xs.iter().map(|&x| (x - mean).powi(4)).sum();
    if m2 == 0.0 {
        return 0.0;
    }
    let num = nf * (nf + 1.0) * (nf - 1.0) * m4;
    let den = (nf - 2.0) * (nf - 3.0) * m2 * m2;
    let adj = 3.0 * (nf - 1.0).powi(2) / ((nf - 2.0) * (nf - 3.0));
    num / den - adj
}

/// Element-wise choose: `cond[i] ? a[i] : b[i]`. Backs `where` / `mask`. Generic
/// so it preserves dtype (picks i64 natively). `cond` / `a` / `b` are equal length.
pub fn select<T: Numeric>(cond: &[bool], a: &[T], b: &[T]) -> Vec<T> {
    cond.iter()
        .enumerate()
        .map(|(i, &c)| if c { a[i] } else { b[i] })
        .collect()
}

/// The aligned `(x, y)` pairs with neither value NaN (pandas pairwise NaN drop).
fn pairs(x: &[f64], y: &[f64]) -> Vec<(f64, f64)> {
    x.iter()
        .zip(y)
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(&a, &b)| (a, b))
        .collect()
}

/// Pairwise sample covariance, ddof=1 (pandas `cov`); NaN with fewer than two
/// finite pairs. `x` / `y` are aligned positionally.
pub fn cov(x: &[f64], y: &[f64]) -> f64 {
    let p = pairs(x, y);
    let n = p.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = p.iter().map(|t| t.0).sum::<f64>() / nf;
    let my = p.iter().map(|t| t.1).sum::<f64>() / nf;
    p.iter().map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (nf - 1.0)
}

/// Pairwise Pearson correlation (pandas `corr`); NaN with fewer than two finite
/// pairs or a zero-variance input. `x` / `y` are aligned positionally.
pub fn corr(x: &[f64], y: &[f64]) -> f64 {
    let p = pairs(x, y);
    let n = p.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = p.iter().map(|t| t.0).sum::<f64>() / nf;
    let my = p.iter().map(|t| t.1).sum::<f64>() / nf;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in &p {
        let (dx, dy) = (a - mx, b - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        return f64::NAN;
    }
    // sqrt(sxx*syy) (not sqrt(sxx)*sqrt(syy)) so a column with itself is exactly
    // 1.0; clamp to [-1, 1] against rounding (matching NumPy / pandas).
    (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)
}

/// How tied values share ranks (pandas `rank(method=)`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RankMethod {
    /// Mean of the tied positions (pandas default).
    Average,
    /// Lowest tied position.
    Min,
    /// Highest tied position.
    Max,
    /// Distinct, in order of appearance.
    First,
    /// Ties share one rank; the next group is +1 (no gaps).
    Dense,
}

/// Rank the finite values (pandas `rank`, 1-based, `na_option='keep'` so NaN
/// stays NaN). `pct` divides by the count of finite values (max dense rank for
/// `Dense`).
pub fn rank(v: &[f64], method: RankMethod, ascending: bool, pct: bool) -> Vec<f64> {
    rank_by(
        v.len(),
        |i| !v[i].is_nan(),
        |a, b| v[a].partial_cmp(&v[b]).unwrap(),
        method,
        ascending,
        pct,
    )
}

/// Order-based rank over `n` positions, driven by an `is_valid` predicate and a
/// total `cmp` over present positions — so the one rank algorithm (tie handling,
/// `pct`) serves every ordered dtype (numeric by value, `str` lexically,
/// `datetime` by raw `i64`), not just `f64`. Invalid positions rank `NaN`; `cmp`
/// is only ever called on positions that pass `is_valid`.
pub fn rank_by(
    n: usize,
    is_valid: impl Fn(usize) -> bool,
    cmp: impl Fn(usize, usize) -> Ordering,
    method: RankMethod,
    ascending: bool,
    pct: bool,
) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..n).filter(|&i| is_valid(i)).collect();
    let cnt = idx.len();
    // Sort surviving positions by value (stable, so `First` keeps input order).
    idx.sort_by(|&a, &b| {
        let o = cmp(a, b);
        if ascending {
            o
        } else {
            o.reverse()
        }
    });

    let mut out = vec![f64::NAN; n];
    let mut dense_rank = 0.0;
    let mut i = 0;
    while i < idx.len() {
        // [i, j) is a run of tied values (equal regardless of sort direction).
        let mut j = i + 1;
        while j < idx.len() && cmp(idx[j], idx[i]) == Ordering::Equal {
            j += 1;
        }
        dense_rank += 1.0;
        for (k, &pos) in idx[i..j].iter().enumerate() {
            out[pos] = match method {
                RankMethod::Average => (i + j + 1) as f64 / 2.0, // mean of 1-based [i+1, j]
                RankMethod::Min => (i + 1) as f64,
                RankMethod::Max => j as f64,
                RankMethod::First => (i + k + 1) as f64,
                RankMethod::Dense => dense_rank,
            };
        }
        i = j;
    }

    if pct && cnt > 0 {
        let denom = if method == RankMethod::Dense {
            dense_rank
        } else {
            cnt as f64
        };
        for x in out.iter_mut() {
            if !x.is_nan() {
                *x /= denom;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nan() -> f64 {
        f64::NAN
    }

    fn eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(&x, &y)| (x == y) || (x.is_nan() && y.is_nan()))
    }

    #[test]
    fn cumulatives_skip_nan() {
        assert!(eq(&cumsum(&[1.0, nan(), 2.0, 3.0]), &[1.0, nan(), 3.0, 6.0]));
        assert!(eq(&cummax(&[1.0, nan(), 2.0, 4.0]), &[1.0, nan(), 2.0, 4.0]));
        assert!(eq(&cummin(&[3.0, nan(), 1.0, 2.0]), &[3.0, nan(), 1.0, 1.0]));
        assert!(eq(&cumprod(&[1.0, nan(), 2.0, 4.0]), &[1.0, nan(), 2.0, 8.0]));
        // leading NaN: accumulator stays unset until the first finite value
        assert!(eq(&cummax(&[nan(), 5.0, 3.0]), &[nan(), 5.0, 5.0]));
        assert!(eq(&cumsum(&[]), &[]));
    }

    #[test]
    fn reductions_skip_missing_and_preserve_dtype() {
        // float: skip NaN; empty/all-NaN -> identity
        assert_eq!(prod(&[1.0, nan(), 2.0, 3.0]), 6.0);
        assert_eq!(prod(&[nan(), nan()]), 1.0);
        assert_eq!(prod::<f64>(&[]), 1.0);
        assert_eq!(sum(&[1.0, nan(), 2.0, 3.0]), 6.0);
        assert_eq!(sum::<f64>(&[]), 0.0);
        // int: native i64 (wrapping like pandas)
        assert_eq!(sum(&[1_i64, 2, 3]), 6);
        assert_eq!(prod(&[2_i64, 3, 4]), 24);
        assert_eq!(sum(&[i64::MAX, 1]), i64::MIN); // wraps
        // extreme: skip NaN, None when empty/all-missing
        assert_eq!(extreme(&[3.0, nan(), 1.0, 2.0], false), Some(1.0));
        assert_eq!(extreme(&[3.0, nan(), 1.0, 2.0], true), Some(3.0));
        assert_eq!(extreme(&[1_i64, 5, 2], true), Some(5));
        assert_eq!(extreme::<f64>(&[], false), None);
        assert_eq!(extreme(&[nan(), nan()], true), None);
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * a.abs().max(1.0)
    }

    #[test]
    fn moments_match_pandas() {
        // values cross-checked against pandas 3.0
        let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!(close(sem(&v), 0.755_928_946_018));
        assert!(close(skew(&v), 0.818_487_553_4));
        assert!(close(kurt(&v), 0.940_625));
        // NaN is skipped before counting
        assert!(close(sem(&[2.0, nan(), 4.0]), sem(&[2.0, 4.0])));
        // too-few-values guards
        assert!(sem(&[5.0]).is_nan());
        assert!(skew(&[1.0, 2.0]).is_nan());
        assert!(kurt(&[1.0, 2.0, 3.0]).is_nan());
        // zero-variance samples -> 0.0
        assert_eq!(skew(&[3.0, 3.0, 3.0]), 0.0);
        assert_eq!(kurt(&[3.0, 3.0, 3.0, 3.0]), 0.0);
    }

    #[test]
    fn rank_methods_and_pct() {
        use RankMethod::*;
        let v = [3.0, 1.0, 1.0, 2.0, nan()];
        let r = |m, asc, pct| rank(&v, m, asc, pct);
        assert!(eq(&r(Average, true, false), &[4.0, 1.5, 1.5, 3.0, nan()]));
        assert!(eq(&r(Min, true, false), &[4.0, 1.0, 1.0, 3.0, nan()]));
        assert!(eq(&r(Max, true, false), &[4.0, 2.0, 2.0, 3.0, nan()]));
        assert!(eq(&r(First, true, false), &[4.0, 1.0, 2.0, 3.0, nan()]));
        assert!(eq(&r(Dense, true, false), &[3.0, 1.0, 1.0, 2.0, nan()]));
        // descending
        assert!(eq(&r(Min, false, false), &[1.0, 3.0, 3.0, 2.0, nan()]));
        // pct: average / count(=4); dense / max-dense(=3)
        assert!(eq(&r(Average, true, true), &[1.0, 0.375, 0.375, 0.75, nan()]));
        assert!(eq(&r(Dense, true, true), &[1.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, nan()]));
        // all-NaN: every rank is NaN, pct is a no-op
        assert!(eq(&rank(&[nan(), nan()], Average, true, true), &[nan(), nan()]));
    }

    #[test]
    fn corr_cov_pairwise() {
        assert!(close(corr(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 1.0));
        assert!(close(corr(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), -1.0));
        assert!(close(cov(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 1.0)); // == sample var
        // NaN pairs are dropped (drops index 1 -> corr([1,3],[1,3]) == 1)
        assert!(close(corr(&[1.0, nan(), 3.0], &[1.0, 5.0, 3.0]), 1.0));
        // guards: fewer than two pairs / zero variance -> NaN
        assert!(corr(&[1.0], &[2.0]).is_nan());
        assert!(cov(&[1.0], &[2.0]).is_nan());
        assert!(corr(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_nan());
    }

    #[test]
    fn select_picks_per_condition() {
        let cond = [true, false, true];
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        assert_eq!(select(&cond, &a, &b), vec![1.0, 20.0, 3.0]);
        // backs `mask` by swapping the branches
        assert_eq!(select(&cond, &b, &a), vec![10.0, 2.0, 30.0]);
    }
}
