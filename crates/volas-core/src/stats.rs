//! Pure NaN-skipping numeric algorithms over `f64` slices, shared by the Series
//! / DataFrame numeric methods in the binding layer. They live here (not in the
//! pyo3 binding) so they stay unit-tested and coverage-measured.
//!
//! NaN handling mirrors pandas's `skipna=True`: a NaN does not contribute to a
//! running accumulator / reduction, and in the cumulative forms it is kept as
//! NaN in place.

/// Cumulative sum (pandas `cumsum`, skipna=True).
pub fn cumsum(v: &[f64]) -> Vec<f64> {
    let mut acc = 0.0;
    v.iter()
        .map(|&x| {
            if x.is_nan() {
                f64::NAN
            } else {
                acc += x;
                acc
            }
        })
        .collect()
}

/// Cumulative maximum (pandas `cummax`, skipna=True).
pub fn cummax(v: &[f64]) -> Vec<f64> {
    cum_extreme(v, f64::max)
}

/// Cumulative minimum (pandas `cummin`, skipna=True).
pub fn cummin(v: &[f64]) -> Vec<f64> {
    cum_extreme(v, f64::min)
}

/// Shared running-extreme for `cummax` / `cummin`.
fn cum_extreme(v: &[f64], pick: fn(f64, f64) -> f64) -> Vec<f64> {
    let mut acc = f64::NAN;
    v.iter()
        .map(|&x| {
            if x.is_nan() {
                f64::NAN
            } else {
                acc = if acc.is_nan() { x } else { pick(acc, x) };
                acc
            }
        })
        .collect()
}

/// Cumulative product (pandas `cumprod`, skipna=True).
pub fn cumprod(v: &[f64]) -> Vec<f64> {
    let mut acc = 1.0;
    v.iter()
        .map(|&x| {
            if x.is_nan() {
                f64::NAN
            } else {
                acc *= x;
                acc
            }
        })
        .collect()
}

/// NaN-skipping product (1.0 when empty / all-NaN, matching pandas).
pub fn prod(v: &[f64]) -> f64 {
    v.iter().filter(|x| !x.is_nan()).product()
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
    let mut idx: Vec<usize> = (0..v.len()).filter(|&i| !v[i].is_nan()).collect();
    let cnt = idx.len();
    // Sort surviving positions by value (stable, so `First` keeps input order).
    idx.sort_by(|&a, &b| {
        let o = v[a].partial_cmp(&v[b]).unwrap();
        if ascending {
            o
        } else {
            o.reverse()
        }
    });

    let mut out = vec![f64::NAN; v.len()];
    let mut dense_rank = 0.0;
    let mut i = 0;
    while i < idx.len() {
        // [i, j) is a run of tied values.
        let mut j = i + 1;
        while j < idx.len() && v[idx[j]] == v[idx[i]] {
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
    fn product_skips_nan_and_empty_is_one() {
        assert_eq!(prod(&[1.0, nan(), 2.0, 3.0]), 6.0);
        assert_eq!(prod(&[nan(), nan()]), 1.0);
        assert_eq!(prod(&[]), 1.0);
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
}
