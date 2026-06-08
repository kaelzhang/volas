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
}
