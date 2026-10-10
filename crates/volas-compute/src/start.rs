//! Where a series starts.
//!
//! A leading run of rows where an input is missing (NaN) is not data — the series
//! simply starts after it. A recursive kernel (EMA cascade, Wilder smoothing,
//! cumulative sums, the Hilbert transform, SAR) would otherwise seed from, or be
//! poisoned by, those rows. So it runs on its inputs from their first complete row and
//! pads the result back with NaN: a series with a missing head then agrees, row for
//! row, with the same series started later. The carried-state captures
//! (`*_final_state`) start the same way, so an incremental resume continues the same
//! recursion. Finite-window kernels get this for free from window locality (see
//! [`crate::guard`]).

/// The first row at which every input is present (`n` when none is).
pub(crate) fn first_complete_row(inputs: &[&[f64]]) -> usize {
    let n = inputs.first().map_or(0, |x| x.len());
    (0..n)
        .find(|&i| inputs.iter().all(|x| !x[i].is_nan()))
        .unwrap_or(n)
}

/// Prefix `tail` (the output of a run that started at row `start`) with `start` NaN
/// rows, restoring the full length.
pub(crate) fn pad_front(start: usize, tail: Vec<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(start + tail.len());
    out.resize(start, f64::NAN);
    out.extend(tail);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_complete_row_skips_rows_missing_any_input() {
        let a = [f64::NAN, 1.0, 2.0, 3.0];
        let b = [0.0, f64::NAN, 2.0, 3.0];
        assert_eq!(first_complete_row(&[&a[..], &b[..]]), 2);
        assert_eq!(first_complete_row(&[&b[2..]]), 0);
        assert_eq!(first_complete_row(&[&[f64::NAN, f64::NAN][..]]), 2);
        assert_eq!(first_complete_row(&[]), 0);
    }

    #[test]
    fn an_infinity_is_a_value_not_a_missing_row() {
        assert_eq!(first_complete_row(&[&[f64::INFINITY, 1.0][..]]), 0);
    }

    #[test]
    fn pad_front_restores_the_length() {
        let out = pad_front(2, vec![5.0, 6.0]);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!(&out[2..], &[5.0, 6.0]);
        assert_eq!(pad_front(0, vec![1.0]), vec![1.0]);
    }
}
