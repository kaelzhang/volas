//! Technical indicators as pure-Rust functions over `f64` / `bool` slices.
//!
//! Ported from stock-pandas's Rust core (the pyo3 wrappers are stripped; the math
//! is unchanged). All float results use `NaN` for warm-up / undefined values.
//!
//! Organised by responsibility into one submodule per indicator family; every
//! function is re-exported here, so callers keep using `indicators::<name>`
//! regardless of which submodule owns it.

use ndarray::ArrayView1;

/// View a slice as an `ndarray` 1-D view (the kernels' input type). Shared by the
/// submodules that delegate to `kernels`.
#[inline]
pub(crate) fn av(s: &[f64]) -> ArrayView1<'_, f64> {
    ArrayView1::from(s)
}

mod bands;
mod candles;
mod directional;
mod math_ops;
mod momentum;
mod oscillators;
mod statistic;
mod tools;
mod transform;
mod trend;
mod volume;

pub use bands::*;
pub use candles::*;
pub use directional::*;
pub use math_ops::*;
pub use momentum::*;
pub use oscillators::*;
pub use statistic::*;
pub use tools::*;
pub use transform::*;
pub use trend::*;
pub use volume::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ma_basic() {
        let r = ma(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert!(r[0].is_nan() && r[1].is_nan());
        assert!((r[2] - 2.0).abs() < 1e-10);
        assert!((r[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn tr_atr_basic() {
        let high = [10.0, 12.0, 11.0];
        let low = [8.0, 9.0, 10.0];
        let close = [9.0, 11.0, 10.5];
        let t = tr(&high, &low, &close);
        assert!(t[0].is_nan()); // TA-Lib: no TR at index 0
                                // tr[1] = max(hl=3, hc=|12-9|=3, lc=|9-9|=0) = 3 ; tr[2] = max(1,0,1) = 1
        assert!((t[1] - 3.0).abs() < 1e-10);
        assert!((t[2] - 1.0).abs() < 1e-10);
        // atr:2 SMA-seeded Wilder -> seed at idx 2 = mean(tr[1], tr[2]) = mean(3,1) = 2
        let a = atr(&high, &low, &close, 2);
        assert!(a[0].is_nan() && a[1].is_nan());
        assert!((a[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn rsi_bounds() {
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let r = rsi(&close, 14);
        // strictly increasing -> RSI 100 once defined
        assert!((r[29] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn style_and_repeat() {
        assert_eq!(
            style(Style::Bullish, &[1.0, 2.0], &[2.0, 1.0]),
            vec![true, false]
        );
        assert_eq!(
            style(Style::Bearish, &[1.0, 2.0], &[2.0, 1.0]),
            vec![false, true]
        );
        let r = repeat(&[true, true, true, false], 2);
        assert_eq!(r, vec![false, true, true, false]);
    }

    #[test]
    fn nan_delta_and_oversized_windows() {
        // rsi skips NaN deltas (the `continue` branch)
        let r = rsi(&[1.0, f64::NAN, 3.0, 4.0, 5.0], 2);
        assert_eq!(r.len(), 5);
        // increase / repeat return all-false when the window exceeds the length
        assert_eq!(increase(&[1.0, 2.0], 5, 1), vec![false, false]);
        assert_eq!(repeat(&[true, true], 5), vec![false, false]);
    }
}
