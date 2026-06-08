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
mod group_a;
mod hilbert;
mod math_ops;
mod momentum;
mod oscillators;
mod statistic;
mod stochastic;
mod tools;
mod transform;
mod trend;
mod volume;

pub use bands::*;
pub use candles::*;
pub use directional::*;
pub use group_a::*;
pub use hilbert::*;
pub use math_ops::*;
pub use momentum::*;
pub use oscillators::*;
pub use statistic::*;
pub use stochastic::*;
pub use tools::*;
pub use transform::*;
pub use trend::*;
pub use volume::*;

/// Shared deterministic test fixtures + the bit-exact state-carry oracle, reused by
/// every submodule's `#[cfg(test)] mod tests` (via `use crate::indicators::test_support::*;`).
/// Kept in the parent so the generators and the `assert_bits` comparator are not
/// duplicated across files.
#[cfg(test)]
pub(crate) mod test_support {
    /// A non-degenerate oscillating series (sine carrier + slow drift), long enough
    /// that every indicator is well past its warm-up. Iterating a resume's `from`
    /// across this naturally fires the rare rolling-extreme rebuild / new-extreme arms.
    pub fn series(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                100.0 + 0.05 * t + 6.0 * (t * 0.30).sin() + 2.0 * (t * 0.11).cos()
            })
            .collect()
    }

    /// Oscillating high/low/close triplet. The carrier swings hard enough (and the
    /// two sines beat against each other) that a Parabolic-SAR walk reverses in BOTH
    /// directions repeatedly, and the directional-movement extremes flip sign — so a
    /// resume iterated over many `from` offsets exercises every reversal arm.
    pub fn ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            let mid = 100.0 + 0.04 * t + 8.0 * (t * 0.25).sin() + 3.0 * (t * 0.13).cos();
            let span = 1.5 + 0.8 * (t * 0.37).sin().abs();
            high.push(mid + span);
            low.push(mid - span);
            // Close walks within the bar, sometimes near the high, sometimes the low.
            close.push(mid + span * 0.6 * (t * 0.41).sin());
        }
        (high, low, close)
    }

    /// Compare two slices for exact bit equality (NaN == NaN) — the state-carry oracle:
    /// a resume fed the carried state of a full compute over the head must reproduce
    /// the tail of a full compute over the whole input, bit-for-bit.
    pub fn assert_bits(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{what}: length {} != {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                "{what}: bar {i}: resume {x:?} != full {y:?}",
            );
        }
    }
}

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

    /// Zero-denominator / flat-data branches (no range, no variance, no directional
    /// movement) — the `… == 0 { 0.0 }` fall-backs that real price series never reach.
    #[test]
    fn flat_data_zero_denominator_branches() {
        const N: usize = 40;
        let f = vec![100.0; N]; // perfectly flat OHLC: every range / variance is 0
        let vol = vec![1000.0; N];
        // willr flat range; bop no range; cci zero deviation; mfi zero/again-flat flow.
        assert!(willr(&f, &f, &f, 14).iter().any(|x| *x == 0.0));
        assert!(bop(&f, &f, &f, &f).iter().all(|x| *x == 0.0));
        assert!(cci(&f, &f, &f, 14).iter().any(|x| *x == 0.0));
        assert!(mfi(&f, &f, &f, &vol, 14).iter().any(|x| *x == 0.0));
        // directional: smoothed TR is 0, so +DI / −DI / DX are 0.
        assert!(plus_di(&f, &f, &f, 14).iter().any(|x| *x == 0.0));
        assert!(minus_di(&f, &f, &f, 14).iter().any(|x| *x == 0.0));
        assert!(dx(&f, &f, &f, 14).iter().any(|x| *x == 0.0));
        // statistics: zero variance / zero regression denominator.
        assert!(stddev(&f, 5, 1.0).iter().any(|x| *x == 0.0));
        assert!(correl(&f, &f, 30).iter().any(|x| *x == 0.0));
        assert!(beta(&f, &f, 5).iter().any(|x| *x == 0.0));
        // oscillators: flat stochastic range; CMO's flat-window guard; volume: zero
        // money-flow multiplier.
        assert!(stoch_fastk(&f, &f, &f, 14).iter().any(|x| *x == 0.0));
        assert!(cmo(&f, 14).iter().any(|x| *x == 0.0));
        assert!(ad(&f, &f, &f, &vol).iter().any(|x| *x == 0.0));
        // kama efficiency ratio defaults to 1.0 when there is no net change.
        assert_eq!(kama(&f, 30).len(), N);

        // cmo skips NaN deltas (the shared gains/losses helper's `continue`).
        assert_eq!(cmo(&[100.0, f64::NAN, 102.0, 103.0, 104.0], 2).len(), 5);

        // DX inner zero: inside bars give nonzero TR but no directional movement, so
        // +DI + −DI == 0 while smoothed TR != 0.
        let hi: Vec<f64> = (0..N).map(|i| 110.0 - i as f64 * 0.3).collect();
        let lo: Vec<f64> = (0..N).map(|i| 90.0 + i as f64 * 0.3).collect();
        assert!(dx(&hi, &lo, &f, 14).iter().any(|x| *x == 0.0));

        // roc / beta: a zero prior price yields a 0 result (the divide-by-zero guard).
        let z = [0.0, 0.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        assert_eq!(roc(&z, 2)[2], 0.0);
        assert!(beta(&z, &f[..z.len()].to_vec(), 5)
            .iter()
            .any(|x| *x == 0.0));

        // accbands: high + low == 0 falls back to the bare high / low edge.
        let pos = vec![5.0; N];
        let neg = vec![-5.0; N];
        assert!(accbands_upper(&pos, &neg, 20)
            .iter()
            .any(|x| (*x - 5.0).abs() < 1e-9));
        assert!(accbands_lower(&pos, &neg, 20)
            .iter()
            .any(|x| (*x + 5.0).abs() < 1e-9));

        // adosc with fast == slow == 1 has lookback 0, so it emits at index 0.
        assert!(!adosc(&f, &f, &f, &vol, 1, 1)[0].is_nan());
    }

    /// SAR initial-direction branches (down-opening) and SAREXT start-value sign
    /// branches, which the upward-opening tencent series does not exercise.
    #[test]
    fn sar_initial_direction_branches() {
        // Down-opening: −DM at bar 1 is positive, so SAR starts short.
        let hi = [110.0, 105.0, 104.0, 106.0, 108.0, 107.0];
        let lo = [100.0, 95.0, 94.0, 96.0, 98.0, 97.0];
        assert_eq!(sar(&hi, &lo, 0.02, 0.2).len(), 6);
        // SAREXT with an explicit positive then negative start value.
        assert_eq!(
            sarext(&hi, &lo, 1.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2).len(),
            6
        );
        assert_eq!(
            sarext(&hi, &lo, -1.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2).len(),
            6
        );
    }

    #[test]
    fn assert_bits_reports_length_mismatch() {
        let err = std::panic::catch_unwind(|| {
            test_support::assert_bits(&[1.0], &[], "length-check");
        });
        assert!(err.is_err());
    }

    /// Warm-up guards reached only by degenerate periods / inputs.
    #[test]
    fn warmup_guards() {
        assert!(mom(&[1.0, 2.0], 5).iter().all(|x| x.is_nan())); // period >= n
        assert!(roc(&[1.0, 2.0], 5).iter().all(|x| x.is_nan()));
        assert!(rocp(&[1.0, 2.0], 5).iter().all(|x| x.is_nan()));
        assert_eq!(rocr(&[0.0, 2.0], 1)[1], 0.0);
        assert!(imi(&[1.0, 2.0], &[1.0, 2.0], 0).iter().all(|x| x.is_nan()));
        assert!(aroon_up(&[1.0], &[1.0], 0).iter().all(|x| x.is_nan())); // period == 0
        let f = vec![100.0; 5];
        assert!(ultosc(&f, &f, &f, 0, 14, 28).iter().all(|x| x.is_nan())); // p1 == 0
        assert!(adx(&f, &f, &f, 14).iter().all(|x| x.is_nan())); // 2*period-1 >= n
        assert!(t3(&f, 10, 0.7).iter().all(|x| x.is_nan())); // 6*(period-1) >= n
    }
}
