//! TA-Lib's generic moving average `TA_MA`: one MA-type code selecting the average —
//! 0 SMA, 1 EMA, 2 WMA, 3 DEMA, 4 TEMA, 5 TRIMA, 6 KAMA, 7 MAMA, 8 T3. The one owner of
//! what a `(period, matype)` pair means: the type actually run, its warm-up, whether it
//! is recursive, and the dispatch. Every generic-MA use (`ma`, `apo`, `ppo`, `macdext`,
//! `mavp`, the stochastics, and [`super::FiniteMa`]'s resumable stages) reads it here.

/// The MA-type code `TA_MA` actually runs over `period`: a period of 1 is the identity
/// for every type — `TA_MA` checks it before dispatching on the type, so even KAMA and
/// MAMA (which otherwise ignores the period) return the input, with no warm-up. volas
/// runs that identity as the 1-bar simple average, whose value is the input.
pub fn generic_ma_type(period: usize, matype: usize) -> usize {
    if period == 1 { 0 } else { matype }
}

/// Whether the generic MA of `period` and `matype` is recursive: EMA, DEMA, TEMA, KAMA,
/// MAMA, T3 — not SMA, WMA, TRIMA, nor any type at period 1.
pub fn is_recursive_generic_ma(period: usize, matype: usize) -> bool {
    matches!(generic_ma_type(period, matype), 1 | 3 | 4 | 6 | 7 | 8)
}

/// The generic MA's lookback: DEMA `2·(period-1)`, TEMA `3·(period-1)`, KAMA `period`,
/// T3 `6·(period-1)`, MAMA a fixed `32` (period-independent), SMA / EMA / WMA / TRIMA
/// `period-1` — and 0 for any type at period 1.
pub fn generic_ma_lookback(period: usize, matype: usize) -> usize {
    match generic_ma_type(period, matype) {
        3 => 2 * period.saturating_sub(1),
        4 => 3 * period.saturating_sub(1),
        6 => period,
        7 => 32,
        8 => 6 * period.saturating_sub(1),
        _ => period.saturating_sub(1),
    }
}

/// The generic moving average of `period` and `matype` over `data` (row 0 at absolute
/// position `origin`). MAMA returns its primary line with `TA_MA`'s fixed 0.5 / 0.05
/// limits, T3 its fixed volume factor 0.7. An unknown type code gives all-NaN (the
/// directive layer rejects it when binding).
pub fn generic_ma(data: &[f64], period: usize, matype: usize, origin: usize) -> Vec<f64> {
    match generic_ma_type(period, matype) {
        0 => super::ma(data, period, origin),
        1 => super::ema(data, period),
        2 => super::wma(data, period, origin),
        3 => super::dema(data, period),
        4 => super::tema(data, period),
        5 => super::trima(data, period, origin),
        6 => super::kama(data, period, origin),
        7 => crate::raw::indicators::mama(data, 0.5, 0.05).0,
        8 => super::t3(data, period, 0.7),
        _ => vec![f64::NAN; data.len()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn period_one_is_the_input_for_every_type() {
        let x = [3.0, 1.5, f64::NAN, 4.25, 2.0];
        for matype in 0..9 {
            let got = generic_ma(&x, 1, matype, 0);
            assert!(got.iter().zip(&x).all(|(g, v)| g.to_bits() == v.to_bits() || g.is_nan() && v.is_nan()));
            assert_eq!(generic_ma_lookback(1, matype), 0);
            assert!(!is_recursive_generic_ma(1, matype));
        }
        assert!(is_recursive_generic_ma(2, 6) && !is_recursive_generic_ma(2, 5));
        assert_eq!(generic_ma_lookback(30, 7), 32);
        assert!(generic_ma(&x, 2, 9, 0).iter().all(|v| v.is_nan()));
    }

    /// The resumable stages read the rule too: any type at period 1 is the identity SMA
    /// stage, so a caller cannot reach KAMA / MAMA at period 1 through them.
    #[test]
    fn resumable_stages_apply_the_period_one_identity() {
        use crate::raw::indicators::{ma_state, ma_step, FiniteMa};
        assert!(FiniteMa::new(1, 6).is_some() && FiniteMa::new(2, 6).is_none());
        let x = [3.0, 1.5, 4.25];
        let state = ma_state(&x[..2], 1, 7, 0).expect("period 1 resumes for every type");
        let mut next = state.clone();
        assert_eq!(ma_step(&x, 1, 7, 2, &state, &mut next), Some(4.25));
    }
}
