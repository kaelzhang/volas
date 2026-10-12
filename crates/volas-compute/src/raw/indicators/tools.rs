// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

// The boolean tools return truths that carry presence (contract #27): `None` is a
// missing truth — a comparison with a missing (NaN) operand, or a window reaching
// before the series start — and windows combine them with Kleene logic.

/// Kleene AND of each row's last `k` truths: `false` as soon as one present `false`
/// is in the window, `true` when the window is full and every truth in it is `true`,
/// missing otherwise (rows before the series start are missing). `k == 0` has no
/// window: every row is missing (the directive boundary rejects it).
pub fn all_of_last(truths: &[Option<bool>], k: usize) -> Vec<Option<bool>> {
    (0..truths.len())
        .map(|i| {
            let window = &truths[(i + 1).saturating_sub(k)..=i];
            if k == 0 {
                None
            } else if window.contains(&Some(false)) {
                Some(false)
            } else if i + 1 >= k && window.iter().all(|t| *t == Some(true)) {
                Some(true)
            } else {
                None
            }
        })
        .collect()
}

/// Whether values strictly increase (`direction > 0`) / decrease (`< 0`) over each
/// row's last `repeat` steps (the last `repeat + 1` values): the Kleene AND of the
/// steps, a step with a missing value being missing. `direction == 0` never holds.
pub fn increase(data: &[f64], repeat: usize, direction: i32) -> Vec<Option<bool>> {
    let want = match direction.signum() {
        1 => Some(std::cmp::Ordering::Greater),
        -1 => Some(std::cmp::Ordering::Less),
        _ => None,
    };
    let steps: Vec<Option<bool>> = (0..data.len())
        .map(|j| {
            let prev = data.get(j.wrapping_sub(1))?; // the first value has no step
            data[j].partial_cmp(prev).map(|o| Some(o) == want)
        })
        .collect();
    all_of_last(&steps, repeat)
}

/// Candlestick style. Parsing the DSL string (`"bullish"` / `"bearish"`) is the
/// directive layer's job — this numeric kernel takes a typed value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// close > open.
    Bullish,
    /// close < open.
    Bearish,
}

/// Whether each candle matches `style` (`Bullish` => close > open); missing when
/// either price is.
pub fn style(style: Style, open: &[f64], close: &[f64]) -> Vec<Option<bool>> {
    let want = match style {
        Style::Bullish => std::cmp::Ordering::Greater,
        Style::Bearish => std::cmp::Ordering::Less,
    };
    (0..open.len()).map(|i| close[i].partial_cmp(&open[i]).map(|o| o == want)).collect()
}

/// Whether a boolean condition holds for `repeat` consecutive periods (the Kleene
/// AND of [`all_of_last`]).
pub fn repeat(data: &[Option<bool>], repeat: usize) -> Vec<Option<bool>> {
    all_of_last(data, repeat)
}

/// Percentage change over `period` (`data[i] / data[i-(period-1)] - 1`).
pub fn change(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    // `period == 0` would underflow `period - 1`; the directive boundary
    // rejects it — this is the defensive in-kernel guard (all-NaN).
    if period == 0 {
        return result;
    }
    let shift = period - 1;
    for i in shift..n {
        let prev = data[i - shift];
        // An exact zero test, as TA-Lib's ROCP / ROCR: an absolute threshold would
        // drop the real changes of a tiny-priced series. A zero base stays NaN.
        if prev != 0.0 {
            result[i] = data[i] / prev - 1.0;
        }
    }
    result
}
