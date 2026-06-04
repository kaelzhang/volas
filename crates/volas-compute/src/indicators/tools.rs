// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// Whether values increase (`direction = 1`) / decrease (`-1`) over a window.
pub fn increase(data: &[f64], repeat: usize, direction: i32) -> Vec<bool> {
    let n = data.len();
    let period = repeat + 1;
    let mut result = vec![false; n];
    if period > n {
        return result;
    }
    for i in (period - 1)..n {
        let mut is_increasing = true;
        let mut current = if direction == 1 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        for j in (i + 1 - period)..=i {
            let value = data[j];
            if (value - current) * (direction as f64) > 0.0 {
                current = value;
            } else {
                is_increasing = false;
                break;
            }
        }
        result[i] = is_increasing;
    }
    result
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

/// Whether each candle matches `style` (`Bullish` => close > open).
pub fn style(style: Style, open: &[f64], close: &[f64]) -> Vec<bool> {
    (0..open.len())
        .map(|i| match style {
            Style::Bullish => close[i] > open[i],
            Style::Bearish => close[i] < open[i],
        })
        .collect()
}

/// Whether a boolean condition holds for `repeat` consecutive periods.
pub fn repeat(data: &[bool], repeat: usize) -> Vec<bool> {
    let n = data.len();
    if repeat == 1 {
        return data.to_vec();
    }
    let mut result = vec![false; n];
    if repeat > n {
        return result;
    }
    for i in (repeat - 1)..n {
        let mut all_true = true;
        for &d in &data[(i + 1 - repeat)..=i] {
            if !d {
                all_true = false;
                break;
            }
        }
        result[i] = all_true;
    }
    result
}

/// Percentage change over `period` (`data[i] / data[i-(period-1)] - 1`).
pub fn change(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let shift = period - 1;
    let mut result = vec![f64::NAN; n];
    for i in shift..n {
        let prev = data[i - shift];
        if prev.abs() > 1e-10 {
            result[i] = data[i] / prev - 1.0;
        }
    }
    result
}
