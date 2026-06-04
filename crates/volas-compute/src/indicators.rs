//! Technical indicators as pure-Rust functions over `f64` / `bool` slices.
//!
//! Ported from stock-pandas's Rust core (the pyo3 wrappers are stripped; the math
//! is unchanged). All float results use `NaN` for warm-up / undefined values.

use ndarray::{Array1, ArrayView1};

use crate::simd;

#[inline]
fn av(s: &[f64]) -> ArrayView1<'_, f64> {
    ArrayView1::from(s)
}

// ---------------------------------------------------------------------------
// Trend-following
// ---------------------------------------------------------------------------

/// Simple moving average.
pub fn ma(close: &[f64], period: usize) -> Vec<f64> {
    simd::sma(av(close), period).to_vec()
}

/// Exponential moving average (`com = (period - 1) / 2`).
pub fn ema(close: &[f64], period: usize) -> Vec<f64> {
    let com = (period as f64 - 1.0) / 2.0;
    simd::ewma_com(av(close), com, true, false, period).to_vec()
}

/// Smoothed moving average.
pub fn smma(close: &[f64], period: usize) -> Vec<f64> {
    simd::smma(av(close), period).to_vec()
}

fn macd_line(close: &[f64], fast: usize, slow: usize) -> Array1<f64> {
    let data = av(close);
    let com_fast = (fast as f64 - 1.0) / 2.0;
    let com_slow = (slow as f64 - 1.0) / 2.0;
    // fast and slow EWMA in one fused pass (was two ewma_com traversals)
    let (f, s) = simd::dual_ewma(data, com_fast, fast, data, com_slow, slow, true, false);
    &f - &s
}

/// MACD line (DIF).
pub fn macd(close: &[f64], fast: usize, slow: usize) -> Vec<f64> {
    macd_line(close, fast, slow).to_vec()
}

/// MACD signal line (DEA).
pub fn macd_signal(close: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let line = macd_line(close, fast, slow);
    let com = (signal as f64 - 1.0) / 2.0;
    simd::ewma_com(line.view(), com, true, false, signal).to_vec()
}

/// MACD histogram (`2 * (MACD - signal)`).
pub fn macd_histogram(close: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let line = macd_line(close, fast, slow);
    let com = (signal as f64 - 1.0) / 2.0;
    let sig = simd::ewma_com(line.view(), com, true, false, signal);
    (2.0 * (&line - &sig)).to_vec()
}

/// Bull and Bear Index (`mean of ma:a, ma:b, ma:c, ma:d`).
pub fn bbi(close: &[f64], a: usize, b: usize, c: usize, d: usize) -> Vec<f64> {
    let data = av(close);
    let ma_a = simd::sma(data, a);
    let ma_b = simd::sma(data, b);
    let ma_c = simd::sma(data, c);
    let ma_d = simd::sma(data, d);
    ((&ma_a + &ma_b + &ma_c + &ma_d) / 4.0).to_vec()
}

/// True Range.
pub fn tr(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut tr = vec![f64::NAN; n];
    if n > 0 {
        tr[0] = high[0] - low[0];
    }
    for i in 1..n {
        let prev_close = close[i - 1];
        let hl = high[i] - low[i];
        let hc = (high[i] - prev_close).abs();
        let lc = (low[i] - prev_close).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    tr
}

/// Average True Range (`ma:period` of TR).
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let tr = tr(high, low, close);
    simd::sma(av(&tr), period).to_vec()
}

// ---------------------------------------------------------------------------
// Support / resistance
// ---------------------------------------------------------------------------

/// Bollinger middle band (= MA).
pub fn boll(close: &[f64], period: usize) -> Vec<f64> {
    simd::sma(av(close), period).to_vec()
}

/// Bollinger upper band (`ma + times * std`, population std).
pub fn boll_upper(close: &[f64], period: usize, times: f64) -> Vec<f64> {
    let data = av(close);
    let ma = simd::sma(data, period);
    let std = simd::rolling_std(data, period, 0);
    (&ma + times * &std).to_vec()
}

/// Bollinger lower band (`ma - times * std`, population std).
pub fn boll_lower(close: &[f64], period: usize, times: f64) -> Vec<f64> {
    let data = av(close);
    let ma = simd::sma(data, period);
    let std = simd::rolling_std(data, period, 0);
    (&ma - times * &std).to_vec()
}

/// Bollinger Band Width (`4 * std / ma`).
pub fn bbw(close: &[f64], period: usize) -> Vec<f64> {
    let data = av(close);
    let ma = simd::sma(data, period);
    let std = simd::rolling_std(data, period, 0);
    (4.0 * &std / &ma).to_vec()
}

/// Historical Volatility.
pub fn hv(close: &[f64], period: usize, minutes: i64, trading_days: i64) -> Vec<f64> {
    let n = close.len();
    let mut log_return = Array1::from_elem(n, f64::NAN);
    for i in 1..n {
        if close[i - 1] > 0.0 && close[i] > 0.0 {
            log_return[i] = (close[i] / close[i - 1]).ln();
        }
    }
    let std = simd::rolling_std(log_return.view(), period, 1);
    let day_minutes = 1440.0;
    let annualization = ((trading_days as f64) * day_minutes / (minutes as f64)).sqrt();
    (&std * annualization).to_vec()
}

// ---------------------------------------------------------------------------
// Overbought / oversold
// ---------------------------------------------------------------------------

/// Lowest of low values.
pub fn llv(data: &[f64], period: usize) -> Vec<f64> {
    simd::rolling_min(av(data), period).to_vec()
}

/// Highest of high values.
pub fn hhv(data: &[f64], period: usize) -> Vec<f64> {
    simd::rolling_max(av(data), period).to_vec()
}

/// Raw Stochastic Value.
pub fn rsv(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let llv = simd::rolling_min(av(low), period);
    let hhv = simd::rolling_max(av(high), period);
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        let denom = hhv[i] - llv[i];
        if denom.abs() > 1e-10 {
            result[i] = (close[i] - llv[i]) / denom * 100.0;
        } else {
            result[i] = 0.0;
        }
    }
    result
}

fn kdj_rsv(high: &[f64], low: &[f64], close: &[f64], period_rsv: usize) -> Array1<f64> {
    let llv = simd::rolling_min(av(low), period_rsv);
    let hhv = simd::rolling_max(av(high), period_rsv);
    let n = close.len();
    let mut rsv = Array1::from_elem(n, 0.0);
    for i in 0..n {
        let denom = hhv[i] - llv[i];
        if denom.abs() > 1e-10 {
            rsv[i] = (close[i] - llv[i]) / denom * 100.0;
        }
    }
    rsv
}

/// KDJ %K line.
pub fn kdj_k(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    init: f64,
) -> Vec<f64> {
    let rsv = kdj_rsv(high, low, close, period_rsv);
    simd::ewma_with_init(rsv.view(), period_k, init).to_vec()
}

/// KDJ %D line.
pub fn kdj_d(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    init: f64,
) -> Vec<f64> {
    let rsv = kdj_rsv(high, low, close, period_rsv);
    let k = simd::ewma_with_init(rsv.view(), period_k, init);
    simd::ewma_with_init(k.view(), period_d, init).to_vec()
}

/// KDJ %J line (`3K - 2D`).
pub fn kdj_j(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    init: f64,
) -> Vec<f64> {
    let rsv = kdj_rsv(high, low, close, period_rsv);
    let k = simd::ewma_with_init(rsv.view(), period_k, init);
    let d = simd::ewma_with_init(k.view(), period_d, init);
    (3.0 * &k - 2.0 * &d).to_vec()
}

/// Relative Strength Index.
pub fn rsi(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let delta = simd::diff(av(close));
    let mut gains = Array1::from_elem(n, f64::NAN);
    let mut losses = Array1::from_elem(n, f64::NAN);
    for i in 1..n {
        let d = delta[i];
        if d.is_nan() {
            continue;
        }
        gains[i] = d.max(0.0);
        losses[i] = (-d).max(0.0);
    }
    // both smoothed averages in one fused pass (smma = ewma_com, com = period-1)
    let com = (period - 1) as f64;
    let (sg, sl) = simd::dual_ewma(gains.view(), com, period, losses.view(), com, period, true, false);
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        if sg[i].is_nan() || sl[i].is_nan() {
            continue;
        }
        if sl[i].abs() < 1e-10 {
            result[i] = 100.0;
        } else {
            result[i] = 100.0 - 100.0 / (1.0 + sg[i] / sl[i]);
        }
    }
    result
}

/// Donchian middle channel (`(hhv + llv) / 2`).
pub fn donchian(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let hhv = simd::rolling_max(av(high), period);
    let llv = simd::rolling_min(av(low), period);
    ((&hhv + &llv) / 2.0).to_vec()
}

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
        assert!((t[0] - 2.0).abs() < 1e-10);
        // tr[1] = max(hl=3, hc=|12-9|=3, lc=|9-9|=0) = 3
        assert!((t[1] - 3.0).abs() < 1e-10);
        let a = atr(&high, &low, &close, 2);
        assert!(a[0].is_nan());
        assert!((a[1] - 2.5).abs() < 1e-10);
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
        assert_eq!(style(Style::Bullish, &[1.0, 2.0], &[2.0, 1.0]), vec![true, false]);
        assert_eq!(style(Style::Bearish, &[1.0, 2.0], &[2.0, 1.0]), vec![false, true]);
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
