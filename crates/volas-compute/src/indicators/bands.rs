use ndarray::Array1;

use super::av;
use crate::kernels;

// ---------------------------------------------------------------------------
// Support / resistance
// ---------------------------------------------------------------------------

/// Bollinger middle band (= MA).
pub fn boll(close: &[f64], period: usize) -> Vec<f64> {
    kernels::sma(av(close), period).to_vec()
}

/// Bollinger upper band (`ma + times * std`, population std).
pub fn boll_upper(close: &[f64], period: usize, times: f64) -> Vec<f64> {
    let (ma, std) = kernels::rolling_mean_std(av(close), period, 0);
    (&ma + times * &std).to_vec()
}

/// Bollinger lower band (`ma - times * std`, population std).
pub fn boll_lower(close: &[f64], period: usize, times: f64) -> Vec<f64> {
    let (ma, std) = kernels::rolling_mean_std(av(close), period, 0);
    (&ma - times * &std).to_vec()
}

/// Bollinger Band Width (`4 * std / ma`).
pub fn bbw(close: &[f64], period: usize) -> Vec<f64> {
    let (ma, std) = kernels::rolling_mean_std(av(close), period, 0);
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
    let std = kernels::rolling_std(log_return.view(), period, 1);
    let day_minutes = 1440.0;
    let annualization = ((trading_days as f64) * day_minutes / (minutes as f64)).sqrt();
    (&std * annualization).to_vec()
}

/// The raw (pre-SMA) Acceleration Band edge for each bar: `high·(1 + 4·(h-l)/(h+l))`
/// for the upper edge, `low·(1 − 4·(h-l)/(h+l))` for the lower. A zero `high+low`
/// falls back to the bare high/low (TA-Lib's guard).
fn accbands_raw(high: &[f64], low: &[f64], upper: bool) -> Vec<f64> {
    (0..high.len())
        .map(|i| {
            let sum = high[i] + low[i];
            if sum.abs() < 1e-14 {
                if upper {
                    high[i]
                } else {
                    low[i]
                }
            } else {
                let factor = 4.0 * (high[i] - low[i]) / sum;
                if upper {
                    high[i] * (1.0 + factor)
                } else {
                    low[i] * (1.0 - factor)
                }
            }
        })
        .collect()
}

/// Acceleration Bands middle (TA-Lib ACCBANDS middle) — the SMA of close, identical
/// to the Bollinger middle band. Lookback `period-1`.
pub fn accbands_middle(close: &[f64], period: usize) -> Vec<f64> {
    kernels::sma(av(close), period).to_vec()
}

/// Acceleration Bands upper (TA-Lib ACCBANDS upper) — SMA of the raw upper edge.
pub fn accbands_upper(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let raw = accbands_raw(high, low, true);
    kernels::sma(av(&raw), period).to_vec()
}

/// Acceleration Bands lower (TA-Lib ACCBANDS lower) — SMA of the raw lower edge.
pub fn accbands_lower(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let raw = accbands_raw(high, low, false);
    kernels::sma(av(&raw), period).to_vec()
}
