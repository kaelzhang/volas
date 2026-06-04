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
