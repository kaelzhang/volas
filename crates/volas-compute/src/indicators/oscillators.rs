use ndarray::Array1;

use super::av;
use crate::kernels;

// ---------------------------------------------------------------------------
// Overbought / oversold
// ---------------------------------------------------------------------------

/// Lowest of low values.
pub fn llv(data: &[f64], period: usize) -> Vec<f64> {
    kernels::rolling_min(av(data), period).to_vec()
}

/// Highest of high values.
pub fn hhv(data: &[f64], period: usize) -> Vec<f64> {
    kernels::rolling_max(av(data), period).to_vec()
}

/// Raw Stochastic Value.
pub fn rsv(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let llv = kernels::rolling_min(av(low), period);
    let hhv = kernels::rolling_max(av(high), period);
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
    let llv = kernels::rolling_min(av(low), period_rsv);
    let hhv = kernels::rolling_max(av(high), period_rsv);
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
    kernels::ewma_with_init(rsv.view(), period_k, init).to_vec()
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
    let k = kernels::ewma_with_init(rsv.view(), period_k, init);
    kernels::ewma_with_init(k.view(), period_d, init).to_vec()
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
    let k = kernels::ewma_with_init(rsv.view(), period_k, init);
    let d = kernels::ewma_with_init(k.view(), period_d, init);
    (3.0 * &k - 2.0 * &d).to_vec()
}

/// SMA-seeded Wilder average gain and average loss of `data`'s bar-to-bar changes
/// — the shared core of RSI and CMO. Both outputs are NaN until the first smoothed
/// value (at index `period`).
fn wilder_gain_loss(data: &[f64], period: usize) -> (Array1<f64>, Array1<f64>) {
    let n = data.len();
    let delta = kernels::diff(av(data));
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
    (
        kernels::wilder(gains.view(), period),
        kernels::wilder(losses.view(), period),
    )
}

/// Relative Strength Index (TA-Lib RSI): `100·avgGain/(avgGain+avgLoss)`.
pub fn rsi(close: &[f64], period: usize) -> Vec<f64> {
    let (sg, sl) = wilder_gain_loss(close, period);
    let n = close.len();
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

/// Chande Momentum Oscillator (TA-Lib CMO): `100·(avgGain−avgLoss)/(avgGain+avgLoss)`
/// over the same Wilder-smoothed gains/losses as RSI; a flat window (gain+loss = 0)
/// yields 0. Lookback `period`. (Algebraically `2·RSI − 100`; computed directly so
/// the flat-window guard matches TA-Lib exactly rather than inheriting RSI's.)
pub fn cmo(close: &[f64], period: usize) -> Vec<f64> {
    let (sg, sl) = wilder_gain_loss(close, period);
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        if sg[i].is_nan() || sl[i].is_nan() {
            continue;
        }
        let denom = sg[i] + sl[i];
        result[i] = if denom < 1e-14 {
            0.0
        } else {
            100.0 * (sg[i] - sl[i]) / denom
        };
    }
    result
}

/// Donchian middle channel (`(hhv + llv) / 2`).
pub fn donchian(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let hhv = kernels::rolling_max(av(high), period);
    let llv = kernels::rolling_min(av(low), period);
    ((&hhv + &llv) / 2.0).to_vec()
}

/// Midpoint over `period` of a single series: `(max + min) / 2` (TA-Lib MIDPOINT).
/// Lookback `period-1`.
pub fn midpoint(data: &[f64], period: usize) -> Vec<f64> {
    let hh = kernels::rolling_max(av(data), period);
    let ll = kernels::rolling_min(av(data), period);
    ((&hh + &ll) / 2.0).to_vec()
}

/// Midpoint price over `period`: `(max(high) + min(low)) / 2` (TA-Lib MIDPRICE).
/// Lookback `period-1`. (Same arithmetic as the Donchian middle channel.)
pub fn midprice(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let hh = kernels::rolling_max(av(high), period);
    let ll = kernels::rolling_min(av(low), period);
    ((&hh + &ll) / 2.0).to_vec()
}
