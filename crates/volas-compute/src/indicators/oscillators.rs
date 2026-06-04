use ndarray::Array1;

use super::av;
use crate::kernels;

// ---------------------------------------------------------------------------
// Overbought / oversold
// ---------------------------------------------------------------------------

/// Lowest of low values.
pub fn llv(data: &[f64], period: usize) -> Vec<f64> {
    // Move the kernel's owned buffer out (no copy) rather than `to_vec`.
    kernels::rolling_min(av(data), period)
        .into_raw_vec_and_offset()
        .0
}

/// Highest of high values.
pub fn hhv(data: &[f64], period: usize) -> Vec<f64> {
    kernels::rolling_max(av(data), period)
        .into_raw_vec_and_offset()
        .0
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

/// TA-Lib stochastic raw %K: `100·(close − LL) / (HH − LL)` over `period` (HH/LL the
/// highest high / lowest low). A flat range yields 0, but — unlike [`rsv`] — the
/// warm-up is NaN, so the smoothing MAs in `stoch`/`stochf` begin at the right row.
/// Lookback `period-1`.
pub fn stoch_fastk(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let hh = kernels::rolling_max(av(high), period);
    let ll = kernels::rolling_min(av(low), period);
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if !hh[i].is_nan() {
            let diff = hh[i] - ll[i];
            out[i] = if diff != 0.0 {
                100.0 * (close[i] - ll[i]) / diff
            } else {
                0.0
            };
        }
    }
    out
}

/// TA-Lib StochRSI raw %K: the stochastic %K of the RSI line — `stoch_fastk` applied
/// to `rsi(close, rsi_period)` over `fastk_period`. Because RSI itself warms up, the
/// %K is masked to NaN until a full `fastk_period` window of finite RSI is available
/// (index `rsi_period + fastk_period - 1`). The `fastd` line is then `ma_typed` of this.
pub fn stochrsi_fastk(close: &[f64], rsi_period: usize, fastk_period: usize) -> Vec<f64> {
    let rsi = rsi(close, rsi_period);
    let mut fk = stoch_fastk(&rsi, &rsi, &rsi, fastk_period);
    // rolling max/min skip NaN, so they would emit over a partial RSI window; suppress
    // those rows — the %K is valid only once the whole window of RSI is finite.
    let start = (rsi_period + fastk_period).saturating_sub(1).min(fk.len());
    for v in fk.iter_mut().take(start) {
        *v = f64::NAN;
    }
    fk
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
///
/// Single-pass / single-allocation: seed the Wilder average gain & loss as the SMA
/// of the first `period` bar-to-bar changes, emit from index `period`, then
/// Wilder-smooth in place. Bit-identical to `wilder(gains)` / `wilder(losses)` +
/// combine (same seed sum, same recursion, same flat-window guard) but ~one sixth
/// the memory traffic — no `diff` / `gains` / `losses` / two smoothed arrays.
pub fn rsi(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return out;
    }
    let pf = period as f64;
    let p1 = pf - 1.0;
    let emit = |g: f64, l: f64| {
        if l.abs() < 1e-10 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + g / l)
        }
    };
    // Seed: SMA of the first `period` gains / losses (deltas at indices 1..=period).
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in 1..=period {
        let d = close[i] - close[i - 1];
        if d > 0.0 {
            avg_gain += d;
        } else {
            avg_loss -= d;
        }
    }
    avg_gain /= pf;
    avg_loss /= pf;
    out[period] = emit(avg_gain, avg_loss);
    for i in (period + 1)..n {
        let d = close[i] - close[i - 1];
        let (gain, loss) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
        avg_gain = (avg_gain * p1 + gain) / pf;
        avg_loss = (avg_loss * p1 + loss) / pf;
        out[i] = emit(avg_gain, avg_loss);
    }
    out
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
