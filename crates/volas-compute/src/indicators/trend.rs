use ndarray::Array1;

use super::av;
use crate::kernels;

// ---------------------------------------------------------------------------
// Trend-following
// ---------------------------------------------------------------------------

/// Simple moving average.
pub fn ma(close: &[f64], period: usize) -> Vec<f64> {
    kernels::sma(av(close), period).to_vec()
}

/// Exponential moving average (TA-Lib: SMA-seeded, `k = 2/(period+1)`).
pub fn ema(close: &[f64], period: usize) -> Vec<f64> {
    kernels::ema_seeded(av(close), period).to_vec()
}

/// Smoothed moving average (Wilder's RMA: SMA-seeded, `alpha = 1/period`).
pub fn smma(close: &[f64], period: usize) -> Vec<f64> {
    kernels::wilder(av(close), period).to_vec()
}

/// Weighted moving average — linearly increasing weights `1..=period`, the newest
/// bar weighted heaviest (TA-Lib WMA). O(n) via a running sum + running weighted
/// sum. Lookback `period-1`.
pub fn wma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let denom = (period * (period + 1) / 2) as f64; // sum of weights 1..=period
    let pf = period as f64;
    // Seed the first full window directly.
    let mut sum = 0.0; // plain window sum
    let mut wsum = 0.0; // weighted sum: newest bar * period ... oldest * 1
    for j in 0..period {
        sum += data[j];
        wsum += data[j] * (j + 1) as f64;
    }
    out[period - 1] = wsum / denom;
    // Slide: dropping the oldest (weight 1) raises every retained weight by one.
    for i in period..n {
        wsum += pf * data[i] - sum;
        sum += data[i] - data[i - period];
        out[i] = wsum / denom;
    }
    out
}

/// Double EMA: `2*EMA - EMA(EMA)` (TA-Lib DEMA). Lookback `2*(period-1)`.
pub fn dema(data: &[f64], period: usize) -> Vec<f64> {
    let e1 = kernels::ema_seeded(av(data), period);
    let e2 = kernels::ema_seeded(e1.view(), period);
    (0..data.len()).map(|i| 2.0 * e1[i] - e2[i]).collect()
}

/// Triple EMA: `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))` (TA-Lib TEMA).
/// Lookback `3*(period-1)`.
pub fn tema(data: &[f64], period: usize) -> Vec<f64> {
    let e1 = kernels::ema_seeded(av(data), period);
    let e2 = kernels::ema_seeded(e1.view(), period);
    let e3 = kernels::ema_seeded(e2.view(), period);
    (0..data.len())
        .map(|i| 3.0 * e1[i] - 3.0 * e2[i] + e3[i])
        .collect()
}

/// Triangular moving average (TA-Lib TRIMA): a double SMA whose net weights rise to
/// the window centre then fall. Since convolution commutes, the two SMA windows are
/// `((n+1)/2, (n+1)/2)` for odd `n` and `(n/2, n/2+1)` for even `n` (either order).
/// Lookback `period-1`.
pub fn trima(data: &[f64], period: usize) -> Vec<f64> {
    let (a, b) = if period % 2 == 1 {
        let m = (period + 1) / 2;
        (m, m)
    } else {
        let m = period / 2;
        (m, m + 1)
    };
    let inner = kernels::sma(av(data), a);
    kernels::sma(inner.view(), b).to_vec()
}

/// Tillson T3 (TA-Lib T3): `c1·e6 + c2·e5 + c3·e4 + c4·e3` over six cascaded
/// SMA-seeded EMAs, with `vfactor`-derived coefficients `c1=-v³`, `c2=3(v²−c1)`,
/// `c3=-6v²−3(v−c1)`, `c4=1+3v−c1+3v²` (computed in TA-Lib's exact float order).
/// Default period 5, vfactor 0.7; lookback `6·(period-1)`.
pub fn t3(data: &[f64], period: usize, vfactor: f64) -> Vec<f64> {
    let e1 = kernels::ema_seeded(av(data), period);
    let e2 = kernels::ema_seeded(e1.view(), period);
    let e3 = kernels::ema_seeded(e2.view(), period);
    let e4 = kernels::ema_seeded(e3.view(), period);
    let e5 = kernels::ema_seeded(e4.view(), period);
    let e6 = kernels::ema_seeded(e5.view(), period);
    let v2 = vfactor * vfactor;
    let c1 = -(v2 * vfactor);
    let c2 = 3.0 * (v2 - c1);
    let c3 = -6.0 * v2 - 3.0 * (vfactor - c1);
    let c4 = 1.0 + 3.0 * vfactor - c1 + 3.0 * v2;
    (0..data.len())
        .map(|i| c1 * e6[i] + c2 * e5[i] + c3 * e4[i] + c4 * e3[i])
        .collect()
}

fn macd_line(close: &[f64], fast: usize, slow: usize) -> Array1<f64> {
    let data = av(close);
    // TA-Lib MACD line = fast EMA - slow EMA (SMA-seeded EMAs). Best practice: the
    // line is emitted from its natural start (the slow EMA's first valid row), not
    // delayed to the signal line's start as TA-Lib's aligned 3-output form does.
    let f = kernels::ema_seeded(data, fast);
    let s = kernels::ema_seeded(data, slow);
    &f - &s
}

/// MACD line (DIF).
pub fn macd(close: &[f64], fast: usize, slow: usize) -> Vec<f64> {
    macd_line(close, fast, slow).to_vec()
}

/// MACD signal line (DEA) — SMA-seeded EMA of the MACD line.
pub fn macd_signal(close: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let line = macd_line(close, fast, slow);
    kernels::ema_seeded(line.view(), signal).to_vec()
}

/// MACD histogram — TA-Lib convention `MACD - signal` (not the stock-pandas `2x`).
pub fn macd_histogram(close: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let line = macd_line(close, fast, slow);
    let sig = kernels::ema_seeded(line.view(), signal);
    (&line - &sig).to_vec()
}

/// Bull and Bear Index (`mean of ma:a, ma:b, ma:c, ma:d`).
pub fn bbi(close: &[f64], a: usize, b: usize, c: usize, d: usize) -> Vec<f64> {
    let data = av(close);
    let ma_a = kernels::sma(data, a);
    let ma_b = kernels::sma(data, b);
    let ma_c = kernels::sma(data, c);
    let ma_d = kernels::sma(data, d);
    ((&ma_a + &ma_b + &ma_c + &ma_d) / 4.0).to_vec()
}

/// True Range.
pub fn tr(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut tr = vec![f64::NAN; n];
    // TA-Lib TRANGE: index 0 has no prior close, so it has no TR (NaN); TR is
    // defined from index 1 onward.
    for i in 1..n {
        let prev_close = close[i - 1];
        let hl = high[i] - low[i];
        let hc = (high[i] - prev_close).abs();
        let lc = (low[i] - prev_close).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    tr
}

/// Average True Range — TA-Lib semantics: SMA-seeded Wilder smoothing of TR (the
/// first ATR, at index `period`, is the SMA of the first `period` TRs; thereafter
/// `ATR[i] = (ATR[i-1]*(period-1) + TR[i]) / period`).
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let tr = tr(high, low, close);
    kernels::wilder(av(&tr), period).to_vec()
}

/// Normalized Average True Range — `ATR/close · 100` (TA-Lib NATR), expressing ATR
/// as a percentage of price. Lookback = period (same as ATR).
pub fn natr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let atr = atr(high, low, close, period);
    (0..close.len())
        .map(|i| atr[i] / close[i] * 100.0)
        .collect()
}
