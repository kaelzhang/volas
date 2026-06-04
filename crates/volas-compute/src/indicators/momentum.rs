use super::av;
use crate::kernels;

// ---------------------------------------------------------------------------
// Momentum — change relative to the price `period` bars earlier
// ---------------------------------------------------------------------------

/// Momentum: `data[i] - data[i-period]` (TA-Lib MOM). NaN during warm-up.
pub fn mom(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        out[i] = data[i] - data[i - period];
    }
    out
}

/// Shared shape for the rate-of-change ratios (ROC/ROCP/ROCR/ROCR100): relate
/// each row to the price `period` bars earlier via `f(current, prior)`, NaN
/// during warm-up. A prior price of exactly zero yields `0.0`, matching TA-Lib's
/// divide-by-zero guard (purely theoretical for a positive price series).
fn roc_ratio(data: &[f64], period: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        let prior = data[i - period];
        out[i] = if prior == 0.0 { 0.0 } else { f(data[i], prior) };
    }
    out
}

/// Rate of change: `100 * (data/data[period ago] - 1)` (TA-Lib ROC).
pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| (cur / prior - 1.0) * 100.0)
}

/// Rate of change percentage: `data/data[period ago] - 1` (TA-Lib ROCP).
pub fn rocp(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| cur / prior - 1.0)
}

/// Rate of change ratio: `data/data[period ago]` (TA-Lib ROCR).
pub fn rocr(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| cur / prior)
}

/// Rate of change ratio ×100: `100 * data/data[period ago]` (TA-Lib ROCR100).
pub fn rocr100(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| cur / prior * 100.0)
}

/// Williams %R: `-100 * (HH - close) / (HH - LL)` over `period`, where HH/LL are
/// the highest high / lowest low (TA-Lib WILLR). A flat range (HH == LL) yields 0.
/// Lookback `period-1`. The operation order mirrors TA-Lib bit-for-bit.
pub fn willr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let hh = kernels::rolling_max(av(high), period);
    let ll = kernels::rolling_min(av(low), period);
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let diff = (hh[i] - ll[i]) / -100.0; // NaN during warm-up -> stays NaN below
        if diff != 0.0 {
            out[i] = (hh[i] - close[i]) / diff;
        } else if !hh[i].is_nan() {
            out[i] = 0.0; // finite flat range
        }
    }
    out
}

/// Balance of Power: `(close − open)/(high − low)` (TA-Lib BOP). A bar with no
/// range (`high − low < ε`) yields 0. Lookback 0.
pub fn bop(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len())
        .map(|i| {
            let range = high[i] - low[i];
            if range < 1e-14 {
                0.0
            } else {
                (close[i] - open[i]) / range
            }
        })
        .collect()
}
