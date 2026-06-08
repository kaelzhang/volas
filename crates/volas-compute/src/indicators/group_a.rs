//! Group A non-TA-Lib indicators (gap report 2026-06-07).
//!
//! Pure O(n) kernels, verified against the source-pinned reference oracle
//! (`test/oracle_reference.py`). The cumulative members (pvt / nvi / pvi) carry no
//! state-carry kernel yet: with a lookback of 0 the directive engine refreshes them on
//! `append` through its exact full-recompute fallback.

use crate::indicators::av;
use crate::kernels;

/// PSY (心理线): `100 * mean(up, n)`, where `up[i] = 1` when `close[i] > close[i-1]` (the
/// first bar is not rising). NaN until `n` bars are available.
/// Source: Eastmoney 心理线.
pub fn psy(close: &[f64], n: usize) -> Vec<f64> {
    let up: Vec<f64> = (0..close.len())
        .map(|i| f64::from(i > 0 && close[i] > close[i - 1]))
        .collect();
    (kernels::sma(av(&up), n) * 100.0).to_vec()
}

/// PVT Price Volume Trend (cumulative): `PVT_i = PVT_{i-1} + (C_i - C_{i-1}) / C_{i-1} * V_i`,
/// with `PVT_0 = 0`.
/// Source: StockCharts / Investopedia — Price Volume Trend.
pub fn pvt(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; close.len()];
    let mut acc = 0.0;
    for i in 1..close.len() {
        acc += (close[i] - close[i - 1]) / close[i - 1] * volume[i];
        out[i] = acc;
    }
    out
}

/// NVI Negative Volume Index (base 1000): on a down-volume bar `*= 1 + ROC`, otherwise held.
/// Source: StockCharts — Negative Volume Index.
pub fn nvi(close: &[f64], volume: &[f64]) -> Vec<f64> {
    volume_index(close, volume, true)
}

/// PVI Positive Volume Index (base 1000): on an up-volume bar `*= 1 + ROC`, otherwise held.
/// Source: StockCharts — Positive Volume Index.
pub fn pvi(close: &[f64], volume: &[f64]) -> Vec<f64> {
    volume_index(close, volume, false)
}

/// Shared NVI / PVI engine. `on_down` selects the negative (volume-falling) variant.
fn volume_index(close: &[f64], volume: &[f64], on_down: bool) -> Vec<f64> {
    const BASE: f64 = 1000.0;
    let mut out = vec![BASE; close.len()];
    for i in 1..close.len() {
        let triggered = if on_down {
            volume[i] < volume[i - 1]
        } else {
            volume[i] > volume[i - 1]
        };
        out[i] = if triggered {
            out[i - 1] * (1.0 + (close[i] - close[i - 1]) / close[i - 1])
        } else {
            out[i - 1]
        };
    }
    out
}
