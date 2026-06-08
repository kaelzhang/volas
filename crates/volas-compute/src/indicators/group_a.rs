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

/// DPO Detrended Price Oscillator: `Price[(n/2 + 1) ago] - SMA_n`.
/// Source: StockCharts — Detrended Price Oscillator (displaced form).
pub fn dpo(close: &[f64], n: usize) -> Vec<f64> {
    let sma = kernels::sma(av(close), n);
    let shift = n / 2 + 1;
    (0..close.len())
        .map(|i| if i >= shift { close[i - shift] - sma[i] } else { f64::NAN })
        .collect()
}

/// CMF Chaikin Money Flow: `sum_n(MFV) / sum_n(volume)`, `MFV = ((C-L)-(H-C))/(H-L) * V`.
/// Source: StockCharts / Fidelity — Chaikin Money Flow.
pub fn cmf(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], n: usize) -> Vec<f64> {
    let mfv: Vec<f64> = (0..close.len())
        .map(|i| ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i]) * volume[i])
        .collect();
    let sum_mfv = super::sum(&mfv, n);
    let sum_vol = super::sum(volume, n);
    (0..close.len()).map(|i| sum_mfv[i] / sum_vol[i]).collect()
}

/// CHOP Choppiness Index: `100 * log10( sum_n(TR) / (HHV_n(high) - LLV_n(low)) ) / log10(n)`.
/// Source: TradingView — Choppiness Index.
pub fn chop(high: &[f64], low: &[f64], close: &[f64], n: usize) -> Vec<f64> {
    let tr = super::tr(high, low, close);
    // TR[0] is NaN; `sma` skips the leading-NaN prefix, so `sma * n` is the n-window TR sum.
    let sum_tr = kernels::sma(av(&tr), n) * n as f64;
    let hh = super::hhv(high, n);
    let ll = super::llv(low, n);
    let denom = (n as f64).log10();
    (0..close.len())
        .map(|i| 100.0 * (sum_tr[i] / (hh[i] - ll[i])).log10() / denom)
        .collect()
}

/// KST Know Sure Thing (Pring): `1*RCMA1 + 2*RCMA2 + 3*RCMA3 + 4*RCMA4`, where
/// `RCMA1=SMA10(ROC10)`, `RCMA2=SMA10(ROC15)`, `RCMA3=SMA10(ROC20)`, `RCMA4=SMA15(ROC30)`.
/// Source: StockCharts — Know Sure Thing.
pub fn kst(close: &[f64]) -> Vec<f64> {
    let roc = |p: usize| -> Vec<f64> {
        (0..close.len())
            .map(|i| if i >= p { (close[i] / close[i - p] - 1.0) * 100.0 } else { f64::NAN })
            .collect()
    };
    let (roc10, roc15, roc20, roc30) = (roc(10), roc(15), roc(20), roc(30));
    let r1 = kernels::sma(av(&roc10), 10);
    let r2 = kernels::sma(av(&roc15), 10);
    let r3 = kernels::sma(av(&roc20), 10);
    let r4 = kernels::sma(av(&roc30), 15);
    (0..close.len())
        .map(|i| r1[i] + 2.0 * r2[i] + 3.0 * r3[i] + 4.0 * r4[i])
        .collect()
}
