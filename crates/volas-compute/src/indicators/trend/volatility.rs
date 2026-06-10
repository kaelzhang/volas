//! True-range volatility: TR / ATR / NATR and the ATR state-carry resume.

use crate::indicators::av;
use crate::kernels;

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

// --- ATR / NATR state-carry (additive; the full-recompute fallback stays correct) ---
//
// ATR is the SMA-seeded Wilder *average* of True Range, and NATR rescales it by price.
// The carried state is the running ATR `[atr_{from-1}]`. `tr` at bar `i` reads the prior
// close, so a resume reads only `high/low/close[from-1..]`; the steady-state step is the
// exact fused `prev·a + tr·b` of `kernels::wilder`. A resume at `from == 0` (no prior bar)
// returns `None` and falls back. `*_final_state` returns `None` before ATR seeds (the
// column is all-NaN), so the caller keeps the correct fallback.

/// One-period true range at bar `i` (needs the prior close) — the per-bar input the ATR
/// Wilder average smooths. Matches `tr`'s `index >= 1` definition.
#[inline]
fn tr1(high: &[f64], low: &[f64], close: &[f64], i: usize) -> f64 {
    let hl = high[i] - low[i];
    let hc = (high[i] - close[i - 1]).abs();
    let lc = (low[i] - close[i - 1]).abs();
    hl.max(hc).max(lc)
}

/// Final ATR state `[atr]` after a full [`atr`] compute, or `None` if ATR never seeds
/// (`period == 0 || period >= n`, since `tr[0]` is NaN so the SMA of the first `period`
/// TRs seeds at index `period`). Reproduces `kernels::wilder(tr)` exactly: SMA-seed the
/// first `period` TRs (indices `1..=period`), then the fused Wilder step.
pub fn atr_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Option<Vec<f64>> {
    let n = high.len();
    if period == 0 || period >= n {
        return None;
    }
    let pf = period as f64;
    let (a, b) = ((pf - 1.0) / pf, 1.0 / pf);
    // Seed = SMA of tr[1..=period] (tr[0] is undefined / NaN), placed at index `period`.
    let mut atr = 0.0;
    for i in 1..=period {
        atr += tr1(high, low, close, i);
    }
    atr /= pf;
    for i in (period + 1)..n {
        atr = atr.mul_add(a, tr1(high, low, close, i) * b);
    }
    Some(vec![atr])
}

/// Resume [`atr`] from `state = [atr_{from-1}]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `high/low/close[from-1..]`.
pub fn atr_resume(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from == 0 {
        return None;
    }
    let n = high.len();
    let pf = period as f64;
    let (a, b) = ((pf - 1.0) / pf, 1.0 / pf);
    let mut atr = state[0];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        atr = atr.mul_add(a, tr1(high, low, close, i) * b);
        out.push(atr);
    }
    Some((out, vec![atr]))
}

/// Resume [`natr`] from `state = [atr_{from-1}]` over rows `[from, n)`: the ATR resume,
/// rescaled per row by `atr/close·100`. `None` at `from == 0`. Reads only
/// `high/low/close[from-1..]`.
pub fn natr_resume(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let (atr_tail, new_state) = atr_resume(high, low, close, period, from, state)?;
    let vals = (from..high.len())
        .map(|i| atr_tail[i - from] / close[i] * 100.0)
        .collect();
    Some((vals, new_state))
}
