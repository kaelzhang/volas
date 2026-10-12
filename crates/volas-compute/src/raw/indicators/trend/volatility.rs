//! True-range volatility: TR / ATR / NATR and the ATR state-carry resume.

use crate::missing;
use crate::raw::indicators::av;
use crate::raw::kernels;

/// True Range (TA-Lib TRANGE): [`tr_at`] at every row; row 0 has no prior close, so
/// it has no TR (NaN).
pub fn tr(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut tr = vec![f64::NAN; n];
    for (i, slot) in tr.iter_mut().enumerate().skip(1) {
        *slot = tr_at(high, low, close, i);
    }
    tr
}

/// The true range of bar `i >= 1`: `max(high − low, |high − prev close|, |low − prev
/// close|)`. The one owner of the TR term — TR, ATR / NATR, CHOP, Vortex, ULTOSC and
/// the live resumes all read it here. It reads `high[i]`, `low[i]` and `close[i - 1]`,
/// and is missing when one of them is (see [`crate::missing::term`]): `f64::max` would
/// otherwise step over a missing value into an ordinary range.
#[inline]
pub fn tr_at(high: &[f64], low: &[f64], close: &[f64], i: usize) -> f64 {
    let (h, l, prev_close) = (high[i], low[i], close[i - 1]);
    let range = (h - l).max((h - prev_close).abs()).max((l - prev_close).abs());
    missing::term(&[h, l, prev_close], range)
}

/// Average True Range — TA-Lib semantics: SMA-seeded Wilder smoothing of TR (the
/// first ATR, at index `period`, is the SMA of the first `period` TRs; thereafter
/// `ATR[i] = (ATR[i-1]*(period-1) + TR[i]) / period`). Kept out of line: inlined into
/// the directive's dispatch arm it compiles to a slower loop.
#[inline(never)]
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

/// Final ATR state `[atr]` after a full [`atr`] compute, or `None` if ATR never seeds:
/// the Wilder smoothing of the true ranges captured by [`kernels::sma_seeded_last`] —
/// the full kernel's own seed rule (the first `period` present TRs; `tr[0]` and a true
/// range of infinite prices are not present) and its own step.
pub fn atr_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Option<Vec<f64>> {
    let tr = tr(high, low, close);
    Some(vec![kernels::sma_seeded_last(&tr, period, kernels::wilder_step(period))?])
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
        atr = atr.mul_add(a, tr_at(high, low, close, i) * b);
        out.push(atr);
    }
    Some((out, vec![atr]))
}

/// Scalar single-row twin of [`atr_resume`]: the Wilder ATR at `row` continued from
/// `state = [atr_{row-1}]`, with the state as of `row` (`[atr_row]`) written into `next`
/// — no allocation. Used by the single-bar append / live forming-row refresh.
/// Bit-identical to `atr_resume` (the same fused step). `None` at `row == 0`, empty
/// `state`, `next` not one value, or `row` out of range.
pub fn atr_resume_one(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row == 0 || state.is_empty() || row >= high.len() {
        return None;
    }
    let [slot] = next else { return None };
    let pf = period as f64;
    let (a, b) = ((pf - 1.0) / pf, 1.0 / pf);
    *slot = state[0].mul_add(a, tr_at(high, low, close, row) * b);
    Some(*slot)
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
