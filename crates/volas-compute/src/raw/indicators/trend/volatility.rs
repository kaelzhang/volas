//! True-range volatility: TR / ATR / NATR and the ATR state-carry resume.

use crate::missing;
use crate::raw::smoothing;

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
/// and is missing when one of them is (see [`crate::missing::term`]). Otherwise follow
/// `ta_TRANGE.c`: start with high − low, then replace it only when either distance
/// from the prior close is greater. Unlike `f64::max`, this preserves an arithmetic
/// NaN from `inf − inf`, and the initial range's signed zero on a tie.
#[inline]
pub fn tr_at(high: &[f64], low: &[f64], close: &[f64], i: usize) -> f64 {
    let (h, l, prev_close) = (high[i], low[i], close[i - 1]);
    let mut range = h - l;
    let high_gap = (prev_close - h).abs();
    if high_gap > range {
        range = high_gap;
    }
    let low_gap = (prev_close - l).abs();
    if low_gap > range {
        range = low_gap;
    }
    missing::term(&[h, l, prev_close], range)
}

/// ATR's fixed seed and recurrence, shared by the full output and state capture.
/// Input missing rows are removed by the public wrapper; a NaN *computed* by TR is
/// part of the seed, not a missing price bar. Period 1 is TR itself (`ta_ATR.c`).
fn atr_fold(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mut row: impl FnMut(usize, f64),
) -> Option<f64> {
    let n = high.len();
    if period == 0 || period >= n {
        return None;
    }
    if period == 1 {
        let mut value = f64::NAN;
        for i in 1..n {
            value = tr_at(high, low, close, i);
            row(i, value);
        }
        return Some(value);
    }
    let mut value = 0.0;
    for i in 1..=period {
        value += tr_at(high, low, close, i);
    }
    value /= period as f64;
    row(period, value);
    let average = smoothing::Wilder::new(period);
    for i in period + 1..n {
        value = average.step(value, tr_at(high, low, close, i));
        row(i, value);
    }
    Some(value)
}

/// Average True Range — TA-Lib semantics: SMA-seeded Wilder smoothing of TR (the
/// first ATR, at index `period`, is the SMA of the first `period` TRs; thereafter
/// `ATR[i] = (ATR[i-1]*(period-1) + TR[i]) / period`). Kept out of line: inlined into
/// the directive's dispatch arm it compiles to a slower loop.
#[inline(never)]
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; high.len()];
    atr_fold(high, low, close, period, |i, value| out[i] = value);
    out
}

/// Normalized Average True Range — `ATR/close · 100` (TA-Lib NATR), expressing ATR
/// as a percentage of price. Lookback = period (same as ATR).
pub fn natr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let atr = atr(high, low, close, period);
    (0..close.len()).map(|i| natr_value(atr[i], close[i])).collect()
}

/// One row's NATR from its ATR and close: `ATR/close · 100`, and 0 where the close is
/// exactly zero (TA-Lib's guard, applied at the current row; TA-Lib's `TA_IS_ZERO` is an
/// absolute epsilon, volas tests exact zero as for PPO). An ATR still warming up stays
/// NaN. The one owner of the rule for the full computation and every resume.
#[inline]
pub fn natr_value(atr: f64, close: f64) -> f64 {
    if close == 0.0 && !atr.is_nan() {
        0.0
    } else {
        atr / close * 100.0
    }
}

// --- ATR / NATR state-carry (additive; the full-recompute fallback stays correct) ---
//
// ATR is the SMA-seeded Wilder *average* of True Range, and NATR rescales it by price.
// The carried state is the running ATR `[atr_{from-1}]`. `tr` at bar `i` reads the prior
// close, so a resume reads only `high/low/close[from-1..]`; the steady-state step is the
// exact Wilder step of [`smoothing::Wilder`]. A resume at `from == 0` (no prior bar)
// returns `None` and falls back. `*_final_state` returns `None` before ATR seeds (the
// column is all-NaN), so the caller keeps the correct fallback.

/// Final ATR state `[atr]` after a full [`atr`] compute, or `None` if ATR never seeds:
/// the full kernel's own fixed seed and recurrence ([`atr_fold`]). An arithmetic NaN
/// is still a seeded state, and period 1 carries the last TR even when it is NaN.
pub fn atr_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Option<Vec<f64>> {
    Some(vec![atr_fold(high, low, close, period, |_, _| {})?])
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
    let wilder = smoothing::Wilder::new(period);
    let mut atr = state[0];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        atr = wilder.step(atr, tr_at(high, low, close, i));
        out.push(atr);
    }
    Some((out, vec![atr]))
}

/// Scalar single-row twin of [`atr_resume`]: the Wilder ATR at `row` continued from
/// `state = [atr_{row-1}]`, with the state as of `row` (`[atr_row]`) written into `next`
/// — no allocation. Used by the single-bar append / live forming-row refresh.
/// Bit-identical to `atr_resume` (the same step). `None` at `row == 0`, empty
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
    *slot = smoothing::Wilder::new(period).step(state[0], tr_at(high, low, close, row));
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
        .map(|i| natr_value(atr_tail[i - from], close[i]))
        .collect();
    Some((vals, new_state))
}
