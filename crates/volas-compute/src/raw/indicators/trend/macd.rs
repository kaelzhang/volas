//! MACD line / signal / histogram and their state-carry resume.

use ndarray::Array1;

use crate::raw::indicators::av;
use crate::raw::smoothing::{self, ema_k};
use crate::raw::kernels;

/// TA-Lib's price-oscillator rule (`APO`, `PPO`, `MACD`, `MACDEXT`): the slow average is
/// the one with the longer period — when `slow < fast` the two averages, each with what
/// it carries (`T`: its MA type, or nothing), are swapped, so the line is always the
/// shorter average minus the longer one. The one owner of that rule for the MACD kernels
/// here and the directive's generic-MA oscillators.
pub fn oscillator_order<T>(fast: (usize, T), slow: (usize, T)) -> ((usize, T), (usize, T)) {
    if slow.0 < fast.0 { (slow, fast) } else { (fast, slow) }
}

/// [`oscillator_order`] of two bare periods.
fn oscillator_periods(fast: usize, slow: usize) -> (usize, usize) {
    let ((fast, ()), (slow, ())) = oscillator_order((fast, ()), (slow, ()));
    (fast, slow)
}

fn macd_line(close: &[f64], fast: usize, slow: usize) -> Array1<f64> {
    // TA-Lib MACD line = fast EMA - slow EMA (SMA-seeded EMAs). Best practice: the
    // line is emitted from its natural start (the slow EMA's first valid row), not
    // delayed to the signal line's start as TA-Lib's aligned 3-output form does.
    // `ema_diff_seeded` fuses both EMAs into one interleaved pass (ILP over the two
    // independent recurrences) and emits the difference directly.
    kernels::ema_diff_seeded(av(close), fast, slow)
}

/// MACD line (DIF).
pub fn macd(close: &[f64], fast: usize, slow: usize) -> Vec<f64> {
    let (fast, slow) = oscillator_periods(fast, slow);
    macd_line(close, fast, slow).to_vec()
}

/// MACD signal line (DEA) — SMA-seeded EMA of the MACD line.
pub fn macd_signal(close: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let (fast, slow) = oscillator_periods(fast, slow);
    let line = macd_line(close, fast, slow);
    kernels::ema_seeded(line.view(), signal).to_vec()
}

/// MACD histogram — TA-Lib convention `MACD - signal` (not the stock-pandas `2x`).
pub fn macd_histogram(close: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let (fast, slow) = oscillator_periods(fast, slow);
    let line = macd_line(close, fast, slow);
    let sig = kernels::ema_seeded(line.view(), signal);
    (&line - &sig).to_vec()
}

/// The fast/slow EMA pair `(pf, ps)` as of the last row after a full MACD-line compute,
/// or `None` if the slow EMA never seeds (the line is all-NaN → keep the fallback).
/// Mirrors `kernels::ema_diff_seeded`: each EMA SMA-seeds at its own period-th finite
/// value, then advances with the EMA step ([`smoothing::ema`]). Callers pass `fast <= slow`.
fn macd_emas_final(close: &[f64], fast: usize, slow: usize) -> Option<(f64, f64)> {
    let pf = kernels::sma_seeded_last(close, fast, kernels::ema_step(fast))?;
    let ps = kernels::sma_seeded_last(close, slow, kernels::ema_step(slow))?;
    Some((pf, ps))
}

/// Final MACD-line state `[pf, ps]`, or `None` if unseeded. Pairs with [`macd_resume`].
pub fn macd_final_state(close: &[f64], fast: usize, slow: usize) -> Option<Vec<f64>> {
    let (fast, slow) = oscillator_periods(fast, slow);
    let (pf, ps) = macd_emas_final(close, fast, slow)?;
    Some(vec![pf, ps])
}

/// Resume the MACD line from `state = [pf, ps]` over rows `[from, n)`, bit-identical to a
/// full recompute (`fast EMA − slow EMA`, same step). Reads only `close[from..]`.
pub fn macd_resume(
    close: &[f64],
    fast: usize,
    slow: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (fast, slow) = oscillator_periods(fast, slow);
    let (kf, ks) = (ema_k(fast), ema_k(slow));
    let n = close.len();
    let (mut pf, mut ps) = (state[0], state[1]);
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &close[from..n] {
        pf = smoothing::ema(pf, x, kf);
        ps = smoothing::ema(ps, x, ks);
        out.push(pf - ps);
    }
    (out, vec![pf, ps])
}

/// Final MACD signal/histogram state `[pf, ps, sig]`: the line's fast/slow EMAs plus the
/// signal EMA (an SMA-seeded EMA of the line), all as of the last row. `None` if the
/// signal never seeds. Shared by macd.signal and macd.histogram (their per-row outputs
/// differ — `sig` vs `line − sig` — but the carried recursion is identical).
pub fn macd_signal_final_state(
    close: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> Option<Vec<f64>> {
    let (fast, slow) = oscillator_periods(fast, slow);
    // The line over full history (its NaN warm-up is what the signal SMA-seeds past).
    let line = macd_line(close, fast, slow);
    let line = line.as_slice().expect("macd line is contiguous");
    let (pf, ps) = macd_emas_final(close, fast, slow)?;
    let sig = kernels::sma_seeded_last(line, signal, kernels::ema_step(signal))?;
    Some(vec![pf, ps, sig])
}

/// Resume MACD signal/histogram from `state = [pf, ps, sig]` over rows `[from, n)`. The
/// `histogram` flag selects the per-row output (`line − sig` vs `sig`); both advance the
/// same fast/slow/signal recursion, bit-identical to the full recompute. Reads only
/// `close[from..]`.
pub fn macd_signal_resume(
    close: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    histogram: bool,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (fast, slow) = oscillator_periods(fast, slow);
    let (kf, ks, ksig) = (ema_k(fast), ema_k(slow), ema_k(signal));
    let n = close.len();
    let (mut pf, mut ps, mut sig) = (state[0], state[1], state[2]);
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &close[from..n] {
        pf = smoothing::ema(pf, x, kf);
        ps = smoothing::ema(ps, x, ks);
        let line = pf - ps;
        sig = smoothing::ema(sig, line, ksig);
        out.push(if histogram { line - sig } else { sig });
    }
    (out, vec![pf, ps, sig])
}


/// Scalar single-row twin of [`macd_resume`]: the MACD line at `row` from `state =
/// [pf, ps]` (fast/slow EMA as of `row-1`), the state as of `row` written into `next`,
/// zero-alloc, bit-identical to one Vec-kernel iteration. Reads only `close[row]`.
pub fn macd_resume_one(
    close: &[f64],
    fast: usize,
    slow: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let (fast, slow) = oscillator_periods(fast, slow);
    if row == 0 || state.len() != 2 || next.len() != 2 || row >= close.len() {
        return None;
    }
    let (kf, ks) = (ema_k(fast), ema_k(slow));
    let (pf, ps) = (state[0], state[1]);
    let x = close[row];
    let pf = smoothing::ema(pf, x, kf);
    let ps = smoothing::ema(ps, x, ks);
    next.copy_from_slice(&[pf, ps]);
    Some(pf - ps)
}

/// Scalar single-row twin of [`macd_signal_resume`]: the MACD signal (`histogram ==
/// false`) or histogram (`true`) at `row` from `state = [pf, ps, sig]` (as of `row-1`),
/// the state as of `row` written into `next`, zero-alloc, bit-identical to one Vec-kernel
/// iteration. Reads only `close[row]`.
#[allow(clippy::too_many_arguments)]
pub fn macd_signal_resume_one(
    close: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    histogram: bool,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let (fast, slow) = oscillator_periods(fast, slow);
    if row == 0 || state.len() != 3 || next.len() != 3 || row >= close.len() {
        return None;
    }
    let (kf, ks, ksig) = (ema_k(fast), ema_k(slow), ema_k(signal));
    let (pf, ps, sig) = (state[0], state[1], state[2]);
    let x = close[row];
    let pf = smoothing::ema(pf, x, kf);
    let ps = smoothing::ema(ps, x, ks);
    let line = pf - ps;
    let sig = smoothing::ema(sig, line, ksig);
    next.copy_from_slice(&[pf, ps, sig]);
    Some(if histogram { line - sig } else { sig })
}

