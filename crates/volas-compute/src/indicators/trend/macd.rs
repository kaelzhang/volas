//! MACD line / signal / histogram and their state-carry resume.

use ndarray::Array1;

use crate::indicators::av;
use crate::indicators::trend::ma::{ema_k, ema_seed_idx};
use crate::kernels;

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

/// The fast/slow EMA pair `(pf, ps)` as of the last row after a full MACD-line compute,
/// or `None` if the slow EMA never seeds (the line is all-NaN → keep the fallback).
/// Mirrors `kernels::ema_diff_seeded`: each EMA SMA-seeds at its own period-th finite
/// value, then advances with the fused `(x-prev)·k+prev` step. Requires `fast <= slow`.
fn macd_emas_final(close: &[f64], fast: usize, slow: usize) -> Option<(f64, f64)> {
    let (kf, ks) = (ema_k(fast), ema_k(slow));
    let sf = ema_seed_idx(close, fast)?;
    let ss = ema_seed_idx(close, slow)?;
    let mut pf = close[sf + 1 - fast..=sf].iter().sum::<f64>() / fast as f64;
    for &x in &close[sf + 1..] {
        pf = (x - pf).mul_add(kf, pf);
    }
    let mut ps = close[ss + 1 - slow..=ss].iter().sum::<f64>() / slow as f64;
    for &x in &close[ss + 1..] {
        ps = (x - ps).mul_add(ks, ps);
    }
    Some((pf, ps))
}

/// Final MACD-line state `[pf, ps]`, or `None` if unseeded. Pairs with [`macd_resume`].
pub fn macd_final_state(close: &[f64], fast: usize, slow: usize) -> Option<Vec<f64>> {
    let (pf, ps) = macd_emas_final(close, fast, slow)?;
    Some(vec![pf, ps])
}

/// Resume the MACD line from `state = [pf, ps]` over rows `[from, n)`, bit-identical to a
/// full recompute (`fast EMA − slow EMA`, same fused step). Reads only `close[from..]`.
pub fn macd_resume(
    close: &[f64],
    fast: usize,
    slow: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (kf, ks) = (ema_k(fast), ema_k(slow));
    let n = close.len();
    let (mut pf, mut ps) = (state[0], state[1]);
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &close[from..n] {
        pf = (x - pf).mul_add(kf, pf);
        ps = (x - ps).mul_add(ks, ps);
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
    // The line over full history (its NaN warm-up is what the signal SMA-seeds past).
    let line = macd_line(close, fast, slow);
    let line = line.as_slice().expect("macd line is contiguous");
    let (pf, ps) = macd_emas_final(close, fast, slow)?;
    let ksig = ema_k(signal);
    let si = ema_seed_idx(line, signal)?;
    let mut sig = line[si + 1 - signal..=si].iter().sum::<f64>() / signal as f64;
    for &x in &line[si + 1..] {
        sig = (x - sig).mul_add(ksig, sig);
    }
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
    let (kf, ks, ksig) = (ema_k(fast), ema_k(slow), ema_k(signal));
    let n = close.len();
    let (mut pf, mut ps, mut sig) = (state[0], state[1], state[2]);
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &close[from..n] {
        pf = (x - pf).mul_add(kf, pf);
        ps = (x - ps).mul_add(ks, ps);
        let line = pf - ps;
        sig = (line - sig).mul_add(ksig, sig);
        out.push(if histogram { line - sig } else { sig });
    }
    (out, vec![pf, ps, sig])
}


/// Scalar single-row twin of [`macd_resume`]: the MACD line at `row` from `state =
/// [pf, ps]` (fast/slow EMA as of `row-1`), zero-alloc, bit-identical to one Vec-kernel
/// iteration. Reads only `close[row]`.
pub fn macd_resume_one(
    close: &[f64],
    fast: usize,
    slow: usize,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || state.len() < 2 || row >= close.len() {
        return None;
    }
    let (kf, ks) = (ema_k(fast), ema_k(slow));
    let (pf, ps) = (state[0], state[1]);
    let x = close[row];
    let pf = (x - pf).mul_add(kf, pf);
    let ps = (x - ps).mul_add(ks, ps);
    Some(pf - ps)
}

/// Scalar single-row twin of [`macd_signal_resume`]: the MACD signal (`histogram ==
/// false`) or histogram (`true`) at `row` from `state = [pf, ps, sig]` (as of `row-1`),
/// zero-alloc, bit-identical to one Vec-kernel iteration. Reads only `close[row]`.
pub fn macd_signal_resume_one(
    close: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    histogram: bool,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || state.len() < 3 || row >= close.len() {
        return None;
    }
    let (kf, ks, ksig) = (ema_k(fast), ema_k(slow), ema_k(signal));
    let (pf, ps, sig) = (state[0], state[1], state[2]);
    let x = close[row];
    let pf = (x - pf).mul_add(kf, pf);
    let ps = (x - ps).mul_add(ks, ps);
    let line = pf - ps;
    let sig = (line - sig).mul_add(ksig, sig);
    Some(if histogram { line - sig } else { sig })
}

