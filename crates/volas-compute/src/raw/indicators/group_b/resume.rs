//! Resumable forms of the group-B finite-window indicators: the running sums of each
//! full computation, in its exact operation order, so a resume of one row at a time is
//! bit-identical to recomputing the column. The per-bar terms are shared with the full
//! computations in the parent module.

use crate::guard::{ring_push, state_after, Slide};
use crate::missing;
use crate::raw::indicators::{wma_initial, wma_row, FiniteMa, WMA_STATE};
use crate::raw::kernels::{self, SMA_STATE};

/// The `p`-bar rate of change of `close` at `i` (NaN before `p` bars exist).
#[inline]
pub(super) fn roc_at(close: &[f64], p: usize, i: usize) -> f64 {
    if i >= p { (close[i] / close[i - p] - 1.0) * 100.0 } else { f64::NAN }
}

/// The 4-bar symmetric weighted average of `x` at `i`, weights `[1, 2, 2, 1] / 6`
/// (NaN until 3 prior bars exist).
#[inline]
pub(super) fn swma4_at(x: impl Fn(usize) -> f64, i: usize) -> f64 {
    if i >= 3 { (x(i) + 2.0 * x(i - 1) + 2.0 * x(i - 2) + x(i - 3)) / 6.0 } else { f64::NAN }
}

/// DKX's midpoint `(3·C + L + O + H) / 6` at `i`.
#[inline]
pub(super) fn dkx_mid(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> f64 {
    (3.0 * close[i] + low[i] + open[i] + high[i]) / 6.0
}

/// WVAD's per-bar term `(C − O) / (H − L) · V` at `i`: a range flow like AD's
/// ([`super::super::volume::range_flow`]), so a bar without range carries no flow.
#[inline]
pub(super) fn wvad_term(open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64], i: usize) -> f64 {
    super::super::volume::range_flow(close[i] - open[i], high[i], low[i], volume[i])
}

/// VR's per-bar `(up, down, flat)` volumes at `i` (NaN at the first bar). They read the
/// close, the prior close and the volume, and are missing when one of them is (the
/// per-row term rule, [`missing::terms`]): a comparison with a missing close would read
/// as a flat day.
#[inline]
pub(super) fn vr_volumes(close: &[f64], volume: &[f64], i: usize) -> [f64; 3] {
    if i == 0 {
        return [f64::NAN; 3];
    }
    let (c, prev, v) = (close[i], close[i - 1], volume[i]);
    let split = if c > prev {
        [v, 0.0, 0.0]
    } else if c < prev {
        [0.0, v, 0.0]
    } else {
        [0.0, 0.0, v]
    };
    missing::terms(&[c, prev, v], split)
}

/// BRAR BR's per-bar `[max(0, H − Cᵧ), max(0, Cᵧ − L)]` at `i` (`Cᵧ` the prior close; NaN
/// at the first bar). Each is missing when an input it reads is (the per-row term rule,
/// [`missing::term`]): the `max(0, …)` clamp would turn a missing value into `0`.
#[inline]
pub(super) fn br_terms(high: &[f64], low: &[f64], close: &[f64], i: usize) -> [f64; 2] {
    if i == 0 {
        return [f64::NAN; 2];
    }
    let (h, l, cy) = (high[i], low[i], close[i - 1]);
    [missing::term(&[h, cy], (h - cy).max(0.0)), missing::term(&[cy, l], (cy - l).max(0.0))]
}

/// One row of the `n`-bar rolling sum (`SMA · n`) of `term`, from `state` into `next`.
#[inline]
fn rolling_sum_row(n: usize, row: usize, term: impl Fn(usize) -> f64, state: &[f64], next: &mut [f64]) -> Option<f64> {
    Some(kernels::sma_step(n, row, term, state, next)? * n as f64)
}

/// [`super::coppock`]'s state after the rows of `close`: its WMA's.
pub fn coppock_state(
    close: &[f64],
    w: usize,
    roc_long: usize,
    roc_short: usize,
    origin: usize,
) -> Option<Vec<f64>> {
    state_after(wma_initial(origin), close.len(), |row, prev, next| {
        coppock_step(close, w, roc_long, roc_short, row, prev, next)
    })
}

/// One row of [`super::coppock`] at `row`, from `state` into `next`.
pub fn coppock_step(
    close: &[f64],
    w: usize,
    roc_long: usize,
    roc_short: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row >= close.len() || state.len() != WMA_STATE || next.len() != WMA_STATE {
        return None;
    }
    let sum = |j: usize| roc_at(close, roc_long, j) + roc_at(close, roc_short, j);
    wma_row(w, row, &sum, state, next)
}

/// The length of [`super::relative_vigor`]'s state (its two SMAs), plus with `signal`
/// the last three RVI values its smoother reads.
fn rvi_state_len(signal: bool) -> usize {
    2 * SMA_STATE + if signal { 3 } else { 0 }
}

/// [`super::relative_vigor`] (or its `signal`)'s state after the rows.
pub fn relative_vigor_state(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    n: usize,
    signal: bool,
    origin: usize,
) -> Option<Vec<f64>> {
    let mut init = Slide::<1>::initial(origin).repeat(2);
    init.resize(rvi_state_len(signal), 0.0);
    state_after(init, close.len(), |row, prev, next| {
        relative_vigor_step(open, high, low, close, n, signal, row, prev, next)
    })
}

/// One row of [`super::relative_vigor`] (or its `signal`) at `row`, from `state` into
/// `next`.
#[allow(clippy::too_many_arguments)]
pub fn relative_vigor_step(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    n: usize,
    signal: bool,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    const S: usize = SMA_STATE;
    if row >= close.len() || state.len() != rvi_state_len(signal) || next.len() != state.len() {
        return None;
    }
    let co = |j: usize| close[j] - open[j];
    let hl = |j: usize| high[j] - low[j];
    let (n_num, rest) = next.split_at_mut(S);
    let (n_den, n_ring) = rest.split_at_mut(S);
    let num = kernels::sma_step(n, row, |j| swma4_at(co, j), &state[..S], n_num)?;
    let den = kernels::sma_step(n, row, |j| swma4_at(hl, j), &state[S..2 * S], n_den)?;
    let rvi = num / den;
    if !signal {
        return Some(rvi);
    }
    // the RVI at row j: `ring` holds rows `row-3 .. row-1`, oldest first
    let ring = &state[2 * S..];
    let at = |j: usize| if j == row { rvi } else { ring[3 - (row - j)] };
    let value = swma4_at(at, row);
    ring_push(ring, rvi, n_ring);
    Some(value)
}

/// The length of [`super::dkx`]'s state (its WMA), plus with `ma` (the period of
/// [`super::dkx_ma`]) its SMA and the last `ma` DKX values that SMA reads.
fn dkx_state_len(ma: Option<usize>) -> usize {
    WMA_STATE + ma.map_or(0, |m| SMA_STATE + m)
}

/// [`super::dkx`] (or with `ma`, [`super::dkx_ma`])'s state after the rows.
pub fn dkx_state(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    ma: Option<usize>,
    origin: usize,
) -> Option<Vec<f64>> {
    let mut init = wma_initial(origin);
    if let Some(m) = ma {
        init.extend(FiniteMa::new(m, 0)?.chained_initial(origin));
    }
    state_after(init, close.len(), |row, prev, next| {
        dkx_step(open, high, low, close, ma, row, prev, next)
    })
}

/// One row of [`super::dkx`] (or with `ma`, [`super::dkx_ma`]) at `row`, from `state`
/// into `next`.
#[allow(clippy::too_many_arguments)]
pub fn dkx_step(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    ma: Option<usize>,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row >= close.len() || state.len() != dkx_state_len(ma) || next.len() != state.len() {
        return None;
    }
    let mid = |j: usize| dkx_mid(open, high, low, close, j);
    let (n_wma, n_rest) = next.split_at_mut(WMA_STATE);
    let dkx = wma_row(20, row, &mid, &state[..WMA_STATE], n_wma)?;
    let Some(m) = ma else { return Some(dkx) };
    let value = FiniteMa::new(m, 0)?.step_chained(row, dkx, &state[WMA_STATE..], n_rest)?;
    Some(value)
}

/// [`super::wvad`]'s state after the rows: its rolling sum's.
pub fn wvad_state(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    n: usize,
    origin: usize,
) -> Option<Vec<f64>> {
    state_after(Slide::<1>::initial(origin), close.len(), |row, prev, next| {
        wvad_step(open, high, low, close, volume, n, row, prev, next)
    })
}

/// One row of [`super::wvad`] at `row`, from `state` into `next`.
#[allow(clippy::too_many_arguments)]
pub fn wvad_step(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    n: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row >= close.len() {
        return None;
    }
    rolling_sum_row(n, row, |j| wvad_term(open, high, low, close, volume, j), state, next)
}

/// [`super::vr`]'s state after the rows: its three rolling sums'.
pub fn vr_state(close: &[f64], volume: &[f64], n: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<1>::initial(origin).repeat(3), close.len(), |row, prev, next| {
        vr_step(close, volume, n, row, prev, next)
    })
}

/// One row of [`super::vr`] at `row`, from `state` into `next`.
pub fn vr_step(close: &[f64], volume: &[f64], n: usize, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    const S: usize = SMA_STATE;
    if row >= close.len() || state.len() != 3 * S || next.len() != 3 * S {
        return None;
    }
    let mut sums = [0.0; 3];
    for (k, sum) in sums.iter_mut().enumerate() {
        let term = |j: usize| vr_volumes(close, volume, j)[k];
        *sum = rolling_sum_row(n, row, term, &state[k * S..(k + 1) * S], &mut next[k * S..(k + 1) * S])?;
    }
    let [suv, sdv, spv] = sums;
    Some((suv + 0.5 * spv) / (sdv + 0.5 * spv) * 100.0)
}
