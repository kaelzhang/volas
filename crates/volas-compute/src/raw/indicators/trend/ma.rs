//! Moving averages: SMA / EMA / SMMA / WMA / BBI, with the EMA-family
//! state-carry helpers (`ema_k` / `ema_seed_idx` shared with the cascade and
//! MACD families).

use crate::accum::{anchors_at, Accumulator, Anchors, Csum};
use crate::guard::{all_finite, own_sums, ring_push, state_after, window_sums, Order, Slide, Window};
use crate::guard::{NonFiniteRows, RowGuard, Unguarded};
use crate::missing;
use crate::raw::indicators::av;
use crate::raw::kernels;

/// Simple moving average; row 0 is at absolute position `origin` (see
/// [`crate::accum`]).
pub fn ma(close: &[f64], period: usize, origin: usize) -> Vec<f64> {
    // Move the kernel's owned buffer out (no copy) rather than `to_vec`.
    kernels::sma(av(close), period, origin).into_raw_vec_and_offset().0
}

/// Exponential moving average (TA-Lib: SMA-seeded, `k = 2/(period+1)`).
pub fn ema(close: &[f64], period: usize) -> Vec<f64> {
    kernels::ema_seeded(av(close), period)
        .into_raw_vec_and_offset()
        .0
}

/// Smoothed moving average (Wilder's RMA: SMA-seeded, `alpha = 1/period`).
pub fn smma(close: &[f64], period: usize) -> Vec<f64> {
    kernels::wilder(av(close), period).to_vec()
}

// --- EMA-family state-carry (additive; the full-recompute fallback stays correct) ---
//
// A recursive EMA-style indicator compresses its whole history into a small fixed
// state: the single carried EMA (ema), the Wilder running value (smma), or the vector
// of cascaded sub-EMA stage values (dema/tema/t3/trix/macd). `*_final_state` captures
// that state after a full compute (returning `None` before the indicator has seeded, so
// the caller keeps the correct fallback); `*_resume` continues the recursion over only
// the new rows `[from, n)`, reading nothing before `from`, with arithmetic bit-identical
// to each kernel's steady-state loop. `from` is the cache's `valid_rows`; the carried
// state is the internal state as of row `from - 1`, which the plumbing guarantees is at
// or past the seed (a fresh full compute captured it on a non-empty column, and a slice
// only carries it when its end reaches `valid_rows`). The resume never reads `data[< from]`,
// so it continues correctly across a head-dropping slice.

/// The k used by every SMA-seeded EMA stage: `2/(period+1)`.
#[inline]
pub(super) fn ema_k(period: usize) -> f64 {
    2.0 / (period as f64 + 1.0)
}

/// Final single-EMA state `[ema]` after a full [`ema`] compute, or `None` if `data`
/// never seeded (the column is all-NaN → nothing to carry, keep the fallback). The EMA
/// is advanced with the same fused `(x-prev)·k+prev` step as `kernels::ema_seeded`.
pub fn ema_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    Some(vec![kernels::sma_seeded_last(data, period, kernels::ema_step(period))?])
}

/// Resume [`ema`] from `state = [ema_{from-1}]` over rows `[from, n)`, bit-identical to a
/// full recompute. Reads only `data[from..]`.
pub fn ema_resume(data: &[f64], period: usize, from: usize, state: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let k = ema_k(period);
    let n = data.len();
    let mut e = state[0];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &data[from..n] {
        e = ema_step(x, e, k);
        out.push(e);
    }
    (out, vec![e])
}

/// One EMA recurrence step, shared by [`ema_resume`] and [`ema_resume_one`] so the
/// full and single-row paths cannot drift: `e' = (x - e)·k + e`.
#[inline]
pub(super) fn ema_step(x: f64, e: f64, k: f64) -> f64 {
    (x - e).mul_add(k, e)
}

/// Single new-row EMA resume: the value at `row`, with the carried state as of `row`
/// (`[e]`, the value itself) written into `next` — no allocation (the live single-bar
/// fast path, bit-identical to [`ema_resume`] from the same state). `None` if the state
/// is empty, `next` is not the state's length, or `row` is out of range.
pub fn ema_resume_one(
    data: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let e = *state.first()?;
    let x = *data.get(row)?;
    let [slot] = next else { return None };
    *slot = ema_step(x, e, ema_k(period));
    Some(*slot)
}

/// Final Wilder/SMMA state `[rma]` after a full [`smma`] compute, or `None` if `data`
/// never seeded. Uses `kernels::wilder`'s exact fused `prev·a + x·b` step.
pub fn smma_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    Some(vec![kernels::sma_seeded_last(data, period, kernels::wilder_step(period))?])
}

/// Resume [`smma`] from `state = [rma_{from-1}]` over rows `[from, n)`. Reads only
/// `data[from..]`.
pub fn smma_resume(
    data: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let pf = period as f64;
    let (a, b) = ((pf - 1.0) / pf, 1.0 / pf);
    let n = data.len();
    let mut w = state[0];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &data[from..n] {
        w = smma_step(x, w, a, b);
        out.push(w);
    }
    (out, vec![w])
}

/// One SMMA/Wilder recurrence step, shared by [`smma_resume`] and [`smma_resume_one`]
/// so they cannot drift: `w' = w·a + x·b`.
#[inline]
pub(super) fn smma_step(x: f64, w: f64, a: f64, b: f64) -> f64 {
    w.mul_add(a, x * b)
}

/// Single new-row SMMA resume: the value at `row`, with the state as of `row` (`[w]`,
/// the value itself) written into `next` — no allocation (bit-identical to
/// [`smma_resume`]). `None` if the state is empty, `next` is not the state's length, or
/// `row` is out of range.
pub fn smma_resume_one(
    data: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let w = *state.first()?;
    let x = *data.get(row)?;
    let [slot] = next else { return None };
    let pf = period as f64;
    *slot = smma_step(x, w, (pf - 1.0) / pf, 1.0 / pf);
    Some(*slot)
}

/// Weighted moving average — linearly increasing weights `1..=period`, the newest
/// bar weighted heaviest (TA-Lib WMA). O(n) via a running sum + running weighted
/// sum, both exact ([`Csum`], re-anchored at absolute positions; the weighted terms
/// exact products — [`wma_slide`]), so a window's average is its own however long the
/// slide. Row 0 is at absolute position `origin`. Lookback `period-1`.
pub fn wma(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    // Skip a leading-NaN warm-up prefix (derived inputs — e.g. the stochastic %K line —
    // warm up with NaN): compute from the first finite value onward, mirroring sma/ema.
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        let sub = wma(&data[start..], period, origin + start);
        out[start..].copy_from_slice(&sub);
        return out;
    }
    let (out, finite) = wma_with::<Unguarded>(data, period, origin);
    if finite {
        out
    } else {
        wma_with::<NonFiniteRows>(data, period, origin).0
    }
}

/// The sliding WMA under a row guard (`period <= data.len()`); also returns whether
/// the running sums stayed finite. A window holding a non-finite row is weighted
/// directly on its own rows.
fn wma_with<G: RowGuard>(data: &[f64], period: usize, origin: usize) -> (Vec<f64>, bool) {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let denom = wma_weight_sum(period);
    let pf = period as f64;
    let window = |end: usize| {
        data[end + 1 - period..=end]
            .iter()
            .enumerate()
            .fold(0.0, |acc, (j, &x)| acc + x * (j + 1) as f64)
            / denom
    };
    let mut guard = G::default();
    let mut anchors = Anchors::from(origin, period);
    let mut finite = true;
    // Re-sum at an anchor or when a sum went inexact. The fast pass stops once a
    // non-finite row has poisoned it (the careful pass recomputes every row); the
    // careful pass, whose sums hold finite rows only, always re-sums.
    let mut reanchor = |guard: &G, sum: &mut Csum, wsum: &mut Csum, end: usize| {
        finite &= sum.value().is_finite() && wsum.value().is_finite();
        if finite || !guard.counts(false) {
            (*sum, *wsum) = wma_sums(data[end + 1 - period..=end].iter().copied(), |x| guard.counts(x.is_finite()));
        }
    };
    let exact = |a: &Csum, b: &Csum| a.is_exact() && b.is_exact();
    // Seed the first full window directly.
    let mut sum = Csum::default(); // plain window sum
    let mut wsum = Csum::default(); // weighted sum: newest bar * period ... oldest * 1
    #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
    for j in 0..period {
        if guard.enter(data[j].is_finite()) {
            wma_seed(&mut sum, &mut wsum, j, data[j]);
        }
        let anchor = anchors.tick();
        if (anchor || !exact(&sum, &wsum)) && j + 1 == period {
            reanchor(&guard, &mut sum, &mut wsum, j);
        }
    }
    out[period - 1] = if guard.clean() { wsum.mean(denom) } else { window(period - 1) };
    // Slide: dropping the oldest (weight 1) raises every retained weight by one.
    for i in period..n {
        let entering = data[i];
        let leaving = data[i - period];
        let came_in = guard.enter(entering.is_finite());
        let went_out = guard.leave(leaving.is_finite());
        wma_slide(&mut sum, &mut wsum, pf, came_in.then_some(entering), went_out.then_some(leaving));
        let anchor = anchors.tick();
        if anchor || !exact(&sum, &wsum) {
            reanchor(&guard, &mut sum, &mut wsum, i);
        }
        out[i] = if guard.clean() { wsum.mean(denom) } else { window(i) };
    }
    (out, finite && sum.value().is_finite() && wsum.value().is_finite())
}

/// The WMA sums `(Σx, Σw·x)` of one window, its rows oldest first (weight `1..`),
/// those `admit` keeps — the seed of a first window, and every re-anchoring.
fn wma_sums(rows: impl Iterator<Item = f64> + Clone, admit: impl Fn(f64) -> bool) -> (Csum, Csum) {
    let [sum, wsum] = Csum::settle(wma_sums_in(rows.clone(), &admit), || wma_sums_in(rows, &admit));
    (sum, wsum)
}

/// [`wma_sums`] over an [`Accumulator`].
fn wma_sums_in<A: Accumulator>(rows: impl Iterator<Item = f64>, admit: &impl Fn(f64) -> bool) -> [A; 2] {
    let (mut sum, mut wsum) = (A::default(), A::default());
    for (j, x) in rows.enumerate() {
        if admit(x) {
            wma_seed(&mut sum, &mut wsum, j, x);
        }
    }
    [sum, wsum]
}

/// The sum of the WMA weights `1..=period`, `period·(period+1)/2`, formed in `u128` —
/// exact for every `period` — and rounded to `f64` once.
fn wma_weight_sum(period: usize) -> f64 {
    let q = period as u128;
    (q * (q + 1) / 2) as f64
}

/// Seed the first WMA window with its `j`-th row (weight `j + 1`), the weighted term
/// exact.
#[inline(always)]
fn wma_seed<A: Accumulator>(sum: &mut A, wsum: &mut A, j: usize, x: f64) {
    sum.add(x);
    wsum.add_prod(x, (j + 1) as f64);
}

/// Slide the WMA sums one row: the entering row (if in the sums) joins `Σw·x` at
/// weight `period` (exactly), every retained weight drops by one (`Σw·x` loses the old
/// `Σx`, both its words), then `Σx` takes the entering row and drops the leaving one.
#[inline(always)]
fn wma_slide(sum: &mut Csum, wsum: &mut Csum, pf: f64, entering: Option<f64>, leaving: Option<f64>) {
    if let Some(e) = entering {
        wsum.add_prod(pf, e);
    }
    wsum.sub_sum(sum);
    if let Some(e) = entering {
        sum.add(e);
    }
    if let Some(l) = leaving {
        sum.sub(l);
    }
}

/// Volume-Weighted Moving Average (TradingView `ta.vwma`): `Σ(close·volume, n) /
/// Σ(volume, n)` over each trailing `period` window. A window whose volume sums
/// to zero yields `NaN` (no weight). Row 0 is at absolute position `origin`. Lookback
/// `period-1`. O(n) sliding, exact sums ([`Csum`]).
pub fn vwma(close: &[f64], volume: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut pv = Csum::default(); // Σ close·volume
    let mut vv = Csum::default(); // Σ volume
    let mut anchors = Anchors::from(origin, period);
    let mut clean = true;
    for i in 0..n {
        pv.add(close[i] * volume[i]);
        vv.add(volume[i]);
        if i >= period {
            pv.sub(close[i - period] * volume[i - period]);
            vv.sub(volume[i - period]);
        }
        let anchor = anchors.tick();
        if (anchor || !(pv.is_exact() && vv.is_exact())) && i + 1 >= period {
            clean &= pv.value().is_finite() && vv.value().is_finite();
            if clean {
                [pv, vv] = Csum::of_rows((i + 1 - period..=i).map(|j| [close[j] * volume[j], volume[j]]));
            }
        }
        if i >= period - 1 {
            out[i] = vwma_value([pv.value(), vv.value()]);
        }
    }
    if clean && pv.value().is_finite() && vv.value().is_finite() {
        return out;
    }
    // A non-finite bar entered: window-local careful path (see `crate::guard`).
    window_sums(n, period, Order::AddDrop, origin, |i| [close[i] * volume[i], volume[i]], |i, s| {
        out[i] = vwma_value(s);
    });
    out
}

/// VWMA from its window's `[Σc·v, Σv]`: NaN for a zero volume.
#[inline]
fn vwma_value([pv, vv]: [f64; 2]) -> f64 {
    if vv != 0.0 { pv / vv } else { f64::NAN }
}

/// Arnaud Legoux Moving Average (TradingView `ta.alma`): a Gaussian-weighted
/// window MA. `m = offset·(period-1)` (NOT floored — matching Pine's standard
/// form), `s = period/sigma`, weight `wᵢ = exp(-(i-m)²/(2s²))` for `i = 0..period`
/// applied oldest→newest, normalized to sum 1. Lookback `period-1`.
pub fn alma(close: &[f64], period: usize, offset: f64, sigma: f64) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let m = offset * (period - 1) as f64;
    let s = period as f64 / sigma;
    let mut w = vec![0.0; period];
    let mut norm = 0.0;
    for (i, wi) in w.iter_mut().enumerate() {
        let d = i as f64 - m;
        *wi = (-(d * d) / (2.0 * s * s)).exp();
        norm += *wi;
    }
    for wi in w.iter_mut() {
        *wi /= norm;
    }
    for i in (period - 1)..n {
        let mut acc = 0.0;
        for (k, &wk) in w.iter().enumerate() {
            acc += close[i + 1 - period + k] * wk;
        }
        out[i] = acc;
    }
    out
}

/// Hull Moving Average (TradingView `ta.hma`): `WMA(2·WMA(close, n/2) −
/// WMA(close, n), round(√n))`, with `n/2` floored. Lower lag than a plain WMA.
/// Row 0 is at absolute position `origin`. Lookback `n + round(√n) − 2`.
pub fn hma(close: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let n = close.len();
    if period == 0 || period > n {
        return vec![f64::NAN; n];
    }
    // period 1 is degenerate (half = 0 → `wma(close, 0)` is all-NaN); like the rest of
    // the MA family at period 1, return the source unchanged.
    if period == 1 {
        return close.to_vec();
    }
    let half = period / 2;
    let sq = (period as f64).sqrt().round() as usize;
    let wma_half = wma(close, half, origin);
    let wma_full = wma(close, period, origin);
    let raw: Vec<f64> = (0..n).map(|i| 2.0 * wma_half[i] - wma_full[i]).collect();
    wma(&raw, sq, origin)
}

/// Symmetrically-Weighted Moving Average (TradingView `ta.swma`): fixed 4-bar
/// window, weights `[1/6, 2/6, 2/6, 1/6]` (oldest→newest). Lookback 3.
pub fn swma(close: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    for i in 3..n {
        out[i] = close[i - 3] / 6.0
            + close[i - 2] * 2.0 / 6.0
            + close[i - 1] * 2.0 / 6.0
            + close[i] / 6.0;
    }
    out
}

/// Bull and Bear Index (`mean of ma:a, ma:b, ma:c, ma:d`); row 0 at absolute position
/// `origin`.
pub fn bbi(close: &[f64], a: usize, b: usize, c: usize, d: usize, origin: usize) -> Vec<f64> {
    let data = av(close);
    let ma_a = kernels::sma(data, a, origin);
    let ma_b = kernels::sma(data, b, origin);
    let ma_c = kernels::sma(data, c, origin);
    let ma_d = kernels::sma(data, d, origin);
    ((&ma_a + &ma_b + &ma_c + &ma_d) / 4.0).to_vec()
}

// ---------------------------------------------------------------------------
// Resumable forms: each finite-window average carries the running sums of its full
// computation (in its exact operation order), so a resume of one row at a time is
// bit-identical to recomputing the whole column — and replacing the newest row costs
// one step.
// ---------------------------------------------------------------------------

/// The resumable state of [`wma`]: `[Σx, Σw·x]` (each a [`Csum`]), the non-finite rows
/// in the window, the rows since the first non-NaN value (the seed of the first window,
/// then the slide), the absolute position of the next row (for re-anchoring).
pub(crate) const WMA_STATE: usize = 2 * Csum::STATE + 3;

/// The state of a WMA before its first row, at absolute position `origin`.
pub(crate) fn wma_initial(origin: usize) -> Vec<f64> {
    let mut v = vec![0.0; WMA_STATE];
    v[6] = origin as f64;
    v
}

/// One row of [`wma`] over the series `term` (the careful [`wma_with`] loop, its
/// leading-NaN skip included), from `state` into `next`.
pub(crate) fn wma_row(period: usize, row: usize, term: &impl Fn(usize) -> f64, state: &[f64], next: &mut [f64]) -> Option<f64> {
    // Every caller (`FiniteMa::step`, the HMA / DKX / Coppock steps) checks both lengths.
    debug_assert!(state.len() == WMA_STATE && next.len() == WMA_STATE);
    let mut sum = Csum::from_state(&state[..2])?;
    let mut wsum = Csum::from_state(&state[2..4])?;
    let (mut bad, seen, pos) = (state[4] as usize, state[5] as usize, state[6] as usize);
    let x = term(row);
    if period == 0 || (seen == 0 && x.is_nan()) {
        // the leading-NaN warm-up: skipped (not a row of the average yet)
        next.copy_from_slice(state);
        next[6] = (pos + 1) as f64;
        return Some(f64::NAN);
    }
    let denom = wma_weight_sum(period);
    let pf = period as f64;
    let window = || {
        (row + 1 - period..=row)
            .enumerate()
            .fold(0.0, |acc, (j, k)| acc + term(k) * (j + 1) as f64)
            / denom
    };
    let value = if seen < period {
        // seed the first full window directly
        if x.is_finite() {
            wma_seed(&mut sum, &mut wsum, seen, x);
        } else {
            bad += 1;
        }
        if seen + 1 == period {
            if anchors_at(pos, period) || !(sum.is_exact() && wsum.is_exact()) {
                (sum, wsum) = wma_sums((row + 1 - period..=row).map(term), f64::is_finite);
            }
            if bad == 0 { wsum.mean(denom) } else { window() }
        } else {
            f64::NAN
        }
    } else {
        let leaving = term(row - period);
        let came_in = x.is_finite();
        let went_out = leaving.is_finite();
        if !came_in {
            bad += 1;
        }
        if !went_out {
            bad -= 1;
        }
        wma_slide(&mut sum, &mut wsum, pf, came_in.then_some(x), went_out.then_some(leaving));
        if anchors_at(pos, period) || !(sum.is_exact() && wsum.is_exact()) {
            (sum, wsum) = wma_sums((row + 1 - period..=row).map(term), f64::is_finite);
        }
        if bad == 0 { wsum.mean(denom) } else { window() }
    };
    sum.write_state(&mut next[..2]);
    wsum.write_state(&mut next[2..4]);
    next[4] = bad as f64;
    next[5] = (seen + 1).min(period) as f64;
    next[6] = (pos + 1) as f64;
    Some(value)
}

/// The period `mavp` (TA-Lib MAVP) averages a row over: the row's `periods` value
/// truncated toward zero and clamped to `[min_period, max_period]`. `None` when that value
/// is missing — the row reads no average and is missing (the per-row term rule, see
/// [`crate::missing`]); a cast would read it as `0` and clamp it to `min_period`.
#[inline]
pub fn mavp_period(value: f64, min_period: usize, max_period: usize) -> Option<usize> {
    if missing::is_missing(value) {
        return None;
    }
    Some((value as i64).clamp(min_period as i64, max_period as i64) as usize)
}

/// A finite-window moving average of a TA-Lib MA type — SMA (`0`), WMA (`2`) or TRIMA
/// (`5`) — as a resumable stage over any series, for the commands composed of them
/// (`ma`, `apo`, `ppo`, `macdext`, `bias`, `dma`, `mavp`, the stochastics). Its step is
/// the kernel's own row ([`kernels::sma_row`], [`wma_row`], [`trima_row`]).
#[derive(Clone, Copy, Debug)]
pub struct FiniteMa {
    period: usize,
    matype: usize,
}

impl FiniteMa {
    /// The stage of `matype` over `period`, or `None` for a type that is not
    /// finite-window (a recursive average carries its own state).
    pub fn new(period: usize, matype: usize) -> Option<Self> {
        matches!(matype, 0 | 2 | 5).then_some(FiniteMa { period, matype })
    }

    /// The length of its state.
    pub fn state_len(&self) -> usize {
        match self.matype {
            0 => kernels::SMA_STATE,
            2 => WMA_STATE,
            _ => super::trima_state_len(self.period),
        }
    }

    /// How many rows before the current one its step reads (a stage over another
    /// stage's output keeps that many of them).
    pub fn reach(&self) -> usize {
        match self.matype {
            5 => super::trima_reach(self.period),
            _ => self.period,
        }
    }

    /// One row of the average of `term` at `row`, from `state` into `next`.
    pub fn step(
        &self,
        row: usize,
        term: impl Fn(usize) -> f64,
        state: &[f64],
        next: &mut [f64],
    ) -> Option<f64> {
        if next.len() != self.state_len() || state.len() != next.len() {
            return None;
        }
        match self.matype {
            0 => kernels::sma_step(self.period, row, term, state, next),
            2 => wma_row(self.period, row, &term, state, next),
            _ => super::trima_row(self.period, row, &term, state, next),
        }
    }

    /// Its state before its first row, at absolute position `origin`.
    pub fn initial(&self, origin: usize) -> Vec<f64> {
        match self.matype {
            0 => Slide::<1>::initial(origin),
            2 => wma_initial(origin),
            _ => super::trima_initial(self.period, origin),
        }
    }

    /// Its state as a stage over a series produced one row at a time, before its first
    /// row (at absolute position `origin`): [`initial`](Self::initial), then the
    /// series' (not yet seen) previous values.
    pub fn chained_initial(&self, origin: usize) -> Vec<f64> {
        let mut v = self.initial(origin);
        v.resize(self.chained_len(), 0.0);
        v
    }

    /// The length of its state as a stage over a series produced one row at a time:
    /// its own state, then the last [`reach`](Self::reach) values of that series.
    pub fn chained_len(&self) -> usize {
        self.state_len() + self.reach()
    }

    /// One row of the average of a series produced one row at a time — `newest` is its
    /// value at `row` — from `state` into `next` (both [`chained_len`](Self::chained_len)
    /// long: the stage's state, then the series' previous values, oldest first).
    pub fn step_chained(&self, row: usize, newest: f64, state: &[f64], next: &mut [f64]) -> Option<f64> {
        let (len, reach) = (self.state_len(), self.reach());
        if state.len() != len + reach || next.len() != state.len() {
            return None;
        }
        let (own, ring) = state.split_at(len);
        let (n_own, n_ring) = next.split_at_mut(len);
        // the series at row j: `ring` holds rows `row-reach .. row-1`, oldest first
        let at = |j: usize| if j == row { newest } else { ring[reach - (row - j)] };
        let value = self.step(row, at, own, n_own)?;
        ring_push(ring, newest, n_ring);
        Some(value)
    }

    /// The state after the rows `0..n` of `term`, row 0 at absolute position `origin`.
    pub fn state(&self, n: usize, origin: usize, term: impl Fn(usize) -> f64) -> Option<Vec<f64>> {
        state_after(self.initial(origin), n, |row, prev, next| self.step(row, &term, prev, next))
    }
}

/// [`ma`]'s state after the rows of `close` (row 0 at absolute position `origin`) —
/// see [`FiniteMa`].
pub fn ma_state(close: &[f64], period: usize, matype: usize, origin: usize) -> Option<Vec<f64>> {
    FiniteMa::new(period, matype)?.state(close.len(), origin, |j| close[j])
}

/// One row of [`ma`] (of `matype`) at `row`, from `state` into `next`.
pub fn ma_step(
    close: &[f64],
    period: usize,
    matype: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row >= close.len() {
        return None;
    }
    FiniteMa::new(period, matype)?.step(row, |j| close[j], state, next)
}

/// [`vwma`]'s state (row 0 at absolute position `origin`): the [`Slide`] of `[Σc·v, Σv]`.
pub fn vwma_state(close: &[f64], volume: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<2>::initial(origin), close.len(), |row, prev, next| {
        vwma_step(close, volume, period, row, prev, next)
    })
}

/// One row of [`vwma`] at `row`, from `state` into `next`.
pub fn vwma_step(
    close: &[f64],
    volume: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || row >= close.len() || next.len() != Slide::<2>::STATE {
        return None;
    }
    let term = |i: usize| [close[i] * volume[i], volume[i]];
    let mut slide = Slide::<2>::from_state(state)?;
    let value = match slide.row(Order::AddDrop, period, row, |j| all_finite(term(j))) {
        Window::Partial => f64::NAN,
        Window::Sums(s) => vwma_value(s.map(|c| c.value())),
        Window::Bad => vwma_value(own_sums(row + 1 - period..=row, term)),
    };
    slide.write_state(next);
    Some(value)
}

/// [`hma`]'s state: its two WMAs of close, its WMA of their blend, and the last
/// `round(√period)` blended values that last WMA reads.
fn hma_parts(period: usize) -> (usize, usize) {
    (period / 2, (period as f64).sqrt().round() as usize)
}

/// The length of [`hma`]'s state.
fn hma_state_len(period: usize) -> usize {
    3 * WMA_STATE + hma_parts(period).1
}

/// [`hma`]'s state after the rows of `close` (row 0 at absolute position `origin`).
pub fn hma_state(close: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    if period < 2 {
        return None;
    }
    let mut init = [wma_initial(origin), wma_initial(origin), wma_initial(origin)].concat();
    init.resize(hma_state_len(period), 0.0);
    state_after(init, close.len(), |row, prev, next| {
        hma_step(close, period, row, prev, next)
    })
}

/// One row of [`hma`] at `row`, from `state` into `next`.
pub fn hma_step(
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period < 2 || row >= close.len() || state.len() != hma_state_len(period) || next.len() != state.len() {
        return None;
    }
    let (half, sq) = hma_parts(period);
    let c = |j: usize| close[j];
    let (s_half, rest) = state.split_at(WMA_STATE);
    let (s_full, rest) = rest.split_at(WMA_STATE);
    let (s_out, ring) = rest.split_at(WMA_STATE);
    let (n_half, nrest) = next.split_at_mut(WMA_STATE);
    let (n_full, nrest) = nrest.split_at_mut(WMA_STATE);
    let (n_out, n_ring) = nrest.split_at_mut(WMA_STATE);
    let wh = wma_row(half, row, &c, s_half, n_half)?;
    let wf = wma_row(period, row, &c, s_full, n_full)?;
    let raw = 2.0 * wh - wf;
    // the blend at row j: `ring` holds rows `row-sq .. row-1`, oldest first
    let blend = |j: usize| if j == row { raw } else { ring[sq - (row - j)] };
    let value = wma_row(sq, row, &blend, s_out, n_out)?;
    ring_push(ring, raw, n_ring);
    Some(value)
}

/// [`bbi`]'s state: the four SMAs of close (row 0 at absolute position `origin`).
pub fn bbi_state(close: &[f64], a: usize, b: usize, c: usize, d: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<1>::initial(origin).repeat(4), close.len(), |row, prev, next| {
        bbi_step(close, a, b, c, d, row, prev, next)
    })
}

/// One row of [`bbi`] at `row`, from `state` into `next`.
#[allow(clippy::too_many_arguments)]
pub fn bbi_step(
    close: &[f64],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    const S: usize = kernels::SMA_STATE;
    if row >= close.len() || state.len() != 4 * S || next.len() != 4 * S {
        return None;
    }
    let term = |j: usize| close[j];
    let mut m = [0.0; 4];
    for (k, p) in [a, b, c, d].into_iter().enumerate() {
        m[k] = kernels::sma_step(p, row, term, &state[k * S..(k + 1) * S], &mut next[k * S..(k + 1) * S])?;
    }
    Some((m[0] + m[1] + m[2] + m[3]) / 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `hma:1` is degenerate (`half = period / 2 = 0`); like every other MA at period 1
    /// (`sma` / `ema` / `wma` / `vwma` / `alma`), it returns the source rather than the
    /// all-NaN that `wma(close, 0)` used to produce.
    #[test]
    fn hma_period_one_returns_source() {
        assert_eq!(hma(&[1.0, 2.0, 3.0], 1, 0), vec![1.0, 2.0, 3.0]);
    }
}
