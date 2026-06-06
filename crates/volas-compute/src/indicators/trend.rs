use ndarray::Array1;

use super::av;
use crate::kernels;

// ---------------------------------------------------------------------------
// Trend-following
// ---------------------------------------------------------------------------

/// Simple moving average.
pub fn ma(close: &[f64], period: usize) -> Vec<f64> {
    // Move the kernel's owned buffer out (no copy) rather than `to_vec`.
    kernels::sma(av(close), period).into_raw_vec_and_offset().0
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
fn ema_k(period: usize) -> f64 {
    2.0 / (period as f64 + 1.0)
}

/// Index of the EMA seed (the `period`-th finite value of `data`), or `None` if fewer
/// than `period` finite values exist. Mirrors `kernels::sma_seeded`'s seeding scan, so a
/// captured state corresponds to the same seed the full kernel used.
fn ema_seed_idx(data: &[f64], period: usize) -> Option<usize> {
    if period == 0 {
        return None;
    }
    let mut count = 0usize;
    for (i, &x) in data.iter().enumerate() {
        if !x.is_nan() {
            count += 1;
            if count == period {
                return Some(i);
            }
        }
    }
    None
}

/// Final single-EMA state `[ema]` after a full [`ema`] compute, or `None` if `data`
/// never seeded (the column is all-NaN → nothing to carry, keep the fallback). The EMA
/// is advanced with the same fused `(x-prev)·k+prev` step as `kernels::ema_seeded`.
pub fn ema_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let k = ema_k(period);
    let si = ema_seed_idx(data, period)?;
    let mut e = data[si + 1 - period..=si].iter().sum::<f64>() / period as f64;
    for &x in &data[si + 1..] {
        e = (x - e).mul_add(k, e);
    }
    Some(vec![e])
}

/// Resume [`ema`] from `state = [ema_{from-1}]` over rows `[from, n)`, bit-identical to a
/// full recompute. Reads only `data[from..]`.
pub fn ema_resume(data: &[f64], period: usize, from: usize, state: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let k = ema_k(period);
    let n = data.len();
    let mut e = state[0];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &data[from..n] {
        e = (x - e).mul_add(k, e);
        out.push(e);
    }
    (out, vec![e])
}

/// Final Wilder/SMMA state `[rma]` after a full [`smma`] compute, or `None` if `data`
/// never seeded. Uses `kernels::wilder`'s exact fused `prev·a + x·b` step.
pub fn smma_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let pf = period as f64;
    let (a, b) = ((pf - 1.0) / pf, 1.0 / pf);
    let si = ema_seed_idx(data, period)?;
    let mut w = data[si + 1 - period..=si].iter().sum::<f64>() / pf;
    for &x in &data[si + 1..] {
        w = w.mul_add(a, x * b);
    }
    Some(vec![w])
}

/// Resume [`smma`] from `state = [rma_{from-1}]` over rows `[from, n)`. Reads only
/// `data[from..]`.
pub fn smma_resume(data: &[f64], period: usize, from: usize, state: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let pf = period as f64;
    let (a, b) = ((pf - 1.0) / pf, 1.0 / pf);
    let n = data.len();
    let mut w = state[0];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &data[from..n] {
        w = w.mul_add(a, x * b);
        out.push(w);
    }
    (out, vec![w])
}

/// Weighted moving average — linearly increasing weights `1..=period`, the newest
/// bar weighted heaviest (TA-Lib WMA). O(n) via a running sum + running weighted
/// sum. Lookback `period-1`.
pub fn wma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    // Skip a leading-NaN warm-up prefix (derived inputs — e.g. the stochastic %K line —
    // warm up with NaN): compute from the first finite value onward, mirroring sma/ema.
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        let sub = wma(&data[start..], period);
        out[start..].copy_from_slice(&sub);
        return out;
    }
    let denom = (period * (period + 1) / 2) as f64; // sum of weights 1..=period
    let pf = period as f64;
    // Seed the first full window directly.
    let mut sum = 0.0; // plain window sum
    let mut wsum = 0.0; // weighted sum: newest bar * period ... oldest * 1
    for j in 0..period {
        sum += data[j];
        wsum += data[j] * (j + 1) as f64;
    }
    out[period - 1] = wsum / denom;
    // Slide: dropping the oldest (weight 1) raises every retained weight by one.
    for i in period..n {
        wsum += pf * data[i] - sum;
        sum += data[i] - data[i - period];
        out[i] = wsum / denom;
    }
    out
}

/// Staggered warm-up for a cascade of `S` SMA-seeded EMAs (`k = 2/(period+1)`): each
/// stage SMA-seeds over the first `period` finite values of its predecessor, so stage
/// `j` seeds at `j·(period-1)`. Returns `(lookback, stage values at lookback)` where
/// `lookback = S·(period-1)` is the first fully-valid bar, or `None` if it never seeds.
/// Shared by DEMA/TEMA/T3 — their combine and steady-state recurrence differ, but the
/// warm-up is identical (and bit-identical to chaining `ema_seeded` `S` times).
fn cascade_warmup<const S: usize>(data: &[f64], period: usize, k: f64) -> Option<(usize, [f64; S])> {
    let n = data.len();
    if period == 0 {
        return None;
    }
    // Account for a leading-NaN warm-up prefix (derived inputs — e.g. the stochastic
    // %K line — warm up with NaN). The cascade seeds over the first finite values, so
    // the last stage is ready at `start + S*(period-1)`, not from index 0; computing
    // the lookback from index 0 returns unseeded stages (garbage) too early.
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    let lookback = start + S * (period - 1);
    if lookback >= n {
        return None;
    }
    let mut e = [0.0f64; S];
    let mut acc = [0.0f64; S];
    let mut cnt = [0usize; S];
    let mut seeded = [false; S];
    for &raw in &data[..=lookback] {
        let mut x = raw;
        for s in 0..S {
            if seeded[s] {
                e[s] = (x - e[s]).mul_add(k, e[s]);
                x = e[s];
            } else if !x.is_nan() {
                acc[s] += x;
                cnt[s] += 1;
                if cnt[s] == period {
                    e[s] = acc[s] / period as f64;
                    seeded[s] = true;
                    x = e[s];
                } else {
                    x = f64::NAN;
                }
            } else {
                x = f64::NAN;
            }
        }
    }
    Some((lookback, e))
}

/// Double EMA: `2*EMA - EMA(EMA)` (TA-Lib DEMA). Lookback `2*(period-1)`. Single-pass
/// lattice over the two cascaded EMAs (vs two `ema_seeded` passes + a combine);
/// bit-identical to `2·e1 − e2`.
pub fn dema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let k = 2.0 / (period as f64 + 1.0);
    let Some((lookback, e)) = cascade_warmup::<2>(data, period, k) else {
        return out;
    };
    out[lookback] = 2.0 * e[0] - e[1];
    let (mut e0, mut e1) = (e[0], e[1]);
    for i in (lookback + 1)..n {
        e0 = (data[i] - e0).mul_add(k, e0);
        e1 = (e0 - e1).mul_add(k, e1);
        out[i] = 2.0 * e0 - e1;
    }
    out
}

/// Triple EMA: `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))` (TA-Lib TEMA). Lookback
/// `3*(period-1)`. Single-pass lattice over the three cascaded EMAs; bit-identical to
/// `3·e1 − 3·e2 + e3`.
pub fn tema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let k = 2.0 / (period as f64 + 1.0);
    let Some((lookback, e)) = cascade_warmup::<3>(data, period, k) else {
        return out;
    };
    out[lookback] = 3.0 * e[0] - 3.0 * e[1] + e[2];
    let (mut e0, mut e1, mut e2) = (e[0], e[1], e[2]);
    for i in (lookback + 1)..n {
        e0 = (data[i] - e0).mul_add(k, e0);
        e1 = (e0 - e1).mul_add(k, e1);
        e2 = (e1 - e2).mul_add(k, e2);
        out[i] = 3.0 * e0 - 3.0 * e1 + e2;
    }
    out
}

/// Triangular moving average (TA-Lib TRIMA): a double SMA whose net weights rise to
/// the window centre then fall. Since convolution commutes, the two SMA windows are
/// `((n+1)/2, (n+1)/2)` for odd `n` and `(n/2, n/2+1)` for even `n` (either order).
/// Lookback `period-1`.
pub fn trima(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    // Skip a leading-NaN warm-up prefix (see `wma`).
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        let sub = trima(&data[start..], period);
        out[start..].copy_from_slice(&sub);
        return out;
    }
    let (a, b) = if period % 2 == 1 {
        let m = (period + 1) / 2;
        (m, m)
    } else {
        let m = period / 2;
        (m, m + 1)
    };
    // Fuse the two SMA convolutions into one pass (no intermediate `inner` array): a
    // running `sum1` slides the inner a-window, each inner mean feeds `sum2` — the
    // outer b-window sum over a ring of the last `b` inner means. Both windows use SMA's
    // exact add-then-subtract order, and the outer slide begins at the first finite
    // inner mean (index a-1, mirroring the leading-NaN skip), so this is bit-identical
    // to `sma(sma(data, a), b)`.
    let af = a as f64;
    let bf = b as f64;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut ring = vec![0.0; b];
    let mut slot = 0usize;
    let mut inner_cnt = 0usize;
    for i in 0..n {
        sum1 += data[i];
        if i >= a {
            sum1 -= data[i - a];
        }
        if i + 1 >= a {
            let inner = sum1 / af;
            sum2 += inner;
            if inner_cnt >= b {
                sum2 -= ring[slot]; // the inner mean leaving the outer window
            }
            ring[slot] = inner;
            slot += 1;
            if slot == b {
                slot = 0;
            }
            inner_cnt += 1;
            if inner_cnt >= b {
                out[i] = sum2 / bf;
            }
        }
    }
    out
}

/// Tillson T3 (TA-Lib T3): `c1·e6 + c2·e5 + c3·e4 + c4·e3` over six cascaded
/// SMA-seeded EMAs, with `vfactor`-derived coefficients `c1=-v³`, `c2=3(v²−c1)`,
/// `c3=-6v²−3(v−c1)`, `c4=1+3v−c1+3v²` (computed in TA-Lib's exact float order).
/// Default period 5, vfactor 0.7; lookback `6·(period-1)`.
pub fn t3(data: &[f64], period: usize, vfactor: f64) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let v2 = vfactor * vfactor;
    let c1 = -(v2 * vfactor);
    let c2 = 3.0 * (v2 - c1);
    let c3 = -6.0 * v2 - 3.0 * (vfactor - c1);
    let c4 = 1.0 + 3.0 * vfactor - c1 + 3.0 * v2;
    let k = 2.0 / (period as f64 + 1.0);
    // Single-pass lattice instead of six sequential `ema_seeded` passes: the six EMAs
    // cascade (stage j consumes stage j-1's *current* output) so they share one
    // staggered-warmup traversal ([`cascade_warmup`]); the steady state then advances a
    // six-deep FMA chain per bar (TA-Lib's lattice) and emits the combine in place.
    // Bit-identical to the six-call form.
    let Some((lookback, e)) = cascade_warmup::<6>(data, period, k) else {
        return out;
    };
    out[lookback] = c1 * e[5] + c2 * e[4] + c3 * e[3] + c4 * e[2];
    // Steady state: all six stages seeded. The cascade is inherently sequential (a
    // six-deep FMA dependency chain per bar, exactly as in TA-Lib's lattice), so we
    // unroll it into registers and emit the combine in the same pass.
    let (mut e0, mut e1, mut e2, mut e3, mut e4, mut e5) =
        (e[0], e[1], e[2], e[3], e[4], e[5]);
    for i in (lookback + 1)..n {
        e0 = (data[i] - e0).mul_add(k, e0);
        e1 = (e0 - e1).mul_add(k, e1);
        e2 = (e1 - e2).mul_add(k, e2);
        e3 = (e2 - e3).mul_add(k, e3);
        e4 = (e3 - e4).mul_add(k, e4);
        e5 = (e4 - e5).mul_add(k, e5);
        out[i] = c1 * e5 + c2 * e4 + c3 * e3 + c4 * e2;
    }
    out
}

/// Final DEMA state `[e0, e1]` (the two cascaded EMAs as of the last row), or `None` if
/// unseeded. Pairs with [`dema_resume`].
pub fn dema_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let e = kernels::ema_cascade_final::<2>(data, period)?;
    Some(e.to_vec())
}

/// Resume [`dema`] from `state = [e0, e1]` over rows `[from, n)`, bit-identical to a full
/// recompute (the same `2·e0 − e1` lattice). Reads only `data[from..]`.
pub fn dema_resume(data: &[f64], period: usize, from: usize, state: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let k = ema_k(period);
    let n = data.len();
    let mut e = [state[0], state[1]];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &data[from..n] {
        kernels::ema_cascade_step(&mut e, x, k);
        out.push(2.0 * e[0] - e[1]);
    }
    (out, e.to_vec())
}

/// Final TEMA state `[e0, e1, e2]`, or `None` if unseeded. Pairs with [`tema_resume`].
pub fn tema_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let e = kernels::ema_cascade_final::<3>(data, period)?;
    Some(e.to_vec())
}

/// Resume [`tema`] from `state = [e0, e1, e2]` over rows `[from, n)`, bit-identical to a
/// full recompute (the `3·e0 − 3·e1 + e2` lattice). Reads only `data[from..]`.
pub fn tema_resume(data: &[f64], period: usize, from: usize, state: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let k = ema_k(period);
    let n = data.len();
    let mut e = [state[0], state[1], state[2]];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &data[from..n] {
        kernels::ema_cascade_step(&mut e, x, k);
        out.push(3.0 * e[0] - 3.0 * e[1] + e[2]);
    }
    (out, e.to_vec())
}

/// Final T3 state `[e0..e5]`, or `None` if unseeded. Pairs with [`t3_resume`].
pub fn t3_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let e = kernels::ema_cascade_final::<6>(data, period)?;
    Some(e.to_vec())
}

/// Resume [`t3`] from `state = [e0..e5]` over rows `[from, n)`, bit-identical to a full
/// recompute (the same six-deep lattice + `c1·e5 + c2·e4 + c3·e3 + c4·e2` combine, with
/// `vfactor`-derived coefficients in TA-Lib's float order). Reads only `data[from..]`.
pub fn t3_resume(
    data: &[f64],
    period: usize,
    vfactor: f64,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let v2 = vfactor * vfactor;
    let c1 = -(v2 * vfactor);
    let c2 = 3.0 * (v2 - c1);
    let c3 = -6.0 * v2 - 3.0 * (vfactor - c1);
    let c4 = 1.0 + 3.0 * vfactor - c1 + 3.0 * v2;
    let k = ema_k(period);
    let n = data.len();
    let mut e = [state[0], state[1], state[2], state[3], state[4], state[5]];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &data[from..n] {
        kernels::ema_cascade_step(&mut e, x, k);
        out.push(c1 * e[5] + c2 * e[4] + c3 * e[3] + c4 * e[2]);
    }
    (out, e.to_vec())
}

/// Kaufman Adaptive Moving Average (TA-Lib KAMA). The smoothing constant adapts each
/// bar via the efficiency ratio `ER = |price[i]-price[i-period]| / Σ|1-bar changes|`:
/// `SC = (ER·(fast−slow) + slow)²` with fast `=2/3`, slow `=2/31`; then
/// `KAMA[i] = KAMA[i-1] + SC·(price[i] − KAMA[i-1])`, seeded `KAMA[period-1] =
/// price[period-1]`. First value at index `period` (lookback = period). Faithful port
/// of TA-Lib's sliding-sum recurrence.
pub fn kama(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period + 1 > n {
        return out;
    }
    // Skip a leading-NaN warm-up prefix (see `wma`).
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        let sub = kama(&data[start..], period);
        out[start..].copy_from_slice(&sub);
        return out;
    }
    // sumROC1 = Σ_{j=0}^{period-1} |price[j] − price[j+1]| (the first window).
    let mut sum_roc1 = 0.0;
    for j in 0..period {
        sum_roc1 += (data[j] - data[j + 1]).abs();
    }
    let mut trailing_idx = 0usize;
    let mut today = period;
    // First KAMA (index `period`): the prior KAMA is seeded with yesterday's price.
    let mut prev_kama = data[today - 1];
    let mut trailing_value = data[trailing_idx];
    let period_roc = data[today] - trailing_value;
    trailing_idx += 1;
    let sc = kama_sc(period_roc, sum_roc1);
    // `(price − prev)·sc + prev` fused: one rounding off the recurrence's critical
    // path. KAMA is contractive (sc ≤ (2/3)² < 1), so the ~1e-16 divergence decays —
    // within the 1e-9 TA-Lib parity tolerance.
    prev_kama = (data[today] - prev_kama).mul_add(sc, prev_kama);
    out[today] = prev_kama;
    today += 1;

    while today < n {
        let tr2 = data[trailing_idx];
        trailing_idx += 1;
        let period_roc = data[today] - tr2;
        sum_roc1 -= (trailing_value - tr2).abs(); // drop the oldest 1-bar change
        sum_roc1 += (data[today] - data[today - 1]).abs(); // add the newest
        trailing_value = tr2;
        let sc = kama_sc(period_roc, sum_roc1);
        prev_kama = (data[today] - prev_kama).mul_add(sc, prev_kama);
        out[today] = prev_kama;
        today += 1;
    }
    out
}

/// KAMA's adaptive smoothing constant `SC = (ER·(fast−slow) + slow)²` from the period
/// ROC and the sliding sum of 1-bar changes, with `ER` clamped to 1 on a tiny/degenerate
/// denominator (TA-Lib's guard). Hoisted so [`kama`], [`kama_final_state`], and
/// [`kama_resume`] share byte-identical arithmetic (`fast = 2/3`, `slow = 2/31`).
#[inline]
fn kama_sc(period_roc: f64, sum_roc1: f64) -> f64 {
    const CONST_MAX: f64 = 2.0 / (30.0 + 1.0); // slow smoothing constant
    let const_diff = 2.0 / (2.0 + 1.0) - CONST_MAX; // fast − slow
    let er = if sum_roc1 <= period_roc || sum_roc1.abs() < 1e-14 {
        1.0
    } else {
        (period_roc / sum_roc1).abs()
    };
    let sc = er.mul_add(const_diff, CONST_MAX);
    sc * sc
}

/// Final KAMA state `[prev_kama, sum_roc1]` as of the last row: the KAMA value at `n−1`
/// and the sliding |1-bar change| sum for that row's window — the two scalars a
/// [`kama_resume`] needs to advance at `n`. `None` if KAMA never seeds (`period+1 > n`,
/// or a leading-NaN prefix leaves too few finite values → all-NaN, keep the fallback).
/// Runs [`kama`]'s exact sliding-sum recurrence so the captured `sum_roc1` matches.
pub fn kama_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let n = data.len();
    if period == 0 || period + 1 > n {
        return None;
    }
    // Mirror `kama`'s leading-NaN skip: seed over the first finite values. The carried
    // scalars are position-independent, so recursing on the finite tail is equivalent.
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        return kama_final_state(&data[start..], period);
    }
    let mut sum_roc1 = 0.0;
    for j in 0..period {
        sum_roc1 += (data[j] - data[j + 1]).abs();
    }
    let mut trailing_idx = 0usize;
    let mut today = period;
    let mut prev_kama = data[today - 1];
    let mut trailing_value = data[trailing_idx];
    let period_roc = data[today] - trailing_value;
    trailing_idx += 1;
    prev_kama = (data[today] - prev_kama).mul_add(kama_sc(period_roc, sum_roc1), prev_kama);
    today += 1;
    while today < n {
        let tr2 = data[trailing_idx];
        trailing_idx += 1;
        let period_roc = data[today] - tr2;
        sum_roc1 -= (trailing_value - tr2).abs();
        sum_roc1 += (data[today] - data[today - 1]).abs();
        trailing_value = tr2;
        prev_kama = (data[today] - prev_kama).mul_add(kama_sc(period_roc, sum_roc1), prev_kama);
        today += 1;
    }
    Some(vec![prev_kama, sum_roc1])
}

/// Resume [`kama`] from `state = [prev_kama_{from-1}, sum_roc1_{from-1}]` over rows
/// `[from, n)`, bit-identical to a full recompute. The sliding-sum recurrence reads the
/// trailing window `data[from-period-1 ..= from-1]` plus `data[from..]`; after a slice
/// those trailing rows are still present in the carried frame (close is carried verbatim),
/// so this reads nothing the frame lacks. Returns `None` (→ caller's full-recompute
/// fallback) when `from <= period` (the trailing window would underflow), which the slice
/// carry can produce at a very short retained head; that case is rare and stays correct
/// via the fallback.
pub fn kama_resume(data: &[f64], period: usize, from: usize, state: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = data.len();
    // Need `data[from-period-1]` (the oldest 1-bar change to drop) and `data[from-1]`.
    if period == 0 || from <= period || from > n {
        return None;
    }
    let mut prev_kama = state[0];
    let mut sum_roc1 = state[1];
    // `trailing_value` = data[from-period-1]; `trailing_idx` walks data[from-period..].
    let mut trailing_value = data[from - period - 1];
    let mut trailing_idx = from - period;
    let mut out = Vec::with_capacity(n - from);
    for today in from..n {
        let tr2 = data[trailing_idx];
        trailing_idx += 1;
        let period_roc = data[today] - tr2;
        sum_roc1 -= (trailing_value - tr2).abs();
        sum_roc1 += (data[today] - data[today - 1]).abs();
        trailing_value = tr2;
        prev_kama = (data[today] - prev_kama).mul_add(kama_sc(period_roc, sum_roc1), prev_kama);
        out.push(prev_kama);
    }
    Some((out, vec![prev_kama, sum_roc1]))
}

/// Parabolic SAR (TA-Lib SAR). Faithful port of TA-Lib's recurrence: initial trend is
/// chosen from the first bar's −DM1; each step trails the stop by `af·(ep − sar)`,
/// `af` ramping by `acceleration` (capped at `maximum`) on every new extreme, resetting
/// on a reversal; the SAR is clamped within the prior two bars' range. Default
/// acceleration 0.02, maximum 0.2; lookback 1. (TA-Lib applies no rounding.)
pub fn sar(high: &[f64], low: &[f64], acceleration: f64, maximum: f64) -> Vec<f64> {
    let n = high.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 {
        return out;
    }
    let af_init = acceleration.min(maximum); // TA-Lib clamps the step to the cap
                                             // Initial direction from the one-period −DM at bar 1: a positive −DM ⇒ short.
    let diff_p = high[1] - high[0];
    let diff_m = low[0] - low[1];
    let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m {
        diff_m
    } else {
        0.0
    };
    let mut is_long = !(minus_dm1 > 0.0);

    let mut af = af_init;
    let (mut ep, mut sar) = if is_long {
        (high[1], low[0])
    } else {
        (low[1], high[0])
    };
    // "Cheat" the first iteration: prime new high/low with bar 1 (as TA-Lib does).
    let mut new_low = low[1];
    let mut new_high = high[1];

    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                // Reverse to short: stop becomes the extreme point, clamped up.
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                out[today] = sar;
                af = af_init;
                ep = new_low;
                sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out[today] = sar;
                if new_high > ep {
                    ep = new_high;
                    af = (af + af_init).min(maximum);
                }
                sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            // Reverse to long: stop becomes the extreme point, clamped down.
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            out[today] = sar;
            af = af_init;
            ep = new_high;
            sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out[today] = sar;
            if new_low < ep {
                ep = new_low;
                af = (af + af_init).min(maximum);
            }
            sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    out
}

// SAR state-carry: the whole history compresses into the recurrence's loop state as of the
// last valid bar — `[is_long, af, ep, sar, prev_high, prev_low]`, where `prev_high`/`prev_low`
// are bar `from-1`'s high/low (the `new_high`/`new_low` that become `prev_*` at the next bar).
// The step reads only `high/low[from..]` plus this state, so a resume never indexes before
// `from` — sound after a head-dropping slice. A resume at `from < 2` (the SAR bootstrap reads
// bars 0 and 1) returns `None` and falls back; `sar_final_state` likewise returns `None` when
// `n < 2` (the column is all-NaN, no state to carry).

/// Final SAR state `[is_long, af, ep, sar, prev_high, prev_low]` after a full [`sar`] compute,
/// or `None` when `n < 2` (SAR never produces a value). Replays [`sar`]'s exact recurrence and
/// captures the loop variables as of the last bar (`n-1`) — i.e. the entering state for bar `n`.
pub fn sar_final_state(high: &[f64], low: &[f64], acceleration: f64, maximum: f64) -> Option<Vec<f64>> {
    let n = high.len();
    if n < 2 {
        return None;
    }
    let af_init = acceleration.min(maximum);
    let diff_p = high[1] - high[0];
    let diff_m = low[0] - low[1];
    let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m { diff_m } else { 0.0 };
    let mut is_long = !(minus_dm1 > 0.0);
    let mut af = af_init;
    let (mut ep, mut sar) = if is_long { (high[1], low[0]) } else { (low[1], high[0]) };
    let mut new_low = low[1];
    let mut new_high = high[1];
    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                af = af_init;
                ep = new_low;
                sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
            } else {
                if new_high > ep {
                    ep = new_high;
                    af = (af + af_init).min(maximum);
                }
                sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            af = af_init;
            ep = new_high;
            sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
        } else {
            if new_low < ep {
                ep = new_low;
                af = (af + af_init).min(maximum);
            }
            sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    Some(vec![is_long as u8 as f64, af, ep, sar, new_high, new_low])
}

/// Resume [`sar`] from `state = [is_long, af, ep, sar, prev_high, prev_low]` (as of row
/// `from - 1`) over rows `[from, n)`, bit-identical to a full recompute. `None` at `from < 2`
/// (the bootstrap needs bars 0 and 1, never re-run here). Reads only `high/low[from..]`; the
/// prior bar's extremes come from the carried state.
pub fn sar_resume(
    high: &[f64],
    low: &[f64],
    acceleration: f64,
    maximum: f64,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from < 2 {
        return None;
    }
    let n = high.len();
    let af_init = acceleration.min(maximum);
    let mut is_long = state[0] != 0.0;
    let mut af = state[1];
    let mut ep = state[2];
    let mut sar = state[3];
    let mut new_high = state[4];
    let mut new_low = state[5];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                out.push(sar);
                af = af_init;
                ep = new_low;
                sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out.push(sar);
                if new_high > ep {
                    ep = new_high;
                    af = (af + af_init).min(maximum);
                }
                sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            out.push(sar);
            af = af_init;
            ep = new_high;
            sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out.push(sar);
            if new_low < ep {
                ep = new_low;
                af = (af + af_init).min(maximum);
            }
            sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    Some((out, vec![is_long as u8 as f64, af, ep, sar, new_high, new_low]))
}

/// Parabolic SAR Extended (TA-Lib SAREXT). As [`sar`], but with separate long/short
/// acceleration (init/step/max), an optional start value (`>0` forces an initial long at
/// that level, `<0` an initial short at `|start|`, `0` = SAR's directional bootstrap), an
/// `offset_on_reverse` that nudges the stop on each reversal, and a **signed** output —
/// negative while short, positive while long — so reversals are visible. Lookback 1.
#[allow(clippy::too_many_arguments)]
pub fn sarext(
    high: &[f64],
    low: &[f64],
    start_value: f64,
    offset_on_reverse: f64,
    accel_init_long: f64,
    accel_long: f64,
    accel_max_long: f64,
    accel_init_short: f64,
    accel_short: f64,
    accel_max_short: f64,
) -> Vec<f64> {
    let n = high.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 {
        return out;
    }
    // TA-Lib clamps the init/step factors to their caps.
    let af_long_init = accel_init_long.min(accel_max_long);
    let af_short_init = accel_init_short.min(accel_max_short);
    let accel_long = accel_long.min(accel_max_long);
    let accel_short = accel_short.min(accel_max_short);

    // Initial direction: forced by a non-zero start value, else SAR's -DM1 bootstrap.
    let mut is_long = if start_value == 0.0 {
        let diff_p = high[1] - high[0];
        let diff_m = low[0] - low[1];
        let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m { diff_m } else { 0.0 };
        !(minus_dm1 > 0.0)
    } else {
        start_value > 0.0
    };

    let (mut ep, mut sar) = if start_value == 0.0 {
        if is_long { (high[1], low[0]) } else { (low[1], high[0]) }
    } else if start_value > 0.0 {
        (high[1], start_value)
    } else {
        (low[1], start_value.abs())
    };

    let (mut af_long, mut af_short) = (af_long_init, af_short_init);
    let mut new_low = low[1];
    let mut new_high = high[1];

    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                if offset_on_reverse != 0.0 {
                    sar += sar * offset_on_reverse;
                }
                out[today] = -sar;
                af_short = af_short_init;
                ep = new_low;
                sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out[today] = sar;
                if new_high > ep {
                    ep = new_high;
                    af_long = (af_long + accel_long).min(accel_max_long);
                }
                sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            if offset_on_reverse != 0.0 {
                sar -= sar * offset_on_reverse;
            }
            out[today] = sar;
            af_long = af_long_init;
            ep = new_high;
            sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out[today] = -sar;
            if new_low < ep {
                ep = new_low;
                af_short = (af_short + accel_short).min(accel_max_short);
            }
            sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    out
}

// SAREXT state-carry: like SAR, but carry both per-direction acceleration factors —
// `[is_long, af_long, af_short, ep, sar, prev_high, prev_low]`. The inactive direction's `af`
// is preserved across the active run (only ramped while in that direction, reset to its init on
// re-entry), so both must be carried for a bit-exact resume. `start_value` only steers the bar-1
// bootstrap, so it is irrelevant to a resume (`from >= 2`) and not threaded through `*_resume`;
// `offset_on_reverse` and the long/short accel step/cap still are. A resume at `from < 2` falls
// back (`None`), as does `*_final_state` when `n < 2`.

/// Final SAREXT state `[is_long, af_long, af_short, ep, sar, prev_high, prev_low]` after a full
/// [`sarext`] compute, or `None` when `n < 2`. Replays [`sarext`]'s exact recurrence and captures
/// the loop variables as of the last bar (`n-1`) — the entering state for bar `n`.
#[allow(clippy::too_many_arguments)]
pub fn sarext_final_state(
    high: &[f64],
    low: &[f64],
    start_value: f64,
    offset_on_reverse: f64,
    accel_init_long: f64,
    accel_long: f64,
    accel_max_long: f64,
    accel_init_short: f64,
    accel_short: f64,
    accel_max_short: f64,
) -> Option<Vec<f64>> {
    let n = high.len();
    if n < 2 {
        return None;
    }
    let af_long_init = accel_init_long.min(accel_max_long);
    let af_short_init = accel_init_short.min(accel_max_short);
    let accel_long = accel_long.min(accel_max_long);
    let accel_short = accel_short.min(accel_max_short);
    let mut is_long = if start_value == 0.0 {
        let diff_p = high[1] - high[0];
        let diff_m = low[0] - low[1];
        let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m { diff_m } else { 0.0 };
        !(minus_dm1 > 0.0)
    } else {
        start_value > 0.0
    };
    let (mut ep, mut sar) = if start_value == 0.0 {
        if is_long { (high[1], low[0]) } else { (low[1], high[0]) }
    } else if start_value > 0.0 {
        (high[1], start_value)
    } else {
        (low[1], start_value.abs())
    };
    let (mut af_long, mut af_short) = (af_long_init, af_short_init);
    let mut new_low = low[1];
    let mut new_high = high[1];
    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                if offset_on_reverse != 0.0 {
                    sar += sar * offset_on_reverse;
                }
                af_short = af_short_init;
                ep = new_low;
                sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
            } else {
                if new_high > ep {
                    ep = new_high;
                    af_long = (af_long + accel_long).min(accel_max_long);
                }
                sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            if offset_on_reverse != 0.0 {
                sar -= sar * offset_on_reverse;
            }
            af_long = af_long_init;
            ep = new_high;
            sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
        } else {
            if new_low < ep {
                ep = new_low;
                af_short = (af_short + accel_short).min(accel_max_short);
            }
            sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    Some(vec![is_long as u8 as f64, af_long, af_short, ep, sar, new_high, new_low])
}

/// Resume [`sarext`] from `state = [is_long, af_long, af_short, ep, sar, prev_high, prev_low]`
/// (as of row `from - 1`) over rows `[from, n)`, bit-identical to a full recompute. `None` at
/// `from < 2`. `start_value` is omitted (it only steers the bar-1 bootstrap, never re-run here).
/// Reads only `high/low[from..]`; the prior bar's extremes come from the carried state.
#[allow(clippy::too_many_arguments)]
pub fn sarext_resume(
    high: &[f64],
    low: &[f64],
    offset_on_reverse: f64,
    accel_init_long: f64,
    accel_long: f64,
    accel_max_long: f64,
    accel_init_short: f64,
    accel_short: f64,
    accel_max_short: f64,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from < 2 {
        return None;
    }
    let n = high.len();
    let af_long_init = accel_init_long.min(accel_max_long);
    let af_short_init = accel_init_short.min(accel_max_short);
    let accel_long = accel_long.min(accel_max_long);
    let accel_short = accel_short.min(accel_max_short);
    let mut is_long = state[0] != 0.0;
    let mut af_long = state[1];
    let mut af_short = state[2];
    let mut ep = state[3];
    let mut sar = state[4];
    let mut new_high = state[5];
    let mut new_low = state[6];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                if offset_on_reverse != 0.0 {
                    sar += sar * offset_on_reverse;
                }
                out.push(-sar);
                af_short = af_short_init;
                ep = new_low;
                sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out.push(sar);
                if new_high > ep {
                    ep = new_high;
                    af_long = (af_long + accel_long).min(accel_max_long);
                }
                sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            if offset_on_reverse != 0.0 {
                sar -= sar * offset_on_reverse;
            }
            out.push(sar);
            af_long = af_long_init;
            ep = new_high;
            sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out.push(-sar);
            if new_low < ep {
                ep = new_low;
                af_short = (af_short + accel_short).min(accel_max_short);
            }
            sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    Some((out, vec![is_long as u8 as f64, af_long, af_short, ep, sar, new_high, new_low]))
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

/// Bull and Bear Index (`mean of ma:a, ma:b, ma:c, ma:d`).
pub fn bbi(close: &[f64], a: usize, b: usize, c: usize, d: usize) -> Vec<f64> {
    let data = av(close);
    let ma_a = kernels::sma(data, a);
    let ma_b = kernels::sma(data, b);
    let ma_c = kernels::sma(data, c);
    let ma_d = kernels::sma(data, d);
    ((&ma_a + &ma_b + &ma_c + &ma_d) / 4.0).to_vec()
}

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
pub fn atr_final_state(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Option<Vec<f64>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::test_support::*;

    /// SAR resume, fed the carried `[is_long, af, ep, sar, prev_high, prev_low]` of a full
    /// compute over the head, reproduces the tail of a full compute over the whole input —
    /// bit-for-bit. Iterating `from` over an oscillating OHLC series places reversals on
    /// BOTH sides of the cut, firing the long->short (731-737) and short->long (746-752)
    /// reversal arms plus both trend-continue arms.
    #[test]
    fn sar_resume_is_bit_identical_to_full() {
        let (high, low, _close) = ohlc(160);
        let (accel, max) = (0.02, 0.2);
        let full = sar(&high, &low, accel, max);
        for &from in &[2usize, 3, 10, 40, 80, 120, 159] {
            let st = sar_final_state(&high[..from], &low[..from], accel, max).unwrap();
            let (tail, _) = sar_resume(&high, &low, accel, max, from, &st).unwrap();
            assert_bits(&tail, &full[from..], "sar");
        }
        // The oscillating series must actually reverse in both directions for the resume to
        // have exercised both reversal arms — confirm the full SAR sign flips at least once
        // each way (SAR itself is unsigned, so detect reversals via the value jumps).
        let mut saw_up = false;
        let mut saw_down = false;
        for w in full.windows(2) {
            if w[1] > w[0] {
                saw_up = true;
            }
            if w[1] < w[0] {
                saw_down = true;
            }
        }
        assert!(saw_up && saw_down, "fixture must swing both ways");
    }

    /// SAREXT resume (signed output) reproduces the full tail bit-for-bit, with a non-zero
    /// `offset_on_reverse` so the reversal-offset nudges (997-998 / 1015-1016) fire too.
    #[test]
    fn sarext_resume_is_bit_identical_to_full() {
        let (high, low, _close) = ohlc(160);
        // Asymmetric long/short acceleration + a reversal offset.
        let (offset, ail, al, aml, ais, as_, ams) = (0.1, 0.02, 0.02, 0.2, 0.03, 0.03, 0.25);
        for &start in &[0.0_f64, 1.0, -1.0] {
            let full = sarext(&high, &low, start, offset, ail, al, aml, ais, as_, ams);
            for &from in &[2usize, 5, 30, 70, 110, 159] {
                let st = sarext_final_state(
                    &high[..from], &low[..from], start, offset, ail, al, aml, ais, as_, ams,
                )
                .unwrap();
                let (tail, _) =
                    sarext_resume(&high, &low, offset, ail, al, aml, ais, as_, ams, from, &st)
                        .unwrap();
                assert_bits(&tail, &full[from..], "sarext");
            }
        }
    }

    /// SAREXT `*_final_state` bootstrap branches: a positive start forces an initial long at
    /// that level (900 / 904-905), a negative start an initial short at `|start|` (906-907),
    /// and a non-zero offset nudges the very first reversal (922 / 938).
    #[test]
    fn sarext_final_state_bootstrap_and_offset() {
        let (high, low, _close) = ohlc(80);
        let (al, aml, as_, ams) = (0.02, 0.2, 0.02, 0.2);
        // Exercise all three bootstrap arms of `*_final_state`: the SAR `-DM1` directional
        // bootstrap (start == 0, 902-903) and the forced long / short starts (start > 0 at
        // 904-905, start < 0 at 906-907). The first *computed* SAREXT value (`out[1]`) is
        // seeded directly from `sar`, so the three starts produce visibly different series.
        let bootstrap = |start: f64| -> (Vec<f64>, Vec<f64>) {
            let full = sarext(&high, &low, start, 0.0, al, al, aml, as_, as_, ams);
            let st = sarext_final_state(&high, &low, start, 0.0, al, al, aml, as_, as_, ams).unwrap();
            (full, st)
        };
        let (full_zero, _) = bootstrap(0.0);
        let (full_long, _) = bootstrap(5.0); // forced long: sar seeded from +start (904-905)
        let (full_short, _) = bootstrap(-5.0); // forced short: sar seeded from |start| (906-907)
        // The three distinct bootstrap arms seed distinct stops, so the first computed value
        // differs across all three (proving the forced-long and forced-short arms both ran,
        // not just the directional `start == 0` arm).
        assert_ne!(full_zero[1].to_bits(), full_long[1].to_bits(), "long bootstrap distinct");
        assert_ne!(full_zero[1].to_bits(), full_short[1].to_bits(), "short bootstrap distinct");
        assert_ne!(full_long[1].to_bits(), full_short[1].to_bits(), "long != short bootstrap");

        // With offset != 0 the carried `sar` differs from the offset == 0 run (the reversal
        // nudge at 922/938 changed it). A long enough oscillating series has reversed by n-1.
        let s_no_off = sarext_final_state(&high, &low, 0.0, 0.0, al, al, aml, as_, as_, ams).unwrap();
        let s_off = sarext_final_state(&high, &low, 0.0, 0.1, al, al, aml, as_, as_, ams).unwrap();
        assert_ne!(s_no_off[4].to_bits(), s_off[4].to_bits(), "offset must move the stop");
    }

    /// SAR / SAREXT warm-up guards: `*_final_state` declines for `n < 2` (no SAR value), and
    /// `*_resume` declines for `from < 2` (the bootstrap reads bars 0 and 1, never re-run).
    #[test]
    fn sar_guards_decline() {
        let one_h = [10.0];
        let one_l = [8.0];
        // n < 2 -> no state (654 / 888).
        assert!(sar_final_state(&one_h, &one_l, 0.02, 0.2).is_none());
        assert!(sarext_final_state(&one_h, &one_l, 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2).is_none());
        // from < 2 -> resume declines (714 / 973). State is unread on the None path.
        let st = vec![1.0, 0.02, 10.0, 8.0, 10.0, 8.0];
        let (h, l, _c) = ohlc(20);
        assert!(sar_resume(&h, &l, 0.02, 0.2, 1, &st).is_none());
        let st7 = vec![1.0, 0.02, 0.02, 10.0, 8.0, 10.0, 8.0];
        assert!(sarext_resume(&h, &l, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2, 1, &st7).is_none());
    }

    /// The remaining `*_final_state` / `*_resume` None guards in the MA-family helpers:
    /// `ema_seed_idx` period==0 (53), `kama_final_state` no-seed (500) + leading-NaN
    /// recursion (506), `kama_resume` underflow (545), and `atr_final_state` /
    /// `atr_resume` guards (1226 / 1253).
    #[test]
    fn ma_family_state_guards() {
        let data = series(60);
        // ema_seed_idx period == 0 -> ema_final_state returns None (trend.rs:53).
        assert!(ema_final_state(&data, 0).is_none());

        // kama_final_state: period == 0 / period + 1 > n -> None (trend.rs:500).
        assert!(kama_final_state(&data, 0).is_none());
        assert!(kama_final_state(&data[..5], 30).is_none());
        // kama_final_state leading-NaN prefix -> recurse on the finite tail (trend.rs:506).
        let mut nan_head = series(60);
        for x in nan_head.iter_mut().take(3) {
            *x = f64::NAN;
        }
        // A finite-tail KAMA state still computes; it must equal the state of the tail alone.
        let via_head = kama_final_state(&nan_head, 10).unwrap();
        let via_tail = kama_final_state(&nan_head[3..], 10).unwrap();
        assert_eq!(via_head[0].to_bits(), via_tail[0].to_bits(), "kama leading-NaN recursion");
        assert_eq!(via_head[1].to_bits(), via_tail[1].to_bits());

        // kama_resume underflow: period == 0 / from <= period / from > n -> None (trend.rs:545).
        let st = kama_final_state(&data, 10).unwrap();
        assert!(kama_resume(&data, 0, 20, &st).is_none());
        assert!(kama_resume(&data, 10, 10, &st).is_none()); // from <= period
        assert!(kama_resume(&data, 10, 1000, &st).is_none()); // from > n

        // atr_final_state: period == 0 / period >= n -> None (trend.rs:1226).
        let (h, l, c) = ohlc(60);
        assert!(atr_final_state(&h, &l, &c, 0).is_none());
        assert!(atr_final_state(&h, &l, &c, 100).is_none());
        // atr_resume: from == 0 -> None (trend.rs:1253).
        let st = atr_final_state(&h, &l, &c, 14).unwrap();
        assert!(atr_resume(&h, &l, &c, 14, 0, &st).is_none());
    }
}
