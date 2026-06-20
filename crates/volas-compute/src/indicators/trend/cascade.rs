//! Cascaded EMA trend MAs: DEMA / TEMA / TRIMA / T3, sharing the staggered
//! `cascade_warmup` and the EMA lattice resume.

use crate::indicators::trend::ma::ema_k;
use crate::kernels;

/// Staggered warm-up for a cascade of `S` SMA-seeded EMAs (`k = 2/(period+1)`): each
/// stage SMA-seeds over the first `period` finite values of its predecessor, so stage
/// `j` seeds at `j·(period-1)`. Returns `(lookback, stage values at lookback)` where
/// `lookback = S·(period-1)` is the first fully-valid bar, or `None` if it never seeds.
/// Shared by DEMA/TEMA/T3 — their combine and steady-state recurrence differ, but the
/// warm-up is identical (and bit-identical to chaining `ema_seeded` `S` times).
fn cascade_warmup<const S: usize>(
    data: &[f64],
    period: usize,
    k: f64,
) -> Option<(usize, [f64; S])> {
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
        let m = period.div_ceil(2);
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
    let (mut e0, mut e1, mut e2, mut e3, mut e4, mut e5) = (e[0], e[1], e[2], e[3], e[4], e[5]);
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
pub fn dema_resume(
    data: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
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
pub fn tema_resume(
    data: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
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


/// Scalar single-row twin of [`dema_resume`]: the value at `row` from `state = [e0, e1]`,
/// zero-alloc and bit-identical to one Vec-kernel iteration (same `ema_cascade_step`
/// lattice + `2·e0 − e1` combine). The carried `state` is as-of `row-1`.
pub fn dema_resume_one(data: &[f64], period: usize, row: usize, state: &[f64]) -> Option<f64> {
    if state.len() < 2 || row >= data.len() {
        return None;
    }
    let k = ema_k(period);
    let mut e = [state[0], state[1]];
    kernels::ema_cascade_step(&mut e, data[row], k);
    Some(2.0 * e[0] - e[1])
}

/// Scalar single-row twin of [`tema_resume`]: the value at `row` from `state = [e0, e1, e2]`,
/// zero-alloc and bit-identical to one Vec-kernel iteration (same `ema_cascade_step`
/// lattice + `3·e0 − 3·e1 + e2` combine). The carried `state` is as-of `row-1`.
pub fn tema_resume_one(data: &[f64], period: usize, row: usize, state: &[f64]) -> Option<f64> {
    if state.len() < 3 || row >= data.len() {
        return None;
    }
    let k = ema_k(period);
    let mut e = [state[0], state[1], state[2]];
    kernels::ema_cascade_step(&mut e, data[row], k);
    Some(3.0 * e[0] - 3.0 * e[1] + e[2])
}

/// Scalar single-row twin of [`t3_resume`]: the value at `row` from `state = [e0..e5]`,
/// zero-alloc and bit-identical to one Vec-kernel iteration (same six-deep
/// `ema_cascade_step` lattice + `c1·e5 + c2·e4 + c3·e3 + c4·e2` combine, with the
/// `vfactor`-derived coefficients in TA-Lib's exact float order). State is as-of `row-1`.
pub fn t3_resume_one(
    data: &[f64],
    period: usize,
    vfactor: f64,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if state.len() < 6 || row >= data.len() {
        return None;
    }
    let v2 = vfactor * vfactor;
    let c1 = -(v2 * vfactor);
    let c2 = 3.0 * (v2 - c1);
    let c3 = -6.0 * v2 - 3.0 * (vfactor - c1);
    let c4 = 1.0 + 3.0 * vfactor - c1 + 3.0 * v2;
    let k = ema_k(period);
    let mut e = [
        state[0], state[1], state[2], state[3], state[4], state[5],
    ];
    kernels::ema_cascade_step(&mut e, data[row], k);
    Some(c1 * e[5] + c2 * e[4] + c3 * e[3] + c4 * e[2])
}

