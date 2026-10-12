//! Cascaded EMA trend MAs: DEMA / TEMA / TRIMA / T3, sharing the staggered
//! [`kernels::cascade_warmup`] and the cascade lattice resume.

use crate::accum::Csum;
use crate::guard::{ring_push, Slide};
use crate::raw::kernels;
use crate::raw::smoothing::{self, ema_k, Weighted};

/// Double EMA: `2*EMA - EMA(EMA)` (TA-Lib DEMA). Lookback `2*(period-1)`. Single-pass
/// lattice over the two cascaded EMAs (vs two `ema_seeded` passes + a combine);
/// bit-identical to `2·e1 − e2`.
pub fn dema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let k = ema_k(period);
    let step = |prev: f64, x: f64| smoothing::ema(prev, x, k);
    let Some((lookback, e)) = kernels::cascade_warmup::<2>(data, period, step) else {
        return out;
    };
    out[lookback] = 2.0 * e[0] - e[1];
    let (mut e0, mut e1) = (e[0], e[1]);
    for i in (lookback + 1)..n {
        e0 = step(e0, data[i]);
        e1 = step(e1, e0);
        out[i] = 2.0 * e0 - e1;
    }
    out
}

/// Triple EMA: `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))` (TA-Lib TEMA). Lookback
/// `3*(period-1)`. Single-pass lattice over the three cascaded EMAs; bit-identical to
/// TA-Lib's `e3 + ((3·e1) − (3·e2))` (the sum commutes).
pub fn tema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let k = ema_k(period);
    let step = |prev: f64, x: f64| smoothing::ema(prev, x, k);
    let Some((lookback, e)) = kernels::cascade_warmup::<3>(data, period, step) else {
        return out;
    };
    out[lookback] = 3.0 * e[0] - 3.0 * e[1] + e[2];
    let (mut e0, mut e1, mut e2) = (e[0], e[1], e[2]);
    for i in (lookback + 1)..n {
        e0 = step(e0, data[i]);
        e1 = step(e1, e0);
        e2 = step(e2, e1);
        out[i] = 3.0 * e0 - 3.0 * e1 + e2;
    }
    out
}

/// Triangular moving average (TA-Lib TRIMA): a double SMA whose net weights rise to
/// the window centre then fall. Since convolution commutes, the two SMA windows are
/// `((n+1)/2, (n+1)/2)` for odd `n` and `(n/2, n/2+1)` for even `n` (either order).
/// Row 0 is at absolute position `origin`. Lookback `period-1`.
pub fn trima(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    // Skip a leading-NaN warm-up prefix (see `wma`).
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        let sub = trima(&data[start..], period, origin + start);
        out[start..].copy_from_slice(&sub);
        return out;
    }
    let (a, b) = trima_windows(period);
    // Fuse the two SMA convolutions into one pass (no intermediate `inner` array): a
    // running `sum1` slides the inner a-window, each inner mean feeds `sum2` — the
    // outer b-window sum over a ring of the last `b` inner means. Both follow `sma`'s
    // fast path exactly — add then subtract, re-summed where inexact — and
    // the outer slide begins at the first finite inner mean (index a-1, `sma`'s
    // leading-NaN skip), so this is bit-identical to `sma(sma(data, a), b)`.
    let af = a as f64;
    let bf = b as f64;
    let mut sum1 = Csum::default();
    let mut sum2 = Csum::default();
    let mut clean = true;
    let mut ring = vec![0.0; b];
    let mut slot = 0usize;
    let mut inner_cnt = 0usize;
    for i in 0..n {
        sum1.add(data[i]);
        if i >= a {
            sum1.sub(data[i - a]);
        }
        if !sum1.is_exact() && i + 1 >= a {
            clean &= sum1.value().is_finite();
            if clean {
                sum1 = Csum::of(data[i + 1 - a..=i].iter().copied());
            }
        }
        if i + 1 >= a {
            let inner = sum1.mean(af);
            sum2.add(inner);
            if inner_cnt >= b {
                sum2.sub(ring[slot]); // the inner mean leaving the outer window
            }
            ring[slot] = inner;
            slot += 1;
            if slot == b {
                slot = 0;
            }
            inner_cnt += 1;
            if !sum2.is_exact() && i + 1 >= b {
                clean &= sum2.value().is_finite();
                if clean {
                    // the window's inner means, oldest first (fewer than `b` at the start)
                    let m = inner_cnt.min(b);
                    sum2 = Csum::of((0..m).map(|k| ring[(slot + b - m + k) % b]));
                }
            }
            if inner_cnt >= b {
                out[i] = sum2.mean(bf);
            }
        }
    }
    if clean && sum1.value().is_finite() && sum2.value().is_finite() {
        return out;
    }
    // A non-finite row entered: fall back to the reference composition this fused pass
    // is bit-identical to. Both SMAs are window-local (see `crate::guard`), so a bad
    // row reaches only the windows holding it.
    let ma = crate::raw::indicators::trend::ma::ma;
    ma(&ma(data, a, origin), b, origin)
}

/// T3's arithmetic (TA-Lib `ta_T3.c`): the weighted EMA form `(k·x) + ((1 − k)·e)` of
/// its six stages ([`Weighted`] — T3's own form, not `ta_EMA.c`'s) and the
/// `vfactor`-derived combine coefficients `c1=-v³`, `c2=3(v²−c1)`, `c3=-6v²−3(v−c1)`,
/// `c4=1+3v−c1+3v²`, computed in TA-Lib's float order. The one owner for the full
/// computation, its captured state and its resumes.
#[derive(Clone, Copy)]
struct T3 {
    stage: Weighted,
    c: [f64; 4],
}

impl T3 {
    fn new(period: usize, vfactor: f64) -> Self {
        let v2 = vfactor * vfactor;
        let c1 = -(v2 * vfactor);
        let c2 = 3.0 * (v2 - c1);
        let c3 = -6.0 * v2 - 3.0 * (vfactor - c1);
        let c4 = 1.0 + 3.0 * vfactor - c1 + 3.0 * v2;
        T3 { stage: t3_stage(period), c: [c1, c2, c3, c4] }
    }

    /// `c1·e6 + c2·e5 + c3·e4 + c4·e3` over the stages `e = [e1..e6]`.
    #[inline(always)]
    fn combine(&self, e: &[f64; 6]) -> f64 {
        let [c1, c2, c3, c4] = self.c;
        c1 * e[5] + c2 * e[4] + c3 * e[3] + c4 * e[2]
    }

    /// Advance the six stages by `x` and combine them.
    #[inline(always)]
    fn advance(&self, e: &mut [f64; 6], x: f64) -> f64 {
        let stage = self.stage;
        kernels::cascade_step(e, x, move |prev: f64, x: f64| stage.step(prev, x));
        self.combine(e)
    }
}

/// The step of each T3 stage over `period`: `k = 2/(period + 1)` in the weighted form.
fn t3_stage(period: usize) -> Weighted {
    Weighted::new(ema_k(period))
}

/// Tillson T3 (TA-Lib T3): `c1·e6 + c2·e5 + c3·e4 + c4·e3` over six cascaded
/// SMA-seeded EMAs in TA-Lib's arithmetic ([`T3`]). Default period 5, vfactor 0.7;
/// lookback `6·(period-1)`.
pub fn t3(data: &[f64], period: usize, vfactor: f64) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    let t3 = T3::new(period, vfactor);
    let stage = t3.stage;
    // Single-pass lattice instead of six sequential passes: the six stages cascade
    // (stage j consumes stage j-1's *current* output) so they share one
    // staggered-warmup traversal ([`kernels::cascade_warmup`]); the steady state then
    // advances the six-deep chain per bar (TA-Lib's lattice) and emits the combine in
    // place.
    let Some((lookback, mut e)) = kernels::cascade_warmup::<6>(data, period, |p, x| stage.step(p, x)) else {
        return out;
    };
    out[lookback] = t3.combine(&e);
    for i in (lookback + 1)..n {
        out[i] = t3.advance(&mut e, data[i]);
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
/// Appends rows `[from, n)` to `out` and returns the carried state.
pub fn dema_resume(
    data: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
    out: &mut Vec<f64>,
) -> Vec<f64> {
    let n = data.len();
    let mut e = [state[0], state[1]];
    out.reserve(n.saturating_sub(from));
    for &x in &data[from..n] {
        kernels::ema_cascade_step(&mut e, x, period);
        out.push(2.0 * e[0] - e[1]);
    }
    e.to_vec()
}

/// Final TEMA state `[e0, e1, e2]`, or `None` if unseeded. Pairs with [`tema_resume`].
pub fn tema_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let e = kernels::ema_cascade_final::<3>(data, period)?;
    Some(e.to_vec())
}

/// Resume [`tema`] from `state = [e0, e1, e2]` over rows `[from, n)`, bit-identical to a
/// full recompute (the `3·e0 − 3·e1 + e2` lattice). Reads only `data[from..]`.
/// Appends rows `[from, n)` to `out` and returns the carried state.
pub fn tema_resume(
    data: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
    out: &mut Vec<f64>,
) -> Vec<f64> {
    let n = data.len();
    let mut e = [state[0], state[1], state[2]];
    out.reserve(n.saturating_sub(from));
    for &x in &data[from..n] {
        kernels::ema_cascade_step(&mut e, x, period);
        out.push(3.0 * e[0] - 3.0 * e[1] + e[2]);
    }
    e.to_vec()
}

/// Final T3 state `[e0..e5]`, or `None` if unseeded. Pairs with [`t3_resume`].
pub fn t3_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let stage = t3_stage(period);
    let e = kernels::cascade_final::<6>(data, period, move |p: f64, x: f64| stage.step(p, x))?;
    Some(e.to_vec())
}

/// Resume [`t3`] from `state = [e0..e5]` over rows `[from, n)`, bit-identical to a full
/// recompute (the same six-deep lattice and combine, [`T3`]). Reads only `data[from..]`.
/// Appends rows `[from, n)` to `out` and returns the carried state.
pub fn t3_resume(
    data: &[f64],
    period: usize,
    vfactor: f64,
    from: usize,
    state: &[f64],
    out: &mut Vec<f64>,
) -> Vec<f64> {
    let t3 = T3::new(period, vfactor);
    let n = data.len();
    let mut e = [state[0], state[1], state[2], state[3], state[4], state[5]];
    out.reserve(n.saturating_sub(from));
    out.extend(data[from..n].iter().map(|&x| t3.advance(&mut e, x)));
    e.to_vec()
}


/// Scalar single-row twin of [`dema_resume`]: the value at `row` from `state = [e0, e1]`,
/// zero-alloc and bit-identical to one Vec-kernel iteration (same `ema_cascade_step`
/// lattice + `2·e0 − e1` combine). The carried `state` is as-of `row-1`; the cascade as
/// of `row` is written into `next`.
pub fn dema_resume_one(
    data: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let (Ok(st), Ok(e)) = (<[f64; 2]>::try_from(state), <&mut [f64; 2]>::try_from(next)) else {
        return None;
    };
    let x = *data.get(row)?;
    *e = st;
    kernels::ema_cascade_step(e, x, period);
    Some(2.0 * e[0] - e[1])
}

/// Scalar single-row twin of [`tema_resume`]: the value at `row` from `state = [e0, e1, e2]`,
/// zero-alloc and bit-identical to one Vec-kernel iteration (same `ema_cascade_step`
/// lattice + `3·e0 − 3·e1 + e2` combine). The carried `state` is as-of `row-1`; the
/// cascade as of `row` is written into `next`.
pub fn tema_resume_one(
    data: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let (Ok(st), Ok(e)) = (<[f64; 3]>::try_from(state), <&mut [f64; 3]>::try_from(next)) else {
        return None;
    };
    let x = *data.get(row)?;
    *e = st;
    kernels::ema_cascade_step(e, x, period);
    Some(3.0 * e[0] - 3.0 * e[1] + e[2])
}

/// Scalar single-row twin of [`t3_resume`]: the value at `row` from `state = [e0..e5]`,
/// zero-alloc and bit-identical to one Vec-kernel iteration (the same [`T3`] lattice
/// step and combine). State is as-of `row-1`; the cascade as of `row` is written into
/// `next`.
pub fn t3_resume_one(
    data: &[f64],
    period: usize,
    vfactor: f64,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let (Ok(st), Ok(e)) = (<[f64; 6]>::try_from(state), <&mut [f64; 6]>::try_from(next)) else {
        return None;
    };
    let x = *data.get(row)?;
    *e = st;
    Some(T3::new(period, vfactor).advance(e, x))
}


/// TRIMA's two SMA windows `(a, b)` for `period` (see [`trima`]).
fn trima_windows(period: usize) -> (usize, usize) {
    if period % 2 == 1 {
        let m = period.div_ceil(2);
        (m, m)
    } else {
        let m = period / 2;
        (m, m + 1)
    }
}

/// How many input rows before the current one a [`trima`] step reads (the inner SMA's).
pub(crate) fn trima_reach(period: usize) -> usize {
    trima_windows(period).0
}

/// The length of [`trima`]'s resumable state: the inner and outer SMAs' slides, then the
/// last `b` inner means the outer one reads.
pub(crate) fn trima_state_len(period: usize) -> usize {
    2 * kernels::SMA_STATE + trima_windows(period).1
}

/// [`trima`]'s state before its first row, at absolute position `origin`.
pub(crate) fn trima_initial(period: usize, origin: usize) -> Vec<f64> {
    let mut v = Slide::<1>::initial(origin).repeat(2);
    v.resize(trima_state_len(period), 0.0);
    v
}

/// One row of [`trima`] over the series `term` at `row` — `sma(sma(term, a), b)`, which
/// the fused pass is bit-identical to — from `state` into `next`.
pub(crate) fn trima_row(
    period: usize,
    row: usize,
    term: &impl Fn(usize) -> f64,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    const S: usize = kernels::SMA_STATE;
    let (a, b) = trima_windows(period);
    if period == 0 || state.len() != 2 * S + b || next.len() != state.len() {
        return None;
    }
    let (s_in, rest) = state.split_at(S);
    let (s_out, ring) = rest.split_at(S);
    let (n_in, nrest) = next.split_at_mut(S);
    let (n_out, n_ring) = nrest.split_at_mut(S);
    let inner = kernels::sma_step(a, row, term, s_in, n_in)?;
    // the inner mean at row j: `ring` holds rows `row-b .. row-1`, oldest first
    let means = |j: usize| if j == row { inner } else { ring[b - (row - j)] };
    let value = kernels::sma_step(b, row, means, s_out, n_out)?;
    ring_push(ring, inner, n_ring);
    Some(value)
}

/// [`trima`]'s state after the rows of `data` (row 0 at absolute position `origin`).
pub fn trima_state(data: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    super::FiniteMa::new(period, 5)?.state(data.len(), origin, |j| data[j])
}

/// One row of [`trima`] at `row`, from `state` into `next`.
pub fn trima_step(data: &[f64], period: usize, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    if row >= data.len() {
        return None;
    }
    super::FiniteMa::new(period, 5)?.step(row, |j| data[j], state, next)
}
