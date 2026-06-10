//! Moving averages: SMA / EMA / SMMA / WMA / BBI, with the EMA-family
//! state-carry helpers (`ema_k` / `ema_seed_idx` shared with the cascade and
//! MACD families).

use crate::indicators::av;
use crate::kernels;

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
pub(super) fn ema_k(period: usize) -> f64 {
    2.0 / (period as f64 + 1.0)
}

/// Index of the EMA seed (the `period`-th finite value of `data`), or `None` if fewer
/// than `period` finite values exist. Mirrors `kernels::sma_seeded`'s seeding scan, so a
/// captured state corresponds to the same seed the full kernel used.
pub(super) fn ema_seed_idx(data: &[f64], period: usize) -> Option<usize> {
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

/// Bull and Bear Index (`mean of ma:a, ma:b, ma:c, ma:d`).
pub fn bbi(close: &[f64], a: usize, b: usize, c: usize, d: usize) -> Vec<f64> {
    let data = av(close);
    let ma_a = kernels::sma(data, a);
    let ma_b = kernels::sma(data, b);
    let ma_c = kernels::sma(data, c);
    let ma_d = kernels::sma(data, d);
    ((&ma_a + &ma_b + &ma_c + &ma_d) / 4.0).to_vec()
}
