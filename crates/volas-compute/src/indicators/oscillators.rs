use ndarray::Array1;

use super::av;
use crate::kernels;

// ---------------------------------------------------------------------------
// Overbought / oversold
// ---------------------------------------------------------------------------

/// Lowest of low values.
pub fn llv(data: &[f64], period: usize) -> Vec<f64> {
    // Move the kernel's owned buffer out (no copy) rather than `to_vec`.
    kernels::rolling_min(av(data), period)
        .into_raw_vec_and_offset()
        .0
}

/// Highest of high values.
pub fn hhv(data: &[f64], period: usize) -> Vec<f64> {
    kernels::rolling_max(av(data), period)
        .into_raw_vec_and_offset()
        .0
}

/// Raw Stochastic Value.
pub fn rsv(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let llv = kernels::rolling_min(av(low), period);
    let hhv = kernels::rolling_max(av(high), period);
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        let denom = hhv[i] - llv[i];
        if denom.abs() > 1e-10 {
            result[i] = (close[i] - llv[i]) / denom * 100.0;
        } else {
            result[i] = 0.0;
        }
    }
    result
}

/// TA-Lib stochastic raw %K: `100·(close − LL) / (HH − LL)` over `period` (HH/LL the
/// highest high / lowest low). A flat range yields 0, but — unlike [`rsv`] — the
/// warm-up is NaN, so the smoothing MAs in `stoch`/`stochf` begin at the right row.
/// Lookback `period-1`.
pub fn stoch_fastk(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    // One fused pass for max(high) and min(low) instead of two separate van Herk sweeps.
    let (hh, ll) = kernels::rolling_max_min(high, low, period);
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if !hh[i].is_nan() {
            let diff = hh[i] - ll[i];
            out[i] = if diff != 0.0 {
                100.0 * (close[i] - ll[i]) / diff
            } else {
                0.0
            };
        }
    }
    out
}

/// TA-Lib StochRSI raw %K: the stochastic %K of the RSI line — `stoch_fastk` applied
/// to `rsi(close, rsi_period)` over `fastk_period`. Because RSI itself warms up, the
/// %K is masked to NaN until a full `fastk_period` window of finite RSI is available
/// (index `rsi_period + fastk_period - 1`). The `fastd` line is then `ma_typed` of this.
pub fn stochrsi_fastk(close: &[f64], rsi_period: usize, fastk_period: usize) -> Vec<f64> {
    let rsi = rsi(close, rsi_period);
    let n = rsi.len();
    let mut fk = vec![f64::NAN; n];
    // RSI warms up with a leading-NaN prefix, so a rolling min/max over the whole line
    // fails the NaN-free check and drops to the slow monotonic-deque path — StochRSI's
    // hot spot. Run the stochastic %K on RSI's *finite tail* instead (NaN-free → the
    // van Herk fast path) and place it back at `start`. The first valid value then
    // lands at the first fully-finite RSI window — exactly the rows the old explicit
    // mask kept — and the result is bit-identical (min/max over a finite window is
    // order-independent and exact), so the separate masking pass is no longer needed.
    let start = rsi.iter().position(|x| !x.is_nan()).unwrap_or(n);
    let tail = &rsi[start..];
    if tail.len() >= fastk_period {
        let fk_tail = stoch_fastk(tail, tail, tail, fastk_period);
        fk[start..].copy_from_slice(&fk_tail);
    }
    fk
}

fn kdj_rsv(high: &[f64], low: &[f64], close: &[f64], period_rsv: usize) -> Array1<f64> {
    let llv = kernels::rolling_min(av(low), period_rsv);
    let hhv = kernels::rolling_max(av(high), period_rsv);
    let n = close.len();
    let mut rsv = Array1::from_elem(n, 0.0);
    for i in 0..n {
        let denom = hhv[i] - llv[i];
        if denom.abs() > 1e-10 {
            rsv[i] = (close[i] - llv[i]) / denom * 100.0;
        }
    }
    rsv
}

/// KDJ %K line.
pub fn kdj_k(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    init: f64,
) -> Vec<f64> {
    let rsv = kdj_rsv(high, low, close, period_rsv);
    kernels::ewma_with_init(rsv.view(), period_k, init).to_vec()
}

/// KDJ %D line.
pub fn kdj_d(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    init: f64,
) -> Vec<f64> {
    let rsv = kdj_rsv(high, low, close, period_rsv);
    let k = kernels::ewma_with_init(rsv.view(), period_k, init);
    kernels::ewma_with_init(k.view(), period_d, init).to_vec()
}

/// KDJ %J line (`3K - 2D`).
pub fn kdj_j(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    init: f64,
) -> Vec<f64> {
    let rsv = kdj_rsv(high, low, close, period_rsv);
    let k = kernels::ewma_with_init(rsv.view(), period_k, init);
    let d = kernels::ewma_with_init(k.view(), period_d, init);
    (3.0 * &k - 2.0 * &d).to_vec()
}

/// SMA-seeded Wilder average gain and average loss of `data`'s bar-to-bar changes
/// — the shared core of RSI and CMO. Both outputs are NaN until the first smoothed
/// value (at index `period`).
fn wilder_gain_loss(data: &[f64], period: usize) -> (Array1<f64>, Array1<f64>) {
    let n = data.len();
    let delta = kernels::diff(av(data));
    let mut gains = Array1::from_elem(n, f64::NAN);
    let mut losses = Array1::from_elem(n, f64::NAN);
    for i in 1..n {
        let d = delta[i];
        if d.is_nan() {
            continue;
        }
        gains[i] = d.max(0.0);
        losses[i] = (-d).max(0.0);
    }
    (
        kernels::wilder(gains.view(), period),
        kernels::wilder(losses.view(), period),
    )
}

/// Relative Strength Index (TA-Lib RSI): `100·avgGain/(avgGain+avgLoss)`.
///
/// Single-pass / single-allocation: seed the Wilder average gain & loss as the SMA
/// of the first `period` bar-to-bar changes, emit from index `period`, then
/// Wilder-smooth in place. Bit-identical to `wilder(gains)` / `wilder(losses)` +
/// combine (same seed sum, same recursion, same flat-window guard) but ~one sixth
/// the memory traffic — no `diff` / `gains` / `losses` / two smoothed arrays.
pub fn rsi(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return out;
    }
    let pf = period as f64;
    let p1 = pf - 1.0;
    let emit = |g: f64, l: f64| {
        if l.abs() < 1e-10 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + g / l)
        }
    };
    // Seed: SMA of the first `period` gains / losses (deltas at indices 1..=period).
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in 1..=period {
        let d = close[i] - close[i - 1];
        if d > 0.0 {
            avg_gain += d;
        } else {
            avg_loss -= d;
        }
    }
    avg_gain /= pf;
    avg_loss /= pf;
    out[period] = emit(avg_gain, avg_loss);
    for i in (period + 1)..n {
        let d = close[i] - close[i - 1];
        let (gain, loss) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
        avg_gain = (avg_gain * p1 + gain) / pf;
        avg_loss = (avg_loss * p1 + loss) / pf;
        out[i] = emit(avg_gain, avg_loss);
    }
    out
}

/// Chande Momentum Oscillator (TA-Lib CMO): `100·(avgGain−avgLoss)/(avgGain+avgLoss)`
/// over the same Wilder-smoothed gains/losses as RSI; a flat window (gain+loss = 0)
/// yields 0. Lookback `period`. (Algebraically `2·RSI − 100`; computed directly so
/// the flat-window guard matches TA-Lib exactly rather than inheriting RSI's.)
pub fn cmo(close: &[f64], period: usize) -> Vec<f64> {
    let (sg, sl) = wilder_gain_loss(close, period);
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        if sg[i].is_nan() || sl[i].is_nan() {
            continue;
        }
        let denom = sg[i] + sl[i];
        result[i] = if denom < 1e-14 {
            0.0
        } else {
            100.0 * (sg[i] - sl[i]) / denom
        };
    }
    result
}

// --- RSI / CMO state-carry (additive; the full-recompute fallback stays correct) ---
//
// Both smooth bar-to-bar gains and losses with a Wilder average, so the carried state is
// the pair `[avg_gain, avg_loss]` as of row `from-1`. A delta needs the prior close, so a
// resume reads only `close[from-1..]`; `from == 0` returns `None` (falls back). NOTE the
// two use DIFFERENT recurrences — `rsi` divides each step (`(avg·(p-1)+x)/p`) while `cmo`
// goes through `kernels::wilder`'s fused `avg·a + x·b` — so each has its own kernel,
// bit-identical to its `pub fn`. `*_final_state` returns `None` before the seed (`n <=
// period`), keeping the fallback.

/// Final RSI state `[avg_gain, avg_loss]` after a full [`rsi`] compute, or `None` if it
/// never seeds (`period == 0 || n <= period`). Reproduces `rsi`'s exact seed (SMA of the
/// first `period` gains/losses, deltas `1..=period`) and `(avg·(p-1)+x)/p` recurrence.
pub fn rsi_final_state(close: &[f64], period: usize) -> Option<Vec<f64>> {
    let n = close.len();
    if period == 0 || n <= period {
        return None;
    }
    let pf = period as f64;
    let p1 = pf - 1.0;
    let (mut avg_gain, mut avg_loss) = (0.0, 0.0);
    for i in 1..=period {
        let d = close[i] - close[i - 1];
        if d > 0.0 {
            avg_gain += d;
        } else {
            avg_loss -= d;
        }
    }
    avg_gain /= pf;
    avg_loss /= pf;
    for i in (period + 1)..n {
        let d = close[i] - close[i - 1];
        let (gain, loss) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
        avg_gain = (avg_gain * p1 + gain) / pf;
        avg_loss = (avg_loss * p1 + loss) / pf;
    }
    Some(vec![avg_gain, avg_loss])
}

/// Resume [`rsi`] from `state = [avg_gain, avg_loss]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `close[from-1..]`. The recurrence and the
/// `100 - 100/(1 + g/l)` (flat-loss → 100) output match [`rsi`] bit-for-bit.
pub fn rsi_resume(close: &[f64], period: usize, from: usize, state: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    if from == 0 {
        return None;
    }
    let n = close.len();
    let pf = period as f64;
    let p1 = pf - 1.0;
    let emit = |g: f64, l: f64| {
        if l.abs() < 1e-10 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + g / l)
        }
    };
    let (mut avg_gain, mut avg_loss) = (state[0], state[1]);
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        let d = close[i] - close[i - 1];
        let (gain, loss) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
        avg_gain = (avg_gain * p1 + gain) / pf;
        avg_loss = (avg_loss * p1 + loss) / pf;
        out.push(emit(avg_gain, avg_loss));
    }
    Some((out, vec![avg_gain, avg_loss]))
}

/// Final CMO state `[avg_gain, avg_loss]` after a full [`cmo`] compute, or `None` if it
/// never seeds. CMO smooths via `kernels::wilder` (fused `avg·a + x·b`), seeded as the
/// SMA of the first `period` gains/losses — the same seed index as RSI but a different
/// recurrence, reproduced here exactly.
pub fn cmo_final_state(close: &[f64], period: usize) -> Option<Vec<f64>> {
    let n = close.len();
    if period == 0 || n <= period {
        return None;
    }
    let pf = period as f64;
    let (a, b) = ((pf - 1.0) / pf, 1.0 / pf);
    let (mut avg_gain, mut avg_loss) = (0.0, 0.0);
    for i in 1..=period {
        let d = close[i] - close[i - 1];
        avg_gain += d.max(0.0);
        avg_loss += (-d).max(0.0);
    }
    avg_gain /= pf;
    avg_loss /= pf;
    for i in (period + 1)..n {
        let d = close[i] - close[i - 1];
        avg_gain = avg_gain.mul_add(a, d.max(0.0) * b);
        avg_loss = avg_loss.mul_add(a, (-d).max(0.0) * b);
    }
    Some(vec![avg_gain, avg_loss])
}

/// Resume [`cmo`] from `state = [avg_gain, avg_loss]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `close[from-1..]`. The fused Wilder recurrence and the
/// `100·(g-l)/(g+l)` (flat-window → 0) output match [`cmo`] bit-for-bit.
pub fn cmo_resume(close: &[f64], period: usize, from: usize, state: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    if from == 0 {
        return None;
    }
    let n = close.len();
    let pf = period as f64;
    let (a, b) = ((pf - 1.0) / pf, 1.0 / pf);
    let (mut avg_gain, mut avg_loss) = (state[0], state[1]);
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        let d = close[i] - close[i - 1];
        avg_gain = avg_gain.mul_add(a, d.max(0.0) * b);
        avg_loss = avg_loss.mul_add(a, (-d).max(0.0) * b);
        let denom = avg_gain + avg_loss;
        out.push(if denom < 1e-14 {
            0.0
        } else {
            100.0 * (avg_gain - avg_loss) / denom
        });
    }
    Some((out, vec![avg_gain, avg_loss]))
}

/// Donchian middle channel (`(hhv + llv) / 2`).
pub fn donchian(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let hhv = kernels::rolling_max(av(high), period);
    let llv = kernels::rolling_min(av(low), period);
    ((&hhv + &llv) / 2.0).to_vec()
}

/// Midpoint over `period` of a single series: `(max + min) / 2` (TA-Lib MIDPOINT).
/// Lookback `period-1`.
pub fn midpoint(data: &[f64], period: usize) -> Vec<f64> {
    let hh = kernels::rolling_max(av(data), period);
    let ll = kernels::rolling_min(av(data), period);
    ((&hh + &ll) / 2.0).to_vec()
}

/// Midpoint price over `period`: `(max(high) + min(low)) / 2` (TA-Lib MIDPRICE).
/// Lookback `period-1`. (Same arithmetic as the Donchian middle channel.)
pub fn midprice(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let hh = kernels::rolling_max(av(high), period);
    let ll = kernels::rolling_min(av(low), period);
    ((&hh + &ll) / 2.0).to_vec()
}
