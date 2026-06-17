use ndarray::Array1;

use super::av;
use super::stochastic::stoch_fastk;
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
    // TA-Lib `MAX` is tuned around a track-and-rescan loop for small default periods.
    // For `hhv:10`, pre-scan once for NaN and then keep that C-shaped tracker across
    // all lengths; NaN-bearing data falls back to `rolling_max`'s precise semantics.
    if period == 10 && period <= data.len() && !data.iter().any(|x| x.is_nan()) {
        return hhv10_no_nan(data);
    }
    kernels::rolling_max(av(data), period)
        .into_raw_vec_and_offset()
        .0
}

fn hhv10_no_nan(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    // NaN-prefilled buffer: the warm-up stays NaN and the loop overwrites the
    // valid region (D2 2026-06-12 — replaces the with_capacity + set_len pattern;
    // the prefill is a vectorized splat, measured at parity by make perf-ab).
    let mut out = vec![f64::NAN; n];

    let src = data.as_ptr();
    let dst = out.as_mut_ptr();
    let mut highest_idx = 0usize;
    let mut highest = unsafe { *src };
    for i in 1..10 {
        let value = unsafe { *src.add(i) };
        if value > highest {
            highest_idx = i;
            highest = value;
        }
    }
    unsafe {
        *dst.add(9) = highest;
    }

    for today in 10..n {
        let trailing = today - 9;
        let x = unsafe { *src.add(today) };
        if highest_idx < trailing {
            highest_idx = trailing;
            highest = unsafe { *src.add(trailing) };
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *src.add(idx) };
                if value > highest {
                    highest_idx = idx;
                    highest = value;
                }
                idx += 1;
            }
        } else if x >= highest {
            highest_idx = today;
            highest = x;
        }
        unsafe {
            *dst.add(today) = highest;
        }
    }
    out
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

// --- StochRSI state-carry (additive; the full-recompute fallback stays correct) ---
//
// StochRSI is a composite: a windowed stochastic %K of the (Wilder-recursive) RSI line,
// with the `.d` line a further SMA of that %K. The %K / SMA stages are finite-memory
// (position-independent windowed min/max / mean over a NaN-free RSI buffer), so the ONLY
// recursive part is the underlying RSI. We continue it bit-exactly in O(new rows) by
// carrying the RSI Wilder pair `[avg_gain, avg_loss]` as of `from-1` PLUS the recent RSI
// VALUES needed to fill the stochastic (and SMA) windows over the new rows:
//   state = [avg_gain, avg_loss, rsi_{from-C}, …, rsi_{from-1}]
// where the RSI-context depth `C` is the stage lookback before `from`:
//   `.k` -> C = fastk_period - 1                       (the %K window)
//   `.d` -> C = (fastd_period - 1) + (fastk_period - 1) (the SMA-of-%K reach)
// On resume we `rsi_resume` the new RSI tail, concatenate the carried context, run the
// (NaN-free) windowed %K — and, for `.d`, the windowed SMA — then slice out `[from, n)`.
// Bit-identical to a fresh `stochrsi_fastk` (+ `ma`), since every windowed reduction over
// a finite window is order/position-independent. A resume that cannot see a full context
// of FINITE RSI (`from - C < rsi_period`, i.e. the tracker is not yet warm) returns `None`
// and falls back; a carried slice keeps `>= lookback` rows, so it is always warm enough.
// Only the canonical SMA `.d` (matype 0) is resumed; a recursive-MA `.d` declines.

/// RSI-context depth carried before `from` for a StochRSI resume (see the module note):
/// the %K window for `.k`, plus the SMA-of-%K reach for `.d`.
fn stochrsi_ctx_depth(fastk_period: usize, is_d: bool, fastd_period: usize) -> usize {
    let k = fastk_period.saturating_sub(1);
    if is_d {
        k + fastd_period.saturating_sub(1)
    } else {
        k
    }
}

/// Final StochRSI state `[avg_gain, avg_loss, rsi_tail…]` after a full compute, or `None`
/// if RSI never warms up (`rsi_period == 0 || n <= rsi_period`) or there are not yet `C+1`
/// finite RSI rows to anchor a resume. `is_d` / `fastd_period` only size the carried RSI
/// tail (the deeper SMA reach). The Wilder pair matches [`rsi_final_state`] exactly.
pub fn stochrsi_final_state(
    close: &[f64],
    rsi_period: usize,
    fastk_period: usize,
    is_d: bool,
    fastd_period: usize,
) -> Option<Vec<f64>> {
    let n = close.len();
    let wilder = rsi_final_state(close, rsi_period)?; // [avg_gain, avg_loss] as of n-1
    let c = stochrsi_ctx_depth(fastk_period, is_d, fastd_period);
    // Need `c` RSI values ending at n-1, all finite (RSI is finite from row `rsi_period`).
    if n < c || n - c < rsi_period {
        return None;
    }
    let rsi = rsi(close, rsi_period);
    let mut state = wilder;
    state.extend_from_slice(&rsi[n - c..n]);
    Some(state)
}

/// Resume StochRSI `.k` / `.d` from `state = [avg_gain, avg_loss, rsi_tail…]` over rows
/// `[from, n)`, returning the new-row values and the updated state. `None` when RSI cannot
/// be resumed (`from == 0`) or the carried context is too short / not warm. Bit-identical
/// to a fresh `stochrsi_fastk` (+ SMA for `.d`).
pub fn stochrsi_resume(
    close: &[f64],
    rsi_period: usize,
    fastk_period: usize,
    is_d: bool,
    fastd_period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = close.len();
    let c = stochrsi_ctx_depth(fastk_period, is_d, fastd_period);
    if state.len() != c + 2 || from == 0 || from > n || from < c {
        return None;
    }
    // Continue RSI over the new rows from the carried Wilder pair (as of `from-1`).
    let (new_rsi, new_wilder) = rsi_resume(close, rsi_period, from, &state[..2])?;
    // RSI buffer covering `[from - c, n)`: the carried context tail ++ the new RSI rows.
    let mut buf = Vec::with_capacity(c + new_rsi.len());
    buf.extend_from_slice(&state[2..]); // rsi[from-c .. from)
    buf.extend_from_slice(&new_rsi); // rsi[from .. n)
                                     // Every buffered RSI must be finite for the van-Herk %K (the warm guard above ensures
                                     // `from - c >= rsi_period`, so it is) — bail out defensively otherwise.
    if buf.iter().any(|x| x.is_nan()) {
        return None;
    }
    // Windowed stochastic %K of the RSI buffer (RSI as high=low=close, matching
    // `stochrsi_fastk`). Buffer index `p` is original row `from - c + p`.
    let fk = stoch_fastk(&buf, &buf, &buf, fastk_period);
    let line = if is_d {
        // `.d` is the SMA of %K; only matype 0 (SMA) is resumed (the caller gates this).
        super::ma(&fk, fastd_period)
    } else {
        fk
    };
    // New rows `[from, n)` are buffer indices `[c, c + (n-from))`.
    let out: Vec<f64> = line[c..].to_vec();
    debug_assert_eq!(out.len(), n - from);
    // Refresh the state: Wilder pair as of n-1, then the trailing `c` RSI values.
    let mut new_state = new_wilder;
    let full_rsi_len = c + new_rsi.len(); // == buf.len()
    new_state.extend_from_slice(&buf[full_rsi_len - c..]);
    Some((out, new_state))
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

/// The KDJ line a resume emits: %K, %D, or %J (`3K - 2D`).
#[derive(Clone, Copy)]
pub enum KdjLine {
    K,
    D,
    J,
}

/// Final KDJ recursive state after a full compute: `[k_last]` for `.k`, `[k_last, d_last]`
/// for `.d` / `.j` (the ⅓-weight SMA-smoothed %K and %D as of the last row). RSV is
/// finite-memory (a `period_rsv` window), so it is NOT carried — a [`kdj_resume`] recomputes
/// it from the windowed high/low/close tail. `want_d` also carries `d_last` (needed by
/// `.d` / `.j`). `None` for an empty series.
#[allow(clippy::too_many_arguments)]
pub fn kdj_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    init: f64,
    want_d: bool,
) -> Option<Vec<f64>> {
    let n = close.len();
    if n == 0 {
        return None;
    }
    let rsv = kdj_rsv(high, low, close, period_rsv);
    let k = kernels::ewma_with_init(rsv.view(), period_k, init);
    if !want_d {
        return Some(vec![k[n - 1]]);
    }
    let d = kernels::ewma_with_init(k.view(), period_d, init);
    Some(vec![k[n - 1], d[n - 1]])
}

/// Resume a KDJ `line` from `state` (`[k_last]` for `.k`, `[k_last, d_last]` for `.d`/`.j`,
/// as of row `from - 1`) over rows `[from, n)` — bit-identical to a full recompute. The %K /
/// %D SMA recurrences continue from the carried values (past the `init` seed) using the same
/// `base·prev + alpha·x` step as [`kernels::ewma_with_init`], while RSV is recomputed over the
/// windowed `high/low/close` tail `[from - period_rsv + 1, n)` (a windowed min/max gives the
/// identical value to the full series). `None` when there is not a full RSV window before
/// `from` (`from + 1 < period_rsv`), `from` is out of range, or the state is too short.
#[allow(clippy::too_many_arguments)]
pub fn kdj_resume(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    line: KdjLine,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = close.len();
    if period_rsv == 0 || from == 0 || from > n || from + 1 < period_rsv || state.is_empty() {
        return None;
    }
    let lo = from + 1 - period_rsv;
    // RSV over [from, n): compute on the windowed tail, then drop its `period_rsv - 1` warm-up.
    let rsv = kdj_rsv(&high[lo..n], &low[lo..n], &close[lo..n], period_rsv);
    let skip = from - lo; // == period_rsv - 1
    let (ak, bk) = (1.0 / period_k as f64, 1.0 - 1.0 / period_k as f64);
    let mut k_prev = state[0];
    match line {
        KdjLine::K => {
            let mut out = Vec::with_capacity(n - from);
            for &r in rsv.iter().skip(skip) {
                k_prev = bk * k_prev + ak * r;
                out.push(k_prev);
            }
            Some((out, vec![k_prev]))
        }
        KdjLine::D | KdjLine::J => {
            if state.len() < 2 {
                return None;
            }
            let (ad, bd) = (1.0 / period_d as f64, 1.0 - 1.0 / period_d as f64);
            let mut d_prev = state[1];
            let is_j = matches!(line, KdjLine::J);
            let mut out = Vec::with_capacity(n - from);
            for &r in rsv.iter().skip(skip) {
                k_prev = bk * k_prev + ak * r;
                d_prev = bd * d_prev + ad * k_prev;
                out.push(if is_j {
                    3.0 * k_prev - 2.0 * d_prev
                } else {
                    d_prev
                });
            }
            Some((out, vec![k_prev, d_prev]))
        }
    }
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
pub fn rsi_resume(
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
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
pub fn cmo_resume(
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
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

/// Center of Gravity oscillator (TradingView `ta.cog`, John Ehlers):
/// `-Σ((1+i)·close[i]) / Σ(close[i])` over the trailing `period`, where `close[i]`
/// is `i` bars back (newest weighted 1, oldest weighted `period`). A zero window
/// sum yields `NaN`. Lookback `period-1`. O(n·period).
pub fn cog(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    for i in (period - 1)..n {
        let mut num = 0.0; // Σ (1+age)·price, age 0 = newest
        let mut den = 0.0;
        for age in 0..period {
            let price = close[i - age];
            num += (1 + age) as f64 * price;
            den += price;
        }
        out[i] = if den != 0.0 { -num / den } else { f64::NAN };
    }
    out
}

/// Rank Correlation Index (TradingView `ta.rci`): Spearman's rank correlation
/// between `close` and the bar index over `period` bars, scaled to `[-100, 100]`.
/// Computed as the Pearson correlation of the (average-tie) value ranks against
/// the time ranks `1..=period`, so ties are handled exactly. A degenerate
/// (zero-variance) window yields `NaN`. Lookback `period-1`. O(n·period·log period).
pub fn rci(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period < 2 || period > n {
        return out;
    }
    // time ranks 1..=period (ascending = chronological); their mean and variance
    // are constant across windows.
    let p = period as f64;
    let t_mean = (p + 1.0) / 2.0;
    let t_ss: f64 = (1..=period).map(|t| (t as f64 - t_mean).powi(2)).sum();
    let mut idx: Vec<usize> = Vec::with_capacity(period);
    let mut prank = vec![0.0f64; period];
    for i in (period - 1)..n {
        let w = &close[i + 1 - period..=i];
        // average ranks of the window values (ascending)
        idx.clear();
        idx.extend(0..period);
        idx.sort_by(|&a, &b| w[a].partial_cmp(&w[b]).unwrap_or(std::cmp::Ordering::Equal));
        let mut k = 0;
        while k < period {
            let mut j = k + 1;
            while j < period && w[idx[j]] == w[idx[k]] {
                j += 1;
            }
            // ranks k+1..=j share the average rank (1-based)
            let avg = ((k + 1 + j) as f64) / 2.0;
            for &pos in &idx[k..j] {
                prank[pos] = avg;
            }
            k = j;
        }
        // Pearson(prank, time-rank). prank mean == t_mean (both are 1..=period).
        let mut cov = 0.0;
        let mut p_ss = 0.0;
        for (t, &pr) in prank.iter().enumerate() {
            let dp = pr - t_mean;
            let dt = (t + 1) as f64 - t_mean;
            cov += dp * dt;
            p_ss += dp * dp;
        }
        let denom = (p_ss * t_ss).sqrt();
        out[i] = if denom > 0.0 { cov / denom * 100.0 } else { f64::NAN };
    }
    out
}

/// Fractal pivot detection (TradingView `ta.pivothigh` / `ta.pivotlow`). A bar
/// `p` is a pivot when its `source` value is the STRICT extremum of the window
/// `[p-left, p+right]` (every other bar strictly lower for a high / strictly
/// higher for a low — a tie disqualifies it, matching Pine). The pivot's value
/// is emitted at the CONFIRMATION bar `p+right` (non-causal: it needs `right`
/// future bars), `NaN` everywhere else. A `NaN` anywhere in the window
/// disqualifies the pivot. Lookback `left+right`. O(n·(left+right)).
pub fn pivot(source: &[f64], left: usize, right: usize, high: bool) -> Vec<f64> {
    let n = source.len();
    let mut out = vec![f64::NAN; n];
    let win = left + right;
    for i in win..n {
        let p = i - right; // candidate pivot position
        let cand = source[p];
        if cand.is_nan() {
            continue;
        }
        let start = i - win; // = p - left
        let is_pivot = (start..=i).all(|j| {
            j == p || if high { source[j] < cand } else { source[j] > cand }
        });
        if is_pivot {
            out[i] = cand;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::stochrsi_fastk;
    use crate::indicators::test_support::*;

    /// StochRSI `.k` and `.d` resumes, fed the carried RSI Wilder pair + context tail of
    /// a full compute over the head, reproduce the tail of a full compute over the whole
    /// input — bit-for-bit. The full `.d` is the SMA (matype 0) of the full `.k`.
    #[test]
    fn stochrsi_resume_is_bit_identical_to_full() {
        let close = series(200);
        let (rp, fk, fd) = (14usize, 14usize, 3usize);
        let k_full = stochrsi_fastk(&close, rp, fk);
        let d_full = crate::indicators::ma(&k_full, fd);

        // `from` past the deepest context (`.d` reach = rp + (fk-1) + (fd-1)) so the head
        // always carries a full finite-RSI context.
        for &from in &[60usize, 70, 120, 199] {
            let head = &close[..from];

            // `.k` line (is_d = false).
            let st = stochrsi_final_state(head, rp, fk, false, fd).unwrap();
            let (tail, _) = stochrsi_resume(&close, rp, fk, false, fd, from, &st).unwrap();
            assert_bits(&tail, &k_full[from..], "stochrsi.k");

            // `.d` line (is_d = true). The `.d` SMA-of-%K rolls a running sum whose start
            // point differs between the windowed resume buffer and the full frame, so the
            // two agree to the production parity tolerance (~1e-9) rather than bit-for-bit.
            let st = stochrsi_final_state(head, rp, fk, true, fd).unwrap();
            let (tail, _) = stochrsi_resume(&close, rp, fk, true, fd, from, &st).unwrap();
            let want = &d_full[from..];
            assert_eq!(tail.len(), want.len(), "stochrsi.d length");
            for (i, (x, y)) in tail.iter().zip(want).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-9 || (x.is_nan() && y.is_nan()),
                    "stochrsi.d bar {i}: resume {x} != full {y}",
                );
            }
        }
    }

    /// StochRSI guards: a too-short close (RSI never accrues `C+1` finite rows) declines
    /// the final state; a bad state length / `from` declines the resume; and an embedded
    /// NaN in the close keeps an RSI row NaN, tripping the resume's NaN-in-buffer bail-out.
    #[test]
    fn stochrsi_guards_decline() {
        let (rp, fk, fd) = (14usize, 14usize, 3usize);

        // n < c || n - c < rsi_period -> final state declines (oscillators.rs:135).
        let short = series(20);
        assert!(stochrsi_final_state(&short, rp, fk, false, fd).is_none());

        // Bad state length / from -> resume declines (oscillators.rs:159).
        let close = series(200);
        let st = stochrsi_final_state(&close[..120], rp, fk, false, fd).unwrap();
        let bad = vec![0.0; 1]; // wrong length (!= c + 2)
        assert!(stochrsi_resume(&close, rp, fk, false, fd, 120, &bad).is_none());
        assert!(stochrsi_resume(&close, rp, fk, false, fd, 0, &st).is_none()); // from == 0

        // NaN-in-buffer -> resume declines (oscillators.rs:170). Embed a NaN late in the
        // close so a resumed RSI row stays NaN; the carried context is still finite (the
        // length/warm guards pass) but the freshly-resumed RSI tail carries the NaN.
        let mut nanclose = series(200);
        nanclose[150] = f64::NAN; // poisons RSI from row 150 onward
        let st = stochrsi_final_state(&nanclose[..140], rp, fk, false, fd).unwrap();
        assert!(stochrsi_resume(&nanclose, rp, fk, false, fd, 140, &st).is_none());
    }

    /// KDJ `.k` / `.d` / `.j` resumes, fed the carried %K (and %D) of a full compute over the
    /// head, reproduce the tail of a full compute over the whole input — bit-for-bit. RSV is
    /// recomputed from the windowed tail (a windowed min/max equals the full-series value),
    /// and the ⅓-weight %K/%D SMA recurrences continue from the carried values — so, unlike
    /// the windowed-SMA stochrsi `.d`, every KDJ line is exact.
    #[test]
    fn kdj_resume_is_bit_identical_to_full() {
        let (high, low, close) = ohlc(150);
        let (p, pk, pd, init) = (9usize, 3usize, 3usize, 50.0);
        let k_full = kdj_k(&high, &low, &close, p, pk, init);
        let d_full = kdj_d(&high, &low, &close, p, pk, pd, init);
        let j_full = kdj_j(&high, &low, &close, p, pk, pd, init);
        // `from` spans the first row a full RSV window exists (`p - 1`) through a generic
        // large offset; the carried %K/%D continue the recursion past the dropped head.
        for &from in &[p - 1, p, 30, 80, 149] {
            // `.k` carries just [%K].
            let st_k = kdj_final_state(
                &high[..from],
                &low[..from],
                &close[..from],
                p,
                pk,
                pd,
                init,
                false,
            )
            .unwrap();
            assert_eq!(st_k.len(), 1, "kdj.k carries [%K]");
            let (tail, ret) =
                kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, from, &st_k).unwrap();
            assert_bits(&tail, &k_full[from..], "kdj.k");
            assert_eq!(ret.len(), 1);

            // `.d` / `.j` carry [%K, %D].
            let st_d = kdj_final_state(
                &high[..from],
                &low[..from],
                &close[..from],
                p,
                pk,
                pd,
                init,
                true,
            )
            .unwrap();
            assert_eq!(st_d.len(), 2, "kdj.d/.j carry [%K, %D]");
            let (tail, ret) =
                kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::D, from, &st_d).unwrap();
            assert_bits(&tail, &d_full[from..], "kdj.d");
            assert_eq!(ret.len(), 2);
            let (tail, _) =
                kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::J, from, &st_d).unwrap();
            assert_bits(&tail, &j_full[from..], "kdj.j");
        }
    }

    /// KDJ guards: an empty series declines the final state; a zero RSV period, a zero /
    /// out-of-range / pre-window `from`, an empty state, or a single-element state for
    /// `.d`/`.j` all decline the resume (each then falls back to the correct full recompute).
    #[test]
    fn kdj_guards_decline() {
        let (high, low, close) = ohlc(60);
        let (p, pk, pd, init) = (9usize, 3usize, 3usize, 50.0);
        let n = close.len();

        assert!(kdj_final_state(&[], &[], &[], p, pk, pd, init, false).is_none()); // empty series

        let st =
            kdj_final_state(&high[..40], &low[..40], &close[..40], p, pk, pd, init, true).unwrap();
        assert!(kdj_resume(&high, &low, &close, 0, pk, pd, KdjLine::K, 40, &st).is_none()); // period_rsv == 0
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, 0, &st).is_none()); // from == 0
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, n + 1, &st).is_none()); // from > n
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, p - 2, &st).is_none()); // from + 1 < p
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, 40, &[]).is_none()); // empty state
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::D, 40, &[1.0]).is_none());
        // `.d` needs [%K, %D]
    }

    /// RSI / CMO resume parity plus the flat-window output arms. A strictly increasing
    /// close drives RSI's `avg_loss == 0` branch (output 100); a flat close drives CMO's
    /// `gain + loss == 0` branch (output 0).
    #[test]
    fn rsi_cmo_resume_and_flat_window_arms() {
        let close = series(120);
        let p = 14usize;
        let rsi_full = rsi(&close, p);
        let cmo_full = cmo(&close, p);
        for &from in &[p + 1, 30, 60, 119] {
            let st = rsi_final_state(&close[..from], p).unwrap();
            let (tail, _) = rsi_resume(&close, p, from, &st).unwrap();
            assert_bits(&tail, &rsi_full[from..], "rsi");

            let st = cmo_final_state(&close[..from], p).unwrap();
            let (tail, _) = cmo_resume(&close, p, from, &st).unwrap();
            assert_bits(&tail, &cmo_full[from..], "cmo");
        }

        // from == 0 -> both resumes decline (oscillators.rs:384, 440).
        let st = rsi_final_state(&close, p).unwrap();
        assert!(rsi_resume(&close, p, 0, &st).is_none());
        let st = cmo_final_state(&close, p).unwrap();
        assert!(cmo_resume(&close, p, 0, &st).is_none());

        // period == 0 / n <= period -> final states decline (oscillators.rs:355, 415).
        assert!(rsi_final_state(&[1.0, 2.0], 5).is_none());
        assert!(cmo_final_state(&[1.0, 2.0], 5).is_none());

        // Strictly increasing close: avg_loss == 0, so RSI's `emit` returns 100 in the
        // resume's flat-loss arm (oscillators.rs:391).
        let up: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let st = rsi_final_state(&up[..20], p).unwrap();
        let (tail, _) = rsi_resume(&up, p, 20, &st).unwrap();
        assert!(tail.iter().all(|&x| x == 100.0), "rsi flat-loss -> 100");

        // Flat close: gain + loss == 0, so CMO's resume returns 0 (oscillators.rs:453).
        let flat = vec![7.0; 40];
        let st = cmo_final_state(&flat[..20], p).unwrap();
        let (tail, _) = cmo_resume(&flat, p, 20, &st).unwrap();
        assert!(tail.iter().all(|&x| x == 0.0), "cmo flat-window -> 0");
    }
}
