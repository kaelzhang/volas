use ndarray::Array1;

use super::av;
use super::stochastic::{stoch_fastk, stochrsi_fastk};
use crate::accum::{anchors_at, Csum};
use crate::guard::Slide;
use crate::missing;
use crate::raw::kernels::{self, total_ge, total_gt};

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
        if total_gt(value, highest) {
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
                if total_gt(value, highest) {
                    highest_idx = idx;
                    highest = value;
                }
                idx += 1;
            }
        } else if total_ge(x, highest) {
            highest_idx = today;
            highest = x;
        }
        unsafe {
            *dst.add(today) = highest;
        }
    }
    out
}

/// Raw Stochastic Value (未成熟随机值): [`rsv_value`] over each `period`-bar window —
/// NaN during the warm-up (no full window yet) and while the window holds a missing
/// value. Lookback `period-1`.
pub fn rsv(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let llv = kernels::rolling_min(av(low), period);
    let hhv = kernels::rolling_max(av(high), period);
    (0..close.len()).map(|i| rsv_value(close[i], hhv[i], llv[i])).collect()
}

/// The raw stochastic value of `close` in a window whose highest high is `hh` and
/// lowest low `ll`: `100·(close − ll)/(hh − ll)`, `0` for a (near-)flat range. The one
/// owner of the RSV formula — [`rsv`] and KDJ both evaluate it. Missing when any of the
/// three is (the per-row term rule, [`missing::term`]): a missing range would otherwise
/// fall to the flat-range `0`.
#[inline]
pub(super) fn rsv_value(close: f64, hh: f64, ll: f64) -> f64 {
    let denom = hh - ll;
    let v = if denom.abs() > 1e-10 { (close - ll) / denom * 100.0 } else { 0.0 };
    missing::term(&[close, hh, ll], v)
}

// --- StochRSI state-carry (additive; the full-recompute fallback stays correct) ---
//
// StochRSI is a composite: a windowed stochastic %K of the (Wilder-recursive) RSI line,
// with the `.d` line a further SMA of that %K. The %K / SMA stages are finite-memory
// (position-independent windowed min/max / mean over a NaN-free RSI buffer), so the ONLY
// recursive part is the underlying RSI. We continue it bit-exactly in O(new rows) by
// carrying the RSI Wilder pair `[avg_gain, avg_loss]` as of `from-1` PLUS the recent RSI
// VALUES needed to fill the stochastic windows over the new rows:
//   `.k` -> state = [avg_gain, avg_loss, rsi_{from-C}, …, rsi_{from-1}]
//   `.d` -> state = [avg_gain, avg_loss, <the SMA's Slide>, rsi_{from-C}, …, rsi_{from-1}]
// where the RSI-context depth `C` is the stage lookback before `from`:
//   `.k` -> C = fastk_period - 1                   (the %K window)
//   `.d` -> C = (fastk_period - 1) + fastd_period  (%K back to the row the SMA drops)
// On resume we `rsi_resume` the new RSI tail, concatenate the carried context and run the
// (NaN-free) windowed %K — bit-identical to a fresh `stochrsi_fastk`, since a min/max
// over a finite window is order/position-independent. The `.d` SMA is a running sum,
// whose value depends on the rows since its last anchor: it is carried (the [`Slide`]
// state of the full SMA as of `from-1`) and continued like the full SMA, so `.d` is
// bit-identical too. Its rows are StochRSI's complete rows, the first at absolute position
// `origin`. A resume that cannot see a full context
// of FINITE RSI (`from - C < rsi_period`, i.e. the tracker is not yet warm) returns `None`
// and falls back; a carried slice keeps `>= lookback` rows, so it is always warm enough.
// Only the canonical SMA `.d` (matype 0) is resumed; a recursive-MA `.d` declines.

/// RSI-context depth carried before `from` for a StochRSI resume (see the module note):
/// the %K window for `.k`, plus the SMA-of-%K reach for `.d`.
fn stochrsi_ctx_depth(fastk_period: usize, is_d: bool, fastd_period: usize) -> usize {
    let k = fastk_period.saturating_sub(1);
    if is_d {
        k + fastd_period
    } else {
        k
    }
}

/// Words of a StochRSI state before the RSI tail: the Wilder pair, then for `.d` the
/// SMA's [`Slide`] state.
fn stochrsi_head_len(is_d: bool) -> usize {
    if is_d {
        2 + Slide::<1>::STATE
    } else {
        2
    }
}

/// The `.d` SMA (`fastd_period`) slid over `%K` row `i` — the full SMA's own slide
/// ([`stochrsi_d_sma`], `stochrsi_d_default_sma`), so the value and the running sum
/// match it bit for bit.
fn stochrsi_d_row(slide: &mut Slide<1>, fastd_period: usize, i: usize, k: impl Fn(usize) -> f64) -> f64 {
    kernels::sma_row(slide, fastd_period, i, k)
}

/// StochRSI `.d` with an SMA `fastd_period` smoothing: `sma(stochrsi_fastk(..))`, row 0
/// at absolute position `origin`.
pub fn stochrsi_d_sma(
    close: &[f64],
    rsi_period: usize,
    fastk_period: usize,
    fastd_period: usize,
    origin: usize,
) -> Vec<f64> {
    let k = stochrsi_fastk(close, rsi_period, fastk_period);
    kernels::sma(super::av(&k), fastd_period, origin).into_raw_vec_and_offset().0
}

/// Final StochRSI state (see the module note) after a full compute, or `None`
/// if RSI never warms up (`rsi_period == 0 || n <= rsi_period`) or there are not yet `C+1`
/// finite RSI rows to anchor a resume. `is_d` / `fastd_period` only size the carried RSI
/// tail (the deeper SMA reach). The Wilder pair matches [`rsi_final_state`] exactly.
pub fn stochrsi_final_state(
    close: &[f64],
    rsi_period: usize,
    fastk_period: usize,
    is_d: bool,
    fastd_period: usize,
    origin: usize,
) -> Option<Vec<f64>> {
    let n = close.len();
    let wilder = rsi_final_state(close, rsi_period)?; // [avg_gain, avg_loss] as of n-1
    let c = stochrsi_ctx_depth(fastk_period, is_d, fastd_period);
    // Need `c` RSI values ending at n-1, all finite (RSI is finite from row `rsi_period`).
    if n < c || n - c < rsi_period {
        return None;
    }
    if fastk_period == 0 || (is_d && fastd_period == 0) {
        return None;
    }
    let rsi = rsi(close, rsi_period);
    let mut state = wilder;
    if is_d {
        let k = stochrsi_fastk(close, rsi_period, fastk_period);
        let mut slide = Slide::<1>::at(origin);
        for i in 0..n {
            stochrsi_d_row(&mut slide, fastd_period, i, |j| k[j]);
        }
        let mut words = [0.0; Slide::<1>::STATE];
        slide.write_state(&mut words);
        state.extend_from_slice(&words);
    }
    state.extend_from_slice(&rsi[n - c..n]);
    Some(state)
}

/// Resume StochRSI `.k` / `.d` from its state (see the module note) over rows
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
    let head = stochrsi_head_len(is_d);
    if state.len() != c + head || from == 0 || from > n || from < c || fastk_period == 0 {
        return None;
    }
    // Continue RSI over the new rows from the carried Wilder pair (as of `from-1`).
    let (new_rsi, new_wilder) = rsi_resume(close, rsi_period, from, &state[..2])?;
    // RSI buffer covering `[from - c, n)`: the carried context tail ++ the new RSI rows.
    let mut buf = Vec::with_capacity(c + new_rsi.len());
    buf.extend_from_slice(&state[head..]); // rsi[from-c .. from)
    buf.extend_from_slice(&new_rsi); // rsi[from .. n)
                                     // Every buffered RSI must be finite for the van-Herk %K (the warm guard above ensures
                                     // `from - c >= rsi_period`, so it is) — bail out defensively otherwise.
    if buf.iter().any(|x| x.is_nan()) {
        return None;
    }
    // Windowed stochastic %K of the RSI buffer (RSI as high=low=close, matching
    // `stochrsi_fastk`). Buffer index `p` is original row `from - c + p`.
    let fk = stoch_fastk(&buf, &buf, &buf, fastk_period);
    let mut new_state = new_wilder;
    // New rows `[from, n)` are buffer indices `[c, c + (n-from))`.
    let out: Vec<f64> = if is_d {
        // `.d` is the SMA of %K; only matype 0 (SMA) is resumed (the caller gates this).
        // Buffer index `p` is row `from - c + p`, so the SMA's row `i` reads `fk[i + c -
        // from]` — present for every row it drops (`>= from - fastd_period`).
        let mut slide = Slide::<1>::from_state(&state[2..head])?;
        let base = from - c;
        let k = |j: usize| fk[j - base];
        let out = (from..n)
            .map(|i| stochrsi_d_row(&mut slide, fastd_period, i, k))
            .collect();
        let mut words = [0.0; Slide::<1>::STATE];
        slide.write_state(&mut words);
        new_state.extend_from_slice(&words);
        out
    } else {
        fk[c..].to_vec()
    };
    debug_assert_eq!(out.len(), n - from);
    // Refresh the state: Wilder pair as of n-1, [the `.d` slide,] the trailing `c` RSI.
    let full_rsi_len = c + new_rsi.len(); // == buf.len()
    new_state.extend_from_slice(&buf[full_rsi_len - c..]);
    Some((out, new_state))
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
    let mut w = Wilder::rsi_seed(close, period);
    out[period] = w.rsi();
    for i in (period + 1)..n {
        w.rsi_step(close[i] - close[i - 1], period);
        out[i] = w.rsi();
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
        result[i] = Wilder { gain: sg[i], loss: sl[i] }.cmo();
    }
    result
}

// --- RSI / CMO state-carry (additive; the full-recompute fallback stays correct) ---
//
// Both smooth bar-to-bar gains and losses with a Wilder average, so the carried state is
// the pair `[avg_gain, avg_loss]` as of row `from-1`. A delta needs the prior close, so a
// resume reads only `close[from-1..]`; `from == 0` returns `None` (falls back). NOTE the
// two use DIFFERENT recurrences — `rsi` divides each step (`(avg·(p-1)+x)/p`) while `cmo`
// goes through `kernels::wilder`'s fused `avg·a + x·b`. Each recurrence, seed and output
// rule (with its flat-window guard) has one owner, [`Wilder`], which every path uses.
// `*_final_state` returns `None` before the seed (`n <= period`), keeping the fallback.

/// The Wilder-smoothed average gain and loss behind RSI and CMO — the one owner of their
/// seeds, steps and output rules.
#[derive(Clone, Copy, Debug)]
struct Wilder {
    gain: f64,
    loss: f64,
}

impl Wilder {
    fn from_state(state: &[f64]) -> Option<Self> {
        let [gain, loss] = *state else { return None };
        Some(Wilder { gain, loss })
    }

    fn state(self) -> Vec<f64> {
        vec![self.gain, self.loss]
    }

    /// RSI's seed: the SMA of the first `period` gains / losses (deltas `1..=period`).
    fn rsi_seed(close: &[f64], period: usize) -> Self {
        let pf = period as f64;
        let (mut gain, mut loss) = (0.0, 0.0);
        for i in 1..=period {
            let d = close[i] - close[i - 1];
            if d > 0.0 {
                gain += d;
            } else {
                loss -= d;
            }
        }
        Wilder { gain: gain / pf, loss: loss / pf }
    }

    /// RSI's step `(avg·(p-1) + x) / p` over the delta `d`.
    #[inline(always)]
    fn rsi_step(&mut self, d: f64, period: usize) {
        let pf = period as f64;
        let p1 = pf - 1.0;
        let (gain, loss) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
        self.gain = (self.gain * p1 + gain) / pf;
        self.loss = (self.loss * p1 + loss) / pf;
    }

    /// RSI `100 − 100/(1 + g/l)`, a flat loss (`|l| < 1e-10`) giving 100.
    #[inline(always)]
    fn rsi(self) -> f64 {
        if self.loss.abs() < 1e-10 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + self.gain / self.loss)
        }
    }

    /// CMO's seed: `kernels::wilder`'s SMA of the first `period` gains / losses.
    fn cmo_seed(close: &[f64], period: usize) -> Self {
        let pf = period as f64;
        let (mut gain, mut loss) = (0.0, 0.0);
        for i in 1..=period {
            let d = close[i] - close[i - 1];
            gain += d.max(0.0);
            loss += (-d).max(0.0);
        }
        Wilder { gain: gain / pf, loss: loss / pf }
    }

    /// CMO's step: `kernels::wilder`'s fused `avg·a + x·b` over the delta `d`.
    #[inline(always)]
    fn cmo_step(&mut self, d: f64, period: usize) {
        let pf = period as f64;
        let (a, b) = ((pf - 1.0) / pf, 1.0 / pf);
        self.gain = self.gain.mul_add(a, d.max(0.0) * b);
        self.loss = self.loss.mul_add(a, (-d).max(0.0) * b);
    }

    /// CMO `100·(g−l)/(g+l)`, a flat window (`g+l < 1e-14`) giving 0.
    #[inline(always)]
    fn cmo(self) -> f64 {
        let denom = self.gain + self.loss;
        if denom < 1e-14 {
            0.0
        } else {
            100.0 * (self.gain - self.loss) / denom
        }
    }
}

/// Final RSI state `[avg_gain, avg_loss]` after a full [`rsi`] compute, or `None` if it
/// never seeds (`period == 0 || n <= period`).
pub fn rsi_final_state(close: &[f64], period: usize) -> Option<Vec<f64>> {
    let n = close.len();
    if period == 0 || n <= period {
        return None;
    }
    let mut w = Wilder::rsi_seed(close, period);
    for i in (period + 1)..n {
        w.rsi_step(close[i] - close[i - 1], period);
    }
    Some(w.state())
}

/// Resume [`rsi`] from `state = [avg_gain, avg_loss]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `close[from-1..]`; bit-identical to [`rsi`].
pub fn rsi_resume(
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from == 0 {
        return None;
    }
    let mut w = Wilder::from_state(state)?;
    let out = (from..close.len())
        .map(|i| {
            w.rsi_step(close[i] - close[i - 1], period);
            w.rsi()
        })
        .collect();
    Some((out, w.state()))
}

/// Final CMO state `[avg_gain, avg_loss]` after a full [`cmo`] compute, or `None` if it
/// never seeds.
pub fn cmo_final_state(close: &[f64], period: usize) -> Option<Vec<f64>> {
    let n = close.len();
    if period == 0 || n <= period {
        return None;
    }
    let mut w = Wilder::cmo_seed(close, period);
    for i in (period + 1)..n {
        w.cmo_step(close[i] - close[i - 1], period);
    }
    Some(w.state())
}

/// Resume [`cmo`] from `state = [avg_gain, avg_loss]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `close[from-1..]`; bit-identical to [`cmo`].
pub fn cmo_resume(
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from == 0 {
        return None;
    }
    let mut w = Wilder::from_state(state)?;
    let out = (from..close.len())
        .map(|i| {
            w.cmo_step(close[i] - close[i - 1], period);
            w.cmo()
        })
        .collect();
    Some((out, w.state()))
}

/// Scalar single-row twin of [`rsi_resume`]: the RSI at `row` continued from
/// `state = [avg_gain, avg_loss]` (as of `row - 1`), with the advanced state as of `row`
/// written into `next` — no allocation. `None` at `row == 0`, a `state` / `next` that is
/// not two values, or `row` out of range.
pub fn rsi_resume_one(
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row == 0 || next.len() != 2 || row >= close.len() {
        return None;
    }
    let mut w = Wilder::from_state(state)?;
    w.rsi_step(close[row] - close[row - 1], period);
    next.copy_from_slice(&[w.gain, w.loss]);
    Some(w.rsi())
}

/// Scalar single-row twin of [`cmo_resume`]: the CMO at `row` continued from
/// `state = [avg_gain, avg_loss]` (as of `row - 1`), with the advanced state as of `row`
/// written into `next` — no allocation. `None` at `row == 0`, a `state` / `next` that is
/// not two values, or `row` out of range.
pub fn cmo_resume_one(
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row == 0 || next.len() != 2 || row >= close.len() {
        return None;
    }
    let mut w = Wilder::from_state(state)?;
    w.cmo_step(close[row] - close[row - 1], period);
    next.copy_from_slice(&[w.gain, w.loss]);
    Some(w.cmo())
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
/// sum yields `NaN`. Lookback `period-1`.
///
/// An **O(n)** sliding recurrence ([`cog_row`]). A window holding a non-finite bar is
/// computed exactly on its own rows (O(period)), and the slide is re-seeded from the
/// first clean window after it — a bad bar reaches only the windows holding it, and a
/// value never depends on the rows after it.
pub fn cog(close: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut state = cog_initial(origin);
    for (i, o) in out.iter_mut().enumerate() {
        *o = cog_row(close, period, i, &mut state);
    }
    out
}

/// The length of [`cog`]'s state: `[Σ(1+age)·price, Σprice, Σ|price|]` (each a
/// [`Csum`]), the non-finite bars in the window, the bars seen (capped), whether the
/// sums describe the window, the absolute position of the next bar (for re-anchoring).
const COG_STATE: usize = 3 * Csum::STATE + 4;

/// [`cog`]'s state before its first bar, at absolute position `origin`.
fn cog_initial(origin: usize) -> [f64; COG_STATE] {
    let mut st = [0.0; COG_STATE];
    st[9] = origin as f64;
    st
}

/// One row of [`cog`] at `i`, advancing `st` (bars `0..i` already seen).
///
/// The first full window is seeded newest-first, then each bar slides it: dropping the
/// oldest (weight `period`) lowers every retained weight, so `num[i] = num[i-1] +
/// den[i-1] - (period+1)·leaving + close[i]` and `den[i] = den[i-1] - leaving +
/// close[i]` (the WMA-slide trick, mirrored to cog's newest-weighted-1 orientation).
/// The sums are compensated and the weighted terms exact, so a bar leaves `num` exactly
/// as it entered it.
/// `aden` (Σ|price|) is the cancellation scale: when the running `den` is negligible
/// against it the window sum has cancelled to ~0, but a running sum lands on float drift
/// (~1e-15) instead of an exact 0.0, so the `den != 0.0` guard would wrongly emit a huge
/// garbage value. There the window is recomputed exactly (O(period), only on
/// near-cancellation) so a true zero sum yields `NaN`, matching the per-window form.
#[inline]
fn cog_row(close: &[f64], period: usize, i: usize, st: &mut [f64; COG_STATE]) -> f64 {
    let sum_at = |k: usize| Csum::from_state(&st[2 * k..2 * k + 2]).unwrap_or_default();
    let (mut num, mut den, mut aden) = (sum_at(0), sum_at(1), sum_at(2));
    let (bad, seen, valid, pos) = (st[6], st[7], st[8], st[9] as usize);
    let (mut bad, seen) = (bad as usize, seen as usize);
    let pf = period as f64;
    if !close[i].is_finite() {
        bad += 1;
    }
    let full = seen + 1 >= period;
    if full && seen + 1 > period && !close[i - period].is_finite() {
        bad -= 1;
    }
    let mut valid = valid != 0.0;
    let value = if !full {
        f64::NAN
    } else if bad > 0 {
        valid = false;
        cog_window_exact(close, i, period)
    } else {
        // Re-anchoring at an absolute position is re-seeding the window (see
        // `crate::accum`).
        if valid && !anchors_at(pos, period) {
            let leaving = close[i - period];
            num.add_sum(&den);
            num.sub_prod(pf + 1.0, leaving);
            num.add(close[i]);
            den.sub(leaving);
            den.add(close[i]);
            aden.sub(leaving.abs());
            aden.add(close[i].abs());
        } else {
            (num, den, aden) = (Csum::default(), Csum::default(), Csum::default());
            for age in 0..period {
                let price = close[i - age]; // age 0 = newest
                num.add_prod((1 + age) as f64, price);
                den.add(price);
                aden.add(price.abs());
            }
            valid = true;
        }
        let d = den.value();
        if d.abs() <= 1e-9 * aden.value() {
            cog_window_exact(close, i, period)
        } else {
            -num.value() / d
        }
    };
    num.write_state(&mut st[0..2]);
    den.write_state(&mut st[2..4]);
    aden.write_state(&mut st[4..6]);
    st[6] = bad as f64;
    st[7] = (seen + 1).min(period + 1) as f64;
    st[8] = f64::from(u8::from(valid));
    st[9] = (pos + 1) as f64;
    value
}

/// [`cog`]'s state after the rows of `close`.
pub fn cog_state(close: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    crate::guard::state_after(cog_initial(origin).to_vec(), close.len(), |row, prev, next| {
        cog_step(close, period, row, prev, next)
    })
}

/// One row of [`cog`] at `row`, from `state` into `next`.
pub fn cog_step(close: &[f64], period: usize, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    if period == 0 || row >= close.len() || next.len() != COG_STATE {
        return None;
    }
    let mut st = <[f64; COG_STATE]>::try_from(state).ok()?;
    let value = cog_row(close, period, row, &mut st);
    next.copy_from_slice(&st);
    Some(value)
}


/// One cog window computed exactly (the O(period) per-window form): the slide's
/// `NaN`-input fallback and its near-zero-denominator recovery. `i` is the newest
/// bar, weighting newest 1 … oldest `period`; an exact zero window sum ⇒ `NaN`.
#[inline]
fn cog_window_exact(close: &[f64], i: usize, period: usize) -> f64 {
    let mut num = 0.0; // Σ (1+age)·price, age 0 = newest
    let mut den = 0.0;
    for age in 0..period {
        let price = close[i - age];
        num += (1 + age) as f64 * price;
        den += price;
    }
    if den != 0.0 {
        -num / den
    } else {
        f64::NAN
    }
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
        // A missing value makes the rank order undefined — disqualify the window (NaN
        // out), matching the other window indicators; never let the sort fallback turn
        // a NaN window into a confident signal.
        if w.iter().any(|x| x.is_nan()) {
            continue;
        }
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
    #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
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
    use crate::raw::indicators::stochrsi_fastk;
    use crate::raw::indicators::test_support::*;

    /// A window containing a missing value (`NaN`) must NOT yield a confident rank
    /// correlation — it disqualifies the window (NaN out), like the other window
    /// indicators (`dev` / `mode` / `pivot`). The sort fallback used to turn
    /// `[1, NaN, 3]` into a perfect `100.0` signal.
    #[test]
    fn rci_nan_in_window_disqualifies() {
        let out = rci(&[1.0, f64::NAN, 3.0, 4.0, 5.0], 3);
        assert!(out[2].is_nan(), "window [1, NaN, 3] must be NaN, got {}", out[2]);
        assert!(out[3].is_nan(), "window [NaN, 3, 4] must be NaN, got {}", out[3]);
        assert!(out[4].is_finite(), "window [3, 4, 5] is clean -> finite, got {}", out[4]);
    }

    /// StochRSI `.k` and `.d` resumes, fed the carried RSI Wilder pair + context tail of
    /// a full compute over the head, reproduce the tail of a full compute over the whole
    /// input — bit-for-bit. The full `.d` is the SMA (matype 0) of the full `.k`.
    #[test]
    fn stochrsi_resume_is_bit_identical_to_full() {
        let close = series(200);
        let (rp, fk, fd) = (14usize, 14usize, 3usize);
        let k_full = stochrsi_fastk(&close, rp, fk);
        let d_full = crate::raw::indicators::ma(&k_full, fd, 0);

        // `from` past the deepest context (`.d` reach = rp + (fk-1) + (fd-1)) so the head
        // always carries a full finite-RSI context.
        for &from in &[60usize, 70, 120, 199] {
            let head = &close[..from];

            // `.k` line (is_d = false).
            let st = stochrsi_final_state(head, rp, fk, false, fd, 0).unwrap();
            let (tail, _) = stochrsi_resume(&close, rp, fk, false, fd, from, &st).unwrap();
            assert_bits(&tail, &k_full[from..], "stochrsi.k");

            // `.d` line (is_d = true): the carried running sum of the `.d` SMA continues
            // the full frame's own, so the two agree bit for bit — also when the resume
            // is chained one row at a time from each returned state.
            let st = stochrsi_final_state(head, rp, fk, true, fd, 0).unwrap();
            let (tail, _) = stochrsi_resume(&close, rp, fk, true, fd, from, &st).unwrap();
            assert_bits(&tail, &d_full[from..], "stochrsi.d");
            let mut st = st;
            for row in from..close.len() {
                let (one, next) = stochrsi_resume(&close[..=row], rp, fk, true, fd, row, &st).unwrap();
                assert_bits(&one, &d_full[row..=row], "stochrsi.d chained");
                st = next;
            }
            // the default fused `.d` kernel is the same SMA
            let d_default = crate::raw::indicators::stochrsi_d_default_sma(&close, 0).unwrap();
            let st = stochrsi_final_state(head, 14, 5, true, 3, 0).unwrap();
            let (tail, _) = stochrsi_resume(&close, 14, 5, true, 3, from, &st).unwrap();
            assert_bits(&tail, &d_default[from..], "stochrsi.d default");
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
        assert!(stochrsi_final_state(&short, rp, fk, false, fd, 0).is_none());

        // Bad state length / from -> resume declines (oscillators.rs:159).
        let close = series(200);
        let st = stochrsi_final_state(&close[..120], rp, fk, false, fd, 0).unwrap();
        let bad = vec![0.0; 1]; // wrong length (!= c + 2)
        assert!(stochrsi_resume(&close, rp, fk, false, fd, 120, &bad).is_none());
        assert!(stochrsi_resume(&close, rp, fk, false, fd, 0, &st).is_none()); // from == 0

        // NaN-in-buffer -> resume declines (oscillators.rs:170). Embed a NaN late in the
        // close so a resumed RSI row stays NaN; the carried context is still finite (the
        // length/warm guards pass) but the freshly-resumed RSI tail carries the NaN.
        let mut nanclose = series(200);
        nanclose[150] = f64::NAN; // poisons RSI from row 150 onward
        let st = stochrsi_final_state(&nanclose[..140], rp, fk, false, fd, 0).unwrap();
        assert!(stochrsi_resume(&nanclose, rp, fk, false, fd, 140, &st).is_none());
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
