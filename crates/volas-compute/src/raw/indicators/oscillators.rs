use super::av;
use super::stochastic::{stoch_fastk, stochrsi_fastk};
use crate::accum::{Accumulator, Csum};
use crate::missing;
use crate::raw::kernels::{self, canonical_zero};
use crate::raw::smoothing;

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
        *dst.add(9) = canonical_zero(highest);
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
            *dst.add(today) = canonical_zero(highest);
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
/// lowest low `ll`: `100·(close − ll)/(hh − ll)`, `0` for a flat range (`hh − ll == 0`,
/// exactly, as TA-Lib's STOCHF tests it — an absolute threshold would zero the real
/// ranges of a tiny-priced series). The one
/// owner of the RSV formula — [`rsv`] and KDJ both evaluate it. Missing when any of the
/// three is (the per-row term rule, [`missing::term`]): a missing range would otherwise
/// fall to the flat-range `0`.
#[inline]
pub(super) fn rsv_value(close: f64, hh: f64, ll: f64) -> f64 {
    let denom = hh - ll;
    let v = if denom != 0.0 { (close - ll) / denom * 100.0 } else { 0.0 };
    missing::term(&[close, hh, ll], v)
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

/// StochRSI's single continuation: RSI's Wilder pair, the recent RSI window,
/// and (for D) the generic MA stage including its input ring. The MA is seeded
/// from the same raw %K series passed to ma_typed by the full computation.
#[allow(clippy::too_many_arguments)]
pub fn stochrsi_final_state(close: &[f64], rsi_period: usize, fastk_period: usize, is_d: bool, fastd_period: usize, matype: usize, origin: usize) -> Option<Vec<f64>> {
    let c = fastk_period.checked_sub(1)?;
    let n = close.len();
    if n < rsi_period + c || fastd_period == 0 { return None; }
    let mut state = rsi_final_state(close, rsi_period)?;
    let rsi = rsi(close, rsi_period);
    state.extend_from_slice(&rsi[n - c..]);
    if is_d {
        let ma = super::MaStage::new(fastd_period, matype)?;
        let k = stochrsi_fastk(close, rsi_period, fastk_period);
        let stage = crate::guard::state_after(ma.chained_initial(origin), n, |row, prev, next| ma.step_chained(row, k[row], prev, next))?;
        state.extend(stage);
    }
    Some(state)
}

/// Append StochRSI rows, continuing RSI and every generic MA type from their own states.
#[allow(clippy::too_many_arguments)]
pub fn stochrsi_resume(close: &[f64], rsi_period: usize, fastk_period: usize, is_d: bool, fastd_period: usize, matype: usize, from: usize, state: &[f64], out: &mut Vec<f64>) -> Option<Vec<f64>> {
    let c = fastk_period.checked_sub(1)?;
    let ma = super::MaStage::new(fastd_period, matype)?;
    if state.len() != 2 + c + if is_d { ma.chained_len() } else { 0 } || from == 0 || from > close.len() { return None; }
    let mut buf = state[2..2 + c].to_vec();
    let mut next = rsi_resume(close, rsi_period, from, &state[..2], &mut buf)?;
    let fk = stoch_fastk(&buf, &buf, &buf, fastk_period);
    next.extend_from_slice(&buf[buf.len() - c..]);
    if is_d {
        let mut prev = state[2 + c..].to_vec();
        let mut new = prev.clone();
        for (i, &value) in fk[c..].iter().enumerate() {
            out.push(ma.step_chained(from + i, value, &prev, &mut new)?);
            std::mem::swap(&mut prev, &mut new);
        }
        next.extend(prev);
    } else { out.extend_from_slice(&fk[c..]); }
    Some(next)
}

/// Relative Strength Index (TA-Lib RSI): `100·(avgGain/(avgGain+avgLoss))` over the
/// Wilder-averaged gains and losses ([`GainLoss`]). Single pass, single allocation: the
/// averages are seeded from the first `period` changes, emitted from index `period`, then
/// stepped per row.
pub fn rsi(close: &[f64], period: usize) -> Vec<f64> {
    gain_loss_series(close, period, GainLoss::rsi)
}

/// Chande Momentum Oscillator (TA-Lib CMO): `100·((avgGain−avgLoss)/(avgGain+avgLoss))`
/// over the same Wilder-averaged gains and losses as RSI ([`GainLoss`]); a flat window
/// (gain + loss = 0) yields 0. Lookback `period`. (Algebraically `2·RSI − 100`; computed
/// directly in TA-Lib's order rather than derived from RSI.)
pub fn cmo(close: &[f64], period: usize) -> Vec<f64> {
    gain_loss_series(close, period, GainLoss::cmo)
}

/// The full RSI / CMO series: NaN before index `period`, then `value` of the averages.
fn gain_loss_series(close: &[f64], period: usize, value: impl Fn(GainLoss) -> f64) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return out;
    }
    let mut w = GainLoss::seed(close, period);
    out[period] = value(w);
    let avg = smoothing::Wilder::new(period);
    for i in (period + 1)..n {
        w.step(avg, close[i] - close[i - 1]);
        out[i] = value(w);
    }
    out
}

// --- RSI / CMO state-carry (additive; the full-recompute fallback stays correct) ---
//
// Both average bar-to-bar gains and losses with TA-Lib's one recurrence (`ta_RSI.c` and
// `ta_CMO.c` share it), so the carried state is the pair `[avg_gain, avg_loss]` as of row
// `from-1`. A change needs the prior close, so a resume reads only `close[from-1..]`;
// `from == 0` returns `None` (falls back). The seed, step and output rules have one
// owner, [`GainLoss`], which every path uses. `*_final_state` returns `None` before the
// seed (`n <= period`), keeping the fallback.

/// The Wilder-averaged gain and loss behind RSI and CMO — the one owner of their seed,
/// step ([`smoothing::gain_loss_seed`], [`smoothing::Wilder::gain_loss`]) and output
/// rules.
#[derive(Clone, Copy, Debug)]
struct GainLoss {
    gain: f64,
    loss: f64,
}

impl GainLoss {
    fn from_state(state: &[f64]) -> Option<Self> {
        let [gain, loss] = *state else { return None };
        Some(GainLoss { gain, loss })
    }

    fn state(self) -> Vec<f64> {
        vec![self.gain, self.loss]
    }

    /// The seed from the first `period` changes (deltas `1..=period`).
    fn seed(close: &[f64], period: usize) -> Self {
        let (gain, loss) = smoothing::gain_loss_seed(period, |i| close[i] - close[i - 1]);
        GainLoss { gain, loss }
    }

    /// One step over the change `d`.
    #[inline(always)]
    fn step(&mut self, avg: smoothing::Wilder, d: f64) {
        (self.gain, self.loss) = avg.gain_loss(self.gain, self.loss, d);
    }

    /// RSI `100·(g/(g+l))` (TA-Lib's form), a flat window (`g + l == 0`, exactly) giving 0.
    /// TA-Lib tests `TA_IS_ZERO` (an absolute `1e-14`), which zeroes real movements of
    /// tiny-priced series; the exact test agrees with it on every flat window and keeps
    /// the scale-free value everywhere else (a deliberate, documented divergence).
    #[inline(always)]
    fn rsi(self) -> f64 {
        let denom = self.gain + self.loss;
        if denom == 0.0 {
            0.0
        } else {
            100.0 * (self.gain / denom)
        }
    }

    /// CMO `100·((g−l)/(g+l))` (TA-Lib's form), a flat window (`g + l == 0`, exactly)
    /// giving 0 — RSI's zero test on the same denominator (see [`GainLoss::rsi`]).
    #[inline(always)]
    fn cmo(self) -> f64 {
        let denom = self.gain + self.loss;
        if denom == 0.0 {
            0.0
        } else {
            100.0 * ((self.gain - self.loss) / denom)
        }
    }
}

/// The averages `[avg_gain, avg_loss]` after a full RSI / CMO compute, or `None` if they
/// never seed (`period == 0 || n <= period`).
fn gain_loss_final_state(close: &[f64], period: usize) -> Option<Vec<f64>> {
    let n = close.len();
    if period == 0 || n <= period {
        return None;
    }
    let mut w = GainLoss::seed(close, period);
    let avg = smoothing::Wilder::new(period);
    for i in (period + 1)..n {
        w.step(avg, close[i] - close[i - 1]);
    }
    Some(w.state())
}

/// Resume RSI / CMO (`value`) from `state = [avg_gain, avg_loss]` over rows `[from, n)`.
/// `None` at `from == 0`. Reads only `close[from-1..]`.
fn gain_loss_resume(
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
    value: impl Fn(GainLoss) -> f64,
    out: &mut Vec<f64>,
) -> Option<Vec<f64>> {
    if from == 0 {
        return None;
    }
    let mut w = GainLoss::from_state(state)?;
    let avg = smoothing::Wilder::new(period);
    out.reserve(close.len().saturating_sub(from));
    out.extend((from..close.len()).map(|i| {
        w.step(avg, close[i] - close[i - 1]);
        value(w)
    }));
    Some(w.state())
}

/// The single-row twin of [`gain_loss_resume`]: `value` at `row` continued from `state`
/// (as of `row - 1`), the state as of `row` written into `next` — no allocation. `None`
/// at `row == 0`, a `state` / `next` that is not two values, or `row` out of range.
fn gain_loss_resume_one(
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
    value: impl Fn(GainLoss) -> f64,
) -> Option<f64> {
    if row == 0 || next.len() != 2 || row >= close.len() {
        return None;
    }
    let mut w = GainLoss::from_state(state)?;
    w.step(smoothing::Wilder::new(period), close[row] - close[row - 1]);
    next.copy_from_slice(&[w.gain, w.loss]);
    Some(value(w))
}

/// Final RSI state `[avg_gain, avg_loss]` after a full [`rsi`] compute, or `None` if it
/// never seeds (`period == 0 || n <= period`).
pub fn rsi_final_state(close: &[f64], period: usize) -> Option<Vec<f64>> {
    gain_loss_final_state(close, period)
}

/// Resume [`rsi`] from `state = [avg_gain, avg_loss]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `close[from-1..]`; bit-identical to [`rsi`].
/// Appends rows `[from, n)` to `out` and returns the carried state.
pub fn rsi_resume(
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
    out: &mut Vec<f64>,
) -> Option<Vec<f64>> {
    gain_loss_resume(close, period, from, state, GainLoss::rsi, out)
}

/// Final CMO state `[avg_gain, avg_loss]` after a full [`cmo`] compute, or `None` if it
/// never seeds.
pub fn cmo_final_state(close: &[f64], period: usize) -> Option<Vec<f64>> {
    gain_loss_final_state(close, period)
}

/// Resume [`cmo`] from `state = [avg_gain, avg_loss]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `close[from-1..]`; bit-identical to [`cmo`].
/// Appends rows `[from, n)` to `out` and returns the carried state.
pub fn cmo_resume(
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
    out: &mut Vec<f64>,
) -> Option<Vec<f64>> {
    gain_loss_resume(close, period, from, state, GainLoss::cmo, out)
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
    gain_loss_resume_one(close, period, row, state, next, GainLoss::rsi)
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
    gain_loss_resume_one(close, period, row, state, next, GainLoss::cmo)
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
/// computed on its own rows (O(period)), and the slide is re-seeded from the
/// first clean window after it — a bad bar reaches only the windows holding it, and a
/// value never depends on the rows after it.
pub fn cog(close: &[f64], period: usize, _origin: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut state = [0.0; COG_STATE];
    for (i, o) in out.iter_mut().enumerate() {
        *o = cog_row(close, period, i, &mut state);
    }
    out
}

/// The length of [`cog`]'s state: `[Σ(1+age)·price, Σprice]` (each a [`Csum`]), the
/// non-finite bars in the window, the bars seen (capped), whether the sums describe the
/// window.
const COG_STATE: usize = 2 * Csum::STATE + 3;

/// One row of [`cog`] at `i`, advancing `st` (bars `0..i` already seen).
///
/// The first full window is seeded newest-first, then each bar slides it: dropping the
/// oldest (weight `period`) lowers every retained weight, so `num[i] = num[i-1] +
/// den[i-1] - (period+1)·leaving + close[i]` and `den[i] = den[i-1] - leaving +
/// close[i]` (the WMA-slide trick, mirrored to cog's newest-weighted-1 orientation).
/// The sums are exact and the weighted terms enter and leave exactly (see
/// [`crate::accum`]), so a window whose prices sum to zero reads exactly zero (`NaN`).
#[inline]
fn cog_row(close: &[f64], period: usize, i: usize, st: &mut [f64; COG_STATE]) -> f64 {
    let sum_at = |k: usize| Csum::from_state(&st[2 * k..2 * k + 2]).unwrap_or_default();
    let (mut num, mut den) = (sum_at(0), sum_at(1));
    let (bad, seen, valid) = (st[4], st[5], st[6]);
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
        let [num, den] = cog_sums(close, i, period);
        cog_value(num, den)
    } else {
        // Re-anchoring at an absolute position, or when a slid sum went inexact, is
        // re-seeding the window (see `crate::accum`).
        let slid = valid;
        if slid {
            let leaving = close[i - period];
            num.add_sum(&den);
            num.sub_prod(pf + 1.0, leaving);
            num.add(close[i]);
            den.sub(leaving);
            den.add(close[i]);
        }
        if !(slid && num.is_exact() && den.is_exact()) {
            [num, den] = cog_sums(close, i, period);
            valid = true;
        }
        cog_value(num, den)
    };
    num.write_state(&mut st[0..2]);
    den.write_state(&mut st[2..4]);
    st[4] = bad as f64;
    st[5] = (seen + 1).min(period + 1) as f64;
    st[6] = f64::from(u8::from(valid));
    value
}

/// `-num / den`, `NaN` for a window whose prices sum to zero.
#[inline]
fn cog_value(num: Csum, den: Csum) -> f64 {
    let d = den.value();
    if d != 0.0 {
        -num.value() / d
    } else {
        f64::NAN
    }
}

/// [`cog`]'s window sums `[Σ(1+age)·price, Σprice]` at `i`, seeded newest first —
/// exact (a window holding a non-finite bar takes their IEEE value). Out of the
/// slide's loop.
#[cold]
#[inline(never)]
fn cog_sums(close: &[f64], i: usize, period: usize) -> [Csum; 2] {
    Csum::settle(cog_seed(close, i, period), || cog_seed(close, i, period))
}

/// [`cog_sums`] over an [`Accumulator`].
fn cog_seed<A: Accumulator>(close: &[f64], i: usize, period: usize) -> [A; 2] {
    let (mut num, mut den) = (A::default(), A::default());
    for age in 0..period {
        let price = close[i - age]; // age 0 = newest
        num.add_prod((1 + age) as f64, price);
        den.add(price);
    }
    [num, den]
}

/// [`cog`]'s state after the rows of `close`.
pub fn cog_state(close: &[f64], period: usize, _origin: usize) -> Option<Vec<f64>> {
    crate::guard::state_after(vec![0.0; COG_STATE], close.len(), |row, prev, next| {
        cog_step(close, period, row, prev, next)
    })
}

/// One row of [`cog`] at `row`, from `state` into `next`.
pub fn cog_step(close: &[f64], period: usize, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    if period == 0 || row >= close.len() || next.len() != COG_STATE {
        return None;
    }
    let mut st = <[f64; COG_STATE]>::try_from(state).ok()?;
    if !crate::guard::holds_window(row, st[5] as usize, period) {
        return None;
    }
    let value = cog_row(close, period, row, &mut st);
    next.copy_from_slice(&st);
    Some(value)
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
            let st = stochrsi_final_state(head, rp, fk, false, fd, 0, 0).unwrap();
            let mut tail = Vec::new();
            let _ = stochrsi_resume(&close, rp, fk, false, fd, 0, from, &st, &mut tail).unwrap();
            assert_bits(&tail, &k_full[from..], "stochrsi.k");

            // `.d` line (is_d = true): the carried running sum of the `.d` SMA continues
            // the full frame's own, so the two agree bit for bit — also when the resume
            // is chained one row at a time from each returned state.
            let st = stochrsi_final_state(head, rp, fk, true, fd, 0, 0).unwrap();
            let mut tail = Vec::new();
            let _ = stochrsi_resume(&close, rp, fk, true, fd, 0, from, &st, &mut tail).unwrap();
            assert_bits(&tail, &d_full[from..], "stochrsi.d");
            let mut st = st;
            for row in from..close.len() {
                let mut one = Vec::new();
                let next = stochrsi_resume(&close[..=row], rp, fk, true, fd, 0, row, &st, &mut one).unwrap();
                assert_bits(&one, &d_full[row..=row], "stochrsi.d chained");
                st = next;
            }
            // the default fused `.d` kernel is the same SMA
            let d_default = crate::raw::indicators::stochrsi_d_default_sma(&close, 0).unwrap();
            let st = stochrsi_final_state(head, 14, 5, true, 3, 0, 0).unwrap();
            let mut tail = Vec::new();
            let _ = stochrsi_resume(&close, 14, 5, true, 3, 0, from, &st, &mut tail).unwrap();
            assert_bits(&tail, &d_default[from..], "stochrsi.d default");
        }
    }

    /// StochRSI guards: a too-short close (RSI never accrues `C+1` finite rows) declines
    /// the final state; a bad state length / `from` declines the resume; and an embedded
    /// NaN in the close keeps the resumed RSI rows NaN — the resume carries on exactly as
    /// the full computation does (each window holding a NaN RSI is NaN).
    #[test]
    fn stochrsi_guards_decline() {
        let (rp, fk, fd) = (14usize, 14usize, 3usize);

        // n < c || n - c < rsi_period -> final state declines (oscillators.rs:135).
        let short = series(20);
        assert!(stochrsi_final_state(&short, rp, fk, false, fd, 0, 0).is_none());

        // Bad state length / from -> resume declines (oscillators.rs:159).
        let close = series(200);
        let st = stochrsi_final_state(&close[..120], rp, fk, false, fd, 0, 0).unwrap();
        let bad = vec![0.0; 1]; // wrong length (!= c + 2)
        assert!(stochrsi_resume(&close, rp, fk, false, fd, 0, 120, &bad, &mut Vec::new()).is_none());
        assert!(stochrsi_resume(&close, rp, fk, false, fd, 0, 0, &st, &mut Vec::new()).is_none()); // from == 0

        // A NaN late in the close poisons the raw RSI from row 150 on: the resumed tail
        // equals the full computation's, NaN windows included.
        let mut nanclose = series(200);
        nanclose[150] = f64::NAN;
        let st = stochrsi_final_state(&nanclose[..140], rp, fk, false, fd, 0, 0).unwrap();
        let mut tail = Vec::new();
        let _ = stochrsi_resume(&nanclose, rp, fk, false, fd, 0, 140, &st, &mut tail).unwrap();
        let full = stochrsi_fastk(&nanclose, rp, fk);
        assert!(tail.iter().zip(&full[140..]).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(tail[10..].iter().all(|v| v.is_nan()));
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
            let mut tail = Vec::new();
            let _ = rsi_resume(&close, p, from, &st, &mut tail).unwrap();
            assert_bits(&tail, &rsi_full[from..], "rsi");

            let st = cmo_final_state(&close[..from], p).unwrap();
            let mut tail = Vec::new();
            let _ = cmo_resume(&close, p, from, &st, &mut tail).unwrap();
            assert_bits(&tail, &cmo_full[from..], "cmo");
        }

        // from == 0 -> both resumes decline (oscillators.rs:384, 440).
        let st = rsi_final_state(&close, p).unwrap();
        assert!(rsi_resume(&close, p, 0, &st, &mut Vec::new()).is_none());
        let st = cmo_final_state(&close, p).unwrap();
        assert!(cmo_resume(&close, p, 0, &st, &mut Vec::new()).is_none());

        // period == 0 / n <= period -> final states decline (oscillators.rs:355, 415).
        assert!(rsi_final_state(&[1.0, 2.0], 5).is_none());
        assert!(cmo_final_state(&[1.0, 2.0], 5).is_none());

        // Strictly increasing close: avg_loss == 0, so RSI's `emit` returns 100 in the
        // resume's flat-loss arm (oscillators.rs:391).
        let up: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let st = rsi_final_state(&up[..20], p).unwrap();
        let mut tail = Vec::new();
        let _ = rsi_resume(&up, p, 20, &st, &mut tail).unwrap();
        assert!(tail.iter().all(|&x| x == 100.0), "rsi flat-loss -> 100");

        // Flat close: gain + loss == 0, so CMO's resume returns 0 (oscillators.rs:453).
        let flat = vec![7.0; 40];
        let st = cmo_final_state(&flat[..20], p).unwrap();
        let mut tail = Vec::new();
        let _ = cmo_resume(&flat, p, 20, &st, &mut tail).unwrap();
        assert!(tail.iter().all(|&x| x == 0.0), "cmo flat-window -> 0");
    }
}
