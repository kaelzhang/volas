//! Kaufman Adaptive Moving Average (KAMA) and its adaptive smoothing constant.

use crate::accum::Csum;

/// Kaufman Adaptive Moving Average (TA-Lib KAMA). The smoothing constant adapts each
/// bar via the efficiency ratio `ER = |price[i]-price[i-period]| / Σ|1-bar changes|`:
/// `SC = (ER·(fast−slow) + slow)²` with fast `=2/3`, slow `=2/31`; then
/// `KAMA[i] = KAMA[i-1] + SC·(price[i] − KAMA[i-1])`, seeded `KAMA[period-1] =
/// price[period-1]`. First value at index `period` (lookback = period). TA-Lib's
/// sliding-sum recurrence, the sum of 1-bar changes exact ([`Csum`], re-summed where it
/// could not hold its sum) so a window's efficiency ratio is its own however long the
/// slide — which is also why it does not depend on `_origin`, the frame's position
/// every kernel takes. Every path — this, [`kama_final_state`], [`kama_resume`],
/// [`kama_resume_one`] — advances by the one [`kama_bar`].
pub fn kama(data: &[f64], period: usize, _origin: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period + 1 > n {
        return out;
    }
    // The caller runs KAMA on the complete rows of its input (see `crate::missing`).
    let mut k = KamaState::seed(data, period);
    for (today, o) in out.iter_mut().enumerate().skip(period) {
        *o = kama_bar(data, period, today, &mut k);
    }
    out
}

/// KAMA's running state as of a bar: the KAMA value and the sum of the window's 1-bar
/// changes.
#[derive(Clone, Copy, Debug)]
struct KamaState {
    kama: f64,
    sum_roc1: Csum,
}

/// The length of a KAMA state: `[kama, sum, comp]`.
const KAMA_STATE: usize = 1 + Csum::STATE;

impl KamaState {
    /// The state before bar `period` (the first KAMA): the prior KAMA seeded with
    /// yesterday's price, the sum not yet formed (the first bar forms it).
    fn seed(data: &[f64], period: usize) -> Self {
        KamaState { kama: data[period - 1], sum_roc1: Csum::default() }
    }

    fn from_state(state: &[f64]) -> Option<Self> {
        let [kama, sum, comp] = *state else { return None };
        Some(KamaState { kama, sum_roc1: Csum::from_state(&[sum, comp])? })
    }

    fn write_state(&self, out: &mut [f64]) {
        out[0] = self.kama;
        self.sum_roc1.write_state(&mut out[1..3]);
    }

    fn to_vec(self) -> Vec<f64> {
        let mut v = vec![0.0; KAMA_STATE];
        self.write_state(&mut v);
        v
    }
}

/// The 1-bar change `|price[j] − price[j−1]|` KAMA sums.
#[inline(always)]
fn change(data: &[f64], j: usize) -> f64 {
    (data[j] - data[j - 1]).abs()
}

/// Advance KAMA over bar `today` (`>= period`): slide the sum of 1-bar changes (drop
/// the oldest, add the newest — or form it from the window at the first bar and where
/// the sum could not hold itself exactly), then step the recurrence. Returns the KAMA at
/// `today`.
#[inline]
fn kama_bar(data: &[f64], period: usize, today: usize, k: &mut KamaState) -> f64 {
    if today != period {
        k.sum_roc1.sub(change(data, today - period)); // drop the oldest 1-bar change
        k.sum_roc1.add(change(data, today)); // add the newest
    }
    if today == period || !k.sum_roc1.is_exact() {
        k.sum_roc1 = Csum::of((today + 1 - period..=today).map(|j| change(data, j)));
    }
    let period_roc = data[today] - data[today - period];
    let sc = kama_sc(period_roc, k.sum_roc1.value());
    // `(price − prev)·sc + prev` fused: one rounding off the recurrence's critical
    // path. KAMA is contractive (sc ≤ (2/3)² < 1), so the ~1e-16 divergence decays —
    // within the 1e-9 TA-Lib parity tolerance.
    k.kama = (data[today] - k.kama).mul_add(sc, k.kama);
    k.kama
}

/// KAMA's adaptive smoothing constant `SC = (ER·(fast−slow) + slow)²` from the period
/// ROC and the sliding sum of 1-bar changes, with `ER` clamped to 1 on a tiny/degenerate
/// denominator (TA-Lib's guard) (`fast = 2/3`, `slow = 2/31`).
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

/// Final KAMA state `[kama, sum_roc1, comp]` as of the last row: the KAMA value at
/// `n−1` and the exact sum of 1-bar changes for that row's window (`_origin`, the
/// frame's position, does not enter it). `None` if KAMA
/// never seeds (`period+1 > n`, or a leading-NaN prefix leaves too few finite values →
/// all-NaN, keep the fallback).
pub fn kama_final_state(data: &[f64], period: usize, _origin: usize) -> Option<Vec<f64>> {
    let n = data.len();
    if period == 0 || period + 1 > n {
        return None;
    }
    // Mirror `kama`'s leading-NaN skip: seed over the first finite values.
    let start = data.iter().position(|x| !x.is_nan()).unwrap_or(n);
    if start > 0 {
        return kama_final_state(&data[start..], period, 0);
    }
    let mut k = KamaState::seed(data, period);
    for today in period..n {
        kama_bar(data, period, today, &mut k);
    }
    Some(k.to_vec())
}

/// Resume [`kama`] from `state` (as of `from-1`, see [`kama_final_state`]) over rows
/// `[from, n)`, bit-identical to a full recompute. Reads the trailing window
/// `data[from-period-1 ..= from-1]` plus `data[from..]`. Returns `None` (→ caller's
/// full-recompute fallback) when `from <= period` (the trailing window would underflow)
/// or the state is malformed.
pub fn kama_resume(
    data: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = data.len();
    if period == 0 || from <= period || from > n {
        return None;
    }
    let mut k = KamaState::from_state(state)?;
    let out = (from..n).map(|today| kama_bar(data, period, today, &mut k)).collect();
    Some((out, k.to_vec()))
}

/// Single new-row [`kama`] resume: the KAMA value at `row` from `state` (as of
/// `row-1`), the state as of `row` written into `next`, with NO allocation —
/// bit-identical to one bar of [`kama_resume`]. `None` on the same guard as
/// [`kama_resume`] plus a `state` / `next` of the wrong length / an out-of-range `row`.
pub fn kama_resume_one(
    data: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let n = data.len();
    if period == 0 || row <= period || row >= n || next.len() != KAMA_STATE {
        return None;
    }
    let mut k = KamaState::from_state(state)?;
    let value = kama_bar(data, period, row, &mut k);
    k.write_state(next);
    Some(value)
}
