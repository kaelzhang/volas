//! Kaufman Adaptive Moving Average (KAMA) and its adaptive smoothing constant.

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
pub fn kama_resume(
    data: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
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
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
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
