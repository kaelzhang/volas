//! Directional-family state carry: the `*_final_state` captures and the `*_resume` /
//! `*_resume_one` continuations that let a cached ±DM / ±DI / DX / ADX / ADXR column
//! extend over appended rows without a full recompute.

use super::*;

// --- Directional-family state-carry (additive; the full-recompute fallback stays
// correct) --------------------------------------------------------------------------
//
// Every directional indicator is a Wilder recurrence over per-bar terms (`dm1` / `tr1`)
// that each read the *prior* bar. The carried state is the running Wilder accumulator(s)
// as of row `from-1`: a single sum (±DM), a sum pair (±DI), a sum triple (DX), the DX
// triple plus the running ADX average (ADX), and additionally the trailing `period`-long
// ADX window (ADXR, which looks `period-1` rows back). `*_final_state` re-runs the exact
// kernel recurrence to capture that state after a full compute (returning `None` before
// the indicator seeds, so the caller keeps the correct fallback); `*_resume` continues
// over rows `[from, n)` reading only `high/low/close[from-1..]`, with arithmetic
// bit-identical to the kernels above. A resume at `from == 0` (no prior bar to read)
// returns `None`, falling back to the full recompute — the plumbing only captures state
// on a seeded column, so `from >= 1` always holds in practice.

/// The Wilder-`sum` running accumulator after seeding at `period-1` and folding every
/// later term — i.e. the value of `s` once `wilder_sum` has consumed all `n` rows. `None`
/// before the seed exists (`period == 0 || period >= n`). `term(i)` is the per-bar
/// one-period quantity (`dm1.0` / `dm1.1` / `tr1`).
fn wilder_sum_final(n: usize, period: usize, term: impl Fn(usize) -> f64) -> Option<f64> {
    if period == 0 || period >= n {
        return None;
    }
    let mut s = 0.0;
    for i in 1..period {
        s += term(i);
    }
    let a = 1.0 - 1.0 / period as f64;
    for i in period..n {
        s = s.mul_add(a, term(i));
    }
    Some(s)
}

/// Resume a Wilder-`sum` from carried `s = state` over rows `[from, n)`, producing each
/// row's running sum and the new `s`. Reads only the bars `[from-1, n)` (each `term(i)`
/// needs bar `i-1`). Bit-identical to [`wilder_sum`]'s steady-state `s = s·a + term`.
fn wilder_sum_resume(
    n: usize,
    period: usize,
    from: usize,
    state: f64,
    term: impl Fn(usize) -> f64,
) -> (Vec<f64>, f64) {
    let a = 1.0 - 1.0 / period as f64;
    let mut s = state;
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        s = s.mul_add(a, term(i));
        out.push(s);
    }
    (out, s)
}

/// Final +DM state `[s]` (running Wilder sum) — pairs with [`plus_dm_resume`].
pub fn plus_dm_final_state(high: &[f64], low: &[f64], period: usize) -> Option<Vec<f64>> {
    // A leading run of missing rows: the series starts after it (see `crate::start`).
    let start = crate::start::first_complete_row(&[high, low]);
    if start > 0 {
        return plus_dm_final_state(&high[start..], &low[start..], period);
    }
    wilder_sum_final(high.len(), period, |i| dm1(high, low, i).0).map(|s| vec![s])
}

/// Resume [`plus_dm`] from `state = [s_{from-1}]` over rows `[from, n)`. `None` at
/// `from == 0` (no prior bar). Reads only `high/low[from-1..]`.
pub fn plus_dm_resume(
    high: &[f64],
    low: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from == 0 {
        return None;
    }
    let (out, s) = wilder_sum_resume(high.len(), period, from, state[0], |i| dm1(high, low, i).0);
    Some((out, vec![s]))
}

/// Final −DM state `[s]` — pairs with [`minus_dm_resume`].
pub fn minus_dm_final_state(high: &[f64], low: &[f64], period: usize) -> Option<Vec<f64>> {
    // A leading run of missing rows: the series starts after it (see `crate::start`).
    let start = crate::start::first_complete_row(&[high, low]);
    if start > 0 {
        return minus_dm_final_state(&high[start..], &low[start..], period);
    }
    wilder_sum_final(high.len(), period, |i| dm1(high, low, i).1).map(|s| vec![s])
}

/// Resume [`minus_dm`] from `state = [s_{from-1}]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `high/low[from-1..]`.
pub fn minus_dm_resume(
    high: &[f64],
    low: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from == 0 {
        return None;
    }
    let (out, s) = wilder_sum_resume(high.len(), period, from, state[0], |i| dm1(high, low, i).1);
    Some((out, vec![s]))
}

/// Final ±DI / DX / ADX shared state: the Wilder-`sum` triple `[sp, sm, st]` as of the
/// last row (the +DM, −DM and TR running sums). `None` before the seed exists. The
/// shared seed every directional ratio resumes from.
fn dm_tr_sums_final(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Option<(f64, f64, f64)> {
    let n = high.len();
    if period == 0 || period >= n {
        return None;
    }
    let (mut p, mut m, mut t) = (0.0, 0.0, 0.0);
    for i in 1..period {
        let (dp, dm) = dm1(high, low, i);
        p += dp;
        m += dm;
        t += tr1(high, low, close, i);
    }
    let a = 1.0 - 1.0 / period as f64;
    for i in period..n {
        let (dp, dm) = dm1(high, low, i);
        p = p.mul_add(a, dp);
        m = m.mul_add(a, dm);
        t = t.mul_add(a, tr1(high, low, close, i));
    }
    Some((p, m, t))
}

/// Advance the +DM/−DM/TR Wilder sums `s = [sp, sm, st]` one bar at index `i` with factor
/// `a = 1 - 1/period` (the exact `dm_tr_sums` steady-state step).
#[inline]
fn dm_tr_step(high: &[f64], low: &[f64], close: &[f64], i: usize, a: f64, s: &mut [f64; 3]) {
    let (dp, dm) = dm1(high, low, i);
    s[0] = s[0].mul_add(a, dp);
    s[1] = s[1].mul_add(a, dm);
    s[2] = s[2].mul_add(a, tr1(high, low, close, i));
}

/// `100·dm_sm/tr_sm` with TA-Lib's ~0-TR guard — the per-row `di` value.
#[inline]
fn di_val(dm_sm: f64, tr_sm: f64) -> f64 {
    if tr_sm.abs() < 1e-14 {
        0.0
    } else {
        100.0 * dm_sm / tr_sm
    }
}

/// Final +DI state `[sp, st]` (+DM and TR running sums) — pairs with [`plus_di_resume`].
pub fn plus_di_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Option<Vec<f64>> {
    // A leading run of missing rows: the series starts after it (see `crate::start`).
    let start = crate::start::first_complete_row(&[high, low, close]);
    if start > 0 {
        return plus_di_final_state(&high[start..], &low[start..], &close[start..], period);
    }
    let (sp, _sm, st) = dm_tr_sums_final(high, low, close, period)?;
    Some(vec![sp, st])
}

/// Resume [`plus_di`] from `state = [sp, st]` over rows `[from, n)`. `None` at `from == 0`.
/// Reads only `high/low/close[from-1..]`.
pub fn plus_di_resume(
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
    let a = 1.0 - 1.0 / period as f64;
    // s = [sp, sm, st]; +DI ignores the −DM component but shares the fused step.
    let mut s = [state[0], 0.0, state[1]];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        dm_tr_step(high, low, close, i, a, &mut s);
        out.push(di_val(s[0], s[2]));
    }
    Some((out, vec![s[0], s[2]]))
}

/// Final −DI state `[sm, st]` (−DM and TR running sums) — pairs with [`minus_di_resume`].
pub fn minus_di_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Option<Vec<f64>> {
    // A leading run of missing rows: the series starts after it (see `crate::start`).
    let start = crate::start::first_complete_row(&[high, low, close]);
    if start > 0 {
        return minus_di_final_state(&high[start..], &low[start..], &close[start..], period);
    }
    let (_sp, sm, st) = dm_tr_sums_final(high, low, close, period)?;
    Some(vec![sm, st])
}

/// Resume [`minus_di`] from `state = [sm, st]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `high/low/close[from-1..]`.
pub fn minus_di_resume(
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
    let a = 1.0 - 1.0 / period as f64;
    // s = [sp, sm, st]; −DI ignores the +DM component but shares the fused step.
    let mut s = [0.0, state[0], state[1]];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        dm_tr_step(high, low, close, i, a, &mut s);
        out.push(di_val(s[1], s[2]));
    }
    Some((out, vec![s[1], s[2]]))
}

/// The per-row DX value from the +DM/−DM/TR sums — the exact arithmetic of [`dx`].
#[inline]
fn dx_val(sp: f64, sm: f64, st: f64) -> f64 {
    if st.abs() < 1e-14 {
        0.0
    } else {
        let plus_di = 100.0 * sp / st;
        let minus_di = 100.0 * sm / st;
        let sum = plus_di + minus_di;
        if sum.abs() < 1e-14 {
            0.0
        } else {
            100.0 * (minus_di - plus_di).abs() / sum
        }
    }
}

/// Final DX state `[sp, sm, st]` (all three running sums) — pairs with [`dx_resume`].
pub fn dx_final_state(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Option<Vec<f64>> {
    // A leading run of missing rows: the series starts after it (see `crate::start`).
    let start = crate::start::first_complete_row(&[high, low, close]);
    if start > 0 {
        return dx_final_state(&high[start..], &low[start..], &close[start..], period);
    }
    let (sp, sm, st) = dm_tr_sums_final(high, low, close, period)?;
    Some(vec![sp, sm, st])
}

/// Resume [`dx`] from `state = [sp, sm, st]` over rows `[from, n)`. `None` at `from == 0`.
/// Reads only `high/low/close[from-1..]`.
pub fn dx_resume(
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
    let a = 1.0 - 1.0 / period as f64;
    let mut s = [state[0], state[1], state[2]];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        dm_tr_step(high, low, close, i, a, &mut s);
        out.push(dx_val(s[0], s[1], s[2]));
    }
    Some((out, s.to_vec()))
}

/// Final ADX state `[sp, sm, st, adx]`: the DX running sums plus the running ADX average
/// (the SMA-seeded Wilder average of DX) as of the last row. `None` before ADX seeds
/// (`2·period-1 >= n`), so the caller keeps the fallback. Pairs with [`adx_resume`].
pub fn adx_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Option<Vec<f64>> {
    // A leading run of missing rows: the series starts after it (see `crate::start`).
    let start = crate::start::first_complete_row(&[high, low, close]);
    if start > 0 {
        return adx_final_state(&high[start..], &low[start..], &close[start..], period);
    }
    let n = high.len();
    if period == 0 {
        return None;
    }
    let first = 2 * period - 1;
    if first >= n {
        return None;
    }
    // Re-run dx + its Wilder-average exactly as `adx`, carrying the DX sums alongside so
    // a resume can keep producing dx[i] without re-seeding.
    let a_sum = 1.0 - 1.0 / period as f64;
    let mut s = [0.0, 0.0, 0.0];
    for i in 1..period {
        let (dp, dm) = dm1(high, low, i);
        s[0] += dp;
        s[1] += dm;
        s[2] += tr1(high, low, close, i);
    }
    // dx[i] is defined from i == period; collect the first `period` DX values (indices
    // `[period, 2*period)`) to seed the ADX average, advancing the sums as we go.
    let pf = period as f64;
    let mut dx_seed_sum = 0.0;
    let mut adx = f64::NAN;
    let (a_avg, b_avg) = ((pf - 1.0) / pf, 1.0 / pf);
    for i in period..n {
        dm_tr_step(high, low, close, i, a_sum, &mut s);
        let dxv = dx_val(s[0], s[1], s[2]);
        if i < 2 * period {
            dx_seed_sum += dxv;
            if i == 2 * period - 1 {
                adx = dx_seed_sum / pf;
            }
        } else {
            adx = adx.mul_add(a_avg, dxv * b_avg);
        }
    }
    Some(vec![s[0], s[1], s[2], adx])
}

/// Resume [`adx`] from `state = [sp, sm, st, adx]` over rows `[from, n)`. `None` at
/// `from == 0`. Reads only `high/low/close[from-1..]`. Advances the DX sums, recomputes
/// each dx, then folds it into the Wilder ADX average — bit-identical to [`adx`].
pub fn adx_resume(
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
    let a_sum = 1.0 - 1.0 / pf;
    let (a_avg, b_avg) = ((pf - 1.0) / pf, 1.0 / pf);
    let mut s = [state[0], state[1], state[2]];
    let mut adx = state[3];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        dm_tr_step(high, low, close, i, a_sum, &mut s);
        let dxv = dx_val(s[0], s[1], s[2]);
        adx = adx.mul_add(a_avg, dxv * b_avg);
        out.push(adx);
    }
    Some((out, vec![s[0], s[1], s[2], adx]))
}

/// Final ADXR state `[sp, sm, st, adx_{i-(period-1)}, …, adx_i]`: the DX sums followed by
/// the trailing `period` ADX values (oldest→newest), so a resume can form
/// `(adx[i] + adx[i-(period-1)])/2`. `None` before ADXR seeds (`3·period-2 >= n`). Pairs
/// with [`adxr_resume`].
pub fn adxr_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Option<Vec<f64>> {
    // A leading run of missing rows: the series starts after it (see `crate::start`).
    let start = crate::start::first_complete_row(&[high, low, close]);
    if start > 0 {
        return adxr_final_state(&high[start..], &low[start..], &close[start..], period);
    }
    let n = high.len();
    if period == 0 {
        return None;
    }
    if 3 * period - 2 >= n {
        return None;
    }
    // Recompute the full ADX series (needed for the trailing window) alongside the live
    // DX sums; capture the last `period` ADX values.
    let a_sum = 1.0 - 1.0 / period as f64;
    let mut s = [0.0, 0.0, 0.0];
    for i in 1..period {
        let (dp, dm) = dm1(high, low, i);
        s[0] += dp;
        s[1] += dm;
        s[2] += tr1(high, low, close, i);
    }
    let pf = period as f64;
    let (a_avg, b_avg) = ((pf - 1.0) / pf, 1.0 / pf);
    let mut adxv = vec![f64::NAN; n];
    let mut dx_seed_sum = 0.0;
    let mut adx = 0.0;
    #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
    for i in period..n {
        dm_tr_step(high, low, close, i, a_sum, &mut s);
        let dxv = dx_val(s[0], s[1], s[2]);
        if i < 2 * period {
            dx_seed_sum += dxv;
            if i == 2 * period - 1 {
                adx = dx_seed_sum / pf;
                adxv[i] = adx;
            }
        } else {
            adx = adx.mul_add(a_avg, dxv * b_avg);
            adxv[i] = adx;
        }
    }
    let mut state = vec![s[0], s[1], s[2]];
    // Trailing `period` ADX values ending at row n-1 (oldest first). They are all valid
    // because n-1 >= 3*period-2 >= 2*period-1 + (period-1), so the window starts at or
    // after the ADX seed.
    state.extend_from_slice(&adxv[n - period..n]);
    Some(state)
}

/// Resume [`adxr`] from `state = [sp, sm, st, adx window…]` over rows `[from, n)`. `None`
/// at `from == 0`. Reads only `high/low/close[from-1..]`. Advances the DX sums + ADX
/// average per bar, maintaining the trailing `period`-long ADX window so each output is
/// `(adx[i] + adx[i-(period-1)])/2`, bit-identical to [`adxr`].
pub fn adxr_resume(
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
    let a_sum = 1.0 - 1.0 / pf;
    let (a_avg, b_avg) = ((pf - 1.0) / pf, 1.0 / pf);
    let mut s = [state[0], state[1], state[2]];
    // `win` carries the last `period` ADX values, adx[from-period .. from-1] (oldest
    // first), so adx[from-1] (the newest) is `win[period-1]`.
    let mut win: Vec<f64> = state[3..].to_vec();
    let mut adx = win[period - 1]; // the newest carried ADX (= adx[from-1])
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for i in from..n {
        dm_tr_step(high, low, close, i, a_sum, &mut s);
        let dxv = dx_val(s[0], s[1], s[2]);
        adx = adx.mul_add(a_avg, dxv * b_avg);
        // Slide the window forward to include adx[i], then `win[0]` is adx[i-(period-1)].
        win.remove(0);
        win.push(adx);
        out.push((adx + win[0]) / 2.0);
    }
    Some((out, {
        let mut st = vec![s[0], s[1], s[2]];
        st.extend_from_slice(&win);
        st
    }))
}


/// Scalar single-row twin of [`plus_di_resume`]: the +DI value at `row` from
/// `state = [sp_{row-1}, st_{row-1}]`, no allocation. Advances the fused +DM/TR sums one
/// bar then applies `di_val`, bit-identical to the Vec loop body.
pub fn plus_di_resume_one(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || state.len() < 2 || row >= high.len() {
        return None;
    }
    let a = 1.0 - 1.0 / period as f64;
    // Mirror the Vec resume's `s = [sp, 0, st]`; −DM slot stays 0 (unused by +DI).
    let mut s = [state[0], 0.0, state[1]];
    dm_tr_step(high, low, close, row, a, &mut s);
    Some(di_val(s[0], s[2]))
}

/// Scalar single-row twin of [`minus_di_resume`]: the −DI value at `row` from
/// `state = [sm_{row-1}, st_{row-1}]`, no allocation. Bit-identical to the Vec loop body.
pub fn minus_di_resume_one(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || state.len() < 2 || row >= high.len() {
        return None;
    }
    let a = 1.0 - 1.0 / period as f64;
    // Mirror the Vec resume's `s = [0, sm, st]`; +DM slot stays 0 (unused by −DI).
    let mut s = [0.0, state[0], state[1]];
    dm_tr_step(high, low, close, row, a, &mut s);
    Some(di_val(s[1], s[2]))
}

/// Scalar single-row twin of [`dx_resume`]: the DX value at `row` from
/// `state = [sp, sm, st]` (as of row-1), no allocation. Advances all three fused sums one
/// bar then applies `dx_val`, bit-identical to the Vec loop body.
pub fn dx_resume_one(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || state.len() < 3 || row >= high.len() {
        return None;
    }
    let a = 1.0 - 1.0 / period as f64;
    let mut s = [state[0], state[1], state[2]];
    dm_tr_step(high, low, close, row, a, &mut s);
    Some(dx_val(s[0], s[1], s[2]))
}

/// Scalar single-row twin of [`adx_resume`]: the ADX value at `row` from
/// `state = [sp, sm, st, adx]` (as of row-1), no allocation. Advances the DX sums, forms
/// `dx[row]`, then folds it into the Wilder ADX average — bit-identical to the Vec body.
pub fn adx_resume_one(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || state.len() < 4 || row >= high.len() {
        return None;
    }
    let pf = period as f64;
    let a_sum = 1.0 - 1.0 / pf;
    let (a_avg, b_avg) = ((pf - 1.0) / pf, 1.0 / pf);
    let mut s = [state[0], state[1], state[2]];
    dm_tr_step(high, low, close, row, a_sum, &mut s);
    let dxv = dx_val(s[0], s[1], s[2]);
    Some(state[3].mul_add(a_avg, dxv * b_avg))
}

/// Scalar single-row twin of [`adxr_resume`]: the ADXR value at `row` from
/// `state = [sp, sm, st, adx[row-period .. row-1]]`, no allocation. Advances the DX sums +
/// ADX average one bar, then forms `(adx[row] + adx[row-(period-1)])/2`, bit-identical to
/// the Vec loop body — without the Vec's `win.remove(0)`/`push` (O(1) index instead).
pub fn adxr_resume_one(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || period == 0 || state.len() < 3 + period || row >= high.len() {
        return None;
    }
    let pf = period as f64;
    let a_sum = 1.0 - 1.0 / pf;
    let (a_avg, b_avg) = ((pf - 1.0) / pf, 1.0 / pf);
    let mut s = [state[0], state[1], state[2]];
    dm_tr_step(high, low, close, row, a_sum, &mut s);
    let dxv = dx_val(s[0], s[1], s[2]);
    // newest carried adx = adx[row-1] = state[3 + period - 1]; fold to get adx[row].
    let adx = state[3 + period - 1].mul_add(a_avg, dxv * b_avg);
    // After the conceptual window slide, win[0] = adx[row-(period-1)]:
    //  period >= 2 -> old window's win[1] = state[4]; period == 1 -> the just-formed adx.
    let oldest = if period >= 2 { state[4] } else { adx };
    Some((adx + oldest) / 2.0)
}



/// Scalar single-row twin of [`plus_dm_resume`]: the running Wilder-sum at `row` from
/// `state = [s_{row-1}]`, no allocation. Bit-identical to one `wilder_sum` steady step.
pub fn plus_dm_resume_one(
    high: &[f64],
    low: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || state.is_empty() || row >= high.len() {
        return None;
    }
    let a = 1.0 - 1.0 / period as f64;
    Some(state[0].mul_add(a, dm1(high, low, row).0))
}

/// Scalar single-row twin of [`minus_dm_resume`]: running Wilder-sum at `row` from
/// `state = [s_{row-1}]`, no allocation. Bit-identical to one `wilder_sum` step.
pub fn minus_dm_resume_one(
    high: &[f64],
    low: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row == 0 || state.is_empty() || row >= high.len() {
        return None;
    }
    let a = 1.0 - 1.0 / period as f64;
    Some(state[0].mul_add(a, dm1(high, low, row).1))
}
