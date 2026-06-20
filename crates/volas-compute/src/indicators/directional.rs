// ---------------------------------------------------------------------------
// Directional movement — Wilder's +DM/-DM/TR, ±DI, DX, ADX, ADXR
// ---------------------------------------------------------------------------
//
// All seven derive from one-period directional movement and true range, smoothed by
// Wilder's *sum* recurrence (not the SMA-seeded average used by ATR): the seed is the
// sum of the first `period-1` one-period values, then `s = s - s/period + term`.
// Faithful port of TA-Lib (which performs no integer rounding).

/// One-period +DM / -DM at bar `i` (needs the prior bar). Per Wilder: the larger of
/// the up-move (`high-prevHigh`) and down-move (`prevLow-low`) wins, the other is 0;
/// ties and non-positive moves yield 0 for both.
#[inline]
fn dm1(high: &[f64], low: &[f64], i: usize) -> (f64, f64) {
    let diff_p = high[i] - high[i - 1];
    let diff_m = low[i - 1] - low[i];
    let plus = if diff_p > 0.0 && diff_p > diff_m {
        diff_p
    } else {
        0.0
    };
    let minus = if diff_m > 0.0 && diff_p < diff_m {
        diff_m
    } else {
        0.0
    };
    (plus, minus)
}

/// One-period true range at bar `i` (needs the prior close).
#[inline]
fn tr1(high: &[f64], low: &[f64], close: &[f64], i: usize) -> f64 {
    let hl = high[i] - low[i];
    let hc = (high[i] - close[i - 1]).abs();
    let lc = (low[i] - close[i - 1]).abs();
    hl.max(hc).max(lc)
}

/// Wilder "sum" smoothing of a one-period `term`: seed `= Σ term(i)` over bars
/// `[1, period-1]`, placed at index `period-1`; then `s = s - s/period + term(i)`.
/// NaN before index `period-1`.
fn wilder_sum(n: usize, period: usize, term: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut out = vec![f64::NAN; n];
    if period == 0 || period >= n {
        return out;
    }
    let mut s = 0.0;
    for i in 1..period {
        s += term(i);
    }
    out[period - 1] = s;
    // `s - s/period + term` == `s*(1 - 1/period) + term`: precompute the factor and
    // fuse (mul_add), taking the per-element division off the recurrence's critical
    // path. Wilder smoothing is contractive, so the ~1e-16 reassociation decays —
    // within the 1e-9 parity tolerance. Speeds every directional indicator.
    let a = 1.0 - 1.0 / period as f64;
    #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
    for i in period..n {
        s = s.mul_add(a, term(i));
        out[i] = s;
    }
    out
}

/// Wilder-smoothed +DM (TA-Lib PLUS_DM). Lookback `period-1`.
pub fn plus_dm(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    wilder_sum(high.len(), period, |i| dm1(high, low, i).0)
}

/// Wilder-smoothed -DM (TA-Lib MINUS_DM). Lookback `period-1`.
pub fn minus_dm(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    wilder_sum(high.len(), period, |i| dm1(high, low, i).1)
}

/// `100 · smoothedDM / smoothedTR`, emitted from index `period` (one bar after the DM
/// seed), with a ~0 TR yielding 0. Shared by ±DI.
fn di(dm_sm: &[f64], tr_sm: &[f64], period: usize) -> Vec<f64> {
    let n = dm_sm.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        let t = tr_sm[i];
        out[i] = if t.abs() < 1e-14 {
            0.0
        } else {
            100.0 * dm_sm[i] / t
        };
    }
    out
}

/// +DI (TA-Lib PLUS_DI). Lookback `period`.
pub fn plus_di(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let sp = wilder_sum(high.len(), period, |i| dm1(high, low, i).0);
    let st = wilder_sum(high.len(), period, |i| tr1(high, low, close, i));
    di(&sp, &st, period)
}

/// -DI (TA-Lib MINUS_DI). Lookback `period`.
pub fn minus_di(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let sm = wilder_sum(high.len(), period, |i| dm1(high, low, i).1);
    let st = wilder_sum(high.len(), period, |i| tr1(high, low, close, i));
    di(&sm, &st, period)
}

/// Fused Wilder sums of +DM, −DM and TR in a single pass — `dm1` (which yields both
/// directional moves) is evaluated once per bar instead of twice, and the three
/// division-free recurrences share one traversal. Bit-identical to three
/// [`wilder_sum`] calls.
fn dm_tr_sums(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let (mut sp, mut sm, mut st) = (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    if period == 0 || period >= n {
        return (sp, sm, st);
    }
    let (mut p, mut m, mut t) = (0.0, 0.0, 0.0);
    for i in 1..period {
        let (dp, dm) = dm1(high, low, i);
        p += dp;
        m += dm;
        t += tr1(high, low, close, i);
    }
    sp[period - 1] = p;
    sm[period - 1] = m;
    st[period - 1] = t;
    let a = 1.0 - 1.0 / period as f64;
    for i in period..n {
        let (dp, dm) = dm1(high, low, i);
        p = p.mul_add(a, dp);
        m = m.mul_add(a, dm);
        t = t.mul_add(a, tr1(high, low, close, i));
        sp[i] = p;
        sm[i] = m;
        st[i] = t;
    }
    (sp, sm, st)
}

/// Directional Movement Index `DX = 100·|+DI − −DI| / (+DI + −DI)` (TA-Lib DX); a ~0
/// TR or ~0 DI-sum yields 0. Lookback `period`.
pub fn dx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let (sp, sm, st) = dm_tr_sums(high, low, close, period);
    let n = sp.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        let t = st[i];
        out[i] = if t.abs() < 1e-14 {
            0.0
        } else {
            let plus_di = 100.0 * sp[i] / t;
            let minus_di = 100.0 * sm[i] / t;
            let sum = plus_di + minus_di;
            if sum.abs() < 1e-14 {
                0.0
            } else {
                100.0 * (minus_di - plus_di).abs() / sum
            }
        };
    }
    out
}

/// Average Directional Movement Index (TA-Lib ADX): SMA-seeded Wilder *average* of DX
/// (`ADX[2p-1] = mean(DX[p..2p-1])`, then `ADX = (ADX_prev·(p-1) + DX)/p`).
/// Lookback `2·period-1`.
pub fn adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let dxv = dx(high, low, close, period);
    let n = dxv.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 {
        return out;
    }
    let first = 2 * period - 1;
    if first >= n {
        return out;
    }
    let pf = period as f64;
    let mut sum = 0.0;
    for v in dxv.iter().take(2 * period).skip(period) {
        sum += v;
    }
    let mut prev = sum / pf;
    out[first] = prev;
    // `(prev*(period-1) + dx)/period` as `prev*a + dx*b` (precomputed reciprocals,
    // fused): no per-element division on the ADX recurrence. Contractive -> within
    // the 1e-9 parity tolerance.
    let a = (pf - 1.0) / pf;
    let b = 1.0 / pf;
    for i in (2 * period)..n {
        prev = prev.mul_add(a, dxv[i] * b);
        out[i] = prev;
    }
    out
}

/// Average Directional Movement Index Rating: `(ADX[i] + ADX[i−(period−1)]) / 2`
/// (TA-Lib ADXR). Lookback `3·period-2`.
pub fn adxr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let a = adx(high, low, close, period);
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 {
        return out;
    }
    let off = period - 1;
    for i in (3 * period - 2)..n {
        out[i] = (a[i] + a[i - off]) / 2.0;
    }
    out
}

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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::test_support::*;

    /// Every directional resume, fed the carried Wilder accumulator(s) of a full compute
    /// over the head, reproduces the tail of a full compute over the whole input —
    /// bit-for-bit. Iterating `from` over an oscillating OHLC series advances the fused
    /// `dm1`/`tr1` recurrences through their steady state for all seven families.
    #[test]
    fn directional_resume_is_bit_identical_to_full() {
        let (high, low, close) = ohlc(120);
        let p = 14usize;

        // ±DM warm up at `period-1`; ±DI/DX at `period`; ADX at `2p-1`; ADXR at `3p-2`.
        // Pick `from` values comfortably past each family's seed (the largest seed is
        // ADXR's 3*14-2 = 40), so every head produces a valid carried state.
        for &from in &[41usize, 45, 60, 90, 119] {
            let h = &high[..from];
            let l = &low[..from];
            let c = &close[..from];

            let st = plus_dm_final_state(h, l, p).unwrap();
            let (tail, _) = plus_dm_resume(&high, &low, p, from, &st).unwrap();
            assert_bits(&tail, &plus_dm(&high, &low, p)[from..], "plus_dm");

            let st = minus_dm_final_state(h, l, p).unwrap();
            let (tail, _) = minus_dm_resume(&high, &low, p, from, &st).unwrap();
            assert_bits(&tail, &minus_dm(&high, &low, p)[from..], "minus_dm");

            let st = plus_di_final_state(h, l, c, p).unwrap();
            let (tail, _) = plus_di_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &plus_di(&high, &low, &close, p)[from..], "plus_di");

            let st = minus_di_final_state(h, l, c, p).unwrap();
            let (tail, _) = minus_di_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &minus_di(&high, &low, &close, p)[from..], "minus_di");

            let st = dx_final_state(h, l, c, p).unwrap();
            let (tail, _) = dx_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &dx(&high, &low, &close, p)[from..], "dx");

            let st = adx_final_state(h, l, c, p).unwrap();
            let (tail, _) = adx_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &adx(&high, &low, &close, p)[from..], "adx");

            let st = adxr_final_state(h, l, c, p).unwrap();
            let (tail, _) = adxr_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &adxr(&high, &low, &close, p)[from..], "adxr");
        }
    }

    /// `from == 0` resumes decline (no prior bar to read the one-period term from) — the
    /// early `None` arm of every `*_resume`.
    #[test]
    fn directional_resume_declines_at_from_zero() {
        let (high, low, close) = ohlc(60);
        let p = 14usize;
        let dummy = vec![0.0; 8]; // longer than any family's state, unread on the None path
        assert!(plus_dm_resume(&high, &low, p, 0, &dummy).is_none()); // :284
        assert!(minus_dm_resume(&high, &low, p, 0, &dummy).is_none()); // :305
        assert!(plus_di_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :378
        assert!(minus_di_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :409
        assert!(dx_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :457
        assert!(adx_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :525
        assert!(adxr_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :605
    }

    /// `*_final_state` declines before the indicator seeds, so the caller keeps the
    /// full-recompute fallback: `period >= n` for the Wilder-sum families, `period == 0`
    /// and the staged `2p-1 >= n` / `3p-2 >= n` seeds for ADX / ADXR.
    #[test]
    fn directional_final_state_declines_before_seed() {
        let (high, low, close) = ohlc(120);
        // period >= n -> the Wilder sum never seeds.
        assert!(plus_dm_final_state(&high, &low, 200).is_none()); // :236
        assert!(minus_dm_final_state(&high, &low, 200).is_none());
        assert!(plus_di_final_state(&high, &low, &close, 200).is_none()); // :322 (via dm_tr_sums_final)
        assert!(minus_di_final_state(&high, &low, &close, 200).is_none());
        assert!(dx_final_state(&high, &low, &close, 200).is_none());
        // ADX: period == 0 (:476) and 2*period-1 >= n (:480).
        assert!(adx_final_state(&high, &low, &close, 0).is_none());
        assert!(adx_final_state(&high, &low, &close, 70).is_none());
        // ADXR: period == 0 (:550) and 3*period-2 >= n (:553).
        assert!(adxr_final_state(&high, &low, &close, 0).is_none());
        assert!(adxr_final_state(&high, &low, &close, 50).is_none());
    }

    /// The TA-Lib ~0-divisor guards in `di_val` / `dx_val` — which are reached only on the
    /// RESUME paths (the public `plus_di`/`dx` use their own inline `di`/loop). Drive the
    /// resumes with degenerate inputs so every zero arm fires.
    #[test]
    fn di_dx_zero_divisor_guards() {
        let p = 14usize;

        // Flat series (constant high == low == close): smoothed TR ~ 0, so di_val (:355)
        // and dx_val's outer guard (:427) both return 0 on resume.
        let flat = vec![50.0; 40];
        let from = 20usize;
        let st = plus_di_final_state(&flat[..from], &flat[..from], &flat[..from], p).unwrap();
        let (tail, _) = plus_di_resume(&flat, &flat, &flat, p, from, &st).unwrap();
        assert!(
            tail.iter().all(|&x| x == 0.0),
            "plus_di resume flat TR -> 0"
        ); // :355
        let st = minus_di_final_state(&flat[..from], &flat[..from], &flat[..from], p).unwrap();
        let (tail, _) = minus_di_resume(&flat, &flat, &flat, p, from, &st).unwrap();
        assert!(
            tail.iter().all(|&x| x == 0.0),
            "minus_di resume flat TR -> 0"
        );
        let st = dx_final_state(&flat[..from], &flat[..from], &flat[..from], p).unwrap();
        let (tail, _) = dx_resume(&flat, &flat, &flat, p, from, &st).unwrap();
        assert!(tail.iter().all(|&x| x == 0.0), "dx resume flat TR -> 0"); // :427

        // +DI + −DI ~ 0 while TR > 0 -> dx_val's inner sum guard (:433). Strictly shrinking
        // inside bars: each high lower than the prior high AND each low higher than the
        // prior low, so every one-period +DM and −DM is 0 (no new extreme), yet the
        // high-low range stays positive so the smoothed TR is non-zero.
        let n = 40usize;
        let high: Vec<f64> = (0..n).map(|i| 100.0 - i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..n).map(|i| 60.0 + i as f64 * 0.5).collect();
        let close: Vec<f64> = (0..n).map(|i| 80.0 + (i as f64 * 0.3).sin()).collect();
        // Sanity: the constructed bars really are inside bars with positive range.
        for i in 1..n {
            assert!(high[i] < high[i - 1] && low[i] > low[i - 1] && high[i] > low[i]);
        }
        let st = dx_final_state(&high[..from], &low[..from], &close[..from], p).unwrap();
        let (tail, _) = dx_resume(&high, &low, &close, p, from, &st).unwrap();
        assert!(tail.iter().all(|&x| x == 0.0), "dx resume zero DI-sum -> 0"); // :433
    }
}
