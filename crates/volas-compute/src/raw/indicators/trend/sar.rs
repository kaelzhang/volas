//! Parabolic SAR and the extended SAREXT, with their state-carry resume.

// `!(minus_dm1 > 0.0)` is exact TA-Lib parity: the initial trend is long unless −DM1
// is strictly positive, so a zero *or NaN* −DM1 goes long. Rewriting to `<= 0.0` would
// flip the NaN case to short. The negated form is deliberate, not a style slip.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

/// Parabolic SAR (TA-Lib SAR). Faithful port of TA-Lib's recurrence: initial trend is
/// chosen from the first bar's −DM1; each step trails the stop by `af·(ep − sar)`,
/// `af` ramping by `acceleration` (capped at `maximum`) on every new extreme, resetting
/// on a reversal; the SAR is clamped within the prior two bars' range. Default
/// acceleration 0.02, maximum 0.2; lookback 1. (TA-Lib applies no rounding.)
pub fn sar(high: &[f64], low: &[f64], acceleration: f64, maximum: f64) -> Vec<f64> {
    let n = high.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 {
        return out;
    }
    let af_init = acceleration.min(maximum); // TA-Lib clamps the step to the cap
    let mut st = sar_seed(high, low, af_init);
    for today in 1..n {
        out[today] = sar_step(&mut st, high[today], low[today], af_init, maximum);
    }
    out
}

// SAR state-carry: the whole history compresses into the recurrence's loop state as of the
// last valid bar — `[is_long, af, ep, sar, prev_high, prev_low]`, where `prev_high`/`prev_low`
// are bar `from-1`'s high/low (the `new_high`/`new_low` that become `prev_*` at the next bar).
// The step reads only `high/low[from..]` plus this state, so a resume never indexes before
// `from` — sound after a head-dropping slice. A resume at `from < 2` (the SAR bootstrap reads
// bars 0 and 1) returns `None` and falls back; `sar_final_state` likewise returns `None` when
// `n < 2` (the column is all-NaN, no state to carry).

/// Final SAR state `[is_long, af, ep, sar, prev_high, prev_low]` after a full [`sar`] compute,
/// or `None` when `n < 2` (SAR never produces a value). Replays [`sar`]'s exact recurrence and
/// captures the loop variables as of the last bar (`n-1`) — i.e. the entering state for bar `n`.
pub fn sar_final_state(
    high: &[f64],
    low: &[f64],
    acceleration: f64,
    maximum: f64,
) -> Option<Vec<f64>> {
    let n = high.len();
    if n < 2 {
        return None;
    }
    let af_init = acceleration.min(maximum);
    let mut st = sar_seed(high, low, af_init);
    for today in 1..n {
        sar_step(&mut st, high[today], low[today], af_init, maximum);
    }
    Some(st.to_vec())
}

/// Resume [`sar`] from `state = [is_long, af, ep, sar, prev_high, prev_low]` (as of row
/// `from - 1`) over rows `[from, n)`, bit-identical to a full recompute. `None` at `from < 2`
/// (the bootstrap needs bars 0 and 1, never re-run here). Reads only `high/low[from..]`; the
/// prior bar's extremes come from the carried state.
/// Appends rows `[from, n)` to `out` and returns the carried state.
pub fn sar_resume(
    high: &[f64],
    low: &[f64],
    acceleration: f64,
    maximum: f64,
    from: usize,
    state: &[f64],
    out: &mut Vec<f64>,
) -> Option<Vec<f64>> {
    if from < 2 {
        return None;
    }
    let n = high.len();
    let af_init = acceleration.min(maximum);
    let mut st = [state[0], state[1], state[2], state[3], state[4], state[5]];
    out.reserve(n.saturating_sub(from));
    for today in from..n {
        out.push(sar_step(&mut st, high[today], low[today], af_init, maximum));
    }
    Some(st.to_vec())
}

/// SAR's bootstrap direction from the one-period −DM at bar 1: a positive −DM ⇒ short.
#[inline]
fn sar_starts_long(high: &[f64], low: &[f64]) -> bool {
    let diff_p = high[1] - high[0];
    let diff_m = low[0] - low[1];
    let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m { diff_m } else { 0.0 };
    !(minus_dm1 > 0.0)
}

/// SAR's state entering bar 1 (`n >= 2`): the bootstrap direction, its stop and extreme
/// point, and bar 1 "cheated" in as the prior bar (as TA-Lib does).
fn sar_seed(high: &[f64], low: &[f64], af_init: f64) -> [f64; 6] {
    let is_long = sar_starts_long(high, low);
    let (ep, sar) = if is_long { (high[1], low[0]) } else { (low[1], high[0]) };
    [is_long as u8 as f64, af_init, ep, sar, high[1], low[1]]
}

/// Limit a stop to the prior and current bar's range, in TA-Lib's comparison order.
/// A computed NaN stop stays NaN; `f64::min` / `max` would replace it with a price.
/// Shared by SAR and SAREXT, for reversals and ordinary steps alike.
#[inline]
fn sar_limit(mut stop: f64, previous: f64, current: f64, long: bool) -> f64 {
    if if long { stop > previous } else { stop < previous } {
        stop = previous;
    }
    if if long { stop > current } else { stop < current } {
        stop = current;
    }
    stop
}

/// One SAR bar — the full computation, [`sar_final_state`], [`sar_resume`] and
/// [`sar_resume_one`] all advance by it, so they cannot drift:
/// advances `st = [is_long, af, ep, sar, prev_high, prev_low]` over the bar
/// `(new_high, new_low)` and returns the bar's stop.
#[inline]
fn sar_step(st: &mut [f64; 6], new_high: f64, new_low: f64, af_init: f64, maximum: f64) -> f64 {
    let [l, mut af, mut ep, mut sar, prev_high, prev_low] = *st;
    let mut is_long = l != 0.0;
    let out;
    if is_long {
        if new_low <= sar {
            is_long = false;
            sar = sar_limit(ep, prev_high, new_high, false);
            out = sar;
            af = af_init;
            ep = new_low;
            sar = sar_limit(sar + af * (ep - sar), prev_high, new_high, false);
        } else {
            out = sar;
            if new_high > ep {
                ep = new_high;
                af = (af + af_init).min(maximum);
            }
            sar = sar_limit(sar + af * (ep - sar), prev_low, new_low, true);
        }
    } else if new_high >= sar {
        is_long = true;
        sar = sar_limit(ep, prev_low, new_low, true);
        out = sar;
        af = af_init;
        ep = new_high;
        sar = sar_limit(sar + af * (ep - sar), prev_low, new_low, true);
    } else {
        out = sar;
        if new_low < ep {
            ep = new_low;
            af = (af + af_init).min(maximum);
        }
        sar = sar_limit(sar + af * (ep - sar), prev_high, new_high, false);
    }
    *st = [is_long as u8 as f64, af, ep, sar, new_high, new_low];
    out
}

/// Parabolic SAR Extended (TA-Lib SAREXT). As [`sar`], but with separate long/short
/// acceleration (init/step/max), an optional start value (`>0` forces an initial long at
/// that level, `<0` an initial short at `|start|`, `0` = SAR's directional bootstrap), an
/// `offset_on_reverse` that nudges the stop on each reversal, and a **signed** output —
/// negative while short, positive while long — so reversals are visible. Lookback 1.
#[allow(clippy::too_many_arguments)]
pub fn sarext(
    high: &[f64],
    low: &[f64],
    start_value: f64,
    offset_on_reverse: f64,
    accel_init_long: f64,
    accel_long: f64,
    accel_max_long: f64,
    accel_init_short: f64,
    accel_short: f64,
    accel_max_short: f64,
) -> Vec<f64> {
    let n = high.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 {
        return out;
    }
    // TA-Lib clamps the init/step factors to their caps.
    let p = SarextParams::new(
        offset_on_reverse, accel_init_long, accel_long, accel_max_long, accel_init_short,
        accel_short, accel_max_short,
    );
    let mut st = sarext_seed(high, low, start_value, &p);
    for today in 1..n {
        out[today] = sarext_step(&mut st, high[today], low[today], &p);
    }
    out
}

// SAREXT state-carry: like SAR, but carry both per-direction acceleration factors —
// `[is_long, af_long, af_short, ep, sar, prev_high, prev_low]`. The inactive direction's `af`
// is preserved across the active run (only ramped while in that direction, reset to its init on
// re-entry), so both must be carried for a bit-exact resume. `start_value` only steers the bar-1
// bootstrap, so it is irrelevant to a resume (`from >= 2`) and not threaded through `*_resume`;
// `offset_on_reverse` and the long/short accel step/cap still are. A resume at `from < 2` falls
// back (`None`), as does `*_final_state` when `n < 2`.

/// Final SAREXT state `[is_long, af_long, af_short, ep, sar, prev_high, prev_low]` after a full
/// [`sarext`] compute, or `None` when `n < 2`. Replays [`sarext`]'s exact recurrence and captures
/// the loop variables as of the last bar (`n-1`) — the entering state for bar `n`.
#[allow(clippy::too_many_arguments)]
pub fn sarext_final_state(
    high: &[f64],
    low: &[f64],
    start_value: f64,
    offset_on_reverse: f64,
    accel_init_long: f64,
    accel_long: f64,
    accel_max_long: f64,
    accel_init_short: f64,
    accel_short: f64,
    accel_max_short: f64,
) -> Option<Vec<f64>> {
    let n = high.len();
    if n < 2 {
        return None;
    }
    let p = SarextParams::new(
        offset_on_reverse, accel_init_long, accel_long, accel_max_long, accel_init_short,
        accel_short, accel_max_short,
    );
    let mut st = sarext_seed(high, low, start_value, &p);
    for today in 1..n {
        sarext_step(&mut st, high[today], low[today], &p);
    }
    Some(st.to_vec())
}

/// Resume [`sarext`] from `state = [is_long, af_long, af_short, ep, sar, prev_high, prev_low]`
/// (as of row `from - 1`) over rows `[from, n)`, bit-identical to a full recompute. `None` at
/// `from < 2`. `start_value` is omitted (it only steers the bar-1 bootstrap, never re-run here).
/// Reads only `high/low[from..]`; the prior bar's extremes come from the carried state.
#[allow(clippy::too_many_arguments)]
/// Appends rows `[from, n)` to `out` and returns the carried state.
pub fn sarext_resume(
    high: &[f64],
    low: &[f64],
    offset_on_reverse: f64,
    accel_init_long: f64,
    accel_long: f64,
    accel_max_long: f64,
    accel_init_short: f64,
    accel_short: f64,
    accel_max_short: f64,
    from: usize,
    state: &[f64],
    out: &mut Vec<f64>,
) -> Option<Vec<f64>> {
    if from < 2 {
        return None;
    }
    let n = high.len();
    let p = SarextParams::new(
        offset_on_reverse, accel_init_long, accel_long, accel_max_long, accel_init_short,
        accel_short, accel_max_short,
    );
    let mut st = [state[0], state[1], state[2], state[3], state[4], state[5], state[6]];
    out.reserve(n.saturating_sub(from));
    for today in from..n {
        out.push(sarext_step(&mut st, high[today], low[today], &p));
    }
    Some(st.to_vec())
}

/// SAREXT's resume parameters, with each acceleration capped by its maximum.
struct SarextParams {
    offset_on_reverse: f64,
    af_long_init: f64,
    accel_long: f64,
    accel_max_long: f64,
    af_short_init: f64,
    accel_short: f64,
    accel_max_short: f64,
}

impl SarextParams {
    #[allow(clippy::too_many_arguments)]
    fn new(
        offset_on_reverse: f64,
        accel_init_long: f64,
        accel_long: f64,
        accel_max_long: f64,
        accel_init_short: f64,
        accel_short: f64,
        accel_max_short: f64,
    ) -> Self {
        SarextParams {
            offset_on_reverse,
            af_long_init: accel_init_long.min(accel_max_long),
            accel_long: accel_long.min(accel_max_long),
            accel_max_long,
            af_short_init: accel_init_short.min(accel_max_short),
            accel_short: accel_short.min(accel_max_short),
            accel_max_short,
        }
    }
}

/// SAREXT's state entering bar 1 (`n >= 2`): a non-zero `start_value` forces the initial
/// direction (and, its magnitude, the stop), else SAR's bootstrap; bar 1 is "cheated" in
/// as the prior bar.
fn sarext_seed(high: &[f64], low: &[f64], start_value: f64, p: &SarextParams) -> [f64; 7] {
    let is_long = if start_value == 0.0 { sar_starts_long(high, low) } else { start_value > 0.0 };
    let (ep, sar) = if start_value == 0.0 {
        if is_long { (high[1], low[0]) } else { (low[1], high[0]) }
    } else if start_value > 0.0 {
        (high[1], start_value)
    } else {
        (low[1], start_value.abs())
    };
    [is_long as u8 as f64, p.af_long_init, p.af_short_init, ep, sar, high[1], low[1]]
}

/// One SAREXT bar — the full computation, [`sarext_final_state`], [`sarext_resume`] and
/// [`sarext_resume_one`] all advance by it, so they cannot drift: advances `st = [is_long, af_long, af_short, ep, sar, prev_high, prev_low]` over
/// the bar `(new_high, new_low)` and returns the bar's signed stop.
#[inline]
fn sarext_step(st: &mut [f64; 7], new_high: f64, new_low: f64, p: &SarextParams) -> f64 {
    let [l, mut af_long, mut af_short, mut ep, mut sar, prev_high, prev_low] = *st;
    let mut is_long = l != 0.0;
    let out;
    if is_long {
        if new_low <= sar {
            is_long = false;
            sar = sar_limit(ep, prev_high, new_high, false);
            if p.offset_on_reverse != 0.0 {
                sar += sar * p.offset_on_reverse;
            }
            out = -sar;
            af_short = p.af_short_init;
            ep = new_low;
            sar = sar_limit(sar + af_short * (ep - sar), prev_high, new_high, false);
        } else {
            out = sar;
            if new_high > ep {
                ep = new_high;
                af_long = (af_long + p.accel_long).min(p.accel_max_long);
            }
            sar = sar_limit(sar + af_long * (ep - sar), prev_low, new_low, true);
        }
    } else if new_high >= sar {
        is_long = true;
        sar = sar_limit(ep, prev_low, new_low, true);
        if p.offset_on_reverse != 0.0 {
            sar -= sar * p.offset_on_reverse;
        }
        out = sar;
        af_long = p.af_long_init;
        ep = new_high;
        sar = sar_limit(sar + af_long * (ep - sar), prev_low, new_low, true);
    } else {
        out = -sar;
        if new_low < ep {
            ep = new_low;
            af_short = (af_short + p.accel_short).min(p.accel_max_short);
        }
        sar = sar_limit(sar + af_short * (ep - sar), prev_high, new_high, false);
    }
    *st = [is_long as u8 as f64, af_long, af_short, ep, sar, new_high, new_low];
    out
}


/// Scalar single-row twin of [`sar_resume`]: the SAR value at `row` from `state`
/// `[is_long, af, ep, sar, prev_high, prev_low]` (as of `row - 1`), the state as of `row`
/// written into `next`, no allocation — the same [`sar_step`] as the Vec kernel. Mirrors
/// the `from < 2` bootstrap guard.
pub fn sar_resume_one(
    high: &[f64],
    low: &[f64],
    acceleration: f64,
    maximum: f64,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row < 2 || row >= high.len() || row >= low.len() {
        return None;
    }
    let (Ok(st), Ok(out)) = (<[f64; 6]>::try_from(state), <&mut [f64; 6]>::try_from(next)) else {
        return None;
    };
    *out = st;
    Some(sar_step(out, high[row], low[row], acceleration.min(maximum), maximum))
}

/// Scalar single-row twin of [`sarext_resume`]: the signed SAREXT value at `row` from
/// `state` `[is_long, af_long, af_short, ep, sar, prev_high, prev_low]` (as of `row - 1`),
/// the state as of `row` written into `next`, no allocation — the same [`sarext_step`] as
/// the Vec kernel. Mirrors the `from < 2` bootstrap guard.
#[allow(clippy::too_many_arguments)]
pub fn sarext_resume_one(
    high: &[f64],
    low: &[f64],
    offset_on_reverse: f64,
    accel_init_long: f64,
    accel_long: f64,
    accel_max_long: f64,
    accel_init_short: f64,
    accel_short: f64,
    accel_max_short: f64,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row < 2 || row >= high.len() || row >= low.len() {
        return None;
    }
    let (Ok(st), Ok(out)) = (<[f64; 7]>::try_from(state), <&mut [f64; 7]>::try_from(next)) else {
        return None;
    };
    let p = SarextParams::new(
        offset_on_reverse, accel_init_long, accel_long, accel_max_long, accel_init_short,
        accel_short, accel_max_short,
    );
    *out = st;
    Some(sarext_step(out, high[row], low[row], &p))
}

