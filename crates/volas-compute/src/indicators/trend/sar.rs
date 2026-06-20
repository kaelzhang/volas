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
                                             // Initial direction from the one-period −DM at bar 1: a positive −DM ⇒ short.
    let diff_p = high[1] - high[0];
    let diff_m = low[0] - low[1];
    let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m {
        diff_m
    } else {
        0.0
    };
    let mut is_long = !(minus_dm1 > 0.0);

    let mut af = af_init;
    let (mut ep, mut sar) = if is_long {
        (high[1], low[0])
    } else {
        (low[1], high[0])
    };
    // "Cheat" the first iteration: prime new high/low with bar 1 (as TA-Lib does).
    let mut new_low = low[1];
    let mut new_high = high[1];

    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                // Reverse to short: stop becomes the extreme point, clamped up.
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                out[today] = sar;
                af = af_init;
                ep = new_low;
                sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out[today] = sar;
                if new_high > ep {
                    ep = new_high;
                    af = (af + af_init).min(maximum);
                }
                sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            // Reverse to long: stop becomes the extreme point, clamped down.
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            out[today] = sar;
            af = af_init;
            ep = new_high;
            sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out[today] = sar;
            if new_low < ep {
                ep = new_low;
                af = (af + af_init).min(maximum);
            }
            sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
        }
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
    let diff_p = high[1] - high[0];
    let diff_m = low[0] - low[1];
    let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m {
        diff_m
    } else {
        0.0
    };
    let mut is_long = !(minus_dm1 > 0.0);
    let mut af = af_init;
    let (mut ep, mut sar) = if is_long {
        (high[1], low[0])
    } else {
        (low[1], high[0])
    };
    let mut new_low = low[1];
    let mut new_high = high[1];
    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                af = af_init;
                ep = new_low;
                sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
            } else {
                if new_high > ep {
                    ep = new_high;
                    af = (af + af_init).min(maximum);
                }
                sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            af = af_init;
            ep = new_high;
            sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
        } else {
            if new_low < ep {
                ep = new_low;
                af = (af + af_init).min(maximum);
            }
            sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    Some(vec![is_long as u8 as f64, af, ep, sar, new_high, new_low])
}

/// Resume [`sar`] from `state = [is_long, af, ep, sar, prev_high, prev_low]` (as of row
/// `from - 1`) over rows `[from, n)`, bit-identical to a full recompute. `None` at `from < 2`
/// (the bootstrap needs bars 0 and 1, never re-run here). Reads only `high/low[from..]`; the
/// prior bar's extremes come from the carried state.
pub fn sar_resume(
    high: &[f64],
    low: &[f64],
    acceleration: f64,
    maximum: f64,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from < 2 {
        return None;
    }
    let n = high.len();
    let af_init = acceleration.min(maximum);
    let mut is_long = state[0] != 0.0;
    let mut af = state[1];
    let mut ep = state[2];
    let mut sar = state[3];
    let mut new_high = state[4];
    let mut new_low = state[5];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                out.push(sar);
                af = af_init;
                ep = new_low;
                sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out.push(sar);
                if new_high > ep {
                    ep = new_high;
                    af = (af + af_init).min(maximum);
                }
                sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            out.push(sar);
            af = af_init;
            ep = new_high;
            sar = (sar + af * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out.push(sar);
            if new_low < ep {
                ep = new_low;
                af = (af + af_init).min(maximum);
            }
            sar = (sar + af * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    Some((
        out,
        vec![is_long as u8 as f64, af, ep, sar, new_high, new_low],
    ))
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
    let af_long_init = accel_init_long.min(accel_max_long);
    let af_short_init = accel_init_short.min(accel_max_short);
    let accel_long = accel_long.min(accel_max_long);
    let accel_short = accel_short.min(accel_max_short);

    // Initial direction: forced by a non-zero start value, else SAR's -DM1 bootstrap.
    let mut is_long = if start_value == 0.0 {
        let diff_p = high[1] - high[0];
        let diff_m = low[0] - low[1];
        let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m {
            diff_m
        } else {
            0.0
        };
        !(minus_dm1 > 0.0)
    } else {
        start_value > 0.0
    };

    let (mut ep, mut sar) = if start_value == 0.0 {
        if is_long {
            (high[1], low[0])
        } else {
            (low[1], high[0])
        }
    } else if start_value > 0.0 {
        (high[1], start_value)
    } else {
        (low[1], start_value.abs())
    };

    let (mut af_long, mut af_short) = (af_long_init, af_short_init);
    let mut new_low = low[1];
    let mut new_high = high[1];

    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                if offset_on_reverse != 0.0 {
                    sar += sar * offset_on_reverse;
                }
                out[today] = -sar;
                af_short = af_short_init;
                ep = new_low;
                sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out[today] = sar;
                if new_high > ep {
                    ep = new_high;
                    af_long = (af_long + accel_long).min(accel_max_long);
                }
                sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            if offset_on_reverse != 0.0 {
                sar -= sar * offset_on_reverse;
            }
            out[today] = sar;
            af_long = af_long_init;
            ep = new_high;
            sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out[today] = -sar;
            if new_low < ep {
                ep = new_low;
                af_short = (af_short + accel_short).min(accel_max_short);
            }
            sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
        }
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
    let af_long_init = accel_init_long.min(accel_max_long);
    let af_short_init = accel_init_short.min(accel_max_short);
    let accel_long = accel_long.min(accel_max_long);
    let accel_short = accel_short.min(accel_max_short);
    let mut is_long = if start_value == 0.0 {
        let diff_p = high[1] - high[0];
        let diff_m = low[0] - low[1];
        let minus_dm1 = if diff_m > 0.0 && diff_p < diff_m {
            diff_m
        } else {
            0.0
        };
        !(minus_dm1 > 0.0)
    } else {
        start_value > 0.0
    };
    let (mut ep, mut sar) = if start_value == 0.0 {
        if is_long {
            (high[1], low[0])
        } else {
            (low[1], high[0])
        }
    } else if start_value > 0.0 {
        (high[1], start_value)
    } else {
        (low[1], start_value.abs())
    };
    let (mut af_long, mut af_short) = (af_long_init, af_short_init);
    let mut new_low = low[1];
    let mut new_high = high[1];
    for today in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                if offset_on_reverse != 0.0 {
                    sar += sar * offset_on_reverse;
                }
                af_short = af_short_init;
                ep = new_low;
                sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
            } else {
                if new_high > ep {
                    ep = new_high;
                    af_long = (af_long + accel_long).min(accel_max_long);
                }
                sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            if offset_on_reverse != 0.0 {
                sar -= sar * offset_on_reverse;
            }
            af_long = af_long_init;
            ep = new_high;
            sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
        } else {
            if new_low < ep {
                ep = new_low;
                af_short = (af_short + accel_short).min(accel_max_short);
            }
            sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    Some(vec![
        is_long as u8 as f64,
        af_long,
        af_short,
        ep,
        sar,
        new_high,
        new_low,
    ])
}

/// Resume [`sarext`] from `state = [is_long, af_long, af_short, ep, sar, prev_high, prev_low]`
/// (as of row `from - 1`) over rows `[from, n)`, bit-identical to a full recompute. `None` at
/// `from < 2`. `start_value` is omitted (it only steers the bar-1 bootstrap, never re-run here).
/// Reads only `high/low[from..]`; the prior bar's extremes come from the carried state.
#[allow(clippy::too_many_arguments)]
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
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from < 2 {
        return None;
    }
    let n = high.len();
    let af_long_init = accel_init_long.min(accel_max_long);
    let af_short_init = accel_init_short.min(accel_max_short);
    let accel_long = accel_long.min(accel_max_long);
    let accel_short = accel_short.min(accel_max_short);
    let mut is_long = state[0] != 0.0;
    let mut af_long = state[1];
    let mut af_short = state[2];
    let mut ep = state[3];
    let mut sar = state[4];
    let mut new_high = state[5];
    let mut new_low = state[6];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[today];
        new_high = high[today];
        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep.max(prev_high).max(new_high);
                if offset_on_reverse != 0.0 {
                    sar += sar * offset_on_reverse;
                }
                out.push(-sar);
                af_short = af_short_init;
                ep = new_low;
                sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
            } else {
                out.push(sar);
                if new_high > ep {
                    ep = new_high;
                    af_long = (af_long + accel_long).min(accel_max_long);
                }
                sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
            }
        } else if new_high >= sar {
            is_long = true;
            sar = ep.min(prev_low).min(new_low);
            if offset_on_reverse != 0.0 {
                sar -= sar * offset_on_reverse;
            }
            out.push(sar);
            af_long = af_long_init;
            ep = new_high;
            sar = (sar + af_long * (ep - sar)).min(prev_low).min(new_low);
        } else {
            out.push(-sar);
            if new_low < ep {
                ep = new_low;
                af_short = (af_short + accel_short).min(accel_max_short);
            }
            sar = (sar + af_short * (ep - sar)).max(prev_high).max(new_high);
        }
    }
    Some((
        out,
        vec![
            is_long as u8 as f64,
            af_long,
            af_short,
            ep,
            sar,
            new_high,
            new_low,
        ],
    ))
}


/// Scalar single-row twin of [`sar_resume`]: the SAR value at `row` from `state`
/// `[is_long, af, ep, sar, prev_high, prev_low]` (as of `row - 1`), no allocation,
/// bit-identical to the Vec kernel's loop body at `today == row` up to its `out.push`.
/// The `af`/`ep` ramp updates that follow the push only mutate carried state (discarded
/// here), so the emitted value depends only on `is_long`, `ep`, `sar`, and the prior/current
/// extremes. Mirrors the `from < 2` bootstrap guard.
pub fn sar_resume_one(
    high: &[f64],
    low: &[f64],
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row < 2 || state.len() < 6 || row >= high.len() || row >= low.len() {
        return None;
    }
    let is_long = state[0] != 0.0;
    let ep = state[2];
    let sar = state[3];
    let prev_high = state[4];
    let prev_low = state[5];
    let new_low = low[row];
    let new_high = high[row];
    let val = if is_long {
        if new_low <= sar {
            // Reverse to short: stop becomes the extreme point, clamped up.
            ep.max(prev_high).max(new_high)
        } else {
            sar
        }
    } else if new_high >= sar {
        // Reverse to long: stop becomes the extreme point, clamped down.
        ep.min(prev_low).min(new_low)
    } else {
        sar
    };
    Some(val)
}

/// Scalar single-row twin of [`sarext_resume`]: the signed SAREXT value at `row` from `state`
/// `[is_long, af_long, af_short, ep, sar, prev_high, prev_low]` (as of `row - 1`), no allocation,
/// bit-identical to the Vec kernel's loop body at `today == row` up to its `out.push`. Negative
/// while short, positive while long. Only `offset_on_reverse` is threaded (it nudges the stop on a
/// reversal, BEFORE the push); the long/short accel step/cap params feed only the post-push af
/// ramp (discarded state) and are omitted. Mirrors the `from < 2` bootstrap guard.
pub fn sarext_resume_one(
    high: &[f64],
    low: &[f64],
    offset_on_reverse: f64,
    row: usize,
    state: &[f64],
) -> Option<f64> {
    if row < 2 || state.len() < 7 || row >= high.len() || row >= low.len() {
        return None;
    }
    let is_long = state[0] != 0.0;
    let ep = state[3];
    let sar = state[4];
    let prev_high = state[5];
    let prev_low = state[6];
    let new_low = low[row];
    let new_high = high[row];
    let val = if is_long {
        if new_low <= sar {
            // Reverse to short: extreme point clamped up, offset-nudged, output negated.
            let mut sar = ep.max(prev_high).max(new_high);
            if offset_on_reverse != 0.0 {
                sar += sar * offset_on_reverse;
            }
            -sar
        } else {
            sar
        }
    } else if new_high >= sar {
        // Reverse to long: extreme point clamped down, offset-nudged.
        let mut sar = ep.min(prev_low).min(new_low);
        if offset_on_reverse != 0.0 {
            sar -= sar * offset_on_reverse;
        }
        sar
    } else {
        -sar
    };
    Some(val)
}

