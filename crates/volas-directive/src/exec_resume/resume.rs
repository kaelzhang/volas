//! Resume directive execution over only the new tail rows, bit-identical to a
//! full recompute (the per-directive `execute_resume*` dispatch).

use crate::complete::BoundCommand;
use crate::exec::{arg_f64, arg_usize};
use crate::operands::Operands;
use crate::spec::Resume;
use crate::types::Ast;
use volas_compute::raw::indicators as ind;
use volas_core::{Column, DataFrame};

use super::{as_command, bound, recursive, resume_of, window_start};

/// Resume `node` from `prev_state` over rows `[from_row, end)` (`end` at most the
/// frame's height), returning those rows as a [`Column`] and the state as of the last
/// of them. `None` when `node` has no resume kernel, or the resume declines (the caller
/// falls back to a recompute). The values are bit-identical to a fresh full recompute,
/// so writing them into the stale rows keeps the cached column exact. Stopping before
/// the last row reads the inputs' leading rows in place (no copy of the frame), so a
/// refresh of several rows obtains the state before the last one too.
///
/// It reads a trailing window of the frame (see `window_start`).
///
/// `since` is the absolute position of the row the column's history starts at: when it
/// lies before the frame's row 0 (the frame dropped its head), a resume missing complete
/// rows of the history it reads declines rather than restart the recursion. The frame's
/// own `origin` (`DataFrame::origin`) places its rows: recursive *value* indicators
/// ignore it; the absolute-position index family adds it, so emitted positions are the
/// series' own on every path.
pub fn execute_resume(
    df: &DataFrame,
    node: &Ast,
    prev_state: &[f64],
    from_row: usize,
    end: usize,
    since: usize,
) -> Option<(Column, Vec<f64>)> {
    if super::chain::chained(node) {
        return super::chain::resume(df, node, prev_state, from_row, end, since);
    }
    let cmd = as_command(node)?;
    let resume = resume_of(&cmd);
    let end = end.min(df.height());
    if resume == Resume::Recompute || from_row > end {
        return None;
    }
    let bc = bound(&cmd);
    let start = window_start(node, &cmd, from_row);
    if recursive(&cmd) {
        // Over the complete rows: a missing new row is NaN, the state untouched.
        return crate::complete::resume(df, &bc, start, prev_state, (from_row, end, since), |ops, from| {
            resume_rows(ops, &bc, prev_state, from, df.origin() + ops.start())
        });
    }
    let ops = Operands::rows(df, &cmd.series, start, end);
    if resume == Resume::Step {
        return step_rows(&ops, &bc, prev_state, from_row - start);
    }
    resume_rows(&ops, &bc, prev_state, from_row - start, df.origin() + start)
}

/// A finite-window command's rows `[from, len)` of `ops`, one step at a time.
fn step_rows(ops: &Operands, cmd: &BoundCommand, prev_state: &[f64], from: usize) -> Option<(Column, Vec<f64>)> {
    let (mut prev, mut next) = (prev_state.to_vec(), prev_state.to_vec());
    let mut out = Vec::with_capacity(ops.len().saturating_sub(from));
    for row in from..ops.len() {
        out.push(super::window::step(ops, cmd.name, cmd.sub, cmd.args, &prev, row, &mut next)?);
        std::mem::swap(&mut prev, &mut next);
    }
    Some((Column::f64(out), prev))
}

/// [`execute_resume`] of the bound command `cmd` over every row of `ops`.
fn resume_rows(
    ops: &Operands,
    cmd: &BoundCommand,
    prev_state: &[f64],
    from_row: usize,
    origin: usize,
) -> Option<(Column, Vec<f64>)> {
    let (name, sub, args) = (cmd.name, cmd.sub, cmd.args);
    let close = || ops.f64(0, "close");
    match (name, sub) {
        ("obv", _) => {
            let real = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let (vals, st) = ind::obv_resume(&real, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("ad", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let volume = ops.f64(3, "volume").ok()?;
            let (vals, st) = ind::ad_resume(&high, &low, &close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("adosc", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let volume = ops.f64(3, "volume").ok()?;
            let fast = arg_usize(args, 0);
            let slow = arg_usize(args, 1);
            let (vals, st) = ind::adosc_resume(
                &high, &low, &close, &volume, fast, slow, from_row, prev_state,
            );
            Some((Column::f64(vals), st))
        }

        // Group A cumulative / EMA-recursion family (volas-exclusive).
        ("pvt", _) => {
            let close = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let (vals, st) = ind::pvt_resume(&close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("nvi", _) => {
            let close = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let (vals, st) = ind::nvi_resume(&close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("pvi", _) => {
            let close = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let (vals, st) = ind::pvi_resume(&close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("efi", _) => {
            let close = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let (vals, st) =
                ind::efi_resume(&close, &volume, arg_usize(args, 0), from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("tsi", _) => {
            let (vals, st) = ind::tsi_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_usize(args, 1),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("mass_index", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let (vals, st) =
                ind::mass_index_resume(&high, &low, arg_usize(args, 0), from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }

        // kcw = 2·mult·ATR/EMA over the Keltner band's EMA + ATR pair.
        ("kcw", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (ema, ema_state) = ind::ema_resume(&close, arg_usize(args, 0), from_row, &prev_state[0..1]);
            let (atr, atr_state) =
                ind::atr_resume(&high, &low, &close, arg_usize(args, 1), from_row, &prev_state[1..2])?;
            let m = arg_f64(args, 2);
            let vals = ema.iter().zip(&atr).map(|(&e, &a)| ind::kcw_value(e, a, m)).collect();
            Some((Column::f64(vals), vec![ema_state[0], atr_state[0]]))
        }
        // Keltner Channels (Group B): middle resumes the EMA; bands resume the EMA + ATR pair.
        ("keltner", None) => {
            let (vals, st) = ind::ema_resume(
                &ops.f64(0, "close").ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("keltner", Some(sub)) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::keltner_band_resume(
                &close,
                &high,
                &low,
                arg_usize(args, 0),
                arg_usize(args, 1),
                arg_f64(args, 2),
                sub == "upper",
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // Group B cumulative family (volas-exclusive): resume the running line.
        ("wad", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::wad_resume(&high, &low, &close, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("asi", _) => {
            let open = ops.f64(0, "open").ok()?;
            let high = ops.f64(1, "high").ok()?;
            let low = ops.f64(2, "low").ok()?;
            let close = ops.f64(3, "close").ok()?;
            let (vals, st) = ind::asi_resume(
                &open,
                &high,
                &low,
                &close,
                arg_f64(args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("supertrend", sub) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::supertrend_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                arg_f64(args, 1),
                sub == Some("direction"),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // SAR family — resume the state machine from the carried tuple. A resume at
        // `from_row < 2` (the SAR bootstrap needs bars 0 and 1) returns `None` and falls back.
        ("sar", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let (vals, st) = ind::sar_resume(
                &high,
                &low,
                arg_f64(args, 0),
                arg_f64(args, 1),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("sarext", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            // `start_value` (arg 0) only steers the bar-1 bootstrap, never re-run on resume.
            let (vals, st) = ind::sarext_resume(
                &high,
                &low,
                arg_f64(args, 1),
                arg_f64(args, 2),
                arg_f64(args, 3),
                arg_f64(args, 4),
                arg_f64(args, 5),
                arg_f64(args, 6),
                arg_f64(args, 7),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // EMA-recursion family — resume each carried sub-EMA from its last value (skipping
        // the SMA seed), bit-identical to the full kernel's steady-state recurrence.
        ("ema", _) => {
            let (vals, st) = ind::ema_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("smma", _) => {
            let (vals, st) = ind::smma_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("dema", _) => {
            let (vals, st) = ind::dema_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("tema", _) => {
            let (vals, st) = ind::tema_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("t3", _) => {
            let (vals, st) = ind::t3_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_f64(args, 1),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("trix", _) => {
            let (vals, st) = ind::trix_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        // KAMA's sliding-sum resume can decline (short retained head) → None falls back.
        ("kama", _) => {
            let (vals, st) = ind::kama_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        ("macd", None) => {
            let (vals, st) = ind::macd_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_usize(args, 1),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("macd", Some(line @ ("signal" | "histogram"))) => {
            let (vals, st) = ind::macd_signal_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_usize(args, 1),
                arg_usize(args, 2),
                line == "histogram",
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("macdfix", None) => {
            let (vals, st) = ind::macd_resume(&close().ok()?, 12, 26, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("macdfix", Some(line @ ("signal" | "histogram"))) => {
            let (vals, st) = ind::macd_signal_resume(
                &close().ok()?,
                12,
                26,
                arg_usize(args, 0),
                line == "histogram",
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }

        // Wilder-smoothing family — resume the running average(s) over the new rows.
        // Each reads only `…[from_row-1..]` (the per-bar term needs the prior bar), so a
        // resume at `from_row == 0` returns `None` (falls back). DM/DI/DX/ADX/ADXR pull
        // high/low(/close); RSI/CMO/ATR/NATR their named series.
        ("rsi", _) => {
            let (vals, st) = ind::rsi_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("cmo", _) => {
            let (vals, st) = ind::cmo_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("atr", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::atr_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("natr", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::natr_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("plus_dm", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let (vals, st) = ind::plus_dm_resume(
                &high,
                &low,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minus_dm", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let (vals, st) = ind::minus_dm_resume(
                &high,
                &low,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("plus_di", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::plus_di_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minus_di", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::minus_di_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("dx", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::dx_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("adx", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::adx_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("adxr", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let (vals, st) = ind::adxr_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // Hilbert-transform family — reconstruct the shared core + per-output tail and
        // step the recurrence over the new rows. A resume at/under the core warm-up
        // (or, for the price-windowed trendline/trendmode, before a full dominant-cycle
        // window is visible) returns `None` and falls back to the full recompute.
        ("ht_dcperiod", _) => {
            let (vals, st) = ind::ht_dcperiod_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_phasor", sub) => {
            let (vals, st) = ind::ht_phasor_resume(
                &close().ok()?,
                sub == Some("quadrature"),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("ht_dcphase", _) => {
            let (vals, st) = ind::ht_dcphase_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_sine", sub) => {
            let (vals, st) = ind::ht_sine_resume(
                &close().ok()?,
                sub == Some("leadsine"),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("ht_trendline", _) => {
            let (vals, st) = ind::ht_trendline_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_trendmode", _) => {
            let (vals, st) = ind::ht_trendmode_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("mama", sub) => {
            let (vals, st) = ind::mama_resume(
                &close().ok()?,
                arg_f64(args, 0),
                arg_f64(args, 1),
                sub == Some("fama"),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // Index family — windowed arg-extreme emitting ABSOLUTE positions. The carried
        // state is the incremental tracker's running extreme `[idx_abs, value]`; `origin`
        // rebases sub-frame positions back to original-absolute. minmaxindex.max / .min
        // are exactly maxindex / minindex (see `execute`).
        ("maxindex", _) | ("minmaxindex", Some("max")) => {
            let (vals, st) = ind::maxindex_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                origin,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minindex", _) | ("minmaxindex", Some("min")) => {
            let (vals, st) = ind::minindex_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                origin,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // StochRSI — a windowed %K (and SMA `.d`) of the Wilder-recursive RSI; resume by
        // carrying the RSI Wilder pair + the recent RSI values feeding the windows. Only
        // the canonical SMA `.d` (matype 0) resumes (the spec's fact).
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let is_d = line == "d";
            let (vals, st) = ind::stochrsi_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_usize(args, 1),
                is_d,
                arg_usize(args, 2),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // KDJ — resume the recursive %K (+ %D for `.d`/`.j`) from the carried state; RSV is
        // finite-memory, recomputed over the windowed high/low/close tail.
        ("kdj", Some(line @ ("k" | "d" | "j"))) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let period_rsv = arg_usize(args, 0);
            let period_k = arg_usize(args, 1);
            let kline = match line {
                "k" => ind::KdjLine::K,
                "d" => ind::KdjLine::D,
                _ => ind::KdjLine::J,
            };
            let period_d = if line == "k" {
                3
            } else {
                arg_usize(args, 2)
            };
            let (vals, st) = ind::kdj_resume(
                &high, &low, &close, period_rsv, period_k, period_d, kline, from_row, prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        _ => None, // LCOV_EXCL_LINE every line whose spec declares a resume has an arm
    }
}
