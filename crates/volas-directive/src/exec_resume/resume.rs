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
    let mut out = Vec::new();
    let st = resume_into(df, node, prev_state, from_row, end, since, &mut out)?;
    Some((Column::f64(out), st))
}

/// Appends rows `[from_row, end)` to `out` and returns the carried state.
/// Chained nodes may produce a boolean column and use `execute_resume` instead.
pub(crate) fn resume_into(
    df: &DataFrame,
    node: &Ast,
    prev_state: &[f64],
    from_row: usize,
    end: usize,
    since: usize,
    out: &mut Vec<f64>,
) -> Option<Vec<f64>> {
    if super::chain::chained(node) {
        return None;
    }
    if !matches!(node.continuation(), crate::types::Continuation::Own) {
        return None;
    }
    let cmd = as_command(node.node())?;
    let resume = resume_of(&cmd);
    let end = end.min(df.height());
    if resume == Resume::Recompute || from_row > end {
        return None;
    }
    let bc = bound(&cmd);
    let start = window_start(node, &cmd, from_row);
    if recursive(&cmd) {
        // Over the complete rows: a missing new row is NaN, the state untouched.
        return crate::complete::resume(df, &bc, start, prev_state, (from_row, end, since), out, |ops, from, out| {
            resume_rows(ops, &bc, prev_state, from, df.origin() + ops.start(), out)
        });
    }
    let ops = Operands::rows(df, &cmd.series, start, end);
    if resume == Resume::Step {
        return step_rows(&ops, &bc, prev_state, from_row - start, out);
    }
    resume_rows(&ops, &bc, prev_state, from_row - start, df.origin() + start, out)
}

/// Appends a finite-window command's rows `[from, len)` of `ops` to `out`, one step at a time.
fn step_rows(ops: &Operands, cmd: &BoundCommand, prev_state: &[f64], from: usize, out: &mut Vec<f64>) -> Option<Vec<f64>> {
    let (mut prev, mut next) = (prev_state.to_vec(), prev_state.to_vec());
    let at = out.len();
    out.reserve(ops.len().saturating_sub(from));
    for row in from..ops.len() {
        let Some(value) = super::window::step(ops, cmd.name, cmd.sub, cmd.args, &prev, row, &mut next) else {
            out.truncate(at);
            return None;
        };
        out.push(value);
        std::mem::swap(&mut prev, &mut next);
    }
    Some(prev)
}

/// Appends the bound command's rows `[from_row, ops.len())` to `out` and returns its state.
fn resume_rows(
    ops: &Operands,
    cmd: &BoundCommand,
    prev_state: &[f64],
    from_row: usize,
    origin: usize,
    out: &mut Vec<f64>,
) -> Option<Vec<f64>> {
    let (name, sub, args) = (cmd.name, cmd.sub, cmd.args);
    let close = || ops.f64(0, "close");
    match (name, sub) {
        ("obv", _) => {
            let real = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let st = ind::obv_resume(&real, &volume, from_row, prev_state, out);
            Some(st)
        }
        ("ad", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let volume = ops.f64(3, "volume").ok()?;
            let st = ind::ad_resume(&high, &low, &close, &volume, from_row, prev_state, out);
            Some(st)
        }
        ("adosc", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let volume = ops.f64(3, "volume").ok()?;
            let fast = arg_usize(args, 0);
            let slow = arg_usize(args, 1);
            let st = ind::adosc_resume(
                &high, &low, &close, &volume, fast, slow, from_row, prev_state,
                out,
            );
            Some(st)
        }

        // Group A cumulative / EMA-recursion family (volas-exclusive).
        ("pvt", _) => {
            let close = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let st = ind::pvt_resume(&close, &volume, from_row, prev_state, out);
            Some(st)
        }
        ("nvi", _) => {
            let close = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let st = ind::nvi_resume(&close, &volume, from_row, prev_state, out);
            Some(st)
        }
        ("pvi", _) => {
            let close = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let st = ind::pvi_resume(&close, &volume, from_row, prev_state, out);
            Some(st)
        }
        ("efi", _) => {
            let close = ops.f64(0, "close").ok()?;
            let volume = ops.f64(1, "volume").ok()?;
            let st =
                ind::efi_resume(&close, &volume, arg_usize(args, 0), from_row, prev_state, out);
            Some(st)
        }
        ("tsi", _) => {
            let st = ind::tsi_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_usize(args, 1),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("mass_index", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let st =
                ind::mass_index_resume(&high, &low, arg_usize(args, 0), from_row, prev_state, out)?;
            Some(st)
        }

        // kcw = 2·mult·ATR/EMA over the Keltner band's EMA + ATR pair.
        ("kcw", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            // the ATR first, into its own rows: a decline leaves `out` untouched
            let mut atr = Vec::new();
            let atr_state =
                ind::atr_resume(&high, &low, &close, arg_usize(args, 1), from_row, &prev_state[1..2], &mut atr)?;
            let at = out.len();
            let ema_state = ind::ema_resume(&close, arg_usize(args, 0), from_row, &prev_state[0..1], out);
            let m = arg_f64(args, 2);
            for (e, a) in out[at..].iter_mut().zip(atr) {
                *e = ind::kcw_value(*e, a, m);
            }
            Some(vec![ema_state[0], atr_state[0]])
        }
        // Keltner Channels (Group B): middle resumes the EMA; bands resume the EMA + ATR pair.
        ("keltner", None) => {
            let st = ind::ema_resume(
                &ops.f64(0, "close").ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("keltner", Some(sub)) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::keltner_band_resume(
                &close,
                &high,
                &low,
                arg_usize(args, 0),
                arg_usize(args, 1),
                arg_f64(args, 2),
                sub == "upper",
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }

        // Group B cumulative family (volas-exclusive): resume the running line.
        ("wad", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::wad_resume(&high, &low, &close, from_row, prev_state, out);
            Some(st)
        }
        ("asi", _) => {
            let open = ops.f64(0, "open").ok()?;
            let high = ops.f64(1, "high").ok()?;
            let low = ops.f64(2, "low").ok()?;
            let close = ops.f64(3, "close").ok()?;
            let st = ind::asi_resume(
                &open,
                &high,
                &low,
                &close,
                arg_f64(args, 0),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("supertrend", sub) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::supertrend_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                arg_f64(args, 1),
                sub == Some("direction"),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }

        // SAR family — resume the state machine from the carried tuple. A resume at
        // `from_row < 2` (the SAR bootstrap needs bars 0 and 1) returns `None` and falls back.
        ("sar", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let st = ind::sar_resume(
                &high,
                &low,
                arg_f64(args, 0),
                arg_f64(args, 1),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("sarext", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            // `start_value` (arg 0) only steers the bar-1 bootstrap, never re-run on resume.
            let st = ind::sarext_resume(
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
                out,
            )?;
            Some(st)
        }

        // EMA-recursion family — resume each carried sub-EMA from its last value (skipping
        // the SMA seed), bit-identical to the full kernel's steady-state recurrence.
        ("ema", _) => {
            let st = ind::ema_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("smma", _) => {
            let st = ind::smma_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("dema", _) => {
            let st = ind::dema_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("tema", _) => {
            let st = ind::tema_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("t3", _) => {
            let st = ind::t3_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_f64(args, 1),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("trix", _) => {
            let st = ind::trix_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        // KAMA's sliding-sum resume can decline (short retained head) → None falls back.
        ("kama", _) => {
            let st = ind::kama_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }

        ("macd", None) => {
            let st = ind::macd_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_usize(args, 1),
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("macd", Some(line @ ("signal" | "histogram"))) => {
            let st = ind::macd_signal_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_usize(args, 1),
                arg_usize(args, 2),
                line == "histogram",
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }
        ("macdfix", None) => {
            let st = ind::macd_resume(&close().ok()?, 12, 26, from_row, prev_state, out);
            Some(st)
        }
        ("macdfix", Some(line @ ("signal" | "histogram"))) => {
            let st = ind::macd_signal_resume(
                &close().ok()?,
                12,
                26,
                arg_usize(args, 0),
                line == "histogram",
                from_row,
                prev_state,
                out,
            );
            Some(st)
        }

        // Wilder-smoothing family — resume the running average(s) over the new rows.
        // Each reads only `…[from_row-1..]` (the per-bar term needs the prior bar), so a
        // resume at `from_row == 0` returns `None` (falls back). DM/DI/DX/ADX/ADXR pull
        // high/low(/close); RSI/CMO/ATR/NATR their named series.
        ("rsi", _) => {
            let st = ind::rsi_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("cmo", _) => {
            let st = ind::cmo_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("atr", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::atr_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("natr", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::natr_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("plus_dm", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let st = ind::plus_dm_resume(
                &high,
                &low,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("minus_dm", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let st = ind::minus_dm_resume(
                &high,
                &low,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("plus_di", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::plus_di_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("minus_di", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::minus_di_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("dx", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::dx_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("adx", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::adx_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("adxr", _) => {
            let high = ops.f64(0, "high").ok()?;
            let low = ops.f64(1, "low").ok()?;
            let close = ops.f64(2, "close").ok()?;
            let st = ind::adxr_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }

        // Hilbert-transform family — reconstruct the shared core + per-output tail and
        // step the recurrence over the new rows. A resume at/under the core warm-up
        // (or, for the price-windowed trendline/trendmode, before a full dominant-cycle
        // window is visible) returns `None` and falls back to the full recompute.
        ("ht_dcperiod", _) => {
            let st = ind::ht_dcperiod_resume(&close().ok()?, from_row, prev_state, out)?;
            Some(st)
        }
        ("ht_phasor", sub) => {
            let st = ind::ht_phasor_resume(
                &close().ok()?,
                sub == Some("quadrature"),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("ht_dcphase", _) => {
            let st = ind::ht_dcphase_resume(&close().ok()?, from_row, prev_state, out)?;
            Some(st)
        }
        ("ht_sine", sub) => {
            let st = ind::ht_sine_resume(
                &close().ok()?,
                sub == Some("leadsine"),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("ht_trendline", _) => {
            let st = ind::ht_trendline_resume(&close().ok()?, from_row, prev_state, out)?;
            Some(st)
        }
        ("ht_trendmode", _) => {
            let st = ind::ht_trendmode_resume(&close().ok()?, from_row, prev_state, out)?;
            Some(st)
        }
        ("mama", sub) => {
            let st = ind::mama_resume(
                &close().ok()?,
                arg_f64(args, 0),
                arg_f64(args, 1),
                sub == Some("fama"),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
        }

        // Index family — windowed arg-extreme emitting ABSOLUTE positions. The carried
        // state is the incremental tracker's running extreme `[idx_abs, value]`; `origin`
        // rebases sub-frame positions back to original-absolute. minmaxindex.max / .min
        // are exactly maxindex / minindex (see `execute`).
        ("maxindex", _) | ("minmaxindex", Some("max")) => {
            let st = ind::maxindex_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                origin,
                prev_state,
                out,
            )?;
            Some(st)
        }
        ("minindex", _) | ("minmaxindex", Some("min")) => {
            let st = ind::minindex_resume(
                &close().ok()?,
                arg_usize(args, 0),
                from_row,
                origin,
                prev_state,
                out,
            )?;
            Some(st)
        }

        // StochRSI — a windowed %K (and SMA `.d`) of the Wilder-recursive RSI; resume by
        // carrying the RSI Wilder pair + the recent RSI values feeding the windows. Only
        // the canonical SMA `.d` (matype 0) resumes (the spec's fact).
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let is_d = line == "d";
            let st = ind::stochrsi_resume(
                &close().ok()?,
                arg_usize(args, 0),
                arg_usize(args, 1),
                is_d,
                arg_usize(args, 2),
                from_row,
                prev_state,
                out,
            )?;
            Some(st)
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
            let st = ind::kdj_resume(
                &high, &low, &close, period_rsv, period_k, period_d, kline, from_row, prev_state,
                out,
            )?;
            Some(st)
        }

        _ => None, // LCOV_EXCL_LINE every line whose spec declares a resume has an arm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(open: Vec<f64>) -> DataFrame {
        DataFrame::new(vec!["open".into()], vec![Column::f64(open)], None).unwrap()
    }

    #[test]
    fn a_sink_keeps_its_prefix_through_retries_scatter_and_empty_tails() {
        let node = crate::parse("rsi:3@open").unwrap();
        let clean: Vec<f64> = (0..50).map(|i| 40.0 + (i as f64 * 0.3).cos()).collect();
        for missing in [vec![], vec![18, 22, 24], (20..50).collect()] {
            let mut open = clean.clone();
            for i in missing {
                open[i] = f64::NAN;
            }
            let df = frame(open);
            let state = super::super::initial_state(&df, &node, 20).unwrap();
            let mut out = vec![-0.0, f64::from_bits(0x7ff8_0000_0000_1234)];
            let prefix: Vec<u64> = out.iter().map(|v| v.to_bits()).collect();
            let next = resume_into(&df, &node, &state, 20, 49, 0, &mut out).unwrap();
            let last = resume_into(&df, &node, &next, 49, 50, 0, &mut out).unwrap();
            let end = resume_into(&df, &node, &last, 50, 50, 0, &mut out).unwrap();
            assert_eq!(out[..2].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), prefix);
            let full = crate::exec::execute(&df, &node).unwrap().to_f64_vec();
            assert_eq!(out.len(), 32);
            assert!(out[2..].iter().zip(&full[20..]).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_eq!(end, last);
        }
    }

    #[test]
    fn a_declined_sink_keeps_its_prefix_after_an_in_place_attempt() {
        let node = crate::parse("rsi:3@open").unwrap();
        let mut open: Vec<f64> = (0..40).map(|i| 40.0 + (i as f64 * 0.3).cos()).collect();
        let state = super::super::initial_state(&frame(open.clone()), &node, 20).unwrap();
        open[20..30].fill(f64::NAN);
        let df = frame(open).slice_data(18, 40);
        let mut out = vec![-0.0];
        assert!(resume_into(&df, &node, &state, 12, df.height(), 0, &mut out).is_none());
        assert_eq!(out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), vec![(-0.0f64).to_bits()]);
        let chained = crate::parse("rsi:3@open > 50").unwrap();
        assert!(resume_into(&df, &chained, &state, 12, df.height(), 0, &mut out).is_none());
        assert_eq!(out.len(), 1);
    }
}
