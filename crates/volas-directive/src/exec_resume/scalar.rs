//! The single-row resume step: one new row of a recursive indicator — its value and the
//! carried state as of that row — with no allocation. Mirrors [`super::execute_resume`]
//! arm-for-arm for every line whose spec declares a step ([`Resume::Step`]): each arm
//! calls the scalar twin of that arm's kernel, bit-identical to one iteration of its
//! loop. A cached recursive column refreshes its newest row through
//! this step (a plain append, a replaced last row, a time-frame forming row), writing the
//! advanced state into a buffer it keeps, so the live path allocates nothing.


use crate::exec::{arg_f64, arg_usize};
use crate::operands::Operands;
use crate::spec::Resume;
use crate::types::{ArgValue, Ast};
use volas_compute::raw::indicators as ind;
use volas_core::DataFrame;

use super::{as_command, bound, resume_of, window_start};

/// One row of `node` at `row`, continued from `prev_state` (the state as of the last
/// complete row before `row`): returns the row's value and writes the state as of `row`
/// into `next` (which must be `prev_state`'s length). It reads a trailing window of the
/// frame, so it costs that window, not the frame's height. A row missing an input of a
/// recursive command is NaN and does not advance the state (`next` is `prev_state`).
/// `None` when the line has no single-row step ([`Resume::Step`]), or the step declines
/// (`row` out of range, a state of the wrong shape, or a frame that dropped complete rows
/// of the history the step reads — `since` as in [`super::execute_resume`]) — the caller
/// then takes the `Vec` resume.
pub fn execute_resume_step(
    df: &DataFrame,
    node: &Ast,
    prev_state: &[f64],
    (row, _since): (usize, usize),
    next: &mut [f64],
) -> Option<f64> {
    if super::chain::chained(node) {
        return super::chain::step(df, node, prev_state, row, next);
    }
    if !matches!(node.continuation(), crate::types::Continuation::Own) {
        return None;
    }
    let cmd = as_command(node.node())?;
    if resume_of(&cmd) != Resume::Step || next.len() != prev_state.len() || row >= df.height() {
        return None;
    }
    let start = window_start(node, &cmd, row);
    super::state::step(df, &bound(&cmd), start, prev_state, row, next)
}

/// The step of the bound command `name` / `sub` at `row` over `ops` (see
/// [`execute_resume_step`]).
#[allow(clippy::too_many_arguments)] // the step's own inputs, passed through
pub(super) fn step_of(
    ops: &Operands,
    name: &str,
    sub: Option<&str>,
    args: &[ArgValue],
    prev: &[f64],
    row: usize,
    next: &mut [f64],
) -> Option<f64> {
    let p = |i| arg_usize(args, i);
    let close = || ops.f64(0, "close").ok();
    let hl = || Some((ops.f64(0, "high").ok()?, ops.f64(1, "low").ok()?));
    let hlc = || Some((ops.f64(0, "high").ok()?, ops.f64(1, "low").ok()?, ops.f64(2, "close").ok()?));
    let cv = || Some((ops.f64(0, "close").ok()?, ops.f64(1, "volume").ok()?));
    match (name, sub) {
        ("ema", _) => ind::ema_resume_one(&close()?, p(0), row, prev, next),
        ("smma", _) => ind::smma_resume_one(&close()?, p(0), row, prev, next),
        ("rsi", _) => ind::rsi_resume_one(&close()?, p(0), row, prev, next),
        ("cmo", _) => ind::cmo_resume_one(&close()?, p(0), row, prev, next),
        ("atr" | "natr", _) => {
            let (h, l, c) = hlc()?;
            let atr = ind::atr_resume_one(&h, &l, &c, p(0), row, prev, next)?;
            // NATR scales the ATR it carries by the row's close.
            Some(if name == "natr" { ind::natr_value(atr, c[row]) } else { atr })
        }
        ("obv", _) => {
            let (c, v) = cv()?;
            ind::obv_resume_one(&c, &v, row, prev, next)
        }
        ("ad", _) => {
            let (h, l, c) = hlc()?;
            let v = ops.f64(3, "volume").ok()?;
            ind::ad_resume_one(&h, &l, &c, &v, row, prev, next)
        }
        ("dema", _) => ind::dema_resume_one(&close()?, p(0), row, prev, next),
        ("tema", _) => ind::tema_resume_one(&close()?, p(0), row, prev, next),
        ("t3", _) => ind::t3_resume_one(&close()?, p(0), arg_f64(args, 1), row, prev, next),
        ("kama", _) => ind::kama_resume_one(&close()?, p(0), row, prev, next),
        ("macd", None) => ind::macd_resume_one(&close()?, p(0), p(1), row, prev, next),
        ("macdfix", None) => ind::macd_resume_one(&close()?, 12, 26, row, prev, next),
        ("macd", Some(line)) => {
            let hist = line == "histogram";
            ind::macd_signal_resume_one(&close()?, p(0), p(1), p(2), hist, row, prev, next)
        }
        // macdfix: fixed 12/26, the signal period is argument 0
        ("macdfix", Some(line)) => {
            let hist = line == "histogram";
            ind::macd_signal_resume_one(&close()?, 12, 26, p(0), hist, row, prev, next)
        }
        ("sar", _) => {
            let (h, l) = hl()?;
            ind::sar_resume_one(&h, &l, arg_f64(args, 0), arg_f64(args, 1), row, prev, next)
        }
        ("sarext", _) => {
            let (h, l) = hl()?;
            let a = |i| arg_f64(args, i);
            ind::sarext_resume_one(
                &h, &l, a(1), a(2), a(3), a(4), a(5), a(6), a(7), row, prev, next,
            )
        }
        ("plus_di", _) => {
            let (h, l, c) = hlc()?;
            ind::plus_di_resume_one(&h, &l, &c, p(0), row, prev, next)
        }
        ("minus_di", _) => {
            let (h, l, c) = hlc()?;
            ind::minus_di_resume_one(&h, &l, &c, p(0), row, prev, next)
        }
        ("dx", _) => {
            let (h, l, c) = hlc()?;
            ind::dx_resume_one(&h, &l, &c, p(0), row, prev, next)
        }
        ("adx", _) => {
            let (h, l, c) = hlc()?;
            ind::adx_resume_one(&h, &l, &c, p(0), row, prev, next)
        }
        ("adxr", _) => {
            let (h, l, c) = hlc()?;
            ind::adxr_resume_one(&h, &l, &c, p(0), row, prev, next)
        }
        ("pvt", _) => {
            let (c, v) = cv()?;
            ind::pvt_resume_one(&c, &v, row, prev, next)
        }
        ("nvi", _) => {
            let (c, v) = cv()?;
            ind::nvi_resume_one(&c, &v, row, prev, next)
        }
        ("pvi", _) => {
            let (c, v) = cv()?;
            ind::pvi_resume_one(&c, &v, row, prev, next)
        }
        ("efi", _) => {
            let (c, v) = cv()?;
            ind::efi_resume_one(&c, &v, p(0), row, prev, next)
        }
        ("tsi", _) => ind::tsi_resume_one(&close()?, p(0), p(1), row, prev, next),
        ("mass_index", _) => {
            let (h, l) = hl()?;
            ind::mass_index_resume_one(&h, &l, p(0), row, prev, next)
        }
        ("wad", _) => {
            let (h, l, c) = hlc()?;
            ind::wad_resume_one(&h, &l, &c, row, prev, next)
        }
        // kcw = 2·mult·ATR/EMA over the Keltner band's EMA + ATR pair
        ("kcw", _) => {
            let (h, l, c) = hlc()?;
            let (n_ema, n_atr) = next.split_at_mut(1);
            let ema = ind::ema_resume_one(&c, p(0), row, &prev[..1], n_ema)?;
            let atr = ind::atr_resume_one(&h, &l, &c, p(1), row, &prev[1..2], n_atr)?;
            Some(ind::kcw_value(ema, atr, arg_f64(args, 2)))
        }
        // the middle line is the EMA of close
        ("keltner", None) => ind::ema_resume_one(&close()?, p(0), row, prev, next),
        ("keltner", Some(band)) => {
            let (h, l, c) = hlc()?;
            let upper = band == "upper";
            let m = arg_f64(args, 2);
            ind::keltner_band_resume_one(&c, &h, &l, p(0), p(1), m, upper, row, prev, next)
        }
        ("supertrend", sub) => {
            let (h, l, c) = hlc()?;
            let dir = sub == Some("direction");
            let m = arg_f64(args, 1);
            ind::supertrend_resume_one(&h, &l, &c, p(0), m, dir, row, prev, next)
        }
        ("trix", _) => ind::trix_resume_one(&close()?, p(0), row, prev, next),
        ("plus_dm", _) => {
            let (h, l) = hl()?;
            ind::plus_dm_resume_one(&h, &l, p(0), row, prev, next)
        }
        ("minus_dm", _) => {
            let (h, l) = hl()?;
            ind::minus_dm_resume_one(&h, &l, p(0), row, prev, next)
        }
        ("crsi", _) => ind::crsi_resume_one(&ops.f64(0, "close").ok()?, arg_usize(args, 0), arg_usize(args, 1), arg_usize(args, 2), row, prev, next),
        _ => unreachable!("a recursive Step command has a scalar kernel"), // LCOV_EXCL_LINE
    }
}
