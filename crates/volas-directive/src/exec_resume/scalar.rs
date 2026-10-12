//! The single-row resume step: one new row of a recursive indicator — its value and the
//! carried state as of that row — with no allocation. Mirrors [`super::execute_resume`]
//! arm-for-arm: each arm calls the scalar twin of that arm's kernel, bit-identical to one
//! iteration of its loop. A cached recursive column refreshes its newest row through
//! this step (a plain append, a replaced last row, a time-frame forming row), writing the
//! advanced state into a buffer it keeps, so the live path allocates nothing.

use std::cell::Cell;

use crate::complete::BoundCommand;
use crate::exec::{arg_f64, arg_usize};
use crate::operands::Operands;
use crate::types::{ArgValue, Ast};
use volas_compute::raw::indicators as ind;
use volas_core::DataFrame;

use super::as_command;

/// One row of the recursive `node` at `row`, continued from `prev_state` (the state as of
/// the last complete row before `row`): returns the row's value and writes the state as of
/// `row` into `next` (which must be `prev_state`'s length). A row missing an input is NaN
/// and does not advance the state (`next` is `prev_state`). `None` when `node` has no
/// single-row step, or the step declines (`row` out of range, a state of the wrong
/// shape) — the caller then takes the `Vec` resume.
pub fn execute_resume_step(
    df: &DataFrame,
    node: &Ast,
    prev_state: &[f64],
    row: usize,
    next: &mut [f64],
) -> Option<f64> {
    let (name, sub, args, series) = as_command(node)?;
    let (name, sub) = (name.as_ref(), sub.as_deref());
    if next.len() != prev_state.len() {
        return None;
    }
    if !crate::spec::is_recursive(name, &args) {
        // a finite-window command resumes from its kernel's running sums
        let ops = Operands::new(df, series);
        return super::window::step(&ops, name, sub, &args, prev_state, row, next);
    }
    if !has_step(name, sub) {
        return None;
    }
    // Over the complete rows (every command with a step is recursive): a missing row is
    // NaN, its state untouched.
    let cmd = BoundCommand { name, sub, args: &args, series };
    let missing = Cell::new(false);
    let out = crate::complete::resume_one(
        df,
        &cmd,
        row,
        || {
            missing.set(true);
            Some(f64::NAN)
        },
        |ops, row| step_of(ops, name, sub, &args, prev_state, row, next),
    );
    if missing.get() {
        next.copy_from_slice(prev_state);
    }
    out
}

/// Whether the command `name` / `sub` has a single-row step. Without one (the index
/// family, `stochrsi`, `adosc`, `asi`, `kdj`, the Hilbert family and `mama`), a refresh
/// takes the `Vec` resume.
fn has_step(name: &str, sub: Option<&str>) -> bool {
    match name {
        "macd" | "macdfix" => matches!(sub, None | Some("signal" | "histogram")),
        _ => matches!(
            name,
            "ema" | "smma" | "atr" | "natr" | "rsi" | "cmo" | "obv" | "ad" | "dema" | "tema"
                | "t3" | "kama" | "sar" | "sarext" | "plus_di" | "minus_di" | "dx" | "adx"
                | "adxr" | "pvt" | "nvi" | "pvi" | "efi" | "tsi" | "mass_index" | "wad"
                | "keltner" | "supertrend" | "trix" | "plus_dm" | "minus_dm"
        ),
    }
}

/// The step of the bound command `name` / `sub` at `row` over `ops` (see
/// [`execute_resume_step`]).
#[allow(clippy::too_many_arguments)] // the step's own inputs, passed through
fn step_of(
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
        // `has_step` admits nothing else
        _ => {
            let (h, l) = hl()?;
            ind::minus_dm_resume_one(&h, &l, p(0), row, prev, next)
        }
    }
}
