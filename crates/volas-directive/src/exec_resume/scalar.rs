//! Scalar value-only resume dispatch: the single forming-row VALUE for a recursive
//! indicator, allocation-free. Mirrors [`super::execute_resume`] arm-for-arm but returns
//! the `f64` value instead of `(Column, Vec)`, so the live tf-fold forming-row refresh of
//! a cached recursive column allocates nothing. Each arm is bit-identical to the matching
//! `execute_resume` arm's single loop iteration. `None` for a directive without a scalar
//! twin (index family, stochrsi `.d`) — the caller then falls back to `execute_resume`.

use crate::exec::{arg_f64, arg_usize, series_f64};
use crate::types::Ast;
use volas_compute::indicators as ind;
use volas_core::DataFrame;

use super::as_command;

/// The single forming-row value at `row` for recursive `node`, continued from
/// `prev_state` (as of `row - 1`), with NO allocation. `None` when no scalar twin exists.
pub fn execute_resume_value(
    df: &DataFrame,
    node: &Ast,
    prev_state: &[f64],
    row: usize,
    _origin: usize,
) -> Option<f64> {
    let (name, sub, args, series) = as_command(node)?;
    let sub = sub.as_deref();
    match (name.as_ref(), sub) {
        ("obv", _) => {
            let real = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::obv_resume_one(&real, &volume, row, prev_state)
        }
        ("ad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            ind::ad_resume_one(&high, &low, &close, &volume, row, prev_state)
        }
        ("dema", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::dema_resume_one(&close, arg_usize(&args, 0), row, prev_state)
        }
        ("tema", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::tema_resume_one(&close, arg_usize(&args, 0), row, prev_state)
        }
        ("t3", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::t3_resume_one(&close, arg_usize(&args, 0), arg_f64(&args, 1), row, prev_state)
        }
        ("kama", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::kama_resume_one(&close, arg_usize(&args, 0), row, prev_state)
        }
        ("macd", None) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::macd_resume_one(&close, arg_usize(&args, 0), arg_usize(&args, 1), row, prev_state)
        }
        // macdfix (fixed 12/26):
        ("macdfix", None) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::macd_resume_one(&close, 12, 26, row, prev_state)
        }
        ("macd", Some(line @ ("signal" | "histogram"))) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::macd_signal_resume_one(
                &close,
                arg_usize(&args, 0),
                arg_usize(&args, 1),
                arg_usize(&args, 2),
                line == "histogram",
                row,
                prev_state,
            )
        }
        // macdfix (fixed 12/26, signal is arg 0):
        ("macdfix", Some(line @ ("signal" | "histogram"))) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::macd_signal_resume_one(
                &close,
                12,
                26,
                arg_usize(&args, 0),
                line == "histogram",
                row,
                prev_state,
            )
        }
        ("sar", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::sar_resume_one(&high, &low, row, prev_state)
        }
        ("sarext", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::sarext_resume_one(&high, &low, arg_f64(&args, 1), row, prev_state)
        }

        // Notes: same two series reads (IDX 0 "high", 1 "low") as the Vec arm. Of the SAREXT params,
        ("plus_di", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::plus_di_resume_one(&high, &low, &close, arg_usize(&args, 0), row, prev_state)
        }
        ("minus_di", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::minus_di_resume_one(&high, &low, &close, arg_usize(&args, 0), row, prev_state)
        }
        ("dx", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::dx_resume_one(&high, &low, &close, arg_usize(&args, 0), row, prev_state)
        }
        ("adx", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::adx_resume_one(&high, &low, &close, arg_usize(&args, 0), row, prev_state)
        }
        ("adxr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::adxr_resume_one(&high, &low, &close, arg_usize(&args, 0), row, prev_state)
        }
        ("pvt", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::pvt_resume_one(&close, &volume, row, prev_state)
        }
        ("nvi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::nvi_resume_one(&close, &volume, row, prev_state)
        }
        ("pvi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::pvi_resume_one(&close, &volume, row, prev_state)
        }
        ("efi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::efi_resume_one(&close, &volume, arg_usize(&args, 0), row, prev_state)
        }
        ("tsi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::tsi_resume_one(&close, arg_usize(&args, 0), arg_usize(&args, 1), row, prev_state)
        }
        ("mass_index", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::mass_index_resume_one(&high, &low, arg_usize(&args, 0), row, prev_state)
        }
        ("wad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::wad_resume_one(&high, &low, &close, row, prev_state)
        }
        ("keltner", None) => {
            // middle line = EMA; resumes through the ema scalar twin directly
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::ema_resume_one(&close, arg_usize(&args, 0), row, prev_state)
        }
        ("keltner", Some(sub)) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::keltner_band_resume_one(
                &close, &high, &low,
                arg_usize(&args, 0), arg_usize(&args, 1), arg_f64(&args, 2),
                sub == "upper", row, prev_state,
            )
        }
        ("supertrend", sub) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::supertrend_resume_one(
                &high, &low, &close,
                arg_usize(&args, 0), arg_f64(&args, 1),
                sub == Some("direction"), row, prev_state,
            )
        }
        ("trix", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            ind::trix_resume_one(&close, arg_usize(&args, 0), row, prev_state)
        }
        ("plus_dm", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::plus_dm_resume_one(&high, &low, arg_usize(&args, 0), row, prev_state)
        }
        ("minus_dm", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::minus_dm_resume_one(&high, &low, arg_usize(&args, 0), row, prev_state)
        }
        _ => None,
    }
}
