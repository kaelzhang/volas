//! Resume directive execution over only the new tail rows, bit-identical to a
//! full recompute (the per-directive `execute_resume*` dispatch).

use crate::exec::{arg_f64, arg_usize, series_f64};
use crate::parse;
use crate::types::Ast;
use volas_compute::indicators as ind;
use volas_core::{Column, DataFrame};

use super::as_command;

/// Single new-row resume for a recursive indicator whose carried state is a single
/// element equal to its output (`ema` / `smma`): returns the value at `row` — which is
/// also the new state `[value]` — with NO `Vec` allocation, the live single-bar fast
/// path. The caller writes the value with `update_computed_f64_value` and the state in
/// place with `update_computed_state_at`. `None` for any other directive (the caller
/// then takes the `Vec`-returning resume path). Bit-identical to the full resume: both
/// go through the same shared `*_step` kernel.
pub fn execute_resume_one(
    df: &DataFrame,
    directive: &str,
    prev_state: &[f64],
    row: usize,
) -> Option<f64> {
    // Cheap name gate FIRST: skip the parse entirely for every directive this scalar
    // path can't serve (rsi / macd / …), so a non-resumable recursive column does not
    // pay a wasted parse here on top of the parse the general path already does.
    let name = directive.split([':', '@']).next().unwrap_or(directive);
    if !matches!(name, "ema" | "smma" | "atr" | "natr" | "rsi" | "cmo") {
        return None;
    }
    if directive.contains('@') {
        return None; // explicit @series -> general path (avoids the whole-column materialize)
    }
    let node = parse(directive).ok()?;
    execute_resume_one_node(df, &node, prev_state, row)
}

/// Node form of [`execute_resume_one`] — the single-bar scalar resume for the
/// `ema`/`smma`/`atr`/`natr`/`rsi`/`cmo` family from an already-parsed `node`, so the
/// live forming-row refresh (which parses the directive once to gate it) does NOT parse
/// it a second time here. Caller guarantees a default-series command node.
pub fn execute_resume_one_node(
    df: &DataFrame,
    node: &Ast,
    prev_state: &[f64],
    row: usize,
) -> Option<f64> {
    let (name, _sub, args, series) = as_command(node)?;
    let name = name.as_ref();
    if !matches!(name, "ema" | "smma" | "atr" | "natr" | "rsi" | "cmo") {
        return None;
    }
    let period = arg_usize(&args, 0);
    if name == "atr" || name == "natr" {
        // Wilder ATR (and NATR = ATR/close·100) read high/low/close; one fused step
        // replaces the two-`Vec` resume.
        let high = series_f64(df, series, 0, "high").ok()?;
        let low = series_f64(df, series, 1, "low").ok()?;
        let close = series_f64(df, series, 2, "close").ok()?;
        let atr = ind::atr_resume_one(&high, &low, &close, period, row, prev_state)?;
        return Some(if name == "natr" { atr / close[row] * 100.0 } else { atr });
    }
    let close = series_f64(df, series, 0, "close").ok()?; // borrowed for an F64 close
    // `name` is one of the gated close-series families above — dispatch directly, the
    // final arm (`cmo`) is the gate's remainder so there is no dead catch-all.
    if name == "ema" {
        ind::ema_resume_one(&close, period, row, prev_state)
    } else if name == "smma" {
        ind::smma_resume_one(&close, period, row, prev_state)
    } else if name == "rsi" {
        ind::rsi_resume_one(&close, period, row, prev_state)
    } else {
        ind::cmo_resume_one(&close, period, row, prev_state)
    }
}

/// Extract the period from a `name:<period>` directive for the momentum / ROC
/// family. These commands are required (no default), so the form is always explicit.
fn period_of(directive: &str, name: &str) -> Option<usize> {
    directive
        .strip_prefix(name)?
        .strip_prefix(':')?
        .parse()
        .ok()
}

/// Parse-free scalar twin of [`execute_resume_default_series`] for the dominant
/// single-bar append case.
pub fn execute_resume_default_series_one(
    df: &DataFrame,
    directive: &str,
    row: usize,
) -> Option<f64> {
    if directive.contains('@') || row >= df.height() {
        return None;
    }
    match directive {
        "avgprice" => {
            let open = series_f64(df, &[], 0, "open").ok()?;
            let high = series_f64(df, &[], 1, "high").ok()?;
            let low = series_f64(df, &[], 2, "low").ok()?;
            let close = series_f64(df, &[], 3, "close").ok()?;
            Some((open[row] + high[row] + low[row] + close[row]) / 4.0)
        }
        "medprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            Some((high[row] + low[row]) / 2.0)
        }
        "bop" => {
            let open = series_f64(df, &[], 0, "open").ok()?;
            let high = series_f64(df, &[], 1, "high").ok()?;
            let low = series_f64(df, &[], 2, "low").ok()?;
            let close = series_f64(df, &[], 3, "close").ok()?;
            let range = high[row] - low[row];
            Some(if range < 1e-14 { 0.0 } else { (close[row] - open[row]) / range })
        }
        "typprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            Some((high[row] + low[row] + close[row]) / 3.0)
        }
        "wclprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            Some((high[row] + low[row] + 2.0 * close[row]) / 4.0)
        }
        "tr" => {
            if row == 0 {
                return Some(f64::NAN);
            }
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            let prev_close = close[row - 1];
            Some(
                (high[row] - low[row])
                    .max((high[row] - prev_close).abs())
                    .max((low[row] - prev_close).abs()),
            )
        }
        _ => {
            let (kind, period) = ["mom", "roc", "rocp", "rocr", "rocr100"]
                .into_iter()
                .find_map(|kind| {
                    period_of(directive, kind).map(|period| (kind, period))
                })?;
            if row < period {
                return Some(f64::NAN);
            }
            let data = series_f64(df, &[], 0, "close").ok()?;
            let prior = data[row - period];
            Some(match kind {
                "mom" => data[row] - prior,
                _ if prior == 0.0 => 0.0,
                "roc" => (data[row] / prior - 1.0) * 100.0,
                "rocp" => data[row] / prior - 1.0,
                "rocr" => data[row] / prior,
                "rocr100" => data[row] / prior * 100.0,
                _ => unreachable!(), // LCOV_EXCL_LINE
            })
        }
    }
}

/// Parse-free resume for canonical default-series finite-memory directives.
/// This exists for the single-bar append hot path: the formula is cheaper than
/// reparsing the directive AST. Directives with explicit series (`@...`) or
/// expressions intentionally fall back to [`execute_resume`].
pub fn execute_resume_default_series(
    df: &DataFrame,
    directive: &str,
    from_row: usize,
) -> Option<(Column, Vec<f64>)> {
    if directive.contains('@') {
        return None;
    }
    match directive {
        "avgprice" => {
            let open = series_f64(df, &[], 0, "open").ok()?;
            let high = series_f64(df, &[], 1, "high").ok()?;
            let low = series_f64(df, &[], 2, "low").ok()?;
            let close = series_f64(df, &[], 3, "close").ok()?;
            let mut out = Vec::with_capacity(open.len().saturating_sub(from_row));
            for i in from_row..open.len() {
                out.push((open[i] + high[i] + low[i] + close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "medprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let mut out = Vec::with_capacity(high.len().saturating_sub(from_row));
            for i in from_row..high.len() {
                out.push((high[i] + low[i]) / 2.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "bop" => {
            let open = series_f64(df, &[], 0, "open").ok()?;
            let high = series_f64(df, &[], 1, "high").ok()?;
            let low = series_f64(df, &[], 2, "low").ok()?;
            let close = series_f64(df, &[], 3, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                let range = high[i] - low[i];
                out.push(if range < 1e-14 { 0.0 } else { (close[i] - open[i]) / range });
            }
            Some((Column::f64(out), Vec::new()))
        }
        "typprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + close[i]) / 3.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "wclprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + 2.0 * close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "tr" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            let mut out = Vec::with_capacity(high.len().saturating_sub(from_row));
            for i in from_row..high.len() {
                if i == 0 {
                    out.push(f64::NAN);
                } else {
                    let prev_close = close[i - 1];
                    out.push(
                        (high[i] - low[i])
                            .max((high[i] - prev_close).abs())
                            .max((low[i] - prev_close).abs()),
                    );
                }
            }
            Some((Column::f64(out), Vec::new()))
        }
        _ => {
            let (kind, period) = ["mom", "roc", "rocp", "rocr", "rocr100"]
                .into_iter()
                .find_map(|kind| {
                    period_of(directive, kind).map(|period| (kind, period))
                })?;
            let data = series_f64(df, &[], 0, "close").ok()?;
            let mut out = Vec::with_capacity(data.len().saturating_sub(from_row));
            for i in from_row..data.len() {
                if i < period {
                    out.push(f64::NAN);
                    continue;
                }
                let prior = data[i - period];
                let value = match kind {
                    "mom" => data[i] - prior,
                    _ if prior == 0.0 => 0.0,
                    "roc" => (data[i] / prior - 1.0) * 100.0,
                    "rocp" => data[i] / prior - 1.0,
                    "rocr" => data[i] / prior,
                    "rocr100" => data[i] / prior * 100.0,
                    _ => unreachable!(), // LCOV_EXCL_LINE
                };
                out.push(value);
            }
            Some((Column::f64(out), Vec::new()))
        }
    }
}

/// Resume `node` from `prev_state` over rows `[from_row, height)`, returning the
/// new-row [`Column`] and the updated state. `None` when `node` has no resume
/// kernel (caller falls back to a full recompute). The values are bit-identical to
/// a fresh full recompute, so writing them into the stale tail keeps the cached
/// column exact.
///
/// `origin` is the original-frame row this (possibly sliced) frame's row 0 maps to
/// (`ComputedMeta::origin`). Recursive *value* indicators ignore it; the
/// absolute-position index family adds it back so emitted positions stay
/// original-absolute across a head-dropping slice.
pub fn execute_resume(
    df: &DataFrame,
    node: &Ast,
    prev_state: &[f64],
    from_row: usize,
    origin: usize,
) -> Option<(Column, Vec<f64>)> {
    let (name, sub, args, series) = as_command(node)?;
    let sub = sub.as_deref();
    let close = || series_f64(df, series, 0, "close");
    match (name.as_ref(), sub) {
        // Stateless finite-memory resume. These kernels depend only on the row
        // being refreshed plus a fixed prior row, so append can produce exactly the
        // stale tail without probing a window or recomputing the full column.
        ("avgprice", _) => {
            let open = series_f64(df, series, 0, "open").ok()?;
            let high = series_f64(df, series, 1, "high").ok()?;
            let low = series_f64(df, series, 2, "low").ok()?;
            let close = series_f64(df, series, 3, "close").ok()?;
            let mut out = Vec::with_capacity(open.len().saturating_sub(from_row));
            for i in from_row..open.len() {
                out.push((open[i] + high[i] + low[i] + close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("medprice", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let mut out = Vec::with_capacity(high.len().saturating_sub(from_row));
            for i in from_row..high.len() {
                out.push((high[i] + low[i]) / 2.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("bop", _) => {
            let open = series_f64(df, series, 0, "open").ok()?;
            let high = series_f64(df, series, 1, "high").ok()?;
            let low = series_f64(df, series, 2, "low").ok()?;
            let close = series_f64(df, series, 3, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                let range = high[i] - low[i];
                out.push(if range < 1e-14 { 0.0 } else { (close[i] - open[i]) / range });
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("typprice", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + close[i]) / 3.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("wclprice", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + 2.0 * close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("tr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let mut out = Vec::with_capacity(high.len().saturating_sub(from_row));
            for i in from_row..high.len() {
                if i == 0 {
                    out.push(f64::NAN);
                } else {
                    let prev_close = close[i - 1];
                    out.push(
                        (high[i] - low[i])
                            .max((high[i] - prev_close).abs())
                            .max((low[i] - prev_close).abs()),
                    );
                }
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("mom" | "roc" | "rocp" | "rocr" | "rocr100", _) => {
            let data = close().ok()?;
            let period = arg_usize(&args, 0);
            let mut out = Vec::with_capacity(data.len().saturating_sub(from_row));
            for i in from_row..data.len() {
                if i < period {
                    out.push(f64::NAN);
                    continue;
                }
                let prior = data[i - period];
                let value = match name.as_ref() {
                    "mom" => data[i] - prior,
                    _ if prior == 0.0 => 0.0,
                    "roc" => (data[i] / prior - 1.0) * 100.0,
                    "rocp" => data[i] / prior - 1.0,
                    "rocr" => data[i] / prior,
                    "rocr100" => data[i] / prior * 100.0,
                    _ => unreachable!(), // LCOV_EXCL_LINE
                };
                out.push(value);
            }
            Some((Column::f64(out), Vec::new()))
        }

        ("obv", _) => {
            let real = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) = ind::obv_resume(&real, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("ad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            let (vals, st) = ind::ad_resume(&high, &low, &close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("adosc", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            let fast = arg_usize(&args, 0);
            let slow = arg_usize(&args, 1);
            let (vals, st) = ind::adosc_resume(
                &high, &low, &close, &volume, fast, slow, from_row, prev_state,
            );
            Some((Column::f64(vals), st))
        }

        // Group A cumulative / EMA-recursion family (volas-exclusive).
        ("pvt", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) = ind::pvt_resume(&close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("nvi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) = ind::nvi_resume(&close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("pvi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) = ind::pvi_resume(&close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("efi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) =
                ind::efi_resume(&close, &volume, arg_usize(&args, 0), from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("tsi", _) => {
            let (vals, st) = ind::tsi_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                arg_usize(&args, 1),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("mass_index", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) =
                ind::mass_index_resume(&high, &low, arg_usize(&args, 0), from_row, prev_state);
            Some((Column::f64(vals), st))
        }

        // Keltner Channels (Group B): middle resumes the EMA; bands resume the EMA + ATR pair.
        ("keltner", None) => {
            let (vals, st) = ind::ema_resume(
                &series_f64(df, series, 0, "close").ok()?,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("keltner", Some(sub)) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::keltner_band_resume(
                &close,
                &high,
                &low,
                arg_usize(&args, 0),
                arg_usize(&args, 1),
                arg_f64(&args, 2),
                sub == "upper",
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // Group B cumulative family (volas-exclusive): resume the running line.
        ("wad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::wad_resume(&high, &low, &close, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("asi", _) => {
            let open = series_f64(df, series, 0, "open").ok()?;
            let high = series_f64(df, series, 1, "high").ok()?;
            let low = series_f64(df, series, 2, "low").ok()?;
            let close = series_f64(df, series, 3, "close").ok()?;
            let (vals, st) = ind::asi_resume(
                &open,
                &high,
                &low,
                &close,
                arg_f64(&args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("supertrend", sub) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::supertrend_resume(
                &high,
                &low,
                &close,
                arg_usize(&args, 0),
                arg_f64(&args, 1),
                sub == Some("direction"),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // SAR family — resume the state machine from the carried tuple. A resume at
        // `from_row < 2` (the SAR bootstrap needs bars 0 and 1) returns `None` and falls back.
        ("sar", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) = ind::sar_resume(
                &high,
                &low,
                arg_f64(&args, 0),
                arg_f64(&args, 1),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("sarext", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            // `start_value` (arg 0) only steers the bar-1 bootstrap, never re-run on resume.
            let (vals, st) = ind::sarext_resume(
                &high,
                &low,
                arg_f64(&args, 1),
                arg_f64(&args, 2),
                arg_f64(&args, 3),
                arg_f64(&args, 4),
                arg_f64(&args, 5),
                arg_f64(&args, 6),
                arg_f64(&args, 7),
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
                arg_usize(&args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("smma", _) => {
            let (vals, st) = ind::smma_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("dema", _) => {
            let (vals, st) = ind::dema_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("tema", _) => {
            let (vals, st) = ind::tema_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("t3", _) => {
            let (vals, st) = ind::t3_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                arg_f64(&args, 1),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("trix", _) => {
            let (vals, st) = ind::trix_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        // KAMA's sliding-sum resume can decline (short retained head) → None falls back.
        ("kama", _) => {
            let (vals, st) = ind::kama_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        ("macd", None) => {
            let (vals, st) = ind::macd_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                arg_usize(&args, 1),
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("macd", Some(line @ ("signal" | "histogram"))) => {
            let (vals, st) = ind::macd_signal_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                arg_usize(&args, 1),
                arg_usize(&args, 2),
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
                arg_usize(&args, 0),
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
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("cmo", _) => {
            let (vals, st) = ind::cmo_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("atr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::atr_resume(
                &high,
                &low,
                &close,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("natr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::natr_resume(
                &high,
                &low,
                &close,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("plus_dm", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) = ind::plus_dm_resume(
                &high,
                &low,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minus_dm", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) = ind::minus_dm_resume(
                &high,
                &low,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("plus_di", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::plus_di_resume(
                &high,
                &low,
                &close,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minus_di", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::minus_di_resume(
                &high,
                &low,
                &close,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("dx", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::dx_resume(
                &high,
                &low,
                &close,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("adx", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::adx_resume(
                &high,
                &low,
                &close,
                arg_usize(&args, 0),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("adxr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::adxr_resume(
                &high,
                &low,
                &close,
                arg_usize(&args, 0),
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
                arg_f64(&args, 0),
                arg_f64(&args, 1),
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
                arg_usize(&args, 0),
                from_row,
                origin,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minindex", _) | ("minmaxindex", Some("min")) => {
            let (vals, st) = ind::minindex_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                from_row,
                origin,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // StochRSI — a windowed %K (and SMA `.d`) of the Wilder-recursive RSI; resume by
        // carrying the RSI Wilder pair + the recent RSI values feeding the windows. Only
        // the canonical SMA `.d` (matype 0) resumes; a recursive-MA `.d` declines.
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let is_d = line == "d";
            if is_d && arg_usize(&args, 3) != 0 {
                return None; // non-SMA `.d` smoothing is recursive — fall back.
            }
            let (vals, st) = ind::stochrsi_resume(
                &close().ok()?,
                arg_usize(&args, 0),
                arg_usize(&args, 1),
                is_d,
                arg_usize(&args, 2),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // KDJ — resume the recursive %K (+ %D for `.d`/`.j`) from the carried state; RSV is
        // finite-memory, recomputed over the windowed high/low/close tail.
        ("kdj", Some(line @ ("k" | "d" | "j"))) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let period_rsv = arg_usize(&args, 0);
            let period_k = arg_usize(&args, 1);
            let kline = match line {
                "k" => ind::KdjLine::K,
                "d" => ind::KdjLine::D,
                _ => ind::KdjLine::J,
            };
            let period_d = if line == "k" {
                3
            } else {
                arg_usize(&args, 2)
            };
            let (vals, st) = ind::kdj_resume(
                &high, &low, &close, period_rsv, period_k, period_d, kline, from_row, prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        _ => None,
    }
}
