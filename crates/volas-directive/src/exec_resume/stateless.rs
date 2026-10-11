//! Parse-free resume of the canonical default-series finite-memory directives (price
//! transforms, TR, the momentum / ROC family): the stale tail is a formula of each row
//! and a fixed prior row, so an append computes exactly the new rows.

use crate::exec::series_f64;
use volas_core::{Column, DataFrame};

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
