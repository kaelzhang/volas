//! Parse-free resume of the canonical default-series finite-memory directives (price
//! transforms, TR, the momentum / ROC family): the stale tail is a formula of each row
//! and a fixed prior row, so an append computes exactly the new rows.

use crate::exec::col_f64;
use volas_compute::raw::indicators as ind;
use volas_core::{Column, DataFrame};

/// The momentum / ROC family member and period of a `name:<period>` directive. These
/// commands are required (no default), so the form is always explicit.
fn momentum_of(directive: &str) -> Option<(ind::Momentum, usize)> {
    let (name, period) = directive.split_once(':')?;
    Some((ind::Momentum::of(name)?, period.parse().ok()?))
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
            let open = col_f64(df, "open").ok()?;
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            Some((open[row] + high[row] + low[row] + close[row]) / 4.0)
        }
        "medprice" => {
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            Some((high[row] + low[row]) / 2.0)
        }
        "bop" => {
            let open = col_f64(df, "open").ok()?;
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            let range = high[row] - low[row];
            Some(if range < 1e-14 { 0.0 } else { (close[row] - open[row]) / range })
        }
        "typprice" => {
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            Some((high[row] + low[row] + close[row]) / 3.0)
        }
        "wclprice" => {
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            Some((high[row] + low[row] + 2.0 * close[row]) / 4.0)
        }
        "tr" => {
            if row == 0 {
                return Some(f64::NAN);
            }
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            Some(ind::tr_at(&high, &low, &close, row))
        }
        _ => {
            let (kind, period) = momentum_of(directive)?;
            Some(kind.at(&col_f64(df, "close").ok()?, period, row))
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
            let open = col_f64(df, "open").ok()?;
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            let mut out = Vec::with_capacity(open.len().saturating_sub(from_row));
            for i in from_row..open.len() {
                out.push((open[i] + high[i] + low[i] + close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "medprice" => {
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let mut out = Vec::with_capacity(high.len().saturating_sub(from_row));
            for i in from_row..high.len() {
                out.push((high[i] + low[i]) / 2.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "bop" => {
            let open = col_f64(df, "open").ok()?;
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                let range = high[i] - low[i];
                out.push(if range < 1e-14 { 0.0 } else { (close[i] - open[i]) / range });
            }
            Some((Column::f64(out), Vec::new()))
        }
        "typprice" => {
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + close[i]) / 3.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "wclprice" => {
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + 2.0 * close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "tr" => {
            let high = col_f64(df, "high").ok()?;
            let low = col_f64(df, "low").ok()?;
            let close = col_f64(df, "close").ok()?;
            let out = (from_row..high.len())
                .map(|i| if i == 0 { f64::NAN } else { ind::tr_at(&high, &low, &close, i) })
                .collect();
            Some((Column::f64(out), Vec::new()))
        }
        _ => {
            let (kind, period) = momentum_of(directive)?;
            let data = col_f64(df, "close").ok()?;
            let out = (from_row..data.len()).map(|i| kind.at(&data, period, i)).collect();
            Some((Column::f64(out), Vec::new()))
        }
    }
}
