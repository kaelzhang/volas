//! Price transforms share the same per-row term in batch and scalar evaluation.
use crate::missing;

/// Average price: `(open + high + low + close) / 4`.
pub fn avgprice(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    avgprice_with(open, high, low, close, false)
}

/// Average price of one bar, missing when an input is missing.
#[inline]
pub fn avgprice_value(open: f64, high: f64, low: f64, close: f64) -> f64 {
    let value = avgprice_ieee(open, high, low, close);
    if value.is_finite() { value } else { missing::nonfinite_term4(open, high, low, close, value) }
}

/// Median price: `(high + low) / 2`.
pub fn medprice(high: &[f64], low: &[f64]) -> Vec<f64> {
    medprice_with(high, low, false)
}

/// Median price of one bar, missing when an input is missing.
#[inline]
pub fn medprice_value(high: f64, low: f64) -> f64 {
    let value = medprice_ieee(high, low);
    if value.is_finite() { value } else { missing::nonfinite_term2(high, low, value) }
}

/// Typical price: `(high + low + close) / 3`.
pub fn typprice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    typprice_with(high, low, close, false)
}

/// Typical price of one bar, missing when an input is missing.
#[inline]
pub fn typprice_value(high: f64, low: f64, close: f64) -> f64 {
    let value = typprice_ieee(high, low, close);
    if value.is_finite() { value } else { missing::nonfinite_term3(high, low, close, value) }
}

/// Weighted close price: `(high + low + 2*close) / 4`.
pub fn wclprice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    wclprice_with(high, low, close, false)
}

/// Weighted close of one bar, missing when an input is missing.
#[inline]
pub fn wclprice_value(high: f64, low: f64, close: f64) -> f64 {
    let value = wclprice_ieee(high, low, close);
    if value.is_finite() { value } else { missing::nonfinite_term3(high, low, close, value) }
}

/// `avgprice` over inputs already admitted as finite.
pub fn avgprice_complete(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    avgprice_with(open, high, low, close, true)
}

/// `medprice` over inputs already admitted as finite.
pub fn medprice_complete(high: &[f64], low: &[f64]) -> Vec<f64> {
    medprice_with(high, low, true)
}

/// `typprice` over inputs already admitted as finite.
pub fn typprice_complete(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    typprice_with(high, low, close, true)
}

/// `wclprice` over inputs already admitted as finite.
pub fn wclprice_complete(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    wclprice_with(high, low, close, true)
}

/// Batch price transform with a remembered input admission fact.
pub fn avgprice_with(open: &[f64], high: &[f64], low: &[f64], close: &[f64], complete: bool) -> Vec<f64> {
    missing::map_terms_with(close.len(), 0, complete, |i| [open[i], high[i], low[i], close[i]], |_, [open, high, low, close]| {
        let value = avgprice_ieee(open, high, low, close);
        // A non-finite original operand always makes this linear result non-finite.
        // A finite-input overflow only enters the exact cold check.
        (value, value)
    })
}

#[inline(always)]
fn avgprice_ieee(open: f64, high: f64, low: f64, close: f64) -> f64 {
    (open + high + low + close) / 4.0
}

/// Batch price transform with a remembered input admission fact.
pub fn medprice_with(high: &[f64], low: &[f64], complete: bool) -> Vec<f64> {
    missing::map_terms_with(high.len(), 0, complete, |i| [high[i], low[i]], |_, [high, low]| {
        let value = medprice_ieee(high, low);
        // A non-finite original operand always makes this linear result non-finite.
        // A finite-input overflow only enters the exact cold check.
        (value, value)
    })
}

#[inline(always)]
fn medprice_ieee(high: f64, low: f64) -> f64 {
    (high + low) / 2.0
}

/// Batch price transform with a remembered input admission fact.
pub fn typprice_with(high: &[f64], low: &[f64], close: &[f64], complete: bool) -> Vec<f64> {
    missing::map_terms_with(close.len(), 0, complete, |i| [high[i], low[i], close[i]], |_, [high, low, close]| {
        let value = typprice_ieee(high, low, close);
        // A non-finite original operand always makes this linear result non-finite.
        // A finite-input overflow only enters the exact cold check.
        (value, value)
    })
}

#[inline(always)]
fn typprice_ieee(high: f64, low: f64, close: f64) -> f64 {
    (high + low + close) / 3.0
}

/// Batch price transform with a remembered input admission fact.
pub fn wclprice_with(high: &[f64], low: &[f64], close: &[f64], complete: bool) -> Vec<f64> {
    missing::map_terms_with(close.len(), 0, complete, |i| [high[i], low[i], close[i]], |_, [high, low, close]| {
        let value = wclprice_ieee(high, low, close);
        // A non-finite original operand always makes this linear result non-finite.
        // A finite-input overflow only enters the exact cold check.
        (value, value)
    })
}

#[inline(always)]
fn wclprice_ieee(high: f64, low: f64, close: f64) -> f64 {
    (high + low + 2.0 * close) / 4.0
}
