//! Price transforms share the same per-row term in batch and scalar evaluation.
use crate::missing;

// Expand the formula where columns are read. A scalar call evaluates all its
// arguments before doing the arithmetic; this keeps the batch loop's loads
// interleaved with its additions, without duplicating the scalar formula.
macro_rules! price_ieee {
    (avgprice, $o:expr, $h:expr, $l:expr, $c:expr) => { ($o + $h + $l + $c) / 4.0 };
    (medprice, $h:expr, $l:expr) => { ($h + $l) / 2.0 };
    (typprice, $h:expr, $l:expr, $c:expr) => { ($h + $l + $c) / 3.0 };
    (wclprice, $h:expr, $l:expr, $c:expr) => { ($h + $l + 2.0 * $c) / 4.0 };
}


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
    (0..close.len()).map(|i| price_ieee!(avgprice, open[i], high[i], low[i], close[i])).collect()
}

/// `medprice` over inputs already admitted as finite.
pub fn medprice_complete(high: &[f64], low: &[f64]) -> Vec<f64> {
    (0..high.len()).map(|i| price_ieee!(medprice, high[i], low[i])).collect()
}

/// `typprice` over inputs already admitted as finite.
pub fn typprice_complete(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len()).map(|i| price_ieee!(typprice, high[i], low[i], close[i])).collect()
}

/// `wclprice` over inputs already admitted as finite.
pub fn wclprice_complete(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len()).map(|i| price_ieee!(wclprice, high[i], low[i], close[i])).collect()
}

/// Batch price transform with a remembered input admission fact.
pub fn avgprice_with(open: &[f64], high: &[f64], low: &[f64], close: &[f64], complete: bool) -> Vec<f64> {
    if complete { return avgprice_complete(open, high, low, close); }
    avgprice_unclassified(open, high, low, close)
}

#[cold]
#[inline(never)]
fn avgprice_unclassified(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    missing::map_terms_with(close.len(), 0, false, |i| [open[i], high[i], low[i], close[i]], |_, [open, high, low, close]| {
        let value = avgprice_ieee(open, high, low, close);
        // A non-finite original operand always makes this linear result non-finite.
        // A finite-input overflow only enters the exact cold check.
        (value, value)
    })
}

#[inline(always)]
fn avgprice_ieee(open: f64, high: f64, low: f64, close: f64) -> f64 {
    price_ieee!(avgprice, open, high, low, close)
}

/// Batch price transform with a remembered input admission fact.
pub fn medprice_with(high: &[f64], low: &[f64], complete: bool) -> Vec<f64> {
    if complete { return medprice_complete(high, low); }
    medprice_unclassified(high, low)
}

#[cold]
#[inline(never)]
fn medprice_unclassified(high: &[f64], low: &[f64]) -> Vec<f64> {
    missing::map_terms_with(high.len(), 0, false, |i| [high[i], low[i]], |_, [high, low]| {
        let value = medprice_ieee(high, low);
        // A non-finite original operand always makes this linear result non-finite.
        // A finite-input overflow only enters the exact cold check.
        (value, value)
    })
}

#[inline(always)]
fn medprice_ieee(high: f64, low: f64) -> f64 {
    price_ieee!(medprice, high, low)
}

/// Batch price transform with a remembered input admission fact.
pub fn typprice_with(high: &[f64], low: &[f64], close: &[f64], complete: bool) -> Vec<f64> {
    if complete { return typprice_complete(high, low, close); }
    typprice_unclassified(high, low, close)
}

#[cold]
#[inline(never)]
fn typprice_unclassified(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    missing::map_terms_with(close.len(), 0, false, |i| [high[i], low[i], close[i]], |_, [high, low, close]| {
        let value = typprice_ieee(high, low, close);
        // A non-finite original operand always makes this linear result non-finite.
        // A finite-input overflow only enters the exact cold check.
        (value, value)
    })
}

#[inline(always)]
fn typprice_ieee(high: f64, low: f64, close: f64) -> f64 {
    price_ieee!(typprice, high, low, close)
}

/// Batch price transform with a remembered input admission fact.
pub fn wclprice_with(high: &[f64], low: &[f64], close: &[f64], complete: bool) -> Vec<f64> {
    if complete { return wclprice_complete(high, low, close); }
    wclprice_unclassified(high, low, close)
}

#[cold]
#[inline(never)]
fn wclprice_unclassified(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    missing::map_terms_with(close.len(), 0, false, |i| [high[i], low[i], close[i]], |_, [high, low, close]| {
        let value = wclprice_ieee(high, low, close);
        // A non-finite original operand always makes this linear result non-finite.
        // A finite-input overflow only enters the exact cold check.
        (value, value)
    })
}

#[inline(always)]
fn wclprice_ieee(high: f64, low: f64, close: f64) -> f64 {
    price_ieee!(wclprice, high, low, close)
}
