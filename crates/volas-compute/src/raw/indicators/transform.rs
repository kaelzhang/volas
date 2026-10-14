//! Price transforms share the same per-row term in batch and scalar evaluation.
use crate::missing;

/// Average price: `(open + high + low + close) / 4`.
pub fn avgprice(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len()).map(|i| avgprice_value(open[i], high[i], low[i], close[i])).collect()
}

/// Average price of one bar, missing when an input is missing.
#[inline]
pub fn avgprice_value(open: f64, high: f64, low: f64, close: f64) -> f64 {
    missing::term(&[open, high, low, close], (open + high + low + close) / 4.0)
}

/// Median price: `(high + low) / 2`.
pub fn medprice(high: &[f64], low: &[f64]) -> Vec<f64> {
    (0..high.len()).map(|i| medprice_value(high[i], low[i])).collect()
}

/// Median price of one bar, missing when an input is missing.
#[inline]
pub fn medprice_value(high: f64, low: f64) -> f64 {
    missing::term(&[high, low], (high + low) / 2.0)
}

/// Typical price: `(high + low + close) / 3`.
pub fn typprice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len()).map(|i| typprice_value(high[i], low[i], close[i])).collect()
}

/// Typical price of one bar, missing when an input is missing.
#[inline]
pub fn typprice_value(high: f64, low: f64, close: f64) -> f64 {
    missing::term(&[high, low, close], (high + low + close) / 3.0)
}

/// Weighted close price: `(high + low + 2*close) / 4`.
pub fn wclprice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len()).map(|i| wclprice_value(high[i], low[i], close[i])).collect()
}

/// Weighted close of one bar, missing when an input is missing.
#[inline]
pub fn wclprice_value(high: f64, low: f64, close: f64) -> f64 {
    missing::term(&[high, low, close], (high + low + 2.0 * close) / 4.0)
}
