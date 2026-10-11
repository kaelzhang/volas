// ---------------------------------------------------------------------------
// Price transform
// ---------------------------------------------------------------------------

/// Average price: `(open + high + low + close) / 4`.
pub fn avgprice(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len())
        .map(|i| (open[i] + high[i] + low[i] + close[i]) / 4.0)
        .collect()
}

/// Median price: `(high + low) / 2`.
pub fn medprice(high: &[f64], low: &[f64]) -> Vec<f64> {
    (0..high.len()).map(|i| (high[i] + low[i]) / 2.0).collect()
}

/// Typical price: `(high + low + close) / 3`.
pub fn typprice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len())
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect()
}

/// Weighted close price: `(high + low + 2*close) / 4`.
pub fn wclprice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len())
        .map(|i| (high[i] + low[i] + 2.0 * close[i]) / 4.0)
        .collect()
}
