// ---------------------------------------------------------------------------
// Statistic — rolling regression & dispersion
// ---------------------------------------------------------------------------

/// Least-squares fit of `y = m·x + b` over each trailing `period`-bar window, in
/// TA-Lib's coordinate convention: `x = 0` at the most recent bar, increasing into
/// the past (`x = period-1` at the oldest). Returns `(slope m, intercept b)` per
/// row, both NaN during warm-up. Shared by the whole linear-regression family.
/// O(n·period), matching TA-Lib's own inner loop (the constants close-form the
/// `Σx` / `Σx²` of `0..period`, so only `Σy` and `Σxy` are summed per window).
fn linreg_fit(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut slope = vec![f64::NAN; n];
    let mut intercept = vec![f64::NAN; n];
    if period == 0 || period > n {
        return (slope, intercept);
    }
    let p = period as f64;
    let sum_x = p * (period - 1) as f64 * 0.5;
    // (period-1)·period·(2·period-1)/6 = Σ_{k=0}^{period-1} k², always integral.
    let sum_x_sqr = (period * (period - 1) * (2 * period - 1) / 6) as f64;
    let divisor = sum_x * sum_x - p * sum_x_sqr;
    for today in (period - 1)..n {
        let mut sum_xy = 0.0;
        let mut sum_y = 0.0;
        for i in (0..period).rev() {
            let y = data[today - i];
            sum_y += y;
            sum_xy += i as f64 * y;
        }
        let m = (p * sum_xy - sum_x * sum_y) / divisor;
        slope[today] = m;
        intercept[today] = (sum_y - m * sum_x) / p;
    }
    (slope, intercept)
}

/// Linear regression value at the current bar: `b + m·(period-1)` (TA-Lib LINEARREG).
pub fn linearreg(data: &[f64], period: usize) -> Vec<f64> {
    let (m, b) = linreg_fit(data, period);
    let x = period.saturating_sub(1) as f64;
    b.iter().zip(&m).map(|(b, m)| b + m * x).collect()
}

/// Linear regression slope `m` (TA-Lib LINEARREG_SLOPE).
pub fn linearreg_slope(data: &[f64], period: usize) -> Vec<f64> {
    linreg_fit(data, period).0
}

/// Linear regression intercept `b` (TA-Lib LINEARREG_INTERCEPT).
pub fn linearreg_intercept(data: &[f64], period: usize) -> Vec<f64> {
    linreg_fit(data, period).1
}

/// Linear regression angle in degrees: `atan(m)·180/π` (TA-Lib LINEARREG_ANGLE).
pub fn linearreg_angle(data: &[f64], period: usize) -> Vec<f64> {
    let deg = 180.0 / std::f64::consts::PI;
    linreg_fit(data, period)
        .0
        .iter()
        .map(|m| m.atan() * deg)
        .collect()
}

/// Time series forecast — regression value one bar ahead: `b + m·period` (TA-Lib TSF).
pub fn tsf(data: &[f64], period: usize) -> Vec<f64> {
    let (m, b) = linreg_fit(data, period);
    let x = period as f64;
    b.iter().zip(&m).map(|(b, m)| b + m * x).collect()
}

/// Rolling population variance over `period`: `Σx²/p − (Σx/p)²`, O(n) via a sliding
/// sum and sum-of-squares — TA-Lib's own approach, with its exact operation order
/// (add current, take means, drop trailing). NaN during warm-up. Shared by `var`
/// and `stddev`. Floating-point cancellation can yield a tiny negative result; the
/// `stddev` caller clamps such (and an exact-zero, flat window) to 0, as TA-Lib does.
fn rolling_var(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let p = period as f64;
    let mut total1 = 0.0; // Σx over the window
    let mut total2 = 0.0; // Σx² over the window
    for &x in &data[..period - 1] {
        total1 += x;
        total2 += x * x;
    }
    let mut trailing = 0;
    for i in (period - 1)..n {
        let x = data[i];
        total1 += x;
        total2 += x * x;
        let mean1 = total1 / p;
        let mean2 = total2 / p;
        let old = data[trailing];
        trailing += 1;
        total1 -= old;
        total2 -= old * old;
        out[i] = mean2 - mean1 * mean1;
    }
    out
}

/// Rolling population variance over `period` (TA-Lib VAR). Lookback period-1.
/// (TA-Lib's `nbdev` parameter is a no-op on the variance; it is dropped here.)
pub fn var(data: &[f64], period: usize) -> Vec<f64> {
    rolling_var(data, period)
}

/// Rolling standard deviation: `nbdev · sqrt(variance)` over `period` (TA-Lib STDDEV).
/// A non-positive variance (fp cancellation, or a perfectly flat window) yields 0,
/// matching TA-Lib. Lookback period-1.
pub fn stddev(data: &[f64], period: usize, nbdev: f64) -> Vec<f64> {
    rolling_var(data, period)
        .into_iter()
        .map(|v| {
            if v.is_nan() {
                f64::NAN
            } else if v > 0.0 {
                v.sqrt() * nbdev
            } else {
                0.0
            }
        })
        .collect()
}
