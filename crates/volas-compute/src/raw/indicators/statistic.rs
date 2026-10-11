use crate::guard::{all_finite, own_sums, state_after, window_sums, Order, Slide, Window};
use crate::guard::{NonFiniteRows, RowGuard, Unguarded};

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
    let (m, b, finite) = linreg_fit_with::<Unguarded>(data, period);
    if finite {
        (m, b)
    } else {
        let (m, b, _) = linreg_fit_with::<NonFiniteRows>(data, period);
        (m, b)
    }
}

/// The sliding fit under a row guard; also returns whether both running sums ended
/// finite. A window holding a non-finite row is fitted directly on its own rows.
fn linreg_fit_with<G: RowGuard>(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, bool) {
    let n = data.len();
    if period == 0 || period > n {
        return (vec![f64::NAN; n], vec![f64::NAN; n], true);
    }
    let p = period as f64;
    let sum_x = p * (period - 1) as f64 * 0.5;
    // (period-1)·period·(2·period-1)/6 = Σ_{k=0}^{period-1} k², always integral.
    let sum_x_sqr = (period * (period - 1) * (2 * period - 1) / 6) as f64;
    let divisor = sum_x * sum_x - p * sum_x_sqr;
    // Single-write (D2): NaN warm-up `[0, period-1)`, each valid row written once.
    let mut slope = crate::buf::OutBuf::warmup(n, period - 1);
    let mut intercept = crate::buf::OutBuf::warmup(n, period - 1);
    let mut emit = |today: usize, sum_y: f64, sum_xy: f64| {
        let m = (p * sum_xy - sum_x * sum_y) / divisor;
        slope.set(today, m);
        intercept.set(today, (sum_y - m * sum_x) / p);
    };
    // A window's own `(Σy, Σ age·y)` — the direct fit for a window holding a bad row.
    let window_sums = |today: usize| {
        data[today + 1 - period..=today]
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(sy, sxy), (k, &y)| (sy + y, sxy + (period - 1 - k) as f64 * y))
    };
    // Seed the first window with TA-Lib's per-bar accumulation order (`i` = age, the
    // newest bar weighted 0, the oldest `period-1`), then slide in O(1): when the
    // window advances one bar every retained point's age rises by one, so `Σ i·y`
    // gains `Σ y` and loses the departing point's full `period·y` term —
    // `sum_xy += sum_y − period·leaving` (old `sum_y`), then `sum_y += entering −
    // leaving`. This turns the rolling fit from O(n·period) to O(n); the running sums
    // drift ~1e-13 relative (well within the 1e-9 TA-Lib parity tolerance), exactly as
    // the WMA slide already relies on. Shared by linearreg / slope / intercept / angle
    // / tsf.
    let mut guard = G::default();
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    for i in (0..period).rev() {
        let y = data[period - 1 - i];
        if guard.enter(y.is_finite()) {
            sum_y += y;
            sum_xy += i as f64 * y;
        }
    }
    if guard.clean() {
        emit(period - 1, sum_y, sum_xy);
    } else {
        let (sy, sxy) = window_sums(period - 1);
        emit(period - 1, sy, sxy);
    }
    for today in period..n {
        let leaving = data[today - period];
        let entering = data[today];
        let came_in = guard.enter(entering.is_finite());
        let went_out = guard.leave(leaving.is_finite());
        if went_out {
            sum_xy += sum_y - p * leaving;
        } else {
            sum_xy += sum_y;
        }
        match (came_in, went_out) {
            (true, true) => sum_y += entering - leaving,
            (true, false) => sum_y += entering,
            (false, true) => sum_y -= leaving,
            (false, false) => {}
        }
        if guard.clean() {
            emit(today, sum_y, sum_xy);
        } else {
            let (sy, sxy) = window_sums(today);
            emit(today, sy, sxy);
        }
    }
    let finite = sum_y.is_finite() && sum_xy.is_finite();
    (slope.finish(), intercept.finish(), finite)
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
/// Window-local: a non-finite row affects only the windows holding it (a window
/// holding one is NaN — see [`crate::guard`]).
fn rolling_var(data: &[f64], period: usize) -> Vec<f64> {
    let (out, finite) = rolling_var_with::<Unguarded>(data, period);
    if finite {
        out
    } else {
        rolling_var_with::<NonFiniteRows>(data, period).0
    }
}

/// The sliding variance under a row guard; also returns whether both running sums
/// ended finite.
fn rolling_var_with<G: RowGuard>(data: &[f64], period: usize) -> (Vec<f64>, bool) {
    let n = data.len();
    if period == 0 || period > n {
        return (vec![f64::NAN; n], true);
    }
    let p = period as f64;
    let mut guard = G::default();
    let mut total1 = 0.0; // Σx over the window
    let mut total2 = 0.0; // Σx² over the window
    for &x in &data[..period - 1] {
        if guard.enter(x.is_finite()) {
            total1 += x;
            total2 += x * x;
        }
    }
    let mut trailing = 0;
    let mut out = crate::buf::OutBuf::warmup(n, period - 1);
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for i in (period - 1)..n {
        let x = data[i];
        if guard.enter(x.is_finite()) {
            total1 += x;
            total2 += x * x;
        }
        let value = if guard.clean() {
            let mean1 = total1 / p;
            let mean2 = total2 / p;
            mean2 - mean1 * mean1
        } else {
            let (s1, s2) = data[i + 1 - period..=i]
                .iter()
                .fold((0.0, 0.0), |(s1, s2), &x| (s1 + x, s2 + x * x));
            let mean1 = s1 / p;
            s2 / p - mean1 * mean1
        };
        let old = data[trailing];
        trailing += 1;
        if guard.leave(old.is_finite()) {
            total1 -= old;
            total2 -= old * old;
        }
        out.set(i, value);
    }
    (out.finish(), total1.is_finite() && total2.is_finite())
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

// ---------------------------------------------------------------------------
// Statistic — two-series relationships
// ---------------------------------------------------------------------------

/// Rolling Pearson correlation of `x` and `y` over `period` (TA-Lib CORREL):
/// `(Σxy − ΣxΣy/n) / sqrt((Σx²−(Σx)²/n)(Σy²−(Σy)²/n))`. A non-positive denominator
/// yields 0 (TA-Lib's guard). Lookback `period-1`.
pub fn correl(x: &[f64], y: &[f64], period: usize) -> Vec<f64> {
    let n = x.len();
    if period == 0 || period > n {
        return vec![f64::NAN; n];
    }
    // Single-write (D2): NaN warm-up `[0, period-1)`, valid region written once via the
    // output pointer below (no prefill memset); `out.finish()` proves it was filled.
    let mut out = crate::buf::OutBuf::warmup(n, period - 1);
    let pf = period as f64;
    // Precompute 1/period: the means are then per-element multiplies, not divisions
    // (TA-Lib divides by period three times per bar). The ~1e-16 difference from a true
    // divide is well within the 1e-9 parity tolerance (a near-zero denominator is gated
    // to 0 either way, so the catastrophic-cancellation case is unaffected).
    let inv_pf = 1.0 / pf;
    let (mut sx, mut sy, mut sx2, mut sy2, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.ptr();
    for i in 0..period {
        // `period <= n` and the hot loop below keeps both indices in range.
        let (xi, yi) = unsafe { (*x_ptr.add(i), *y_ptr.add(i)) };
        sx += xi;
        sy += yi;
        sx2 += xi * xi;
        sy2 += yi * yi;
        sxy += xi * yi;
    }
    let value = |sx: f64, sy: f64, sx2: f64, sy2: f64, sxy: f64| {
        let denom = (sx2 - sx * sx * inv_pf) * (sy2 - sy * sy * inv_pf);
        if denom < 1e-14 {
            0.0
        } else {
            (sxy - sx * sy * inv_pf) / denom.sqrt()
        }
    };
    unsafe {
        *out_ptr.add(period - 1) = value(sx, sy, sx2, sy2, sxy);
    }
    let mut trailing = 0;
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for i in period..n {
        // Drop the trailing values, then add the new ones (TA-Lib's order).
        let (tx, ty) = unsafe { (*x_ptr.add(trailing), *y_ptr.add(trailing)) };
        trailing += 1;
        sx -= tx;
        sx2 -= tx * tx;
        sxy -= tx * ty;
        sy -= ty;
        sy2 -= ty * ty;
        let (xi, yi) = unsafe { (*x_ptr.add(i), *y_ptr.add(i)) };
        sx += xi;
        sx2 += xi * xi;
        sxy += xi * yi;
        sy += yi;
        sy2 += yi * yi;
        unsafe {
            *out_ptr.add(i) = value(sx, sy, sx2, sy2, sxy);
        }
    }
    if [sx, sy, sx2, sy2, sxy].iter().all(|v| v.is_finite()) {
        return out.finish();
    }
    // A non-finite row entered: window-local careful path (see `crate::guard`).
    let mut out = crate::buf::OutBuf::warmup(n, period - 1);
    window_sums(n, period, Order::DropAdd, |i| {
        let (xi, yi) = (x[i], y[i]);
        [xi, yi, xi * xi, yi * yi, xi * yi]
    }, |i, [sx, sy, sx2, sy2, sxy]| out.set(i, value(sx, sy, sx2, sy2, sxy)));
    out.finish()
}

/// Rolling beta of `x` against `y` over `period` (TA-Lib BETA): the slope
/// `(n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)` of the two series' one-bar **returns**
/// `(p[i]−p[i-1])/p[i-1]` (a zero prior price gives a 0 return; a ~0 denominator
/// gives 0). First value at index `period` (lookback = period).
pub fn beta(x: &[f64], y: &[f64], period: usize) -> Vec<f64> {
    let n = x.len();
    if period == 0 || period + 1 > n {
        return vec![f64::NAN; n];
    }
    // Single-write (D2): NaN warm-up `[0, period)`, valid region written once via
    // `out.set` (no prefill memset); `out.finish()` proves it was filled.
    let mut out = crate::buf::OutBuf::warmup(n, period);
    let ret = |arr: &[f64], i: usize| -> f64 {
        let prev = arr[i - 1];
        if prev.abs() < 1e-14 {
            0.0
        } else {
            (arr[i] - prev) / prev
        }
    };
    let pf = period as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    // Keep the return values that are currently inside the rolling window. TA-Lib
    // conceptually drops a return after emitting a row; storing it once avoids
    // recalculating the departing high/low returns, which are division-heavy.
    let mut rx_ring = vec![0.0; period];
    let mut ry_ring = vec![0.0; period];
    for i in 1..period {
        let (rx, ry) = (ret(x, i), ret(y, i));
        rx_ring[i] = rx;
        ry_ring[i] = ry;
        sx += rx;
        sy += ry;
        sxx += rx * rx;
        sxy += rx * ry;
    }
    for i in period..n {
        let (rx, ry) = (ret(x, i), ret(y, i));
        rx_ring[i % period] = rx;
        ry_ring[i % period] = ry;
        sx += rx;
        sy += ry;
        sxx += rx * rx;
        sxy += rx * ry;
        let denom = pf * sxx - sx * sx;
        out.set(i, if denom.abs() < 1e-14 {
            0.0
        } else {
            (pf * sxy - sx * sy) / denom
        });
        let leaving = i + 1 - period;
        let (tx, ty) = (rx_ring[leaving % period], ry_ring[leaving % period]);
        sx -= tx;
        sy -= ty;
        sxx -= tx * tx;
        sxy -= tx * ty;
    }
    if [sx, sy, sxx, sxy].iter().all(|v| v.is_finite()) {
        return out.finish();
    }
    // A non-finite return entered: window-local careful path (see `crate::guard`).
    // Returns exist for bars `1..n`; term `j` is bar `j + 1`'s.
    let mut out = crate::buf::OutBuf::warmup(n, period);
    window_sums(n - 1, period, Order::AddReadDrop, |j| {
        let (rx, ry) = (ret(x, j + 1), ret(y, j + 1));
        [rx, ry, rx * rx, rx * ry]
    }, |j, [sx, sy, sxx, sxy]| {
        let denom = pf * sxx - sx * sx;
        out.set(j + 1, if denom.abs() < 1e-14 { 0.0 } else { (pf * sxy - sx * sy) / denom });
    });
    out.finish()
}

// ---------------------------------------------------------------------------
// Resumable forms: the running sums of each full computation, in its exact operation
// order, so a resume of one row at a time is bit-identical to recomputing the column.
// ---------------------------------------------------------------------------

/// Which value of the least-squares fit a linear-regression command emits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinregLine {
    /// `b + m·(period-1)` ([`linearreg`]).
    Value,
    /// `m` ([`linearreg_slope`]).
    Slope,
    /// `b` ([`linearreg_intercept`]).
    Intercept,
    /// `atan(m)` in degrees ([`linearreg_angle`]).
    Angle,
    /// `b + m·period` ([`tsf`]).
    Forecast,
}

/// The length of [`linreg_fit`]'s resumable state: `[Σy, Σage·y, non-finite rows in
/// window, rows of the first window seen]`.
const LINREG_STATE: usize = 4;

/// [`linreg_fit`]'s state after the rows of `data` (shared by the whole family).
pub fn linreg_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    state_after(vec![0.0; LINREG_STATE], data.len(), |row, prev, next| {
        linreg_step(data, period, LinregLine::Slope, row, prev, next)
    })
}

/// One row of the linear-regression `line` at `row` — the careful [`linreg_fit_with`]
/// loop's row — from `state` into `next`.
pub fn linreg_step(
    data: &[f64],
    period: usize,
    line: LinregLine,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || row >= data.len() || next.len() != LINREG_STATE {
        return None;
    }
    let [mut sum_y, mut sum_xy, bad, seen] = <[f64; LINREG_STATE]>::try_from(state).ok()?;
    let (mut bad, seen) = (bad as usize, seen as usize);
    let p = period as f64;
    let sum_x = p * (period - 1) as f64 * 0.5;
    let sum_x_sqr = (period * (period - 1) * (2 * period - 1) / 6) as f64;
    let divisor = sum_x * sum_x - p * sum_x_sqr;
    let emit = |sum_y: f64, sum_xy: f64| {
        let m = (p * sum_xy - sum_x * sum_y) / divisor;
        let b = (sum_y - m * sum_x) / p;
        match line {
            LinregLine::Value => b + m * period.saturating_sub(1) as f64,
            LinregLine::Slope => m,
            LinregLine::Intercept => b,
            LinregLine::Angle => m.atan() * (180.0 / std::f64::consts::PI),
            LinregLine::Forecast => b + m * period as f64,
        }
    };
    let own = || {
        data[row + 1 - period..=row]
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(sy, sxy), (k, &y)| (sy + y, sxy + (period - 1 - k) as f64 * y))
    };
    let y = data[row];
    let value = if seen < period {
        // the first window, accumulated in TA-Lib's per-bar order
        if y.is_finite() {
            sum_y += y;
            sum_xy += (period - 1 - seen) as f64 * y;
        } else {
            bad += 1;
        }
        if seen + 1 < period {
            f64::NAN
        } else if bad == 0 {
            emit(sum_y, sum_xy)
        } else {
            let (sy, sxy) = own();
            emit(sy, sxy)
        }
    } else {
        let leaving = data[row - period];
        let came_in = y.is_finite();
        let went_out = leaving.is_finite();
        if !came_in {
            bad += 1;
        }
        if !went_out {
            bad -= 1;
        }
        if went_out {
            sum_xy += sum_y - p * leaving;
        } else {
            sum_xy += sum_y;
        }
        match (came_in, went_out) {
            (true, true) => sum_y += y - leaving,
            (true, false) => sum_y += y,
            (false, true) => sum_y -= leaving,
            (false, false) => {}
        }
        if bad == 0 {
            emit(sum_y, sum_xy)
        } else {
            let (sy, sxy) = own();
            emit(sy, sxy)
        }
    };
    next.copy_from_slice(&[sum_y, sum_xy, bad as f64, (seen + 1).min(period) as f64]);
    Some(value)
}

/// [`rolling_var`]'s state after the rows of `data`: `[Σx, Σx², non-finite rows]`.
pub fn var_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    state_after(vec![0.0; Slide::<2>::STATE], data.len(), |row, prev, next| {
        var_step(data, period, None, row, prev, next)
    })
}

/// One row of [`var`] (or, with `Some(nbdev)`, [`stddev`]) at `row`, from `state` into
/// `next` — the careful [`rolling_var_with`] loop's row.
pub fn var_step(
    data: &[f64],
    period: usize,
    stddev_nbdev: Option<f64>,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || row >= data.len() || next.len() != Slide::<2>::STATE {
        return None;
    }
    let p = period as f64;
    let mut slide = Slide::<2>::from_state(state)?;
    let variance = match slide.row(Order::AddReadDrop, period, row, |j| {
        let x = data[j];
        ([x, x * x], x.is_finite())
    }) {
        Window::Partial => f64::NAN,
        Window::Sums([total1, total2]) => {
            let mean1 = total1 / p;
            let mean2 = total2 / p;
            mean2 - mean1 * mean1
        }
        Window::Bad => {
            let (s1, s2) = data[row + 1 - period..=row]
                .iter()
                .fold((0.0, 0.0), |(s1, s2), &x| (s1 + x, s2 + x * x));
            let mean1 = s1 / p;
            s2 / p - mean1 * mean1
        }
    };
    slide.write_state(next);
    Some(match stddev_nbdev {
        None => variance,
        Some(_) if variance.is_nan() => f64::NAN,
        Some(nbdev) if variance > 0.0 => variance.sqrt() * nbdev,
        Some(_) => 0.0,
    })
}

/// [`correl`]'s state after the rows of `x` / `y`: `[Σx, Σy, Σx², Σy², Σxy, non-finite rows]`.
pub fn correl_state(x: &[f64], y: &[f64], period: usize) -> Option<Vec<f64>> {
    state_after(vec![0.0; Slide::<5>::STATE], x.len(), |row, prev, next| {
        correl_step(x, y, period, row, prev, next)
    })
}

/// One row of [`correl`] at `row`, from `state` into `next` (drop the trailing row, then
/// add the new one — TA-Lib's order, on every path).
pub fn correl_step(
    x: &[f64],
    y: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || row >= x.len() || next.len() != Slide::<5>::STATE {
        return None;
    }
    let inv_pf = 1.0 / period as f64;
    let value = |[sx, sy, sx2, sy2, sxy]: [f64; 5]| {
        let denom = (sx2 - sx * sx * inv_pf) * (sy2 - sy * sy * inv_pf);
        if denom < 1e-14 {
            0.0
        } else {
            (sxy - sx * sy * inv_pf) / denom.sqrt()
        }
    };
    let term = |i: usize| {
        let (xi, yi) = (x[i], y[i]);
        [xi, yi, xi * xi, yi * yi, xi * yi]
    };
    let mut slide = Slide::<5>::from_state(state)?;
    let out = match slide.row(Order::DropAdd, period, row, |j| all_finite(term(j))) {
        Window::Partial => f64::NAN,
        Window::Sums(s) => value(s),
        Window::Bad => value(own_sums(row + 1 - period..=row, term)),
    };
    slide.write_state(next);
    Some(out)
}

/// [`beta`]'s state after the rows of `x` / `y`: the sums of the one-bar returns.
pub fn beta_state(x: &[f64], y: &[f64], period: usize) -> Option<Vec<f64>> {
    state_after(vec![0.0; Slide::<4>::STATE], x.len(), |row, prev, next| {
        beta_step(x, y, period, row, prev, next)
    })
}

/// One row of [`beta`] at `row` (its return is term `row - 1`), from `state` into `next`.
pub fn beta_step(
    x: &[f64],
    y: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || row >= x.len() || next.len() != Slide::<4>::STATE {
        return None;
    }
    if row == 0 {
        next.copy_from_slice(state);
        return Some(f64::NAN);
    }
    let ret = |arr: &[f64], i: usize| {
        let prev = arr[i - 1];
        if prev.abs() < 1e-14 { 0.0 } else { (arr[i] - prev) / prev }
    };
    let pf = period as f64;
    let term = |j: usize| {
        let (rx, ry) = (ret(x, j + 1), ret(y, j + 1));
        [rx, ry, rx * rx, rx * ry]
    };
    let value = |[sx, sy, sxx, sxy]: [f64; 4]| {
        let denom = pf * sxx - sx * sx;
        if denom.abs() < 1e-14 { 0.0 } else { (pf * sxy - sx * sy) / denom }
    };
    let mut slide = Slide::<4>::from_state(state)?;
    let j = row - 1;
    let out = match slide.row(Order::AddReadDrop, period, j, |k| all_finite(term(k))) {
        Window::Partial => f64::NAN,
        Window::Sums(s) => value(s),
        Window::Bad => value(own_sums(j + 1 - period..=j, term)),
    };
    slide.write_state(next);
    Some(out)
}
