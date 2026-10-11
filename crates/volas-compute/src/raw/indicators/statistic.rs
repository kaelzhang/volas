use crate::accum::{anchors_at, Anchors, Csum};
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
fn linreg_fit(data: &[f64], period: usize, origin: usize) -> (Vec<f64>, Vec<f64>) {
    let (m, b, finite) = linreg_fit_with::<Unguarded>(data, period, origin);
    if finite {
        (m, b)
    } else {
        let (m, b, _) = linreg_fit_with::<NonFiniteRows>(data, period, origin);
        (m, b)
    }
}

/// The sliding fit under a row guard (row 0 at absolute position `origin`); also returns
/// whether both running sums stayed finite. A window holding a non-finite row is fitted
/// directly on its own rows.
fn linreg_fit_with<G: RowGuard>(data: &[f64], period: usize, origin: usize) -> (Vec<f64>, Vec<f64>, bool) {
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
    // leaving`. This turns the rolling fit from O(n·period) to O(n). Both sums are
    // compensated, and the weighted products enter and leave exactly
    // ([`Csum::add_prod`]), so a point leaves `Σ age·y` exactly as it entered it — the
    // fit of a window is its own however long the slide; the sums re-anchor at absolute
    // positions (see `crate::accum`). Shared by linearreg / slope / intercept / angle /
    // tsf.
    let mut guard = G::default();
    let mut anchors = Anchors::from(origin + period - 1, period);
    let mut finite = true;
    let mut sum_y = Csum::default();
    let mut sum_xy = Csum::default();
    let mut reanchor = |guard: &G, sum_y: &mut Csum, sum_xy: &mut Csum, today: usize| {
        finite &= sum_y.value().is_finite() && sum_xy.value().is_finite();
        let rows = data[today + 1 - period..=today].iter().copied();
        (*sum_y, *sum_xy) = linreg_sums(rows, period, |y| guard.counts(y.is_finite()));
    };
    for i in (0..period).rev() {
        let y = data[period - 1 - i];
        if guard.enter(y.is_finite()) {
            linreg_admit_first(&mut sum_y, &mut sum_xy, i, y);
        }
    }
    if anchors.tick() {
        reanchor(&guard, &mut sum_y, &mut sum_xy, period - 1);
    }
    if guard.clean() {
        emit(period - 1, sum_y.value(), sum_xy.value());
    } else {
        let (sy, sxy) = window_sums(period - 1);
        emit(period - 1, sy, sxy);
    }
    for today in period..n {
        let leaving = data[today - period];
        let entering = data[today];
        let came_in = guard.enter(entering.is_finite());
        let went_out = guard.leave(leaving.is_finite());
        linreg_slide(&mut sum_y, &mut sum_xy, p, came_in.then_some(entering), went_out.then_some(leaving));
        if anchors.tick() {
            reanchor(&guard, &mut sum_y, &mut sum_xy, today);
        }
        if guard.clean() {
            emit(today, sum_y.value(), sum_xy.value());
        } else {
            let (sy, sxy) = window_sums(today);
            emit(today, sy, sxy);
        }
    }
    let finite = finite && sum_y.value().is_finite() && sum_xy.value().is_finite();
    (slope.finish(), intercept.finish(), finite)
}

/// The fit sums `(Σy, Σ age·y)` of one window, its rows oldest first (age `period-1`
/// down to 0), those `admit` keeps — the seed of a first window, and every
/// re-anchoring.
fn linreg_sums(rows: impl Iterator<Item = f64>, period: usize, admit: impl Fn(f64) -> bool) -> (Csum, Csum) {
    let (mut sum_y, mut sum_xy) = (Csum::default(), Csum::default());
    for (k, y) in rows.enumerate() {
        if admit(y) {
            linreg_admit_first(&mut sum_y, &mut sum_xy, period - 1 - k, y);
        }
    }
    (sum_y, sum_xy)
}

/// Admit a point of age `age` into the first window's `(Σy, Σ age·y)`.
#[inline(always)]
fn linreg_admit_first(sum_y: &mut Csum, sum_xy: &mut Csum, age: usize, y: f64) {
    sum_y.add(y);
    sum_xy.add_prod(age as f64, y);
}

/// Slide `(Σy, Σ age·y)` one bar: every retained point ages by one (`Σ age·y` gains
/// the old `Σy`), the leaving point (if it was in the sums) drops its `period·y`, then
/// `Σy` takes the entering point and drops the leaving one.
#[inline(always)]
fn linreg_slide(sum_y: &mut Csum, sum_xy: &mut Csum, p: f64, entering: Option<f64>, leaving: Option<f64>) {
    sum_xy.add_sum(sum_y);
    if let Some(l) = leaving {
        sum_xy.sub_prod(p, l);
    }
    if let Some(e) = entering {
        sum_y.add(e);
    }
    if let Some(l) = leaving {
        sum_y.sub(l);
    }
}

/// Linear regression value at the current bar: `b + m·(period-1)` (TA-Lib LINEARREG).
pub fn linearreg(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let (m, b) = linreg_fit(data, period, origin);
    let x = period.saturating_sub(1) as f64;
    b.iter().zip(&m).map(|(b, m)| b + m * x).collect()
}

/// Linear regression slope `m` (TA-Lib LINEARREG_SLOPE).
pub fn linearreg_slope(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    linreg_fit(data, period, origin).0
}

/// Linear regression intercept `b` (TA-Lib LINEARREG_INTERCEPT).
pub fn linearreg_intercept(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    linreg_fit(data, period, origin).1
}

/// Linear regression angle in degrees: `atan(m)·180/π` (TA-Lib LINEARREG_ANGLE).
pub fn linearreg_angle(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let deg = 180.0 / std::f64::consts::PI;
    linreg_fit(data, period, origin)
        .0
        .iter()
        .map(|m| m.atan() * deg)
        .collect()
}

/// Time series forecast — regression value one bar ahead: `b + m·period` (TA-Lib TSF).
pub fn tsf(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let (m, b) = linreg_fit(data, period, origin);
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
fn rolling_var(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let (out, finite) = rolling_var_with::<Unguarded>(data, period, origin);
    if finite {
        out
    } else {
        rolling_var_with::<NonFiniteRows>(data, period, origin).0
    }
}

/// The sliding variance under a row guard (row 0 at absolute position `origin`); also
/// returns whether both running sums stayed finite.
fn rolling_var_with<G: RowGuard>(data: &[f64], period: usize, origin: usize) -> (Vec<f64>, bool) {
    let n = data.len();
    if period == 0 || period > n {
        return (vec![f64::NAN; n], true);
    }
    let p = period as f64;
    let mut guard = G::default();
    let mut total1 = Csum::default(); // Σx over the window
    let mut total2 = Csum::default(); // Σx² over the window
    for &x in &data[..period - 1] {
        if guard.enter(x.is_finite()) {
            total1.add(x);
            total2.add(x * x);
        }
    }
    let mut trailing = 0;
    let mut anchors = Anchors::from(origin + period - 1, period);
    let mut finite = true;
    let mut out = crate::buf::OutBuf::warmup(n, period - 1);
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for i in (period - 1)..n {
        let x = data[i];
        if guard.enter(x.is_finite()) {
            total1.add(x);
            total2.add(x * x);
        }
        if anchors.tick() {
            finite &= total1.value().is_finite() && total2.value().is_finite();
            let window = || data[i + 1 - period..=i].iter().copied().filter(|x| guard.counts(x.is_finite()));
            total1 = Csum::of(window());
            total2 = Csum::of(window().map(|x| x * x));
        }
        let value = if guard.clean() {
            var_of_sums(&total1, &total2, p)
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
            total1.sub(old);
            total2.sub(old * old);
        }
        out.set(i, value);
    }
    (out.finish(), finite && total1.value().is_finite() && total2.value().is_finite())
}

/// The population variance `mean(x²) − mean(x)²` of a window from its exact sums — a
/// flat window's is exactly zero.
#[inline(always)]
fn var_of_sums(total1: &Csum, total2: &Csum, p: f64) -> f64 {
    let mean1 = total1.mean(p);
    total2.mean(p) - mean1 * mean1
}

/// Rolling population variance over `period` (TA-Lib VAR). Lookback period-1.
/// (TA-Lib's `nbdev` parameter is a no-op on the variance; it is dropped here.)
pub fn var(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    rolling_var(data, period, origin)
}

/// Rolling standard deviation: `nbdev · sqrt(variance)` over `period` (TA-Lib STDDEV).
/// A non-positive variance (fp cancellation, or a perfectly flat window) yields 0,
/// matching TA-Lib. Lookback period-1.
pub fn stddev(data: &[f64], period: usize, nbdev: f64, origin: usize) -> Vec<f64> {
    rolling_var(data, period, origin)
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
/// yields 0 (TA-Lib's guard). Row 0 is at absolute position `origin`. Lookback
/// `period-1`.
pub fn correl(x: &[f64], y: &[f64], period: usize, origin: usize) -> Vec<f64> {
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
    let mut s = [Csum::default(); 5]; // Σx, Σy, Σx², Σy², Σxy
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.ptr();
    let terms = |xi: f64, yi: f64| [xi, yi, xi * xi, yi * yi, xi * yi];
    for i in 0..period {
        // `period <= n` and the hot loop below keeps both indices in range.
        let (xi, yi) = unsafe { (*x_ptr.add(i), *y_ptr.add(i)) };
        for (a, t) in s.iter_mut().zip(terms(xi, yi)) {
            a.add(t);
        }
    }
    let read = |s: &[Csum; 5]| s.map(|a| a.value());
    let mut anchors = Anchors::from(origin + period - 1, period);
    let mut clean = true;
    let mut reanchor = |s: &mut [Csum; 5], end: usize| {
        clean &= read(s).iter().all(|v| v.is_finite());
        *s = [Csum::default(); 5];
        for j in end + 1 - period..=end {
            for (a, t) in s.iter_mut().zip(terms(x[j], y[j])) {
                a.add(t);
            }
        }
    };
    let value = |sx: f64, sy: f64, sx2: f64, sy2: f64, sxy: f64| {
        let denom = (sx2 - sx * sx * inv_pf) * (sy2 - sy * sy * inv_pf);
        if denom < 1e-14 {
            0.0
        } else {
            (sxy - sx * sy * inv_pf) / denom.sqrt()
        }
    };
    if anchors.tick() {
        reanchor(&mut s, period - 1);
    }
    unsafe {
        let [sx, sy, sx2, sy2, sxy] = read(&s);
        *out_ptr.add(period - 1) = value(sx, sy, sx2, sy2, sxy);
    }
    let mut trailing = 0;
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for i in period..n {
        // Drop the trailing values, then add the new ones (TA-Lib's order).
        let (tx, ty) = unsafe { (*x_ptr.add(trailing), *y_ptr.add(trailing)) };
        trailing += 1;
        for (a, t) in s.iter_mut().zip(terms(tx, ty)) {
            a.sub(t);
        }
        let (xi, yi) = unsafe { (*x_ptr.add(i), *y_ptr.add(i)) };
        for (a, t) in s.iter_mut().zip(terms(xi, yi)) {
            a.add(t);
        }
        if anchors.tick() {
            reanchor(&mut s, i);
        }
        unsafe {
            let [sx, sy, sx2, sy2, sxy] = read(&s);
            *out_ptr.add(i) = value(sx, sy, sx2, sy2, sxy);
        }
    }
    if clean && read(&s).iter().all(|v| v.is_finite()) {
        return out.finish();
    }
    // A non-finite row entered: window-local careful path (see `crate::guard`).
    let mut out = crate::buf::OutBuf::warmup(n, period - 1);
    window_sums(n, period, Order::DropAdd, origin, |i| {
        let (xi, yi) = (x[i], y[i]);
        [xi, yi, xi * xi, yi * yi, xi * yi]
    }, |i, [sx, sy, sx2, sy2, sxy]| out.set(i, value(sx, sy, sx2, sy2, sxy)));
    out.finish()
}

/// Rolling beta of `x` against `y` over `period` (TA-Lib BETA): the slope
/// `(n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)` of the two series' one-bar **returns**
/// `(p[i]−p[i-1])/p[i-1]` (a zero prior price gives a 0 return; a ~0 denominator
/// gives 0). Row 0 is at absolute position `origin` (return `j`, of bar `j + 1`, at
/// `origin + j`). First value at index `period` (lookback = period).
pub fn beta(x: &[f64], y: &[f64], period: usize, origin: usize) -> Vec<f64> {
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
    let mut s = [Csum::default(); 4]; // Σrx, Σry, Σrx², Σrx·ry
    let terms = |rx: f64, ry: f64| [rx, ry, rx * rx, rx * ry];
    // Keep the return values that are currently inside the rolling window. TA-Lib
    // conceptually drops a return after emitting a row; storing it once avoids
    // recalculating the departing high/low returns, which are division-heavy.
    let mut rx_ring = vec![0.0; period];
    let mut ry_ring = vec![0.0; period];
    for i in 1..period {
        let (rx, ry) = (ret(x, i), ret(y, i));
        rx_ring[i] = rx;
        ry_ring[i] = ry;
        for (a, t) in s.iter_mut().zip(terms(rx, ry)) {
            a.add(t);
        }
    }
    let mut anchors = Anchors::from(origin + period - 1, period);
    let mut clean = true;
    for i in period..n {
        let (rx, ry) = (ret(x, i), ret(y, i));
        rx_ring[i % period] = rx;
        ry_ring[i % period] = ry;
        for (a, t) in s.iter_mut().zip(terms(rx, ry)) {
            a.add(t);
        }
        if anchors.tick() {
            clean &= s.iter().all(|a| a.value().is_finite());
            s = [Csum::default(); 4];
            for bar in i + 1 - period..=i {
                for (a, t) in s.iter_mut().zip(terms(rx_ring[bar % period], ry_ring[bar % period])) {
                    a.add(t);
                }
            }
        }
        let [sx, sy, sxx, sxy] = s.map(|a| a.value());
        let denom = pf * sxx - sx * sx;
        out.set(i, if denom.abs() < 1e-14 {
            0.0
        } else {
            (pf * sxy - sx * sy) / denom
        });
        let leaving = i + 1 - period;
        let (tx, ty) = (rx_ring[leaving % period], ry_ring[leaving % period]);
        for (a, t) in s.iter_mut().zip(terms(tx, ty)) {
            a.sub(t);
        }
    }
    if clean && s.iter().all(|a| a.value().is_finite()) {
        return out.finish();
    }
    // A non-finite return entered: window-local careful path (see `crate::guard`).
    // Returns exist for bars `1..n`; term `j` is bar `j + 1`'s.
    let mut out = crate::buf::OutBuf::warmup(n, period);
    window_sums(n - 1, period, Order::AddReadDrop, origin, |j| {
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

/// The length of [`linreg_fit`]'s resumable state: `[Σy, Σage·y]` (each a [`Csum`]),
/// the non-finite rows in the window, the rows of the first window seen, the absolute
/// position of the next row (for re-anchoring).
const LINREG_STATE: usize = 2 * Csum::STATE + 3;

/// [`linreg_fit`]'s state after the rows of `data` (shared by the whole family).
pub fn linreg_state(data: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    let mut init = vec![0.0; LINREG_STATE];
    init[6] = origin as f64;
    state_after(init, data.len(), |row, prev, next| {
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
    if state.len() != LINREG_STATE {
        return None;
    }
    let mut sum_y = Csum::from_state(&state[..2])?;
    let mut sum_xy = Csum::from_state(&state[2..4])?;
    let (mut bad, seen, pos) = (state[4] as usize, state[5] as usize, state[6] as usize);
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
    let anchor = seen + 1 >= period && anchors_at(pos, period);
    let reanchor = |sum_y: &mut Csum, sum_xy: &mut Csum| {
        let rows = data[row + 1 - period..=row].iter().copied();
        (*sum_y, *sum_xy) = linreg_sums(rows, period, f64::is_finite);
    };
    let value = if seen < period {
        // the first window, accumulated in TA-Lib's per-bar order
        if y.is_finite() {
            linreg_admit_first(&mut sum_y, &mut sum_xy, period - 1 - seen, y);
        } else {
            bad += 1;
        }
        if anchor {
            reanchor(&mut sum_y, &mut sum_xy);
        }
        if seen + 1 < period {
            f64::NAN
        } else if bad == 0 {
            emit(sum_y.value(), sum_xy.value())
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
        linreg_slide(&mut sum_y, &mut sum_xy, p, came_in.then_some(y), went_out.then_some(leaving));
        if anchor {
            reanchor(&mut sum_y, &mut sum_xy);
        }
        if bad == 0 {
            emit(sum_y.value(), sum_xy.value())
        } else {
            let (sy, sxy) = own();
            emit(sy, sxy)
        }
    };
    sum_y.write_state(&mut next[..2]);
    sum_xy.write_state(&mut next[2..4]);
    next[4] = bad as f64;
    next[5] = (seen + 1).min(period) as f64;
    next[6] = (pos + 1) as f64;
    Some(value)
}

/// [`rolling_var`]'s state after the rows of `data`: the [`Slide`] of `[Σx, Σx²]`.
pub fn var_state(data: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<2>::initial(origin), data.len(), |row, prev, next| {
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
        Window::Sums([total1, total2]) => var_of_sums(&total1, &total2, p),
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

/// [`correl`]'s state after the rows of `x` / `y`: the [`Slide`] of `[Σx, Σy, Σx², Σy², Σxy]`.
pub fn correl_state(x: &[f64], y: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<5>::initial(origin), x.len(), |row, prev, next| {
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
        Window::Sums(s) => value(s.map(|c| c.value())),
        Window::Bad => value(own_sums(row + 1 - period..=row, term)),
    };
    slide.write_state(next);
    Some(out)
}

/// [`beta`]'s state after the rows of `x` / `y`: the sums of the one-bar returns.
pub fn beta_state(x: &[f64], y: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<4>::initial(origin), x.len(), |row, prev, next| {
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
        Window::Sums(s) => value(s.map(|c| c.value())),
        Window::Bad => value(own_sums(j + 1 - period..=j, term)),
    };
    slide.write_state(next);
    Some(out)
}
