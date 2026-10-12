use crate::accum::{anchors_at, comoment, square_terms, two_prod, Accumulator, Anchors, Csum};
use crate::guard::{all_finite, state_after, Order, Slide, Window};
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
    let (sum_x, divisor) = linreg_x_sums(period);
    // Single-write (D2): NaN warm-up `[0, period-1)`, each valid row written once.
    let mut slope = crate::buf::OutBuf::warmup(n, period - 1);
    let mut intercept = crate::buf::OutBuf::warmup(n, period - 1);
    let mut emit = |today: usize, sum_y: &Csum, sum_xy: &Csum| {
        let (m, b) = linreg_fit_of(p, sum_x, divisor, sum_y, sum_xy);
        slope.set(today, m);
        intercept.set(today, b);
    };
    // A window's own `(Σy, Σ age·y)` — the direct fit for a window holding a bad row.
    let window_sums = |today: usize| linreg_sums(data[today + 1 - period..=today].iter().copied(), period, |_| true);
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
    // Re-sum at an anchor or when a sum went inexact. The fast pass stops once a
    // non-finite row has poisoned it (the careful pass recomputes every row); the
    // careful pass, whose sums hold finite rows only, always re-sums.
    let mut reanchor = |guard: &G, sum_y: &mut Csum, sum_xy: &mut Csum, today: usize| {
        finite &= sum_y.value().is_finite() && sum_xy.value().is_finite();
        if finite || !guard.counts(false) {
            let rows = data[today + 1 - period..=today].iter().copied();
            (*sum_y, *sum_xy) = linreg_sums(rows, period, |y| guard.counts(y.is_finite()));
        }
    };
    let exact = |a: &Csum, b: &Csum| a.is_exact() && b.is_exact();
    for i in (0..period).rev() {
        let y = data[period - 1 - i];
        if guard.enter(y.is_finite()) {
            linreg_admit_first(&mut sum_y, &mut sum_xy, i, y);
        }
    }
    if anchors.tick() || !exact(&sum_y, &sum_xy) {
        reanchor(&guard, &mut sum_y, &mut sum_xy, period - 1);
    }
    if guard.clean() {
        emit(period - 1, &sum_y, &sum_xy);
    } else {
        let (sy, sxy) = window_sums(period - 1);
        emit(period - 1, &sy, &sxy);
    }
    for today in period..n {
        let leaving = data[today - period];
        let entering = data[today];
        let came_in = guard.enter(entering.is_finite());
        let went_out = guard.leave(leaving.is_finite());
        linreg_slide(&mut sum_y, &mut sum_xy, p, came_in.then_some(entering), went_out.then_some(leaving));
        let anchor = anchors.tick();
        if anchor || !exact(&sum_y, &sum_xy) {
            reanchor(&guard, &mut sum_y, &mut sum_xy, today);
        }
        if guard.clean() {
            emit(today, &sum_y, &sum_xy);
        } else {
            let (sy, sxy) = window_sums(today);
            emit(today, &sy, &sxy);
        }
    }
    let finite = finite && sum_y.value().is_finite() && sum_xy.value().is_finite();
    (slope.finish(), intercept.finish(), finite)
}

/// The fit sums `(Σy, Σ age·y)` of one window, its rows oldest first (age `period-1`
/// down to 0), those `admit` keeps — the seed of a first window, and every
/// re-anchoring.
fn linreg_sums(rows: impl Iterator<Item = f64> + Clone, period: usize, admit: impl Fn(f64) -> bool) -> (Csum, Csum) {
    let [sum_y, sum_xy] = Csum::settle(linreg_sums_in(rows.clone(), period, &admit), || {
        linreg_sums_in(rows, period, &admit)
    });
    (sum_y, sum_xy)
}

/// [`linreg_sums`] over an [`Accumulator`].
fn linreg_sums_in<A: Accumulator>(rows: impl Iterator<Item = f64>, period: usize, admit: &impl Fn(f64) -> bool) -> [A; 2] {
    let (mut sum_y, mut sum_xy) = (A::default(), A::default());
    for (k, y) in rows.enumerate() {
        if admit(y) {
            linreg_admit_first(&mut sum_y, &mut sum_xy, period - 1 - k, y);
        }
    }
    [sum_y, sum_xy]
}

/// The fit `(m, b)` of a window of `p` rows from its exact sums `Σy` and `Σ age·y`
/// (`sum_x`, `divisor` from [`linreg_x_sums`]): the slope's numerator `p·Σ age·y −
/// Σage·Σy` by [`comoment`] — accurate however the raw moments cancel, exactly zero
/// for a flat window.
#[inline]
fn linreg_fit_of(p: f64, sum_x: f64, divisor: f64, sum_y: &Csum, sum_xy: &Csum) -> (f64, f64) {
    let m = comoment(p, (sum_x, 0.0), sum_y.words(), std::slice::from_ref(sum_xy)) / divisor;
    (m, (sum_y.value() - m * sum_x) / p)
}

/// The fit's closed-form `x` sums over ages `0..period`: `(Σx, Σx·Σx − period·Σx²)`.
/// `Σx² = (period−1)·period·(2·period−1)/6` is formed in `u128` — exact for every
/// `period` — and rounded to `f64` once (in `usize` it overflows from period
/// 2,097,153).
fn linreg_x_sums(period: usize) -> (f64, f64) {
    let p = period as f64;
    let sum_x = p * (period - 1) as f64 * 0.5;
    let q = period as u128;
    let sum_x_sqr = (q * (q - 1) * (2 * q - 1) / 6) as f64;
    (sum_x, sum_x * sum_x - p * sum_x_sqr)
}

/// Admit a point of age `age` into the first window's `(Σy, Σ age·y)`.
#[inline(always)]
fn linreg_admit_first<A: Accumulator>(sum_y: &mut A, sum_xy: &mut A, age: usize, y: f64) {
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
    let mut sums = [Csum::default(); 3]; // [Σx, Σx²] over the window (see `square_terms`)
    let enter = |sums: &mut [Csum; 3], x: f64| {
        for (s, t) in sums.iter_mut().zip(square_terms(x)) {
            s.add(t);
        }
    };
    for &x in &data[..period - 1] {
        if guard.enter(x.is_finite()) {
            enter(&mut sums, x);
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
            enter(&mut sums, x);
        }
        let anchor = anchors.tick();
        if anchor || !sums.iter().all(Csum::is_exact) {
            // the fast pass stops re-summing once poisoned (see `linreg_fit_with`)
            finite &= sums.iter().all(|s| s.value().is_finite());
            if finite || !guard.counts(false) {
                let window = data[i + 1 - period..=i].iter().filter(|x| guard.counts(x.is_finite()));
                sums = Csum::of_rows(window.map(|&x| square_terms(x)));
            }
        }
        let value = if guard.clean() { var_of_sums(&sums, p) } else { var_of_window(&data[i + 1 - period..=i], p) };
        let old = data[trailing];
        trailing += 1;
        if guard.leave(old.is_finite()) {
            for (s, t) in sums.iter_mut().zip(square_terms(old)) {
                s.sub(t);
            }
        }
        out.set(i, value);
    }
    (out.finish(), finite && sums.iter().all(|s| s.value().is_finite()))
}

/// The population variance of a window from its exact sums `[Σx, Σx²]` (see
/// [`square_terms`]): `(n·Σx² − (Σx)²) / n²` by [`comoment`] — accurate however the raw
/// moments cancel, exactly zero for a flat window.
#[inline(always)]
fn var_of_sums(sums: &[Csum; 3], p: f64) -> f64 {
    let s = sums[0].words();
    comoment(p, s, s, &sums[1..]) / (p * p)
}

/// The population variance of a window holding a non-finite row, on its own rows (the
/// IEEE value of that window: NaN).
fn var_of_window(rows: &[f64], p: f64) -> f64 {
    let (s1, s2) = rows.iter().fold((0.0, 0.0), |(s1, s2), &x| (s1 + x, s2 + x * x));
    let mean1 = s1 / p;
    s2 / p - mean1 * mean1
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

/// Rolling Pearson correlation of `x` and `y` over `period`: `Σ(x−x̄)(y−ȳ) /
/// sqrt(Σ(x−x̄)²·Σ(y−ȳ)²)` over each window, from its exact sums ([`comoment`] of exact
/// `Σx`, `Σy`, `Σx²`, `Σy²`, `Σxy`), so `|r| ≤ 1` and a flat window is not rounding
/// noise. A window where `x` or `y` is constant (zero variance, tested exactly) yields
/// 0 — TA-Lib CORREL's result there, which TA-Lib reaches through an absolute `1e-14`
/// threshold on the raw-moment denominator instead (a deliberate divergence: TA-Lib's
/// raw moments also give `|r| > 1`, and `±1` for some flat windows). Row 0 is at
/// absolute position `origin`. Lookback `period-1`.
pub fn correl(x: &[f64], y: &[f64], period: usize, origin: usize) -> Vec<f64> {
    if !super::same_len(&[x, y]) {
        return Vec::new();
    }
    let n = x.len();
    if period == 0 || period > n {
        return vec![f64::NAN; n];
    }
    let p = period as f64;
    // Fast path: drop the trailing row, then add the new one (TA-Lib's order) — the
    // careful slide's arithmetic ([`correl_row`]) on clean data.
    let mut out = crate::buf::OutBuf::warmup(n, period - 1);
    let mut s = [Csum::default(); 8];
    let mut anchors = Anchors::from(origin + period - 1, period);
    let mut clean = true;
    for i in 0..n {
        if i >= period {
            for (a, t) in s.iter_mut().zip(correl_terms(x[i - period], y[i - period])) {
                a.sub(t);
            }
        }
        for (a, t) in s.iter_mut().zip(correl_terms(x[i], y[i])) {
            a.add(t);
        }
        if i + 1 < period {
            continue;
        }
        let anchor = anchors.tick();
        if anchor || !s.iter().all(Csum::is_exact) {
            clean &= s.iter().all(|a| a.value().is_finite());
            if clean {
                s = Csum::of_rows((i + 1 - period..=i).map(|j| correl_terms(x[j], y[j])));
            }
        }
        out.set(i, correl_of_sums(&s, p));
    }
    if clean && s.iter().all(|a| a.value().is_finite()) {
        return out.finish();
    }
    // A non-finite row entered: window-local careful path (see `crate::guard`).
    let mut slide = Slide::<8>::at(origin);
    (0..n).map(|i| correl_row(&mut slide, x, y, period, i)).collect()
}

/// The per-row terms of [`correl`]'s exact sums: `x`, `y`, then `x²`, `y²` and `x·y`
/// each split into its rounded value and its rounding error.
#[inline(always)]
fn correl_terms(x: f64, y: f64) -> [f64; 8] {
    let (xx, xxe) = two_prod(x, x);
    let (yy, yye) = two_prod(y, y);
    let (xy, xye) = two_prod(x, y);
    [x, y, xx, xxe, yy, yye, xy, xye]
}

/// Pearson's r of a window of `p` rows from its exact sums ([`correl_terms`]); 0 when
/// either variance is exactly zero. Clamped to `[-1, 1]` as a last resort: the
/// co-moments are each within an ulp or two of exact, so Cauchy–Schwarz can only be
/// overshot by that rounding.
#[inline]
fn correl_of_sums(s: &[Csum; 8], p: f64) -> f64 {
    let (x, y) = (s[0].words(), s[1].words());
    let dx = comoment(p, x, x, &s[2..4]);
    let dy = comoment(p, y, y, &s[4..6]);
    if dx <= 0.0 || dy <= 0.0 {
        return 0.0;
    }
    let dxy = comoment(p, x, y, &s[6..8]);
    (dxy / (dx.sqrt() * dy.sqrt())).clamp(-1.0, 1.0)
}

/// One row of [`correl`] at `i`, resumed from `slide` (drop, then add) — the careful
/// slide, which the fast loop follows on clean windows. A window holding a non-finite
/// row takes the value of its own sums.
fn correl_row(slide: &mut Slide<8>, x: &[f64], y: &[f64], period: usize, i: usize) -> f64 {
    let term = |j: usize| correl_terms(x[j], y[j]);
    match slide.row(Order::DropAdd, period, i, |j| all_finite(term(j))) {
        Window::Partial => f64::NAN,
        Window::Sums(s) => correl_of_sums(&s, period as f64),
        Window::Bad => correl_of_sums(&Csum::of_rows((i + 1 - period..=i).map(term)), period as f64),
    }
}

/// Rolling beta of `x` against `y` over `period` (TA-Lib BETA): the slope
/// `Σ(rx−r̄x)(ry−r̄y) / Σ(rx−r̄x)²` of the two series' one-bar **returns**
/// `(p[i]−p[i-1])/p[i-1]` (a zero prior price gives a 0 return), from exact sums
/// ([`comoment`]). A window whose `x` returns are constant (zero variance, tested
/// exactly) gives 0 — TA-Lib's result, reached there through an absolute `1e-14`
/// threshold instead. Row 0 is at absolute position `origin` (return `j`, of bar
/// `j + 1`, at `origin + j`). First value at index `period` (lookback = period).
pub fn beta(x: &[f64], y: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let n = x.len();
    if period == 0 || period + 1 > n {
        return vec![f64::NAN; n];
    }
    let p = period as f64;
    // Fast path: add the new return, read, drop the one the next window loses — the
    // careful slide's arithmetic ([`beta_row`]) on clean data. The returns in the
    // window are kept, so a leaving one is not divided out again.
    let mut out = crate::buf::OutBuf::warmup(n, period);
    let mut ring = vec![[0.0; 6]; period];
    let mut s = [Csum::default(); 6];
    let mut anchors = Anchors::from(origin + period - 1, period);
    let mut clean = true;
    for j in 0..n - 1 {
        let t = beta_terms(x, y, j);
        ring[j % period] = t;
        for (a, v) in s.iter_mut().zip(t) {
            a.add(v);
        }
        if j + 1 < period {
            continue;
        }
        let anchor = anchors.tick();
        if anchor || !s.iter().all(Csum::is_exact) {
            clean &= s.iter().all(|a| a.value().is_finite());
            if clean {
                s = Csum::of_rows((j + 1 - period..=j).map(|k| ring[k % period]));
            }
        }
        out.set(j + 1, beta_of_sums(&s, p));
        for (a, v) in s.iter_mut().zip(ring[(j + 1 - period) % period]) {
            a.sub(v);
        }
    }
    if clean && s.iter().all(|a| a.value().is_finite()) {
        return out.finish();
    }
    // A non-finite return entered: window-local careful path (see `crate::guard`).
    let mut out = crate::buf::OutBuf::warmup(n, period);
    let mut slide = Slide::<6>::at(origin);
    for j in 0..n - 1 {
        let v = beta_row(&mut slide, x, y, period, j);
        if j + 1 >= period {
            out.set(j + 1, v);
        }
    }
    out.finish()
}

/// The one-bar return of `arr` at bar `i` (`>= 1`); 0 after a (near-)zero price.
#[inline(always)]
fn beta_return(arr: &[f64], i: usize) -> f64 {
    let prev = arr[i - 1];
    if prev.abs() < 1e-14 { 0.0 } else { (arr[i] - prev) / prev }
}

/// The terms of [`beta`]'s exact sums for return `j` (bar `j + 1`): `rx`, `ry`, then
/// `rx²` and `rx·ry` each split into its rounded value and its rounding error.
#[inline(always)]
fn beta_terms(x: &[f64], y: &[f64], j: usize) -> [f64; 6] {
    let (rx, ry) = (beta_return(x, j + 1), beta_return(y, j + 1));
    let (xx, xxe) = two_prod(rx, rx);
    let (xy, xye) = two_prod(rx, ry);
    [rx, ry, xx, xxe, xy, xye]
}

/// Beta of a window of `p` returns from its exact sums ([`beta_terms`]); 0 when the
/// `x` returns' variance is exactly zero.
#[inline]
fn beta_of_sums(s: &[Csum; 6], p: f64) -> f64 {
    let x = s[0].words();
    let dx = comoment(p, x, x, &s[2..4]);
    if dx <= 0.0 {
        return 0.0;
    }
    comoment(p, x, s[1].words(), &s[4..6]) / dx
}

/// One row of [`beta`] over return `j`, resumed from `slide` (add, read, drop).
fn beta_row(slide: &mut Slide<6>, x: &[f64], y: &[f64], period: usize, j: usize) -> f64 {
    let term = |k: usize| beta_terms(x, y, k);
    match slide.row(Order::AddReadDrop, period, j, |k| all_finite(term(k))) {
        Window::Partial => f64::NAN,
        Window::Sums(s) => beta_of_sums(&s, period as f64),
        Window::Bad => beta_of_sums(&Csum::of_rows((j + 1 - period..=j).map(term)), period as f64),
    }
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
    let (sum_x, divisor) = linreg_x_sums(period);
    let emit = |sum_y: &Csum, sum_xy: &Csum| {
        let (m, b) = linreg_fit_of(p, sum_x, divisor, sum_y, sum_xy);
        match line {
            LinregLine::Value => b + m * period.saturating_sub(1) as f64,
            LinregLine::Slope => m,
            LinregLine::Intercept => b,
            LinregLine::Angle => m.atan() * (180.0 / std::f64::consts::PI),
            LinregLine::Forecast => b + m * period as f64,
        }
    };
    let own = || linreg_sums(data[row + 1 - period..=row].iter().copied(), period, |_| true);
    let y = data[row];
    let scheduled = anchors_at(pos, period);
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
        if seen + 1 >= period && (scheduled || !(sum_y.is_exact() && sum_xy.is_exact())) {
            reanchor(&mut sum_y, &mut sum_xy);
        }
        if seen + 1 < period {
            f64::NAN
        } else if bad == 0 {
            emit(&sum_y, &sum_xy)
        } else {
            let (sy, sxy) = own();
            emit(&sy, &sxy)
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
        if scheduled || !(sum_y.is_exact() && sum_xy.is_exact()) {
            reanchor(&mut sum_y, &mut sum_xy);
        }
        if bad == 0 {
            emit(&sum_y, &sum_xy)
        } else {
            let (sy, sxy) = own();
            emit(&sy, &sxy)
        }
    };
    sum_y.write_state(&mut next[..2]);
    sum_xy.write_state(&mut next[2..4]);
    next[4] = bad as f64;
    next[5] = (seen + 1).min(period) as f64;
    next[6] = (pos + 1) as f64;
    Some(value)
}

/// [`rolling_var`]'s state after the rows of `data`: the [`Slide`] of `[Σx, Σx²]` (see
/// [`square_terms`]).
pub fn var_state(data: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<3>::initial(origin), data.len(), |row, prev, next| {
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
    if period == 0 || row >= data.len() || next.len() != Slide::<3>::STATE {
        return None;
    }
    let p = period as f64;
    let mut slide = Slide::<3>::from_state(state)?;
    let variance = match slide.row(Order::AddReadDrop, period, row, |j| {
        let x = data[j];
        (square_terms(x), x.is_finite())
    }) {
        Window::Partial => f64::NAN,
        Window::Sums(sums) => var_of_sums(&sums, p),
        Window::Bad => var_of_window(&data[row + 1 - period..=row], p),
    };
    slide.write_state(next);
    Some(match stddev_nbdev {
        None => variance,
        Some(_) if variance.is_nan() => f64::NAN,
        Some(nbdev) if variance > 0.0 => variance.sqrt() * nbdev,
        Some(_) => 0.0,
    })
}

/// [`correl`]'s state after the rows of `x` / `y`: the [`Slide`] of its exact sums
/// ([`correl_terms`]).
pub fn correl_state(x: &[f64], y: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<8>::initial(origin), x.len(), |row, prev, next| {
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
    if period == 0 || row >= x.len() || next.len() != Slide::<8>::STATE {
        return None;
    }
    let mut slide = Slide::<8>::from_state(state)?;
    let out = correl_row(&mut slide, x, y, period, row);
    slide.write_state(next);
    Some(out)
}

/// [`beta`]'s state after the rows of `x` / `y`: the [`Slide`] of its exact sums
/// ([`beta_terms`]).
pub fn beta_state(x: &[f64], y: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<6>::initial(origin), x.len(), |row, prev, next| {
        beta_step(x, y, period, row, prev, next)
    })
}

/// One row of [`beta`] at `row`, from `state` into `next`.
pub fn beta_step(
    x: &[f64],
    y: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || row >= x.len() || next.len() != Slide::<6>::STATE {
        return None;
    }
    if row == 0 {
        next.copy_from_slice(state);
        return Some(f64::NAN);
    }
    let mut slide = Slide::<6>::from_state(state)?;
    let out = beta_row(&mut slide, x, y, period, row - 1);
    slide.write_state(next);
    Some(out)
}
