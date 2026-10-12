use crate::accum::{centred, deviation, shift_row, spread, Accumulator, Anchors, Csum};
use crate::raw::kernels::shifted_terms;
use crate::guard::{state_after, Order, Slide, Window};
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
///
/// The sums are `Σy` and `Σ age·y` (the newest bar has age 0), both exact — the weighted
/// terms are exact products ([`Csum::add_prod`]) — so they are first-moment slides,
/// re-summed only where one went inexact. They slide in O(1): when the window advances
/// one bar every retained point's age rises by one, so `Σ age·y` gains `Σy` and loses
/// the departing point's full `period·y` — `sum_xy += sum_y − period·leaving` (old
/// `sum_y`), then `sum_y += entering − leaving`. The fit reads them shifted by the
/// window's anchor-row value `K` ([`linreg_fit_of`]), so the slope's numerator carries
/// no cancellation between the level and the trend. Shared by linearreg / slope /
/// intercept / angle / tsf.
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
    let mut guard = G::default();
    let mut anchors = Anchors::from(origin, period);
    let mut k = data[0];
    let mut finite = true;
    let (mut sum_d, mut sum_ad) = (Csum::default(), Csum::default());
    for i in 0..n {
        if anchors.tick() {
            k = data[i];
        }
        let entering = data[i];
        let came_in = guard.enter(entering.is_finite());
        let leaving = (i >= period).then(|| data[i - period]);
        let went_out = leaving.is_some_and(|l| guard.leave(l.is_finite()));
        linreg_slide(&mut sum_d, &mut sum_ad, p, came_in.then_some(entering), leaving.filter(|_| went_out));
        let full = i + 1 >= period;
        let rows = &data[(i + 1).saturating_sub(period)..=i];
        if full && !(sum_d.is_exact() && sum_ad.is_exact()) {
            // The fast pass stops once a non-finite row has poisoned it (the careful
            // pass recomputes every row); the careful pass always re-sums.
            finite &= sum_d.value().is_finite() && sum_ad.value().is_finite();
            if finite || !guard.counts(false) {
                (sum_d, sum_ad) = linreg_sums(rows, |y| guard.counts(y.is_finite()));
            }
        }
        if !full {
            continue;
        }
        let (m, b) = if guard.clean() {
            linreg_fit_of(p, sum_x, divisor, k, &sum_d, &sum_ad)
        } else {
            let (sd, sad) = linreg_sums(rows, |_| true);
            linreg_fit_of(p, sum_x, divisor, k, &sd, &sad)
        };
        slope.set(i, m);
        intercept.set(i, b);
    }
    let finite = finite && sum_d.value().is_finite() && sum_ad.value().is_finite();
    (slope.finish(), intercept.finish(), finite)
}

/// The fit sums `(Σy, Σ age·y)` of one window's `rows` (oldest first, the newest at age
/// 0), of the rows `admit` keeps — every re-sum, and a window holding a bad row (out of
/// the slide's loop).
#[cold]
#[inline(never)]
fn linreg_sums(rows: &[f64], admit: impl Fn(f64) -> bool) -> (Csum, Csum) {
    let [sum_y, sum_xy] = Csum::settle(linreg_sums_in(rows, &admit), || linreg_sums_in(rows, &admit));
    (sum_y, sum_xy)
}

/// [`linreg_sums`] over an [`Accumulator`].
fn linreg_sums_in<A: Accumulator>(rows: &[f64], admit: &impl Fn(f64) -> bool) -> [A; 2] {
    let (mut sum_y, mut sum_xy) = (A::default(), A::default());
    for (j, &y) in rows.iter().enumerate() {
        if admit(y) {
            sum_y.add(y);
            sum_xy.add_prod((rows.len() - 1 - j) as f64, y);
        }
    }
    [sum_y, sum_xy]
}

/// The fit `(m, b)` of a window of `p` rows from its exact sums `Σy`, `Σ age·y`
/// (`sum_x`, `divisor` from [`linreg_x_sums`]), read on the data shifted by `k`, the
/// window's anchor-row value ([`deviation`]): `m = (p·Σ age·(y − k) − Σage·Σ(y − k)) /
/// divisor`, `b = k + (Σ(y − k) − m·Σage) / p` — within `90·2⁻⁵³·R/(p + 1) + 2⁻⁵³·|m|`
/// of the exact slope for a window of range `R`, exactly 0 for a flat window. A window
/// holding a non-finite value reads its IEEE value.
#[inline(always)]
fn linreg_fit_of(p: f64, sum_x: f64, divisor: f64, k: f64, sum_y: &Csum, sum_xy: &Csum) -> (f64, f64) {
    let total = sum_y.value();
    if !total.is_finite() {
        let m = (p * sum_xy.value() - sum_x * total) / divisor;
        return (m, (total - m * sum_x) / p);
    }
    let dy = deviation(sum_y, p, k);
    let m = (p * deviation(sum_xy, sum_x, k) - sum_x * dy) / divisor;
    (m, k + (dy - m * sum_x) / p)
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

/// Rolling population variance over `period`: `(Σd² − (Σd)²/p) / p` on the data shifted
/// by the window's anchor-row value (`d = x − K`, see [`crate::accum`]'s second moments),
/// O(n) via sliding exact sums of `x` and `d²` in TA-Lib's operation order (add the
/// current row, read, drop the trailing one) — within `(36p + 2)·2⁻⁵³` of the exact
/// variance however the window's level dwarfs its spread, never negative, exactly zero
/// for a flat window (TA-Lib's one-pass `Σx²/p − (Σx/p)²` instead cancels to rounding
/// noise, which can be negative). NaN during warm-up. Shared by `var` and
/// `stddev`. Window-local: a non-finite row affects only the windows holding it (a
/// window holding one is NaN — see [`crate::guard`]).
fn rolling_var(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let (out, finite) = rolling_var_with::<Unguarded>(data, period, origin);
    if finite {
        out
    } else {
        rolling_var_with::<NonFiniteRows>(data, period, origin).0
    }
}

/// The sliding variance under a row guard (row 0 at absolute position `origin`); also
/// returns whether both running sums stayed finite. The sums are `[Σx, Σ(x − K)²]` with
/// `K` the value at the window's anchor row ([`shift_row`]): the latest row whose
/// absolute position is a multiple of `period` (the first row before one). They are
/// rebuilt at every anchor — where `K` changes — and whenever one went inexact.
fn rolling_var_with<G: RowGuard>(data: &[f64], period: usize, origin: usize) -> (Vec<f64>, bool) {
    let n = data.len();
    if period == 0 || period > n {
        return (vec![f64::NAN; n], true);
    }
    let p = period as f64;
    let mut guard = G::default();
    let mut sums = [Csum::default(); 2];
    let mut anchors = Anchors::from(origin, period);
    let mut k = data[0];
    let mut finite = true;
    let mut out = crate::buf::OutBuf::warmup(n, period - 1);
    for i in 0..n {
        let x = data[i];
        let anchor = anchors.tick();
        if anchor {
            k = x;
        }
        if guard.enter(x.is_finite()) {
            for (s, t) in sums.iter_mut().zip(shifted_terms(x, k)) {
                s.add(t);
            }
        }
        let full = i + 1 >= period;
        let rows = &data[(i + 1).saturating_sub(period)..=i];
        if anchor || (full && !sums.iter().all(Csum::is_exact)) {
            // the fast pass stops re-summing once poisoned (see `linreg_fit_with`)
            finite &= sums.iter().all(|s| s.value().is_finite());
            if finite || !guard.counts(false) {
                let window = rows.iter().filter(|x| guard.counts(x.is_finite()));
                sums = Csum::of_rows(window.map(|&x| shifted_terms(x, k)));
            }
        }
        if !full {
            continue;
        }
        let value = if guard.clean() { var_of_sums(&sums, k, p) } else { var_of_window(rows, k, p) };
        let old = data[i + 1 - period];
        if guard.leave(old.is_finite()) {
            for (s, t) in sums.iter_mut().zip(shifted_terms(old, k)) {
                s.sub(t);
            }
        }
        out.set(i, value);
    }
    (out.finish(), finite && sums.iter().all(|s| s.value().is_finite()))
}

/// The population variance of a window of `p` rows from its exact sums
/// `[Σx, Σ(x − k)²]` ([`shifted_terms`]): `(Σd² − (Σd)²/p) / p`.
#[inline(always)]
fn var_of_sums([sum, sq]: &[Csum; 2], k: f64, p: f64) -> f64 {
    spread(p, sum, k, sq).1 / p
}

/// The population variance of a window holding a non-finite row, on its own rows: its
/// exact sums, then [`var_of_sums`] (the IEEE value of that window: NaN).
fn var_of_window(rows: &[f64], k: f64, p: f64) -> f64 {
    var_of_sums(&Csum::of_rows(rows.iter().map(|&x| shifted_terms(x, k))), k, p)
}

/// Rolling population variance over `period` (TA-Lib VAR). Lookback period-1.
/// (TA-Lib's `nbdev` parameter is a no-op on the variance; it is dropped here.)
pub fn var(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    rolling_var(data, period, origin)
}

/// Rolling standard deviation: `nbdev · sqrt(variance)` over `period` (TA-Lib STDDEV),
/// the variance of [`rolling_var`]; a flat window (zero variance) yields 0. Lookback
/// period-1.
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
/// sqrt(Σ(x−x̄)²·Σ(y−ȳ)²)` over each window, from exact sums of the data shifted by the
/// window's anchor-row values ([`crate::accum`]'s second moments) — no cancellation
/// between the level and the spread, `|r| ≤ 1`, and a flat window is not rounding
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
    let mut s = [Csum::default(); 5];
    let mut anchors = Anchors::from(origin, period);
    let mut k = (x[0], y[0]);
    let mut clean = true;
    for i in 0..n {
        let anchor = anchors.tick();
        if anchor {
            k = (x[i], y[i]);
        }
        if i >= period {
            for (a, t) in s.iter_mut().zip(correl_terms(x[i - period], y[i - period], k)) {
                a.sub(t);
            }
        }
        for (a, t) in s.iter_mut().zip(correl_terms(x[i], y[i], k)) {
            a.add(t);
        }
        let full = i + 1 >= period;
        if anchor || (full && !s.iter().all(Csum::is_exact)) {
            clean &= s.iter().all(|a| a.value().is_finite());
            if clean {
                let rows = (i + 1).saturating_sub(period)..i + 1;
                s = Csum::of_rows(rows.map(|j| correl_terms(x[j], y[j], k)));
            }
        }
        if full {
            out.set(i, correl_of_sums(&s, k, p));
        }
    }
    if clean && s.iter().all(|a| a.value().is_finite()) {
        return out.finish();
    }
    // A non-finite row entered: window-local careful path (see `crate::guard`).
    let mut slide = Slide::<5>::at(origin);
    (0..n).map(|i| correl_row(&mut slide, x, y, period, i)).collect()
}

/// The per-row terms of [`correl`]'s exact sums under the shift `(kx, ky)`: `x`, `y`,
/// and the shifted `dx²`, `dy²`, `dx·dy`.
#[inline(always)]
fn correl_terms(x: f64, y: f64, (kx, ky): (f64, f64)) -> [f64; 5] {
    let (dx, dy) = (x - kx, y - ky);
    [x, y, dx * dx, dy * dy, dx * dy]
}

/// Pearson's r of a window of `p` rows from its exact sums ([`correl_terms`]); 0 when
/// either variance is exactly zero. Clamped to `[-1, 1]` as a last resort: each centred
/// moment is within `(36p + 2)·2⁻⁵³` of exact relative to `sqrt(Cxx·Cyy)`, so
/// Cauchy–Schwarz can only be overshot by that rounding.
#[inline]
fn correl_of_sums(s: &[Csum; 5], (kx, ky): (f64, f64), p: f64) -> f64 {
    let cxx = centred(p, (&s[0], kx), (&s[0], kx), &s[2]);
    let cyy = centred(p, (&s[1], ky), (&s[1], ky), &s[3]);
    if cxx <= 0.0 || cyy <= 0.0 {
        return 0.0;
    }
    let cxy = centred(p, (&s[0], kx), (&s[1], ky), &s[4]);
    (cxy / (cxx.sqrt() * cyy.sqrt())).clamp(-1.0, 1.0)
}

/// One row of [`correl`] at `i`, resumed from `slide` (drop, then add) — the careful
/// slide, which the fast loop follows on clean windows. A window holding a non-finite
/// row takes the value of its own sums.
fn correl_row(slide: &mut Slide<5>, x: &[f64], y: &[f64], period: usize, i: usize) -> f64 {
    let a = shift_row(i, slide.pos(), period);
    let k = (x[a], y[a]);
    let term = |j: usize| correl_terms(x[j], y[j], k);
    match slide.row_shifted(Order::DropAdd, period, i, |j| (term(j), x[j].is_finite() && y[j].is_finite())) {
        Window::Partial => f64::NAN,
        Window::Sums(s) => correl_of_sums(&s, k, period as f64),
        Window::Bad => correl_of_sums(&Csum::of_rows((i + 1 - period..=i).map(term)), k, period as f64),
    }
}

/// Rolling beta of `x` against `y` over `period` (TA-Lib BETA): the slope
/// `Σ(rx−r̄x)(ry−r̄y) / Σ(rx−r̄x)²` of the two series' one-bar **returns**
/// `(p[i]−p[i-1])/p[i-1]` (a zero prior price gives a 0 return), from exact sums of the
/// returns shifted by the window's anchor-row returns ([`crate::accum`]'s second
/// moments). A window whose `x` returns are constant (zero variance, tested exactly)
/// gives 0 — TA-Lib's result, reached there through an absolute `1e-14` threshold
/// instead. Row 0 is at absolute position `origin` (return `j`, of bar `j + 1`, at
/// `origin + j`). First value at index `period` (lookback = period).
pub fn beta(x: &[f64], y: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let n = x.len();
    if period == 0 || period + 1 > n {
        return vec![f64::NAN; n];
    }
    let p = period as f64;
    // Fast path: add the new return, read, drop the one the next window loses — the
    // careful slide's arithmetic ([`beta_row`]) on clean data. The returns in the window
    // are kept, so a leaving one is not divided out again.
    let mut out = crate::buf::OutBuf::warmup(n, period);
    let mut ring = vec![(0.0, 0.0); period];
    let mut s = [Csum::default(); 4];
    let mut anchors = Anchors::from(origin, period);
    let mut k = beta_returns(x, y, 0);
    let mut clean = true;
    for j in 0..n - 1 {
        let r = beta_returns(x, y, j);
        ring[j % period] = r;
        let anchor = anchors.tick();
        if anchor {
            k = r;
        }
        for (a, v) in s.iter_mut().zip(beta_terms(r, k)) {
            a.add(v);
        }
        let full = j + 1 >= period;
        if anchor || (full && !s.iter().all(Csum::is_exact)) {
            clean &= s.iter().all(|a| a.value().is_finite());
            if clean {
                let rows = (j + 1).saturating_sub(period)..j + 1;
                s = Csum::of_rows(rows.map(|t| beta_terms(ring[t % period], k)));
            }
        }
        if !full {
            continue;
        }
        out.set(j + 1, beta_of_sums(&s, k, p));
        for (a, v) in s.iter_mut().zip(beta_terms(ring[(j + 1 - period) % period], k)) {
            a.sub(v);
        }
    }
    if clean && s.iter().all(|a| a.value().is_finite()) {
        return out.finish();
    }
    // A non-finite return entered: window-local careful path (see `crate::guard`).
    let mut out = crate::buf::OutBuf::warmup(n, period);
    let mut slide = Slide::<4>::at(origin);
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

/// The returns `(rx, ry)` of return `j` (bar `j + 1`).
#[inline(always)]
fn beta_returns(x: &[f64], y: &[f64], j: usize) -> (f64, f64) {
    (beta_return(x, j + 1), beta_return(y, j + 1))
}

/// The terms of [`beta`]'s exact sums for the returns `(rx, ry)` under the shift `(kx,
/// ky)`: `rx`, `ry`, and the shifted `drx²`, `drx·dry`.
#[inline(always)]
fn beta_terms((rx, ry): (f64, f64), (kx, ky): (f64, f64)) -> [f64; 4] {
    let (dx, dy) = (rx - kx, ry - ky);
    [rx, ry, dx * dx, dx * dy]
}

/// Beta of a window of `p` returns from its exact sums ([`beta_terms`]); 0 when the `x`
/// returns' variance is exactly zero.
#[inline]
fn beta_of_sums(s: &[Csum; 4], (kx, ky): (f64, f64), p: f64) -> f64 {
    let cxx = centred(p, (&s[0], kx), (&s[0], kx), &s[2]);
    if cxx <= 0.0 {
        return 0.0;
    }
    centred(p, (&s[0], kx), (&s[1], ky), &s[3]) / cxx
}

/// One row of [`beta`] over return `j`, resumed from `slide` (add, read, drop).
fn beta_row(slide: &mut Slide<4>, x: &[f64], y: &[f64], period: usize, j: usize) -> f64 {
    let k = beta_returns(x, y, shift_row(j, slide.pos(), period));
    let term = |t: usize| beta_terms(beta_returns(x, y, t), k);
    let finite = |t: usize| {
        let (rx, ry) = beta_returns(x, y, t);
        rx.is_finite() && ry.is_finite()
    };
    match slide.row_shifted(Order::AddReadDrop, period, j, |t| (term(t), finite(t))) {
        Window::Partial => f64::NAN,
        Window::Sums(s) => beta_of_sums(&s, k, period as f64),
        Window::Bad => beta_of_sums(&Csum::of_rows((j + 1 - period..=j).map(term)), k, period as f64),
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
    let mut sum_d = Csum::from_state(&state[..2])?;
    let mut sum_ad = Csum::from_state(&state[2..4])?;
    let (mut bad, seen, pos) = (state[4] as usize, state[5] as usize, state[6] as usize);
    let p = period as f64;
    let (sum_x, divisor) = linreg_x_sums(period);
    let k = data[shift_row(row, pos, period)];
    let y = data[row];
    if !y.is_finite() {
        bad += 1;
    }
    let leaving = (seen >= period).then(|| data[row - period]);
    if leaving.is_some_and(|l| !l.is_finite()) {
        bad -= 1;
    }
    linreg_slide(&mut sum_d, &mut sum_ad, p, y.is_finite().then_some(y), leaving.filter(|l| l.is_finite()));
    let full = seen + 1 >= period;
    let rows = &data[(row + 1).saturating_sub(period)..=row];
    if full && !(sum_d.is_exact() && sum_ad.is_exact()) {
        (sum_d, sum_ad) = linreg_sums(rows, f64::is_finite);
    }
    let value = if !full {
        f64::NAN
    } else {
        let (m, b) = if bad == 0 {
            linreg_fit_of(p, sum_x, divisor, k, &sum_d, &sum_ad)
        } else {
            let (sd, sad) = linreg_sums(rows, |_| true);
            linreg_fit_of(p, sum_x, divisor, k, &sd, &sad)
        };
        match line {
            LinregLine::Value => b + m * period.saturating_sub(1) as f64,
            LinregLine::Slope => m,
            LinregLine::Intercept => b,
            LinregLine::Angle => m.atan() * (180.0 / std::f64::consts::PI),
            LinregLine::Forecast => b + m * period as f64,
        }
    };
    sum_d.write_state(&mut next[..2]);
    sum_ad.write_state(&mut next[2..4]);
    next[4] = bad as f64;
    next[5] = (seen + 1).min(period) as f64;
    next[6] = (pos + 1) as f64;
    Some(value)
}

/// [`rolling_var`]'s state after the rows of `data`: the [`Slide`] of `[Σx, Σ(x − K)²]`
/// (see [`rolling_var_with`]).
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
    let k = data[shift_row(row, slide.pos(), period)];
    let variance = match slide.row_shifted(Order::AddReadDrop, period, row, |j| {
        let x = data[j];
        (shifted_terms(x, k), x.is_finite())
    }) {
        Window::Partial => f64::NAN,
        Window::Sums(sums) => var_of_sums(&sums, k, p),
        Window::Bad => var_of_window(&data[row + 1 - period..=row], k, p),
    };
    slide.write_state(next);
    Some(match stddev_nbdev {
        None => variance,
        Some(_) if variance.is_nan() => f64::NAN,
        Some(nbdev) if variance > 0.0 => variance.sqrt() * nbdev,
        Some(_) => 0.0,
    })
}

/// [`correl`]'s state after the rows of `x` / `y`: the [`Slide`] of its exact shifted
/// sums ([`correl_terms`]).
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
    let mut slide = Slide::<5>::from_state(state)?;
    let out = correl_row(&mut slide, x, y, period, row);
    slide.write_state(next);
    Some(out)
}

/// [`beta`]'s state after the rows of `x` / `y`: the [`Slide`] of its exact sums
/// ([`beta_terms`]).
pub fn beta_state(x: &[f64], y: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<4>::initial(origin), x.len(), |row, prev, next| {
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
    if period == 0 || row >= x.len() || next.len() != Slide::<4>::STATE {
        return None;
    }
    if row == 0 {
        next.copy_from_slice(state);
        return Some(f64::NAN);
    }
    let mut slide = Slide::<4>::from_state(state)?;
    let out = beta_row(&mut slide, x, y, period, row - 1);
    slide.write_state(next);
    Some(out)
}
