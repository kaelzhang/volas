//! The `rolling` / `expanding` / `ewm` window API — thin bindings over the
//! [`volas_compute::window`] kernels (pandas window semantics: NaN = missing,
//! skipped; `min_periods` gates each cell).
//!
//! This surface exists for pandas COMPATIBILITY (research / labeling code that
//! moves over verbatim). It is deliberately not the recommended path: a window
//! result is a plain Series — it does not join the directive cache and is NOT
//! incrementally refreshed by `append` / `fulfill`. In a live trading system
//! use the directive forms (`df['ma:20']`, `df['median:30']`, …), which are
//! the same kernels with caching and O(lookback) per-bar refresh.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_compute::window as w;
use volas_core::{Column, Series};

use crate::series::PySeries;

/// Resolve pandas's four exclusive EW decay spellings to the smoothing factor.
pub(crate) fn resolve_alpha(
    com: Option<f64>,
    span: Option<f64>,
    halflife: Option<f64>,
    alpha: Option<f64>,
) -> PyResult<f64> {
    let given = [com.is_some(), span.is_some(), halflife.is_some(), alpha.is_some()];
    if given.iter().filter(|&&g| g).count() != 1 {
        return Err(PyValueError::new_err(
            "ewm requires exactly one of com / span / halflife / alpha",
        ));
    }
    // A NaN or infinite decay degenerates to alpha=0 / NaN — an all-NA or
    // frozen no-signal column. alpha=0 is already rejected, so the other
    // spellings must reject their equivalent too (same-guard symmetry, E8).
    if let Some(c) = com {
        if !(c >= 0.0 && c.is_finite()) {
            return Err(PyValueError::new_err("ewm com must be finite and >= 0"));
        }
        return Ok(1.0 / (1.0 + c));
    }
    if let Some(s) = span {
        if !(s >= 1.0 && s.is_finite()) {
            return Err(PyValueError::new_err("ewm span must be finite and >= 1"));
        }
        return Ok(2.0 / (s + 1.0));
    }
    if let Some(h) = halflife {
        if !(h > 0.0 && h.is_finite()) {
            return Err(PyValueError::new_err("ewm halflife must be finite and > 0"));
        }
        return Ok(1.0 - (-(2.0f64.ln()) / h).exp());
    }
    let a = alpha.expect("exactly one spelling is set");
    if !(a > 0.0 && a <= 1.0) {
        return Err(PyValueError::new_err("ewm alpha must satisfy 0 < alpha <= 1"));
    }
    Ok(a)
}

fn validate_interpolation(interpolation: &str) -> PyResult<()> {
    match interpolation {
        "linear" | "lower" | "higher" | "nearest" | "midpoint" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "interpolation must be one of linear/lower/higher/nearest/midpoint, got {other:?}"
        ))),
    }
}

/// R-1: a negative ddof must be a clean ValueError, never pyo3's
/// unsigned-conversion OverflowError leak.
pub(crate) fn validate_ddof(ddof: i64) -> PyResult<usize> {
    if ddof < 0 {
        return Err(PyValueError::new_err("ddof must be >= 0"));
    }
    Ok(ddof as usize)
}

fn validate_rank_method(method: &str) -> PyResult<()> {
    match method {
        "average" | "min" | "max" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "rank method must be 'average', 'min' or 'max', got {other:?}"
        ))),
    }
}

/// The shared engine: a window specification over one series. `window ==
/// usize::MAX` is the expanding window (`center` is rolling-only, like pandas).
pub(crate) struct WinSpec {
    pub(crate) window: usize,
    pub(crate) min_periods: usize,
    pub(crate) center: bool,
}

impl WinSpec {
    /// pandas `center=True` labels each window at its center: the window for
    /// label `i` is `[i - window/2, i + (window - window/2 - 1)]`, CLIPPED at
    /// both edges (a clipped window still emits once `min_periods` present
    /// values remain). Implemented by padding `fwd` trailing NaNs — padding is
    /// "missing", so the trailing kernels produce exactly the clipped
    /// centered semantics — then re-slicing the labels. The kernel receives the
    /// trailing window length to use (see [`geometry`](Self::geometry)).
    fn run(&self, data: &[f64], k: impl FnOnce(&[f64], usize) -> Vec<f64>) -> Vec<f64> {
        let n = data.len();
        let (window, fwd) = self.geometry(n);
        if fwd == 0 {
            return k(data, window);
        }
        let mut padded = data.to_vec();
        padded.extend(std::iter::repeat_n(f64::NAN, fwd));
        let out = k(&padded, window);
        out[fwd..fwd + n].to_vec()
    }

    /// The trailing `(window, fwd)` that realizes this spec over `n` rows: the
    /// window as given and no forward reach when trailing; when centered, the
    /// backward reach `window / 2` and forward reach `window - window / 2 - 1`,
    /// each **bounded by `n`** — the window is clipped at both edges, so reaching
    /// past the data is the same as reaching exactly to it. The padding (and every
    /// kernel buffer) is therefore bounded by the data, never by the window.
    fn geometry(&self, n: usize) -> (usize, usize) {
        if !self.center {
            return (self.window, 0);
        }
        let fwd = self.window - self.window / 2 - 1;
        let (back, fwd) = ((self.window / 2).min(n), fwd.min(n));
        (back + fwd + 1, fwd)
    }

    /// A float64 aggregation result carrying the source's name + index.
    fn f64_series(&self, s: &Series, out: Vec<f64>) -> PySeries {
        PySeries {
            inner: Series::new(s.name.clone(), Column::f64(out), Arc::clone(&s.index)),
        }
    }

    /// An int64 count-like result (`count` / `nunique`): NaN cells become NA.
    fn i64_series(&self, s: &Series, out: Vec<f64>) -> PySeries {
        // a count is a small non-negative integer, exact in float64
        let cells = out.iter().map(|&x| (!x.is_nan()).then_some(x as i64));
        PySeries {
            inner: Series::new(s.name.clone(), Column::i64_from_options(cells), Arc::clone(&s.index)),
        }
    }

    /// The window input for the float64 aggregation `op`: an int64 past 2^53
    /// raises (contract #26) instead of aggregating rounded values.
    fn data(&self, s: &Series, op: &str) -> PyResult<Vec<f64>> {
        s.data.to_f64_domain(op).map_err(crate::pyerr)
    }

    pub(crate) fn count(&self, s: &Series) -> PySeries {
        // count gates on window COVERAGE (see the kernel), so it handles
        // `center` itself instead of going through the NaN-padding `run`.
        let out = w::count(&s.data.to_f64_vec(), self.window, self.min_periods, self.center);
        self.i64_series(s, out)
    }
    pub(crate) fn nunique(&self, s: &Series) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "nunique")?, |d, win| w::nunique(d, win, self.min_periods));
        Ok(self.i64_series(s, out))
    }
    pub(crate) fn sum(&self, s: &Series) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "sum")?, |d, win| w::sum(d, win, self.min_periods));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn mean(&self, s: &Series) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "mean")?, |d, win| w::mean(d, win, self.min_periods));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn median(&self, s: &Series) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "median")?, |d, win| w::median(d, win, self.min_periods));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn min(&self, s: &Series) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "min")?, |d, win| w::min(d, win, self.min_periods));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn max(&self, s: &Series) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "max")?, |d, win| w::max(d, win, self.min_periods));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn var(&self, s: &Series, ddof: usize) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "var")?, |d_, win| w::var(d_, win, self.min_periods, ddof));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn std(&self, s: &Series, ddof: usize) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "std")?, |d_, win| w::std(d_, win, self.min_periods, ddof));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn sem(&self, s: &Series, ddof: usize) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "sem")?, |d_, win| w::sem(d_, win, self.min_periods, ddof));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn skew(&self, s: &Series) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "skew")?, |d, win| w::skew(d, win, self.min_periods));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn kurt(&self, s: &Series) -> PyResult<PySeries> {
        let out = self.run(&self.data(s, "kurt")?, |d, win| w::kurt(d, win, self.min_periods));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn quantile(&self, s: &Series, q: f64, interpolation: &str) -> PyResult<PySeries> {
        validate_interpolation(interpolation)?;
        if !(0.0..=1.0).contains(&q) {
            return Err(PyValueError::new_err("quantile q must be in [0, 1]"));
        }
        let out = self.run(&self.data(s, "quantile")?, |d, win| {
            w::quantile(d, win, self.min_periods, q, interpolation)
        });
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn rank(&self, s: &Series, method: &str, ascending: bool, pct: bool) -> PyResult<PySeries> {
        validate_rank_method(method)?;
        // a centered window ending in the padding ranks its last DATA row
        let n = s.len();
        let out = self.run(&self.data(s, "rank")?, |d, win| {
            w::rank_padded(d, win, self.min_periods, method, ascending, pct, n)
        });
        Ok(self.f64_series(s, out))
    }
    /// `first` / `last` preserve the source dtype: gather the edge positions.
    pub(crate) fn edge(&self, s: &Series, last: bool) -> PySeries {
        let n = s.len();
        let mut valid: Vec<bool> = (0..n).map(|i| s.data.is_valid(i)).collect();
        // the same pad-and-relabel trick as `run` (padding is never valid, so it
        // is never selected as an edge position).
        let (window, fwd) = self.geometry(n);
        valid.extend(std::iter::repeat_n(false, fwd));
        let pos = w::edge_positions(&valid, window, self.min_periods, last)[fwd..fwd + n].to_vec();
        PySeries {
            inner: Series::new(
                s.name.clone(),
                s.data.take_optional(&pos),
                Arc::clone(&s.index),
            ),
        }
    }
    pub(crate) fn corr(&self, s: &Series, other: &PySeries) -> PyResult<PySeries> {
        let (x, y) = pair(s, other)?;
        let out = self.run2(&x, &y, |a, b, win| w::corr(a, b, win, self.min_periods));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn cov(&self, s: &Series, other: &PySeries, ddof: usize) -> PyResult<PySeries> {
        let (x, y) = pair(s, other)?;
        let out = self.run2(&x, &y, |a, b, win| w::cov(a, b, win, self.min_periods, ddof));
        Ok(self.f64_series(s, out))
    }

    /// Two-series variant of `run` (corr / cov under `center=True`).
    fn run2(
        &self,
        x: &[f64],
        y: &[f64],
        k: impl FnOnce(&[f64], &[f64], usize) -> Vec<f64>,
    ) -> Vec<f64> {
        let n = x.len();
        let (window, fwd) = self.geometry(n);
        if fwd == 0 {
            return k(x, y, window);
        }
        let (mut xp, mut yp) = (x.to_vec(), y.to_vec());
        xp.extend(std::iter::repeat_n(f64::NAN, fwd));
        yp.extend(std::iter::repeat_n(f64::NAN, fwd));
        k(&xp, &yp, window)[fwd..fwd + n].to_vec()
    }
}

/// `s.rolling(window)` — fixed-window aggregation, pandas semantics (NaN
/// skipped, `min_periods` gates each cell, `center=True` labels the window at
/// its center). Compatibility surface — prefer directives in live systems.
#[pyclass(name = "Rolling")]
pub struct PyRolling {
    pub(crate) series: Series,
    pub(crate) spec: WinSpec,
}

/// `s.expanding()` — the cumulative (all-history) window aggregator.
#[pyclass(name = "Expanding")]
pub struct PyExpanding {
    pub(crate) series: Series,
    pub(crate) spec: WinSpec,
}

/// `s.ewm(...)` — the exponentially-weighted aggregator (pandas `ewm`):
/// exactly one of `com` / `span` / `halflife` / `alpha`, both `adjust` modes,
/// both `ignore_na` modes.
#[pyclass(name = "Ewm")]
pub struct PyEwm {
    pub(crate) series: Series,
    pub(crate) alpha: f64,
    pub(crate) adjust: bool,
    pub(crate) ignore_na: bool,
    pub(crate) min_periods: usize,
}

#[pymethods]
impl PyRolling {
    /// Present-value count per window (pandas `count`); int64, gaps are NA.
    fn count(&self) -> PySeries {
        self.spec.count(&self.series)
    }
    /// Distinct present values per window (pandas `nunique`); int64.
    fn nunique(&self) -> PyResult<PySeries> {
        self.spec.nunique(&self.series)
    }
    /// Rolling sum (pandas `sum`).
    fn sum(&self) -> PyResult<PySeries> {
        self.spec.sum(&self.series)
    }
    /// Rolling mean (pandas `mean`).
    fn mean(&self) -> PyResult<PySeries> {
        self.spec.mean(&self.series)
    }
    /// Rolling median (pandas `median`).
    fn median(&self) -> PyResult<PySeries> {
        self.spec.median(&self.series)
    }
    /// Rolling minimum (pandas `min`).
    fn min(&self) -> PyResult<PySeries> {
        self.spec.min(&self.series)
    }
    /// Rolling maximum (pandas `max`).
    fn max(&self) -> PyResult<PySeries> {
        self.spec.max(&self.series)
    }
    /// Rolling sample variance (pandas `var(ddof=1)`).
    #[pyo3(signature = (ddof = 1))]
    fn var(&self, ddof: i64) -> PyResult<PySeries> {
        self.spec.var(&self.series, validate_ddof(ddof)?)
    }
    /// Rolling sample standard deviation (pandas `std(ddof=1)`).
    #[pyo3(signature = (ddof = 1))]
    fn std(&self, ddof: i64) -> PyResult<PySeries> {
        self.spec.std(&self.series, validate_ddof(ddof)?)
    }
    /// Standard error of the mean (pandas `sem`).
    #[pyo3(signature = (ddof = 1))]
    fn sem(&self, ddof: i64) -> PyResult<PySeries> {
        self.spec.sem(&self.series, validate_ddof(ddof)?)
    }
    /// Bias-corrected sample skewness (pandas `skew`; >= 3 present values).
    fn skew(&self) -> PyResult<PySeries> {
        self.spec.skew(&self.series)
    }
    /// Bias-corrected excess kurtosis (pandas `kurt`; >= 4 present values).
    fn kurt(&self) -> PyResult<PySeries> {
        self.spec.kurt(&self.series)
    }
    /// Rolling quantile (pandas `quantile(q, interpolation='linear')`).
    #[pyo3(signature = (q, interpolation = "linear"))]
    fn quantile(&self, q: f64, interpolation: &str) -> PyResult<PySeries> {
        self.spec.quantile(&self.series, q, interpolation)
    }
    /// Rank of each row's value within its own window (pandas `rank`).
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PySeries> {
        self.spec.rank(&self.series, method, ascending, pct)
    }
    /// First present value per window (pandas `first`), dtype-preserving.
    fn first(&self) -> PySeries {
        self.spec.edge(&self.series, false)
    }
    /// Last present value per window (pandas `last`), dtype-preserving.
    fn last(&self) -> PySeries {
        self.spec.edge(&self.series, true)
    }
    /// Rolling Pearson correlation against another series (pandas `corr`).
    fn corr(&self, other: &PySeries) -> PyResult<PySeries> {
        self.spec.corr(&self.series, other)
    }
    /// Rolling sample covariance against another series (pandas `cov`).
    #[pyo3(signature = (other, ddof = 1))]
    fn cov(&self, other: &PySeries, ddof: i64) -> PyResult<PySeries> {
        self.spec.cov(&self.series, other, validate_ddof(ddof)?)
    }
}

#[pymethods]
impl PyExpanding {
    /// Expanding present-value count; int64.
    fn count(&self) -> PySeries {
        self.spec.count(&self.series)
    }
    /// Expanding distinct present values; int64.
    fn nunique(&self) -> PyResult<PySeries> {
        self.spec.nunique(&self.series)
    }
    /// Expanding sum.
    fn sum(&self) -> PyResult<PySeries> {
        self.spec.sum(&self.series)
    }
    /// Expanding mean.
    fn mean(&self) -> PyResult<PySeries> {
        self.spec.mean(&self.series)
    }
    /// Expanding median.
    fn median(&self) -> PyResult<PySeries> {
        self.spec.median(&self.series)
    }
    /// Expanding minimum.
    fn min(&self) -> PyResult<PySeries> {
        self.spec.min(&self.series)
    }
    /// Expanding maximum.
    fn max(&self) -> PyResult<PySeries> {
        self.spec.max(&self.series)
    }
    /// Expanding sample variance (`ddof=1`).
    #[pyo3(signature = (ddof = 1))]
    fn var(&self, ddof: i64) -> PyResult<PySeries> {
        self.spec.var(&self.series, validate_ddof(ddof)?)
    }
    /// Expanding sample standard deviation (`ddof=1`).
    #[pyo3(signature = (ddof = 1))]
    fn std(&self, ddof: i64) -> PyResult<PySeries> {
        self.spec.std(&self.series, validate_ddof(ddof)?)
    }
    /// Expanding standard error of the mean.
    #[pyo3(signature = (ddof = 1))]
    fn sem(&self, ddof: i64) -> PyResult<PySeries> {
        self.spec.sem(&self.series, validate_ddof(ddof)?)
    }
    /// Expanding skewness.
    fn skew(&self) -> PyResult<PySeries> {
        self.spec.skew(&self.series)
    }
    /// Expanding excess kurtosis.
    fn kurt(&self) -> PyResult<PySeries> {
        self.spec.kurt(&self.series)
    }
    /// Expanding quantile.
    #[pyo3(signature = (q, interpolation = "linear"))]
    fn quantile(&self, q: f64, interpolation: &str) -> PyResult<PySeries> {
        self.spec.quantile(&self.series, q, interpolation)
    }
    /// Expanding rank of each row's value within all history so far.
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PySeries> {
        self.spec.rank(&self.series, method, ascending, pct)
    }
    /// First present value so far, dtype-preserving.
    fn first(&self) -> PySeries {
        self.spec.edge(&self.series, false)
    }
    /// Last present value so far, dtype-preserving.
    fn last(&self) -> PySeries {
        self.spec.edge(&self.series, true)
    }
    /// Expanding Pearson correlation against another series.
    fn corr(&self, other: &PySeries) -> PyResult<PySeries> {
        self.spec.corr(&self.series, other)
    }
    /// Expanding sample covariance against another series.
    #[pyo3(signature = (other, ddof = 1))]
    fn cov(&self, other: &PySeries, ddof: i64) -> PyResult<PySeries> {
        self.spec.cov(&self.series, other, validate_ddof(ddof)?)
    }
}

impl PyEwm {
    /// The ewm input: float64 math, so an int64 past 2^53 raises (contract #26).
    fn data(&self) -> PyResult<Vec<f64>> {
        self.series.data.to_f64_domain("ewm").map_err(crate::pyerr)
    }

    fn wrap(&self, out: Vec<f64>) -> PySeries {
        PySeries {
            inner: Series::new(
                self.series.name.clone(),
                Column::f64(out),
                Arc::clone(&self.series.index),
            ),
        }
    }
}

/// The two operands of a pairwise window statistic (`corr` / `cov`), paired by
/// position under the shared alignment rule ([`crate::require_aligned`]): equal
/// labels or an error, never a silent positional pairing of different labels.
fn pair(s: &Series, other: &PySeries) -> PyResult<(Vec<f64>, Vec<f64>)> {
    crate::require_aligned(&s.index, &other.inner.index)?;
    other.inner.data.require_numeric().map_err(crate::pyerr)?;
    let f = |c: &volas_core::Column| c.to_f64_domain("a pairwise window statistic").map_err(crate::pyerr);
    Ok((f(&s.data)?, f(&other.inner.data)?))
}

#[pymethods]
impl PyEwm {
    /// Exponentially-weighted mean (pandas `ewm(...).mean()`).
    pub(crate) fn mean(&self) -> PyResult<PySeries> {
        Ok(self.wrap(w::ewm_mean(
            &self.data()?,
            self.alpha,
            self.adjust,
            self.ignore_na,
            self.min_periods,
        )))
    }
    /// Exponentially-weighted (un-normalized) sum; like pandas, only defined
    /// for `adjust=True`.
    pub(crate) fn sum(&self) -> PyResult<PySeries> {
        if !self.adjust {
            return Err(PyValueError::new_err(
                "ewm sum is not implemented with adjust=False (pandas raises too)",
            ));
        }
        Ok(self.wrap(w::ewm_sum(
            &self.data()?,
            self.alpha,
            self.ignore_na,
            self.min_periods,
        )))
    }
    /// Exponentially-weighted variance (pandas `var(bias=False)`).
    #[pyo3(signature = (bias = false))]
    pub(crate) fn var(&self, bias: bool) -> PyResult<PySeries> {
        Ok(self.wrap(w::ewm_var(
            &self.data()?,
            self.alpha,
            self.adjust,
            self.ignore_na,
            bias,
            self.min_periods.max(1),
        )))
    }
    /// Exponentially-weighted standard deviation.
    #[pyo3(signature = (bias = false))]
    pub(crate) fn std(&self, bias: bool) -> PyResult<PySeries> {
        let v = w::ewm_var(
            &self.data()?,
            self.alpha,
            self.adjust,
            self.ignore_na,
            bias,
            self.min_periods.max(1),
        );
        Ok(self.wrap(v.into_iter().map(f64::sqrt).collect()))
    }
    /// Exponentially-weighted covariance against another series.
    #[pyo3(signature = (other, bias = false))]
    fn cov(&self, other: &PySeries, bias: bool) -> PyResult<PySeries> {
        let (x, y) = pair(&self.series, other)?;
        Ok(self.wrap(w::ewm_cov(
            &x,
            &y,
            self.alpha,
            self.adjust,
            self.ignore_na,
            bias,
            self.min_periods.max(1),
        )))
    }
    /// Exponentially-weighted correlation against another series.
    fn corr(&self, other: &PySeries) -> PyResult<PySeries> {
        let (x, y) = pair(&self.series, other)?;
        Ok(self.wrap(w::ewm_corr(
            &x,
            &y,
            self.alpha,
            self.adjust,
            self.ignore_na,
            self.min_periods.max(1),
        )))
    }
}
