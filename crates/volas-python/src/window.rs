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
    /// centered semantics — then re-slicing the labels.
    fn run(&self, data: &[f64], k: impl FnOnce(&[f64]) -> Vec<f64>) -> Vec<f64> {
        if !self.center {
            return k(data);
        }
        let n = data.len();
        let fwd = self.window - self.window / 2 - 1;
        let mut padded = data.to_vec();
        padded.extend(std::iter::repeat(f64::NAN).take(fwd));
        let out = k(&padded);
        out[fwd..fwd + n].to_vec()
    }

    /// A float64 aggregation result carrying the source's name + index.
    fn f64_series(&self, s: &Series, out: Vec<f64>) -> PySeries {
        PySeries {
            inner: Series::new(s.name.clone(), Column::f64(out), Arc::clone(&s.index)),
        }
    }

    /// An int64 count-like result (`count` / `nunique`): NaN cells become NA.
    fn i64_series(&self, s: &Series, out: Vec<f64>) -> PySeries {
        let validity = volas_core::Validity::from_valid_iter(
            out.len(),
            out.iter().map(|x| !x.is_nan()),
        );
        let vals = out.iter().map(|&x| if x.is_nan() { 0 } else { x as i64 }).collect();
        PySeries {
            inner: Series::new(
                s.name.clone(),
                Column::i64_with(vals, validity),
                Arc::clone(&s.index),
            ),
        }
    }

    fn data(&self, s: &Series) -> Vec<f64> {
        s.data.to_f64_vec()
    }

    pub(crate) fn count(&self, s: &Series) -> PySeries {
        // count gates on window COVERAGE (see the kernel), so it handles
        // `center` itself instead of going through the NaN-padding `run`.
        let out = w::count(&self.data(s), self.window, self.min_periods, self.center);
        self.i64_series(s, out)
    }
    pub(crate) fn nunique(&self, s: &Series) -> PySeries {
        let out = self.run(&self.data(s), |d| w::nunique(d, self.window, self.min_periods));
        self.i64_series(s, out)
    }
    pub(crate) fn sum(&self, s: &Series) -> PySeries {
        let out = self.run(&self.data(s), |d| w::sum(d, self.window, self.min_periods));
        self.f64_series(s, out)
    }
    pub(crate) fn mean(&self, s: &Series) -> PySeries {
        let out = self.run(&self.data(s), |d| w::mean(d, self.window, self.min_periods));
        self.f64_series(s, out)
    }
    pub(crate) fn median(&self, s: &Series) -> PySeries {
        let out = self.run(&self.data(s), |d| w::median(d, self.window, self.min_periods));
        self.f64_series(s, out)
    }
    pub(crate) fn min(&self, s: &Series) -> PySeries {
        let out = self.run(&self.data(s), |d| w::min(d, self.window, self.min_periods));
        self.f64_series(s, out)
    }
    pub(crate) fn max(&self, s: &Series) -> PySeries {
        let out = self.run(&self.data(s), |d| w::max(d, self.window, self.min_periods));
        self.f64_series(s, out)
    }
    pub(crate) fn var(&self, s: &Series, ddof: usize) -> PySeries {
        let out = self.run(&self.data(s), |d_| w::var(d_, self.window, self.min_periods, ddof));
        self.f64_series(s, out)
    }
    pub(crate) fn std(&self, s: &Series, ddof: usize) -> PySeries {
        let out = self.run(&self.data(s), |d_| w::std(d_, self.window, self.min_periods, ddof));
        self.f64_series(s, out)
    }
    pub(crate) fn sem(&self, s: &Series, ddof: usize) -> PySeries {
        let out = self.run(&self.data(s), |d_| w::sem(d_, self.window, self.min_periods, ddof));
        self.f64_series(s, out)
    }
    pub(crate) fn skew(&self, s: &Series) -> PySeries {
        let out = self.run(&self.data(s), |d| w::skew(d, self.window, self.min_periods));
        self.f64_series(s, out)
    }
    pub(crate) fn kurt(&self, s: &Series) -> PySeries {
        let out = self.run(&self.data(s), |d| w::kurt(d, self.window, self.min_periods));
        self.f64_series(s, out)
    }
    pub(crate) fn quantile(&self, s: &Series, q: f64, interpolation: &str) -> PyResult<PySeries> {
        validate_interpolation(interpolation)?;
        if !(0.0..=1.0).contains(&q) {
            return Err(PyValueError::new_err("quantile q must be in [0, 1]"));
        }
        let out = self.run(&self.data(s), |d| {
            w::quantile(d, self.window, self.min_periods, q, interpolation)
        });
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn rank(&self, s: &Series, method: &str, ascending: bool, pct: bool) -> PyResult<PySeries> {
        validate_rank_method(method)?;
        let out = self.run(&self.data(s), |d| {
            w::rank(d, self.window, self.min_periods, method, ascending, pct)
        });
        Ok(self.f64_series(s, out))
    }
    /// `first` / `last` preserve the source dtype: gather the edge positions.
    pub(crate) fn edge(&self, s: &Series, last: bool) -> PySeries {
        let n = s.len();
        let mut valid: Vec<bool> = (0..n).map(|i| s.data.is_valid(i)).collect();
        let pos = if self.center {
            // the same pad-and-relabel trick as `run` (padding is never valid,
            // so it is never selected as an edge position).
            let fwd = self.window - self.window / 2 - 1;
            valid.extend(std::iter::repeat(false).take(fwd));
            w::edge_positions(&valid, self.window, self.min_periods, last)[fwd..fwd + n].to_vec()
        } else {
            w::edge_positions(&valid, self.window, self.min_periods, last)
        };
        PySeries {
            inner: Series::new(
                s.name.clone(),
                s.data.take_optional(&pos),
                Arc::clone(&s.index),
            ),
        }
    }
    fn pair<'a>(&self, s: &Series, other: &'a PySeries) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if other.inner.len() != s.len() {
            return Err(PyValueError::new_err(format!(
                "cannot align series of different lengths ({} vs {})",
                s.len(),
                other.inner.len()
            )));
        }
        other.inner.data.require_numeric().map_err(crate::pyerr)?;
        Ok((s.data.to_f64_vec(), other.inner.data.to_f64_vec()))
    }
    pub(crate) fn corr(&self, s: &Series, other: &PySeries) -> PyResult<PySeries> {
        let (x, y) = self.pair(s, other)?;
        let out = self.run2(&x, &y, |a, b| w::corr(a, b, self.window, self.min_periods));
        Ok(self.f64_series(s, out))
    }
    pub(crate) fn cov(&self, s: &Series, other: &PySeries, ddof: usize) -> PyResult<PySeries> {
        let (x, y) = self.pair(s, other)?;
        let out = self.run2(&x, &y, |a, b| w::cov(a, b, self.window, self.min_periods, ddof));
        Ok(self.f64_series(s, out))
    }

    /// Two-series variant of `run` (corr / cov under `center=True`).
    fn run2(&self, x: &[f64], y: &[f64], k: impl FnOnce(&[f64], &[f64]) -> Vec<f64>) -> Vec<f64> {
        if !self.center {
            return k(x, y);
        }
        let n = x.len();
        let fwd = self.window - self.window / 2 - 1;
        let (mut xp, mut yp) = (x.to_vec(), y.to_vec());
        xp.extend(std::iter::repeat(f64::NAN).take(fwd));
        yp.extend(std::iter::repeat(f64::NAN).take(fwd));
        k(&xp, &yp)[fwd..fwd + n].to_vec()
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
    fn nunique(&self) -> PySeries {
        self.spec.nunique(&self.series)
    }
    /// Rolling sum (pandas `sum`).
    fn sum(&self) -> PySeries {
        self.spec.sum(&self.series)
    }
    /// Rolling mean (pandas `mean`).
    fn mean(&self) -> PySeries {
        self.spec.mean(&self.series)
    }
    /// Rolling median (pandas `median`).
    fn median(&self) -> PySeries {
        self.spec.median(&self.series)
    }
    /// Rolling minimum (pandas `min`).
    fn min(&self) -> PySeries {
        self.spec.min(&self.series)
    }
    /// Rolling maximum (pandas `max`).
    fn max(&self) -> PySeries {
        self.spec.max(&self.series)
    }
    /// Rolling sample variance (pandas `var(ddof=1)`).
    #[pyo3(signature = (ddof = 1))]
    fn var(&self, ddof: i64) -> PyResult<PySeries> {
        Ok(self.spec.var(&self.series, validate_ddof(ddof)?))
    }
    /// Rolling sample standard deviation (pandas `std(ddof=1)`).
    #[pyo3(signature = (ddof = 1))]
    fn std(&self, ddof: i64) -> PyResult<PySeries> {
        Ok(self.spec.std(&self.series, validate_ddof(ddof)?))
    }
    /// Standard error of the mean (pandas `sem`).
    #[pyo3(signature = (ddof = 1))]
    fn sem(&self, ddof: i64) -> PyResult<PySeries> {
        Ok(self.spec.sem(&self.series, validate_ddof(ddof)?))
    }
    /// Bias-corrected sample skewness (pandas `skew`; >= 3 present values).
    fn skew(&self) -> PySeries {
        self.spec.skew(&self.series)
    }
    /// Bias-corrected excess kurtosis (pandas `kurt`; >= 4 present values).
    fn kurt(&self) -> PySeries {
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
    fn nunique(&self) -> PySeries {
        self.spec.nunique(&self.series)
    }
    /// Expanding sum.
    fn sum(&self) -> PySeries {
        self.spec.sum(&self.series)
    }
    /// Expanding mean.
    fn mean(&self) -> PySeries {
        self.spec.mean(&self.series)
    }
    /// Expanding median.
    fn median(&self) -> PySeries {
        self.spec.median(&self.series)
    }
    /// Expanding minimum.
    fn min(&self) -> PySeries {
        self.spec.min(&self.series)
    }
    /// Expanding maximum.
    fn max(&self) -> PySeries {
        self.spec.max(&self.series)
    }
    /// Expanding sample variance (`ddof=1`).
    #[pyo3(signature = (ddof = 1))]
    fn var(&self, ddof: i64) -> PyResult<PySeries> {
        Ok(self.spec.var(&self.series, validate_ddof(ddof)?))
    }
    /// Expanding sample standard deviation (`ddof=1`).
    #[pyo3(signature = (ddof = 1))]
    fn std(&self, ddof: i64) -> PyResult<PySeries> {
        Ok(self.spec.std(&self.series, validate_ddof(ddof)?))
    }
    /// Expanding standard error of the mean.
    #[pyo3(signature = (ddof = 1))]
    fn sem(&self, ddof: i64) -> PyResult<PySeries> {
        Ok(self.spec.sem(&self.series, validate_ddof(ddof)?))
    }
    /// Expanding skewness.
    fn skew(&self) -> PySeries {
        self.spec.skew(&self.series)
    }
    /// Expanding excess kurtosis.
    fn kurt(&self) -> PySeries {
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
    fn wrap(&self, out: Vec<f64>) -> PySeries {
        PySeries {
            inner: Series::new(
                self.series.name.clone(),
                Column::f64(out),
                Arc::clone(&self.series.index),
            ),
        }
    }
    fn pair(&self, other: &PySeries) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if other.inner.len() != self.series.len() {
            return Err(PyValueError::new_err(format!(
                "cannot align series of different lengths ({} vs {})",
                self.series.len(),
                other.inner.len()
            )));
        }
        other.inner.data.require_numeric().map_err(crate::pyerr)?;
        Ok((self.series.data.to_f64_vec(), other.inner.data.to_f64_vec()))
    }
}

#[pymethods]
impl PyEwm {
    /// Exponentially-weighted mean (pandas `ewm(...).mean()`).
    pub(crate) fn mean(&self) -> PySeries {
        self.wrap(w::ewm_mean(
            &self.series.data.to_f64_vec(),
            self.alpha,
            self.adjust,
            self.ignore_na,
            self.min_periods,
        ))
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
            &self.series.data.to_f64_vec(),
            self.alpha,
            self.ignore_na,
            self.min_periods,
        )))
    }
    /// Exponentially-weighted variance (pandas `var(bias=False)`).
    #[pyo3(signature = (bias = false))]
    pub(crate) fn var(&self, bias: bool) -> PySeries {
        self.wrap(w::ewm_var(
            &self.series.data.to_f64_vec(),
            self.alpha,
            self.adjust,
            self.ignore_na,
            bias,
            self.min_periods.max(1),
        ))
    }
    /// Exponentially-weighted standard deviation.
    #[pyo3(signature = (bias = false))]
    pub(crate) fn std(&self, bias: bool) -> PySeries {
        let v = w::ewm_var(
            &self.series.data.to_f64_vec(),
            self.alpha,
            self.adjust,
            self.ignore_na,
            bias,
            self.min_periods.max(1),
        );
        self.wrap(v.into_iter().map(f64::sqrt).collect())
    }
    /// Exponentially-weighted covariance against another series.
    #[pyo3(signature = (other, bias = false))]
    fn cov(&self, other: &PySeries, bias: bool) -> PyResult<PySeries> {
        let (x, y) = self.pair(other)?;
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
        let (x, y) = self.pair(other)?;
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
