//! The rolling / expanding / ewm frame aggregators returned by
//! `DataFrame.rolling` / `.expanding` / `.ewm`.

use volas_core::DataFrame;

#[allow(unused_imports)]
use crate::*;

#[pyclass(name = "RollingFrame")]
pub struct PyRollingFrame {
    pub(crate) frame: DataFrame,
    pub(crate) window: usize,
    pub(crate) min_periods: usize,
    pub(crate) center: bool,
}

/// `df.expanding()` — per-numeric-column expanding aggregation -> DataFrame.
#[pyclass(name = "ExpandingFrame")]
pub struct PyExpandingFrame {
    pub(crate) frame: DataFrame,
    pub(crate) min_periods: usize,
}

/// `df.ewm(...)` — per-numeric-column EW aggregation -> DataFrame.
#[pyclass(name = "EwmFrame")]
pub struct PyEwmFrame {
    pub(crate) frame: DataFrame,
    pub(crate) alpha: f64,
    pub(crate) adjust: bool,
    pub(crate) ignore_na: bool,
    pub(crate) min_periods: usize,
}

/// Apply a Series-level window op over every column of `frame` (a non-numeric
/// column errors, like pandas with `numeric_only=False`). A frame-level window
/// is a BULK read — it would aggregate a stale cached-directive column as if
/// it were data — so it carries the same fulfill guard as to_numpy / iloc
/// (same-guard symmetry, E8; self-audit SA2-3).
fn frame_window(
    frame: &DataFrame,
    op: impl Fn(&PySeries) -> PyResult<PySeries>,
) -> PyResult<PyDataFrame> {
    crate::ensure_fresh(frame)?;
    let wrapper = PyDataFrame::plain(frame.clone());
    wrapper.map_cols(|s| {
        s.inner.data.require_numeric().map_err(pyerr)?;
        op(s)
    })
}

impl PyRollingFrame {
    fn spec(&self) -> crate::window::WinSpec {
        crate::window::WinSpec {
            window: self.window,
            min_periods: self.min_periods,
            center: self.center,
        }
    }
    fn agg(
        &self,
        op: impl Fn(&crate::window::WinSpec, &volas_core::Series) -> PyResult<PySeries>,
    ) -> PyResult<PyDataFrame> {
        let spec = self.spec();
        frame_window(&self.frame, |s| op(&spec, &s.inner))
    }
}

impl PyExpandingFrame {
    fn agg(
        &self,
        op: impl Fn(&crate::window::WinSpec, &volas_core::Series) -> PyResult<PySeries>,
    ) -> PyResult<PyDataFrame> {
        let spec = crate::window::WinSpec {
            window: usize::MAX,
            min_periods: self.min_periods,
            center: false,
        };
        frame_window(&self.frame, |s| op(&spec, &s.inner))
    }
}

#[pymethods]
impl PyRollingFrame {
    /// Per-column rolling count (int64) -> DataFrame.
    fn count(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.count(s)))
    }
    /// Per-column rolling distinct-count (int64).
    fn nunique(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.nunique(s)))
    }
    /// Per-column rolling sum.
    fn sum(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.sum(s)))
    }
    /// Per-column rolling mean.
    fn mean(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.mean(s)))
    }
    /// Per-column rolling median.
    fn median(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.median(s)))
    }
    /// Per-column rolling minimum.
    fn min(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.min(s)))
    }
    /// Per-column rolling maximum.
    fn max(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.max(s)))
    }
    /// Per-column rolling sample variance.
    #[pyo3(signature = (ddof = 1))]
    fn var(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.var(s, ddof)))
    }
    /// Per-column rolling sample standard deviation.
    #[pyo3(signature = (ddof = 1))]
    fn std(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.std(s, ddof)))
    }
    /// Per-column standard error of the mean.
    #[pyo3(signature = (ddof = 1))]
    fn sem(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.sem(s, ddof)))
    }
    /// Per-column rolling skewness.
    fn skew(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.skew(s)))
    }
    /// Per-column rolling excess kurtosis.
    fn kurt(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.kurt(s)))
    }
    /// Per-column rolling quantile.
    #[pyo3(signature = (q, interpolation = "linear"))]
    fn quantile(&self, q: f64, interpolation: &str) -> PyResult<PyDataFrame> {
        self.agg(|w, s| w.quantile(s, q, interpolation))
    }
    /// Per-column rolling rank of the current value within its window.
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PyDataFrame> {
        self.agg(|w, s| w.rank(s, method, ascending, pct))
    }
    /// Per-column first present value per window, dtype-preserving.
    fn first(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.edge(s, false)))
    }
    /// Per-column last present value per window, dtype-preserving.
    fn last(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.edge(s, true)))
    }
}

#[pymethods]
impl PyExpandingFrame {
    /// Per-column expanding count (int64) -> DataFrame.
    fn count(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.count(s)))
    }
    /// Per-column expanding distinct-count (int64).
    fn nunique(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.nunique(s)))
    }
    /// Per-column expanding sum.
    fn sum(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.sum(s)))
    }
    /// Per-column expanding mean.
    fn mean(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.mean(s)))
    }
    /// Per-column expanding median.
    fn median(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.median(s)))
    }
    /// Per-column expanding minimum.
    fn min(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.min(s)))
    }
    /// Per-column expanding maximum.
    fn max(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.max(s)))
    }
    /// Per-column expanding sample variance.
    #[pyo3(signature = (ddof = 1))]
    fn var(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.var(s, ddof)))
    }
    /// Per-column expanding sample standard deviation.
    #[pyo3(signature = (ddof = 1))]
    fn std(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.std(s, ddof)))
    }
    /// Per-column expanding standard error of the mean.
    #[pyo3(signature = (ddof = 1))]
    fn sem(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.sem(s, ddof)))
    }
    /// Per-column expanding skewness.
    fn skew(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.skew(s)))
    }
    /// Per-column expanding excess kurtosis.
    fn kurt(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.kurt(s)))
    }
    /// Per-column expanding quantile.
    #[pyo3(signature = (q, interpolation = "linear"))]
    fn quantile(&self, q: f64, interpolation: &str) -> PyResult<PyDataFrame> {
        self.agg(|w, s| w.quantile(s, q, interpolation))
    }
    /// Per-column expanding rank.
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PyDataFrame> {
        self.agg(|w, s| w.rank(s, method, ascending, pct))
    }
    /// Per-column first present value so far, dtype-preserving.
    fn first(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.edge(s, false)))
    }
    /// Per-column last present value so far, dtype-preserving.
    fn last(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.edge(s, true)))
    }
}

#[pymethods]
impl PyEwmFrame {
    /// Per-column exponentially-weighted mean -> DataFrame.
    fn mean(&self) -> PyResult<PyDataFrame> {
        frame_window(&self.frame, |s| {
            Ok(crate::window::PyEwm {
                series: s.inner.clone(),
                alpha: self.alpha,
                adjust: self.adjust,
                ignore_na: self.ignore_na,
                min_periods: self.min_periods,
            }
            .mean())
        })
    }
    /// Per-column exponentially-weighted sum (`adjust=True` only).
    fn sum(&self) -> PyResult<PyDataFrame> {
        frame_window(&self.frame, |s| {
            crate::window::PyEwm {
                series: s.inner.clone(),
                alpha: self.alpha,
                adjust: self.adjust,
                ignore_na: self.ignore_na,
                min_periods: self.min_periods,
            }
            .sum()
        })
    }
    /// Per-column exponentially-weighted variance.
    #[pyo3(signature = (bias = false))]
    fn var(&self, bias: bool) -> PyResult<PyDataFrame> {
        frame_window(&self.frame, |s| {
            Ok(crate::window::PyEwm {
                series: s.inner.clone(),
                alpha: self.alpha,
                adjust: self.adjust,
                ignore_na: self.ignore_na,
                min_periods: self.min_periods,
            }
            .var(bias))
        })
    }
    /// Per-column exponentially-weighted standard deviation.
    #[pyo3(signature = (bias = false))]
    fn std(&self, bias: bool) -> PyResult<PyDataFrame> {
        frame_window(&self.frame, |s| {
            Ok(crate::window::PyEwm {
                series: s.inner.clone(),
                alpha: self.alpha,
                adjust: self.adjust,
                ignore_na: self.ignore_na,
                min_periods: self.min_periods,
            }
            .std(bias))
        })
    }
}
