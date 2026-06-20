//! `Series` reductions (sum / mean / quantile / describe / corr / idxmax / …).

use std::sync::Arc;

use pyo3::prelude::*;
use volas_core::{
    stats, Column, Index, Series,
};

#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PySeries {

    // Reductions return numpy scalars (pandas' boundary representation). The
    // dtype-preserving ones (sum/prod/min/max) carry the column's result dtype
    // (np.int64 for an int column, etc.); the always-float statistics box np.float64.

    // Each numeric reduction first asserts the column is numeric — a str/datetime
    // reduction used to funnel through to_f64_vec and silently return 0.0 / NaN,
    // which the API contract (C4) forbids (V3).

    /// NaN-skipping mean (pandas `mean`) -> `np.float64`.
    pub(crate) fn mean(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.mean_f64()))
    }
    /// Sum (pandas `sum`), dtype-preserving: float -> `np.float64`, int / bool ->
    /// `np.int64` (computed natively).
    pub(crate) fn sum(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(scalar_to_numpy(py, self.inner.data.sum()))
    }
    /// Product (pandas `prod`), dtype-preserving.
    pub(crate) fn prod(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(scalar_to_numpy(py, self.inner.data.prod()))
    }
    /// Minimum (pandas `min`). Order-based, so it serves any ordered dtype:
    /// numeric/bool reduce to a numpy scalar (int -> `np.int64`, exact even past
    /// 2^53), str to a Python str, datetime to `np.datetime64` — not the f64
    /// funnel. See [`extreme_value`].
    pub(crate) fn min(&self, py: Python<'_>) -> Py<PyAny> {
        extreme_value(py, &self.inner.data, false)
    }
    /// Maximum (pandas `max`), order-based and dtype-typed. See [`extreme_value`].
    pub(crate) fn max(&self, py: Python<'_>) -> Py<PyAny> {
        extreme_value(py, &self.inner.data, true)
    }
    /// Sample variance (`ddof=1`, pandas `var`) -> `np.float64`.
    pub(crate) fn var(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.var_f64()))
    }
    /// Sample standard deviation (`ddof=1`, pandas `std`) -> `np.float64`.
    pub(crate) fn std(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.var_f64().sqrt()))
    }
    /// Median (pandas `median`) -> `np.float64`.
    pub(crate) fn median(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.median_f64()))
    }
    /// Standard error of the mean (`ddof=1`, pandas `sem`) -> `np.float64`.
    pub(crate) fn sem(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, stats::sem(&self.inner.data.to_f64_vec())))
    }
    /// Adjusted Fisher-Pearson skewness (pandas `skew`) -> `np.float64`.
    pub(crate) fn skew(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, stats::skew(&self.inner.data.to_f64_vec())))
    }
    /// Excess kurtosis, Fisher's definition (pandas `kurt`) -> `np.float64`.
    pub(crate) fn kurt(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, stats::kurt(&self.inner.data.to_f64_vec())))
    }

    /// Pairwise Pearson correlation with `other` (pandas `corr`); positional
    /// alignment (volas does not reindex), dropping NaN pairs.
    pub(crate) fn corr(&self, other: &PySeries) -> PyResult<f64> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        other.inner.data.require_numeric().map_err(pyerr)?;
        Ok(stats::corr(&self.inner.data.to_f64_vec(), &other.inner.data.to_f64_vec()))
    }

    /// Pairwise sample covariance with `other`, ddof=1 (pandas `cov`); positional
    /// alignment, dropping NaN pairs.
    pub(crate) fn cov(&self, other: &PySeries) -> PyResult<f64> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        other.inner.data.require_numeric().map_err(pyerr)?;
        Ok(stats::cov(&self.inner.data.to_f64_vec(), &other.inner.data.to_f64_vec()))
    }

    /// Summary statistics (pandas `describe`): a Series indexed by
    /// `count / mean / std / min / 25% / 50% / 75% / max`.
    pub(crate) fn describe(&self) -> PyResult<PySeries> {
        // describe is a numeric summary (mean/std/quantiles); a str/datetime column
        // would funnel through to_f64_vec to nonsense, so it raises (C4) until a
        // dtype-aware categorical/datetime describe is designed.
        self.inner.data.require_numeric().map_err(pyerr)?;
        let v = self.inner.data.to_f64_vec();
        let count = non_nan(&self.inner.data).len() as f64;
        let vals = vec![
            count,
            self.mean_f64(),
            self.var_f64().sqrt(),
            stats::extreme(&v, false).unwrap_or(f64::NAN),
            self.quantile_f64(0.25)?,
            self.quantile_f64(0.5)?,
            self.quantile_f64(0.75)?,
            stats::extreme(&v, true).unwrap_or(f64::NAN),
        ];
        let labels = describe_labels();
        Ok(PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                Column::f64(vals),
                Arc::new(Index::str(labels)),
            ),
        })
    }

    /// Number of non-missing values (pandas `count`) -> `int`.
    pub(crate) fn count(&self) -> usize {
        self.inner.data.count()
    }

    /// Number of distinct non-missing values (pandas `nunique`) -> `int`.
    pub(crate) fn nunique(&self) -> usize {
        self.inner.data.nunique()
    }

    /// True if any element is truthy (NaN skipped) — pandas `any` -> `np.bool_`.
    /// A bool/numeric truthiness reduction, so str/datetime raise rather than
    /// funnel to a silent (and dtype-dependent) answer (C4).
    pub(crate) fn any(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        let r = match &self.inner.data {
            // skipna: a NA bool is its `false` placeholder in the buffer, so read the
            // validity — only a *present* true counts (matching pandas nullable any).
            Column::Bool(v, val) => v.iter().enumerate().any(|(i, &b)| val.is_valid(i) && b),
            other => other.to_f64_vec().iter().any(|&x| !x.is_nan() && x != 0.0),
        };
        Ok(np_bool(py, r))
    }

    /// True if every non-missing element is truthy (empty / all-NA -> True) — pandas
    /// `all` -> `np.bool_`, default `skipna=True`. Bool/numeric only (see `any`).
    pub(crate) fn all(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        let r = match &self.inner.data {
            // skipna: a NA is ignored (vacuously satisfies), only a present false fails.
            Column::Bool(v, val) => v.iter().enumerate().all(|(i, &b)| !val.is_valid(i) || b),
            other => other.to_f64_vec().iter().all(|&x| x.is_nan() || x != 0.0),
        };
        Ok(np_bool(py, r))
    }

    /// The `q`-quantile in `[0, 1]` (linear interpolation, NaN-skipping) — pandas
    /// `quantile` -> `np.float64`.
    #[pyo3(signature = (q = 0.5))]
    pub(crate) fn quantile(&self, py: Python<'_>, q: f64) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.quantile_f64(q)?))
    }

    /// The index **label** of the maximum value (NaN-skipping); raises on an
    /// all-NA series (pandas `idxmax`).
    pub(crate) fn idxmax(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(label_to_py(
            py,
            &self.inner.index,
            argext(&self.inner.data, true)?,
        ))
    }

    /// The index **label** of the minimum value (NaN-skipping); raises on an
    /// all-NA series (pandas `idxmin`).
    pub(crate) fn idxmin(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(label_to_py(
            py,
            &self.inner.index,
            argext(&self.inner.data, false)?,
        ))
    }
}
