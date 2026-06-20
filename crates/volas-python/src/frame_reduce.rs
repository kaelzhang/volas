//! `DataFrame` column-wise reductions (sum / mean / quantile / describe / corr / …).

use std::sync::Arc;

use pyo3::prelude::*;
use volas_core::{
    stats, Column, DataFrame, Index, Series,
};

#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PyDataFrame {

    /// Per-column count of non-missing values (pandas `count`) -> a Series indexed
    /// by column name (`int64`), reading each column's validity.
    pub(crate) fn count(&self) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let names: Vec<String> = df.names().to_vec();
        let counts: Vec<i64> = df.columns().iter().map(|c| c.count() as i64).collect();
        Ok(PySeries {
            inner: Series::new(None, Column::i64(counts), Arc::new(Index::str(names))),
        })
    }

    /// Per-column NaN-skipping sum (pandas `df.sum()`; non-numeric skipped).
    pub(crate) fn sum(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| {
            let v = c.to_f64_vec();
            (0..c.len()).filter(|&i| c.is_valid(i) && !v[i].is_nan()).map(|i| v[i]).sum()
        })
    }
    /// Per-column NaN-skipping product (pandas `df.prod()`).
    pub(crate) fn prod(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| {
            let v = c.to_f64_vec();
            (0..c.len())
                .filter(|&i| c.is_valid(i) && !v[i].is_nan())
                .map(|i| v[i])
                .product()
        })
    }
    /// Per-column NaN-skipping mean (pandas `df.mean()`).
    pub(crate) fn mean(&self) -> PyResult<PySeries> {
        self.reduce_with(|s| s.mean_f64())
    }
    /// Per-column sample variance (ddof=1, pandas `df.var()`).
    pub(crate) fn var(&self) -> PyResult<PySeries> {
        self.reduce_with(|s| s.var_f64())
    }
    /// Per-column sample standard deviation (pandas `df.std()`).
    pub(crate) fn std(&self) -> PyResult<PySeries> {
        self.reduce_with(|s| s.var_f64().sqrt())
    }
    /// Per-column NaN-skipping median (pandas `df.median()`).
    pub(crate) fn median(&self) -> PyResult<PySeries> {
        self.try_reduce_with(|s| s.quantile_f64(0.5))
    }
    /// Per-column `q`-quantile (pandas `df.quantile(q)`).
    #[pyo3(signature = (q = 0.5))]
    pub(crate) fn quantile(&self, q: f64) -> PyResult<PySeries> {
        self.try_reduce_with(|s| s.quantile_f64(q))
    }
    /// Per-column NaN-skipping minimum (pandas `df.min()`; numeric columns).
    pub(crate) fn min(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| {
            let v = c.to_f64_vec();
            (0..c.len())
                .filter(|&i| c.is_valid(i) && !v[i].is_nan())
                .map(|i| v[i])
                .fold(f64::NAN, f64::min)
        })
    }
    /// Per-column NaN-skipping maximum (pandas `df.max()`).
    pub(crate) fn max(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| {
            let v = c.to_f64_vec();
            (0..c.len())
                .filter(|&i| c.is_valid(i) && !v[i].is_nan())
                .map(|i| v[i])
                .fold(f64::NAN, f64::max)
        })
    }
    /// Per-column count of distinct present values (pandas `df.nunique()`).
    pub(crate) fn nunique(&self) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let names: Vec<String> = df.names().to_vec();
        let counts: Vec<i64> = df
            .columns()
            .iter()
            .map(|c| {
                let mut seen = std::collections::HashSet::new();
                (0..c.len())
                    .filter_map(|i| crate::series::cell_key(c, i))
                    .filter(|k| seen.insert(k.clone()))
                    .count() as i64
            })
            .collect();
        Ok(PySeries {
            inner: Series::new(None, Column::i64(counts), Arc::new(Index::str(names))),
        })
    }
    /// Per-column truthiness `any` (pandas `df.any()`): a present, non-zero /
    /// True / non-empty cell counts.
    pub(crate) fn any(&self) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        Ok(self.bool_reduce(true))
    }
    /// Per-column truthiness `all` (pandas `df.all()`), NA-skipping.
    pub(crate) fn all(&self) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        Ok(self.bool_reduce(false))
    }
    /// Per-column index label of the maximum (pandas `df.idxmax()`).
    pub(crate) fn idxmax(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        ensure_fresh(&self.inner)?;
        self.idx_extreme(py, true)
    }
    /// Per-column index label of the minimum (pandas `df.idxmin()`).
    pub(crate) fn idxmin(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        ensure_fresh(&self.inner)?;
        self.idx_extreme(py, false)
    }

    // --- column-wise reductions (-> a Series indexed by column name; numeric
    // columns only, pandas df.sem() etc.). -------------------------------------

    /// Per-column standard error of the mean (pandas `sem`).
    pub(crate) fn sem(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| stats::sem(&c.to_f64_vec()))
    }
    /// Per-column unbiased skewness (pandas `skew`).
    pub(crate) fn skew(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| stats::skew(&c.to_f64_vec()))
    }
    /// Per-column unbiased excess kurtosis (pandas `kurt`).
    pub(crate) fn kurt(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| stats::kurt(&c.to_f64_vec()))
    }

    /// Per-column summary statistics over the numeric columns (pandas `describe`):
    /// a frame indexed by `count / mean / std / min / 25% / 50% / 75% / max`.
    pub(crate) fn describe(&self) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let idx = df.index();
        let mut names = Vec::new();
        let mut cols = Vec::new();
        for (name, col) in df.names().iter().zip(df.columns()) {
            if col.dtype().is_numeric() {
                let s = PySeries {
                    inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(idx)),
                };
                names.push(name.clone());
                cols.push(s.describe()?.inner.data);
            }
        }
        // No numeric columns (empty / str-only / datetime-only frame) -> a 0x0
        // frame, consistent with corr / cov — not an 8-row describe index over zero
        // columns, which the core rejects as a height mismatch. volas describe is
        // numeric-only (pandas would return object-column stats here instead).
        let index = if cols.is_empty() {
            Index::str(Vec::new())
        } else {
            Index::str(describe_labels())
        };
        DataFrame::new(names, cols, Some(index))
            .map(PyDataFrame::plain)
            .map_err(pyerr)
    }

    /// Pairwise Pearson correlation matrix over the numeric columns (pandas
    /// `corr`): a square frame indexed and labelled by those column names.
    pub(crate) fn corr(&self) -> PyResult<PyDataFrame> {
        self.corr_cov(stats::corr)
    }

    /// Pairwise sample covariance matrix over the numeric columns (pandas `cov`).
    pub(crate) fn cov(&self) -> PyResult<PyDataFrame> {
        self.corr_cov(stats::cov)
    }
}

impl PyDataFrame {
    /// Per-numeric-column reduce via a Series-level helper -> f64 Series keyed
    /// by column name (non-numeric columns are skipped, like `reduce_cols`).
    fn reduce_with(&self, op: impl Fn(&PySeries) -> f64) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let idx = df.index();
        let mut names = Vec::new();
        let mut vals = Vec::new();
        for (name, col) in df.names().iter().zip(df.columns()) {
            if col.require_numeric().is_ok() {
                let s = PySeries {
                    inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(idx)),
                };
                names.push(name.clone());
                vals.push(op(&s));
            }
        }
        Ok(PySeries {
            inner: Series::new(None, Column::f64(vals), Arc::new(Index::str(names))),
        })
    }

    /// Like [`Self::reduce_with`] for fallible helpers (quantile).
    fn try_reduce_with(&self, op: impl Fn(&PySeries) -> PyResult<f64>) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let idx = df.index();
        let mut names = Vec::new();
        let mut vals = Vec::new();
        for (name, col) in df.names().iter().zip(df.columns()) {
            if col.require_numeric().is_ok() {
                let s = PySeries {
                    inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(idx)),
                };
                names.push(name.clone());
                vals.push(op(&s)?);
            }
        }
        Ok(PySeries {
            inner: Series::new(None, Column::f64(vals), Arc::new(Index::str(names))),
        })
    }

    /// Per-column truthiness any/all (NA-skipping) -> bool Series by name.
    fn bool_reduce(&self, want_any: bool) -> PySeries {
        let view = self.logical();
        let df = view.as_ref();
        let names: Vec<String> = df.names().to_vec();
        let vals: Vec<bool> = df
            .columns()
            .iter()
            .map(|c| {
                let truth = to_bool_vec(c);
                let present = (0..c.len()).filter(|&i| c.is_valid(i));
                if want_any {
                    present.into_iter().any(|i| truth[i])
                } else {
                    present.into_iter().all(|i| truth[i])
                }
            })
            .collect();
        PySeries {
            inner: Series::new(None, Column::bool(vals), Arc::new(Index::str(names))),
        }
    }

    /// Per-column index label of the extreme -> a Series of labels keyed by
    /// column name (the label dtype follows the index kind).
    fn idx_extreme(&self, py: Python<'_>, want_max: bool) -> PyResult<Py<PyAny>> {
        let view = self.logical();
        let df = view.as_ref();
        let names: Vec<String> = df.names().to_vec();
        let mut positions = Vec::with_capacity(names.len());
        for col in df.columns() {
            positions.push(argext(col, want_max)?);
        }
        let index = df.index();
        let labels = index.take(&positions).to_column();
        let s = PySeries {
            inner: Series::new(None, labels, Arc::new(Index::str(names))),
        };
        Ok(Py::new(py, s)?.into_any())
    }
}
