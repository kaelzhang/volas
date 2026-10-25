//! `DataFrame` column-wise reductions (sum / mean / quantile / describe / corr / …).

use std::sync::Arc;

use pyo3::prelude::*;
use volas_core::{
    stats, Column, DType, DataFrame, Index, IndexKind, Scalar, Series,
};

#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PyDataFrame {

    /// Per-column count of non-missing values (pandas `count`) -> a Series indexed
    /// by column name (`int64`), reading each column's validity.
    pub(crate) fn count(&self) -> PyResult<PySeries> {
        let view = self.read()?;
        let df = view.as_ref();
        let names: Vec<String> = df.names().to_vec();
        let counts: Vec<i64> = df.columns().iter().map(|c| c.count() as i64).collect();
        Ok(PySeries {
            inner: Series::new(None, Column::i64(counts), Arc::new(Index::new(None, volas_core::IndexKind::Str(names)).map_err(pyerr)?)).map_err(pyerr)?,
        })
    }

    /// Per-column sum (pandas `df.sum()`; str / datetime skipped, bool counts), in the columns'
    /// own dtype: an all-int frame sums in int64, exactly (contract #26), and an
    /// int64 overflow raises (#25).
    pub(crate) fn sum(&self) -> PyResult<PySeries> {
        self.reduce_typed(|c| c.sum().map(|s| (s.dtype(), s)))
    }
    /// Per-column product (pandas `df.prod()`), typed like `sum`.
    pub(crate) fn prod(&self) -> PyResult<PySeries> {
        self.reduce_typed(|c| c.prod().map(|s| (s.dtype(), s)))
    }
    /// Per-column NaN-skipping mean (pandas `df.mean()`).
    pub(crate) fn mean(&self) -> PyResult<PySeries> {
        self.try_reduce_with(|s| s.mean_f64())
    }
    /// Per-column sample variance (ddof=1, pandas `df.var()`).
    pub(crate) fn var(&self) -> PyResult<PySeries> {
        self.try_reduce_with(|s| s.var_f64())
    }
    /// Per-column sample standard deviation (pandas `df.std()`).
    pub(crate) fn std(&self) -> PyResult<PySeries> {
        self.try_reduce_with(|s| Ok(s.var_f64()?.sqrt()))
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
    /// Per-column minimum (pandas `df.min()`; numeric columns), in the columns'
    /// own dtype: an all-int frame stays int, exactly (contract #26).
    pub(crate) fn min(&self) -> PyResult<PySeries> {
        self.reduce_typed(|c| Ok((c.dtype(), c.extreme(false))))
    }
    /// Per-column maximum (pandas `df.max()`), typed like `min`.
    pub(crate) fn max(&self) -> PyResult<PySeries> {
        self.reduce_typed(|c| Ok((c.dtype(), c.extreme(true))))
    }
    /// Per-column count of distinct present values (pandas `df.nunique()`).
    pub(crate) fn nunique(&self) -> PyResult<PySeries> {
        let view = self.read()?;
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
            inner: Series::new(None, Column::i64(counts), Arc::new(Index::new(None, volas_core::IndexKind::Str(names)).map_err(pyerr)?)).map_err(pyerr)?,
        })
    }
    /// Per-column truthiness `any` (pandas `df.any()`): a present, non-zero /
    /// True / non-empty cell counts.
    pub(crate) fn any(&self) -> PyResult<PySeries> {
        Self::bool_reduce(&self.read()?, true)
    }
    /// Per-column truthiness `all` (pandas `df.all()`), NA-skipping.
    pub(crate) fn all(&self) -> PyResult<PySeries> {
        Self::bool_reduce(&self.read()?, false)
    }
    /// Per-column index label of the maximum (pandas `df.idxmax()`).
    pub(crate) fn idxmax(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::idx_extreme(py, &self.read()?, true)
    }
    /// Per-column index label of the minimum (pandas `df.idxmin()`).
    pub(crate) fn idxmin(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::idx_extreme(py, &self.read()?, false)
    }

    // --- column-wise reductions (-> a Series indexed by column name; numeric and
    // bool columns, pandas df.sem() etc.). -------------------------------------

    /// Per-column standard error of the mean (pandas `sem`).
    pub(crate) fn sem(&self) -> PyResult<PySeries> {
        self.reduce_cols("sem", stats::sem)
    }
    /// Per-column unbiased skewness (pandas `skew`).
    pub(crate) fn skew(&self) -> PyResult<PySeries> {
        self.reduce_cols("skew", stats::skew)
    }
    /// Per-column unbiased excess kurtosis (pandas `kurt`).
    pub(crate) fn kurt(&self) -> PyResult<PySeries> {
        self.reduce_cols("kurt", stats::kurt)
    }

    /// Per-column summary statistics over the numeric columns (pandas `describe`):
    /// a frame indexed by `count / mean / std / min / 25% / 50% / 75% / max`.
    pub(crate) fn describe(&self) -> PyResult<PyDataFrame> {
        let view = self.read()?;
        let df = view.as_ref();
        let idx = df.index();
        let mut names = Vec::new();
        let mut cols = Vec::new();
        for (name, col) in df.names().iter().zip(df.columns()) {
            if col.dtype().is_numeric() {
                let s = PySeries {
                    inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(idx)).map_err(pyerr)?,
                };
                names.push(name.clone());
                cols.push(s.describe()?.inner.into_data());
            }
        }
        // No numeric columns (empty / str-only / datetime-only frame) -> a 0x0
        // frame, consistent with corr / cov — not an 8-row describe index over zero
        // columns, which the core rejects as a height mismatch. volas describe is
        // numeric-only (pandas would return object-column stats here instead).
        let index = if cols.is_empty() {
            Index::new(None, volas_core::IndexKind::Str(Vec::new())).map_err(pyerr)?
        } else {
            Index::new(None, volas_core::IndexKind::Str(describe_labels())).map_err(pyerr)?
        };
        DataFrame::new(names, cols, Some(index))
            .map(PyDataFrame::plain)
            .map_err(pyerr)
    }

    /// Pairwise Pearson correlation matrix over the numeric columns (pandas
    /// `corr`): a square frame indexed and labelled by those column names.
    pub(crate) fn corr(&self) -> PyResult<PyDataFrame> {
        self.corr_cov("corr", stats::corr)
    }

    /// Pairwise sample covariance matrix over the numeric columns (pandas `cov`).
    pub(crate) fn cov(&self) -> PyResult<PyDataFrame> {
        self.corr_cov("cov", stats::cov)
    }
}

impl PyDataFrame {
    /// Per-reducible-column (`frame::reduces`, like `reduce_cols`) typed reduce:
    /// `op` gives each column's `(contributed dtype, value)` and the result Series
    /// is assembled by the core [`Column::from_scalars`] — the supertype of the
    /// parts, each value converted losslessly (no float64 round trip).
    fn reduce_typed(&self, op: impl Fn(&Column) -> volas_core::Result<(DType, Scalar)>) -> PyResult<PySeries> {
        let view = self.read()?;
        let df = view.as_ref();
        let mut names = Vec::new();
        let mut parts = Vec::new();
        for (name, col) in df.names().iter().zip(df.columns()) {
            if crate::frame::reduces(col) {
                names.push(name.clone());
                parts.push(op(col).map_err(pyerr)?);
            }
        }
        let data = Column::from_scalars(&parts).map_err(pyerr)?;
        Ok(PySeries { inner: Series::new(None, data, Arc::new(Index::new(None, volas_core::IndexKind::Str(names)).map_err(pyerr)?)).map_err(pyerr)? })
    }


    /// Per-numeric-column reduce via a fallible Series-level float64 helper ->
    /// f64 Series keyed by column name (non-numeric columns are skipped).
    fn try_reduce_with(&self, op: impl Fn(&PySeries) -> PyResult<f64>) -> PyResult<PySeries> {
        let view = self.read()?;
        let df = view.as_ref();
        let idx = df.index();
        let mut names = Vec::new();
        let mut vals = Vec::new();
        for (name, col) in df.names().iter().zip(df.columns()) {
            if crate::frame::reduces(col) {
                let s = PySeries {
                    inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(idx)).map_err(pyerr)?,
                };
                names.push(name.clone());
                vals.push(op(&s)?);
            }
        }
        Ok(PySeries {
            inner: Series::new(None, Column::f64(vals), Arc::new(Index::new(None, volas_core::IndexKind::Str(names)).map_err(pyerr)?)).map_err(pyerr)?,
        })
    }

    /// Per-column truthiness any/all over the numeric / bool columns (a str /
    /// datetime column has no truth value, as `Series.any` raises for it), NA
    /// skipped — the same core truth owner as the Series surface (#27).
    fn bool_reduce(df: &Fresh<'_>, want_any: bool) -> PyResult<PySeries> {
        let (mut names, mut vals) = (Vec::new(), Vec::new());
        for (name, c) in df.names().iter().zip(df.columns()) {
            if crate::frame::reduces(c) {
                let mut truth = (0..c.len()).filter_map(|i| c.truth_at(i));
                names.push(name.clone());
                vals.push(if want_any { truth.any(|t| t) } else { truth.all(|t| t) });
            }
        }
        Ok(PySeries {
            inner: Series::new(None, Column::bool(vals), Arc::new(Index::new(None, volas_core::IndexKind::Str(names)).map_err(pyerr)?)).map_err(pyerr)?,
        })
    }

    /// Per-column index label of the extreme -> a Series of labels keyed by
    /// column name (the label dtype follows the index kind).
    fn idx_extreme(py: Python<'_>, df: &Fresh<'_>, want_max: bool) -> PyResult<Py<PyAny>> {
        let names: Vec<String> = df.names().to_vec();
        let mut positions = Vec::with_capacity(names.len());
        for col in df.columns() {
            positions.push(argext(col, want_max)?);
        }
        let index = df.index();
        // These are result values, not a new row axis: several columns may
        // choose the same label, and their extrema need not follow time order.
        let labels = match index.kind() {
            IndexKind::Range { start, .. } => Column::i64(positions.iter().map(|&i| start + i as i64).collect()),
            IndexKind::Int64(v) => Column::i64(positions.iter().map(|&i| v[i]).collect()),
            IndexKind::Datetime(v, _) => Column::datetime(positions.iter().map(|&i| v[i]).collect()),
            IndexKind::Str(v) => Column::str(positions.iter().map(|&i| v[i].clone()).collect()),
        };
        let s = PySeries {
            inner: Series::new(None, labels, Arc::new(Index::new(None, volas_core::IndexKind::Str(names)).map_err(pyerr)?)).map_err(pyerr)?,
        };
        Ok(Py::new(py, s)?.into_any())
    }
}
