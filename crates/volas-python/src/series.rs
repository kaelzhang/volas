//! The `volas.Series` pyclass and its positional `.iloc` accessor.

use volas_core::ColumnView;
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use volas_core::{
    binary_supertype, stats, Column, DType, Series,
};

#[allow(unused_imports)]
use crate::*;

/// ``volas.Series`` — a single named, indexed column (usually obtained from
/// ``df['col']`` or a directive like ``df['ma:5']``).
///
/// Supports NaN-skipping reductions (``mean`` / ``sum`` / ``min`` / ``max`` /
/// ``std`` / ``var`` / ``median``), element-wise arithmetic / comparison /
/// boolean operators (``+ - * /``, ``< <= == != >= >``, ``& | ^ ~``) against a
/// scalar or another equal-length Series, the TA-Lib math transforms
/// (``sin`` / ``sqrt`` / ``ln`` / …), and ``shift`` / ``diff`` / ``fillna`` /
/// ``isna`` / ``notna`` / ``dropna``. Index by position via ``s.iloc[...]`` or
/// label via ``s.loc[...]``; export with ``to_numpy`` / ``to_list``.
///
/// Usage::
///
///     close = df['close']
///     close.mean()            # NaN-skipping mean
///     (close - df['open'])    # element-wise difference
///     close.shift(1)          # lag by one bar
///     close.iloc[-1]          # last value
#[pyclass(name = "Series")]
pub struct PySeries {
    pub(crate) inner: Series,
}

impl PySeries {
    /// A new Series whose data and index are gathered by `idx` (backs
    /// `sort_values` and any fancy-index reorder).
    pub(crate) fn reindexed(&self, idx: &[usize]) -> PyResult<PySeries> {
        let index = Arc::new(self.inner.index().take(idx).map_err(pyerr)?);
        Ok(PySeries {
            inner: Series::new(
                self.inner.name().map(str::to_owned),
                self.inner.data().take(idx),
                index,
            ).map_err(pyerr)?,
        })
    }

    /// A new Series with the `[start, end)` row slice of both data and index
    /// (backs `head` / `tail`).
    pub(crate) fn sliced(&self, start: usize, end: usize) -> PyResult<PySeries> {
        Ok(PySeries {
            inner: Series::new(
                self.inner.name().map(str::to_owned),
                self.inner.data().slice(start, end),
                Arc::new(self.inner.index().slice(start, end)),
            ).map_err(pyerr)?,
        })
    }

    /// Box a float statistic as the column's float dtype: `np.float32` for an f32
    /// column, else `np.float64` (pandas: `f32.mean() -> np.float32`).
    pub(crate) fn box_float(&self, py: Python<'_>, value: f64) -> PyResult<Py<PyAny>> {
        if self.inner.data().dtype() == DType::F32 {
            np_f32(py, value as f32)
        } else {
            np_f64(py, value)
        }
    }

    // Raw f64 reduction values (the public methods box these as numpy scalars;
    // `describe` reuses them, so they stay unboxed here). Each is float64 math, so
    // an int64 past 2^53 raises (contract #26) through `to_f64_domain`.
    pub(crate) fn mean_f64(&self) -> PyResult<f64> {
        let v = non_nan(self.inner.data(), "mean")?;
        Ok(if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        })
    }
    pub(crate) fn var_f64(&self) -> PyResult<f64> {
        Ok(stats::var(&self.f64_domain("var")?))
    }
    pub(crate) fn median_f64(&self) -> PyResult<f64> {
        let mut v = non_nan(self.inner.data(), "median")?;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(stats::sorted_median(&v))
    }
    /// The values as `f64` for the float64 operation `op` (raises past 2^53).
    pub(crate) fn f64_domain(&self, op: &str) -> PyResult<Vec<f64>> {
        self.inner.data().to_f64_domain(op).map_err(pyerr)
    }
    /// The `q`-quantile as a raw f64 (linear interpolation, NaN-skipping).
    pub(crate) fn quantile_f64(&self, q: f64) -> PyResult<f64> {
        if !(0.0..=1.0).contains(&q) {
            return Err(PyValueError::new_err("quantile: q must be in [0, 1]"));
        }
        let mut v = non_nan(self.inner.data(), "quantile")?;
        if v.is_empty() {
            return Ok(f64::NAN);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(stats::sorted_quantile(&v, q))
    }

    /// The boolean mask for a `where` / `mask` condition, validating it matches
    /// this series' length (pandas requires equal-shape conditionals).
    pub(crate) fn cond_mask(&self, cond: &PySeries) -> PyResult<Vec<bool>> {
        if !matches!(cond.inner.data().view(), ColumnView::Bool(..)) {
            return Err(PyTypeError::new_err(format!(
                "where/mask: `cond` must be a boolean Series, got {}",
                cond.inner.data().dtype()
            )));
        }
        let c = bool_mask_vec(cond.inner.data())?;
        if c.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "Array conditional must be same shape as self ({} != {})",
                c.len(),
                self.inner.len()
            )));
        }
        // the condition pairs with `self` by position, so its labels must align
        require_aligned(self.inner.index(), cond.inner.index())?;
        Ok(c)
    }

    /// `where` (`invert = false`) / `mask` (`invert = true`) shared core: pick
    /// `self` where the (possibly inverted) condition holds, else `other`, in the
    /// promoted dtype. `mask` is `where(!cond)`.
    pub(crate) fn select_where(
        &self,
        cond: &PySeries,
        other: Option<&Bound<'_, PyAny>>,
        invert: bool,
    ) -> PyResult<PySeries> {
        let mut c = self.cond_mask(cond)?;
        if invert {
            c.iter_mut().for_each(|b| *b = !*b);
        }
        self.select_with(&c, other)
    }

    /// Keep `self` where `keep[i]` is true, else take the resolved `other` (the
    /// typed-scalar / Series / default-NA fill). The shared core of `where` /
    /// `mask` / `fillna`, so all three resolve a fill the same dtype-typed way
    /// (str scalar, parsed timestamp, NA-preserving) instead of funnelling to f64.
    pub(crate) fn select_with(&self, keep: &[bool], other: Option<&Bound<'_, PyAny>>) -> PyResult<PySeries> {
        // Lazy: if every cell is kept, no fill is applied, so the fill's type /
        // alignment is irrelevant — return self unchanged (dtype intact). This keeps
        // an all-keep where/mask (and a dense-column fillna) from type-checking a
        // fill it never uses, matching the per-column DataFrame surface.
        if keep.iter().all(|&b| b) {
            return Ok(PySeries { inner: self.inner.clone() });
        }
        let (other_col, other_dt) = where_other_resolve(other, &self.inner)?;
        // A float column keeps its float dtype (it absorbs any fill); a same-dtype
        // `other` (incl. the default NA, and bool/str/datetime) keeps that dtype so
        // it is not funneled to f64; a mixed int/float promotes by the supertype.
        let self_dt = self.inner.data().dtype();
        let target = if self_dt.is_float() || self_dt == other_dt {
            self_dt
        } else {
            binary_supertype(self_dt, other_dt)
        };
        col_to_series(
            &self.inner,
            self.inner.data().select(keep, &other_col, target).map_err(pyerr)?,
        )
    }

    /// Apply an element-wise `f64 -> f64` map, preserving name and index. The single
    /// guard for the whole Math Transform family: a `str` / `datetime` column would
    /// funnel through `to_f64_vec` to silent `NaN` (str) or `sin(epoch-as-f64)`
    /// (datetime), which the contract (C4) forbids, so it raises here.
    pub(crate) fn map_f64(&self, f: impl Fn(f64) -> f64) -> PyResult<PySeries> {
        self.inner.data().require_numeric().map_err(pyerr)?;
        let data = Column::f64(self.f64_domain("a math transform")?.iter().map(|&x| f(x)).collect());
        Ok(PySeries {
            inner: Series::new(self.inner.name().map(str::to_owned), data, Arc::clone(self.inner.index())).map_err(pyerr)?,
        })
    }

    /// Directional fill (`forward` = ffill, else bfill), dtype-aware over every
    /// dtype (int / bool / str / datetime / float). Shared by `ffill` / `bfill`.
    pub(crate) fn fill_dir(&self, forward: bool) -> PyResult<PySeries> {
        col_to_series(&self.inner, self.inner.data().fill_dir(forward))
    }
}

/// `series.iloc[...]` positional indexer.
#[pyclass]
pub struct SeriesILoc {
    pub(crate) inner: Series,
}

#[pymethods]
impl SeriesILoc {
    pub(crate) fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, self.inner.len())?;
            return np_scalar_to_py(py, self.inner.data(), i);
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            return Ok(Py::new(py, slice_series(&self.inner, slice)?)?.into_any());
        }
        Err(PyIndexError::new_err(
            "iloc key must be an integer or slice",
        ))
    }
}

/// A value-exact, hash-able key for a cell: `None` for a missing cell.
/// Numeric integers share keys across storage dtypes; other floats use bits.
/// Signed zeros share the integer zero key. Backs
/// value_counts / mode / isin / duplicated / replace.
pub(crate) fn cell_key(col: &Column, i: usize) -> Option<String> {
    if !col.is_valid(i) {
        return None;
    }
    // Integral floats share an integer key only inside the exact int64 domain.
    // Never cast an integer to float: adjacent large integers must stay distinct.
    let float = |x: f64| {
        if x.fract() == 0.0 && x >= i64::MIN as f64 && x < -(i64::MIN as f64) {
            format!("i{}", x as i64)
        } else {
            format!("f{:x}", x.to_bits())
        }
    };
    Some(match col.view() {
        ColumnView::F64(v) => float(v[i]),
        ColumnView::F32(v) => float(v[i] as f64),
        ColumnView::I64(v, _) => format!("i{}", v[i]),
        ColumnView::I32(v, _) => format!("i{}", v[i]),
        ColumnView::Bool(v, _) => format!("b{}", v[i]),
        ColumnView::Str(v, _) => format!("s{}", v.get(i)),
        ColumnView::Datetime(v) => format!("d{}", v[i]),
    })
}

/// Duplicate mask honoring `keep`: with `'first'`, later occurrences are True;
/// with `'last'`, earlier ones are. NA cells share one "missing" identity.
pub(crate) fn duplicated_mask_keep(col: &Column, keep: &str) -> PyResult<Vec<bool>> {
    let n = col.len();
    let mut seen: std::collections::HashSet<Option<String>> = std::collections::HashSet::new();
    match keep {
        "first" => Ok((0..n).map(|i| !seen.insert(cell_key(col, i))).collect()),
        "last" => {
            let mut out = vec![false; n];
            for i in (0..n).rev() {
                out[i] = !seen.insert(cell_key(col, i));
            }
            Ok(out)
        }
        other => Err(PyValueError::new_err(format!(
            "keep must be 'first' or 'last', got {other:?}"
        ))),
    }
}
