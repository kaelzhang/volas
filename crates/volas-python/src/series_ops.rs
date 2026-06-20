//! `Series` operators (arithmetic / comparison / logical) and element access
//! (`__getitem__` / `__setitem__`).


use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use volas_core::{
    BinOp, BoolOp, CmpOp,
};

#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PySeries {

    pub(crate) fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Add, false)
    }
    pub(crate) fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Sub, false)
    }
    pub(crate) fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Mul, false)
    }
    pub(crate) fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_div(&self.inner, other, false)
    }
    pub(crate) fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Add, true)
    }
    pub(crate) fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Sub, true)
    }
    pub(crate) fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Mul, true)
    }
    pub(crate) fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_div(&self.inner, other, true)
    }
    pub(crate) fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_floordiv(&self.inner, other, false)
    }
    pub(crate) fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_floordiv(&self.inner, other, true)
    }

    // Element-wise comparisons -> bool Series (pandas-style), dtype-aware.
    pub(crate) fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Lt)
    }
    pub(crate) fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Le)
    }
    pub(crate) fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Eq)
    }
    pub(crate) fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Ne)
    }
    pub(crate) fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Ge)
    }
    pub(crate) fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Gt)
    }

    // Element-wise boolean logic -> bool Series (operands coerced to bool).
    pub(crate) fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::And)
    }
    pub(crate) fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Or)
    }
    pub(crate) fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Xor)
    }
    pub(crate) fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::And)
    }
    pub(crate) fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Or)
    }
    pub(crate) fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Xor)
    }
    pub(crate) fn __invert__(&self) -> PySeries {
        col_to_series(&self.inner, self.inner.data.not())
    }

    /// `series[key]`: an integer position, a datetime label, or a slice.
    pub(crate) fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // boolean mask -> the True rows, as a new Series (pandas `s[bool_mask]`)
        if let Some(mask) = bool_mask_key(key)? {
            let sub = self.inner.filter_mask(&mask).map_err(pyerr)?;
            return Ok(Py::new(py, PySeries { inner: sub })?.into_any());
        }
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, self.inner.len())?;
            return Ok(np_scalar_to_py(py, &self.inner.data, i));
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            return Ok(Py::new(py, slice_series(&self.inner, slice)?)?.into_any());
        }
        // label lookup
        let label = parse_label(key, &self.inner.index)?;
        let pos = self
            .inner
            .index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(np_scalar_to_py(py, &self.inner.data, pos))
    }

    /// In-place assignment by boolean mask (`s[mask] = v`) or integer position
    /// (`s[i] = v`). Follows pandas 3.0 dtype rules: the column dtype is kept when
    /// the value fits (an integral number stays in an int series), `NaN` upcasts
    /// an int series to float, and a lossy write (e.g. `2.5` into an int series)
    /// raises `TypeError`.
    pub(crate) fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.len();
        let positions: Vec<usize> = if let Some(mask) = bool_mask_key(key)? {
            if mask.len() != n {
                return Err(PyValueError::new_err(format!(
                    "boolean mask length {} != series length {n}",
                    mask.len()
                )));
            }
            mask.iter()
                .enumerate()
                .filter_map(|(i, &m)| m.then_some(i))
                .collect()
        } else if let Ok(i) = key.extract::<isize>() {
            vec![norm_idx(i, n)?]
        } else {
            return Err(PyTypeError::new_err(
                "Series assignment takes a boolean mask or an integer position",
            ));
        };
        // One assignment path for every value kind (number, bool, string, datetime
        // string, None/NaN): convert to a typed single-cell column for this dtype
        // and scatter it — identical rules to the DataFrame indexers and mask
        // assignment (keep dtype, update validity, lossy values error).
        self.inner.data = scatter_scalar(&self.inner.data, &positions, value)?;
        Ok(())
    }
}
