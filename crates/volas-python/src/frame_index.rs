//! The DataFrame indexing surface: the `.iloc` / `.loc` / `.iat` / `.at`
//! accessors, axis resolution, and assignment-value coercion.

use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PySliceIndices, PyTuple};
use volas_core::{
    Column, DType, DataFrame, Index, Label, Series,
};

#[allow(unused_imports)]
use crate::*;


/// `df.iloc[...]` positional indexer.
// --- indexer assignment helpers (PD-12) ------------------------------------

/// Whether `v` is a missing-value scalar: Python `None`, a `NaN` float, or the
/// `volas.NA` singleton. The one predicate every scalar boundary shares, so the
/// canonical `volas.NA` symbol (what `to_list()` returns) is usable wherever
/// `None` is — constructor, Series setitem, DataFrame indexers, mask assignment,
/// and `where` / `mask` `other`.
pub(crate) fn is_na_like_py(v: &Bound<'_, PyAny>) -> bool {
    let py = v.py();
    v.is_none() || v.is(na(py).bind(py)) || v.extract::<f64>().is_ok_and(|x| x.is_nan())
}

/// Build a length-1 [`Column`] from a Python scalar, coerced toward the target
/// column's dtype (so a string can land in a datetime column, etc.). An `I64`
/// target given a float yields an `F64` value — core then widens the column.
pub(crate) fn scalar_to_column(v: &Bound<'_, PyAny>, target: DType) -> PyResult<Column> {
    // `None` / `NaN` / `volas.NA` -> a typed single-cell NA (marks the position
    // missing while keeping the dtype), so `x[i] = None / nan / volas.NA` is
    // uniform across every dtype and every assignment surface.
    if is_na_like_py(v) {
        return Ok(Column::na_of(target, 1));
    }
    match target {
        DType::F64 => {
            let x = v
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("expected a number"))?;
            Ok(Column::f64(vec![x]))
        }
        DType::I64 => {
            if let Ok(i) = v.extract::<i64>() {
                Ok(Column::i64(vec![i]))
            } else {
                let x = v
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("expected a number"))?;
                Ok(Column::f64(vec![x]))
            }
        }
        DType::F32 => {
            let x = v
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("expected a number"))?;
            Ok(Column::f32(vec![x as f32]))
        }
        DType::I32 => match v.extract::<i64>() {
            Ok(i) => match i32::try_from(i) {
                Ok(v32) => Ok(Column::i32(vec![v32])),
                Err(_) => Ok(Column::i64(vec![i])), // out of i32 range -> i64 (core widens)
            },
            Err(_) => {
                let x = v
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("expected a number"))?;
                Ok(Column::f64(vec![x]))
            }
        },
        DType::Bool => {
            let b = v
                .extract::<bool>()
                .map_err(|_| PyTypeError::new_err("expected a bool"))?;
            Ok(Column::bool(vec![b]))
        }
        DType::Utf8 => {
            let s = v
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("expected a string"))?;
            Ok(Column::str(vec![s]))
        }
        DType::Datetime => Ok(Column::datetime(vec![parse_ts(v)?])),
    }
}

/// Assign a Python **scalar** into `col` at `positions`, via the shared
/// `scalar_to_column` + [`Column::scatter`] primitive — the single assignment
/// path behind Series setitem and DataFrame boolean-mask assignment (the
/// `.loc/.iloc/.at/.iat` indexers reach `scatter` through `assign_positions`).
///
/// A column with **no selected positions** is returned unchanged, so a typed fill
/// (a string into a str column, say) errors only when it actually targets a cell
/// of an incompatible column — the mixed-frame atomic rule: nothing is written
/// unless every targeted column accepts the value.
pub(crate) fn scatter_scalar(
    col: &Column,
    positions: &[usize],
    value: &Bound<'_, PyAny>,
) -> PyResult<Column> {
    if positions.is_empty() {
        return Ok(col.clone());
    }
    let src = scalar_to_column(value, col.dtype())?;
    col.scatter(positions, &src).map_err(pyerr)
}

/// Convert an array-like assignment value (list / NumPy array / `Series`) to a
/// [`Column`]; core coerces it toward the target dtype.
pub(crate) fn value_to_column(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(s) = v.extract::<PyRef<PySeries>>() {
        return Ok(s.inner.data.clone());
    }
    pyany_to_column(v)
}

/// Resolve a `df[...] = value` right-hand side for `n` selected rows: a scalar is
/// broadcast (length-1 column), an array-like must match `n`.
pub(crate) fn resolve_assignment(v: &Bound<'_, PyAny>, target: DType, n: usize) -> PyResult<Column> {
    // A Python str has `__len__` but is a scalar here; everything else with
    // `__len__` (list / ndarray / Series) is array-like.
    let is_str = v.extract::<String>().is_ok();
    let arraylike = !is_str && v.hasattr("__len__").unwrap_or(false);
    if arraylike {
        let col = value_to_column(v)?;
        if col.len() != n {
            return Err(PyValueError::new_err(format!(
                "cannot assign {} values to {n} selected rows",
                col.len()
            )));
        }
        Ok(col)
    } else {
        scalar_to_column(v, target)
    }
}

/// If `sel` is a boolean mask of length `height` (NumPy bool array, bool `Series`,
/// or `list[bool]`), return the selected row positions; else `None`.
pub(crate) fn as_bool_mask(sel: &Bound<'_, PyAny>, height: usize) -> Option<Vec<usize>> {
    let collect = |bits: &[bool]| -> Option<Vec<usize>> {
        (bits.len() == height).then(|| {
            bits.iter()
                .enumerate()
                .filter_map(|(i, &b)| b.then_some(i))
                .collect()
        })
    };
    if let Ok(a) = sel.extract::<PyReadonlyArray1<bool>>() {
        return a.as_slice().ok().and_then(collect);
    }
    if let Ok(ser) = sel.extract::<PyRef<PySeries>>() {
        if let Column::Bool(v, _) = &ser.inner.data {
            return collect(v);
        }
        return None;
    }
    if let Ok(v) = sel.extract::<Vec<bool>>() {
        return collect(&v);
    }
    None
}

/// Resolve an `iloc` row selector (int / slice / int-list / bool-mask) to row
/// positions.
pub(crate) fn iloc_positions(sel: &Bound<'_, PyAny>, height: usize) -> PyResult<Vec<usize>> {
    if let Some(pos) = as_bool_mask(sel, height) {
        return Ok(pos);
    }
    if let Ok(i) = sel.extract::<isize>() {
        return Ok(vec![norm_idx(i, height)?]);
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let info = slice.indices(height as isize)?;
        return Ok(strided(info.start, info.stop, info.step));
    }
    if let Ok(idxs) = sel.extract::<Vec<isize>>() {
        return idxs.into_iter().map(|i| norm_idx(i, height)).collect();
    }
    Err(PyTypeError::new_err(
        "iloc row selector must be an int, slice, int list, or boolean mask",
    ))
}

/// Resolve a `loc` row selector (bool-mask / label-slice / label / label-list) to
/// row positions.
pub(crate) fn loc_positions(sel: &Bound<'_, PyAny>, index: &Index, height: usize) -> PyResult<Vec<usize>> {
    if let Some(pos) = as_bool_mask(sel, height) {
        return Ok(pos);
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let (lo, hi) = label_bounds(slice, index)?;
        let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
        return Ok((a..b).collect());
    }
    if let Ok(list) = sel.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            let label = parse_label(&item, index)?;
            out.push(
                index
                    .position_of(&label)
                    .ok_or_else(|| PyKeyError::new_err("label not found"))?,
            );
        }
        return Ok(out);
    }
    let label = parse_label(sel, index)?;
    let pos = index
        .position_of(&label)
        .ok_or_else(|| PyKeyError::new_err("label not found"))?;
    Ok(vec![pos])
}

/// One axis of a 2-D `iloc` / `loc` get: a single scalar position (the axis is
/// reduced away, pandas-style) or a list of positions (the axis is kept).
pub(crate) enum AxisSel {
    One(usize),
    Many(Vec<usize>),
}

/// Resolve an `iloc` row axis: a bare int reduces the axis (`AxisSel::One`); a
/// slice / int-list / boolean mask keeps it (`AxisSel::Many`).
pub(crate) fn iloc_row_axis(sel: &Bound<'_, PyAny>, height: usize) -> PyResult<AxisSel> {
    if let Ok(i) = sel.extract::<isize>() {
        return Ok(AxisSel::One(norm_idx(i, height)?));
    }
    Ok(AxisSel::Many(iloc_positions(sel, height)?))
}

/// Resolve an `iloc` column axis (int / slice / int-list) to column positions.
pub(crate) fn iloc_col_axis(sel: &Bound<'_, PyAny>, width: usize) -> PyResult<AxisSel> {
    if let Ok(j) = sel.extract::<isize>() {
        return Ok(AxisSel::One(norm_idx(j, width)?));
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let info = slice.indices(width as isize)?;
        return Ok(AxisSel::Many(strided(info.start, info.stop, info.step)));
    }
    if let Ok(idxs) = sel.extract::<Vec<isize>>() {
        return Ok(AxisSel::Many(
            idxs.into_iter()
                .map(|j| norm_idx(j, width))
                .collect::<PyResult<_>>()?,
        ));
    }
    Err(PyTypeError::new_err(
        "iloc column selector must be an int, slice, or int list",
    ))
}

/// Resolve a `loc` row axis: a single label reduces the axis (`AxisSel::One`); a
/// label-slice / label-list / boolean mask keeps it (`AxisSel::Many`).
pub(crate) fn loc_row_axis(sel: &Bound<'_, PyAny>, index: &Index, height: usize) -> PyResult<AxisSel> {
    if as_bool_mask(sel, height).is_some()
        || sel.downcast::<PySlice>().is_ok()
        || sel.downcast::<PyList>().is_ok()
    {
        return Ok(AxisSel::Many(loc_positions(sel, index, height)?));
    }
    let label = parse_label(sel, index)?;
    let pos = index
        .position_of(&label)
        .ok_or_else(|| PyKeyError::new_err("label not found"))?;
    Ok(AxisSel::One(pos))
}

/// Resolve a `loc` column axis (name / name-list / inclusive name-slice) to
/// column positions.
pub(crate) fn loc_col_axis(sel: &Bound<'_, PyAny>, df: &DataFrame) -> PyResult<AxisSel> {
    let pos_of = |name: &str| {
        df.column_pos(name)
            .ok_or_else(|| PyKeyError::new_err(format!("column {name:?} not found")))
    };
    if let Ok(name) = sel.extract::<String>() {
        return Ok(AxisSel::One(pos_of(&name)?));
    }
    if let Ok(names) = sel.extract::<Vec<String>>() {
        return Ok(AxisSel::Many(
            names.iter().map(|n| pos_of(n)).collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let start = slice.getattr("start")?;
        let stop = slice.getattr("stop")?;
        let lo = if start.is_none() {
            0
        } else {
            pos_of(&start.extract::<String>()?)?
        };
        let hi = if stop.is_none() {
            df.width().saturating_sub(1)
        } else {
            pos_of(&stop.extract::<String>()?)?
        };
        return Ok(AxisSel::Many((lo..=hi).collect()));
    }
    Err(PyTypeError::new_err(
        "loc column selector must be a name, name list, or name slice",
    ))
}

/// Project `df` onto `cols` (by position), carrying the index — the column-axis
/// counterpart of `DataFrame::take` (which selects rows).
pub(crate) fn project_cols(df: &DataFrame, cols: &[usize]) -> PyResult<DataFrame> {
    let names: Vec<String> = cols.iter().map(|&j| df.names()[j].clone()).collect();
    let data: Vec<Column> = cols.iter().map(|&j| df.columns()[j].clone()).collect();
    let idx = (*df.index().as_ref()).clone();
    DataFrame::new(names, data, Some(idx)).map_err(pyerr)
}

/// Build a 2-D `iloc` / `loc` get result from already-resolved row & column
/// positions, reproducing pandas's shape rules: scalar×scalar -> cell,
/// rows×col -> a column Series, row×cols -> the row (volas's 1-row frame), and
/// rows×cols -> a sub-frame.
pub(crate) fn select_2d(py: Python<'_>, df: &DataFrame, rows: AxisSel, cols: AxisSel) -> PyResult<Py<PyAny>> {
    match (rows, cols) {
        (AxisSel::One(i), AxisSel::One(j)) => Ok(np_scalar_to_py(py, &df.columns()[j], i)),
        (AxisSel::Many(r), AxisSel::One(j)) => {
            let sub = project_cols(df, &[j])?.take(&r);
            let name = sub.names()[0].clone();
            let col = sub.columns()[0].clone();
            let series = PySeries {
                inner: Series::new(Some(name), col, Arc::clone(sub.index())),
            };
            Ok(Py::new(py, series)?.into_any())
        }
        (AxisSel::One(i), AxisSel::Many(c)) => {
            let sub = project_cols(df, &c)?.take(&[i]);
            Ok(Py::new(py, PyRow { inner: sub })?.into_any())
        }
        (AxisSel::Many(r), AxisSel::Many(c)) => {
            let sub = project_cols(df, &c)?.take(&r);
            Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any())
        }
    }
}

/// Split a `df.loc[rows, col]` / `df.iloc[rows, col]` assignment key into its two
/// parts, with a clear error directing to the supported 2-tuple form.
pub(crate) fn split_row_col<'py>(
    key: &Bound<'py, PyAny>,
    accessor: &str,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let tup = key.downcast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{accessor} assignment needs a (rows, column) key, e.g. df.{accessor}[mask, 'col'] = value"
        ))
    })?;
    if tup.len() != 2 {
        return Err(PyTypeError::new_err(format!(
            "{accessor} assignment key must be (rows, column)"
        )));
    }
    Ok((tup.get_item(0)?, tup.get_item(1)?))
}

#[pyclass]
pub struct DataFrameILoc {
    pub(crate) parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameILoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        // 2-D positional get: df.iloc[rows, cols], symmetric with __setitem__.
        if let Ok(tup) = key.downcast::<PyTuple>() {
            if tup.len() == 2 {
                let rows = iloc_row_axis(&tup.get_item(0)?, pf.inner.height())?;
                let cols = iloc_col_axis(&tup.get_item(1)?, pf.inner.width())?;
                return select_2d(py, &pf.inner, rows, cols);
            }
        }
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, pf.inner.height())?;
            return Ok(Py::new(py, row_at(&pf.inner, i))?.into_any());
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let info = slice.indices(pf.inner.height() as isize)?;
            let sub = positional_slice(&pf.inner, &info);
            return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
        }
        // int-list / boolean-mask row selection -> sub-frame.
        let positions = iloc_positions(key, pf.inner.height())?;
        Ok(Py::new(py, PyDataFrame::plain(take_frame(&pf.inner, &positions)))?.into_any())
    }

    /// `df.iloc[i, j] = scalar` or `df.iloc[rows, j] = scalar | array` (positional;
    /// copy-on-write). `rows` is an int / slice / int-list / boolean mask; `j` is a
    /// column position.
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let (rows, col) = split_row_col(key, "iloc")?;
        let height = pf.inner.height();
        let j = norm_idx(col.extract::<isize>()?, pf.inner.width())?;
        let positions = iloc_positions(&rows, height)?;
        let target = pf.inner.columns()[j].dtype();
        let val = resolve_assignment(value, target, positions.len())?;
        pf.inner
            .assign_positions(j, &positions, &val)
            .map_err(pyerr)
    }
}

pub(crate) fn row_at(df: &DataFrame, i: usize) -> PyRow {
    // `take` materializes the index label (Range -> Int64([i])) and preserves
    // every column's dtype — a faithful 1-row frame.
    PyRow {
        inner: df.take(&[i]),
    }
}

pub(crate) fn take_frame(df: &DataFrame, positions: &[usize]) -> DataFrame {
    // Delegates to core `take`, which carries column aliases onto the new frame.
    df.take(positions)
}

/// A positional slice: a contiguous `step == 1` slice uses `DataFrame::slice` (a
/// contiguous copy); a strided slice gathers the individual positions.
pub(crate) fn positional_slice(df: &DataFrame, info: &PySliceIndices) -> DataFrame {
    if info.step == 1 {
        df.slice(info.start.max(0) as usize, info.stop.max(0) as usize)
    } else {
        take_frame(df, &strided(info.start, info.stop, info.step))
    }
}

/// Slice a frame by a Python slice — positional for integer bounds, label-based
/// (DatetimeIndex) for string bounds.
pub(crate) fn slice_frame(df: &DataFrame, slice: &Bound<'_, PySlice>) -> PyResult<DataFrame> {
    let start_obj = slice.getattr("start")?;
    let stop_obj = slice.getattr("stop")?;
    let is_label = start_obj.extract::<String>().is_ok() || stop_obj.extract::<String>().is_ok();
    if is_label {
        let index = df.index();
        let lo = if start_obj.is_none() {
            None
        } else {
            Some(parse_label(&start_obj, index)?)
        };
        let hi = if stop_obj.is_none() {
            None
        } else {
            Some(parse_label(&stop_obj, index)?)
        };
        let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
        Ok(df.slice(a, b))
    } else {
        let info = slice.indices(df.height() as isize)?;
        Ok(positional_slice(df, &info))
    }
}

pub(crate) fn label_bounds(
    slice: &Bound<'_, PySlice>,
    index: &Index,
) -> PyResult<(Option<Label>, Option<Label>)> {
    let start_obj = slice.getattr("start")?;
    let stop_obj = slice.getattr("stop")?;
    let lo = if start_obj.is_none() {
        None
    } else {
        Some(parse_label(&start_obj, index)?)
    };
    let hi = if stop_obj.is_none() {
        None
    } else {
        Some(parse_label(&stop_obj, index)?)
    };
    Ok((lo, hi))
}

/// `df.loc[...]` label indexer.
#[pyclass]
pub struct DataFrameLoc {
    pub(crate) parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameLoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        let index = pf.inner.index();
        // 2-D label get: df.loc[rows, col], symmetric with __setitem__.
        if let Ok(tup) = key.downcast::<PyTuple>() {
            if tup.len() == 2 {
                let rows = loc_row_axis(&tup.get_item(0)?, index, pf.inner.height())?;
                let cols = loc_col_axis(&tup.get_item(1)?, &pf.inner)?;
                return select_2d(py, &pf.inner, rows, cols);
            }
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (lo, hi) = label_bounds(slice, index)?;
            let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
            return Ok(Py::new(py, PyDataFrame::plain(pf.inner.slice(a, b)))?.into_any());
        }
        // boolean-mask / label-list row selection -> sub-frame.
        if as_bool_mask(key, pf.inner.height()).is_some() || key.downcast::<PyList>().is_ok() {
            let positions = loc_positions(key, index, pf.inner.height())?;
            return Ok(
                Py::new(py, PyDataFrame::plain(take_frame(&pf.inner, &positions)))?.into_any(),
            );
        }
        let label = parse_label(key, index)?;
        let pos = index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(Py::new(py, row_at(&pf.inner, pos))?.into_any())
    }

    /// `df.loc[rows, col] = scalar | array` (label-based; copy-on-write). `rows` is
    /// a boolean mask, a label slice, a single label, or a label list; `col` is a
    /// single column name. The classic `df.loc[mask, 'signal'] = 1`.
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let (rows, col) = split_row_col(key, "loc")?;
        let colname: String = col.extract().map_err(|_| {
            PyTypeError::new_err("loc assignment column must be a single column name")
        })?;
        let height = pf.inner.height();
        let positions = {
            let index = pf.inner.index();
            loc_positions(&rows, index, height)?
        };
        let j = pf
            .inner
            .column_pos(&colname)
            .ok_or_else(|| PyKeyError::new_err(format!("column {colname:?} not found")))?;
        let target = pf.inner.columns()[j].dtype();
        let val = resolve_assignment(value, target, positions.len())?;
        pf.inner
            .assign_positions(j, &positions, &val)
            .map_err(pyerr)
    }
}

/// `df.iat[i, j]` scalar access by position.
#[pyclass]
pub struct DataFrameIat {
    pub(crate) parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameIat {
    fn __getitem__(&self, py: Python<'_>, key: (isize, isize)) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        let i = norm_idx(key.0, pf.inner.height())?;
        let j = norm_idx(key.1, pf.inner.width())?;
        Ok(np_scalar_to_py(py, &pf.inner.columns()[j], i))
    }

    /// `df.iat[i, j] = scalar` — set a single cell by position (copy-on-write).
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: (isize, isize),
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let i = norm_idx(key.0, pf.inner.height())?;
        let j = norm_idx(key.1, pf.inner.width())?;
        let target = pf.inner.columns()[j].dtype();
        let val = scalar_to_column(value, target)?;
        pf.inner.assign_positions(j, &[i], &val).map_err(pyerr)
    }
}

/// `df.at[label, col]` scalar access by label + column name.
#[pyclass]
pub struct DataFrameAt {
    pub(crate) parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameAt {
    fn __getitem__(&self, py: Python<'_>, key: (Py<PyAny>, String)) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        let index = pf.inner.index();
        let label = parse_label(key.0.bind(py), index)?;
        let i = index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        let col = pf.inner.column(&key.1).map_err(pyerr)?;
        Ok(np_scalar_to_py(py, col, i))
    }

    /// `df.at[label, col] = scalar` — set a single cell by label + column name
    /// (copy-on-write).
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: (Py<PyAny>, String),
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let i = {
            let index = pf.inner.index();
            let label = parse_label(key.0.bind(py), index)?;
            index
                .position_of(&label)
                .ok_or_else(|| PyKeyError::new_err("label not found"))?
        };
        let j = pf
            .inner
            .column_pos(&key.1)
            .ok_or_else(|| PyKeyError::new_err(format!("column {:?} not found", key.1)))?;
        let target = pf.inner.columns()[j].dtype();
        let val = scalar_to_column(value, target)?;
        pf.inner.assign_positions(j, &[i], &val).map_err(pyerr)
    }
}

/// `series.loc[...]` label indexer.
#[pyclass]
pub struct SeriesLoc {
    pub(crate) inner: Series,
}

#[pymethods]
impl SeriesLoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (lo, hi) = label_bounds(slice, &self.inner.index)?;
            let (a, b) = self.inner.index.label_slice(lo.as_ref(), hi.as_ref());
            let positions: Vec<usize> = (a..b).collect();
            let data = self.inner.data.take(&positions);
            let index = Arc::new(self.inner.index.take(&positions));
            return Ok(Py::new(
                py,
                PySeries {
                    inner: Series::new(self.inner.name.clone(), data, index),
                },
            )?
            .into_any());
        }
        let label = parse_label(key, &self.inner.index)?;
        let pos = self
            .inner
            .index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(np_scalar_to_py(py, &self.inner.data, pos))
    }
}
