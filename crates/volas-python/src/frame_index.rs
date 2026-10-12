//! The DataFrame indexing surface: the `.iloc` / `.loc` / `.iat` / `.at`
//! accessors, axis resolution, and assignment-value coercion.

use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyKeyError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PySliceIndices, PyString, PyTuple};
use volas_core::{
    Column, DType, DataFrame, Index, IndexKind, Label, Series, SliceBound,
};

#[allow(unused_imports)]
use crate::*;

// --- indexer assignment helpers (PD-12) ------------------------------------

/// Whether `v` is a missing-value scalar: Python `None`, a `NaN` float, or the
/// `volas.NA` singleton. The one predicate every scalar boundary shares, so the
/// canonical `volas.NA` symbol (what `to_list()` returns) is usable wherever
/// `None` is — constructor, Series setitem, DataFrame indexers, mask assignment,
/// and `where` / `mask` `other`.
pub(crate) fn is_na_like_py(v: &Bound<'_, PyAny>) -> bool {
    let py = v.py();
    v.is_none()
        || v.is(na(py).bind(py))
        || v.extract::<f64>().is_ok_and(|x| x.is_nan())
        // pd.NA (pandas's NAType) is a legitimate missing-value synonym at the
        // boundary (F42) — volas's own NAType is already covered by `is(na)`.
        || v.get_type().name().map(|n| n == "NAType").unwrap_or(false)
}

/// Whether an assigned value is an array of values rather than one value — the one
/// classification every assignment surface shares: a list, a tuple, a `Series`, or
/// anything with one or more dimensions (an `ndarray`, a pandas object) is an array;
/// a Python / NumPy scalar and a 0-d array are one value. Never decided by whether
/// the value converts to a number (NumPy converts a one-element array, deprecated).
pub(crate) fn is_array(v: &Bound<'_, PyAny>) -> bool {
    use pyo3::types::{PyBool, PyFloat, PyInt};
    if v.is_exact_instance_of::<PyFloat>()
        || v.is_exact_instance_of::<PyInt>()
        || v.is_exact_instance_of::<PyBool>()
        || v.is_instance_of::<PyString>()
        || v.is_none()
    {
        return false; // the live bar cells: no attribute lookup
    }
    if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() || v.extract::<PyRef<PySeries>>().is_ok() {
        return true;
    }
    v.getattr("ndim").ok().and_then(|n| n.extract::<usize>().ok()).is_some_and(|n| n >= 1)
}

/// Build a length-1 [`Column`] of EXACTLY the `target` dtype from a Python scalar
/// — the one scalar-write conversion behind every target-typed write (`iat` /
/// `iloc` / `at` / `loc` / Series setitem / mask assignment / dict `append`). A
/// missing scalar becomes the target's NA; a number converts through the core
/// lossless owner ([`Column::convert_lossless`], contract #20): exact converts,
/// inexact raises (`2**53 + 1` into float64, `2.5` into int64, `1e300` into
/// float32), never a silent round or saturation. A string lands in a str / datetime
/// column; a bool column takes a bool.
pub(crate) fn scalar_to_column(v: &Bound<'_, PyAny>, target: DType) -> PyResult<Column> {
    // `None` / `NaN` / `volas.NA` -> a typed single-cell NA (marks the position
    // missing while keeping the dtype), so `x[i] = None / nan / volas.NA` is
    // uniform across every dtype and every assignment surface.
    if is_na_like_py(v) {
        return Ok(Column::na_of(target, 1));
    }
    if is_array(v) {
        return Err(PyTypeError::new_err(format!(
            "a cell takes one value, not an array ({}) — pass the element itself",
            v.get_type().name()?
        )));
    }
    match target {
        DType::F64 | DType::F32 | DType::I64 | DType::I32 => {
            number_column(v, target)?.convert_lossless(target).map_err(pyerr)
        }
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

/// A Python number as its natural 1-cell column, before any conversion: a bool
/// -> `Bool`, an integer (a Python `int` or a NumPy integer, via `__index__`) ->
/// `I64`, a float -> `F64`. An integer past int64 has no column of its own: it
/// may still land in a float `target` when float64 holds it exactly (and float32
/// too, for a float32 target), checked by Python's exact int / float comparison;
/// otherwise it raises — never a rounded or saturated value (contract #20).
fn number_column(v: &Bound<'_, PyAny>, target: DType) -> PyResult<Column> {
    // fast path for the live bar-append cells: a Python float is a float64 as-is
    if let Ok(f) = v.downcast_exact::<pyo3::types::PyFloat>() {
        return Ok(Column::f64(vec![f.value()]));
    }
    if v.is_exact_instance_of::<pyo3::types::PyInt>() {
        if let Ok(i) = v.extract::<i64>() {
            return Ok(Column::i64(vec![i]));
        }
    }
    if let Ok(b) = v.extract::<bool>() {
        return Ok(Column::bool(vec![b])); // a Python or NumPy bool
    }
    if let Ok(idx) = v.call_method0("__index__") {
        if let Ok(i) = idx.extract::<i64>() {
            return Ok(Column::i64(vec![i]));
        }
        if !target.is_float() {
            return Err(PyOverflowError::new_err(format!("{idx} is out of the {target} range")));
        }
        let f = idx
            .extract::<f64>()
            .map_err(|_| PyOverflowError::new_err(format!("{idx} is out of the {target} range")))?;
        let narrow_ok = target == DType::F64 || (f as f32) as f64 == f;
        if !(narrow_ok && idx.eq(f)?) {
            return Err(PyTypeError::new_err(format!(
                "{idx} is not exactly representable as {target}"
            )));
        }
        return Ok(Column::f64(vec![f]));
    }
    v.extract::<f64>()
        .map(|x| Column::f64(vec![x]))
        .map_err(|_| PyTypeError::new_err("expected a number"))
}

/// Assign a Python **scalar** into `col` at `positions`, via the shared
/// `scalar_to_column` + [`Column::scatter`] primitive — the Series setitem path (a
/// DataFrame's cell writes reach `scatter` through `PyDataFrame::write_cells`).
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
    if is_array(v) {
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
pub(crate) fn as_bool_mask(
    sel: &Bound<'_, PyAny>,
    height: usize,
) -> PyResult<Option<Vec<usize>>> {
    let collect = |bits: &[bool]| -> Option<Vec<usize>> {
        (bits.len() == height).then(|| {
            bits.iter()
                .enumerate()
                .filter_map(|(i, &b)| b.then_some(i))
                .collect()
        })
    };
    if let Ok(a) = sel.extract::<PyReadonlyArray1<bool>>() {
        return Ok(a.as_slice().ok().and_then(collect));
    }
    if let Ok(ser) = sel.extract::<PyRef<PySeries>>() {
        if let Column::Bool(..) = &ser.inner.data {
            // O5: a bool Series mask carrying any volas.NA is rejected — an unknown
            // signal must not silently filter like a deliberate False.
            return Ok(collect(&bool_mask_vec(&ser.inner.data)?));
        }
        return Ok(None);
    }
    if let Ok(v) = sel.extract::<Vec<bool>>() {
        return Ok(collect(&v));
    }
    Ok(None)
}

/// Resolve an `iloc` row selector (int / slice / int-list / bool-mask) to row
/// positions.
pub(crate) fn iloc_positions(sel: &Bound<'_, PyAny>, height: usize) -> PyResult<Vec<usize>> {
    if let Some(pos) = as_bool_mask(sel, height)? {
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
    if let Some(pos) = as_bool_mask(sel, height)? {
        return Ok(pos);
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let (lo, hi) = label_bounds(slice, index)?;
        let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref()).map_err(pyerr)?;
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
    if as_bool_mask(sel, height)?.is_some()
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
/// positions of the visible rows of `pf` (`view`), reproducing pandas's shape rules:
/// scalar×scalar -> cell, rows×col -> a column Series, row×cols -> the row (volas's
/// 1-row frame) — reads of values — and rows×cols -> a sub-frame: the derived frame of
/// those rows ([`PyDataFrame::derive_rows`]) projected onto those columns. Rows that are
/// not contiguous are a new series, which carries no cached column: naming one raises.
pub(crate) fn select_2d(
    py: Python<'_>,
    pf: &PyDataFrame,
    view: &Fresh<'_>,
    rows: AxisSel,
    cols: AxisSel,
) -> PyResult<Py<PyAny>> {
    let df: &DataFrame = view;
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
            let names: Vec<String> = c.iter().map(|&j| df.names()[j].clone()).collect();
            let sel = RowSel::Positions(r);
            let rows = pf.derive_rows(view, &sel)?;
            if let Some(cached) = names.iter().find(|n| df.is_computed(n) && !rows.inner.has_column(n)) {
                return Err(uncarried_cached(cached));
            }
            if names.as_slice() == rows.inner.names() {
                return Ok(Py::new(py, rows)?.into_any());
            }
            let inner = rows.inner.select(&names).map_err(pyerr)?;
            Ok(Py::new(py, rows.derive_projection(inner)?)?.into_any())
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
        // Position the read against the logical M view (zero-cost borrow when
        // unbounded), so `iloc[i]` indexes the visible window, never the margin.
        let view = pf.read()?;
        let df = view.as_ref();
        // 2-D positional get: df.iloc[rows, cols], symmetric with __setitem__.
        if let Ok(tup) = key.downcast::<PyTuple>() {
            if tup.len() == 2 {
                let rows = iloc_row_axis(&tup.get_item(0)?, df.height())?;
                let cols = iloc_col_axis(&tup.get_item(1)?, df.width())?;
                return select_2d(py, &pf, &view, rows, cols);
            }
        }
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, df.height())?;
            return Ok(Py::new(py, row_at(df, i))?.into_any());
        }
        let sel = match key.downcast::<PySlice>() {
            Ok(slice) => positional_slice(&slice.indices(df.height() as isize)?),
            // int-list / boolean-mask row selection
            Err(_) => RowSel::Positions(iloc_positions(key, df.height())?),
        };
        Ok(Py::new(py, pf.derive_rows(&view, &sel)?)?.into_any())
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
        let (rows, col) = split_row_col(key, "iloc")?;
        // Positions of the visible rows; the write path maps them onto the frame.
        let len = pf.meta().range().1;
        let j = norm_idx(col.extract::<isize>()?, pf.inner.width())?;
        let positions = iloc_positions(&rows, len)?;
        let target = pf.inner.columns()[j].dtype();
        let val = resolve_assignment(value, target, positions.len())?;
        pf.write_cells(vec![(j, positions, val)])
    }
}

pub(crate) fn row_at(df: &DataFrame, i: usize) -> PyRow {
    // `take` materializes the index label (Range -> Int64([i])) and preserves
    // every column's dtype — a faithful 1-row frame.
    PyRow {
        inner: df.take(&[i]),
    }
}

/// A positional slice of the visible rows: a contiguous `step == 1` slice is a range;
/// a strided one the individual positions.
pub(crate) fn positional_slice(info: &PySliceIndices) -> RowSel {
    if info.step == 1 {
        RowSel::Range(info.start.max(0) as usize, info.stop.max(0) as usize)
    } else {
        RowSel::Positions(strided(info.start, info.stop, info.step))
    }
}

/// The rows a Python slice selects — label-based when either bound is a string
/// (`df['a':'b']`), positional otherwise.
pub(crate) fn slice_rows(df: &DataFrame, slice: &Bound<'_, PySlice>) -> PyResult<RowSel> {
    let is_text = |o: &Bound<'_, PyAny>| o.is_instance_of::<PyString>();
    if is_text(&slice.getattr("start")?) || is_text(&slice.getattr("stop")?) {
        label_rows(df.index(), slice)
    } else {
        Ok(positional_slice(&slice.indices(df.height() as isize)?))
    }
}

/// The rows a label slice (`df.loc[lo:hi]`, both bounds inclusive) selects.
pub(crate) fn label_rows(index: &Index, slice: &Bound<'_, PySlice>) -> PyResult<RowSel> {
    let (lo, hi) = label_bounds(slice, index)?;
    let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref()).map_err(pyerr)?;
    Ok(RowSel::Range(a, b))
}

/// The two bounds of a label slice (`None` for an open end), each tagged with
/// whether it was written as a string — the one place a Python slice bound
/// becomes a [`SliceBound`]. A bound of the other label kind (a string against a
/// numeric index, an integer against a string index) is kept as written, so
/// [`Index::label_slice`] rejects it with a `TypeError`.
///
/// A string bound on a `DatetimeIndex` written as a partial date (`'2024-01'`,
/// `'2024-01-02 10'`, …) names a period, as pandas reads it: a start bound is the
/// period's first instant and a stop bound its last, so `df.loc['2024-01':'2024-02']`
/// holds both whole months ([`Tz::period_of_text`](volas_core::Tz::period_of_text)).
pub(crate) fn label_bounds(
    slice: &Bound<'_, PySlice>,
    index: &Index,
) -> PyResult<(Option<SliceBound>, Option<SliceBound>)> {
    let bound = |obj: Bound<'_, PyAny>, stop: bool| -> PyResult<Option<SliceBound>> {
        if obj.is_none() {
            return Ok(None);
        }
        let text = obj.is_instance_of::<PyString>();
        let label = match index.kind() {
            IndexKind::Range { .. } | IndexKind::Int64(_) if text => Label::Str(obj.extract()?),
            IndexKind::Str(_) if !text => match obj.extract::<i64>() {
                Ok(x) => Label::I64(x),
                Err(_) => parse_label(&obj, index)?,
            },
            IndexKind::Datetime(_, tz) if text => match tz.period_of_text(&obj.extract::<String>()?) {
                Some((start, end, _)) => Label::I64(if stop { end - 1 } else { start }),
                None => parse_label(&obj, index)?,
            },
            _ => parse_label(&obj, index)?,
        };
        Ok(Some(if text { SliceBound::text(label) } else { SliceBound::exact(label) }))
    };
    Ok((bound(slice.getattr("start")?, false)?, bound(slice.getattr("stop")?, true)?))
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
        // Resolve labels against the logical M view (zero-cost borrow when
        // unbounded), so a margin label is never addressable through `loc`.
        let view = pf.read()?;
        let df = view.as_ref();
        let index = df.index();
        // 2-D label get: df.loc[rows, col], symmetric with __setitem__.
        if let Ok(tup) = key.downcast::<PyTuple>() {
            if tup.len() == 2 {
                let rows = loc_row_axis(&tup.get_item(0)?, index, df.height())?;
                let cols = loc_col_axis(&tup.get_item(1)?, df)?;
                return select_2d(py, &pf, &view, rows, cols);
            }
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let sel = label_rows(index, slice)?;
            return Ok(Py::new(py, pf.derive_rows(&view, &sel)?)?.into_any());
        }
        // boolean-mask / label-list row selection -> sub-frame.
        if as_bool_mask(key, df.height())?.is_some() || key.downcast::<PyList>().is_ok() {
            let sel = RowSel::Positions(loc_positions(key, index, df.height())?);
            return Ok(Py::new(py, pf.derive_rows(&view, &sel)?)?.into_any());
        }
        // F28: partial-string datetime indexing — a year ("2021") or month
        // ("2021-02") string selects the whole period's rows (pandas parity).
        if let volas_core::IndexKind::Datetime(_, tz) = index.kind() {
            if let Ok(sk) = key.extract::<String>() {
                // a year or a month names its whole period (a finer string, its label)
                if let Some((lo, hi, _)) = tz.period_of_text(&sk).filter(|p| p.2 <= 2) {
                    let positions: Vec<usize> = (0..df.height())
                        .filter(|&i| {
                            index
                                .label_at(i)
                                .as_i64()
                                .is_some_and(|ns| ns >= lo && ns < hi)
                        })
                        .collect();
                    if positions.is_empty() {
                        return Err(PyKeyError::new_err(format!("label {sk:?} not found")));
                    }
                    // contiguous on an increasing index: the selection continues
                    let sel = RowSel::Positions(positions);
                    return Ok(Py::new(py, pf.derive_rows(&view, &sel)?)?.into_any());
                }
            }
        }
        let label = parse_label(key, index)?;
        let pos = index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(Py::new(py, row_at(df, pos))?.into_any())
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
        let (rows, col) = split_row_col(key, "loc")?;
        let colname: String = col.extract().map_err(|_| {
            PyTypeError::new_err("loc assignment column must be a single column name")
        })?;
        // Resolve labels against the visible rows; the write path maps them onto the frame.
        let len = pf.meta().range().1;
        let positions: Vec<usize> = loc_positions(&rows, &pf.meta().labels(), len)?;
        let j = pf
            .inner
            .column_pos(&colname)
            .ok_or_else(|| PyKeyError::new_err(format!("column {colname:?} not found")))?;
        let target = pf.inner.columns()[j].dtype();
        let val = resolve_assignment(value, target, positions.len())?;
        pf.write_cells(vec![(j, positions, val)])
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
        let view = pf.read()?;
        let i = norm_idx(key.0, view.rows())?;
        let (col, at) = view.cells(norm_idx(key.1, view.width())?);
        Ok(np_scalar_to_py(py, col, at + i))
    }

    /// `df.iat[i, j] = scalar` — set a single cell by position (copy-on-write).
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: (isize, isize),
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        let i = norm_idx(key.0, pf.meta().range().1)?;
        let j = norm_idx(key.1, pf.inner.width())?;
        let target = pf.inner.columns()[j].dtype();
        let val = scalar_to_column(value, target)?;
        pf.write_cells(vec![(j, vec![i], val)])
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
        let view = pf.read()?;
        let meta = pf.meta();
        let lidx = meta.labels();
        let label = parse_label(key.0.bind(py), &lidx)?;
        let i = lidx
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        let j = view.column_pos(&key.1).ok_or_else(|| {
            PyKeyError::new_err(format!("column \"{}\" not found", key.1))
        })?;
        let (col, at) = view.cells(j);
        Ok(np_scalar_to_py(py, col, at + i))
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
        let i = {
            let meta = pf.meta();
            let lidx = meta.labels();
            let label = parse_label(key.0.bind(py), &lidx)?;
            lidx.position_of(&label).ok_or_else(|| PyKeyError::new_err("label not found"))?
        };
        let j = pf
            .inner
            .column_pos(&key.1)
            .ok_or_else(|| PyKeyError::new_err(format!("column {:?} not found", key.1)))?;
        let target = pf.inner.columns()[j].dtype();
        let val = scalar_to_column(value, target)?;
        pf.write_cells(vec![(j, vec![i], val)])
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
            let (a, b) = self.inner.index.label_slice(lo.as_ref(), hi.as_ref()).map_err(pyerr)?;
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

