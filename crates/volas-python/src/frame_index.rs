//! The DataFrame indexing surface: the `.iloc` / `.loc` / `.iat` / `.at`
//! accessors, axis resolution, and assignment-value coercion.

use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PySliceIndices, PyString, PyTuple};
use volas_core::{Column, DType, DataFrame, Index, IndexKind, Label, Series, SliceBound};

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
    if v.is_instance_of::<PyList>()
        || v.is_instance_of::<PyTuple>()
        || v.extract::<PyRef<PySeries>>().is_ok()
    {
        return true;
    }
    v.getattr("ndim")
        .ok()
        .and_then(|n| n.extract::<usize>().ok())
        .is_some_and(|n| n >= 1)
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
        DType::F64 | DType::F32 | DType::I64 | DType::I32 => number_column(v, target)?
            .convert_lossless(target)
            .map_err(pyerr),
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
            return Err(PyOverflowError::new_err(format!(
                "{idx} is out of the {target} range"
            )));
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

/// Convert an array-like assignment value (list / NumPy array / `Series`) to a
/// [`Column`]; core coerces it toward the target dtype.
pub(crate) fn value_to_column(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(s) = v.extract::<PyRef<PySeries>>() {
        return Ok(s.inner.data().clone());
    }
    pyany_to_column(v)
}

/// Resolve a `df[...] = value` right-hand side for `n` selected rows: a scalar is
/// broadcast (length-1 column), an array-like must match `n`.
pub(crate) fn resolve_assignment(
    v: &Bound<'_, PyAny>,
    target: DType,
    n: usize,
) -> PyResult<Column> {
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

/// Resolve a one-dimensional bool mask for either axis. A recognized bool dtype
/// never becomes integer positions; an untyped empty list remains a list selector.
pub(crate) fn as_bool_mask<'a>(
    sel: &Bound<'_, PyAny>, height: usize,
    labels: impl FnOnce() -> PyResult<std::borrow::Cow<'a, Index>>,
) -> PyResult<Option<Vec<usize>>> {
    if sel.is_exact_instance_of::<pyo3::types::PyBool>() {
        return Err(PyTypeError::new_err("boolean mask must be one-dimensional"));
    }
    if sel.is_exact_instance_of::<pyo3::types::PyInt>()
        || sel.is_instance_of::<PyString>()
        || sel.downcast::<PySlice>().is_ok()
    {
        return Ok(None);
    }
    let collect = |bits: &[bool]| -> PyResult<Option<Vec<usize>>> {
        if bits.len() != height {
            return Err(PyValueError::new_err(format!("boolean mask length {} != axis length {height}", bits.len())));
        }
        Ok(Some({
            bits.iter()
                .enumerate()
                .filter_map(|(i, &b)| b.then_some(i))
                .collect()
        }))
    };
    if let Some(bits) = bool_mask_key(sel, labels)? {
        return collect(&bits.values());
    }
    let bool_dtype = sel.getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .is_ok_and(|kind| kind == "b");
    if bool_dtype {
        return Err(PyTypeError::new_err("boolean mask must be a one-dimensional bool array, Series or list"));
    }
    Ok(None)
}

/// Resolve an `iloc` row selector (int / slice / int-list / bool-mask) to row
/// positions.
pub(crate) fn iloc_positions(sel: &Bound<'_, PyAny>, labels: &Labels<'_>) -> PyResult<Vec<usize>> {
    let height = labels.rows.len();

    if let Some(pos) = as_bool_mask(sel, height, || Ok(labels.index()))? {
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

/// The labels of a frame's visible rows, looked up where they lie: a windowed frame's
/// visible rows are a range of its stored index, never copied out to resolve a label.
pub(crate) struct Labels<'a> {
    index: &'a Index,
    rows: std::ops::Range<usize>,
}

impl<'a> Labels<'a> {
    /// The labels at positions `rows` of `index`.
    pub(crate) fn of(index: &'a Index, rows: std::ops::Range<usize>) -> Labels<'a> {
        Labels { index, rows }
    }

    /// Every label of `index`.
    pub(crate) fn whole(index: &'a Index) -> Labels<'a> {
        Labels::of(index, 0..index.len())
    }

    /// The visible position of the row `key` labels — the first, when several do — in
    /// O(log rows) over ordered labels ([`Index::position_in`]); a `KeyError` when none
    /// does.
    pub(crate) fn locate(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        let label = parse_label(key, self.index)?;
        self.index
            .position_in(&label, self.rows.clone())
            .map(|p| p - self.rows.start)
            .ok_or_else(|| PyKeyError::new_err("label not found"))
    }

    /// The visible labels as an index of their own — for a label slice, an alignment,
    /// an export (a copy when they are not every stored label).
    pub(crate) fn index(&self) -> std::borrow::Cow<'a, Index> {
        if self.rows == (0..self.index.len()) {
            std::borrow::Cow::Borrowed(self.index)
        } else {
            std::borrow::Cow::Owned(self.index.slice(self.rows.start, self.rows.end))
        }
    }
}

/// Resolve a `loc` row selector (bool-mask / label-slice / label / label-list) to
/// row positions.
pub(crate) fn loc_positions(
    sel: &Bound<'_, PyAny>,
    labels: &Labels<'_>,
    height: usize,
) -> PyResult<Vec<usize>> {
    if let Some(pos) = as_bool_mask(sel, height, || Ok(labels.index()))? {
        return Ok(pos);
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let index = labels.index();
        let (lo, hi) = label_bounds(slice, &index)?;
        let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref()).map_err(pyerr)?;
        return Ok((a..b).collect());
    }
    if let Ok(list) = sel.downcast::<PyList>() {
        return list.iter().map(|item| labels.locate(&item)).collect();
    }
    Ok(vec![labels.locate(sel)?])
}

/// One axis of a 2-D `iloc` / `loc` get: a single scalar position (the axis is
/// reduced away, pandas-style) or a list of positions (the axis is kept).
pub(crate) enum AxisSel {
    One(usize),
    Many(Vec<usize>),
}

/// Resolve an `iloc` row axis: a bare int reduces the axis (`AxisSel::One`); a
/// slice / int-list / boolean mask keeps it (`AxisSel::Many`).
pub(crate) fn iloc_row_axis(sel: &Bound<'_, PyAny>, index: &Index) -> PyResult<RowAxis> {
    let height = index.len();

    if let Some(positions) = as_bool_mask(sel, height, || Ok(std::borrow::Cow::Borrowed(index)))? {
        return Ok(RowAxis::Rows(RowSel::Positions(positions)));
    }
    if let Ok(i) = sel.extract::<isize>() {
        return Ok(RowAxis::One(norm_idx(i, height)?));
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        return Ok(RowAxis::Rows(positional_slice(
            &slice.indices(height as isize)?,
        )));
    }
    Ok(RowAxis::Rows(RowSel::Positions(iloc_positions(
        sel, &Labels::whole(index),
    )?)))
}

/// The row axis of a 2-D `iloc` / `loc` get: a single row (the axis is reduced away,
/// pandas-style) or a selection of rows (the axis is kept) — a range staying a range,
/// so the one derivation decides it as the 1-D forms do.
pub(crate) enum RowAxis {
    One(usize),
    Rows(RowSel),
}

/// Resolve an `iloc` column axis, including the same bool masks as the row axis.
pub(crate) fn iloc_col_axis(sel: &Bound<'_, PyAny>, df: &DataFrame) -> PyResult<AxisSel> {
    let width = df.width();

    if let Some(positions) = as_bool_mask(sel, width, || Index::new(None, volas_core::IndexKind::Str(df.names().to_vec())).map(std::borrow::Cow::Owned).map_err(pyerr))? {
        return Ok(AxisSel::Many(positions));
    }
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

/// Resolve a `loc` row axis over `index`: a single label reduces the axis
/// (`RowAxis::One`); a label slice, a year / month string (its period), a label list or
/// a boolean mask keeps it (`RowAxis::Rows`).
pub(crate) fn loc_row_axis(
    sel: &Bound<'_, PyAny>,
    index: &Index,
    height: usize,
) -> PyResult<RowAxis> {
    if let Ok(slice) = sel.downcast::<PySlice>() {
        return Ok(RowAxis::Rows(label_rows(index, slice)?));
    }
    let labels = Labels::whole(index);
    if let Some(positions) = as_bool_mask(sel, height, || Ok(std::borrow::Cow::Borrowed(index)))? {
        return Ok(RowAxis::Rows(RowSel::Positions(positions)));
    }
    if sel.downcast::<PyList>().is_ok() {
        return Ok(RowAxis::Rows(RowSel::Positions(loc_positions(
            sel, &labels, height,
        )?)));
    }
    if let Some(period) = period_rows(sel, index)? {
        return Ok(RowAxis::Rows(period));
    }
    Ok(RowAxis::One(labels.locate(sel)?))
}

/// The rows a year or a month string selects on a datetime index — its whole period
/// (pandas partial-string indexing); `None` for any other key. A period no row falls in
/// is a `KeyError`.
pub(crate) fn period_rows(key: &Bound<'_, PyAny>, index: &Index) -> PyResult<Option<RowSel>> {
    let (IndexKind::Datetime(_, tz), Ok(text)) = (index.kind(), key.extract::<String>()) else {
        return Ok(None);
    };
    let Some((lo, hi, _)) = tz.period_of_text(&text).filter(|p| p.2 <= 2) else {
        return Ok(None);
    };
    let positions: Vec<usize> = (0..index.len())
        .filter(|&i| {
            index
                .label_at(i)
                .as_i64()
                .is_some_and(|ns| ns >= lo && ns < hi)
        })
        .collect();
    if positions.is_empty() {
        return Err(PyKeyError::new_err(format!("label {text:?} not found")));
    }
    // contiguous on an increasing index: the selection continues
    Ok(Some(RowSel::Positions(positions)))
}

/// Resolve a `loc` column axis (name / name-list / inclusive name-slice) to
/// column positions.
pub(crate) fn loc_col_axis(sel: &Bound<'_, PyAny>, df: &DataFrame) -> PyResult<AxisSel> {
    if let Some(positions) = as_bool_mask(sel, df.width(), || Index::new(None, volas_core::IndexKind::Str(df.names().to_vec())).map(std::borrow::Cow::Owned).map_err(pyerr))? {
        return Ok(AxisSel::Many(positions));
    }
    let pos_of = |name: &str| PyDataFrame::column_position(df, name);
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
    rows: RowAxis,
    cols: AxisSel,
    explicit_projection: bool,
) -> PyResult<Py<PyAny>> {
    let df: &DataFrame = view;
    match (rows, cols) {
        (RowAxis::One(i), AxisSel::One(j)) => np_scalar_to_py(py, &df.columns()[j], i),
        (RowAxis::Rows(r), AxisSel::One(j)) => {
            let sub = project_cols(df, &[j])?.take(&r.positions()).map_err(pyerr)?;
            let series = PySeries { inner: sub.series_at(0) };
            Ok(Py::new(py, series)?.into_any())
        }
        (RowAxis::One(i), AxisSel::Many(c)) => {
            let sub = project_cols(df, &c)?.take(&[i]).map_err(pyerr)?;
            Ok(Py::new(py, PyRow { inner: sub })?.into_any())
        }
        (RowAxis::Rows(sel), AxisSel::Many(c)) => {
            let names: Vec<String> = c.iter().map(|&j| df.names()[j].clone()).collect();
            let rows = pf.derive_rows(view, &sel)?;
            if let Some(cached) = names
                .iter()
                .find(|n| df.is_computed(n) && !rows.inner.has_column(n))
            {
                return Err(uncarried_cached(cached));
            }
            if !explicit_projection && names.as_slice() == rows.inner.names() {
                return Ok(Py::new(py, rows)?.into_any());
            }
            let inner = rows.inner.project(&names, 0, rows.inner.height()).map_err(pyerr)?;
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
                let rows = iloc_row_axis(&tup.get_item(0)?, df.index())?;
                let cols = iloc_col_axis(&tup.get_item(1)?, df)?;
                return select_2d(py, &pf, &view, rows, cols, tup.get_item(1)?.downcast::<PySlice>().is_err());
            }
        }
        match iloc_row_axis(key, df.index())? {
            RowAxis::One(i) => Ok(Py::new(py, row_at(df, i)?)?.into_any()),
            RowAxis::Rows(sel) => Ok(Py::new(py, pf.derive_rows(&view, &sel)?)?.into_any()),
        }
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
        let columns = match as_bool_mask(&col, pf.inner.width(), || Index::new(None, volas_core::IndexKind::Str(pf.inner.names().to_vec())).map(std::borrow::Cow::Owned).map_err(pyerr))? {
            Some(selected) => AxisSel::Many(selected),
            None => AxisSel::One(norm_idx(col.extract::<isize>()?, pf.inner.width())?),
        };
        let positions = iloc_positions(&rows, &pf.meta().labels())?;
        pf.assign_columns(columns, positions, value)
    }
}

impl PyDataFrame {
    /// Resolve every target value before the existing atomic cell-write owner runs.
    fn assign_columns(&mut self, columns: AxisSel, positions: Vec<usize>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let columns = match columns {
            AxisSel::One(j) => {
                let values = resolve_assignment(value, self.inner.columns()[j].dtype(), positions.len())?;
                return self.write_cells(vec![(j, positions, values)]);
            }
            AxisSel::Many(cols) => cols,
        };
        if columns.len() > 1 && is_array(value) {
            return Err(PyTypeError::new_err("array assignment requires a single selected column"));
        }
        let count = columns.len();
        let mut positions = positions;
        let mut writes = Vec::with_capacity(count);
        for (i, j) in columns.into_iter().enumerate() {
            let values = resolve_assignment(value, self.inner.columns()[j].dtype(), positions.len())?;
            let rows = if i + 1 == count { std::mem::take(&mut positions) } else { positions.clone() };
            writes.push((j, rows, values));
        }
        self.write_cells(writes)
    }
}

pub(crate) fn row_at(df: &DataFrame, i: usize) -> PyResult<PyRow> {
    // `take` materializes the index label (Range -> Int64([i])) and preserves
    // every column's dtype — a faithful 1-row frame.
    Ok(PyRow {
        inner: df.take(&[i]).map_err(pyerr)?,
    })
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
            // a partial date names a period: its first instant starts, its last stops
            IndexKind::Datetime(_, tz)
                if text && tz.period_of_text(&obj.extract::<String>()?).is_some() =>
            {
                let (start, end, _) = tz
                    .period_of_text(&obj.extract::<String>()?)
                    .expect("checked above");
                Label::I64(if stop { end - 1 } else { start })
            }
            // text naming no instant (not a datetime, a DST gap / fold wall clock) cannot
            // bound a datetime slice: a TypeError with the reason, as pandas raises
            IndexKind::Datetime(_, tz) if text => {
                Label::I64(crate::convert::parse_ts_in_tz(&obj, tz).map_err(|e| {
                    let py = obj.py();
                    if e.is_instance_of::<pyo3::exceptions::PyValueError>(py) {
                        PyTypeError::new_err(format!(
                            "cannot slice a DatetimeIndex with {}: {}",
                            obj.repr()
                                .map_or_else(|_| "?".to_string(), |r| r.to_string()),
                            e.value(py)
                        ))
                    } else {
                        e
                    }
                })?)
            }
            _ => parse_label(&obj, index)?,
        };
        Ok(Some(if text {
            SliceBound::text(label)
        } else {
            SliceBound::exact(label)
        }))
    };
    Ok((
        bound(slice.getattr("start")?, false)?,
        bound(slice.getattr("stop")?, true)?,
    ))
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
                return select_2d(py, &pf, &view, rows, cols, tup.get_item(1)?.downcast::<PySlice>().is_err());
            }
        }
        match loc_row_axis(key, index, df.height())? {
            RowAxis::One(i) => Ok(Py::new(py, row_at(df, i)?)?.into_any()),
            RowAxis::Rows(sel) => Ok(Py::new(py, pf.derive_rows(&view, &sel)?)?.into_any()),
        }
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
        let columns = match as_bool_mask(&col, pf.inner.width(), || Index::new(None, volas_core::IndexKind::Str(pf.inner.names().to_vec())).map(std::borrow::Cow::Owned).map_err(pyerr))? {
            Some(selected) => AxisSel::Many(selected),
            None => {
                let name = col.extract::<String>().map_err(|_| PyTypeError::new_err("loc assignment column must be a single column name or boolean mask"))?;
                AxisSel::One(PyDataFrame::column_position(&pf.inner, &name)?)
            }
        };
        let len = pf.meta().range().1;
        let positions = loc_positions(&rows, &pf.meta().labels(), len)?;
        pf.assign_columns(columns, positions, value)
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
        np_scalar_to_py(py, col, at + i)
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
        let i = pf.meta().labels().locate(key.0.bind(py))?;
        let j = PyDataFrame::column_position(&view, &key.1)?;
        let (col, at) = view.cells(j);
        np_scalar_to_py(py, col, at + i)
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
        let i = pf.meta().labels().locate(key.0.bind(py))?;
        let j = PyDataFrame::column_position(&pf.inner, &key.1)?;
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
            let (lo, hi) = label_bounds(slice, self.inner.index())?;
            let (a, b) = self
                .inner
                .index()
                .label_slice(lo.as_ref(), hi.as_ref())
                .map_err(pyerr)?;
            let positions: Vec<usize> = (a..b).collect();
            let index = Arc::new(self.inner.index().take(&positions).map_err(pyerr)?);
            let data = self.inner.data().take(&positions);
            return Ok(Py::new(
                py,
                PySeries {
                    inner: Series::new(self.inner.name().map(str::to_owned), data, index).map_err(pyerr)?,
                },
            )?
            .into_any());
        }
        let label = parse_label(key, self.inner.index())?;
        let pos = self
            .inner
            .index()
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        np_scalar_to_py(py, self.inner.data(), pos)
    }
}
