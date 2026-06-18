//! `DataFrame` methods, part 2: indexing assignment / lookup, directive
//! execution, IO (CSV / pandas / NumPy), structural ops, and rendering.

use std::collections::HashMap;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PySlice};
use volas_core::{
    Column, DataFrame, Label,
};
use volas_directive::{execute, parse};
use volas_time::Cumulator;

use crate::format::{
    cell_to_csv, csv_escape, index_label_csv, render_frame, render_frame_html, Dimensions,
    DisplayOpts, NA_REPR,
};
use crate::timeframe::{build_agg_spec_for, resolve_time_frame};
#[allow(unused_imports)]
use crate::*;

/// R4-P2-01: appended rows must not introduce a column absent from the target.
/// The name-aligned append NaN-pads a *missing* column (fine), but it silently
/// dropped an *extra* one — so an exchange adding a field would lose data without
/// a trace. Reject the extra column instead.
fn require_no_new_columns(target: &DataFrame, src: &DataFrame) -> PyResult<()> {
    if let Some(name) = src.names().iter().find(|n| !target.has_column(n)) {
        return Err(PyValueError::new_err(format!(
            "append: column {name:?} is not in the target frame — appended rows must not \
             introduce a new column (a missing column is NaN-padded; an extra one is \
             rejected so data is never silently dropped)"
        )));
    }
    Ok(())
}

/// The frame's 2-D NA mask (row-major `h × w`, `True` where the cell is missing) — the
/// index for an `na_value` fill into an exported NumPy matrix.
fn frame_na_mask<'py>(py: Python<'py>, df: &DataFrame) -> PyResult<Bound<'py, PyAny>> {
    let (h, w) = (df.height(), df.width());
    let cols = df.columns();
    let mut data = Vec::with_capacity(h * w);
    for i in 0..h {
        for c in cols {
            data.push(!c.is_valid(i));
        }
    }
    Ok(ndarray::Array2::from_shape_vec((h, w), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py)
        .into_any())
}

/// Substitute `na_value` into the missing cells (`mask`) of an exported array, in place;
/// a no-op when either the mask or the value is absent.
fn fill_na_2d<'py>(
    arr: Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    na_value: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let (Some(m), Some(nv)) = (mask, na_value) {
        arr.set_item(m, nv)?;
    }
    Ok(arr)
}

#[pymethods]
impl PyDataFrame {
    /// `df[key] = value`. With a column name, add or replace that column —
    /// `value` may be a scalar (broadcast), a 1-D array / list, or a Series
    /// (positional, length must equal the frame height). With a boolean mask and
    /// a scalar fill, assign by mask: a boolean Series / array sets whole rows
    /// (`df[df['a'] > 0] = 0`), a boolean frame sets cells (`df[df.isna()] = 0`).
    /// Copy-on-write: a prior `copy()` is unaffected.
    pub(crate) fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Boolean-mask assignment with a scalar fill: df[mask] = v
        if let Some(mask) = bool_mask_key(key)? {
            return self.assign_row_mask(&mask, value);
        }
        if let Ok(cond) = key.extract::<PyRef<PyDataFrame>>() {
            return self.assign_cell_mask(&cond, value);
        }
        // Column assignment: df[name] = value
        let name: String = key.extract().map_err(|_| {
            PyTypeError::new_err("DataFrame key must be a column name or a boolean mask")
        })?;
        let h = self.inner.height();
        let col = if let Ok(s) = value.extract::<PyRef<PySeries>>() {
            s.inner.data.clone()
        } else if let Ok(b) = value.extract::<bool>() {
            Column::bool(vec![b; h])
        } else if let Ok(scalar) = value.extract::<f64>() {
            Column::f64(vec![scalar; h])
        } else {
            pyany_to_column(value)?
        };
        // Overwriting an EXISTING column may invalidate any cached indicator derived
        // from it (e.g. `df['close'] = …` stales `ma:20`); mark those for recompute on
        // next access. Adding a brand-new column cannot affect existing caches.
        let existed = self.inner.has_column(&name);
        self.inner.set_column(&name, col).map_err(pyerr)?;
        if existed {
            self.inner.invalidate_computed_on_write(&name);
        }
        Ok(())
    }

    // `df[key]` — column name / indicator directive / list / boolean mask /
    // slice. The user-facing usage lives in the class docstring (pyo3 implements
    // `__getitem__` as a type slot and does not surface its doc comment).
    pub(crate) fn __getitem__(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // boolean mask (Series or numpy)
        if let Ok(s) = key.extract::<PyRef<PySeries>>() {
            if let Column::Bool(..) = &s.inner.data {
                // O5: reject an NA-carrying mask (an unknown signal is not False).
                let mask = bool_mask_vec(&s.inner.data)?;
                let sub = self.inner.filter_mask(&mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
            }
        }
        if let Ok(arr) = key.extract::<PyReadonlyArray1<bool>>() {
            let sub = self.inner.filter_mask(arr.as_slice()?).map_err(pyerr)?;
            return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
        }
        // boolean mask as a plain Python list (df[[True, False, ...]]). An empty
        // list is an empty column projection, not a mask, so it falls through.
        if let Ok(mask) = key.extract::<Vec<bool>>() {
            if !mask.is_empty() {
                if mask.len() != self.inner.height() {
                    return Err(PyIndexError::new_err(format!(
                        "boolean index has wrong length: {} instead of {}",
                        mask.len(),
                        self.inner.height()
                    )));
                }
                let sub = self.inner.filter_mask(&mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
            }
        }
        // label / positional slice: df[:'date'], df[1:5]
        if let Ok(slice) = key.downcast::<PySlice>() {
            let sub = slice_frame(&self.inner, slice)?;
            return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
        }
        // column name or directive — materialize + auto-refresh a stale cached
        // directive (O(lookback), not O(n)) so the Series is always fresh.
        if let Ok(name) = key.extract::<String>() {
            let (resolved, col) = self.materialize_refresh(&name)?;
            return Ok(Py::new(py, self.wrap_series(resolved, col))?.into_any());
        }
        // list of names / directives — each entry auto-refreshes exactly like the
        // single-name form, so `df[['ma:3']]` and `df['ma:3']` stay consistent.
        if let Ok(list) = key.extract::<Vec<String>>() {
            let mut cols = Vec::with_capacity(list.len());
            for n in &list {
                let (_, col) = self.materialize_refresh(n)?;
                cols.push(col);
            }
            let idx = (*self.inner.index().as_ref()).clone();
            let df = DataFrame::new(list, cols, Some(idx)).map_err(pyerr)?;
            return Ok(Py::new(py, PyDataFrame::plain(df))?.into_any());
        }
        Err(PyKeyError::new_err(
            "key must be a column name, directive, list, boolean mask, or slice",
        ))
    }

    /// Evaluate an indicator directive and return its values as a NumPy array.
    ///
    /// Unlike ``df['ma:5']`` (which returns a Series and caches the column),
    /// ``exec`` returns the raw array; pass ``create_column=True`` to also cache
    /// it on the frame under its canonical name.
    ///
    /// Args:
    ///     directive (str): the directive, e.g. ``'macd'``, ``'boll.upper:20'``,
    ///         ``'close > open'``.
    ///     create_column (bool): if True, materialize and cache the result as a
    ///         column (default False).
    ///
    /// Usage::
    ///
    ///     df.exec('ma:5')               # ndarray of SMA(5)
    ///     df.exec('kdj.j', create_column=True)  # also caches the column
    ///
    /// Returns:
    ///     numpy.ndarray
    #[pyo3(signature = (directive, create_column = false))]
    pub(crate) fn exec<'py>(
        &mut self,
        py: Python<'py>,
        directive: &str,
        create_column: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.inner.has_column(directive) {
            let col = self.inner.column(directive).map_err(pyerr)?.clone();
            return Ok(column_to_numpy(py, &col));
        }
        let node = parse(directive).map_err(directive_err)?;
        if create_column {
            // Materialize + cache under the canonical name, exactly like `df[directive]`.
            let canonical = volas_directive::stringify(&node);
            if self.inner.has_column(&canonical) {
                self.refresh_computed(Some(&canonical))?;
            } else {
                let col = execute(&self.inner, &node).map_err(value_err)?;
                let lookback = volas_directive::lookback::lookback(&node);
                let state = volas_directive::exec::initial_state(&self.inner, &node, &col);
                self.inner.set_column(&canonical, col).map_err(pyerr)?;
                self.inner
                    .set_computed(&canonical, canonical.clone(), lookback);
                self.inner.set_computed_state(&canonical, state);
            }
            let col = self.inner.column(&canonical).map_err(pyerr)?.clone();
            Ok(column_to_numpy(py, &col))
        } else {
            let col = execute(&self.inner, &node).map_err(value_err)?;
            Ok(column_into_numpy(py, col))
        }
    }

    /// Gets a column from the frame by name (alias-aware), as a Series.
    ///
    /// Args:
    ///     key (str): the column name.
    ///
    /// Returns:
    ///     Series
    pub(crate) fn get_column(&self, key: &str) -> PyResult<PySeries> {
        let col = self.inner.column(key).map_err(pyerr)?.clone();
        Ok(self.wrap_series(key.to_string(), col))
    }

    /// A copy of the frame — preserving the cached directive columns / cursor and
    /// (for a tf-aware frame) the cumulation state, so the copy keeps folding.
    pub(crate) fn copy(&self) -> PyDataFrame {
        PyDataFrame {
            inner: self.inner.clone(),
            tf: self.tf.clone(),
        }
    }

    /// Convert to a `pandas.DataFrame`. pandas is imported lazily (only here), so
    /// volas stays pandas-free at import.
    #[pyo3(signature = (dtype_backend = "numpy"))]
    pub(crate) fn to_pandas<'py>(&self, py: Python<'py>, dtype_backend: &str) -> PyResult<Bound<'py, PyAny>> {
        ensure_fresh(&self.inner)?;
        // 'numpy' (default): an int/bool column with NA exports as float64+NaN, the
        // most ecosystem-compatible form. 'numpy_nullable': a faithful, lossless
        // masked Int64 / boolean. Mirrors pandas' own `dtype_backend`.
        let nullable = match dtype_backend {
            "numpy" => false,
            "numpy_nullable" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "dtype_backend must be 'numpy' or 'numpy_nullable', got {other:?}"
                )))
            }
        };
        let pd = py.import("pandas")?;
        let data = PyDict::new(py);
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            data.set_item(name, column_to_pandas(py, &pd, col, nullable)?)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("index", index_to_numpy(py, self.inner.index())?)?;
        let pdf = pd.call_method("DataFrame", (data,), Some(&kwargs))?;
        // A tz-aware frame exports a UTC-naive datetime64 index (index_to_numpy); restore the
        // display zone so the pandas index is tz-aware — a faithful round-trip with from_pandas.
        if let Some(tz) = self.tz() {
            let aware = pdf
                .getattr("index")?
                .call_method1("tz_localize", ("UTC",))?
                .call_method1("tz_convert", (&tz,))?;
            pdf.setattr("index", aware)?;
        }
        // Carry the index name onto the pandas index (pandas parity).
        if let Some(name) = self.inner.index().name() {
            let renamed = pdf.getattr("index")?.call_method1("rename", (name,))?;
            pdf.setattr("index", renamed)?;
        }
        Ok(pdf)
    }

    /// Write the frame as CSV (pandas-subset). With no `path`, returns the CSV
    /// string. Datetime columns are written as formatted strings (round-trips
    /// with `read_csv`).
    // One parameter per pandas `to_csv` keyword — a struct would break the API.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path = None, sep = ",", index = true, header = true, na_rep = "", columns = None, float_format = None))]
    pub(crate) fn to_csv(
        &self,
        path: Option<std::path::PathBuf>,
        sep: &str,
        index: bool,
        header: bool,
        na_rep: &str,
        columns: Option<Vec<String>>,
        float_format: Option<&str>,
    ) -> PyResult<Option<String>> {
        ensure_fresh(&self.inner)?;
        let ff = parse_ff(float_format)?;
        let names = self.inner.names();
        let positions: Vec<usize> = match &columns {
            Some(cols) => cols
                .iter()
                .map(|n| {
                    self.inner
                        .column_pos(n)
                        .ok_or_else(|| PyKeyError::new_err(format!("column \"{n}\" not found")))
                })
                .collect::<PyResult<_>>()?,
            None => (0..self.inner.width()).collect(),
        };
        let mut out = String::new();
        if header {
            if index {
                // pandas writes the index name, or an empty field for an unnamed index.
                out.push_str(self.inner.index().name().unwrap_or(""));
                out.push_str(sep);
            }
            let hdr: Vec<String> = positions
                .iter()
                .map(|&j| csv_escape(names[j].clone(), sep))
                .collect();
            out.push_str(&hdr.join(sep));
            out.push('\n');
        }
        for i in 0..self.inner.height() {
            if index {
                out.push_str(&index_label_csv(self.inner.index(), i));
                out.push_str(sep);
            }
            let cells: Vec<String> = positions
                .iter()
                .map(|&j| csv_escape(cell_to_csv(&self.inner.columns()[j], i, na_rep, ff), sep))
                .collect();
            out.push_str(&cells.join(sep));
            out.push('\n');
        }
        match path {
            Some(p) => {
                std::fs::write(&p, out).map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(None)
            }
            None => Ok(Some(out)),
        }
    }

    /// Drop rows by index label (`axis=0`) or columns by name (`axis=1`) —
    /// returns a new DataFrame. Row labels are parsed against the index kind.
    #[pyo3(signature = (labels, axis = 0, errors = "raise"))]
    pub(crate) fn drop(&self, py: Python<'_>, labels: Vec<Py<PyAny>>, axis: i64, errors: &str) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let ignore_missing = match errors {
            "raise" => false,
            "ignore" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "drop: errors must be 'raise' or 'ignore', got {other:?}"
                )))
            }
        };
        if axis == 1 {
            let drop_names: Vec<String> = labels
                .iter()
                .map(|l| l.bind(py).extract::<String>())
                .collect::<PyResult<_>>()?;
            // F37: a missing label is an error (pandas KeyError), not a silent
            // no-op — unless explicitly opted out with errors='ignore' (F44).
            let names = self.inner.names();
            if !ignore_missing {
                for n in &drop_names {
                    if !names.iter().any(|m| m == n) {
                        return Err(PyKeyError::new_err(format!("[{n:?}] not found in axis")));
                    }
                }
            }
            let keep: Vec<String> = names
                .iter()
                .filter(|n| !drop_names.contains(n))
                .cloned()
                .collect();
            return Ok(PyDataFrame::plain(self.inner.select(&keep).map_err(pyerr)?));
        }
        let index = self.inner.index();
        let targets: Vec<Label> = labels
            .iter()
            .map(|l| parse_label(l.bind(py), index))
            .collect::<PyResult<_>>()?;
        // F37 (row axis): every label must exist in the index, else KeyError —
        // unless errors='ignore' (F44).
        if !ignore_missing {
            let present: Vec<Label> = (0..self.inner.height()).map(|i| index.label_at(i)).collect();
            for t in &targets {
                if !present.contains(t) {
                    return Err(PyKeyError::new_err("label not found in axis"));
                }
            }
        }
        let positions: Vec<usize> = (0..self.inner.height())
            .filter(|&i| !targets.contains(&index.label_at(i)))
            .collect();
        Ok(PyDataFrame::plain(take_frame(&self.inner, &positions)))
    }

    /// Append the rows of another DataFrame or a single Row **in place** and
    /// return the same frame (amortized O(1), like ``list.append`` — the live
    /// single-bar hot path, no full-column copy).
    ///
    /// On a **time_frame** frame (see the constructor / ``cumulate``) the rows
    /// are treated as *finer* bars and folded into the current period: a bar in
    /// the open period updates the forming last row (``df.iloc[-1]``), a bar in a
    /// new period rolls over into a fresh row. A re-sent forming bar (same
    /// timestamp) updates rather than double-counts.
    ///
    /// Missing columns are NaN-padded; cached directive columns go stale until
    /// ``fulfill()``. A snapshot taken via ``copy()`` / ``iloc`` is unaffected
    /// (it pays one copy-on-write the next time *it* is appended to).
    ///
    /// Args:
    ///     other (DataFrame | Row): the rows to append (fine bars if tf-aware).
    ///
    /// Usage::
    ///
    ///     df.append(bar)           # append / fold one bar
    ///     df.append(other_frame)   # append / fold many bars
    ///
    /// Returns:
    ///     DataFrame: ``self`` (enabling chaining).
    pub(crate) fn append<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        if let Ok(df) = other.extract::<PyRef<PyDataFrame>>() {
            if slf.as_ptr() == other.as_ptr() {
                // `df.append(df)` needs an owned snapshot before taking `self` mutably.
                let other_inner = df.inner.clone();
                drop(df);
                let mut me = slf.borrow_mut();
                require_no_new_columns(&me.inner, &other_inner)?;
                if me.tf.is_some() {
                    me.fold_append(&other_inner)?;
                } else {
                    me.inner.append(&other_inner).map_err(pyerr)?;
                }
                return Ok(slf);
            }
            // Normal live path: append a distinct one-row frame without cloning it.
            let mut me = slf.borrow_mut();
            require_no_new_columns(&me.inner, &df.inner)?;
            if me.tf.is_some() {
                me.fold_append(&df.inner)?;
            } else {
                me.inner.append(&df.inner).map_err(pyerr)?;
            }
            return Ok(slf);
        }
        if let Ok(row) = other.extract::<PyRef<PyRow>>() {
            let mut me = slf.borrow_mut();
            require_no_new_columns(&me.inner, &row.inner)?;
            if me.tf.is_some() {
                me.fold_append(&row.inner)?;
            } else {
                me.inner.append(&row.inner).map_err(pyerr)?;
            }
            return Ok(slf);
        }
        Err(PyTypeError::new_err("append expects a DataFrame or Row"))
    }

    /// Arrow PyCapsule stream protocol — exposes the whole frame to Arrow consumers
    /// (`pa.table(df)`, `pl.from_dataframe(df)`) as a single zero-copy `RecordBatch`.
    /// `requested_schema` is accepted and ignored (the native dtypes are exported).
    #[pyo3(signature = (requested_schema = None))]
    pub(crate) fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        ensure_fresh(&self.inner)?;
        crate::arrow::frame_c_stream(py, self.inner.names(), self.inner.columns())
    }

    /// Build a DataFrame from any object exposing the Arrow stream protocol
    /// (`__arrow_c_stream__`) — a pyarrow `Table`, a polars `DataFrame`, … — zero-copy
    /// where dtypes match. The result carries a fresh `RangeIndex`.
    #[staticmethod]
    pub(crate) fn from_arrow(data: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let (names, cols) = crate::arrow::frame_from_arrow_obj(data)?;
        Ok(PyDataFrame::plain(DataFrame::new(names, cols, None).map_err(pyerr)?))
    }

    /// Export as a `pyarrow.Table` (zero-copy where dtypes match; requires pyarrow).
    pub(crate) fn to_arrow<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("pyarrow")?.call_method1("table", (slf,))
    }

    /// The frame as a 2-D NumPy array (pandas `to_numpy`). `to_numpy` is an export
    /// boundary — leaving volas — so an explicit `dtype` is honored per cell like
    /// pandas (the internal no-lossy contract governs *computation*, not what the
    /// caller asks to convert *out* to). The matrix:
    ///
    /// | frame \\ dtype | `None` (default)        | `"object"`          | int / bool          | float                 |
    /// |---|---|---|---|---|
    /// | numeric / bool | `float64` matrix        | typed-cell object   | exact cast          | cast                  |
    /// | datetime       | 2-D `datetime64[ns]`    | `Timestamp` / `NA`  | **exact epoch-ns** (NaT→`i64::MIN`) | epoch-ns as float (lossy past 2⁵³, NaT→NaN) |
    /// | mixed          | object (typed cells)    | typed-cell object   | exact cast          | cast (str → error)    |
    /// | contains str   | object (typed cells)    | str kept            | **error**           | **error**             |
    ///
    /// So `dtype="object"` is always lossless (each cell its own typed value —
    /// `Timestamp` / `volas.NA` / str / number), an integer dtype takes the exact
    /// `i64` channel (datetime never round-trips through `f64`), a float dtype is
    /// the caller's opt-in lossy export, and a `str` column rejects any numeric
    /// dtype (no numeric meaning) pointing at `dtype="object"`.
    #[pyo3(signature = (dtype = None, na_value = None))]
    pub(crate) fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&str>,
        na_value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        ensure_fresh(&self.inner)?;
        let cols = self.inner.columns();
        let has_str = cols.iter().any(|c| matches!(c, Column::Str(..)));

        // `na_value` fill plumbing: with a value AND any missing cell, build the 2-D NA
        // mask once and substitute `na_value` into the holes of whichever array is built.
        let nv = na_value.as_ref();
        let mask = (nv.is_some() && cols.iter().any(|c| c.null_count() > 0))
            .then(|| frame_na_mask(py, &self.inner))
            .transpose()?;
        let mask = mask.as_ref();

        if let Some(dt) = dtype {
            // `object`: a lossless typed-cell array (never the f64 channel) — the
            // inspection / interop export that keeps datetime, str and NA intact.
            if dt == "object" || dt == "O" {
                return fill_na_2d(self.object_array(py)?, mask, nv);
            }
            // Any numeric / temporal target: a str column has no numeric value
            // (pandas raises here too), so reject it and point at the object route.
            if has_str {
                return Err(PyValueError::new_err(format!(
                    "cannot convert a string column to {dt}; use dtype='object' to keep strings"
                )));
            }
            // A float target rides the (lossy) f64 channel — the caller opted into
            // float, so a datetime epoch-ns past 2⁵³ quantises and a NaT becomes
            // NaN. A non-float (integer / bool / datetime) target takes the EXACT
            // i64 channel so a datetime exports its true epoch-ns (NaT→i64::MIN)
            // and a large i64 column survives, instead of an f64 round-trip.
            let floaty = dt.contains("float")
                || dt == "f32"
                || dt == "f64"
                || dt == "double"
                || dt == "single";
            // The two channels carry different Vec element types, so each builds its
            // own (type-erased) NumPy matrix; the fill + cast is then shared.
            let arr = if floaty {
                let (data, h, w) = self.inner.to_row_major_f64();
                ndarray::Array2::from_shape_vec((h, w), data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py)
                    .into_any()
            } else {
                // The i64 channel serves both integer and `datetime64` targets. A
                // datetime NaT → `i64::MIN` is its documented exact export, but a plain
                // int/float NA has no integer representation — raise for a true integer
                // target (pandas-aligned) UNLESS `na_value` gives an explicit fill,
                // exempting datetime columns' sentinel.
                if nv.is_none()
                    && is_integer_dtype(py, dt)?
                    && cols
                        .iter()
                        .any(|c| !matches!(c, Column::Datetime(..)) && c.null_count() > 0)
                {
                    return Err(PyValueError::new_err(format!(
                        "cannot convert a frame with missing values to integer NumPy dtype \
                         '{dt}' (an NA has no integer representation) — pass na_value=, or use \
                         a float dtype"
                    )));
                }
                let (data, h, w) = self.inner.to_row_major_i64();
                ndarray::Array2::from_shape_vec((h, w), data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py)
                    .into_any()
            };
            return fill_na_2d(arr, mask, nv)?.call_method1("astype", (dt,));
        }

        // Default (no dtype) — the honest representation chosen by the dtypes.
        // Fast path: an entirely numeric/bool frame is exactly the f64 matrix (an
        // empty frame counts — `all` over no columns is true, checked first).
        let all_numeric = cols.iter().all(|c| {
            matches!(
                c,
                Column::F64(_)
                    | Column::F32(_)
                    | Column::I64(..)
                    | Column::I32(..)
                    | Column::Bool(..)
            )
        });
        if all_numeric {
            let (data, h, w) = self.inner.to_row_major_f64();
            let arr = ndarray::Array2::from_shape_vec((h, w), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any();
            return fill_na_2d(arr, mask, nv);
        }
        // A datetime-only frame -> a 2-D `datetime64[ns]` built DIRECTLY from the
        // raw i64 ns buffer (ns-exact, NaT = i64::MIN native). The old path boxed
        // each cell into a Python `Timestamp` then let NumPy re-coerce it, which
        // truncated ns and failed outright on a NaT cell (P1-01 / D1 / D2).
        if cols.iter().all(|c| matches!(c, Column::Datetime(_))) {
            let (data, h, w) = self.inner.to_row_major_i64();
            let arr = ndarray::Array2::from_shape_vec((h, w), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .call_method1("astype", ("datetime64[ns]",))?;
            return fill_na_2d(arr, mask, nv);
        }
        // Any other mix -> a lossless object array, each cell its own typed value.
        fill_na_2d(self.object_array(py)?, mask, nv)
    }

    /// Value equality (same columns + index + values, `NaN == NaN`).
    pub(crate) fn equals(&self, other: &PyDataFrame) -> PyResult<bool> {
        ensure_fresh(&self.inner)?;
        ensure_fresh(&other.inner)?;
        Ok(self.inner.equals(&other.inner))
    }

    /// Resample to a coarser timeframe (OHLCV cumulation / down-sampling),
    /// returning a **tf-aware** DataFrame you can keep ``append``-ing finer bars
    /// into (the forming period is the live last row).
    ///
    /// Requires a DatetimeIndex. Each column is aggregated with a sensible
    /// default (open=first, high=max, low=min, close=last, volume=sum); override
    /// per column via ``cumulators``. If the source already has a ``time_frame``,
    /// the target must be a whole multiple of it (e.g. 5m→15m, not 5m→7m, and not
    /// a week/3-day into a month); cumulating to the *same* frame is a ``copy()``.
    ///
    /// Args:
    ///     time_frame (str | TimeFrame): the target bucket, e.g. ``'1d'``,
    ///         ``'15m'``, ``'1w'``.
    ///     cumulators (dict[str, str], optional): per-column aggregator
    ///         overrides, e.g. ``{'volume': 'sum', 'close': 'last'}``.
    ///
    /// Usage::
    ///
    ///     daily = df.cumulate('1d')          # a tf-aware 1d frame
    ///     daily.append(intraday_bar)         # folds into the forming day
    ///
    /// Returns:
    ///     DataFrame: the resampled, tf-aware frame.
    #[pyo3(signature = (time_frame, cumulators = None))]
    pub(crate) fn cumulate(
        &self,
        time_frame: &Bound<'_, PyAny>,
        cumulators: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let target = resolve_time_frame(time_frame)?;
        if let Some(tfs) = &self.tf {
            // Same frame is a no-op resample == copy() (keeps the cursor & state).
            if target == tfs.time_frame {
                return Ok(self.copy());
            }
            if !tfs.time_frame.can_coarsen(target) {
                return Err(PyValueError::new_err(format!(
                    "cannot cumulate {} -> {}: the target is not a whole multiple of the source frame",
                    tfs.time_frame.label(),
                    target.label()
                )));
            }
        }
        let spec = build_agg_spec_for(cumulators, Some(self.inner.names()))?;
        let mut cum = Cumulator::new(target, spec.clone());
        cum.append(&self.inner).map_err(pyerr)?;
        let frame = cum.frame().map_err(pyerr)?;
        // The result is a fresh frame (no cached directive columns -> cursor 0)
        // that carries the open period's fine bars so further appends fold in.
        Ok(PyDataFrame {
            inner: frame,
            tf: Some(TfState {
                time_frame: target,
                cumulators: spec,
                open: cum.open_clone(),
            }),
        })
    }

    /// Rename columns (pandas `rename(columns={old: new})`), returning a new
    /// frame.
    #[pyo3(signature = (columns))]
    pub(crate) fn rename(&self, columns: &Bound<'_, PyDict>) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let mut mapping = HashMap::new();
        for (k, v) in columns.iter() {
            mapping.insert(k.extract::<String>()?, v.extract::<String>()?);
        }
        // F39: a rename must not collide two columns onto one name (duplicate
        // column names violate the unique-name contract) — fail-loud (C4).
        let result: Vec<String> = self
            .inner
            .names()
            .iter()
            .map(|n| mapping.get(n).cloned().unwrap_or_else(|| n.clone()))
            .collect();
        let mut seen = std::collections::HashSet::new();
        for n in &result {
            if !seen.insert(n) {
                return Err(PyValueError::new_err(format!(
                    "rename would produce duplicate column {n:?}"
                )));
            }
        }
        Ok(PyDataFrame::plain(
            self.inner.rename(&mapping).map_err(pyerr)?,
        ))
    }

    /// Move a column into the row index (pandas `set_index(col)`), returning a
    /// new frame. A datetime / int / string column becomes the matching index.
    #[pyo3(signature = (keys))]
    pub(crate) fn set_index(&self, keys: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::plain(
            self.inner.set_index(keys).map_err(pyerr)?,
        ))
    }

    /// Cast columns to new dtypes (pandas `astype({col: dtype})`), returning a
    /// new frame.
    pub(crate) fn astype(&self, dtypes: &Bound<'_, PyDict>) -> PyResult<PyDataFrame> {
        let mut df = self.inner.clone();
        let mut mapping = HashMap::new();
        for (k, v) in dtypes.iter() {
            let name = k.extract::<String>()?;
            let dt = v.extract::<String>()?;
            if let Some(unit) = datetime_unit_of(&dt) {
                // datetime target: parse a string column, or scale a numeric epoch
                // column by the dtype's unit (truncating, like a NumPy
                // `datetime64[unit]` cast).
                let col = df.column(&name).map_err(pyerr)?.clone();
                let converted = match &col {
                    Column::Datetime(_) | Column::Str(_, _) => col.to_datetime().map_err(pyerr)?,
                    _ => col.epoch_to_datetime(unit).map_err(pyerr)?,
                };
                df.set_column(&name, converted).map_err(pyerr)?;
            } else {
                mapping.insert(name, parse_dtype(&dt)?);
            }
        }
        if !mapping.is_empty() {
            df = df.astype(&mapping).map_err(pyerr)?;
        }
        Ok(PyDataFrame::plain(df))
    }

    /// Define a column / directive alias: `as_name` resolves to `src_name`
    /// everywhere a column is looked up (mutates in place, pandas-like).
    pub(crate) fn alias(&mut self, as_name: &str, src_name: &str) -> PyResult<()> {
        self.inner = self.inner.with_alias(as_name, src_name).map_err(pyerr)?;
        Ok(())
    }

    /// Refresh the stale tail of every materialized (auto-cached) directive
    /// column at once — the batch form needed before any read that is NOT a
    /// column projection (`to_numpy`, `.iloc`, the reductions, `to_csv`, …),
    /// since those fail loud while the frame is stale. A column read
    /// (`df[directive]` / `df[[...]]`) already auto-refreshes on its own. In
    /// place, incremental — O(lookback + new rows) per column, not O(n).
    pub(crate) fn fulfill(&mut self) -> PyResult<()> {
        self.refresh_computed(None)
    }

    /// pandas-style aligned-table repr: a left-justified index column + right-
    /// justified data columns, truncating to 5 head + 5 tail rows past 60
    /// (`display.max_rows` / `min_rows`) with a `[N rows x M columns]` footer.
    /// `str` and `repr` are identical.
    pub(crate) fn __repr__(&self) -> PyResult<String> {
        ensure_fresh(&self.inner)?;
        let truncate = if self.inner.height() > 60 { Some(5) } else { None };
        let opts = DisplayOpts {
            header: true,
            index: true,
            na_rep: NA_REPR,
            float_format: None,
            dimensions: Dimensions::OnTruncate,
            truncate,
        };
        let cols: Vec<usize> = (0..self.inner.width()).collect();
        Ok(render_frame(&self.inner, &cols, &opts))
    }

    pub(crate) fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    /// Render the whole frame as text (pandas `DataFrame.to_string`), implementing
    /// the core parameters. No truncation by default; `max_rows` truncates to 5
    /// head + 5 tail (or `min_rows`). Legacy / non-applicable pandas params
    /// (`sparsify`, `index_names`, `col_space`, `justify`, `formatters`,
    /// `line_width`, `encoding`, `decimal`, `buf`) are intentionally omitted.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (columns = None, header = true, index = true, na_rep = NA_REPR, float_format = None, max_rows = None, min_rows = None, show_dimensions = false))]
    pub(crate) fn to_string(
        &self,
        columns: Option<Vec<String>>,
        header: bool,
        index: bool,
        na_rep: &str,
        float_format: Option<&str>,
        max_rows: Option<usize>,
        min_rows: Option<usize>,
        show_dimensions: bool,
    ) -> PyResult<String> {
        ensure_fresh(&self.inner)?;
        let ff = parse_ff(float_format)?;
        let col_pos: Vec<usize> = match &columns {
            Some(cols) => cols
                .iter()
                .map(|n| {
                    self.inner
                        .column_pos(n)
                        .ok_or_else(|| PyKeyError::new_err(format!("column \"{n}\" not found")))
                })
                .collect::<PyResult<_>>()?,
            None => (0..self.inner.width()).collect(),
        };
        let truncate = match max_rows {
            Some(m) if self.inner.height() > m => Some((min_rows.unwrap_or(m) / 2).max(1)),
            _ => None,
        };
        let opts = DisplayOpts {
            header,
            index,
            na_rep,
            float_format: ff,
            dimensions: if show_dimensions {
                Dimensions::Always
            } else {
                Dimensions::Never
            },
            truncate,
        };
        Ok(render_frame(&self.inner, &col_pos, &opts))
    }

    /// Rich HTML table for Jupyter (`_repr_html_`). pandas defines this only on
    /// DataFrame — a Series falls back to its text repr — so volas matches and
    /// exposes it on DataFrame alone.
    pub(crate) fn _repr_html_(&self) -> PyResult<String> {
        ensure_fresh(&self.inner)?;
        Ok(render_frame_html(&self.inner))
    }
}

// Internal helpers (a plain impl, NOT `#[pymethods]`, so they are not exposed to
// Python — they back the methods above).
impl PyDataFrame {
    /// Resolve `name` (a real column or a directive), materializing + caching a
    /// directive on first use and refreshing a cached directive's stale tail
    /// (O(lookback); a no-op for a plain column or an already-fresh one). Returns
    /// the resolved column name and a clone of its now-fresh data. Shared by the
    /// single-name and list forms of `__getitem__` so both auto-refresh identically
    /// after an append.
    fn materialize_refresh(&mut self, name: &str) -> PyResult<(String, Column)> {
        if self.inner.has_column(name) {
            self.refresh_computed(Some(name))?;
            let col = self.inner.column(name).map_err(pyerr)?.clone();
            return Ok((name.to_string(), col));
        }
        let node = parse(name).map_err(directive_err)?;
        let canonical = volas_directive::stringify(&node);
        if self.inner.has_column(&canonical) {
            self.refresh_computed(Some(&canonical))?;
        } else {
            let col = execute(&self.inner, &node).map_err(value_err)?;
            let lookback = volas_directive::lookback::lookback(&node);
            // Capture the recursive resume state (if any) BEFORE moving the column in,
            // so a later append can continue in O(new rows) instead of recomputing.
            let state = volas_directive::exec::initial_state(&self.inner, &node, &col);
            self.inner.set_column(&canonical, col).map_err(pyerr)?;
            self.inner
                .set_computed(&canonical, canonical.clone(), lookback);
            self.inner.set_computed_state(&canonical, state);
        }
        let col = self.inner.column(&canonical).map_err(pyerr)?.clone();
        Ok((canonical, col))
    }

    /// A lossless 2-D `object` NumPy array: each cell its own typed Python value
    /// (`volas.Timestamp` / `volas.NA` / str / number) via `scalar_to_py`. Backs
    /// the default mixed-frame export and `to_numpy(dtype="object")`.
    fn object_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cols = self.inner.columns();
        let (h, w) = (self.inner.height(), self.inner.width());
        let rows = PyList::empty(py);
        for i in 0..h {
            let row = PyList::empty(py);
            for col in cols {
                row.append(scalar_to_py(py, col, i))?;
            }
            rows.append(row)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "object")?;
        let arr = py.import("numpy")?.call_method("array", (rows,), Some(&kwargs))?;
        // An empty or single-column frame can collapse to the wrong ndim; pin (h, w).
        arr.call_method1("reshape", ((h, w),))
    }
}
