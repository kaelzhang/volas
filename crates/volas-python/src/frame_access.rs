//! `DataFrame` element access and the live-stream lifecycle
//! (`__getitem__`/`__setitem__`, `append`, `fulfill`).


use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use volas_core::{
    Column, DataFrame,
};
use volas_directive::{execute, parse};

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
        // A windowed frame's column spans the full physical buffer, but the user
        // supplies a value for the logical window M. A scalar broadcasts across the
        // whole buffer (the visible rows see it); an array-like is logical-length and
        // NA-pads the hidden margin (front). Identity when unbounded (start == 0).
        let (start, logical_h) = self.logical_range();
        let pad = |c: Column| -> PyResult<Column> {
            if start == 0 {
                return Ok(c);
            }
            if c.len() != logical_h {
                return Err(PyValueError::new_err(format!(
                    "length of values ({}) does not match the window length ({logical_h})",
                    c.len()
                )));
            }
            let positions: Vec<usize> = (start..h).collect();
            Column::na_of(c.dtype(), h).scatter(&positions, &c).map_err(pyerr)
        };
        let col = if let Ok(s) = value.extract::<PyRef<PySeries>>() {
            pad(s.inner.data.clone())?
        } else if let Ok(b) = value.extract::<bool>() {
            Column::bool(vec![b; h])
        } else if let Ok(scalar) = value.extract::<f64>() {
            Column::f64(vec![scalar; h])
        } else {
            pad(pyany_to_column(value)?)?
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
        // Row-selecting reads (mask / slice) operate on the logical M view, so a
        // windowed frame's hidden margin is never selectable (a zero-cost borrow
        // when unbounded).
        // boolean mask (Series or numpy)
        if let Ok(s) = key.extract::<PyRef<PySeries>>() {
            if let Column::Bool(..) = &s.inner.data {
                // O5: reject an NA-carrying mask (an unknown signal is not False).
                let mask = bool_mask_vec(&s.inner.data)?;
                let sub = self.logical().filter_mask(&mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
            }
        }
        if let Ok(arr) = key.extract::<PyReadonlyArray1<bool>>() {
            let sub = self.logical().filter_mask(arr.as_slice()?).map_err(pyerr)?;
            return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
        }
        // boolean mask as a plain Python list (df[[True, False, ...]]). An empty
        // list is an empty column projection, not a mask, so it falls through.
        if let Ok(mask) = key.extract::<Vec<bool>>() {
            if !mask.is_empty() {
                let height = self.logical_range().1;
                if mask.len() != height {
                    return Err(PyIndexError::new_err(format!(
                        "boolean index has wrong length: {} instead of {}",
                        mask.len(),
                        height
                    )));
                }
                let sub = self.logical().filter_mask(&mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
            }
        }
        // label / positional slice: df[:'date'], df[1:5]
        if let Ok(slice) = key.downcast::<PySlice>() {
            let sub = slice_frame(self.logical().as_ref(), slice)?;
            return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
        }
        // column name or directive — materialize + auto-refresh a stale cached
        // directive (O(lookback), not O(n)) on the physical frame, then present the
        // result sliced to the logical M view.
        if let Ok(name) = key.extract::<String>() {
            let (resolved, col) = self.materialize_refresh(&name)?;
            return Ok(Py::new(py, self.present_series(resolved, col))?.into_any());
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
            let (start, len) = self.logical_range();
            let df = if self.window.is_some() { df.slice(start, start + len) } else { df };
            return Ok(Py::new(py, PyDataFrame::plain(df))?.into_any());
        }
        Err(PyKeyError::new_err(
            "key must be a column name, directive, list, boolean mask, or slice",
        ))
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
    ///     other (DataFrame | Row | dict): the rows to append (fine bars if tf-aware).
    ///         A scalar dict is one bar — every data column plus the timestamp under the
    ///         key equal to the index's name (a RangeIndex auto-extends).
    ///
    /// Usage::
    ///
    ///     df.append(bar)                          # append / fold one bar (Row / 1-row frame)
    ///     df.append({'time': ts, 'open': o, ...}) # append / fold one bar (scalar dict)
    ///     df.append(other_frame)                  # append / fold many bars
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
                me.maybe_compact()?;
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
            me.maybe_compact()?;
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
            me.maybe_compact()?;
            return Ok(slf);
        }
        // A scalar bar dict — the timestamp under the index-name key. Built straight into a
        // one-row frame (no intermediate Python DataFrame), then folded / appended.
        if let Ok(dict) = other.downcast::<pyo3::types::PyDict>() {
            let mut me = slf.borrow_mut();
            let bar = me.bar_from_dict(dict)?;
            if me.tf.is_some() {
                me.fold_append(&bar)?;
            } else {
                me.inner.append(&bar).map_err(pyerr)?;
            }
            me.maybe_compact()?;
            return Ok(slf);
        }
        Err(PyTypeError::new_err("append expects a DataFrame, a Row, or a bar dict"))
    }

    /// Whether the column ``name`` is a cached **directive (computed)** column —
    /// one derived from a directive (e.g. ``df['rsi:14']``) and refreshed by
    /// ``fulfill`` — rather than a plain data column supplied per bar.
    ///
    /// Raises:
    ///     KeyError: if ``name`` is not a column of the frame.
    ///
    /// Returns:
    ///     bool
    pub(crate) fn is_computed(&self, name: &str) -> PyResult<bool> {
        if !self.inner.has_column(name) {
            return Err(PyKeyError::new_err(format!("column {name:?} not found")));
        }
        Ok(self.inner.is_computed(name))
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
}

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
            self.inner.set_computed(&canonical, lookback);
            self.inner.set_computed_state(&canonical, state);
        }
        let col = self.inner.column(&canonical).map_err(pyerr)?.clone();
        Ok((canonical, col))
    }
}
