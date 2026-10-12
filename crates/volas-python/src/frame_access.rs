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

/// A requested name, resolved against the frame (see `PyDataFrame::resolve`).
enum Resolved {
    /// An existing column: data, or a cached directive under its canonical name.
    Column(String),
    /// A new directive: as written, and parsed.
    New(String, std::rc::Rc<volas_directive::Ast>),
}

/// A new directive computed over the binding frame, not yet cached.
struct Computed {
    canonical: String,
    reach: volas_core::Reach,
    col: Column,
    state: Option<Vec<f64>>,
    before: Option<Vec<f64>>,
}

/// A value assigned to a whole column as the one cell it names, when it is a scalar —
/// classified by its Python type, never by whether it converts to a number: a Python
/// `int` / `float` / `bool` / `str` (an int an exact int64), a NumPy scalar
/// (`np.generic`, keeping its width: `np.int32` an int32 column) or a 0-d array (its
/// one element, likewise). An array (any dimension but 0), a list or a tuple is a
/// sequence of values: `None`.
fn scalar_cell(value: &Bound<'_, PyAny>) -> PyResult<Option<Column>> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString};
    let py = value.py();
    if value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyString>()
    {
        return pyany_to_column(&PyList::new(py, [value])?.into_any()).map(Some);
    }
    let np = py.import("numpy")?;
    if crate::frame_index::is_array(value) {
        return Ok(None);
    }
    if value.is_instance(&np.getattr("ndarray")?)? || value.is_instance(&np.getattr("generic")?)? {
        // one element of the value's own dtype
        return pyany_to_column(&np.call_method1("asarray", (value,))?.call_method1("reshape", (1,))?)
            .map(Some);
    }
    Ok(None)
}

#[pymethods]
impl PyDataFrame {
    /// `df[key] = value`. With a column name, add or replace that column —
    /// `value` may be a scalar (broadcast, keeping its exact type: an int is an int64
    /// column), a 1-D array / list, or a Series (its index must equal the frame's —
    /// the alignment rule of every operand; pairing is by position). With a boolean mask and
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
            match &self.window {
                None => require_aligned(self.inner.index(), &s.inner.index)?,
                Some(_) => self.meta().labels().require_aligned(&s.inner.index).map_err(pyerr)?,
            }
            pad(s.inner.data.clone())?
        } else if let Some(one) = scalar_cell(value)? {
            // one cell, broadcast over every row
            one.take(&vec![0; h])
        } else {
            pad(pyany_to_column(value)?)?
        };
        // A data column under a name a cached directive reads as a command is refused;
        // replacing a column stales the caches that may read it (inside `set_column`).
        if !self.inner.has_column(&name) || self.inner.is_computed(&name) {
            self.check_new_data_column(&name)?;
        }
        self.inner.set_column(&name, col).map_err(pyerr)?;
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
                let sub = self.read()?.filter_mask(&mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
            }
        }
        if let Ok(arr) = key.extract::<PyReadonlyArray1<bool>>() {
            let sub = self.read()?.filter_mask(arr.as_slice()?).map_err(pyerr)?;
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
                let sub = self.read()?.filter_mask(&mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
            }
        }
        // label / positional slice: df[:'date'], df[1:5]
        if let Ok(slice) = key.downcast::<PySlice>() {
            let sub = slice_frame(self.read()?.as_ref(), slice)?;
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
            let df = self.project(&list)?;
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
    /// On a **time_frame** frame (see the constructor / ``cumulate``) each row is
    /// a bar folded into the current period, whose bars are kept in a small frame: a
    /// bar of the open period joins it — replacing a bar with the same timestamp, so a
    /// re-sent bar never double-counts — and the cumulated period replaces the forming
    /// last row (``df.iloc[-1]``); a bar of a later period starts a new row. A bar of
    /// the frame's own time frame stamped at the period start (an exchange kline
    /// re-sent while it forms) therefore replaces the last row whole. The last
    /// constructed row is the open period's first bar, so it must not already hold
    /// the finer bars appended after it.
    ///
    /// On any other frame each row is keyed by its index label: a row labelled like
    /// the last row replaces it whole; a ``DatetimeIndex`` only increases (a row
    /// labelled before the last raises ``ValueError``) and an integer / string index
    /// stays unique (a row labelled like an earlier row raises), the frame unchanged on
    /// an error; unlabelled rows (a ``RangeIndex``) take the next labels.
    ///
    /// Missing columns are NaN-padded; cached directive columns go stale until
    /// ``fulfill()``. The rows may carry only the frame's data columns (a cached
    /// directive column raises ``ValueError`` — select the data columns first), values
    /// its dtypes hold exactly (a lossy value raises ``TypeError``), and labels of its
    /// index kind and awareness; a source frame must be fresh, and only its visible rows
    /// are read. The rows are checked whole before anything is written, so on an error
    /// the frame is unchanged. A snapshot taken via ``copy()`` / ``iloc`` is unaffected
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
            // A source frame is read like any bulk read: its visible rows only, and never
            // while a cached column of it is stale (its placeholders are not data).
            if slf.as_ptr() == other.as_ptr() {
                // `df.append(df)` needs an owned snapshot before taking `self` mutably.
                let rows = df.read()?.into_owned();
                drop(df);
                slf.borrow_mut().append_rows(&rows)?;
                return Ok(slf);
            }
            // Normal live path: append a distinct frame without cloning it.
            let rows = df.read()?;
            slf.borrow_mut().append_rows(&rows)?;
            return Ok(slf);
        }
        if let Ok(row) = other.extract::<PyRef<PyRow>>() {
            slf.borrow_mut().append_rows(&row.inner)?;
            return Ok(slf);
        }
        // A scalar bar dict — the timestamp under the index-name key. Built straight into a
        // one-row frame (no intermediate Python DataFrame), then folded / appended.
        if let Ok(dict) = other.downcast::<pyo3::types::PyDict>() {
            let mut me = slf.borrow_mut();
            let bar = me.bar_from_dict(dict)?;
            me.append_rows(&bar)?;
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
        let column = self
            .column_name(name)
            .ok_or_else(|| PyKeyError::new_err(format!("column {name:?} not found")))?;
        Ok(self.inner.is_computed(&column))
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
    /// The column `name` names: a column of that exact name, or the cached directive a
    /// directive spelled `name` is cached under (`'close * change'` names the column
    /// `close*change`). Never computes; `None` when there is none.
    pub(crate) fn column_name(&self, name: &str) -> Option<String> {
        if self.inner.has_column(name) {
            return Some(name.to_string());
        }
        let canonical = volas_directive::stringify(&*parse(name).ok()?);
        self.inner.is_computed(&canonical).then_some(canonical)
    }

    /// The one path every append form takes once its rows are built: a time-frame frame
    /// folds them, any other frame appends them keyed by label; then a windowed frame
    /// compacts. The rows are checked whole before the first write, so on an error the
    /// frame is unchanged.
    pub(crate) fn append_rows(&mut self, rows: &DataFrame) -> PyResult<()> {
        if self.tf.is_some() {
            self.fold_append(rows)?;
        } else {
            // a windowed frame keeps its labels unique over the rows it retains (the last
            // window + max_lookback), whatever the physical buffer still holds
            let shown = self.logical_range().0;
            let from = match &self.window {
                Some(w) => self.inner.height().saturating_sub(w.capacity),
                None => 0,
            };
            let retained = volas_core::Retained { from, shown };
            self.inner.append_retaining(rows, retained).map_err(pyerr)?;
        }
        self.maybe_compact()
    }

    /// Resolve `name` (a real column or a directive), materializing + caching a
    /// directive on first use and refreshing a cached directive's stale tail
    /// (O(lookback); a no-op for a plain column or an already-fresh one). Returns
    /// the resolved column name and a clone of its now-fresh data. Shared by the
    /// single-name and list forms of `__getitem__` so both auto-refresh identically
    /// after an append.
    fn materialize_refresh(&mut self, name: &str) -> PyResult<(String, Column)> {
        match self.resolve(name)? {
            Resolved::Column(name) => {
                self.refresh_computed(Some(&name))?;
                let col = self.inner.column(&name).map_err(pyerr)?.clone();
                Ok((name, col))
            }
            Resolved::New(text, node) => {
                let new = self.compute_new(&*self.binding_frame()?, &node, &text)?;
                let canonical = new.canonical.clone();
                Ok((canonical, self.cache(new)?))
            }
        }
    }

    /// What a requested name reads: an existing column (data, or a cached directive under
    /// its canonical name), or a new directive to compute. Writes nothing.
    fn resolve(&self, name: &str) -> PyResult<Resolved> {
        if self.inner.has_column(name) {
            return Ok(Resolved::Column(name.to_string()));
        }
        let node = parse(name).map_err(directive_err)?;
        let canonical = volas_directive::stringify(&node);
        if self.inner.is_computed(&canonical) {
            return Ok(Resolved::Column(canonical));
        }
        Ok(Resolved::New(name.to_string(), node))
    }

    /// Compute a new directive over `frame` (the binding frame) — writing nothing — with
    /// the resume states as of its last row and the row before, so a later append
    /// continues in O(new rows) and a replaced last row recomputes in one step.
    fn compute_new(&self, frame: &DataFrame, node: &volas_directive::Ast, text: &str) -> PyResult<Computed> {
        let col = self.evaluate(frame, node, text)?;
        let (state, before) = Self::capture_resume_states(frame, node, &col);
        Ok(Computed {
            canonical: volas_directive::stringify(node),
            reach: volas_directive::lookback::reach(node),
            col,
            state,
            before,
        })
    }

    /// Cache a computed directive under its canonical name and return its values. A
    /// data column holding that name (`change:` renders as `change`) keeps it: the
    /// command's values are returned, not cached over the data.
    fn cache(&mut self, new: Computed) -> PyResult<Column> {
        if self.inner.has_column(&new.canonical) {
            return Ok(new.col);
        }
        let Computed { canonical, reach, col, state, before } = new;
        self.inner.set_column(&canonical, col).map_err(pyerr)?;
        self.inner.set_computed(&canonical, reach);
        self.inner.set_computed_states(&canonical, state, before);
        Ok(self.inner.column(&canonical).map_err(pyerr)?.clone())
    }

    /// `df[[...]]` (and the constructor's `columns=`): the named columns / directives
    /// as a plain frame over the visible rows. Every entry is resolved and every new
    /// directive computed before any is cached, so a bad entry caches nothing; cached
    /// entries are refreshed like single-column reads.
    pub(crate) fn project(&mut self, list: &[String]) -> PyResult<DataFrame> {
        let resolved = list.iter().map(|n| self.resolve(n)).collect::<PyResult<Vec<_>>>()?;
        let mut computed: Vec<Computed> = Vec::new();
        {
            let frame = self.binding_frame()?;
            for r in &resolved {
                if let Resolved::New(text, node) = r {
                    let canonical = volas_directive::stringify(node);
                    if !computed.iter().any(|c| c.canonical == canonical) {
                        computed.push(self.compute_new(&frame, node, text)?);
                    }
                }
            }
        }
        let mut new_cols: Vec<(String, Column)> = Vec::with_capacity(computed.len());
        for c in computed {
            let canonical = c.canonical.clone();
            new_cols.push((canonical, self.cache(c)?));
        }
        let mut cols = Vec::with_capacity(list.len());
        for r in &resolved {
            cols.push(match r {
                Resolved::Column(name) => {
                    self.refresh_computed(Some(name))?;
                    self.inner.column(name).map_err(pyerr)?.clone()
                }
                Resolved::New(_, node) => {
                    let canonical = volas_directive::stringify(node);
                    new_cols.iter().find(|(n, _)| *n == canonical).expect("computed above").1.clone()
                }
            });
        }
        let idx = (*self.inner.index().as_ref()).clone();
        let df = DataFrame::new(list.to_vec(), cols, Some(idx)).map_err(pyerr)?.with_origin(self.inner.origin());
        let (start, len) = self.logical_range();
        Ok(if self.window.is_some() { df.slice(start, start + len) } else { df })
    }

    /// Bind and compute a new directive (`directive` as written, parsed to `node`) over
    /// `frame` — the binding frame: a bare name that is both a data column and a
    /// built-in command is rejected first.
    pub(crate) fn evaluate(
        &self,
        frame: &DataFrame,
        node: &volas_directive::Ast,
        directive: &str,
    ) -> PyResult<Column> {
        volas_directive::check_ambiguous_names(node, |n| frame.has_column(n), directive)
            .map_err(value_err)?;
        execute(frame, node).map_err(value_err)
    }

    /// Refuse a data column `name` that a cached directive reads as the built-in
    /// command of that name (a bare name in it): the column would change what the cached
    /// directive refreshes. Off the hot path (a column add / overwrite).
    fn check_new_data_column(&self, name: &str) -> PyResult<()> {
        let mut users = Vec::new();
        for cached in self.inner.names().iter().filter(|n| n.as_str() != name && self.inner.is_computed(n)) {
            let node = parse(cached).map_err(directive_err)?;
            if volas_directive::uses_bare_command(&node, name) {
                users.push(format!("'{cached}'"));
            }
        }
        if users.is_empty() {
            return Ok(());
        }
        let (which, verb, them) = if users.len() == 1 {
            (format!("the cached directive {}", users[0]), "uses", "the cached directive")
        } else {
            (format!("the cached directives {}", users.join(", ")), "use", "the cached directives")
        };
        Err(PyValueError::new_err(format!(
            "cannot add column '{name}': {which} {verb} the built-in command '{command}' under \
             that name. Rename the new column, or drop {them} first — \
             df.drop([{list}], axis=1).",
            command = volas_directive::spec::normalize(name),
            list = users.join(", ")
        )))
    }
}
