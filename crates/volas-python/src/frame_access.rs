//! `DataFrame` element access and the live-stream lifecycle
//! (`__getitem__`/`__setitem__`, `append`, `fulfill`).

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use volas_core::{Column, DataFrame};
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
    /// The directive bound against the binding frame (what a refresh continues).
    directive: volas_core::Directive,
    reach: volas_core::Reach,
    col: Column,
    state: Option<Vec<f64>>,
    before: Option<Vec<f64>>,
    /// The checkpoints of its history at the anchor rows it passed.
    anchors: Vec<volas_core::Checkpoint>,
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
        return pyany_to_column(
            &np.call_method1("asarray", (value,))?
                .call_method1("reshape", (1,))?,
        )
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
    /// On a bounded (`window=`) frame a column assignment writes the visible rows only:
    /// the hidden margin keeps its history. Copy-on-write: a prior `copy()` is unaffected.
    pub(crate) fn __setitem__(
        &mut self,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
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
        // The values of the visible rows: a Series (aligned with them), one cell broadcast
        // over them, or an array-like of their length.
        let rows = self.logical_range().1;
        let values = if let Ok(s) = value.extract::<PyRef<PySeries>>() {
            match &self.window {
                None => require_aligned(self.inner.index(), &s.inner.index)?,
                Some(_) => self
                    .meta()
                    .labels()
                    .index()
                    .require_aligned(&s.inner.index)
                    .map_err(pyerr)?,
            }
            s.inner.data.clone()
        } else if let Some(one) = scalar_cell(value)? {
            one.take(&vec![0; rows])
        } else {
            pyany_to_column(value)?
        };
        let col = self.whole_column(&name, values)?;
        // A data column under a name a cached directive reads as a command is refused;
        // replacing a column stales the caches that may read it (inside `set_column`).
        if !self.inner.has_column(&name) || self.inner.is_computed(&name) {
            self.check_new_data_column(&name)?;
        }
        // A time-frame frame takes the column through its one write entry: the open
        // period follows the forming cell it leaves, or the write is refused unchanged.
        match self.tf.as_mut() {
            Some(tfs) => {
                let mut next = self.inner.clone(); // shares every buffer the write keeps
                next.set_column(&name, col).map_err(pyerr)?;
                tfs.write(TfWrite::Frame {
                    inner: &mut self.inner,
                    next,
                })?
            }
            None => self.inner.set_column(&name, col).map_err(pyerr)?,
        }
        Ok(())
    }

    // `df[key]` — column name / indicator directive / list / boolean mask /
    // slice. The user-facing usage lives in the class docstring (pyo3 implements
    // `__getitem__` as a type slot and does not surface its doc comment).
    pub(crate) fn __getitem__(
        &mut self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        // Row-selecting reads (mask / slice) operate on the logical M view, so a
        // windowed frame's hidden margin is never selectable (a zero-cost borrow
        // when unbounded).
        // boolean mask (Series or numpy)
        if let Ok(s) = key.extract::<PyRef<PySeries>>() {
            if let Column::Bool(..) = &s.inner.data {
                // O5: reject an NA-carrying mask (an unknown signal is not False).
                let mask = bool_mask_vec(&s.inner.data)?;
                return Ok(Py::new(py, self.masked(&mask)?)?.into_any());
            }
        }
        if let Ok(arr) = key.extract::<PyReadonlyArray1<bool>>() {
            return Ok(Py::new(py, self.masked(&crate::coerce::numpy_values(&arr))?)?.into_any());
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
                return Ok(Py::new(py, self.masked(&mask)?)?.into_any());
            }
        }
        // label / positional slice: df[:'date'], df[1:5]
        if let Ok(slice) = key.downcast::<PySlice>() {
            let view = self.read()?;
            let sel = slice_rows(&view, slice)?;
            return Ok(Py::new(py, self.derive_rows(&view, &sel)?)?.into_any());
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
            return Ok(Py::new(py, self.derive_projection(df)?)?.into_any());
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
    /// the finer bars appended after it. Every bar is first made a complete bar of the
    /// frame's data columns (a column it leaves out is a missing value); a column
    /// written onto the frame later is missing in the open period's earlier bars.
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
    ///         A scalar dict is one bar — its data columns (one left out is missing, as
    ///         in every form) plus the timestamp under the key equal to the index's
    ///         name (a RangeIndex auto-extends).
    ///
    /// Usage::
    ///
    ///     df.append(bar)                          # append / fold one bar (Row / 1-row frame)
    ///     df.append({'time': ts, 'open': o, ...}) # append / fold one bar (scalar dict)
    ///     df.append(other_frame)                  # append / fold many bars
    ///
    /// Returns:
    ///     DataFrame: ``self`` (enabling chaining).
    pub(crate) fn append<'py>(
        slf: Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
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
        Err(PyTypeError::new_err(
            "append expects a DataFrame, a Row, or a bar dict",
        ))
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
    /// An equivalent canonical spelling among the frame's data columns.
    /// The stored label remains exact so bound data operands keep their owner.
    fn canonical_column_name(frame: &DataFrame, canonical: &str) -> Option<String> {
        frame.names().iter().find(|actual| {
            !frame.is_computed(actual)
                && volas_directive::canonical_data_name(actual).ok().as_deref() == Some(canonical)
        }).cloned()
    }

    /// Resolve a data spelling to the physical label, including directive aliases.
    /// Matching stays at bind/read time; cached trees keep the resulting exact label.
    fn data_column_name(frame: &DataFrame, name: &str) -> Option<String> {
        if frame.has_column(name) && !frame.is_computed(name) {
            return Some(name.to_string());
        }
        let canonical = volas_directive::canonical_data_name(name).ok()?;
        Self::canonical_column_name(frame, &canonical)
    }

    /// An existing physical column named by exact label or an equivalent directive.
    /// Never computes; data columns take precedence over cached directives.
    pub(crate) fn column_name(&self, name: &str) -> Option<String> {
        if self.inner.has_column(name) {
            return Some(name.to_string());
        }
        let canonical = volas_directive::canonical_data_name(name).ok()?;
        if let Some(actual) = Self::canonical_column_name(&self.inner, &canonical) {
            return Some(actual);
        }
        // A cached expression is named by its bound tree, whose data nodes carry
        // physical labels. Resolve those through the same binder as computation;
        // normalizing the requested text alone cannot find such a cache.
        let node = parse(name).ok()?;
        let bound = Self::bind(&self.inner, &node, name).ok()?;
        let canonical = volas_directive::stringify(&bound);
        self.inner.is_computed(&canonical).then_some(canonical)
    }

    /// The column `df[name] = values` stores, `values` being the visible rows' values —
    /// THE one write of a whole column. An unbounded frame stores `values` as they are. A
    /// windowed frame writes its visible rows only: its hidden margin keeps the history it
    /// holds — an existing column's own values, in the new values' dtype, which must hold
    /// them exactly — or, for a new column, missing values.
    fn whole_column(&self, name: &str, values: Column) -> PyResult<Column> {
        let (start, rows) = self.logical_range();
        if start == 0 {
            return Ok(values);
        }
        if values.len() != rows {
            return Err(PyValueError::new_err(format!(
                "length of values ({}) does not match the window length ({rows})",
                values.len()
            )));
        }
        let dtype = values.dtype();
        let inexact = || {
            PyTypeError::new_err(format!(
                "cannot assign {dtype} values to column '{name}' of a windowed frame: its \
                 hidden rows (the max_lookback margin) hold values {dtype} cannot hold exactly"
            ))
        };
        let mut col = match self.inner.column(name) {
            Ok(old) => {
                let hidden = old.slice(0, start);
                let kept = hidden.convert_lossless(dtype).map_err(|_| inexact())?;
                // a float narrowing rounds by design: every hidden value must survive it
                if dtype.is_float() && hidden.require_numeric().is_ok() {
                    let (was, now) = (hidden.to_f64_vec(), kept.to_f64_vec());
                    if was
                        .iter()
                        .zip(&now)
                        .any(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
                    {
                        return Err(inexact());
                    }
                }
                kept
            }
            Err(_) => Column::na_of(dtype, start),
        };
        col.append(&values).map_err(pyerr)?;
        Ok(col)
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
                Some(w) => self.inner.height().saturating_sub(w.window_and_margin()),
                None => 0,
            };
            let retained = volas_core::Retained { from, shown };
            self.inner.append_retaining(rows, retained).map_err(pyerr)?;
        }
        self.maybe_compact();
        Ok(())
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
                self.admit_new(&node, &text)?;
                self.cache_new(&text, &node)
            }
        }
    }

    /// Compute the new directive `node` (spelled `text`) over the binding frame and cache
    /// it: its canonical name and values.
    pub(crate) fn cache_new(
        &mut self,
        text: &str,
        node: &volas_directive::Ast,
    ) -> PyResult<(String, Column)> {
        let new = self.compute_new(&*self.binding_frame()?, node, text)?;
        let canonical = new.canonical.clone();
        Ok((canonical, self.cache(new)?))
    }

    /// What a requested name reads: an existing column (data, or a cached directive under
    /// its canonical name), or a new directive to compute. Writes nothing.
    fn resolve(&self, name: &str) -> PyResult<Resolved> {
        if let Some(actual) = self.column_name(name) {
            return Ok(Resolved::Column(actual));
        }
        Ok(Resolved::New(
            name.to_string(),
            parse(name).map_err(directive_err)?,
        ))
    }

    /// Compute a new directive over `frame` (the binding frame) — writing nothing — with
    /// the resume states as of its last row and the row before, so a later append
    /// continues in O(new rows) and a replaced last row recomputes in one step.
    fn compute_new(
        &self,
        frame: &DataFrame,
        node: &volas_directive::Ast,
        text: &str,
    ) -> PyResult<Computed> {
        let bound = Self::bind(frame, node, text)?;
        let (col, state, before, anchors) =
            volas_directive::exec::execute_for_cache(frame, &bound)
                .map_err(value_err)?;
        Ok(Computed {
            canonical: volas_directive::stringify(&bound),
            reach: volas_directive::lookback::reach(&bound),
            directive: std::sync::Arc::new(bound),
            col,
            state,
            before,
            anchors,
        })
    }

    /// Cache a computed directive under its canonical name and return its values. A
    /// data column already holding that name keeps it: the directive's values are
    /// returned, not cached over the data.
    fn cache(&mut self, new: Computed) -> PyResult<Column> {
        if self.inner.has_column(&new.canonical) {
            return Ok(new.col);
        }
        let Computed {
            canonical,
            directive,
            reach,
            col,
            state,
            before,
            anchors,
        } = new;
        self.inner.set_column(&canonical, col).map_err(pyerr)?;
        self.inner.set_computed(&canonical, directive, reach);
        self.inner.set_computed_states(&canonical, state, before);
        // computed over the binding frame from its row 0: the anchors its history passed
        self.inner.keep_computed_anchors(&canonical, anchors);
        Ok(self.inner.column(&canonical).map_err(pyerr)?.clone())
    }

    /// `df[[...]]` (and the constructor's `columns=`): the named columns / directives
    /// as a plain frame over the visible rows. Every entry is resolved and every new
    /// directive computed before any is cached, so a bad entry caches nothing; cached
    /// entries are refreshed like single-column reads.
    pub(crate) fn project(&mut self, list: &[String]) -> PyResult<DataFrame> {
        let resolved = list
            .iter()
            .map(|n| self.resolve(n))
            .collect::<PyResult<Vec<_>>>()?;
        for r in &resolved {
            if let Resolved::New(text, node) = r {
                self.admit_new(node, text)?;
            }
        }
        let mut computed: Vec<Computed> = Vec::new();
        {
            let frame = self.binding_frame()?;
            for r in &resolved {
                if let Resolved::New(text, node) = r {
                    let canonical = volas_directive::stringify(node.as_ref());
                    if !computed.iter().any(|c| c.canonical == canonical) {
                        computed.push(self.compute_new(&frame, node, text)?);
                    }
                }
            }
        }
        for c in computed {
            self.cache(c)?;
        }
        let mut names = Vec::with_capacity(list.len());
        for r in &resolved {
            names.push(match r {
                Resolved::Column(name) => {
                    self.refresh_computed(Some(name))?;
                    name.clone()
                }
                Resolved::New(text, node) => {
                    let canonical = volas_directive::stringify(node.as_ref());
                    if !self.inner.is_computed(&canonical) {
                        // not cached: a data column holds its name (see `cache`)
                        return Err(PyValueError::new_err(format!(
                            "cannot select '{text}' into a frame: it is named '{canonical}' like a \
                             data column of this DataFrame, so it is not cached under that name. \
                             Read it with df.exec('{text}'), or rename the data column."
                        )));
                    }
                    canonical
                }
            });
        }
        // the listed columns over the same rows, each as data (R40), its visible rows
        let df = self.inner.select(&names).map_err(pyerr)?;
        let (start, len) = self.logical_range();
        Ok(if self.window.is_some() {
            df.rows(start, start + len, &Replayer)
        } else {
            df
        })
    }

    /// Bind a new directive (`directive` as written, parsed to `node`) against `frame` —
    /// the binding frame (see [`bind`](Self::bind)) — and compute it over that frame.
    pub(crate) fn evaluate(
        &self,
        frame: &DataFrame,
        node: &volas_directive::Ast,
        directive: &str,
    ) -> PyResult<Column> {
        execute(frame, &Self::bind(frame, node, directive)?).map_err(value_err)
    }

    /// Bind a new directive (`directive` as written, parsed to `node`) against `frame`:
    /// a physical data column owns every equivalent directive spelling; otherwise
    /// the name binds to its built-in command.
    pub(crate) fn bind(
        frame: &DataFrame,
        node: &volas_directive::Ast,
        directive: &str,
    ) -> PyResult<volas_directive::Ast> {
        volas_directive::bind_frame_resolved(node, &|n| Self::data_column_name(frame, n), directive)
            .map_err(value_err)
    }

    /// Refuse data whose equivalent directive spelling shadows a computation within a
    /// cached bound tree. Checking happens on schema changes, outside the refresh path.
    fn check_new_data_column(&self, name: &str) -> PyResult<()> {
        let canonical =
            volas_directive::canonical_data_name(name).unwrap_or_else(|_| name.to_string());
        let mut users = Vec::new();
        for cached in self.inner.names().iter().filter(|n| n.as_str() != name) {
            let reads = self.inner.computed_directive(cached).map(bound_directive);
            if reads.is_some_and(|node| volas_directive::uses_command_shadow(node, &canonical)) {
                users.push(format!("'{cached}'"));
            }
        }
        if users.is_empty() {
            return Ok(());
        }
        let (which, verb, them) = if users.len() == 1 {
            (
                format!("the cached directive {}", users[0]),
                "uses",
                "the cached directive",
            )
        } else {
            (
                format!("the cached directives {}", users.join(", ")),
                "use",
                "the cached directives",
            )
        };
        Err(PyValueError::new_err(format!(
            "cannot add column '{name}': {which} {verb} the built-in command '{command}' under \
             that name. Rename the new column, or drop {them} first — \
             df.drop([{list}], axis=1).",
            command = canonical,
            list = users.join(", ")
        )))
    }
}
