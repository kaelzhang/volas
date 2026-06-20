//! `DataFrame` construction, the windowing setup, and frame-level accessors
//! (timezone, shape, index, dtypes, the `.iloc`/`.loc`/`.iat`/`.at` accessors).

use std::collections::HashSet;

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use volas_core::{
    CmpOp, DataFrame, Index, IndexKind, Tz,
};

use crate::timeframe::{build_agg_spec_for, resolve_time_frame};
#[allow(unused_imports)]
use crate::*;

/// The windowed-frame lookback bound — how `max_lookback=` is supplied: either an
/// explicit row count, or a list of indicator directives whose largest lookback it is
/// derived from (so the caller never hand-computes a compound indicator's warm-up).
#[derive(FromPyObject)]
pub(crate) enum LookbackBound {
    Count(usize),
    Directives(Vec<String>),
}

/// Validate the windowing params and derive the retention capacity. `max_lookback` only
/// applies with `window`; with `window` it is REQUIRED and supplies the lookback bound
/// (`capacity = window + lookback`) — directly as an int, or as the largest lookback
/// among a list of indicator directives. See `designs/...-windowed-dataframe`.
fn build_window_state(
    window: Option<usize>,
    max_lookback: Option<LookbackBound>,
) -> PyResult<Option<WindowState>> {
    let Some(w) = window else {
        if max_lookback.is_some() {
            return Err(PyValueError::new_err(
                "max_lookback only applies to a windowed frame — pass window=",
            ));
        }
        return Ok(None);
    };
    if w == 0 {
        return Err(PyValueError::new_err("window must be a positive integer"));
    }
    let lookback = match max_lookback {
        None => {
            return Err(PyValueError::new_err(
                "a windowed frame needs its lookback bound — pass max_lookback= \
                 (an int, or a list of indicator directives to derive it from)",
            ))
        }
        Some(LookbackBound::Count(l)) => l,
        Some(LookbackBound::Directives(inds)) => {
            let mut max = 0usize;
            for d in &inds {
                let node = volas_directive::parse(d).map_err(pyerr)?;
                max = max.max(volas_directive::lookback::lookback(&node));
            }
            max
        }
    };
    Ok(Some(WindowState { window: w, capacity: w + lookback }))
}

#[pymethods]
impl PyDataFrame {
    // Constructor — the user-facing argument list & usage live in the class
    // docstring (pyo3 does not surface a `#[new]` doc comment to Python).
    #[new]
    #[pyo3(signature = (data, columns = None, index = None, time_frame = None, cumulators = None, dtype = None, window = None, max_lookback = None))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        data: &Bound<'_, PyAny>,
        columns: Option<Vec<String>>,
        index: Option<&Bound<'_, PyAny>>,
        time_frame: Option<&Bound<'_, PyAny>>,
        cumulators: Option<&Bound<'_, PyDict>>,
        dtype: Option<&str>,
        window: Option<usize>,
        max_lookback: Option<LookbackBound>,
    ) -> PyResult<Self> {
        // `columns`, when given, selects and orders the columns — a strict projection, like
        // `df[[...]]`: a name not present raises KeyError, and an empty list or a duplicate
        // name is rejected. It never silently NaN-fills an absent column.
        if let Some(cols) = &columns {
            if cols.is_empty() {
                return Err(PyValueError::new_err("columns must not be empty"));
            }
            let mut seen = HashSet::with_capacity(cols.len());
            for c in cols {
                if !seen.insert(c.as_str()) {
                    return Err(PyValueError::new_err(format!(
                        "duplicate column \"{c}\" in columns"
                    )));
                }
            }
        }
        // `data` is polymorphic over volas's own inputs: another volas DataFrame (copied —
        // index, aliases and any tf-state carried, exactly like `df.copy()`), or a dict of
        // columns (a fresh RangeIndex); with `columns` the frame is projected onto them. A
        // pandas DataFrame is deliberately NOT accepted here — use `DataFrame.from_pandas`, which keeps
        // volas pandas-free at import. To build a DatetimeIndex from a column, parse it with
        // `to_datetime` then `set_index` (or use `read_csv`).
        let (df, tf) = if let Ok(other) = data.extract::<PyRef<PyDataFrame>>() {
            match &columns {
                None => (other.inner.clone(), other.tf.clone()),
                Some(cols) => {
                    // Project the frame, and a tf-aware frame's forming-period state, onto
                    // `cols`. The cumulator spec is per-column with a default, so the dropped
                    // columns' rules simply go unused — folding stays correct on the kept ones.
                    let inner = other.inner.select(cols).map_err(pyerr)?;
                    let tf = match &other.tf {
                        None => None,
                        Some(t) => Some(TfState {
                            time_frame: t.time_frame,
                            cumulators: t.cumulators.clone(),
                            open: t
                                .open
                                .as_ref()
                                .map(|o| o.select(cols))
                                .transpose()
                                .map_err(pyerr)?,
                            // projection changes the columns -> rebuild the plan lazily.
                            fold_plan: None,
                        }),
                    };
                    (inner, tf)
                }
            }
        } else if let Ok(dict) = data.downcast::<PyDict>() {
            let (names, vcols) = match &columns {
                None => {
                    let mut names = Vec::new();
                    let mut vcols = Vec::new();
                    for (k, v) in dict.iter() {
                        names.push(k.extract::<String>()?);
                        vcols.push(pyany_to_column(&v)?);
                    }
                    (names, vcols)
                }
                Some(cols) => {
                    // Strict select: build only the named columns, in order.
                    let mut vcols = Vec::with_capacity(cols.len());
                    for name in cols {
                        let v = dict.get_item(name)?.ok_or_else(|| {
                            PyKeyError::new_err(format!("column \"{name}\" not found"))
                        })?;
                        vcols.push(pyany_to_column(&v)?);
                    }
                    (cols.clone(), vcols)
                }
            };
            (DataFrame::new(names, vcols, None).map_err(pyerr)?, None)
        } else {
            return Err(PyTypeError::new_err(
                "DataFrame(data): data must be a dict of columns or a volas DataFrame \
                 (for a pandas DataFrame use DataFrame.from_pandas)",
            ));
        };
        // F45: an explicit `index=` attaches row labels at construction (pandas
        // `DataFrame(data, index=...)`): a list / array of int64, str or
        // datetime labels — the same kinds (and uniqueness rules) as set_index.
        let df = match index {
            None => df,
            Some(ix) => {
                let col = pyany_to_column(ix)?;
                let new_index = Index::from_column(&col).map_err(pyerr)?;
                DataFrame::new(df.names().to_vec(), df.columns().to_vec(), Some(new_index))
                    .map_err(pyerr)?
            }
        };
        // `dtype=` casts every column to a single dtype (pandas `DataFrame(data,
        // dtype=...)`), e.g. dtype='float32'.
        let df = match dtype {
            None => df,
            Some(dt_str) => {
                let dt = parse_dtype(dt_str)?;
                let cols = df
                    .columns()
                    .iter()
                    .map(|c| c.cast(dt))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(pyerr)?;
                DataFrame::new(df.names().to_vec(), cols, Some((**df.index()).clone()))
                    .map_err(pyerr)?
            }
        };
        // `window=` makes this a bounded rolling-window frame (see designs): derive the
        // retention capacity (window + max_lookback) and bound the initial data to it.
        let win = build_window_state(window, max_lookback)?;
        let df = match &win {
            Some(w) if df.height() > w.capacity => df.slice(df.height() - w.capacity, df.height()),
            _ => df,
        };
        // A `time_frame` makes this a cumulating frame: the given rows are taken as
        // already-final bars at that frame (not re-aggregated), and later `append`s fold
        // finer bars into them. Requires a DatetimeIndex (build one with `set_index` first).
        if let Some(tf_obj) = time_frame {
            let frame = resolve_time_frame(tf_obj)?;
            match df.index().kind() {
                IndexKind::Datetime(v, _) => {
                    // The rows are taken as final bars and the last row becomes the
                    // forming-period cursor, so the live-fold invariants must hold
                    // from the start: a present, non-decreasing timeline — symmetric
                    // with the append-fold and batch-cumulate guards.
                    if v.contains(&i64::MIN) {
                        return Err(PyValueError::new_err(
                            "time_frame requires a DatetimeIndex without NaT; drop or \
                             fill the missing index timestamps first",
                        ));
                    }
                    if v.windows(2).any(|w| w[1] < w[0]) {
                        return Err(PyValueError::new_err(
                            "time_frame requires a monotonic (time-sorted) \
                             DatetimeIndex; sort the bars by time first",
                        ));
                    }
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "time_frame requires a DatetimeIndex \
                         (build one with to_datetime(df[col]) then df.set_index(col))",
                    ));
                }
            }
            let spec = build_agg_spec_for(cumulators, Some(df.names()))?;
            return Ok(PyDataFrame {
                inner: df,
                tf: Some(TfState {
                    time_frame: frame,
                    cumulators: spec,
                    open: None,
                    fold_plan: None,
                }),
                window: win,
            });
        }
        if cumulators.is_some() {
            return Err(PyValueError::new_err("cumulators requires time_frame"));
        }
        Ok(PyDataFrame { inner: df, tf, window: win })
    }

    /// The DatetimeIndex timezone name (`"UTC"` / `"+08:00"` /
    /// `"America/New_York"`), or `None` for a tz-naive or non-datetime index —
    /// mirroring pandas `df.index.tz`. UTC-aware reports `"UTC"` (F13): a naive
    /// axis and a UTC-anchored one are different states.
    #[getter]
    pub(crate) fn tz(&self) -> Option<String> {
        match self.inner.index().tz() {
            Tz::Naive => None,
            other => Some(other.name()),
        }
    }

    /// Reinterpret the index wall-clock as `tz` (pandas `tz_localize`): the
    /// displayed wall-clock is unchanged, each instant is recomputed. Use when
    /// data was ingested without a tz. Returns a new frame.
    pub(crate) fn tz_localize(&self, tz: &str) -> PyResult<PyDataFrame> {
        let tzv = Tz::parse(tz).map_err(pyerr)?;
        Ok(PyDataFrame::plain(
            self.inner.tz_localize(tzv).map_err(pyerr)?,
        ))
    }

    /// Change the index display / matching tz without moving any instant (pandas
    /// `tz_convert`). Returns a new frame.
    /// Tag the index's zone directly (interop-internal): `from_pandas` imports a
    /// tz-aware pandas index whose instants are ALREADY true UTC, so the zone is
    /// attached without the naive-guard that protects user-facing `tz_convert`.
    pub(crate) fn _set_index_tz(&self, tz: &str) -> PyResult<PyDataFrame> {
        let tzv = Tz::parse(tz).map_err(pyerr)?;
        Ok(PyDataFrame::plain(self.inner.set_index_tz(tzv).map_err(pyerr)?))
    }

    pub(crate) fn tz_convert(&self, tz: &str) -> PyResult<PyDataFrame> {
        let tzv = Tz::parse(tz).map_err(pyerr)?;
        Ok(PyDataFrame::plain(
            self.inner.tz_convert(tzv).map_err(pyerr)?,
        ))
    }

    /// The column names, in order.
    ///
    /// Returns:
    ///     list[str]
    #[getter]
    pub(crate) fn columns(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// The frame dimensions as ``(n_rows, n_columns)`` (pandas ``shape``).
    ///
    /// Returns:
    ///     tuple[int, int]
    #[getter]
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.logical_range().1, self.inner.width())
    }

    /// Whether a windowed frame has warmed up — at least `window + max_lookback`
    /// rows accumulated, so every one of the `window` visible rows has a fully
    /// valid indicator history. Always ``True`` for an unbounded frame (no
    /// warm-up contract). See the ``window=`` constructor argument.
    ///
    /// Returns:
    ///     bool
    #[getter]
    pub(crate) fn ready(&self) -> bool {
        match &self.window {
            Some(w) => self.inner.height() >= w.capacity,
            None => true,
        }
    }

    /// The physical row count actually retained (window M + margin, bounded by
    /// `2*(window + max_lookback)`). Internal — equals ``len(df)`` for an unbounded
    /// frame; exposed so the test suite can assert a windowed frame's memory stays
    /// bounded across unbounded appends.
    #[getter]
    pub(crate) fn _physical_height(&self) -> usize {
        self.inner.height()
    }

    /// The row index as a NumPy array (``datetime64[ns]`` for a DatetimeIndex,
    /// an object array for a string index, else an integer array).
    ///
    /// Returns:
    ///     numpy.ndarray
    #[getter]
    pub(crate) fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, self.logical().index())
    }

    // The indexers hold a live reference to this frame (`Py<PyDataFrame>`), not a
    // snapshot, so `df.iloc[...] = ` / `df.loc[...] = ` mutate the frame in place
    // (copy-on-write under the hood) and reads always see the current rows.

    /// Purely integer-location indexing for selection and assignment.
    ///
    /// Get ``df.iloc[i]`` (a row), ``df.iloc[a:b]`` (a sub-frame),
    /// ``df.iloc[i, j]`` (a cell), ``df.iloc[:, j]`` (a column as a Series), or
    /// ``df.iloc[rows, cols]`` (a sub-frame). Assign ``df.iloc[rows, j] = value``
    /// (copy-on-write; a prior ``copy()`` is unaffected).
    ///
    /// Usage::
    ///
    ///     df.iloc[0]            # first row
    ///     df.iloc[-1, 3]        # last row, 4th column -> scalar
    ///     df.iloc[:, 0]         # first column as a Series
    ///     df.iloc[10:20, 0:2]   # a block
    ///     df.iloc[mask, 1] = 0  # assign a column where a boolean mask is True
    #[getter]
    pub(crate) fn iloc(slf: Bound<'_, Self>) -> DataFrameILoc {
        DataFrameILoc {
            parent: slf.unbind(),
        }
    }

    /// Label-based indexing for selection and assignment.
    ///
    /// Get ``df.loc[label]`` (a row), ``df.loc[a:b]`` (a stop-inclusive label
    /// slice), ``df.loc[label, col]`` (a cell), ``df.loc[:, col]`` (a column),
    /// or ``df.loc[mask, col]``. Assign ``df.loc[mask, 'signal'] = 1``
    /// (copy-on-write).
    ///
    /// Usage::
    ///
    ///     df.loc['2021-01-04']               # row by datetime label
    ///     df.loc['2021-01':'2021-03']        # inclusive label slice
    ///     df.loc[df['close'] > df['open'], 'signal'] = 1
    #[getter]
    pub(crate) fn loc(slf: Bound<'_, Self>) -> DataFrameLoc {
        DataFrameLoc {
            parent: slf.unbind(),
        }
    }

    /// Fast scalar access by integer position: ``df.iat[i, j]`` to get or set a
    /// single cell (copy-on-write).
    ///
    /// Usage::
    ///
    ///     df.iat[0, 3]        # the cell at row 0, column 3
    ///     df.iat[0, 3] = 1.5  # set it
    #[getter]
    pub(crate) fn iat(slf: Bound<'_, Self>) -> DataFrameIat {
        DataFrameIat {
            parent: slf.unbind(),
        }
    }

    /// Fast scalar access by label + column name: ``df.at[label, col]`` to get
    /// or set a single cell (copy-on-write).
    ///
    /// Usage::
    ///
    ///     df.at['2021-01-04', 'close']         # one cell
    ///     df.at['2021-01-04', 'close'] = 100.0 # set it
    #[getter]
    pub(crate) fn at(slf: Bound<'_, Self>) -> DataFrameAt {
        DataFrameAt {
            parent: slf.unbind(),
        }
    }

    pub(crate) fn __len__(&self) -> usize {
        self.logical_range().1
    }

    /// `name in df` — whether a column exists (alias-aware).
    pub(crate) fn __contains__(&self, key: &str) -> bool {
        self.inner.has_column(key)
    }

    /// `for x in df` — iterate the column names (pandas semantics).
    pub(crate) fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let names = PyList::new(py, self.inner.names())?;
        Ok(names.try_iter()?.into_any().unbind())
    }

    /// Guard the ambiguous `if df:` footgun: a DataFrame has no single truth
    /// value (pandas-style).
    pub(crate) fn __bool__(&self) -> PyResult<bool> {
        Err(PyValueError::new_err(
            "The truth value of a DataFrame is ambiguous — use len(df) or an explicit reduction",
        ))
    }

    /// Element-wise `==` -> a bool DataFrame (pandas semantics), not identity. The
    /// operand is another DataFrame (same columns + shared index) or a scalar.
    pub(crate) fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        self.compare(other, CmpOp::Eq)
    }

    /// Element-wise `!=` -> a bool DataFrame.
    pub(crate) fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        self.compare(other, CmpOp::Ne)
    }

    /// Per-column dtypes as `{name: dtype_str}` (pandas `dtypes`).
    #[getter]
    pub(crate) fn dtypes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            d.set_item(name, col.dtype().to_string())?;
        }
        Ok(d)
    }
}
