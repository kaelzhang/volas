//! `DataFrame` methods, part 1: construction, accessors, reductions,
//! element-wise transforms, and missing-value handling.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use volas_core::{
    binary_supertype, stats, CmpOp, Column, DataFrame, Index,
    IndexKind, Series, Tz,
};

use crate::timeframe::{build_agg_spec_for, resolve_time_frame};
#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PyDataFrame {
    // Constructor — the user-facing argument list & usage live in the class
    // docstring (pyo3 does not surface a `#[new]` doc comment to Python).
    #[new]
    #[pyo3(signature = (data, columns = None, index = None, time_frame = None, cumulators = None, dtype = None))]
    pub(crate) fn new(
        data: &Bound<'_, PyAny>,
        columns: Option<Vec<String>>,
        index: Option<&Bound<'_, PyAny>>,
        time_frame: Option<&Bound<'_, PyAny>>,
        cumulators: Option<&Bound<'_, PyDict>>,
        dtype: Option<&str>,
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
        // pandas DataFrame is deliberately NOT accepted here — use `from_pandas`, which keeps
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
                 (for a pandas DataFrame use from_pandas)",
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
                    if v.iter().any(|&t| t == i64::MIN) {
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
                }),
            });
        }
        if cumulators.is_some() {
            return Err(PyValueError::new_err("cumulators requires time_frame"));
        }
        Ok(PyDataFrame { inner: df, tf })
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
        (self.inner.height(), self.inner.width())
    }

    /// The row index as a NumPy array (``datetime64[ns]`` for a DatetimeIndex,
    /// an object array for a string index, else an integer array).
    ///
    /// Returns:
    ///     numpy.ndarray
    #[getter]
    pub(crate) fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, self.inner.index())
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
        self.inner.height()
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

    /// First `n` rows (pandas `head` = `iloc[:n]`, so a negative `n` drops the
    /// last `-n` rows — Python slicing semantics).
    #[pyo3(signature = (n = 5))]
    pub(crate) fn head(&self, n: isize) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let (a, b) = head_tail_window(n, self.inner.height(), true);
        Ok(PyDataFrame::plain(self.inner.slice(a, b)))
    }

    /// Last `n` rows (pandas `tail` = `iloc[-n:]`, so a negative `n` drops the
    /// first `-n` rows — Python slicing semantics).
    #[pyo3(signature = (n = 5))]
    pub(crate) fn tail(&self, n: isize) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let (a, b) = head_tail_window(n, self.inner.height(), false);
        Ok(PyDataFrame::plain(self.inner.slice(a, b)))
    }

    /// Per-column count of non-missing values (pandas `count`) -> a Series indexed
    /// by column name (`int64`), reading each column's validity.
    pub(crate) fn count(&self) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let names: Vec<String> = self.inner.names().to_vec();
        let counts: Vec<i64> = self.inner.columns().iter().map(|c| c.count() as i64).collect();
        Ok(PySeries {
            inner: Series::new(None, Column::i64(counts), Arc::new(Index::str(names))),
        })
    }

    /// Per-column NaN-skipping sum (pandas `df.sum()`; non-numeric skipped).
    pub(crate) fn sum(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| {
            let v = c.to_f64_vec();
            (0..c.len()).filter(|&i| c.is_valid(i) && !v[i].is_nan()).map(|i| v[i]).sum()
        })
    }
    /// Per-column NaN-skipping product (pandas `df.prod()`).
    pub(crate) fn prod(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| {
            let v = c.to_f64_vec();
            (0..c.len())
                .filter(|&i| c.is_valid(i) && !v[i].is_nan())
                .map(|i| v[i])
                .product()
        })
    }
    /// Per-column NaN-skipping mean (pandas `df.mean()`).
    pub(crate) fn mean(&self) -> PyResult<PySeries> {
        self.reduce_with(|s| s.mean_f64())
    }
    /// Per-column sample variance (ddof=1, pandas `df.var()`).
    pub(crate) fn var(&self) -> PyResult<PySeries> {
        self.reduce_with(|s| s.var_f64())
    }
    /// Per-column sample standard deviation (pandas `df.std()`).
    pub(crate) fn std(&self) -> PyResult<PySeries> {
        self.reduce_with(|s| s.var_f64().sqrt())
    }
    /// Per-column NaN-skipping median (pandas `df.median()`).
    pub(crate) fn median(&self) -> PyResult<PySeries> {
        self.try_reduce_with(|s| s.quantile_f64(0.5))
    }
    /// Per-column `q`-quantile (pandas `df.quantile(q)`).
    #[pyo3(signature = (q = 0.5))]
    pub(crate) fn quantile(&self, q: f64) -> PyResult<PySeries> {
        self.try_reduce_with(|s| s.quantile_f64(q))
    }
    /// Per-column NaN-skipping minimum (pandas `df.min()`; numeric columns).
    pub(crate) fn min(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| {
            let v = c.to_f64_vec();
            (0..c.len())
                .filter(|&i| c.is_valid(i) && !v[i].is_nan())
                .map(|i| v[i])
                .fold(f64::NAN, f64::min)
        })
    }
    /// Per-column NaN-skipping maximum (pandas `df.max()`).
    pub(crate) fn max(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| {
            let v = c.to_f64_vec();
            (0..c.len())
                .filter(|&i| c.is_valid(i) && !v[i].is_nan())
                .map(|i| v[i])
                .fold(f64::NAN, f64::max)
        })
    }
    /// Per-column count of distinct present values (pandas `df.nunique()`).
    pub(crate) fn nunique(&self) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let names: Vec<String> = self.inner.names().to_vec();
        let counts: Vec<i64> = self
            .inner
            .columns()
            .iter()
            .map(|c| {
                let mut seen = std::collections::HashSet::new();
                (0..c.len())
                    .filter_map(|i| crate::series::cell_key(c, i))
                    .filter(|k| seen.insert(k.clone()))
                    .count() as i64
            })
            .collect();
        Ok(PySeries {
            inner: Series::new(None, Column::i64(counts), Arc::new(Index::str(names))),
        })
    }
    /// Per-column truthiness `any` (pandas `df.any()`): a present, non-zero /
    /// True / non-empty cell counts.
    pub(crate) fn any(&self) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        Ok(self.bool_reduce(true))
    }
    /// Per-column truthiness `all` (pandas `df.all()`), NA-skipping.
    pub(crate) fn all(&self) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        Ok(self.bool_reduce(false))
    }
    /// Per-column index label of the maximum (pandas `df.idxmax()`).
    pub(crate) fn idxmax(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        ensure_fresh(&self.inner)?;
        self.idx_extreme(py, true)
    }
    /// Per-column index label of the minimum (pandas `df.idxmin()`).
    pub(crate) fn idxmin(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        ensure_fresh(&self.inner)?;
        self.idx_extreme(py, false)
    }

    /// Element-wise membership in `values` -> a boolean frame (pandas `isin`).
    pub(crate) fn isin(&self, values: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.isin(values))
    }
    /// Replace cells equal to `to_replace` with `value`, per column,
    /// dtype-preserving; columns whose dtype cannot hold the scalars are kept
    /// unchanged (pandas `replace`).
    pub(crate) fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        self.map_cols(|s| match s.replace(to_replace, value) {
            Ok(r) => Ok(r),
            Err(_) => Ok(PySeries { inner: s.inner.clone() }),   // dtype can't match -> untouched
        })
    }
    /// The `n` rows with the largest values in `column` (pandas `nlargest`).
    pub(crate) fn nlargest(&self, n: i64, column: &str) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        if n < 0 {
            return Err(PyValueError::new_err("n must be >= 0"));
        }
        self.extreme_rows(n as usize, column, false)
    }
    /// The `n` rows with the smallest values in `column` (pandas `nsmallest`).
    pub(crate) fn nsmallest(&self, n: i64, column: &str) -> PyResult<PyDataFrame> {
        if n < 0 {
            return Err(PyValueError::new_err("n must be >= 0"));
        }
        self.extreme_rows(n as usize, column, true)
    }
    /// Drop later duplicate ROWS, keeping the first (pandas
    /// `drop_duplicates(keep='first')`, over all columns).
    #[pyo3(signature = (keep = "first"))]
    pub(crate) fn drop_duplicates(&self, keep: &str) -> PyResult<PyDataFrame> {
        let dup = self.row_duplicated(keep)?;
        let positions: Vec<usize> = (0..self.inner.height()).filter(|&i| !dup[i]).collect();
        Ok(PyDataFrame::plain(take_frame(&self.inner, &positions)))
    }
    /// True per row for a later duplicate of an earlier row (pandas `duplicated`).
    #[pyo3(signature = (keep = "first"))]
    pub(crate) fn duplicated(&self, keep: &str) -> PyResult<PySeries> {
        let dup = self.row_duplicated(keep)?;
        Ok(PySeries {
            inner: Series::new(None, Column::bool(dup), Arc::clone(self.inner.index())),
        })
    }
    /// The first (smallest-position) mode of each column, as a 1-row frame.
    /// (pandas pads multi-modal columns into extra rows; volas keeps the single
    /// deterministic first mode per column — documented divergence.)
    pub(crate) fn mode(&self) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let mut cols = Vec::with_capacity(self.inner.width());
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            let s = PySeries {
                inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(self.inner.index())),
            };
            let m = s.mode();
            let take: Vec<usize> = if m.inner.is_empty() { vec![] } else { vec![0] };
            cols.push(m.inner.data.take(&take));
        }
        Ok(PyDataFrame::plain(
            DataFrame::new(self.inner.names().to_vec(), cols, None).map_err(pyerr)?,
        ))
    }
    /// Counts of unique values (pandas `df.value_counts()`); volas has no
    /// MultiIndex, so only a single-column frame is supported — call it on the
    /// column (`df[col].value_counts()`) otherwise.
    pub(crate) fn value_counts(&self) -> PyResult<PySeries> {
        if self.inner.width() != 1 {
            return Err(PyTypeError::new_err(
                "DataFrame.value_counts needs a single column (volas has no MultiIndex); \
                 use df[col].value_counts()",
            ));
        }
        let name = self.inner.names()[0].clone();
        let col = self.inner.columns()[0].clone();
        let s = PySeries {
            inner: Series::new(Some(name), col, Arc::clone(self.inner.index())),
        };
        s.value_counts(false, true, false, true)
    }
    /// Linear interpolation per numeric column (pandas `interpolate`).
    pub(crate) fn interpolate(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.interpolate())
    }
    /// A fixed-window rolling aggregator over every numeric column (pandas
    /// `df.rolling(window, min_periods, center)`). Compatibility surface —
    /// prefer directives in live systems (see `Series.rolling`).
    #[pyo3(signature = (window, min_periods = None, center = false))]
    pub(crate) fn rolling(
        &self,
        window: i64,
        min_periods: Option<i64>,
        center: bool,
    ) -> PyResult<PyRollingFrame> {
        if window < 1 {
            return Err(PyValueError::new_err("rolling window must be >= 1"));
        }
        if min_periods.is_some_and(|m| m < 0) {
            return Err(PyValueError::new_err("min_periods must be >= 0"));
        }
        if min_periods.is_some_and(|m| m > window) {
            return Err(PyValueError::new_err("min_periods must be <= window"));
        }
        Ok(PyRollingFrame {
            frame: self.inner.clone(),
            window: window as usize,
            min_periods: min_periods.unwrap_or(window) as usize,
            center,
        })
    }
    /// An expanding (cumulative) aggregator over every numeric column.
    #[pyo3(signature = (min_periods = 1))]
    pub(crate) fn expanding(&self, min_periods: i64) -> PyResult<PyExpandingFrame> {
        if min_periods < 0 {
            return Err(PyValueError::new_err("min_periods must be >= 0"));
        }
        Ok(PyExpandingFrame {
            frame: self.inner.clone(),
            min_periods: min_periods as usize,
        })
    }
    /// An exponentially-weighted aggregator over every numeric column (pandas
    /// `ewm`: exactly one of com / span / halflife / alpha).
    #[pyo3(signature = (com = None, span = None, halflife = None, alpha = None, min_periods = 0, adjust = true, ignore_na = false))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ewm(
        &self,
        com: Option<f64>,
        span: Option<f64>,
        halflife: Option<f64>,
        alpha: Option<f64>,
        min_periods: i64,
        adjust: bool,
        ignore_na: bool,
    ) -> PyResult<PyEwmFrame> {
        if min_periods < 0 {
            return Err(PyValueError::new_err("min_periods must be >= 0"));
        }
        Ok(PyEwmFrame {
            frame: self.inner.clone(),
            alpha: crate::window::resolve_alpha(com, span, halflife, alpha)?,
            adjust,
            ignore_na,
            min_periods: min_periods as usize,
        })
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

    /// Drop rows containing missing values, across every dtype (via the column
    /// validity). `how='any'` (default) drops a row if any column is missing
    /// there; `how='all'` only if every column is missing. An invalid `how`
    /// raises `ValueError`.
    #[pyo3(signature = (how = "any"))]
    pub(crate) fn dropna(&self, how: &str) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        if how != "any" && how != "all" {
            return Err(PyValueError::new_err(format!(
                "dropna: invalid `how` {how:?} (expected 'any' or 'all')"
            )));
        }
        let cols = self.inner.columns();
        let total = cols.len();
        let keep: Vec<usize> = (0..self.inner.height())
            .filter(|&i| {
                let nan = cols.iter().filter(|c| !c.is_valid(i)).count();
                match how {
                    "all" => nan < total.max(1),
                    _ => nan == 0,
                }
            })
            .collect();
        Ok(PyDataFrame::plain(take_frame(&self.inner, &keep)))
    }

    /// Replace missing values with `value` in every column (pandas `fillna`),
    /// dtype-preserving like the Series version: `value` is a typed scalar — a
    /// string fills a str column, a Timestamp / datetime string fills a datetime
    /// column, a number / bool fills the numeric family, and `volas.NA` is a no-op.
    /// A dense (no-hole) column is untouched (so a numeric `fillna(0)` over a mixed
    /// frame skips its holeless str / datetime columns); a column whose hole the
    /// fill can't take raises a `TypeError` (volas has no `object` dtype to mix
    /// types — C4). For directional fill use `ffill` / `bfill` (pandas 3.0 removed
    /// `fillna(method=)`).
    #[pyo3(signature = (value, limit = None))]
    pub(crate) fn fillna(&self, value: &Bound<'_, PyAny>, limit: Option<i64>) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        // Per column, fill the missing cells with the typed scalar (str into str,
        // Timestamp / datetime string into datetime, number / bool into a numeric-
        // family column), mirroring the Series surface. A `volas.NA` fill is a
        // dtype-preserving no-op. A dense (no-hole) column is untouched, so a numeric
        // `fillna(0)` over a mixed frame skips its holeless str / datetime columns
        // and only rejects a column that actually has a hole the fill can't take
        // (C4 — no silent numeric -> non-numeric coercion). Atomic: every column is
        // resolved before any frame is built, so a rejected column mutates nothing.
        if limit.is_some_and(|l| l < 0) {
            return Err(PyValueError::new_err("limit must be >= 0"));
        }
        let na_like = is_na_like_py(value);
        let cols = self
            .inner
            .columns()
            .iter()
            .map(|c| -> PyResult<Column> {
                if c.null_count() == 0 {
                    return Ok(c.clone());
                }
                let n = c.len();
                let kd = c.dtype();
                // `limit` caps how many leading holes are filled, PER COLUMN
                // (pandas semantics); a hole beyond the budget stays missing.
                let mut budget = limit.map(|l| l as usize).unwrap_or(usize::MAX);
                let keep: Vec<bool> = (0..n)
                    .map(|i| {
                        if c.is_valid(i) {
                            true
                        } else if budget > 0 {
                            budget -= 1;
                            false
                        } else {
                            true
                        }
                    })
                    .collect();
                let (other_col, target) = if na_like {
                    (Column::na_of(kd, n), kd)
                } else {
                    let (oc, odt) = scalar_fill_col(value, kd, n)?;
                    let target = if kd.is_float() {
                        kd
                    } else if kd == odt {
                        kd
                    } else {
                        binary_supertype(kd, odt)
                    };
                    (oc, target)
                };
                c.select(&keep, &other_col, target).map_err(pyerr)
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.with_columns(cols)
    }

    /// Forward-fill missing cells in every column (pandas `ffill`), dtype-aware.
    pub(crate) fn ffill(&self) -> PyResult<PyDataFrame> {
        self.fill_dir(true)
    }

    /// Backward-fill missing cells in every column (pandas `bfill`), dtype-aware.
    pub(crate) fn bfill(&self) -> PyResult<PyDataFrame> {
        self.fill_dir(false)
    }

    /// Round each float column to `decimals` places (pandas `round`), banker's
    /// rounding; non-float columns are unchanged.
    #[pyo3(signature = (decimals = 0))]
    pub(crate) fn round(&self, decimals: i32) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        // Round numeric columns dtype-preservingly (banker's f64, integer-exact
        // i64); leave bool / str / datetime untouched, like pandas df.round.
        let cols: Vec<Column> = self
            .inner
            .columns()
            .iter()
            .map(|c| {
                if c.dtype().is_numeric() {
                    c.round(decimals)
                } else {
                    Ok(c.clone())
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(pyerr)?;
        self.with_columns(cols)
    }

    // --- column-wise numeric transforms (-> a new frame, dtype-preserving per
    // column, pandas df.cumsum() etc.). cumulatives / abs / clip keep dtype;
    // diff / shift / rank are always float. -------------------------------------

    /// Column-wise cumulative sum (pandas `cumsum`), dtype-preserving.
    pub(crate) fn cumsum(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cumsum())
    }
    /// Column-wise cumulative maximum (pandas `cummax`).
    pub(crate) fn cummax(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cummax())
    }
    /// Column-wise cumulative minimum (pandas `cummin`).
    pub(crate) fn cummin(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cummin())
    }
    /// Column-wise cumulative product (pandas `cumprod`).
    pub(crate) fn cumprod(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cumprod())
    }
    /// Column-wise absolute value (pandas `abs`), dtype-preserving.
    pub(crate) fn abs(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.abs())
    }
    /// Column-wise clip into `[lower, upper]` (pandas `clip`), dtype-preserving.
    #[pyo3(signature = (lower = None, upper = None))]
    pub(crate) fn clip(&self, lower: Option<f64>, upper: Option<f64>) -> PyResult<PyDataFrame> {
        // F19: inverted interval (lower > upper) -> fail-loud (C5), not silent.
        if let (Some(lo), Some(hi)) = (lower, upper) {
            if lo > hi {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "clip lower ({lo}) must be <= upper ({hi})"
                )));
            }
        }
        self.map_cols(|s| s.clip(lower, upper))
    }
    /// Column-wise discrete difference (pandas `diff`), dtype-preserving; the gap
    /// is missing (`volas.NA` for int/bool).
    #[pyo3(signature = (n = 1))]
    pub(crate) fn diff(&self, n: isize) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.diff(n))
    }
    /// Column-wise shift by `n` rows (pandas `shift`), dtype-preserving; the
    /// vacated cells are missing (`volas.NA` for int/bool).
    #[pyo3(signature = (n = 1))]
    pub(crate) fn shift(&self, n: isize) -> PyResult<PyDataFrame> {
        self.map_cols(|s| Ok(s.shift(n)))
    }
    /// Column-wise rank (pandas `rank`); always float.
    #[pyo3(signature = (method = "average", ascending = true, pct = false, na_option = "keep"))]
    pub(crate) fn rank(&self, method: &str, ascending: bool, pct: bool, na_option: &str) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.rank(method, ascending, pct, na_option))
    }

    // --- column-wise reductions (-> a Series indexed by column name; numeric
    // columns only, pandas df.sem() etc.). -------------------------------------

    /// Per-column standard error of the mean (pandas `sem`).
    pub(crate) fn sem(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| stats::sem(&c.to_f64_vec()))
    }
    /// Per-column unbiased skewness (pandas `skew`).
    pub(crate) fn skew(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| stats::skew(&c.to_f64_vec()))
    }
    /// Per-column unbiased excess kurtosis (pandas `kurt`).
    pub(crate) fn kurt(&self) -> PyResult<PySeries> {
        self.reduce_cols(|c| stats::kurt(&c.to_f64_vec()))
    }

    /// Per-column summary statistics over the numeric columns (pandas `describe`):
    /// a frame indexed by `count / mean / std / min / 25% / 50% / 75% / max`.
    pub(crate) fn describe(&self) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let mut names = Vec::new();
        let mut cols = Vec::new();
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            if col.dtype().is_numeric() {
                let s = PySeries {
                    inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(self.inner.index())),
                };
                names.push(name.clone());
                cols.push(s.describe()?.inner.data);
            }
        }
        // No numeric columns (empty / str-only / datetime-only frame) -> a 0x0
        // frame, consistent with corr / cov — not an 8-row describe index over zero
        // columns, which the core rejects as a height mismatch. volas describe is
        // numeric-only (pandas would return object-column stats here instead).
        let index = if cols.is_empty() {
            Index::str(Vec::new())
        } else {
            Index::str(describe_labels())
        };
        DataFrame::new(names, cols, Some(index))
            .map(PyDataFrame::plain)
            .map_err(pyerr)
    }

    /// Pairwise Pearson correlation matrix over the numeric columns (pandas
    /// `corr`): a square frame indexed and labelled by those column names.
    pub(crate) fn corr(&self) -> PyResult<PyDataFrame> {
        self.corr_cov(stats::corr)
    }

    /// Pairwise sample covariance matrix over the numeric columns (pandas `cov`).
    pub(crate) fn cov(&self) -> PyResult<PyDataFrame> {
        self.corr_cov(stats::cov)
    }

    /// pandas `DataFrame.where`: keep each cell where `cond` is True, else `other`
    /// (a typed scalar, resolved per column like the Series surface; default /
    /// `volas.NA` is a dtype-preserving NA fill). `cond` is a same-shape boolean
    /// frame (e.g. from `isna`). The inverse is `mask`.
    #[pyo3(name = "where", signature = (cond, other = None))]
    pub(crate) fn where_(
        &self,
        cond: &PyDataFrame,
        other: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        self.where_mask(cond, other, true)
    }

    /// pandas `DataFrame.mask`: replace each cell with `other` where `cond` is
    /// True, keep it elsewhere — the inverse of `where`.
    #[pyo3(signature = (cond, other = None))]
    pub(crate) fn mask(
        &self,
        cond: &PyDataFrame,
        other: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        self.where_mask(cond, other, false)
    }

    /// Boolean mask of missing cells (every dtype via the column validity) -> a
    /// bool DataFrame (pandas `isna`).
    pub(crate) fn isna(&self) -> PyResult<PyDataFrame> {
        self.mask_na(true)
    }

    /// Boolean mask of present (non-NaN) cells -> a bool DataFrame (pandas `notna`).
    pub(crate) fn notna(&self) -> PyResult<PyDataFrame> {
        self.mask_na(false)
    }

    /// Sort rows by index label (pandas `sort_index`).
    #[pyo3(signature = (ascending = true))]
    pub(crate) fn sort_index(&self, ascending: bool) -> PyDataFrame {
        let perm = self.inner.index().argsort(ascending);
        PyDataFrame::plain(take_frame(&self.inner, &perm))
    }

    /// Move the row index into an `'index'` column and restore a RangeIndex
    /// (pandas `reset_index`); `drop=True` discards the old index.
    #[pyo3(signature = (drop = false))]
    pub(crate) fn reset_index(&self, drop: bool) -> PyResult<PyDataFrame> {
        let h = self.inner.height();
        let (names, columns): (Vec<String>, Vec<Column>) = if drop {
            (self.inner.names().to_vec(), self.inner.columns().to_vec())
        } else {
            // Restore the index's name as the new column label (pandas parity);
            // an unnamed index falls back to "index".
            let label = self
                .inner
                .index()
                .name()
                .unwrap_or("index")
                .to_string();
            // F39: the restored index label must not collide with an existing
            // column — a duplicate column name violates the unique-name contract.
            if self.inner.names().iter().any(|n| n == &label) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "reset_index: column name {label:?} already exists (would duplicate)"
                )));
            }
            let mut names = vec![label];
            names.extend(self.inner.names().iter().cloned());
            let mut cols = vec![self.inner.index().to_column()];
            cols.extend(self.inner.columns().iter().cloned());
            (names, cols)
        };
        Ok(PyDataFrame::plain(
            DataFrame::new(names, columns, Some(Index::range(h))).map_err(pyerr)?,
        ))
    }
}

impl PyDataFrame {
    /// Per-numeric-column reduce via a Series-level helper -> f64 Series keyed
    /// by column name (non-numeric columns are skipped, like `reduce_cols`).
    fn reduce_with(&self, op: impl Fn(&PySeries) -> f64) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let mut names = Vec::new();
        let mut vals = Vec::new();
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            if col.require_numeric().is_ok() {
                let s = PySeries {
                    inner: Series::new(
                        Some(name.clone()),
                        col.clone(),
                        Arc::clone(self.inner.index()),
                    ),
                };
                names.push(name.clone());
                vals.push(op(&s));
            }
        }
        Ok(PySeries {
            inner: Series::new(None, Column::f64(vals), Arc::new(Index::str(names))),
        })
    }

    /// Like [`Self::reduce_with`] for fallible helpers (quantile).
    fn try_reduce_with(&self, op: impl Fn(&PySeries) -> PyResult<f64>) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let mut names = Vec::new();
        let mut vals = Vec::new();
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            if col.require_numeric().is_ok() {
                let s = PySeries {
                    inner: Series::new(
                        Some(name.clone()),
                        col.clone(),
                        Arc::clone(self.inner.index()),
                    ),
                };
                names.push(name.clone());
                vals.push(op(&s)?);
            }
        }
        Ok(PySeries {
            inner: Series::new(None, Column::f64(vals), Arc::new(Index::str(names))),
        })
    }

    /// Per-column truthiness any/all (NA-skipping) -> bool Series by name.
    fn bool_reduce(&self, want_any: bool) -> PySeries {
        let names: Vec<String> = self.inner.names().to_vec();
        let vals: Vec<bool> = self
            .inner
            .columns()
            .iter()
            .map(|c| {
                let truth = to_bool_vec(c);
                let present = (0..c.len()).filter(|&i| c.is_valid(i));
                if want_any {
                    present.into_iter().any(|i| truth[i])
                } else {
                    present.into_iter().all(|i| truth[i])
                }
            })
            .collect();
        PySeries {
            inner: Series::new(None, Column::bool(vals), Arc::new(Index::str(names))),
        }
    }

    /// Per-column index label of the extreme -> a Series of labels keyed by
    /// column name (the label dtype follows the index kind).
    fn idx_extreme(&self, py: Python<'_>, want_max: bool) -> PyResult<Py<PyAny>> {
        let names: Vec<String> = self.inner.names().to_vec();
        let mut positions = Vec::with_capacity(names.len());
        for col in self.inner.columns() {
            positions.push(argext(col, want_max)?);
        }
        let index = self.inner.index();
        let labels = index.take(&positions).to_column();
        let s = PySeries {
            inner: Series::new(None, labels, Arc::new(Index::str(names))),
        };
        Ok(Py::new(py, s)?.into_any())
    }

    /// The `n` extreme rows by `column` (ascending for nsmallest).
    fn extreme_rows(&self, n: usize, column: &str, ascending: bool) -> PyResult<PyDataFrame> {
        let col = self.inner.column(column).map_err(pyerr)?;
        col.require_numeric().map_err(pyerr)?;
        let v = col.to_f64_vec();
        let mut order: Vec<usize> = (0..v.len()).filter(|&i| col.is_valid(i) && !v[i].is_nan()).collect();
        order.sort_by(|&a, &b| {
            let o = v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal);
            if ascending { o } else { o.reverse() }
        });
        order.truncate(n);
        Ok(PyDataFrame::plain(take_frame(&self.inner, &order)))
    }

    /// Row-level duplicate mask over all columns, honoring `keep` ('first'|'last').
    fn row_duplicated(&self, keep: &str) -> PyResult<Vec<bool>> {
        let h = self.inner.height();
        let key_of = |i: usize| -> Vec<Option<String>> {
            self.inner
                .columns()
                .iter()
                .map(|c| crate::series::cell_key(c, i))
                .collect()
        };
        let mut seen = std::collections::HashSet::with_capacity(h);
        match keep {
            "first" => Ok((0..h).map(|i| !seen.insert(key_of(i))).collect()),
            "last" => {
                let mut out = vec![false; h];
                for i in (0..h).rev() {
                    out[i] = !seen.insert(key_of(i));
                }
                Ok(out)
            }
            other => Err(PyValueError::new_err(format!(
                "keep must be 'first' or 'last', got {other:?}"
            ))),
        }
    }
}

/// `df.rolling(window)` — per-numeric-column rolling aggregation -> DataFrame.
#[pyclass(name = "RollingFrame")]
pub struct PyRollingFrame {
    frame: DataFrame,
    window: usize,
    min_periods: usize,
    center: bool,
}

/// `df.expanding()` — per-numeric-column expanding aggregation -> DataFrame.
#[pyclass(name = "ExpandingFrame")]
pub struct PyExpandingFrame {
    frame: DataFrame,
    min_periods: usize,
}

/// `df.ewm(...)` — per-numeric-column EW aggregation -> DataFrame.
#[pyclass(name = "EwmFrame")]
pub struct PyEwmFrame {
    frame: DataFrame,
    alpha: f64,
    adjust: bool,
    ignore_na: bool,
    min_periods: usize,
}

/// Apply a Series-level window op over every column of `frame` (a non-numeric
/// column errors, like pandas with `numeric_only=False`). A frame-level window
/// is a BULK read — it would aggregate a stale cached-directive column as if
/// it were data — so it carries the same fulfill guard as to_numpy / iloc
/// (same-guard symmetry, E8; self-audit SA2-3).
fn frame_window(
    frame: &DataFrame,
    op: impl Fn(&PySeries) -> PyResult<PySeries>,
) -> PyResult<PyDataFrame> {
    crate::ensure_fresh(frame)?;
    let wrapper = PyDataFrame::plain(frame.clone());
    wrapper.map_cols(|s| {
        s.inner.data.require_numeric().map_err(pyerr)?;
        op(s)
    })
}

impl PyRollingFrame {
    fn spec(&self) -> crate::window::WinSpec {
        crate::window::WinSpec {
            window: self.window,
            min_periods: self.min_periods,
            center: self.center,
        }
    }
    fn agg(
        &self,
        op: impl Fn(&crate::window::WinSpec, &volas_core::Series) -> PyResult<PySeries>,
    ) -> PyResult<PyDataFrame> {
        let spec = self.spec();
        frame_window(&self.frame, |s| op(&spec, &s.inner))
    }
}

impl PyExpandingFrame {
    fn agg(
        &self,
        op: impl Fn(&crate::window::WinSpec, &volas_core::Series) -> PyResult<PySeries>,
    ) -> PyResult<PyDataFrame> {
        let spec = crate::window::WinSpec {
            window: usize::MAX,
            min_periods: self.min_periods,
            center: false,
        };
        frame_window(&self.frame, |s| op(&spec, &s.inner))
    }
}

#[pymethods]
impl PyRollingFrame {
    /// Per-column rolling count (int64) -> DataFrame.
    fn count(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.count(s)))
    }
    /// Per-column rolling distinct-count (int64).
    fn nunique(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.nunique(s)))
    }
    /// Per-column rolling sum.
    fn sum(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.sum(s)))
    }
    /// Per-column rolling mean.
    fn mean(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.mean(s)))
    }
    /// Per-column rolling median.
    fn median(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.median(s)))
    }
    /// Per-column rolling minimum.
    fn min(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.min(s)))
    }
    /// Per-column rolling maximum.
    fn max(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.max(s)))
    }
    /// Per-column rolling sample variance.
    #[pyo3(signature = (ddof = 1))]
    fn var(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.var(s, ddof)))
    }
    /// Per-column rolling sample standard deviation.
    #[pyo3(signature = (ddof = 1))]
    fn std(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.std(s, ddof)))
    }
    /// Per-column standard error of the mean.
    #[pyo3(signature = (ddof = 1))]
    fn sem(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.sem(s, ddof)))
    }
    /// Per-column rolling skewness.
    fn skew(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.skew(s)))
    }
    /// Per-column rolling excess kurtosis.
    fn kurt(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.kurt(s)))
    }
    /// Per-column rolling quantile.
    #[pyo3(signature = (q, interpolation = "linear"))]
    fn quantile(&self, q: f64, interpolation: &str) -> PyResult<PyDataFrame> {
        self.agg(|w, s| w.quantile(s, q, interpolation))
    }
    /// Per-column rolling rank of the current value within its window.
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PyDataFrame> {
        self.agg(|w, s| w.rank(s, method, ascending, pct))
    }
    /// Per-column first present value per window, dtype-preserving.
    fn first(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.edge(s, false)))
    }
    /// Per-column last present value per window, dtype-preserving.
    fn last(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.edge(s, true)))
    }
}

#[pymethods]
impl PyExpandingFrame {
    /// Per-column expanding count (int64) -> DataFrame.
    fn count(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.count(s)))
    }
    /// Per-column expanding distinct-count (int64).
    fn nunique(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.nunique(s)))
    }
    /// Per-column expanding sum.
    fn sum(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.sum(s)))
    }
    /// Per-column expanding mean.
    fn mean(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.mean(s)))
    }
    /// Per-column expanding median.
    fn median(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.median(s)))
    }
    /// Per-column expanding minimum.
    fn min(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.min(s)))
    }
    /// Per-column expanding maximum.
    fn max(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.max(s)))
    }
    /// Per-column expanding sample variance.
    #[pyo3(signature = (ddof = 1))]
    fn var(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.var(s, ddof)))
    }
    /// Per-column expanding sample standard deviation.
    #[pyo3(signature = (ddof = 1))]
    fn std(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.std(s, ddof)))
    }
    /// Per-column expanding standard error of the mean.
    #[pyo3(signature = (ddof = 1))]
    fn sem(&self, ddof: i64) -> PyResult<PyDataFrame> {
        let ddof = crate::window::validate_ddof(ddof)?;
        self.agg(|w, s| Ok(w.sem(s, ddof)))
    }
    /// Per-column expanding skewness.
    fn skew(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.skew(s)))
    }
    /// Per-column expanding excess kurtosis.
    fn kurt(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.kurt(s)))
    }
    /// Per-column expanding quantile.
    #[pyo3(signature = (q, interpolation = "linear"))]
    fn quantile(&self, q: f64, interpolation: &str) -> PyResult<PyDataFrame> {
        self.agg(|w, s| w.quantile(s, q, interpolation))
    }
    /// Per-column expanding rank.
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PyDataFrame> {
        self.agg(|w, s| w.rank(s, method, ascending, pct))
    }
    /// Per-column first present value so far, dtype-preserving.
    fn first(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.edge(s, false)))
    }
    /// Per-column last present value so far, dtype-preserving.
    fn last(&self) -> PyResult<PyDataFrame> {
        self.agg(|w, s| Ok(w.edge(s, true)))
    }
}

#[pymethods]
impl PyEwmFrame {
    /// Per-column exponentially-weighted mean -> DataFrame.
    fn mean(&self) -> PyResult<PyDataFrame> {
        frame_window(&self.frame, |s| {
            Ok(crate::window::PyEwm {
                series: s.inner.clone(),
                alpha: self.alpha,
                adjust: self.adjust,
                ignore_na: self.ignore_na,
                min_periods: self.min_periods,
            }
            .mean())
        })
    }
    /// Per-column exponentially-weighted sum (`adjust=True` only).
    fn sum(&self) -> PyResult<PyDataFrame> {
        frame_window(&self.frame, |s| {
            crate::window::PyEwm {
                series: s.inner.clone(),
                alpha: self.alpha,
                adjust: self.adjust,
                ignore_na: self.ignore_na,
                min_periods: self.min_periods,
            }
            .sum()
        })
    }
    /// Per-column exponentially-weighted variance.
    #[pyo3(signature = (bias = false))]
    fn var(&self, bias: bool) -> PyResult<PyDataFrame> {
        frame_window(&self.frame, |s| {
            Ok(crate::window::PyEwm {
                series: s.inner.clone(),
                alpha: self.alpha,
                adjust: self.adjust,
                ignore_na: self.ignore_na,
                min_periods: self.min_periods,
            }
            .var(bias))
        })
    }
    /// Per-column exponentially-weighted standard deviation.
    #[pyo3(signature = (bias = false))]
    fn std(&self, bias: bool) -> PyResult<PyDataFrame> {
        frame_window(&self.frame, |s| {
            Ok(crate::window::PyEwm {
                series: s.inner.clone(),
                alpha: self.alpha,
                adjust: self.adjust,
                ignore_na: self.ignore_na,
                min_periods: self.min_periods,
            }
            .std(bias))
        })
    }
}
