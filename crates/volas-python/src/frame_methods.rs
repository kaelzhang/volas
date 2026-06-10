//! `DataFrame` methods, part 1: construction, accessors, reductions,
//! element-wise transforms, and missing-value handling.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use volas_core::{
    stats, CmpOp, Column, DataFrame, Index,
    IndexKind, Series, Tz,
};

use crate::timeframe::{build_agg_spec, resolve_time_frame};
#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PyDataFrame {
    // Constructor — the user-facing argument list & usage live in the class
    // docstring (pyo3 does not surface a `#[new]` doc comment to Python).
    #[new]
    #[pyo3(signature = (data, columns = None, time_frame = None, cumulators = None, dtype = None))]
    pub(crate) fn new(
        data: &Bound<'_, PyAny>,
        columns: Option<Vec<String>>,
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
            if !matches!(df.index().kind(), IndexKind::Datetime(..)) {
                return Err(PyValueError::new_err(
                    "time_frame requires a DatetimeIndex \
                     (build one with to_datetime(df[col]) then df.set_index(col))",
                ));
            }
            let spec = build_agg_spec(cumulators)?;
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

    /// The DatetimeIndex timezone name (`"+08:00"` / `"America/New_York"`), or
    /// `None` for a tz-naive (UTC-default) or non-datetime index — mirroring
    /// pandas `df.index.tz`.
    #[getter]
    pub(crate) fn tz(&self) -> Option<String> {
        match self.inner.index().tz() {
            Tz::Utc => None,
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

    /// First `n` rows (pandas `head`).
    #[pyo3(signature = (n = 5))]
    pub(crate) fn head(&self, n: usize) -> PyDataFrame {
        PyDataFrame::plain(self.inner.slice(0, n.min(self.inner.height())))
    }

    /// Last `n` rows (pandas `tail`).
    #[pyo3(signature = (n = 5))]
    pub(crate) fn tail(&self, n: usize) -> PyDataFrame {
        let h = self.inner.height();
        PyDataFrame::plain(self.inner.slice(h.saturating_sub(n), h))
    }

    /// Per-column count of non-missing values (pandas `count`) -> a Series indexed
    /// by column name (`int64`), reading each column's validity.
    pub(crate) fn count(&self) -> PySeries {
        let names: Vec<String> = self.inner.names().to_vec();
        let counts: Vec<i64> = self.inner.columns().iter().map(|c| c.count() as i64).collect();
        PySeries {
            inner: Series::new(None, Column::i64(counts), Arc::new(Index::str(names))),
        }
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
    /// delegating to the per-column validity-aware `Column::fillna` so int / bool
    /// holes are filled dtype-preserving (like the Series version), not just float
    /// NaN. A `str` / `datetime` column with a missing cell raises a `TypeError`
    /// (a numeric fill cannot apply; volas has no `object` dtype) — a dense
    /// (no-hole) str / datetime column is untouched. For directional fill use
    /// `ffill` / `bfill` (pandas 3.0 removed `fillna(method=)`).
    pub(crate) fn fillna(&self, value: f64) -> PyResult<PyDataFrame> {
        let cols: Vec<Column> = self
            .inner
            .columns()
            .iter()
            .map(|c| c.fillna(value))
            .collect::<volas_core::Result<_>>()
            .map_err(pyerr)?;
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
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    pub(crate) fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.rank(method, ascending, pct))
    }

    // --- column-wise reductions (-> a Series indexed by column name; numeric
    // columns only, pandas df.sem() etc.). -------------------------------------

    /// Per-column standard error of the mean (pandas `sem`).
    pub(crate) fn sem(&self) -> PySeries {
        self.reduce_cols(|c| stats::sem(&c.to_f64_vec()))
    }
    /// Per-column unbiased skewness (pandas `skew`).
    pub(crate) fn skew(&self) -> PySeries {
        self.reduce_cols(|c| stats::skew(&c.to_f64_vec()))
    }
    /// Per-column unbiased excess kurtosis (pandas `kurt`).
    pub(crate) fn kurt(&self) -> PySeries {
        self.reduce_cols(|c| stats::kurt(&c.to_f64_vec()))
    }

    /// Per-column summary statistics over the numeric columns (pandas `describe`):
    /// a frame indexed by `count / mean / std / min / 25% / 50% / 75% / max`.
    pub(crate) fn describe(&self) -> PyResult<PyDataFrame> {
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
        DataFrame::new(names, cols, Some(Index::str(describe_labels())))
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
    /// (default NaN). `cond` is a same-shape boolean frame (e.g. from `isna`);
    /// columns are taken as float. The inverse is `mask`.
    #[pyo3(name = "where", signature = (cond, other = None))]
    pub(crate) fn where_(&self, cond: &PyDataFrame, other: Option<f64>) -> PyResult<PyDataFrame> {
        self.where_mask(cond, other, true)
    }

    /// pandas `DataFrame.mask`: replace each cell with `other` where `cond` is
    /// True, keep it elsewhere — the inverse of `where`.
    #[pyo3(signature = (cond, other = None))]
    pub(crate) fn mask(&self, cond: &PyDataFrame, other: Option<f64>) -> PyResult<PyDataFrame> {
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
