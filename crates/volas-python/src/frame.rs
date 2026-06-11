//! The `volas.DataFrame` pyclass: its fields, the non-`#[pymethods]` helpers,
//! and `PyRow`. The method surface lives in frame_methods / frame_methods2.

use std::collections::HashSet;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use volas_core::{
    binary_supertype, CmpOp, Column, DataFrame, Index,
    IndexKind, Series,
};
use volas_directive::{execute, parse};
use volas_time::{aggregate_period, AggSpec, TimeFrame};

use crate::format::render_row;
#[allow(unused_imports)]
use crate::*;

/// A single DataFrame row (the result of `df.iloc[i]` / `df.loc[label]`): a
/// faithful 1-row frame carrying its index label and every column's *typed*
/// value (no lossy f64 coercion, no flag pair to remember the index kind).
#[pyclass(name = "Row")]
pub struct PyRow {
    pub(crate) inner: DataFrame,
}

#[pymethods]
impl PyRow {
    /// The row's index label.
    #[getter]
    pub(crate) fn name(&self, py: Python<'_>) -> Py<PyAny> {
        label_to_py(py, self.inner.index(), 0)
    }

    /// A single value by column name (``row[col]``).
    ///
    /// Returns:
    ///     the typed scalar at that column.
    pub(crate) fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let col = self.inner.column(key).map_err(pyerr)?;
        Ok(np_scalar_to_py(py, col, 0))
    }

    /// The row's values as a ``(1, n_columns)`` float64 NumPy array. Only valid for
    /// an all-numeric row — a str / datetime cell cannot be represented as float64
    /// without a silent NaN, so it errors (contract R2 / C4); read those via
    /// ``to_dict()`` or ``row[col]`` instead.
    ///
    /// Returns:
    ///     numpy.ndarray
    pub(crate) fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        for c in self.inner.columns() {
            c.require_numeric().map_err(pyerr)?;
        }
        // F14: a Row is a single 1-D record -> shape (n,), like pandas
        // df.iloc[0].to_numpy() (was a 2-D (1, n) frame export).
        let (data, _h, _w) = self.inner.to_row_major_f64();
        Ok(data.into_pyarray(py))
    }

    /// The row as a typed `{column: value}` dict (pandas `Series.to_dict`).
    pub(crate) fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            d.set_item(name, scalar_to_py(py, col, 0))?;
        }
        Ok(d)
    }

    /// Vertical repr — `column   value` lines plus a `Name: <row label>` footer.
    /// No `dtype:` is printed: a Row is a typed record, not a Series, and has no
    /// single dtype (pandas prints `dtype: object` only because its row IS an
    /// object Series). `str` and `repr` are identical.
    pub(crate) fn __repr__(&self) -> String {
        render_row(&self.inner, true)
    }

    pub(crate) fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Render the row as text without the `Name` footer (pandas
    /// `Series.to_string`).
    pub(crate) fn to_string(&self) -> String {
        render_row(&self.inner, false)
    }
}

// --- DataFrame -------------------------------------------------------------

/// ``volas.DataFrame`` — an ordered, named, time-indexed OHLCV table with
/// indicator-directive indexing and pandas-compatible positional / label access.
///
/// Construct from a dict of columns, or read a CSV::
///
///     df = volas.DataFrame({'close': [10.0, 11.0], 'volume': [100, 120]})
///     df = volas.read_csv('ohlcv.csv', index_col='time')
///
/// The headline feature is string indexing: a plain column name returns that
/// column, and an *indicator directive* is computed on demand, cached, and
/// incrementally refreshed thereafter::
///
///     df['close']            # a column, as a Series
///     df['ma:5']             # SMA(5) of close (directive) — computed & cached
///     df['macd.signal']      # MACD signal line
///     df['close > open']     # a boolean directive -> bool Series
///     df[['open', 'close']]  # a sub-frame
///     df[df['close'] > 100]  # boolean-mask row filter
///
/// Positional / label access mirrors pandas via ``.iloc`` / ``.loc`` (2-D get +
/// set) and the scalar ``.iat`` / ``.at``; common transforms (``head``,
/// ``tail``, ``dropna``, ``sort_index``, ``reset_index``, ``set_index``,
/// ``rename``, ``astype``, ``to_numpy``, ``to_pandas``, ``to_csv``) follow the
/// pandas spelling. ``cumulate`` resamples to a coarser timeframe; ``append``
/// grows the frame in place for live streaming.
///
/// Args:
///     data (dict[str, Sequence] | DataFrame): a dict of column name -> equal-length
///         values, or another volas DataFrame to copy (its index, aliases and tf-state are
///         carried — like ``df.copy()``). A pandas DataFrame is not accepted; use
///         ``from_pandas``. Build a DatetimeIndex from a column with ``read_csv`` or
///         ``to_datetime`` + ``set_index`` (+ ``tz_localize`` / ``tz_convert``).
///     columns (list[str], optional): select and order the columns to keep (like
///         ``df[[...]]``); a name not present raises ``KeyError``. An empty list or a
///         duplicate name is rejected, and an absent column is never NaN-filled.
///     time_frame (str | TimeFrame, optional): make this a tf-aware (cumulating) frame at
///         this bar interval; the given rows are taken as already-final bars and later
///         ``append``s fold finer bars into them. Requires a DatetimeIndex.
///     cumulators (dict[str, str], optional): per-column aggregator overrides for folding
///         (e.g. ``{'amount': 'sum'}``); only meaningful together with ``time_frame``.
/// Live cumulation state carried by a tf-aware DataFrame (set via the
/// `time_frame` constructor arg or `cumulate`): the target frame, the per-column
/// aggregators, and the raw fine bars of the still-open (forming) period —
/// `df.iloc[-1]` is that period's running bar, which `append` keeps updating.
#[derive(Clone)]
pub(crate) struct TfState {
    pub(crate) time_frame: TimeFrame,
    pub(crate) cumulators: AggSpec,
    /// Raw fine bars of the current open period (`None` until the first
    /// folded append), kept so a re-sent forming bar updates (deduped) rather
    /// than double-counts.
    pub(crate) open: Option<DataFrame>,
}

#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    pub(crate) inner: DataFrame,
    /// Cumulation state when this is a tf-aware frame; `None` for a plain frame.
    pub(crate) tf: Option<TfState>,
}

/// Read cell `i` of a directive-result column as f64 (`Bool` -> 0/1, `I64` -> as f64),
/// for the finite-memory-vs-recursive refresh probe. NaN for other dtypes.
pub(crate) fn col_value(col: &Column, i: usize) -> f64 {
    match col {
        Column::F64(v) => v[i],
        Column::Bool(v, _) => {
            if v[i] {
                1.0
            } else {
                0.0
            }
        }
        Column::I64(v, _) => v[i] as f64,
        _ => f64::NAN,
    }
}

impl PyDataFrame {
    /// Wrap a core frame as a plain (non-cumulating) DataFrame — the default for
    /// every derived frame (slices, projections, head/tail, ...).
    pub(crate) fn plain(inner: DataFrame) -> Self {
        PyDataFrame { inner, tf: None }
    }

    /// Element-wise comparison backing `__eq__` / `__ne__`: against another
    /// DataFrame (identical column names + shared index) or a scalar (broadcast),
    /// producing a bool DataFrame. Compared by position; never auto-aligned.
    pub(crate) fn compare(&self, other: &Bound<'_, PyAny>, op: CmpOp) -> PyResult<PyDataFrame> {
        let cols: Vec<Column> = if let Ok(o) = other.extract::<PyRef<PyDataFrame>>() {
            if self.inner.names() != o.inner.names() {
                return Err(PyValueError::new_err(
                    "cannot compare DataFrames with different columns",
                ));
            }
            require_aligned(self.inner.index(), o.inner.index())?;
            self.inner
                .columns()
                .iter()
                .zip(o.inner.columns())
                .map(|(a, b)| a.compare(b, op))
                .collect::<Result<_, _>>()
                .map_err(pyerr)?
        } else {
            // a scalar is broadcast and typed per column; a column whose dtype the
            // scalar cannot match is a TypeError (no silent all-False mask).
            self.inner
                .columns()
                .iter()
                .map(|c| c.compare(&cmp_scalar_col(other, c.dtype(), c.len())?, op).map_err(pyerr))
                .collect::<PyResult<_>>()?
        };
        self.with_columns(cols)
    }

    /// Rebuild a plain frame from `cols`, reusing this frame's names and index (the
    /// columns must be height-aligned). Backs `compare` / `fillna` / `mask_na`.
    pub(crate) fn with_columns(&self, cols: Vec<Column>) -> PyResult<PyDataFrame> {
        DataFrame::new(
            self.inner.names().to_vec(),
            cols,
            Some((**self.inner.index()).clone()),
        )
        .map(PyDataFrame::plain)
        .map_err(pyerr)
    }

    /// One column as a `PySeries` (carrying its name + the frame index), for
    /// column-wise delegation to Series methods.
    pub(crate) fn col_as_series(&self, name: &str, col: &Column) -> PySeries {
        PySeries {
            inner: Series::new(Some(name.to_string()), col.clone(), Arc::clone(self.inner.index())),
        }
    }

    /// Apply a Series transform to every column -> a new frame (pandas column-wise
    /// `df.cumsum()` etc.). Each column's own dtype rule applies; a column the op
    /// rejects (e.g. a string column under a numeric transform) propagates its error.
    pub(crate) fn map_cols(&self, op: impl Fn(&PySeries) -> PyResult<PySeries>) -> PyResult<PyDataFrame> {
        let cols = self
            .inner
            .names()
            .iter()
            .zip(self.inner.columns())
            .map(|(name, col)| Ok(op(&self.col_as_series(name, col))?.inner.data))
            .collect::<PyResult<Vec<_>>>()?;
        self.with_columns(cols)
    }

    /// Reduce each numeric column to a scalar -> a Series indexed by column name
    /// (pandas column-wise `df.sem()` etc.; non-numeric columns are skipped).
    pub(crate) fn reduce_cols(&self, op: impl Fn(&Column) -> f64) -> PySeries {
        let mut names = Vec::new();
        let mut vals = Vec::new();
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            if col.dtype().is_numeric() {
                names.push(name.clone());
                vals.push(op(col));
            }
        }
        PySeries {
            inner: Series::new(None, Column::f64(vals), Arc::new(Index::str(names))),
        }
    }

    /// Directional fill (`forward` = ffill, else bfill) over every column,
    /// delegating to the per-column validity-aware `Column::fill_dir` so int /
    /// bool / str holes carry directionally too (like the Series version), not
    /// only float NaN. Backs `ffill` / `bfill`.
    pub(crate) fn fill_dir(&self, forward: bool) -> PyResult<PyDataFrame> {
        let cols: Vec<Column> = self
            .inner
            .columns()
            .iter()
            .map(|c| c.fill_dir(forward))
            .collect();
        self.with_columns(cols)
    }

    /// Pairwise matrix (corr / cov) over the numeric columns; result column `j`
    /// is `[op(col_i, col_j) for i]`, indexed and labelled by the column names.
    /// Backs `corr` / `cov`.
    pub(crate) fn corr_cov(&self, op: fn(&[f64], &[f64]) -> f64) -> PyResult<PyDataFrame> {
        let numeric: Vec<(String, Vec<f64>)> = self
            .inner
            .names()
            .iter()
            .zip(self.inner.columns())
            .filter(|(_, c)| c.dtype().is_numeric())
            .map(|(n, c)| (n.clone(), c.to_f64_vec()))
            .collect();
        let names: Vec<String> = numeric.iter().map(|(n, _)| n.clone()).collect();
        let cols: Vec<Column> = numeric
            .iter()
            .map(|(_, cj)| Column::f64(numeric.iter().map(|(_, ci)| op(ci, cj)).collect()))
            .collect();
        DataFrame::new(names.clone(), cols, Some(Index::str(names)))
            .map(PyDataFrame::plain)
            .map_err(pyerr)
    }

    /// `df.where` / `df.mask` shared core: per-cell keep/replace against a
    /// same-shape boolean frame, dtype-preservingly per column (the default
    /// `other = None` keeps each column's dtype and fills NA; an explicit numeric
    /// `other` promotes via the supertype).
    pub(crate) fn where_mask(
        &self,
        cond: &PyDataFrame,
        other: Option<&Bound<'_, PyAny>>,
        is_where: bool,
    ) -> PyResult<PyDataFrame> {
        if cond.inner.width() != self.inner.width() || cond.inner.height() != self.inner.height() {
            return Err(PyValueError::new_err(
                "where/mask: `cond` must have the same shape as the frame",
            ));
        }
        // F36: a DataFrame condition pairs with columns by NAME, never by
        // position — a same-set/different-order cond is reordered to match, and
        // a different name set is an error (silently mis-applying a mask to the
        // wrong column is the worst failure mode for signal filtering).
        let cond_inner = if cond.inner.names() == self.inner.names() {
            cond.inner.clone()
        } else {
            let mut sorted_a: Vec<&String> = self.inner.names().iter().collect();
            let mut sorted_b: Vec<&String> = cond.inner.names().iter().collect();
            sorted_a.sort();
            sorted_b.sort();
            if sorted_a != sorted_b {
                return Err(PyValueError::new_err(
                    "where/mask: `cond` columns must match the frame's columns by name",
                ));
            }
            cond.inner.select(self.inner.names()).map_err(pyerr)?
        };
        // and the row labels must agree — a cond built on a different index would
        // silently filter the wrong rows.
        if *cond_inner.index() != *self.inner.index() {
            return Err(PyValueError::new_err(
                "where/mask: `cond` index must equal the frame's index",
            ));
        }
        let cond = &PyDataFrame::plain(cond_inner);
        // the condition must be boolean — a numeric mask is rejected (pandas-shaped)
        if let Some(cc) = cond
            .inner
            .columns()
            .iter()
            .find(|c| !matches!(c, Column::Bool(..)))
        {
            return Err(PyTypeError::new_err(format!(
                "where/mask: `cond` must be a boolean frame, got a {} column",
                cc.dtype()
            )));
        }
        // a missing `other` (default) or an explicit NA / NaN fill is a dtype-
        // preserving all-NA fill — the typed scalar (str / Timestamp / number /
        // bool) path runs per column otherwise (C2/C4), mirroring the Series surface.
        let na_like = other.map(is_na_like_py).unwrap_or(true);
        let cols = self
            .inner
            .columns()
            .iter()
            .zip(cond.inner.columns())
            .map(|(keep_col, cond_col)| -> PyResult<Column> {
                let mut c = bool_mask_vec(cond_col)?;
                if !is_where {
                    c.iter_mut().for_each(|b| *b = !*b);
                }
                // Lazy: a column whose cells are all kept is returned unchanged, so an
                // all-keep column never type-checks a fill it does not use (parity with
                // Series.select_with).
                if c.iter().all(|&b| b) {
                    return Ok(keep_col.clone());
                }
                let kd = keep_col.dtype();
                let (other_col, target) = if na_like {
                    (Column::na_of(kd, keep_col.len()), kd)
                } else {
                    let (oc, odt) = scalar_fill_col(other.unwrap(), kd, keep_col.len())?;
                    // a float column absorbs any fill; a same-dtype fill keeps that
                    // dtype; a mixed numeric fill promotes by the supertype.
                    let target = if kd.is_float() {
                        kd
                    } else if kd == odt {
                        kd
                    } else {
                        binary_supertype(kd, odt)
                    };
                    (oc, target)
                };
                keep_col.select(&c, &other_col, target).map_err(pyerr)
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.with_columns(cols)
    }

    /// Rebuild `inner` from new columns, preserving names and index (drops the
    /// directive cache, which a write would stale anyway). Backs mask assignment.
    pub(crate) fn rebuild_with(&mut self, cols: Vec<Column>) -> PyResult<()> {
        self.inner = DataFrame::new(
            self.inner.names().to_vec(),
            cols,
            Some((**self.inner.index()).clone()),
        )
        .map_err(pyerr)?;
        Ok(())
    }

    /// `df[row_mask] = v`: set every column's True rows to the scalar, keeping each
    /// column's dtype (pandas' whole-row boolean assignment), via the shared
    /// `scatter_scalar` primitive. Atomic — if any column would take the value
    /// lossily, the per-column map errors and nothing is written.
    pub(crate) fn assign_row_mask(&mut self, mask: &[bool], value: &Bound<'_, PyAny>) -> PyResult<()> {
        if mask.len() != self.inner.height() {
            return Err(PyValueError::new_err(format!(
                "boolean mask length {} != frame height {}",
                mask.len(),
                self.inner.height()
            )));
        }
        let positions: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| m.then_some(i))
            .collect();
        let cols = self
            .inner
            .columns()
            .iter()
            .map(|c| scatter_scalar(c, &positions, value))
            .collect::<PyResult<Vec<_>>>()?;
        self.rebuild_with(cols)
    }

    /// `df[bool_frame] = v`: per-cell assignment where the mask is True, keeping
    /// each column's dtype. Atomic, like `assign_row_mask`. The condition frame
    /// must be boolean — the same contract as `DataFrame.where` (a numeric / string
    /// mask is rejected up front, not coerced through `x != 0.0`).
    pub(crate) fn assign_cell_mask(&mut self, cond: &PyDataFrame, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if cond.inner.width() != self.inner.width() || cond.inner.height() != self.inner.height() {
            return Err(PyValueError::new_err(
                "df[mask] = v: `mask` must have the same shape as the frame",
            ));
        }
        if let Some(cc) = cond
            .inner
            .columns()
            .iter()
            .find(|c| !matches!(c, Column::Bool(..)))
        {
            return Err(PyTypeError::new_err(format!(
                "df[mask] = v: `mask` must be a boolean frame, got a {} column",
                cc.dtype()
            )));
        }
        let cols = self
            .inner
            .columns()
            .iter()
            .zip(cond.inner.columns())
            .map(|(col, cond_col)| {
                let positions: Vec<usize> = bool_mask_vec(cond_col)?
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &m)| m.then_some(i))
                    .collect();
                scatter_scalar(col, &positions, value)
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.rebuild_with(cols)
    }

    /// Per-cell missing mask -> a bool frame; backs `isna` (want_na=true) /
    /// `notna`. Reads the column validity (every dtype), so an int/bool/str NA
    /// and a datetime NaT are detected, not just a float NaN.
    pub(crate) fn mask_na(&self, want_na: bool) -> PyResult<PyDataFrame> {
        let cols = self
            .inner
            .columns()
            .iter()
            .map(|c| Column::bool((0..c.len()).map(|i| (!c.is_valid(i)) == want_na).collect()))
            .collect();
        self.with_columns(cols)
    }

    /// Fold incoming fine bars into a tf-aware frame: each bar either extends the
    /// open period's forming bar (update `inner`'s last row in place + mark its
    /// computed tail stale) or rolls over into a new period (append a fresh
    /// forming row). Assumes `self.tf` is `Some`. A re-sent forming bar (same
    /// timestamp) updates the period rather than double-counting it.
    pub(crate) fn fold_append(&mut self, fine: &DataFrame) -> PyResult<()> {
        let last_dt = |df: &DataFrame| -> i64 {
            match df.index().kind() {
                IndexKind::Datetime(v, _) => v[v.len() - 1],
                _ => unreachable!("checked by caller"),
            }
        };
        let PyDataFrame { inner, tf } = self;
        let tfs = tf.as_mut().expect("fold_append on a plain frame");
        let frame = tfs.time_frame;
        let (fine_ts, tz) = match fine.index().kind() {
            IndexKind::Datetime(v, tz) => (v.clone(), *tz),
            _ => {
                return Err(PyValueError::new_err(
                    "append to a time_frame DataFrame requires a DatetimeIndex",
                ))
            }
        };
        // R4-P1-01 / R4-P1-02: a live fold must see present, non-decreasing
        // timestamps. Validate every bar BEFORE folding any (atomic — a bad bar
        // mutates nothing): a NaT bar has no period (symmetric with the cumulate()
        // entry's D2 rejection), and a bar earlier than the latest one already
        // folded would roll over into a non-monotonic index / fold later bars into
        // the wrong period. Late or disordered feed data must be handled explicitly
        // by the caller, never silently corrupt the OHLCV.
        let mut prev_ts: Option<i64> = match tfs.open.as_ref() {
            Some(o) => Some(last_dt(o)),
            None => match inner.index().kind() {
                IndexKind::Datetime(v, _) if !v.is_empty() => Some(v[v.len() - 1]),
                _ => None,
            },
        };
        for &ts in &fine_ts {
            if ts == i64::MIN {
                return Err(PyValueError::new_err(
                    "cannot append a NaT-timestamped bar to a time_frame DataFrame; a \
                     missing instant has no period (drop it or supply a real timestamp)",
                ));
            }
            if let Some(p) = prev_ts {
                if ts < p {
                    return Err(PyValueError::new_err(
                        "cannot append an out-of-order bar to a time_frame DataFrame \
                         (its timestamp precedes the forming period's latest bar); handle \
                         late / re-ordered feed data before folding so the OHLCV stays \
                         monotonic",
                    ));
                }
            }
            prev_ts = Some(ts);
        }
        for i in 0..fine.height() {
            let bar_ts = fine_ts[i];
            let key = frame.unify_tz(bar_ts, tz);
            let same_period = tfs
                .open
                .as_ref()
                .is_some_and(|open| frame.unify_tz(last_dt(open), tz) == key);
            let bar = fine.slice(i, i + 1);
            if same_period {
                let open = tfs.open.as_mut().unwrap();
                // A re-sent forming bar (same ts) replaces the last open bar.
                if last_dt(open) == bar_ts {
                    *open = open.slice(0, open.height() - 1);
                }
                open.append(&bar).map_err(pyerr)?;
                let agg = aggregate_period(open, &tfs.cumulators).map_err(pyerr)?;
                let last = inner.height() - 1;
                // `assign_positions` invalidates each written column's dependent
                // directive columns, so the forming row's indicators recompute
                // correctly on the next read — no explicit invalidate needed.
                for (name, col) in agg.names().iter().zip(agg.columns()) {
                    if let Some(j) = inner.column_pos(name) {
                        inner.assign_positions(j, &[last], col).map_err(pyerr)?;
                    }
                }
            } else {
                // Roll over: the previous forming bar (if any) is already final in
                // `inner`; start a new open period and append its forming row.
                let agg = aggregate_period(&bar, &tfs.cumulators).map_err(pyerr)?;
                tfs.open = Some(bar);
                inner.append(&agg).map_err(pyerr)?;
            }
        }
        Ok(())
    }

    pub(crate) fn wrap_series(&self, name: String, col: Column) -> PySeries {
        PySeries {
            inner: Series::new(Some(name), col, Arc::clone(self.inner.index())),
        }
    }

    /// Recompute the stale tail of cached directive columns in place — all of
    /// them if `only` is `None`, else just the named one. O(lookback + new rows)
    /// per column. Done against the real (non-computed) columns so a bare-name
    /// directive recomputes and no cached buffer is pinned.
    pub(crate) fn refresh_computed(&mut self, only: Option<&str>) -> PyResult<()> {
        let height = self.inner.height();
        let stale = self.inner.stale_computed_columns(only);
        if stale.is_empty() {
            return Ok(());
        }
        let mut base: Option<DataFrame> = None;
        for (name, meta) in stale {
            let (lb, vr) = (meta.lookback, meta.valid_rows);
            if meta.state.is_some() {
                if height == vr + 1 {
                    if let Some(value) = volas_directive::exec::execute_resume_default_series_one(
                        &self.inner,
                        &meta.directive,
                        vr,
                    ) {
                        self.inner
                            .update_computed_f64_value(&name, vr, value)
                            .map_err(pyerr)?;
                        continue;
                    }
                }
                if let Some((tail, new_state)) =
                    volas_directive::exec::execute_resume_default_series(
                        &self.inner,
                        &meta.directive,
                        vr,
                    )
                {
                    self.inner
                        .update_computed_tail(&name, vr, &tail)
                        .map_err(pyerr)?;
                    self.inner.set_computed_state(&name, Some(new_state));
                    continue;
                }
            }
            let node = parse(&meta.directive).map_err(value_err)?;
            // State-carry fast-path (additive): if this column carries a recursive
            // state, continue the recursion over only the new rows `[vr, height)` —
            // O(new rows), bit-identical to a full recompute — then refresh the carried
            // state. This is the high-performance append path for recursive indicators
            // (and continues correctly across a head-dropping slice, since the state is
            // self-contained and the resume never reads before `vr`). On `None` (no
            // resume kernel for this directive) we fall through to the existing
            // probe / full-recompute path unchanged — always correct.
            if let Some(state) = &meta.state {
                // Default-series resumes only read canonical input columns, so they
                // can skip building a non-computed base frame on the single-column
                // append hot path. Explicit series may reference stale computed
                // columns, so those still use the base-frame fallback below.
                if directive_uses_default_series(&node) {
                    if let Some((tail, new_state)) = volas_directive::exec::execute_resume(
                        &self.inner,
                        &node,
                        state,
                        vr,
                        meta.origin,
                    ) {
                        self.inner
                            .update_computed_tail(&name, vr, &tail)
                            .map_err(pyerr)?;
                        self.inner.set_computed_state(&name, Some(new_state));
                        continue;
                    }
                }
            }
            if base.is_none() {
                let computed_names: HashSet<String> =
                    self.inner.computed_names().into_iter().collect();
                let real_names: Vec<String> = self
                    .inner
                    .names()
                    .iter()
                    .filter(|n| !computed_names.contains(*n))
                    .cloned()
                    .collect();
                base = Some(self.inner.select(&real_names).map_err(pyerr)?);
            }
            let base = base
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("internal base frame was not initialized"))?;
            if let Some(state) = &meta.state {
                if let Some((tail, new_state)) =
                    volas_directive::exec::execute_resume(base, &node, state, vr, meta.origin)
                {
                    self.inner
                        .update_computed_tail(&name, vr, &tail)
                        .map_err(pyerr)?;
                    self.inner.set_computed_state(&name, Some(new_state));
                    continue;
                }
            }
            // A finite-memory indicator (SMA, ROC, price transforms, CDL, …) depends
            // only on a fixed trailing window, so a windowed recompute is exact and
            // O(lookback). A recursive / stateful one (EMA / Wilder / MACD / SAR /
            // cumulative OBV / HT / index) depends on the WHOLE prefix `[0, i]`, so a
            // window re-warms-up and silently diverges (the bug). Probe with a
            // `2*lookback` window that overlaps the last KNOWN row (`vr-1`): if it
            // reproduces that cached value the window is exact, else recompute the full
            // column from row 0 — O(n) but exact for every indicator. (A slice that
            // dropped its head only has the visible rows, so a stateful indicator there
            // cannot be continued past the missing history.)
            let (recomputed, off) = if lb > 0 && vr > 2 * lb {
                let start = vr - 2 * lb;
                let windowed = execute(&base.slice(start, height), &node).map_err(value_err)?;
                let cached_val = col_value(self.inner.column(&name).map_err(pyerr)?, vr - 1);
                let probe = col_value(&windowed, vr - 1 - start);
                if probe.is_finite()
                    && (probe - cached_val).abs() <= 1e-9 * cached_val.abs().max(1.0)
                {
                    (windowed, vr - start)
                } else {
                    (execute(&base, &node).map_err(value_err)?, vr)
                }
            } else {
                (execute(&base, &node).map_err(value_err)?, vr)
            };
            // Write the stale tail back into the column at its original dtype.
            let tail = recomputed.slice(off, recomputed.len());
            self.inner
                .update_computed_tail(&name, vr, &tail)
                .map_err(pyerr)?;
            // The column is now valid for all rows. If this directive supports a
            // resume, (re)capture its recursive state so the NEXT append takes the
            // O(new-rows) fast-path. This repopulates state dropped by an invalidating
            // base-column write or a head-dropping slice. `None` leaves it on the
            // fallback. (`recomputed` is the full column on the full-recompute branch
            // and the window tail otherwise; `initial_state` derives the cumulative
            // family's state from the raw inputs in `base`, so either is fine.)
            let new_state = volas_directive::exec::initial_state(&base, &node, &recomputed);
            if new_state.is_some() {
                self.inner.set_computed_state(&name, new_state);
            }
        }
        Ok(())
    }
}
