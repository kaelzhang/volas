//! The `volas.DataFrame` pyclass: its fields and the non-`#[pymethods]` helpers.
//! `PyRow` lives in `frame_row`; the Python method surface is split by concern
//! across `frame_construct` / `frame_reduce` / `frame_transform` / `frame_reshape`
//! / `frame_io` / `frame_repr` / `frame_access` / `frame_index` / `frame_window`.

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use volas_core::{
    binary_supertype, CmpOp, Column, CombineOp, DataFrame, DType, Index,
    IndexKind, Series,
};
use volas_time::{AggSpec, TimeFrame};

#[allow(unused_imports)]
use crate::*;

// --- DataFrame state -------------------------------------------------------

/// Live cumulation state carried by a tf-aware DataFrame (set via the
/// `time_frame` constructor arg or `cumulate`): the target frame, the per-column
/// aggregators, and the raw fine bars of the still-open (forming) period —
/// `df.iloc[-1]` is that period's running bar, which `append` keeps updating.
#[derive(Clone)]
pub(crate) struct TfState {
    pub(crate) time_frame: TimeFrame,
    pub(crate) cumulators: AggSpec,
    /// The bars of the current open period — the small frame whose cumulation is the
    /// forming last row (`None` only while the frame is empty). It starts as the last
    /// constructed row, or the first bar of a new period; a bar with the same timestamp
    /// as one in it replaces that bar rather than double-counting.
    pub(crate) open: Option<DataFrame>,
    /// The current open period's key — its start on the frame's grid
    /// (`TimeFrame::period_start`) — **lazily memoized** (`None` until recomputed).
    /// The key is invariant while a period is forming — only a rollover changes it —
    /// so caching it lets `fold_append` skip a second `period_start` on every
    /// same-period bar.
    /// `Option` (not a bare `i64`) so a construction path that resets `open` without
    /// touching this can't leave a stale key: `None` just recomputes on next use.
    pub(crate) open_key: Option<i64>,
    /// Cached live-fold plan (`None` until the first fold), reused across appends.
    pub(crate) fold_plan: Option<FoldPlan>,
    /// Cached plan re-cumulating the open period into the forming row (a re-sent bar).
    pub(crate) refold_plan: Option<FoldPlan>,
}

/// The cached forming-row fold plan: the `(dst_col, src_col, combine_op)` triples,
/// plus the inner schema (name `Arc` + dtypes) and bar column names they were built
/// for. The hot append path reuses it so it never re-runs the per-bar `column_pos` /
/// `agg_for` HashMap name lookups (whose SipHash dominated the per-bar cost); it is
/// rebuilt only when a schema actually changes, which `matches` detects without
/// hashing — and, on the inner side, in O(1).
#[derive(Clone)]
pub(crate) struct FoldPlan {
    pub(crate) inner_names: Arc<Vec<String>>,
    pub(crate) inner_dtypes: Vec<DType>,
    pub(crate) bar_names: Vec<String>,
    pub(crate) ops: Vec<(usize, usize, CombineOp)>,
}

impl FoldPlan {
    /// The plan folding `src`'s rows into `inner`'s forming row, or `None` when a column
    /// cannot fold in place (see [`build_fold_ops`]).
    pub(crate) fn build(inner: &DataFrame, src: &DataFrame, spec: &AggSpec) -> Option<FoldPlan> {
        crate::frame_tf::build_fold_ops(inner, src, spec).map(|ops| FoldPlan {
            inner_names: inner.names_arc().clone(),
            inner_dtypes: inner.columns().iter().map(|c| c.dtype()).collect(),
            bar_names: src.names().to_vec(),
            ops,
        })
    }

    /// Whether this plan still matches `inner` + `bar` (no schema change).
    pub(crate) fn matches(&self, inner: &DataFrame, bar: &DataFrame) -> bool {
        // The inner frame is pointer-stable across the fold's row-only mutations
        // (append / forming-row update), so an O(1) `Arc::ptr_eq` replaces the
        // inner-name comparison; the bar is a fresh frame each append, so its (short)
        // names are compared by value. Dtypes are still checked — a cell can be
        // replaced in place, changing dtype without touching the name `Arc` — but
        // that is a cheap enum compare, never a hash.
        Arc::ptr_eq(&self.inner_names, inner.names_arc())
            && self.bar_names.as_slice() == bar.names()
            && self.inner_dtypes.len() == inner.width()
            && self
                .inner_dtypes
                .iter()
                .zip(inner.columns())
                .all(|(d, c)| *d == c.dtype())
    }
}

/// Whether a column takes part in a column-wise reduction (`sum` / `prod` / `min` /
/// `max` / `mean` / `var` / `std` / `median` / `sem` / `skew` / `kurt` / `corr` /
/// `cov` / `any` / `all`): the numeric columns and bool (counted as 0 / 1, pandas)
/// — one predicate, so no reduction drops a column another one keeps.
pub(crate) fn reduces(col: &Column) -> bool {
    col.require_numeric().is_ok()
}

/// A frame's values as a reader may see them: fresh (no cached directive column waits
/// for a refresh) and only the visible rows. Only the read gate
/// ([`PyDataFrame::read`]) makes one, so every helper that takes a `&Fresh` reads
/// values the gate admitted.
pub(crate) struct Fresh<'a> {
    /// The whole (physical) frame.
    frame: &'a DataFrame,
    /// The visible rows `[start, start + len)`; the whole frame when not windowed.
    start: usize,
    len: usize,
    /// The visible rows as their own frame, sliced on first use (windowed only).
    view: std::cell::OnceCell<DataFrame>,
}

impl std::ops::Deref for Fresh<'_> {
    type Target = DataFrame;
    fn deref(&self) -> &DataFrame {
        if self.start == 0 && self.len == self.frame.height() {
            return self.frame;
        }
        self.view.get_or_init(|| self.frame.slice(self.start, self.start + self.len))
    }
}

impl AsRef<DataFrame> for Fresh<'_> {
    fn as_ref(&self) -> &DataFrame {
        self
    }
}

impl Fresh<'_> {
    /// The visible rows as an owned frame.
    pub(crate) fn into_owned(self) -> DataFrame {
        let whole = self.start == 0 && self.len == self.frame.height();
        match self.view.into_inner() {
            Some(view) => view,
            None if whole => self.frame.clone(),
            None => self.frame.slice(self.start, self.start + self.len),
        }
    }

    /// Column `j` as stored with the offset of visible row 0 in it: one cell read in
    /// O(1), without slicing a windowed frame's visible rows.
    pub(crate) fn cells(&self, j: usize) -> (&Column, usize) {
        (&self.frame.columns()[j], self.start)
    }

    /// The number of visible rows.
    pub(crate) fn rows(&self) -> usize {
        self.len
    }
}

/// The value-free view of a frame ([`PyDataFrame::meta`]): what a caller may read
/// without the read gate — names, dtypes, the visible row range and labels.
pub(crate) struct Meta<'a>(&'a PyDataFrame);

impl Meta<'_> {
    /// The visible rows: `(start, len)` in the physical frame.
    pub(crate) fn range(&self) -> (usize, usize) {
        self.0.logical_range()
    }

    /// The visible rows' labels.
    pub(crate) fn labels(&self) -> std::borrow::Cow<'_, Index> {
        self.0.logical_index()
    }
}

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
/// Passing ``window=`` makes a **bounded rolling-window** frame: only the last
/// ``window`` rows are visible, while a hidden margin of older rows
/// (``max_lookback``) is retained behind them, and memory stays bounded no matter how
/// many bars you ``append`` (ideal as a fixed-size NN feature buffer). Every indicator
/// it computes equals an unbounded frame's bit for bit: the directives declared in
/// ``max_lookback`` (computed over all the given rows; a recursion keeps a checkpoint of
/// its history across the periodic front-drop), and any other directive whose values
/// depend on a trailing window the margin holds — anything else raises. Every
/// row-facing surface (indexing, ``to_numpy``, ``to_csv``, reductions, …) sees only the
/// window; ``ready`` reports whether it has warmed up; ``fill_into`` writes the window
/// straight into a preallocated array with no per-bar allocation::
///
///     wf = volas.DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])
///     wf.append(bar); wf.fulfill()        # fold a bar, refresh the cached atr:14
///     x = wf[['close', 'atr:14']].to_numpy('float32')   # the 30×2 feature window
///
/// Args:
///     data (dict[str, Sequence] | DataFrame): a dict of column name -> equal-length
///         values, or another volas DataFrame to copy (its index and tf-state are
///         carried — like ``df.copy()``). A pandas DataFrame is not accepted; use
///         ``DataFrame.from_pandas``. Build a DatetimeIndex from a column with ``read_csv`` or
///         ``to_datetime`` + ``set_index`` (+ ``tz_localize`` / ``tz_convert``).
///     columns (list[str], optional): select and order the columns to keep (like
///         ``df[[...]]``: a stale cached column is refreshed, a windowed source gives its
///         visible rows); a name not present raises ``KeyError``. An empty list or a
///         duplicate name is rejected, and an absent column is never NaN-filled.
///     time_frame (str | TimeFrame, optional): make this a tf-aware (cumulating) frame at
///         this bar interval. The given rows are bars at that frame, each labelled with its
///         period start; the last one is the forming bar, so a later ``append`` of a finer
///         bar in its period folds into it. Requires a DatetimeIndex.
///     cumulators (dict[str, str], optional): per-column aggregator overrides for folding
///         (e.g. ``{'amount': 'sum'}``); only meaningful together with ``time_frame``.
///     dtype (str, optional): cast every column to a single dtype at construction
///         (e.g. ``'float32'``), like pandas ``DataFrame(data, dtype=...)``.
///     window (int, optional): make this a bounded rolling-window frame showing only
///         the last ``window`` rows (see above). Requires ``max_lookback``.
///     max_lookback (int | list[str], optional): REQUIRED with ``window`` (and valid only
///         with it). A **list of indicator directives** declares them: the hidden margin
///         is the largest of their lookbacks (e.g. ``['atr:14', 'ma:50']`` -> margin 49),
///         and each is computed at construction over all the given rows and cached — a
///         recursion (EMA, Wilder smoothing, a cumulative line, …) can only be computed
///         this way; one no carried state covers (inside an ``@`` operand, combined with
///         other values, or a command that recomputes) raises. An **int** is the margin
///         alone. A directive requested later is computed only when it depends on a
///         trailing window of at most the margin's rows, else it raises. A list entry must
///         be an indicator directive (e.g. ``'ma:50'``); a bare/typo'd name (``'ma50'``)
///         and an empty list are rejected.
#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    pub(crate) inner: DataFrame,
    /// Cumulation state when this is a tf-aware frame; `None` for a plain frame.
    pub(crate) tf: Option<TfState>,
    /// Bounded rolling-window state when this is a windowed frame; `None` otherwise.
    pub(crate) window: Option<WindowState>,
}

impl PyDataFrame {
    /// Wrap a core frame as a plain (non-cumulating) DataFrame — the default for
    /// every derived frame (slices, projections, head/tail, ...).
    pub(crate) fn plain(inner: DataFrame) -> Self {
        PyDataFrame { inner, tf: None, window: None }
    }

    /// The logical `[start, len)` row range a windowed frame exposes (the last
    /// `window` rows); `(0, height)` for an unbounded frame.
    pub(crate) fn logical_range(&self) -> (usize, usize) {
        match &self.window {
            Some(w) => {
                let h = self.inner.height();
                let len = h.min(w.window);
                (h - len, len)
            }
            None => (0, self.inner.height()),
        }
    }

    /// THE read gate — the one freshness rule every reader of this frame's values goes
    /// through, and the only way to obtain them: while a cached directive column is stale
    /// (an append or a column write left it to recompute), a read raises instead of
    /// returning or deriving from its placeholders. The check covers the whole physical
    /// frame, a windowed frame's hidden margin included (a visible slice would no longer
    /// know the column is cached). The [`Fresh`] it returns holds the visible rows only.
    /// A column read (`df[d]`, `df[[...]]`, `get_column`) refreshes the columns it reads
    /// first; every other read and every frame operand (comparison right-hand side,
    /// `where` / mask condition, append source, constructor source) reads through this.
    /// O(cached columns).
    pub(crate) fn read(&self) -> PyResult<Fresh<'_>> {
        self.require_fresh()?;
        let (start, len) = self.logical_range();
        Ok(Fresh { frame: &self.inner, start, len, view: std::cell::OnceCell::new() })
    }

    /// The freshness rule shared by the read gate and the write path (`write_cells`).
    fn require_fresh(&self) -> PyResult<()> {
        match self.inner.stale_cause() {
            None => Ok(()),
            Some(cause) => Err(PyValueError::new_err(format!(
                "frame has stale computed (directive) columns after {cause}; call fulfill() \
                 before a bulk or positional read or write"
            ))),
        }
    }

    /// The value-free view of the frame a caller may read without the gate: names,
    /// dtypes, the visible row range and labels — none of a column's values.
    pub(crate) fn meta(&self) -> Meta<'_> {
        Meta(self)
    }

    /// The logical view a windowed frame presents — the last `window` rows, as an
    /// owned sub-frame via `slice` (which carries the directive resume state, SP-9).
    /// An unbounded frame borrows `inner`. Private: values leave through [`read`].
    fn logical(&self) -> std::borrow::Cow<'_, DataFrame> {
        match &self.window {
            Some(_) => {
                let (start, len) = self.logical_range();
                std::borrow::Cow::Owned(self.inner.slice(start, start + len))
            }
            None => std::borrow::Cow::Borrowed(&self.inner),
        }
    }

    /// The logical M view's index (the last `window` labels), as a borrow when
    /// unbounded and an owned slice when windowed — labels, no values (see [`Meta`]).
    fn logical_index(&self) -> std::borrow::Cow<'_, Index> {
        match &self.window {
            Some(_) => {
                let (start, len) = self.logical_range();
                std::borrow::Cow::Owned(self.inner.index().slice(start, start + len))
            }
            None => std::borrow::Cow::Borrowed(self.inner.index()),
        }
    }
    /// Present a full-length column as a Series sliced to the logical M view (with
    /// the matching sliced index) — identity (a plain `wrap_series`) when unbounded.
    /// Backs windowed `df[name]` so a column read never exposes the hidden margin.
    pub(crate) fn present_series(&self, name: String, col: Column) -> PySeries {
        match &self.window {
            Some(_) => {
                let (start, len) = self.logical_range();
                PySeries {
                    inner: Series::new(
                        Some(name),
                        col.slice(start, start + len),
                        Arc::new(self.inner.index().slice(start, start + len)),
                    ),
                }
            }
            None => self.wrap_series(name, col),
        }
    }

    /// Element-wise comparison backing `__eq__` / `__ne__`: against another
    /// DataFrame (identical column names + shared index) or a scalar (broadcast),
    /// producing a bool DataFrame. Compared by position; never auto-aligned.
    pub(crate) fn compare(&self, other: &Bound<'_, PyAny>, op: CmpOp) -> PyResult<PyDataFrame> {
        // Compare the logical M views (zero-cost borrow when unbounded) so the
        // result and the operand both span only the visible window.
        let view = self.read()?;
        let df = view.as_ref();
        let cols: Vec<Column> = if let Ok(o) = other.extract::<PyRef<PyDataFrame>>() {
            let oview = o.read()?;
            let odf = oview.as_ref();
            if df.names() != odf.names() {
                return Err(PyValueError::new_err(
                    "cannot compare DataFrames with different columns",
                ));
            }
            require_aligned(df.index(), odf.index())?;
            df.columns()
                .iter()
                .zip(odf.columns())
                .map(|(a, b)| a.compare(b, op))
                .collect::<Result<_, _>>()
                .map_err(pyerr)?
        } else {
            // a scalar is broadcast and typed per column; a column whose dtype the
            // scalar cannot match is a TypeError (no silent all-False mask).
            df.columns()
                .iter()
                .map(|c| c.compare(&cmp_scalar_col(other, c.dtype(), c.len())?, op).map_err(pyerr))
                .collect::<PyResult<_>>()?
        };
        self.with_columns(cols)
    }

    /// Rebuild a plain frame from `cols`, reusing this frame's names and index (the
    /// columns must be height-aligned). Backs `compare` / `fillna` / `mask_na`.
    pub(crate) fn with_columns(&self, cols: Vec<Column>) -> PyResult<PyDataFrame> {
        // Build over the logical M view's names + index (a zero-cost borrow when
        // unbounded). Callers pass logical-length columns (iterating `logical()`),
        // so a windowed frame's hidden margin never reaches a derived frame.
        let view = self.logical();
        let df = view.as_ref();
        DataFrame::new(df.names().to_vec(), cols, Some((**df.index()).clone()))
            .map(|d| PyDataFrame::plain(d.with_origin(df.origin())))
            .map_err(pyerr)
    }

    /// Column-wise transform chokepoint: map every column of the logical M view
    /// through `op` and rebuild over that view's index. The single windowed-safe
    /// path behind `ffill`/`bfill`/`isna`/`round`/`clip`/… — one logical slice,
    /// margin never leaks.
    pub(crate) fn map_columns(&self, op: impl Fn(&Column) -> PyResult<Column>) -> PyResult<PyDataFrame> {
        let view = self.read()?;
        let df = view.as_ref();
        let cols = df.columns().iter().map(&op).collect::<PyResult<Vec<_>>>()?;
        DataFrame::new(df.names().to_vec(), cols, Some((**df.index()).clone()))
            .map(|d| PyDataFrame::plain(d.with_origin(df.origin())))
            .map_err(pyerr)
    }

    /// Apply a Series transform to every column -> a new frame (pandas column-wise
    /// `df.cumsum()` etc.). Each column's own dtype rule applies; a column the op
    /// rejects (e.g. a string column under a numeric transform) propagates its error.
    pub(crate) fn map_cols(&self, op: impl Fn(&PySeries) -> PyResult<PySeries>) -> PyResult<PyDataFrame> {
        // Windowed: derive from the logical M view (zero-cost borrow when unbounded),
        // so the hidden margin never leaks into a column-wise transform.
        let view = self.read()?;
        let df = view.as_ref();
        let idx = df.index();
        let cols = df
            .names()
            .iter()
            .zip(df.columns())
            .map(|(name, col)| {
                let s = PySeries {
                    inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(idx)),
                };
                Ok(op(&s)?.inner.data)
            })
            .collect::<PyResult<Vec<_>>>()?;
        DataFrame::new(df.names().to_vec(), cols, Some((**idx).clone()))
            .map(|d| PyDataFrame::plain(d.with_origin(df.origin())))
            .map_err(pyerr)
    }

    /// Reduce each reducible column ([`reduces`]) to a float64 statistic -> a Series
    /// indexed by column name (pandas column-wise `df.sem()` etc.). `op` reads the
    /// column's float64 domain values (`op_name` names it): an int64 past 2^53
    /// raises (contract #26).
    pub(crate) fn reduce_cols(&self, op_name: &str, op: impl Fn(&[f64]) -> f64) -> PyResult<PySeries> {
        let view = self.read()?;
        let df = view.as_ref();
        let mut names = Vec::new();
        let mut vals = Vec::new();
        for (name, col) in df.names().iter().zip(df.columns()).filter(|(_, c)| reduces(c)) {
            names.push(name.clone());
            vals.push(op(&col.to_f64_domain(op_name).map_err(pyerr)?));
        }
        Ok(PySeries {
            inner: Series::new(None, Column::f64(vals), Arc::new(Index::str(names))),
        })
    }

    /// Directional fill (`forward` = ffill, else bfill) over every column,
    /// delegating to the per-column validity-aware `Column::fill_dir` so int /
    /// bool / str holes carry directionally too (like the Series version), not
    /// only float NaN. Backs `ffill` / `bfill`.
    pub(crate) fn fill_dir(&self, forward: bool) -> PyResult<PyDataFrame> {
        self.map_columns(|c| Ok(c.fill_dir(forward)))
    }

    /// Pairwise matrix (corr / cov) over the reducible columns ([`reduces`]); result column `j`
    /// is `[op(col_i, col_j) for i]`, indexed and labelled by the column names.
    /// Backs `corr` / `cov`.
    pub(crate) fn corr_cov(&self, op_name: &str, op: fn(&[f64], &[f64]) -> f64) -> PyResult<PyDataFrame> {
        let view = self.read()?;
        let df = view.as_ref();
        // float64 math: an int64 past 2^53 raises (contract #26)
        let numeric: Vec<(String, Vec<f64>)> = df
            .names()
            .iter()
            .zip(df.columns())
            .filter(|(_, c)| reduces(c))
            .map(|(n, c)| Ok((n.clone(), c.to_f64_domain(op_name).map_err(pyerr)?)))
            .collect::<PyResult<_>>()?;
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
        // Operate on the logical M view (zero-cost borrow when unbounded) so a
        // windowed frame's `cond` is matched against — and the result built from —
        // the visible rows, never the hidden margin.
        let view = self.read()?;
        let base = view.as_ref();
        let cview = cond.read()?;
        let cond_df = cview.as_ref();
        if cond_df.width() != base.width() || cond_df.height() != base.height() {
            return Err(PyValueError::new_err(
                "where/mask: `cond` must have the same shape as the frame",
            ));
        }
        // F36: a DataFrame condition pairs with columns by NAME, never by
        // position — a same-set/different-order cond is reordered to match, and
        // a different name set is an error (silently mis-applying a mask to the
        // wrong column is the worst failure mode for signal filtering).
        let cond_inner = if cond_df.names() == base.names() {
            cond_df.clone()
        } else {
            let mut sorted_a: Vec<&String> = base.names().iter().collect();
            let mut sorted_b: Vec<&String> = cond_df.names().iter().collect();
            sorted_a.sort();
            sorted_b.sort();
            if sorted_a != sorted_b {
                return Err(PyValueError::new_err(
                    "where/mask: `cond` columns must match the frame's columns by name",
                ));
            }
            cond_df.select(base.names()).map_err(pyerr)?
        };
        // and the row labels must agree (the one alignment rule) — a cond built on a
        // different index would silently filter the wrong rows.
        require_aligned(base.index(), cond_inner.index())?;
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
        let cols = base
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
                    let target = if kd.is_float() || kd == odt {
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

    /// THE cell write path — every positional, label and mask write (`iloc` / `loc` /
    /// `iat` / `at` assignment, `df[mask] = v`): each `(column, visible rows, values)`
    /// write lands in place (copy-on-write), the values already of the column's dtype
    /// (or one value, broadcast). A stale frame accepts no write — the rule of the read
    /// gate, decided here once. Writing in place keeps every cached column's identity: a
    /// written cached column becomes plain data, every other one goes stale
    /// (`DataFrame::assign_positions`). A write selecting no row changes nothing.
    pub(crate) fn write_cells(&mut self, writes: Vec<(usize, Vec<usize>, Column)>) -> PyResult<()> {
        self.require_fresh()?;
        let start = self.logical_range().0;
        for (col, rows, values) in writes {
            if rows.is_empty() {
                continue;
            }
            let phys: Vec<usize> = rows.into_iter().map(|r| r + start).collect();
            self.inner.assign_positions(col, &phys, &values).map_err(pyerr)?;
        }
        Ok(())
    }

    /// `df[row_mask] = v`: set every column's True rows to the scalar, keeping each
    /// column's dtype (pandas' whole-row boolean assignment). Every column's typed fill
    /// is resolved first (a lossy fill fails before any write), then written through
    /// `write_cells`.
    pub(crate) fn assign_row_mask(&mut self, mask: &[bool], value: &Bound<'_, PyAny>) -> PyResult<()> {
        let len = self.logical_range().1;
        if mask.len() != len {
            return Err(PyValueError::new_err(format!(
                "boolean mask length {} != frame height {}",
                mask.len(),
                len
            )));
        }
        let rows: Vec<usize> = mask.iter().enumerate().filter_map(|(i, &m)| m.then_some(i)).collect();
        let writes = if rows.is_empty() {
            Vec::new()
        } else {
            self.inner
                .columns()
                .iter()
                .enumerate()
                .map(|(j, c)| Ok((j, rows.clone(), scalar_to_column(value, c.dtype())?)))
                .collect::<PyResult<Vec<_>>>()?
        };
        self.write_cells(writes)
    }

    /// `df[bool_frame] = v`: per-cell assignment where the mask is True, keeping
    /// each column's dtype, through `write_cells`. The condition frame is an operand
    /// read through its gate, must be boolean (the contract of `DataFrame.where`: a
    /// numeric / string mask is rejected up front), pairs with the columns by NAME and
    /// with the rows by the one alignment rule.
    pub(crate) fn assign_cell_mask(&mut self, cond: &PyDataFrame, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let len = self.logical_range().1;
        let cview = cond.read()?;
        if cview.width() != self.inner.width() || cview.height() != len {
            return Err(PyValueError::new_err(
                "df[mask] = v: `mask` must have the same shape as the frame",
            ));
        }
        let cond_df = if cview.names() == self.inner.names() {
            cview.into_owned()
        } else {
            let mut want: Vec<&String> = self.inner.names().iter().collect();
            let mut got: Vec<&String> = cview.names().iter().collect();
            want.sort();
            got.sort();
            if want != got {
                return Err(PyValueError::new_err(
                    "df[mask] = v: `mask` columns must match the frame's columns by name",
                ));
            }
            cview.select(self.inner.names()).map_err(pyerr)?
        };
        self.logical_index().require_aligned(cond_df.index()).map_err(pyerr)?;
        if let Some(cc) = cond_df.columns().iter().find(|c| !matches!(c, Column::Bool(..))) {
            return Err(PyTypeError::new_err(format!(
                "df[mask] = v: `mask` must be a boolean frame, got a {} column",
                cc.dtype()
            )));
        }
        let mut writes = Vec::new();
        for (j, (col, cond_col)) in self.inner.columns().iter().zip(cond_df.columns()).enumerate() {
            let rows: Vec<usize> =
                bool_mask_vec(cond_col)?.iter().enumerate().filter_map(|(i, &m)| m.then_some(i)).collect();
            if !rows.is_empty() {
                writes.push((j, rows, scalar_to_column(value, col.dtype())?));
            }
        }
        self.write_cells(writes)
    }
    /// Per-cell missing mask -> a bool frame; backs `isna` (want_na=true) /
    /// `notna`. Reads the column validity (every dtype), so an int/bool/str NA
    /// and a datetime NaT are detected, not just a float NaN.
    pub(crate) fn mask_na(&self, want_na: bool) -> PyResult<PyDataFrame> {
        self.map_columns(|c| {
            Ok(Column::bool((0..c.len()).map(|i| c.is_valid(i) != want_na).collect()))
        })
    }

    /// Build a one-row bar frame from a scalar `dict` for `append`. The timestamp lives
    /// under the key equal to the index's name (a labeled index) and is read in its own
    /// zone — an aware value an instant, a naive one a wall clock, an integer epoch-ns in
    /// the frame's — so the append's index rule judges its awareness; a `RangeIndex`
    /// auto-extends. Every other key becomes a bar column (a data column of the frame
    /// typed by it, any other key as inferred), so the append's admission judges an
    /// unknown key or a cached directive column exactly as for every other form. The
    /// dict form is strict besides: every data column must be supplied.
    pub(crate) fn bar_from_dict(&self, dict: &Bound<'_, PyDict>) -> PyResult<DataFrame> {
        let inner = &self.inner;
        let labeled = !matches!(inner.index().kind(), IndexKind::Range { .. });
        let idx_name: Option<String> = if labeled {
            Some(inner.index().name().map(str::to_string).ok_or_else(|| {
                PyValueError::new_err(
                    "append(dict): the frame's index is unnamed — name it (set_index) so a \
                     bar's timestamp has a key, or append a Row / DataFrame instead",
                )
            })?)
        } else {
            None
        };
        // The frame's data columns, in its order, then any other key.
        let mut names = Vec::new();
        let mut cols = Vec::new();
        for name in inner.names() {
            if inner.is_computed(name) {
                continue;
            }
            let v = dict.get_item(name)?.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "append(dict): missing column {name:?} — every data column must be provided"
                ))
            })?;
            let dt = inner.column(name).map_err(pyerr)?.dtype();
            cols.push(crate::coerce::bar_scalar_to_column(&v, dt)?);
            names.push(name.clone());
        }
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            if idx_name.as_deref() == Some(key.as_str()) || names.contains(&key) {
                continue;
            }
            cols.push(pyany_to_column(&pyo3::types::PyList::new(dict.py(), [v])?.into_any())?);
            names.push(key);
        }
        // The one-row index. A RangeIndex auto-extends (None → a fresh range of len 1).
        let index = match idx_name {
            None => None,
            Some(key) => {
                let ts = dict.get_item(&key)?.ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "append(dict): missing index key {key:?} (the bar's timestamp)"
                    ))
                })?;
                Some(match inner.index().kind() {
                    IndexKind::Datetime(_, frame_tz) => {
                        let tz = match crate::convert::timestamp_zone(&ts)? {
                            crate::convert::Zone::Aware(tz) => tz,
                            crate::convert::Zone::Naive => volas_core::Tz::Naive,
                            crate::convert::Zone::Unzoned => *frame_tz,
                        };
                        let ns = crate::convert::parse_ts_in_tz(&ts, tz)?;
                        Index::datetime(vec![ns], tz)
                    }
                    IndexKind::Int64(_) => {
                        Index::from_column(&crate::coerce::bar_scalar_to_column(&ts, DType::I64)?)
                            .map_err(pyerr)?
                    }
                    IndexKind::Str(_) => {
                        Index::from_column(&crate::coerce::bar_scalar_to_column(&ts, DType::Utf8)?)
                            .map_err(pyerr)?
                    }
                    IndexKind::Range { .. } => unreachable!("labeled excludes Range"), // LCOV_EXCL_LINE
                })
            }
        };
        DataFrame::new(names, cols, index).map_err(pyerr)
    }


    pub(crate) fn wrap_series(&self, name: String, col: Column) -> PySeries {
        PySeries {
            inner: Series::new(Some(name), col, Arc::clone(self.inner.index())),
        }
    }
}
