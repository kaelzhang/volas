//! The `volas.DataFrame` pyclass: its fields and the non-`#[pymethods]` helpers.
//! `PyRow` lives in `frame_row`; the Python method surface is split by concern
//! across `frame_construct` / `frame_reduce` / `frame_transform` / `frame_reshape`
//! / `frame_io` / `frame_repr` / `frame_access` / `frame_index` / `frame_window`.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use volas_core::{
    binary_supertype, CmpOp, Column, CombineOp, DataFrame, DType, Index,
    IndexKind, Series,
};
use volas_directive::parse;
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
    /// Raw fine bars of the current open period (`None` until the first
    /// folded append), kept so a re-sent forming bar updates (deduped) rather
    /// than double-counts.
    pub(crate) open: Option<DataFrame>,
    /// Cached live-fold plan (`None` until the first fold), reused across appends.
    pub(crate) fold_plan: Option<FoldPlan>,
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

/// Bounded rolling-window state when this is a windowed frame (`window=`); `None`
/// for an unbounded frame. The frame physically retains `[capacity, 2*capacity]`
/// rows (`capacity = window + max_lookback`); the user-facing surface shows only the
/// last `window` rows — the margin is hidden history that keeps cached indicators
/// correct across the periodic front-drop. The drop reuses `DataFrame::slice`, which
/// carries the directive resume state (SP-9), so no core change is needed.
#[derive(Clone)]
pub(crate) struct WindowState {
    /// The output window M — the logical row count once warmed.
    pub(crate) window: usize,
    /// Rows kept for correctness: `window + max_lookback`. Physical retention is
    /// bounded by `2*capacity` (compact down to `capacity` when reached).
    pub(crate) capacity: usize,
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
/// ``window`` rows are visible, while enough older rows are retained behind the
/// scenes (``window + max_lookback``) to keep cached indicators consistent across
/// the periodic, automatic front-drop — so memory stays bounded no matter how many
/// bars you ``append`` (ideal as a fixed-size NN feature buffer). Every row-facing
/// surface (indexing, ``to_numpy``, ``to_csv``, reductions, …) sees only the
/// window; ``ready`` reports whether it has warmed up; ``fill_into`` writes the
/// window straight into a preallocated array with no per-bar allocation::
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
///         ``df[[...]]``); a name not present raises ``KeyError``. An empty list or a
///         duplicate name is rejected, and an absent column is never NaN-filled.
///     time_frame (str | TimeFrame, optional): make this a tf-aware (cumulating) frame at
///         this bar interval; the given rows are taken as already-final bars and later
///         ``append``s fold finer bars into them. Requires a DatetimeIndex.
///     cumulators (dict[str, str], optional): per-column aggregator overrides for folding
///         (e.g. ``{'amount': 'sum'}``); only meaningful together with ``time_frame``.
///     dtype (str, optional): cast every column to a single dtype at construction
///         (e.g. ``'float32'``), like pandas ``DataFrame(data, dtype=...)``.
///     window (int, optional): make this a bounded rolling-window frame showing only
///         the last ``window`` rows (see above). Requires ``max_lookback``.
///     max_lookback (int | list[str], optional): REQUIRED with ``window`` (and valid only
///         with it) — the margin of hidden history (``window + max_lookback``) the frame
///         keeps so cached indicators survive the automatic front-drop. Recursive
///         indicators (EMA/Wilder/ATR/RSI/MACD) stay **bit-exact**; finite-window
///         indicators (ma/wma/trima/stddev/…) match an unbounded frame to floating-point
///         tolerance (~1e-13). Give an **int** to state the largest indicator lookback you
///         will use, or a **list of indicator directives** to derive it from the largest of
///         their lookbacks (e.g. ``['atr:14', 'ma:50']`` -> margin 49) — so you never
///         hand-compute a compound indicator's warm-up. Too small a margin silently breaks
///         this guarantee. A list entry must be an indicator directive (e.g. ``'ma:50'``);
///         a bare/typo'd name (``'ma50'``) is rejected.
#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    pub(crate) inner: DataFrame,
    /// Cumulation state when this is a tf-aware frame; `None` for a plain frame.
    pub(crate) tf: Option<TfState>,
    /// Bounded rolling-window state when this is a windowed frame; `None` otherwise.
    pub(crate) window: Option<WindowState>,
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

    /// The logical view a windowed frame presents — the last `window` rows, as an
    /// owned sub-frame via `slice` (which carries the directive resume state, SP-9).
    /// An unbounded frame borrows `inner`. Read / export / display surfaces route
    /// through this so the hidden margin never leaks.
    pub(crate) fn logical(&self) -> std::borrow::Cow<'_, DataFrame> {
        match &self.window {
            Some(_) => {
                let (start, len) = self.logical_range();
                std::borrow::Cow::Owned(self.inner.slice(start, start + len))
            }
            None => std::borrow::Cow::Borrowed(&self.inner),
        }
    }

    /// The logical M view's index (the last `window` labels), as a borrow when
    /// unbounded and an owned slice when windowed. Backs the indexer set paths,
    /// which resolve a position/label against the visible rows before mapping it
    /// back to a physical row via [`Self::logical_range`].
    pub(crate) fn logical_index(&self) -> std::borrow::Cow<'_, Index> {
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

    /// After an append, drop the front history once physical retention reaches
    /// `2*capacity`, keeping the last `capacity` rows. The drop is `DataFrame::slice`,
    /// which carries the directive resume state (SP-9) ONLY when the kept window still
    /// holds the last valid row + its `lookback` warm-up. So we first refresh the stale
    /// directive tail (bringing `valid_rows` up to `height`, anchoring the resume state
    /// inside the kept window) — otherwise an append-many-without-read run would drop the
    /// valid region and the recursion would diverge. Refreshing only at the compaction
    /// boundary (every ~`capacity` appends) keeps the per-bar cost amortized O(lookback),
    /// the same as the lazy read-time refresh. No-op when unbounded.
    pub(crate) fn maybe_compact(&mut self) -> PyResult<()> {
        if let Some(w) = &self.window {
            let cap = w.capacity;
            if self.inner.height() >= 2 * cap {
                self.refresh_computed(None)?;
                let h = self.inner.height();
                self.inner = self.inner.slice(h - cap, h);
            }
        }
        Ok(())
    }

    /// Element-wise comparison backing `__eq__` / `__ne__`: against another
    /// DataFrame (identical column names + shared index) or a scalar (broadcast),
    /// producing a bool DataFrame. Compared by position; never auto-aligned.
    pub(crate) fn compare(&self, other: &Bound<'_, PyAny>, op: CmpOp) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        // Compare the logical M views (zero-cost borrow when unbounded) so the
        // result and the operand both span only the visible window.
        let view = self.logical();
        let df = view.as_ref();
        let cols: Vec<Column> = if let Ok(o) = other.extract::<PyRef<PyDataFrame>>() {
            let oview = o.logical();
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
            .map(PyDataFrame::plain)
            .map_err(pyerr)
    }

    /// Column-wise transform chokepoint: map every column of the logical M view
    /// through `op` and rebuild over that view's index. The single windowed-safe
    /// path behind `ffill`/`bfill`/`isna`/`round`/`clip`/… — one logical slice,
    /// margin never leaks.
    pub(crate) fn map_columns(&self, op: impl Fn(&Column) -> PyResult<Column>) -> PyResult<PyDataFrame> {
        let view = self.logical();
        let df = view.as_ref();
        let cols = df.columns().iter().map(&op).collect::<PyResult<Vec<_>>>()?;
        DataFrame::new(df.names().to_vec(), cols, Some((**df.index()).clone()))
            .map(PyDataFrame::plain)
            .map_err(pyerr)
    }

    /// Apply a Series transform to every column -> a new frame (pandas column-wise
    /// `df.cumsum()` etc.). Each column's own dtype rule applies; a column the op
    /// rejects (e.g. a string column under a numeric transform) propagates its error.
    pub(crate) fn map_cols(&self, op: impl Fn(&PySeries) -> PyResult<PySeries>) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        // Windowed: derive from the logical M view (zero-cost borrow when unbounded),
        // so the hidden margin never leaks into a column-wise transform.
        let view = self.logical();
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
            .map(PyDataFrame::plain)
            .map_err(pyerr)
    }

    /// Reduce each numeric column to a scalar -> a Series indexed by column name
    /// (pandas column-wise `df.sem()` etc.; non-numeric columns are skipped).
    pub(crate) fn reduce_cols(&self, op: impl Fn(&Column) -> f64) -> PyResult<PySeries> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let mut names = Vec::new();
        let mut vals = Vec::new();
        for (name, col) in df.names().iter().zip(df.columns()) {
            if col.dtype().is_numeric() {
                names.push(name.clone());
                vals.push(op(col));
            }
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
        ensure_fresh(&self.inner)?;
        self.map_columns(|c| Ok(c.fill_dir(forward)))
    }

    /// Pairwise matrix (corr / cov) over the numeric columns; result column `j`
    /// is `[op(col_i, col_j) for i]`, indexed and labelled by the column names.
    /// Backs `corr` / `cov`.
    pub(crate) fn corr_cov(&self, op: fn(&[f64], &[f64]) -> f64) -> PyResult<PyDataFrame> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let numeric: Vec<(String, Vec<f64>)> = df
            .names()
            .iter()
            .zip(df.columns())
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
        ensure_fresh(&self.inner)?;
        // Operate on the logical M view (zero-cost borrow when unbounded) so a
        // windowed frame's `cond` is matched against — and the result built from —
        // the visible rows, never the hidden margin.
        let view = self.logical();
        let base = view.as_ref();
        if cond.inner.width() != base.width() || cond.inner.height() != base.height() {
            return Err(PyValueError::new_err(
                "where/mask: `cond` must have the same shape as the frame",
            ));
        }
        // F36: a DataFrame condition pairs with columns by NAME, never by
        // position — a same-set/different-order cond is reordered to match, and
        // a different name set is an error (silently mis-applying a mask to the
        // wrong column is the worst failure mode for signal filtering).
        let cond_inner = if cond.inner.names() == base.names() {
            cond.inner.clone()
        } else {
            let mut sorted_a: Vec<&String> = base.names().iter().collect();
            let mut sorted_b: Vec<&String> = cond.inner.names().iter().collect();
            sorted_a.sort();
            sorted_b.sort();
            if sorted_a != sorted_b {
                return Err(PyValueError::new_err(
                    "where/mask: `cond` columns must match the frame's columns by name",
                ));
            }
            cond.inner.select(base.names()).map_err(pyerr)?
        };
        // and the row labels must agree — a cond built on a different index would
        // silently filter the wrong rows. (Arc identity first: a cond derived
        // from this frame shares the index handle, making the common case O(1).)
        if !std::sync::Arc::ptr_eq(cond_inner.index(), base.index())
            && *cond_inner.index() != *base.index()
        {
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
        let (start, len) = self.logical_range();
        if mask.len() != len {
            return Err(PyValueError::new_err(format!(
                "boolean mask length {} != frame height {}",
                mask.len(),
                len
            )));
        }
        let positions: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| m.then_some(i))
            .collect();
        if self.window.is_none() {
            // Unbounded: rebuild from the scattered columns (the established path).
            let cols = self
                .inner
                .columns()
                .iter()
                .map(|c| scatter_scalar(c, &positions, value))
                .collect::<PyResult<Vec<_>>>()?;
            return self.rebuild_with(cols);
        }
        // Windowed: write in place at the physical rows (offset by the window start)
        // so the hidden margin and the directive cache structure survive. Resolve
        // every column's typed fill FIRST (atomic — a lossy fill fails before any
        // write), then scatter.
        let phys: Vec<usize> = positions.iter().map(|p| p + start).collect();
        let vals: Vec<Option<Column>> = self
            .inner
            .columns()
            .iter()
            .map(|c| {
                if phys.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(scalar_to_column(value, c.dtype())?))
                }
            })
            .collect::<PyResult<_>>()?;
        for (j, v) in vals.into_iter().enumerate() {
            if let Some(v) = v {
                self.inner.assign_positions(j, &phys, &v).map_err(pyerr)?;
            }
        }
        Ok(())
    }

    /// `df[bool_frame] = v`: per-cell assignment where the mask is True, keeping
    /// each column's dtype. Atomic, like `assign_row_mask`. The condition frame
    /// must be boolean — the same contract as `DataFrame.where` (a numeric / string
    /// mask is rejected up front, not coerced through `x != 0.0`).
    pub(crate) fn assign_cell_mask(&mut self, cond: &PyDataFrame, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let (start, len) = self.logical_range();
        if cond.inner.width() != self.inner.width() || cond.inner.height() != len {
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
        if self.window.is_none() {
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
            return self.rebuild_with(cols);
        }
        // Windowed: scatter each column in place at the physical rows the mask
        // selects (offset by the window start), preserving the hidden margin. Resolve
        // the per-column fills first so a lossy fill fails before any write (atomic).
        let plans: Vec<(Vec<usize>, Option<Column>)> = self
            .inner
            .columns()
            .iter()
            .zip(cond.inner.columns())
            .map(|(col, cond_col)| {
                let phys: Vec<usize> = bool_mask_vec(cond_col)?
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &m)| m.then_some(i + start))
                    .collect();
                let v = if phys.is_empty() {
                    None
                } else {
                    Some(scalar_to_column(value, col.dtype())?)
                };
                Ok((phys, v))
            })
            .collect::<PyResult<_>>()?;
        for (j, (phys, v)) in plans.into_iter().enumerate() {
            if let Some(v) = v {
                self.inner.assign_positions(j, &phys, &v).map_err(pyerr)?;
            }
        }
        Ok(())
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
    /// under the key equal to the index's name (a labeled index); a `RangeIndex` auto-
    /// extends. Strict: every plain (non-directive) column must be supplied, the index key
    /// must be present (labeled index), and an unknown or directive key is an error. The
    /// frame's known column dtypes drive the coercion (no inference) — the fast bar path.
    pub(crate) fn bar_from_dict(&self, dict: &Bound<'_, PyDict>) -> PyResult<DataFrame> {
        let inner = &self.inner;
        let labeled = !matches!(inner.index().kind(), IndexKind::Range(_));
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
        // Validate every key: the index key, or a plain column. Reject unknown / directive.
        for k in dict.keys() {
            let ks: String = k.extract()?;
            if idx_name.as_deref() == Some(ks.as_str()) {
                continue;
            }
            if !inner.has_column(&ks) {
                return Err(PyValueError::new_err(format!(
                    "append(dict): unknown key {ks:?} — not a column of the frame nor its index"
                )));
            }
            if inner.is_computed(&ks) {
                return Err(PyValueError::new_err(format!(
                    "append(dict): {ks:?} is a cached directive column — recomputed by \
                     fulfill, not set per bar"
                )));
            }
        }
        // Every plain (non-directive) column must be supplied.
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
        // The one-row index. A RangeIndex auto-extends (None → a fresh range of len 1).
        let index = match idx_name {
            None => None,
            Some(key) => {
                let ts = dict.get_item(&key)?.ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "append(dict): missing index key {key:?} (the bar's timestamp)"
                    ))
                })?;
                let icol = match inner.index().kind() {
                    IndexKind::Datetime(..) => crate::coerce::bar_scalar_to_column(&ts, DType::Datetime)?,
                    IndexKind::Int64(_) => crate::coerce::bar_scalar_to_column(&ts, DType::I64)?,
                    IndexKind::Str(_) => crate::coerce::bar_scalar_to_column(&ts, DType::Utf8)?,
                    IndexKind::Range(_) => unreachable!("labeled excludes Range"), // LCOV_EXCL_LINE
                };
                Some(match inner.index().kind() {
                    IndexKind::Datetime(_, tz) => Index::from_column_tz(&icol, *tz).map_err(pyerr)?,
                    _ => Index::from_column(&icol).map_err(pyerr)?,
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
                    // Recursive single-state single-row fast path (ema/smma): the new
                    // value IS the new state `[value]`, so write the value and update the
                    // state IN PLACE — no tail/state `Vec` allocation. Bit-identical to
                    // the `Vec` resume below (same shared `*_step` kernel).
                    if let Some(state) = meta.state.as_deref() {
                        if let Some(value) =
                            volas_directive::exec::execute_resume_one(&self.inner, &meta.directive, state, vr)
                        {
                            self.inner
                                .update_computed_f64_value(&name, vr, value)
                                .map_err(pyerr)?;
                            self.inner.update_computed_state_at(&name, 0, value);
                            continue;
                        }
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
            let node = parse(&meta.directive).map_err(directive_err)?;
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
            // The base frame strips computed columns so a column-name lookup can
            // never read a STALE computed column. A default-series COMMAND node
            // only reads the canonical input columns (open/high/low/close/volume),
            // which are never computed, so it executes against the live frame
            // directly — skipping the select (a name-filter + frame rebuild) that
            // dominates the probe path's constant cost; this is the same invariant
            // the state-resume fast-path above already relies on. The probe MUST
            // therefore execute via `execute_refresh`, which dispatches a bare
            // NAME node as a command: a bare-canonical directive (`wma`,
            // `linearreg`, ... — the all-defaults spelling) resolved through
            // `execute`'s column lookup would find its own stale cache on the
            // live frame (a self-referential no-op that "verifies" and splices
            // the stale tail back). Explicit `@series` directives still pay for
            // the stripped base frame.
            let use_live = directive_uses_default_series(&node);
            if !use_live && base.is_none() {
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
            let frame: &DataFrame = if use_live {
                &self.inner
            } else {
                base.as_ref().ok_or_else(|| {
                    PyValueError::new_err("internal base frame was not initialized")
                })?
            };
            if let Some(state) = &meta.state {
                if let Some((tail, new_state)) =
                    volas_directive::exec::execute_resume(frame, &node, state, vr, meta.origin)
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
            // A lookback-0 indicator still gets the windowed path with a one-row
            // overlap probe (start = vr-1): elementwise/CDL outputs reproduce the
            // known row and splice in O(new rows); a cumulative lb-0 one (OBV)
            // fails the probe and correctly falls back to the full recompute.
            let win = (2 * lb).max(1);
            let (recomputed, off) = if vr > win {
                let start = vr - win;
                // Read-only probe: `slice_data` skips the per-cached-column ComputedMeta
                // clone (O(K) per probe, O(K²) per fulfill over a K-indicator windowed
                // frame). The probe reads only raw columns and is discarded — never
                // appended — so dropping the resume carry here is sound. (Window
                // compaction at `maybe_compact` keeps `slice`, which carries it.)
                let windowed = volas_directive::exec::execute_refresh(&frame.slice_data(start, height), &node)
                    .map_err(value_err)?;
                let cached_val = col_value(self.inner.column(&name).map_err(pyerr)?, vr - 1);
                let probe = col_value(&windowed, vr - 1 - start);
                if probe.is_finite()
                    && (probe - cached_val).abs() <= 1e-9 * cached_val.abs().max(1.0)
                {
                    (windowed, vr - start)
                } else {
                    (volas_directive::exec::execute_refresh(frame, &node).map_err(value_err)?, vr)
                }
            } else {
                (volas_directive::exec::execute_refresh(frame, &node).map_err(value_err)?, vr)
            };
            // If this directive supports a resume, (re)capture its recursive state
            // so the NEXT append takes the O(new-rows) fast-path. This repopulates
            // state dropped by an invalidating base-column write or a head-dropping
            // slice. `None` leaves it on the fallback. (`recomputed` is the full
            // column on the full-recompute branch and the window tail otherwise;
            // `initial_state` derives the cumulative family's state from the raw
            // inputs, so either is fine. Computed BEFORE the tail write so `frame`'s
            // borrow of the live frame ends before the mutation.)
            let new_state = volas_directive::exec::initial_state(frame, &node, &recomputed);
            // Write the stale tail back into the column at its original dtype.
            let tail = recomputed.slice(off, recomputed.len());
            self.inner
                .update_computed_tail(&name, vr, &tail)
                .map_err(pyerr)?;
            if new_state.is_some() {
                self.inner.set_computed_state(&name, new_state);
            }
        }
        Ok(())
    }
}
