//! The time-frame (tf) state of a cumulating DataFrame and its one owner: the live
//! fold — `fold_append`, the per-bar tf-cumulation hot path that updates the forming
//! bar in place — and the rule every other change of a tf frame's rows or data
//! columns keeps the open period in line with.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::{CellWrites, Column, CombineOp, DType, DataFrame, IndexKind};
use volas_time::{aggregate_period, AggSpec, Cumulator, OpenPeriod, TimeFrame};

#[allow(unused_imports)]
use crate::*;

/// Live cumulation state carried by a tf-aware DataFrame (set via the `time_frame`
/// constructor arg or `cumulate`): the target frame, the per-column aggregators, and
/// the fine bars of the still-open (forming) period — `df.iloc[-1]` is that period's
/// running bar, which `append` keeps updating. Its fields are its own: it is built by
/// its constructors ([`seeded`](Self::seeded), [`from_cumulator`](Self::from_cumulator),
/// [`reseeded`](Self::reseeded), [`projected`](Self::projected)) and changed only by the
/// fold ([`PyDataFrame::fold_append`]) and the one write entry ([`write`](Self::write)).
///
/// Invariant: `open`'s columns are the frame's data columns — every column but the
/// cached directive ones — in the frame's order and dtypes, each supported by its
/// aggregator, and the forming row is the cumulation of `open` — except in a column that
/// became data after the period's bars were folded (added, or a cached column
/// overwritten): the fine bars never carried it, so they hold it as missing, and the
/// written forming cell stands only until the next fold makes it missing (stale and
/// visibly so, never a wrong number).
#[derive(Clone)]
pub(crate) struct TfState {
    time_frame: TimeFrame,
    cumulators: AggSpec,
    /// The bars of the current open period — the small frame whose cumulation is the
    /// forming last row (`None` only while the frame is empty), one bar per timestamp.
    /// It starts as the last constructed row, or the first bar of a new period; a bar
    /// with the same timestamp as its last bar replaces that bar rather than
    /// double-counting.
    open: Option<OpenPeriod>,
    /// The current open period's key — its start on the frame's grid
    /// (`TimeFrame::period_start`) — **lazily memoized** (`None` until recomputed).
    /// The key is invariant while a period is forming — only a rollover changes it —
    /// so caching it lets `fold_append` skip a second `period_start` on every
    /// same-period bar.
    open_key: Option<i64>,
    /// Cached fold plan (`None` until the first fold), reused across appends: a new bar
    /// and the open period share the frame's data-column schema, so one plan folds a
    /// bar into the forming row and re-cumulates the open period into it alike.
    fold_plan: Option<FoldPlan>,
    /// Whether the forming row differs from the cumulation of `open` in a column that
    /// entered the bars as missing (see the invariant): the next bar of the period then
    /// re-cumulates the whole period instead of folding into the forming row, so the
    /// written cell goes missing as the bars say.
    refold_next: bool,
}

/// The cached forming-row fold plan: the `(dst_col, src_col, combine_op)` triples,
/// plus the inner schema (name `Arc` + dtypes) and source column names they were built
/// for. The hot append path reuses it so it never re-runs the per-bar `column_pos` /
/// `agg_for` HashMap name lookups (whose SipHash dominated the per-bar cost); it is
/// rebuilt only when a schema actually changes, which `matches` detects without
/// hashing — and, on the inner side, in O(1).
#[derive(Clone)]
pub(crate) struct FoldPlan {
    inner_names: Arc<Vec<String>>,
    inner_dtypes: Vec<DType>,
    src_names: Vec<String>,
    ops: Vec<(usize, usize, CombineOp)>,
}

impl FoldPlan {
    /// The plan folding `src`'s rows (bars of the frame's data columns, see
    /// [`DataFrame::admit_data_rows`]) into `inner`'s forming row: each data column of
    /// `inner` paired with `src`'s column of that name and its aggregator's combine op.
    /// A cached directive column is not folded (its refresh computes it).
    fn build(inner: &DataFrame, src: &DataFrame, spec: &AggSpec) -> FoldPlan {
        let mut ops = Vec::with_capacity(inner.width());
        for (j, name) in inner.names().iter().enumerate() {
            if inner.is_computed(name) {
                continue;
            }
            let Some(k) = src.column_pos(name) else {
                unreachable!("a tf bar carries every data column") // LCOV_EXCL_LINE
            };
            ops.push((j, k, spec.agg_for(name).as_combine_op()));
        }
        FoldPlan {
            inner_names: inner.names_arc().clone(),
            inner_dtypes: inner.columns().iter().map(|c| c.dtype()).collect(),
            src_names: src.names().to_vec(),
            ops,
        }
    }

    /// Whether this plan still matches `inner` + `src` (no schema change).
    fn matches(&self, inner: &DataFrame, src: &DataFrame) -> bool {
        // The inner frame is pointer-stable across the fold's row-only mutations
        // (append / forming-row update), so an O(1) `Arc::ptr_eq` replaces the
        // inner-name comparison; the source names are compared by value (a column
        // turning from cached to data changes them, not the inner names). Dtypes are
        // still checked — a cell can be replaced in place, changing dtype without
        // touching the name `Arc` — but that is a cheap enum compare, never a hash.
        Arc::ptr_eq(&self.inner_names, inner.names_arc())
            && self.src_names.as_slice() == src.names()
            && self.inner_dtypes.len() == inner.width()
            && self
                .inner_dtypes
                .iter()
                .zip(inner.columns())
                .all(|(d, c)| *d == c.dtype())
    }
}

/// `frame`'s data columns — every column but the cached directive ones — in order.
fn data_columns(frame: &DataFrame) -> Vec<String> {
    frame.names().iter().filter(|n| !frame.is_computed(n)).cloned().collect()
}

/// The last (forming) row of `frame` as a bar of its data columns, at its own label
/// (`None` when the frame has no row).
fn forming_row(frame: &DataFrame) -> Option<DataFrame> {
    frame.height().checked_sub(1).map(|last| frame.data_rows(last, last + 1))
}

/// The last row of `frame` as a bar of its data columns, at its own label — a
/// constructed forming bar, the single entry of an open period.
fn forming_bar(frame: &DataFrame) -> PyResult<OpenPeriod> {
    let row = forming_row(frame).expect("a frame with rows");
    OpenPeriod::new(row).map_err(pyerr)
}

/// A change to a tf frame's cells or data columns, checked against it — every error it
/// can raise already raised — holding the frame it lands on: what [`TfState::write`]
/// takes.
pub(crate) enum TfWrite<'a> {
    /// Cell writes ([`DataFrame::check_writes`]), landing in place.
    Cells(CellWrites<'a>),
    /// A column write: `next` is the frame `inner` as the write leaves it, a copy
    /// sharing every buffer the write does not replace.
    Frame { inner: &'a mut DataFrame, next: DataFrame },
}

impl TfWrite<'_> {
    /// The frame, as it is before the change lands.
    fn frame(&self) -> &DataFrame {
        match self {
            TfWrite::Cells(w) => w.frame(),
            TfWrite::Frame { inner, .. } => inner,
        }
    }

    /// The forming row as this change leaves it: a bar of the data columns after it
    /// (`None` when the frame has no row). O(columns + cells written) — the forming row
    /// alone, never a copy of the frame.
    fn forming(&self) -> Option<DataFrame> {
        match self {
            TfWrite::Cells(w) => w.frame().height().checked_sub(1).map(|last| w.data_row_after(last)),
            TfWrite::Frame { next, .. } => forming_row(next),
        }
    }

    /// Whether this change surely leaves the forming row and the data columns as they
    /// are — cell writes that reach neither the forming row nor a cached column. Then
    /// the open period, which follows only those, stands unjudged.
    fn leaves_forming_row(&self) -> bool {
        match self {
            TfWrite::Cells(w) => w.frame().height().checked_sub(1).is_some_and(|last| !w.may_change_data_row(last)),
            TfWrite::Frame { .. } => false,
        }
    }

    /// Land the change on its frame.
    fn land(self) {
        match self {
            TfWrite::Cells(w) => w.land(),
            TfWrite::Frame { inner, next } => *inner = next,
        }
    }
}

impl TfState {
    /// The tf state of a frame built from `frame`'s rows as bars of `time_frame` — the
    /// one seed rule, for the `time_frame` constructor argument and for a re-labelled or
    /// re-typed copy alike: the rows need a `DatetimeIndex` without `NaT`, in time order;
    /// each row is a bar labelled with its period start, and the last one is the forming
    /// bar — the open period's first and only bar, so a finer bar of its period folds
    /// into it (it must not already hold the finer bars appended after it). Every data
    /// column's aggregator is checked against its dtype first.
    pub(crate) fn seeded(time_frame: TimeFrame, cumulators: AggSpec, frame: &DataFrame) -> PyResult<TfState> {
        match frame.index().kind() {
            IndexKind::Datetime(v, _) => {
                // The last row becomes the open period's first bar, so the live-fold
                // invariants must hold from the start: a present, non-decreasing
                // timeline — symmetric with the append-fold and batch-cumulate guards.
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
        let data = frame.select(&data_columns(frame)).map_err(pyerr)?;
        cumulators.check(&data).map_err(pyerr)?; // dtype x aggregator, before the state exists
        let open = match frame.height() {
            0 => None,
            _ => Some(forming_bar(frame)?),
        };
        Ok(TfState { time_frame, cumulators, open, open_key: None, fold_plan: None, refold_next: false })
    }

    /// The tf state of a `cumulate` result: the cumulator's open period carries on.
    pub(crate) fn from_cumulator(time_frame: TimeFrame, cumulators: AggSpec, cum: &Cumulator) -> TfState {
        let open = cum.open_period().cloned();
        TfState { time_frame, cumulators, open, open_key: None, fold_plan: None, refold_next: false }
    }

    /// This state's time frame and aggregators over the rows of `frame` (a re-labelled or
    /// re-typed copy): [`seeded`](Self::seeded) anew.
    pub(crate) fn reseeded(&self, frame: &DataFrame) -> PyResult<TfState> {
        TfState::seeded(self.time_frame, self.cumulators.clone(), frame)
    }

    /// This state for `inner`, a column projection of its frame (the same rows): the open
    /// period keeps every bar of the data columns `inner` keeps; a column the projection
    /// turns into data enters the bars as missing.
    pub(crate) fn projected(&self, inner: &DataFrame) -> PyResult<TfState> {
        let mut t = self.clone();
        t.fold_plan = None; // the columns changed: rebuilt on the next fold
        let row = forming_row(inner);
        t.follow(row.as_ref(), row.as_ref())?; // every kept value is unchanged
        Ok(t)
    }

    /// The target time frame.
    pub(crate) fn time_frame(&self) -> TimeFrame {
        self.time_frame
    }

    /// THE write entry of a tf frame: `change` (cells written, a column added or
    /// replaced) is checked already, and the open period follows what it LEAVES in the
    /// forming row, however many writes reached it (see [`follow`](Self::follow)) — judged
    /// on the forming row alone, before anything lands. Only then does the change land,
    /// so a refused write leaves the frame unchanged. A change that surely leaves the
    /// forming row and the data columns as they are (cells of closed rows only, no cached
    /// column) leaves the open period as it is: it lands at once, at a plain write's cost.
    pub(crate) fn write(&mut self, change: TfWrite<'_>) -> PyResult<()> {
        if change.leaves_forming_row() {
            change.land();
            return Ok(());
        }
        let mut state = self.clone();
        state.follow(forming_row(change.frame()).as_ref(), change.forming().as_ref())?;
        change.land();
        *self = state;
        Ok(())
    }

    /// Bring the open period in line with `after`, the forming row `before` once the
    /// frame's rows or data columns changed in place (same height) — each a bar of the
    /// frame's data columns at the forming row (`None` for a frame without rows), which
    /// is all the rule reads. For each data column of `after`:
    /// - its aggregator must fold its dtype (else `TypeError`);
    /// - a column the open period's bars do not carry (a new data column, a cached column
    ///   become data) enters them as missing, so the next fold makes its forming cell
    ///   missing;
    /// - a column they carry keeps its bars — converted losslessly to a new dtype — while
    ///   its forming value is unchanged; a new forming value rewrites the period's one
    ///   bar, and over several fine bars (no single bar to rewrite) is refused with
    ///   `ValueError`. A column whose bars cannot take a new dtype losslessly counts as a
    ///   new forming value.
    fn follow(&mut self, before: Option<&DataFrame>, after: Option<&DataFrame>) -> PyResult<()> {
        let (Some(before), Some(after)) = (before, after) else {
            self.open = None;
            self.open_key = None;
            return Ok(());
        };
        let Some(open) = &self.open else {
            self.open = Some(OpenPeriod::new(after.clone()).map_err(pyerr)?); // LCOV_EXCL_LINE (a frame with rows has an open period)
            return Ok(()); // LCOV_EXCL_LINE
        };
        let held = open.bars();
        let n = held.height();
        let (names, mut columns) = (after.names().to_vec(), Vec::new());
        for name in &names {
            let cell = after.column(name).map_err(pyerr)?;
            let dtype = cell.dtype();
            let op = self.cumulators.agg_for(name).as_combine_op();
            if !op.supports(dtype) {
                return Err(pyerr(volas_core::VolasError::DType(format!(
                    "cumulators: cannot {} the {dtype} column {name:?}",
                    op.name()
                ))));
            }
            let Ok(bars) = held.column(name) else {
                self.refold_next = true;
                columns.push(Column::na_of(dtype, n));
                continue;
            };
            let kept = bars.convert_lossless(dtype).ok().filter(|_| {
                let was = before.column(name);
                was.ok().and_then(|w| w.convert_lossless(dtype).ok()).is_some_and(|w| w.equals(cell))
            });
            columns.push(match kept {
                Some(bars) => bars,
                None if n == 1 => cell.clone(),
                None => {
                    return Err(PyValueError::new_err(format!(
                        "cannot overwrite column {name:?} of the forming row: it is the cumulation \
                         of the {n} fine bars folded into the open period, so no single value \
                         replaces it exactly. Write to closed rows only, or append a corrected fine \
                         bar (a bar with the same timestamp replaces the one it repeats)."
                    )))
                }
            });
        }
        let bars = DataFrame::new(names, columns, Some((**held.index()).clone())).map_err(pyerr)?;
        self.open = Some(OpenPeriod::new(bars).map_err(pyerr)?);
        Ok(())
    }
}

impl PyDataFrame {
    /// Fold incoming bars into a tf-aware frame. The open period's bars are kept in a
    /// small frame (`tfs.open`, one bar per timestamp); a bar of that period joins it —
    /// REPLACING the bar with the same timestamp, so a re-sent bar is never
    /// double-counted — and its cumulation replaces `inner`'s last (forming) row,
    /// marking only that row of each cached column stale. A bar of a later period
    /// starts a new small frame and appends a new forming row. Assumes `self.tf` is
    /// `Some`.
    ///
    /// All or nothing: the bars are validated whole first — timestamps present and
    /// in order, then admitted against the frame and given exactly its data columns
    /// ([`DataFrame::admit_data_rows`]: no cached or unknown column, matching index
    /// awareness, values converted losslessly, a data column a bar lacks a missing
    /// value), so the open period, the fold and the refold read one complete bar
    /// whatever the append form left out. Several bars are then folded once on a probe
    /// holding only the forming row (a fold's outcome depends on nothing older), so an
    /// overflow in a later bar is found before the frame is touched. Each step writes
    /// `inner` first, through a mutation that checks itself before its first write,
    /// and only then the open period.
    pub(crate) fn fold_append(&mut self, fine: &DataFrame) -> PyResult<()> {
        let fine_ts: &[i64] = match fine.index().kind() {
            IndexKind::Datetime(v, _) => v.as_slice(),
            _ => {
                return Err(PyValueError::new_err(
                    "append to a time_frame DataFrame requires a DatetimeIndex",
                ))
            }
        };
        // R4-P1-01 / R4-P1-02: a live fold must see present, non-decreasing
        // timestamps: a NaT bar has no period (symmetric with the cumulate() entry's D2
        // rejection), and a bar earlier than the latest one already folded would roll
        // over into a non-monotonic index / fold later bars into the wrong period. Late
        // or disordered feed data must be handled explicitly by the caller, never
        // silently corrupt the OHLCV.
        let tfs = self.tf.as_ref().expect("fold_append on a plain frame");
        let mut prev_ts: Option<i64> = match tfs.open.as_ref() {
            Some(o) => Some(o.last_ts()),
            None => match self.inner.index().kind() {
                IndexKind::Datetime(v, _) if !v.is_empty() => Some(v[v.len() - 1]),
                _ => None,
            },
        };
        if fine_ts.contains(&i64::MIN) {
            return Err(PyValueError::new_err(
                "cannot append a NaT-timestamped bar to a time_frame DataFrame; a \
                 missing instant has no period (drop it or supply a real timestamp)",
            ));
        }
        // Admitted before the order is judged: an instant and a wall clock do not compare.
        let bars = self.inner.admit_data_rows(fine).map_err(pyerr)?;
        let bars = bars.as_ref();
        for &ts in fine_ts {
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
        if bars.height() > 1 {
            // the forming row's data columns only: what the fold reads and writes
            let h = self.inner.height();
            let mut probe = PyDataFrame {
                inner: self.inner.data_rows(h.saturating_sub(1), h),
                tf: self.tf.clone(),
                window: None,
            };
            probe.fold_bars(bars)?;
        }
        self.fold_bars(bars)
    }

    /// Fold admitted bars of the frame's data columns, in time order (see
    /// [`fold_append`](Self::fold_append)), one by one.
    fn fold_bars(&mut self, bars: &DataFrame) -> PyResult<()> {
        let PyDataFrame { inner, tf, window: _ } = self;
        let tfs = tf.as_mut().expect("fold_append on a plain frame");
        let frame = tfs.time_frame;
        // Borrow the bars' timestamps (they outlive this fn) — no per-bar Vec clone.
        let fine_ts: &[i64] = match bars.index().kind() {
            IndexKind::Datetime(v, _) => v.as_slice(),
            _ => unreachable!("checked by fold_append"), // LCOV_EXCL_LINE
        };
        // The period grid belongs to the FRAME's zone (its index zone; a frame with no
        // rows yet takes the first bar's), never to an incoming bar's: bars are instants,
        // and the frame decides which local day / hour they fall in.
        let tz = match inner.index().kind() {
            IndexKind::Datetime(_, tz) => *tz,
            _ => bars.index().tz(), // LCOV_EXCL_LINE (a tf frame's index is datetime)
        };
        for (i, &bar_ts) in fine_ts.iter().enumerate() {
            // A period is identified by its start on the frame's grid.
            let key = frame.period_start(bar_ts, tz).map_err(pyerr)?;
            // The open period's start is invariant while it forms, so memoize it
            // (lazily, also covering construction paths that set `open` without one)
            // and skip the second per-bar `period_start` of the forming bar.
            let same_period = match tfs.open.as_ref() {
                None => false,
                Some(open) => {
                    let open_key = match tfs.open_key {
                        Some(k) => k,
                        None => {
                            let k = frame.period_start(open.last_ts(), tz).map_err(pyerr)?;
                            tfs.open_key = Some(k);
                            k
                        }
                    };
                    open_key == key
                }
            };
            // The forming-bar update only reads the bar, so borrow it directly for the
            // common single-row append (no slice copy).
            let bar = if bars.height() == 1 {
                std::borrow::Cow::Borrowed(bars)
            } else {
                std::borrow::Cow::Owned(bars.slice(i, i + 1))
            };
            let bar = bar.as_ref();
            if !same_period {
                // Roll over: the previous forming bar is final in `inner`; the new row is a
                // stale row like any appended one, resumed from the state as of the closed
                // bar (never stepping the closed bar again). The open period switches only
                // once the row is in.
                let agg =
                    aggregate_period(bar, &tfs.cumulators, tfs.time_frame, tz).map_err(pyerr)?;
                let open = OpenPeriod::new(bar.clone()).map_err(pyerr)?;
                inner.append_new(&agg).map_err(pyerr)?;
                tfs.open = Some(open);
                tfs.refold_next = false;
                tfs.open_key = Some(key); // the new open period's key (no recompute next bar)
                continue;
            }
            let open = tfs.open.as_mut().unwrap();
            // A re-sent forming bar (same ts) replaces the open period's last bar.
            let resent = open.last_ts() == bar_ts;
            let last = inner.height() - 1;
            // A new distinct fine bar updates the forming row IN PLACE — no period
            // re-reduce, no column clone, no allocation. The fold plan is CACHED in
            // `tfs.fold_plan` and reused while the schema is unchanged, so the hot path
            // never re-runs the per-bar `column_pos` / `agg_for` HashMap name lookups
            // (whose SipHash dominated the per-bar cost). A re-sent bar re-cumulates the
            // whole open period instead (a max / min cannot be un-combined) — also in
            // place, from the same plan (the bar and the open period share the frame's
            // data columns).
            if !tfs.fold_plan.as_ref().is_some_and(|fp| fp.matches(inner, bar)) {
                tfs.fold_plan = Some(FoldPlan::build(inner, bar, &tfs.cumulators));
            }
            let ops = &tfs.fold_plan.as_ref().unwrap().ops;
            if resent {
                inner.check_refold(last, open.bars(), bar, ops).map_err(pyerr)?;
                open.push(bar).map_err(pyerr)?;
                inner.refold_forming_row(last, open.bars(), ops).map_err(pyerr)?;
                tfs.refold_next = false;
            } else if tfs.refold_next {
                // the forming row is re-cumulated from the bars, the new one included
                let mut next = open.clone();
                next.push(bar).map_err(pyerr)?;
                inner.check_refold(last, next.bars(), bar, ops).map_err(pyerr)?;
                inner.refold_forming_row(last, next.bars(), ops).map_err(pyerr)?;
                *open = next;
                tfs.refold_next = false;
            } else {
                inner.fold_forming_row(last, bar, 0, ops).map_err(pyerr)?;
                open.push(bar).map_err(pyerr)?;
            }
        }
        Ok(())
    }
}
