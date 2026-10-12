//! The live time-frame fold — `fold_append` (the per-bar tf-cumulation hot path
//! that updates the forming bar in place) and its cached fold-plan builder.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::{CombineOp, DataFrame, IndexKind};
use volas_time::{aggregate_period, AggSpec};

#[allow(unused_imports)]
use crate::*;

/// Plan the in-place forming-row fold: for each of `inner`'s columns the bar carries,
/// the `(dst_col, src_col, combine_op)` triple pairing it with the bar column and its
/// aggregator. A column the bar does not carry (a cached directive) is not folded.
/// Returns `None` (so the caller re-cumulates the period, which validates and
/// converts every column before it writes) if a carried column arrives in another
/// dtype than the frame's, or has an aggregation the support table rejects — so an
/// in-place fold never fails half-way. Every dtype folds in place otherwise
/// (`bool` / `str` columns by their `first` / `last`).
pub(crate) fn build_fold_ops(
    inner: &DataFrame,
    bar: &DataFrame,
    spec: &AggSpec,
) -> Option<Vec<(usize, usize, CombineOp)>> {
    let mut ops = Vec::with_capacity(inner.width());
    for j in 0..inner.width() {
        let name = &inner.names()[j];
        let Some(bj) = bar.column_pos(name) else { continue };
        let (dtype, op) = (inner.columns()[j].dtype(), spec.agg_for(name).as_combine_op());
        if bar.columns()[bj].dtype() != dtype || !op.supports(dtype) {
            return None;
        }
        ops.push((j, bj, op));
    }
    Some(ops)
}

impl PyDataFrame {
    /// Fold incoming bars into a tf-aware frame. The open period's bars are kept in a
    /// small frame (`tfs.open`); a bar of that period joins it — REPLACING a bar with
    /// the same timestamp, so a re-sent bar is never double-counted — and its
    /// cumulation replaces `inner`'s last (forming) row, marking only that row of each
    /// cached column stale. A bar of a later period starts a new small frame and
    /// appends a new forming row. Assumes `self.tf` is `Some`.
    pub(crate) fn fold_append(&mut self, fine: &DataFrame) -> PyResult<()> {
        let last_dt = |df: &DataFrame| -> i64 {
            match df.index().kind() {
                IndexKind::Datetime(v, _) => v[v.len() - 1],
                _ => unreachable!("checked by caller"),
            }
        };
        let PyDataFrame { inner, tf, window: _ } = self;
        let tfs = tf.as_mut().expect("fold_append on a plain frame");
        let frame = tfs.time_frame;
        // Borrow the fine bar's timestamps (it outlives this fn) — no per-bar Vec clone.
        let fine_ts: &[i64] = match fine.index().kind() {
            IndexKind::Datetime(v, _) => v.as_slice(),
            _ => {
                return Err(PyValueError::new_err(
                    "append to a time_frame DataFrame requires a DatetimeIndex",
                ))
            }
        };
        // The period grid belongs to the FRAME's zone (its index zone; a frame with no
        // rows yet takes the first bar's), never to an incoming bar's: bars are instants,
        // and the frame decides which local day / hour they fall in.
        let tz = match inner.index().kind() {
            IndexKind::Datetime(_, tz) => *tz,
            _ => match fine.index().kind() {
                IndexKind::Datetime(_, tz) => *tz,
                _ => unreachable!("checked above"),
            },
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
        for &ts in fine_ts {
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
                            let k = frame.period_start(last_dt(open), tz).map_err(pyerr)?;
                            tfs.open_key = Some(k);
                            k
                        }
                    };
                    open_key == key
                }
            };
            if same_period {
                // The forming-bar update only reads the bar, so borrow it directly
                // for the common single-row append (no slice copy).
                let bar = if fine.height() == 1 {
                    std::borrow::Cow::Borrowed(fine)
                } else {
                    std::borrow::Cow::Owned(fine.slice(i, i + 1))
                };
                let bar = bar.as_ref();
                let open = tfs.open.as_mut().unwrap();
                // A re-sent forming bar (same ts) replaces the last open bar, in place.
                let resent = last_dt(open) == bar_ts;
                if resent {
                    open.replace_last_row(bar, 0).map_err(pyerr)?;
                } else {
                    open.append(bar).map_err(pyerr)?;
                }
                let last = inner.height() - 1;
                // Fast path: a new distinct fine bar whose columns are all foldable
                // (in the frame's own dtypes) updates the forming row IN PLACE — no period
                // re-reduce, no column clone, no allocation. The fold plan is CACHED in
                // `tfs.fold_plan` and reused while the schema is unchanged, so the hot
                // path never re-runs the per-bar `column_pos` / `agg_for` HashMap name
                // lookups (whose SipHash dominated the per-bar cost). A re-sent bar
                // (can't un-combine max/min) or a bar column of another dtype re-cumulates
                // the period instead; both stale only the forming row.
                // A re-sent bar re-cumulates the whole open period instead (a max / min
                // cannot be un-combined) — also in place, from its own cached plan.
                let (src, cached) = if resent {
                    (&*open, &mut tfs.refold_plan)
                } else {
                    (bar, &mut tfs.fold_plan)
                };
                if !cached.as_ref().is_some_and(|fp| fp.matches(inner, src)) {
                    *cached = FoldPlan::build(inner, src, &tfs.cumulators);
                }
                if let Some(fp) = cached.as_ref() {
                    if resent {
                        inner.refold_forming_row(last, src, &fp.ops).map_err(pyerr)?;
                    } else {
                        inner.fold_forming_row(last, src, 0, &fp.ops).map_err(pyerr)?;
                    }
                } else {
                    // A bar column of another dtype (or an aggregation the support table
                    // rejects): re-cumulate the open period, which converts or raises
                    // before writing, and REPLACE
                    // the forming row with it — in place, for every column dtype. Only
                    // that row changed, so the cached columns keep their states.
                    let agg = aggregate_period(open, &tfs.cumulators, tfs.time_frame, tz)
                        .map_err(pyerr)?;
                    inner.replace_last_row(&agg, 0).map_err(pyerr)?;
                }
            } else {
                // Roll over: the previous forming bar is final in `inner`; the new row is
                // a stale row like any appended one, resumed from the state as of the
                // closed bar (never stepping the closed bar again).
                let bar = fine.slice(i, i + 1);
                let agg =
                    aggregate_period(&bar, &tfs.cumulators, tfs.time_frame, tz).map_err(pyerr)?;
                tfs.open = Some(bar);
                tfs.open_key = Some(key); // the new open period's key (no recompute next bar)
                inner.append(&agg).map_err(pyerr)?;
            }
        }
        Ok(())
    }
}
