//! The live time-frame fold — `fold_append` (the per-bar tf-cumulation hot path
//! that updates the forming bar in place) and its cached fold-plan builder.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::{CombineOp, DataFrame, DType, IndexKind};
use volas_time::{aggregate_period, AggSpec};

#[allow(unused_imports)]
use crate::*;

/// Plan the in-place forming-row fold: for each of `inner`'s columns the bar carries,
/// the `(dst_col, src_col, combine_op)` triple pairing it with the bar column and its
/// aggregator. A column the bar does not carry (a cached directive) is not folded.
/// Returns `None` (so the caller re-cumulates the period) if a carried column is
/// `Bool` / `Str`, which `Column::combine_at` cannot fold in place. The common OHLCV
/// (all-numeric) frame is always foldable.
pub(crate) fn build_fold_ops(
    inner: &DataFrame,
    bar: &DataFrame,
    spec: &AggSpec,
) -> Option<Vec<(usize, usize, CombineOp)>> {
    let mut ops = Vec::with_capacity(inner.width());
    for j in 0..inner.width() {
        let name = &inner.names()[j];
        let Some(bj) = bar.column_pos(name) else { continue };
        if matches!(inner.columns()[j].dtype(), DType::Bool | DType::Utf8) {
            return None;
        }
        ops.push((j, bj, spec.agg_for(name).as_combine_op()));
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
    ///
    /// All or nothing: the bars are validated whole first — timestamps present and
    /// in order, then admitted against the frame ([`DataFrame::admit`]: no cached or
    /// unknown column, matching index awareness, values converted losslessly). Several
    /// bars are then folded once on a probe holding only the forming row (a fold's
    /// outcome depends on nothing older), so an overflow in a later bar is found before
    /// the frame is touched. Each step writes `inner` first, through a mutation that
    /// checks itself before its first write, and only then the open period.
    pub(crate) fn fold_append(&mut self, fine: &DataFrame) -> PyResult<()> {
        let last_dt = |df: &DataFrame| -> i64 {
            match df.index().kind() {
                IndexKind::Datetime(v, _) => v[v.len() - 1],
                _ => unreachable!("checked by caller"),
            }
        };
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
            Some(o) => Some(last_dt(o)),
            None => match self.inner.index().kind() {
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
        let bars = self.inner.admit(fine).map_err(pyerr)?;
        let bars = bars.as_ref();
        if bars.height() > 1 {
            let h = self.inner.height();
            let mut probe = PyDataFrame {
                inner: self.inner.slice_data(h.saturating_sub(1), h),
                tf: self.tf.clone(),
                window: None,
            };
            probe.fold_bars(bars)?;
        }
        self.fold_bars(bars)
    }

    /// Fold admitted, ordered bars (see [`fold_append`](Self::fold_append)) one by one.
    fn fold_bars(&mut self, bars: &DataFrame) -> PyResult<()> {
        let last_dt = |df: &DataFrame| -> i64 {
            match df.index().kind() {
                IndexKind::Datetime(v, _) => v[v.len() - 1],
                _ => unreachable!("checked by caller"),
            }
        };
        let PyDataFrame { inner, tf, window: _ } = self;
        let tfs = tf.as_mut().expect("fold_append on a plain frame");
        let frame = tfs.time_frame;
        // Borrow the bars' timestamps (they outlive this fn) — no per-bar Vec clone.
        let (fine_ts, tz): (&[i64], _) = match bars.index().kind() {
            IndexKind::Datetime(v, tz) => (v.as_slice(), *tz),
            _ => unreachable!("checked by fold_append"), // LCOV_EXCL_LINE
        };
        for (i, &bar_ts) in fine_ts.iter().enumerate() {
            let key = frame.unify_tz(bar_ts, tz);
            // The open period's key is invariant while it forms, so memoize it (lazily,
            // also covering construction paths that set `open` without a key) and skip
            // the second per-bar `unify_tz` of the forming bar's timestamp.
            let same_period = match tfs.open.as_ref() {
                None => false,
                Some(open) => {
                    let open_key = match tfs.open_key {
                        Some(k) => k,
                        None => {
                            let k = frame.unify_tz(last_dt(open), tz);
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
                let agg = aggregate_period(bar, &tfs.cumulators, tfs.time_frame).map_err(pyerr)?;
                inner.append(&agg).map_err(pyerr)?;
                tfs.open = Some(bar.clone());
                tfs.open_key = Some(key); // the new open period's key (no recompute next bar)
                continue;
            }
            let open = tfs.open.as_mut().unwrap();
            // A re-sent forming bar (same ts) replaces the last open bar.
            let resent = last_dt(open) == bar_ts;
            let last = inner.height() - 1;
            // Fast path: a new distinct fine bar whose columns are all foldable (numeric /
            // datetime) updates the forming row IN PLACE — no period re-reduce, no column
            // clone, no allocation. The fold plan is CACHED in `tfs.fold_plan` and reused
            // while the schema is unchanged, so the hot path never re-runs the per-bar
            // `column_pos` / `agg_for` HashMap name lookups (whose SipHash dominated the
            // per-bar cost). A re-sent bar re-cumulates the whole open period instead (a
            // max / min cannot be un-combined) — also in place, from its own cached plan.
            let (src, cached) = if resent {
                (&*open, &mut tfs.refold_plan)
            } else {
                (bar, &mut tfs.fold_plan)
            };
            if !cached.as_ref().is_some_and(|fp| fp.matches(inner, src)) {
                *cached = FoldPlan::build(inner, src, &tfs.cumulators);
            }
            match cached.as_ref() {
                Some(fp) if resent => {
                    inner.check_refold(last, open, bar, &fp.ops).map_err(pyerr)?;
                    open.replace_last_row(bar, 0).map_err(pyerr)?;
                    inner.refold_forming_row(last, open, &fp.ops).map_err(pyerr)?;
                }
                Some(fp) => {
                    inner.fold_forming_row(last, bar, 0, &fp.ops).map_err(pyerr)?;
                    let rows = onto(open, bar)?;
                    open.append(&rows).map_err(pyerr)?;
                }
                None => {
                    // A bool / str bar column: re-cumulate the open period and REPLACE the
                    // forming row with it — in place, for every column dtype. Only that row
                    // changed, so the cached columns keep their states.
                    let mut next = open.clone();
                    if resent {
                        next.replace_last_row(bar, 0).map_err(pyerr)?;
                    } else {
                        let rows = onto(&next, bar)?;
                        next.append(&rows).map_err(pyerr)?;
                    }
                    let agg =
                        aggregate_period(&next, &tfs.cumulators, tfs.time_frame).map_err(pyerr)?;
                    let agg = inner.admit(&agg).map_err(pyerr)?;
                    inner.replace_last_row(&agg, 0).map_err(pyerr)?;
                    *open = next;
                }
            }
        }
        Ok(())
    }
}

/// `bar` restricted to the columns of the open period's small frame: a column the
/// period's first bar did not carry is not tracked in it (the fold plan reads the bar
/// itself), so it is left out rather than rejected.
fn onto<'a>(open: &DataFrame, bar: &'a DataFrame) -> PyResult<std::borrow::Cow<'a, DataFrame>> {
    if bar.names().iter().all(|n| open.has_column(n)) {
        return Ok(std::borrow::Cow::Borrowed(bar));
    }
    let kept: Vec<String> = bar.names().iter().filter(|n| open.has_column(n)).cloned().collect();
    Ok(std::borrow::Cow::Owned(bar.select(&kept).map_err(pyerr)?))
}
