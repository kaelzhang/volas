//! The live time-frame fold — `fold_append` (the per-bar tf-cumulation hot path
//! that updates the forming bar in place) and its cached fold-plan builder.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::{CombineOp, DataFrame, DType, IndexKind};
use volas_time::{aggregate_period, AggSpec};

#[allow(unused_imports)]
use crate::*;

/// Plan the in-place forming-row fold: for each of `inner`'s columns, the
/// `(dst_col, src_col, combine_op)` triple pairing it with the matching bar column
/// and its aggregator. Returns `None` (so the caller takes the batch-aggregate
/// path) if any column is `Bool` / `Str`, which `Column::combine_at` cannot fold in
/// place. The common OHLCV (all-numeric) frame is always foldable.
fn build_fold_ops(
    inner: &DataFrame,
    bar: &DataFrame,
    spec: &AggSpec,
) -> Option<Vec<(usize, usize, CombineOp)>> {
    let mut ops = Vec::with_capacity(inner.width());
    for j in 0..inner.width() {
        if matches!(inner.columns()[j].dtype(), DType::Bool | DType::Utf8) {
            return None;
        }
        let name = &inner.names()[j];
        if let Some(bj) = bar.column_pos(name) {
            ops.push((j, bj, spec.agg_for(name).as_combine_op()));
        }
    }
    Some(ops)
}

/// At a period rollover the previous forming row becomes final: advance each
/// default-series recursive column's carried state over that now-closed bar (and
/// write its final value) so the next period's forming row resumes from the correct
/// anchor. Columns without a resume kernel, or explicit-`@series` ones, carry no
/// anchored state and are left to the post-append refresh. Runs once per rollover
/// (not per fine bar), so the per-column snapshot/parse is off the hot path.
fn finalize_forming_computed(inner: &mut DataFrame) -> PyResult<()> {
    let height = inner.height();
    if height == 0 {
        return Ok(());
    }
    let close_row = height - 1;
    // Iterate the (short) directive names only — no per-column `state` Vec clone — and
    // borrow each column's state straight off `inner` just before resuming it.
    for name in inner.computed_names() {
        let Ok(node) = parse(&name) else { continue };
        if !directive_uses_default_series(&node) {
            continue;
        }
        let resumed = inner.computed_resume_state(&name).and_then(|(state, origin)| {
            volas_directive::exec::execute_resume(inner, &node, state, close_row, origin)
        });
        if let Some((tail, new_state)) = resumed {
            inner.update_computed_tail(&name, close_row, &tail).map_err(pyerr)?;
            inner.set_computed_state(&name, Some(new_state));
        }
    }
    Ok(())
}

impl PyDataFrame {
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
        let PyDataFrame { inner, tf, window: _ } = self;
        let tfs = tf.as_mut().expect("fold_append on a plain frame");
        let frame = tfs.time_frame;
        // Borrow the fine bar's timestamps (it outlives this fn) — no per-bar Vec clone.
        let (fine_ts, tz): (&[i64], _) = match fine.index().kind() {
            IndexKind::Datetime(v, tz) => (v.as_slice(), *tz),
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
                // A re-sent forming bar (same ts) replaces the last open bar.
                let resent = last_dt(open) == bar_ts;
                if resent {
                    *open = open.slice(0, open.height() - 1);
                }
                open.append(bar).map_err(pyerr)?;
                let last = inner.height() - 1;
                // Fast path: a new distinct fine bar whose columns are all foldable
                // (numeric / datetime) updates the forming row IN PLACE — no period
                // re-reduce, no column clone, no allocation. The fold plan is CACHED in
                // `tfs.fold_plan` and reused while the schema is unchanged, so the hot
                // path never re-runs the per-bar `column_pos` / `agg_for` HashMap name
                // lookups (whose SipHash dominated the per-bar cost). A re-sent bar
                // (can't un-combine max/min) or a non-numeric column falls back to the
                // batch aggregate; both stale the cached directive tail identically.
                let plan_ok = !resent
                    && tfs.fold_plan.as_ref().is_some_and(|fp| fp.matches(inner, bar));
                if !resent && !plan_ok {
                    tfs.fold_plan = build_fold_ops(inner, bar, &tfs.cumulators).map(|ops| FoldPlan {
                        inner_names: inner.names_arc().clone(),
                        inner_dtypes: inner.columns().iter().map(|c| c.dtype()).collect(),
                        bar_names: bar.names().to_vec(),
                        ops,
                    });
                }
                let plan = if resent { None } else { tfs.fold_plan.as_ref() };
                if let Some(fp) = plan {
                    inner.fold_forming_row(last, bar, 0, &fp.ops).map_err(pyerr)?;
                } else {
                    let agg =
                        aggregate_period(open, &tfs.cumulators, tfs.time_frame).map_err(pyerr)?;
                    for (name, col) in agg.names().iter().zip(agg.columns()) {
                        if let Some(j) = inner.column_pos(name) {
                            inner.assign_positions(j, &[last], col).map_err(pyerr)?;
                        }
                    }
                }
            } else {
                // Roll over: the previous forming bar (if any) is already final in
                // `inner`. Advance each cached recursive column's anchor over that
                // now-closed bar BEFORE opening the next period, so the new forming
                // row resumes from the correct closed-bar state.
                finalize_forming_computed(inner)?;
                let bar = fine.slice(i, i + 1);
                let agg = aggregate_period(&bar, &tfs.cumulators, tfs.time_frame).map_err(pyerr)?;
                tfs.open = Some(bar);
                tfs.open_key = Some(key); // the new open period's key (no recompute next bar)
                inner.append(&agg).map_err(pyerr)?;
            }
        }
        Ok(())
    }
}
