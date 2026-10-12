//! The cached-directive refresh: recomputing the stale tail of materialized
//! directive columns after an append, a replaced last row, or a time-frame forming
//! bar — kept in its own module so the refresh policy lives apart from the frame's
//! construction / access surface.
//!
//! A cached column keeps the directive it computes, bound against the frame when it was
//! cached (`ComputedMeta::directive`): a refresh continues exactly that tree — never a
//! re-parse of the column's name — over the live frame, whose cached columns it never
//! reads (every bare name in the bound tree is a data column or an explicit command).
//!
//! Every frame (plain or time-frame) keeps, per recursive cached column, the state as
//! of its last valid row and the state as of the row before (`ComputedMeta`). A new
//! row resumes from the first in one step; a replaced last row first drops back to the
//! second (`DataFrame::invalidate_computed_forming_row`) and resumes from it — so a
//! refresh never re-steps or recomputes the rows before the stale ones.

use std::borrow::Cow;

use pyo3::prelude::*;
use volas_core::DataFrame;
use volas_directive::types::Ast;

#[allow(unused_imports)]
use crate::*;

/// A stale cached column's cursor as the refresh decides on it: its lookback, valid
/// rows and whether its processed history is past the warm-up.
#[derive(Clone, Copy)]
struct Stale {
    lb: usize,
    resume: usize,
    vr: usize,
    warm: bool,
}

impl PyDataFrame {
    /// The carried resume states `(state, state_before_last)` of a recursive column
    /// freshly computed over `frame` (`col` is that computation): the state as of
    /// `frame`'s last row and as of the row before it. `(None, None)` for a directive
    /// with no resume kernel.
    pub(crate) fn capture_resume_states(
        frame: &DataFrame,
        node: &Ast,
        col: &volas_core::Column,
    ) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        let after = volas_directive::exec::initial_state(frame, node, col);
        let h = frame.height();
        if after.is_none() || h == 0 {
            return (after, None);
        }
        let before = volas_directive::exec::initial_state(
            &frame.slice_data(0, h - 1),
            node,
            &col.slice(0, h - 1),
        );
        (after, before)
    }

    /// The frame a NEW directive binds and computes over (`df[d]`, `exec`): the data
    /// columns only, so a name never reads a cached column — whose rows may be a stale
    /// placeholder — and is never resolved differently from a later refresh. Borrowed
    /// when the frame caches nothing.
    pub(crate) fn binding_frame(&self) -> PyResult<Cow<'_, DataFrame>> {
        self.frame_without(|n| self.inner.is_computed(n))
    }

    /// This frame without the columns `drop` names (borrowed when it names none).
    fn frame_without(&self, drop: impl Fn(&str) -> bool) -> PyResult<Cow<'_, DataFrame>> {
        if !self.inner.names().iter().any(|n| drop(n)) {
            return Ok(Cow::Borrowed(&self.inner));
        }
        let kept: Vec<String> = self.inner.names().iter().filter(|n| !drop(n)).cloned().collect();
        Ok(Cow::Owned(self.inner.select(&kept).map_err(pyerr)?))
    }

    /// Recompute the stale tail of cached directive columns in place — all of
    /// them if `only` is `None`, else just the named one. O(new rows) per recursive
    /// column, O(lookback) per finite-window one.
    pub(crate) fn refresh_computed(&mut self, only: Option<&str>) -> PyResult<()> {
        let height = self.inner.height();
        for stale in self.inner.stale_computed_columns(only) {
            let volas_core::StaleComputed { name, directive, reach, valid_rows: vr, warm } = stale;
            let stale = Stale { lb: reach.lookback, resume: reach.resume, vr, warm };
            self.refresh_column(&name, bound_directive(&directive), stale, height)?;
        }
        Ok(())
    }

    /// Refresh one cached column whose rows `[vr, height)` are stale, continuing its
    /// bound directive `node` over the live frame.
    fn refresh_column(
        &mut self,
        name: &str,
        node: &Ast,
        Stale { lb, resume, vr, warm }: Stale,
        height: usize,
    ) -> PyResult<()> {
        // Past warm-up with a carried state, a column resumes. Warm-up is counted on the
        // column's processed history, not on this frame's row numbers (a tail slice's
        // replaced last row is still past it). The gate keeps the warm-up region on the
        // full recompute: an indicator whose warm-up mask is length-dependent would
        // otherwise freeze an incrementally carried early row that a full recompute
        // masks. A resume reads `resume_reach` rows before the new one (its lookback,
        // plus the row its window drops or SAR's earlier bar): a frame holding fewer (a
        // tail slice) recomputes over its own rows instead.
        let reach = vr >= resume;
        if warm && reach && self.inner.computed_resume_state(name).is_some() {
            // One row at a time through the allocation-free step: each row's value and
            // the state as of it; the last two states are kept.
            let mut row = vr;
            while row < height {
                let value = self.inner.step_computed_state(name, |live, prev, _, next| {
                    volas_directive::exec::execute_resume_step(live, node, prev, row, next)
                });
                let Some(value) = value else { break };
                self.inner.update_computed_f64_value(name, row, value).map_err(pyerr)?;
                row += 1;
            }
            if row == height {
                return Ok(());
            }
            if row > vr {
                // a step declined part-way: the states no longer match `valid_rows`
                self.inner.set_computed_states(name, None, None);
            }
            // No single-row step (or it declined at the first row): the `Vec` resume
            // over the stale rows.
            if row == vr {
                let resumed = self.inner.computed_resume_state(name).and_then(|(state, origin)| {
                    volas_directive::exec::execute_resume(&self.inner, node, state, vr, origin)
                });
                if let Some((tail, new_state)) = resumed {
                    self.inner.update_computed_tail(name, vr, &tail).map_err(pyerr)?;
                    if vr + 1 == height {
                        self.inner.advance_computed_state(name, new_state);
                    } else {
                        // the state before the last of several new rows is not produced
                        self.inner.set_computed_state(name, Some(new_state));
                    }
                    return Ok(());
                }
            }
        }
        let df = &self.inner;
        // A directive whose every command has finite memory and a position-free
        // output (SMA, ROC, price transforms, CDL, …) depends only on a fixed
        // trailing window, so recomputing a window is exact and O(lookback): the
        // window reaches `lb + 1` rows behind the first stale row (one more than the
        // lookback covers indicators that emit from a partial first difference) and
        // doubles that as margin. The decision is structural (`recomputes_on_a_window`),
        // never inferred from the values. It carries no state.
        let win = 2 * (lb + 1);
        if vr > win && volas_directive::lookback::recomputes_on_a_window(node) {
            let start = vr - win;
            // `slice_data` skips the per-cached-column ComputedMeta clone; the window is
            // read-only and discarded.
            let windowed = volas_directive::execute(&df.slice_data(start, height), node).map_err(value_err)?;
            let tail = windowed.slice(vr - start, windowed.len());
            self.inner.update_computed_tail(name, vr, &tail).map_err(pyerr)?;
            return Ok(());
        }
        // Cold / warm-up / no resume: recompute the column and capture both states.
        // Inside warm-up (`vr < lb`) the whole column is rewritten, re-applying the
        // full-series warm-up mask to the early rows, so they are frozen correctly when
        // the resume takes over at the warm boundary; past it only the stale tail is
        // (the rows before are exact — and a head-dropped slice's carried head is not
        // reproducible from its own rows).
        let recomputed = volas_directive::execute(df, node).map_err(value_err)?;
        let (state, before) = Self::capture_resume_states(df, node, &recomputed);
        let from = if warm { vr } else { 0 };
        let tail = recomputed.slice(from, recomputed.len());
        self.inner.update_computed_tail(name, from, &tail).map_err(pyerr)?;
        self.inner.set_computed_states(name, state, before);
        Ok(())
    }
}
