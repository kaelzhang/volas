//! The cached-directive refresh: recomputing the stale tail of materialized
//! directive columns after an append, a replaced last row, or a time-frame forming
//! bar — kept in its own module so the refresh policy lives apart from the frame's
//! construction / access surface.
//!
//! Every frame (plain or time-frame) keeps, per recursive cached column, the state as
//! of its last valid row and the state as of the row before (`ComputedMeta`). A new
//! row resumes from the first in one step; a replaced last row first drops back to the
//! second (`DataFrame::invalidate_computed_forming_row`) and resumes from it — so a
//! refresh never re-steps or recomputes the rows before the stale ones.

use pyo3::prelude::*;
use volas_core::DataFrame;
use volas_directive::types::Ast;

#[allow(unused_imports)]
use crate::*;

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

    /// The frame a cached directive is computed over: the live frame for a
    /// default-series directive (it reads only the canonical input columns, never a
    /// cached one), else the frame stripped of cached columns, so a column-name lookup
    /// can never read a STALE cached column. Built once per refresh and shared.
    fn base_frame(&self) -> PyResult<DataFrame> {
        let real_names: Vec<String> = self
            .inner
            .names()
            .iter()
            .filter(|n| !self.inner.is_computed(n))
            .cloned()
            .collect();
        self.inner.select(&real_names).map_err(pyerr)
    }

    /// Recompute the stale tail of cached directive columns in place — all of
    /// them if `only` is `None`, else just the named one. O(new rows) per recursive
    /// column, O(lookback) per finite-window one.
    pub(crate) fn refresh_computed(&mut self, only: Option<&str>) -> PyResult<()> {
        let height = self.inner.height();
        let stale = self.inner.stale_computed_columns(only);
        if stale.is_empty() {
            return Ok(());
        }
        let mut base: Option<DataFrame> = None;
        for (name, lb, vr) in stale {
            // Parse-free single-row fast path for the stateless finite-memory directives
            // (price transforms, TR, the momentum / ROC family): the newest row is a
            // formula of its own inputs; the (empty) state still advances one row.
            if vr + 1 == height && self.inner.computed_resume_state(&name).is_some() {
                if let Some(value) =
                    volas_directive::exec::execute_resume_default_series_one(&self.inner, &name, vr)
                {
                    self.inner.update_computed_f64_value(&name, vr, value).map_err(pyerr)?;
                    self.inner.advance_computed_state(&name, Vec::new());
                    continue;
                }
            }
            let node = parse(&name).map_err(directive_err)?;
            if !directive_uses_default_series(&node) && base.is_none() {
                base = Some(self.base_frame()?);
            }
            let frame = if directive_uses_default_series(&node) { None } else { base.as_ref() };
            self.refresh_column(&name, &node, frame, lb, vr, height)?;
        }
        Ok(())
    }

    /// Refresh one cached column whose rows `[vr, height)` are stale, computing over
    /// `frame` (the live frame when `None`).
    fn refresh_column(
        &mut self,
        name: &str,
        node: &Ast,
        frame: Option<&DataFrame>,
        lb: usize,
        vr: usize,
        height: usize,
    ) -> PyResult<()> {
        // Past warm-up (`vr >= lb`) with a carried state, a recursive column resumes.
        // The gate keeps the warm-up region on the full recompute: an indicator whose
        // warm-up mask is length-dependent would otherwise freeze an incrementally
        // carried early row that a full recompute masks.
        if vr >= lb && self.inner.computed_resume_state(name).is_some() {
            // One row at a time through the allocation-free step: each row's value and
            // the state as of it; the last two states are kept.
            let mut row = vr;
            while row < height {
                let value = self.inner.step_computed_state(name, |live, prev, _, next| {
                    volas_directive::exec::execute_resume_step(frame.unwrap_or(live), node, prev, row, next)
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
                    let df = frame.unwrap_or(&self.inner);
                    volas_directive::exec::execute_resume(df, node, state, vr, origin)
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
        let df = frame.unwrap_or(&self.inner);
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
            let windowed =
                volas_directive::exec::execute_refresh(&df.slice_data(start, height), node)
                    .map_err(value_err)?;
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
        let recomputed = volas_directive::exec::execute_refresh(df, node).map_err(value_err)?;
        let (state, before) = Self::capture_resume_states(df, node, &recomputed);
        let from = if vr < lb { 0 } else { vr };
        let tail = recomputed.slice(from, recomputed.len());
        self.inner.update_computed_tail(name, from, &tail).map_err(pyerr)?;
        self.inner.set_computed_states(name, state, before);
        Ok(())
    }
}
