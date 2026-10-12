//! The cached-directive refresh: recomputing the stale tail of materialized
//! directive columns after an append, a replaced last row, or a time-frame forming
//! bar — kept in its own module so the refresh policy lives apart from the frame's
//! construction / access surface.
//!
//! Every frame (plain or time-frame) keeps, per recursive cached column, the state as
//! of its last valid row and the state as of the row before (`ComputedMeta`); every
//! successful resume leaves both. A new row resumes from the first in one step; a
//! replaced last row first drops back to the second
//! (`DataFrame::invalidate_computed_forming_row`) and resumes from it — so a refresh
//! never re-steps or recomputes the rows before the stale ones. A recursion whose
//! history starts before the frame's row 0 (a bounded window dropped its head) is
//! recomputed only by replaying its checkpoint (`volas_core::Checkpoint`); without one
//! a recompute raises rather than restart the recursion on the rows the frame holds.

use std::borrow::Cow;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::DataFrame;
use volas_directive::types::Ast;

#[allow(unused_imports)]
use crate::*;

/// A stale cached column's cursor as the refresh decides on it: its reach, valid rows,
/// where its history starts and whether its processed history is past the warm-up.
#[derive(Clone, Copy)]
struct Stale {
    reach: volas_core::Reach,
    vr: usize,
    since: usize,
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

    /// The frame a CACHED directive refreshes over: the data columns, less those named
    /// like a built-in command usable bare. In a cached directive such a bare name is
    /// always the command: a data column of that name present at bind time made the name
    /// ambiguous (an error), and one added later is refused while a cached directive
    /// reads it (`check_new_data_column`) — and the canonical form renders `change:` as
    /// `change`, so the refresh must not read it as the column.
    pub(crate) fn refresh_frame(&self) -> PyResult<DataFrame> {
        Ok(self
            .frame_without(|n| self.inner.is_computed(n) || volas_directive::is_bare_command(n))?
            .into_owned())
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
        let stale = self.inner.stale_computed_columns(only);
        if stale.is_empty() {
            return Ok(());
        }
        let mut base: Option<DataFrame> = None;
        for volas_core::StaleComputed { name, reach, valid_rows: vr, since, warm } in stale {
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
                base = Some(self.refresh_frame()?);
            }
            let frame = if directive_uses_default_series(&node) { None } else { base.as_ref() };
            self.refresh_column(&name, &node, frame, Stale { reach, vr, since, warm }, height)?;
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
        Stale { reach, vr, since, warm }: Stale,
        height: usize,
    ) -> PyResult<()> {
        let lb = reach.lookback;
        // The frame holds the column's whole history (it starts at or after row 0).
        let held = since >= self.inner.origin();
        // Past warm-up with a carried state, a column resumes. Warm-up is counted on the
        // column's processed history, not on this frame's row numbers (a tail slice's
        // replaced last row is still past it). The gate keeps the warm-up region on the
        // full recompute: an indicator whose warm-up mask is length-dependent would
        // otherwise freeze an incrementally carried early row that a full recompute
        // masks. A resume reads `resume` rows before the new one (its lookback, plus the
        // row its window drops or SAR's earlier bar): a frame holding fewer (a tail
        // slice) recomputes over its own rows instead (or replays, see below). A resume
        // that would read complete rows the frame dropped declines (`since`).
        if warm && vr >= reach.resume && self.inner.computed_resume_state(name).is_some() {
            // One row at a time through the allocation-free step: each row's value and
            // the state as of it; the last two states are kept.
            let mut row = vr;
            while row < height {
                let value = self.inner.step_computed_state(name, |live, prev, _, next| {
                    volas_directive::exec::execute_resume_step(frame.unwrap_or(live), node, prev, (row, since), next)
                });
                let Some(value) = value else { break };
                self.inner.update_computed_f64_value(name, row, value).map_err(pyerr)?;
                row += 1;
            }
            if row == height {
                return Ok(());
            }
            // No single-row step (or it declined): the `Vec` resume over the rows left.
            if self.resume_rows(name, node, frame, (row, height, since))? {
                return Ok(());
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
        if reach.history.recursive() && !held {
            // The rows the recursion started on are gone: replay its checkpoint.
            return self.replay_column(name, node, frame, height);
        }
        // Cold / warm-up / no resume: recompute the column and capture both states.
        // Inside warm-up (`vr < lb`) the whole column is rewritten, re-applying the
        // full-series warm-up mask to the early rows, so they are frozen correctly when
        // the resume takes over at the warm boundary; past it only the stale tail is
        // (the rows before are exact — and a head-dropped slice's carried head is not
        // reproducible from its own rows).
        let recomputed = volas_directive::exec::execute_refresh(df, node).map_err(value_err)?;
        let (state, before) = Self::capture_resume_states(df, node, &recomputed);
        let from = if warm { vr } else { 0 };
        let tail = recomputed.slice(from, recomputed.len());
        self.inner.update_computed_tail(name, from, &tail).map_err(pyerr)?;
        self.inner.set_computed_states(name, state, before);
        Ok(())
    }

    /// Resume cached column `name` from its carried state over its stale rows `[from,
    /// height)` (the `Vec` resume), leaving both states: several rows resume up to the
    /// last, then the last from the state before it. `false` (nothing written) when the
    /// column carries no state of row `from - 1` or a resume declines.
    fn resume_rows(
        &mut self,
        name: &str,
        node: &Ast,
        frame: Option<&DataFrame>,
        (from, height, since): (usize, usize, usize),
    ) -> PyResult<bool> {
        let df = frame.unwrap_or(&self.inner);
        let resumed = self.inner.computed_resume_state(name).and_then(|(state, _)| {
            if from + 1 == height {
                return volas_directive::exec::execute_resume(df, node, state, from, height, since)
                    .map(|(tail, last)| (tail, last, None));
            }
            let (mut tail, before) =
                volas_directive::exec::execute_resume(df, node, state, from, height - 1, since)?;
            let (one, last) =
                volas_directive::exec::execute_resume(df, node, &before, height - 1, height, since)?;
            tail.append(&one).ok()?;
            Some((tail, last, Some(before)))
        });
        let Some((tail, last, before)) = resumed else { return Ok(false) };
        self.inner.update_computed_tail(name, from, &tail).map_err(pyerr)?;
        match before {
            None => self.inner.advance_computed_state(name, last),
            Some(before) => self.inner.set_computed_states(name, Some(last), Some(before)),
        }
        Ok(true)
    }

    /// Recompute recursive column `name`, whose history starts before the frame's row 0,
    /// by replaying its checkpoint over the rows after it — up to the row before the last,
    /// then the last — leaving both states. Without a checkpoint (a slice carries none;
    /// a bounded window keeps one for each declared recursion) the frame does not hold
    /// the history the values depend on, and the recompute raises.
    fn replay_column(&mut self, name: &str, node: &Ast, frame: Option<&DataFrame>, height: usize) -> PyResult<()> {
        let lost = || {
            PyValueError::new_err(format!(
                "cannot recompute '{name}' exactly: each of its values depends on every row \
                 before it, and this frame no longer holds the rows its history started on (a \
                 slice continues a recursive column only from its carried state, which a write \
                 discards; a windowed frame keeps the history of the directives declared in \
                 max_lookback). Recompute it on the frame that holds its whole history."
            ))
        };
        let df = frame.unwrap_or(&self.inner);
        let checkpoint = self.inner.computed_checkpoint(name).ok_or_else(lost)?;
        let from = (checkpoint.at + 1).checked_sub(df.origin()).ok_or_else(lost)?;
        let (mut tail, before) =
            volas_directive::exec::replay(df, node, checkpoint, height - 1).ok_or_else(lost)?;
        let (one, last) = volas_directive::exec::replay(df, node, &before, height).ok_or_else(lost)?;
        tail.append(&one).map_err(pyerr)?;
        self.inner
            .continue_computed(name, from, &tail, last.state, before.state)
            .map_err(pyerr)
    }
}
