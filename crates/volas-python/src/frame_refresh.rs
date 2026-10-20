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
        let kept: Vec<String> = self
            .inner
            .names()
            .iter()
            .filter(|n| !drop(n))
            .cloned()
            .collect();
        Ok(Cow::Owned(self.inner.select(&kept).map_err(pyerr)?))
    }

    /// Recompute the stale tail of cached directive columns in place — all of
    /// them if `only` is `None`, else just the named one. O(new rows) per recursive
    /// column, O(lookback) per finite-window one.
    pub(crate) fn refresh_computed(&mut self, only: Option<&str>) -> PyResult<()> {
        let height = self.inner.height();
        for stale in self.inner.shared_stale_computed_columns(only) {
            let volas_core::StaleComputed {
                name,
                directive,
                reach,
                valid_rows: vr,
                since,
                warm,
                state_ready,
            } = stale;
            self.refresh_column(
                &name,
                bound_directive(&directive),
                Stale {
                    reach,
                    vr,
                    since,
                    warm,
                },
                height,
                state_ready,
            )?;
        }
        Ok(())
    }

    /// Refresh one cached column whose rows `[vr, height)` are stale, continuing its
    /// bound directive `node` over the live frame.
    fn refresh_column(
        &mut self,
        name: &str,
        node: &Ast,
        Stale {
            reach,
            vr,
            since,
            warm,
        }: Stale,
        height: usize,
        mut has_state: bool,
    ) -> PyResult<()> {
        let origin = self.inner.origin();
        // The snapshot already checked this cursor while borrowing its metadata.
        // Only a replay or step needs to borrow the state words again.
        let can_resume = warm && vr >= reach.resume;
        let mut has_resume = can_resume && has_state;
        // A selection ending before its source's last row carries a recursion's states
        // lazily: those as of its last valid row are replayed here, the first time the
        // column continues, from the nearest point of its history it keeps.
        if !has_resume
            && can_resume
            && vr > 0
            && matches!(reach.history, volas_core::History::Carried { .. })
        {
            has_resume = self.restore_carried_state(name, vr);
            has_state |= has_resume;
        }
        // The rows before row `vr` that a resume or a recompute of it reads, when the frame
        // does not hold them: its lead keeps them (a contiguous selection keeps the data
        // rows its cached columns read before its row 0), including the seed prefix
        // when a recursive continuation has not yet formed a state.
        let lead = self.inner.lead_rows();
        if since < origin && lead > 0 {
            let stateless = reach.history.recursive() && !has_state;
            let needed = if has_resume && reach.history.recursive() { reach.resume } else { reach.lead() };
            let rows = match stateless {
                true if since + lead >= origin => Some(origin - since),
                _ if vr < needed => Some(needed - vr),
                _ => None,
            };
            if let Some(rows) = rows {
                // Other cached columns may retain an older lead. This column's
                // computation starts at `since`; earlier rows must never warm it up.
                let led = self.inner.led(&self.inner, rows.min(origin - since));
                return self.refresh_led(
                    name,
                    node,
                    &led,
                    Stale {
                        reach,
                        vr,
                        since,
                        warm,
                    },
                    height,
                );
            }
        }
        // The frame holds the column's whole history (it starts at or after row 0).
        let held = since >= origin;
        // Past warm-up with a carried state, a column resumes. Warm-up is counted on the
        // column's processed history, not on this frame's row numbers (a tail slice's
        // replaced last row is still past it). The gate keeps the warm-up region on the
        // full recompute: an indicator whose warm-up mask is length-dependent would
        // otherwise freeze an incrementally carried early row that a full recompute
        // masks. A resume reads `resume` rows before the new one (its lookback, plus the
        // row its window drops or SAR's earlier bar): a frame holding fewer (a tail
        // slice) recomputes over its own rows instead (or replays, see below). A resume
        // that would read complete rows the frame dropped declines (`since`).
        if has_resume {
            // One row at a time through the allocation-free step: each row's value and
            // the state as of it; the last two states are kept. Passing the row the next
            // drop of a bounded frame's head keeps, it leaves the checkpoint there.
            let pending = self.pending_row(name, reach.history, vr);
            let mut row = vr;
            while row < height {
                let value = self.inner.resume_computed_value(name, |live, prev, _, next| {
                    volas_directive::exec::execute_resume_step(live, node, prev, (row, since), next)
                }).map_err(pyerr)?;
                let Some(_) = value else { break };
                if pending == Some(self.inner.origin() + row) {
                    self.keep_pending(name, node, row + 1, held);
                }
                row += 1;
            }
            if row == height {
                return Ok(());
            }
            // No single-row step (or it declined): the `Vec` resume over the rows left.
            if self.resume_rows(name, node, (row, height, since, held), pending)? {
                return Ok(());
            }
        }
        self.recompute_stale(name, node, Stale { reach, vr, since, warm }, height)
    }

    // A lazy selection recovers its state once. Its replay buffers must not
    // enlarge the stack of every later scalar refresh.
    #[cold]
    #[inline(never)]
    fn restore_carried_state(&mut self, name: &str, end: usize) -> bool {
        let Some((state, before)) = volas_core::Replay::continuation(
            &volas_directive::Replayer, &self.inner, name, end,
        ) else { return false; };
        self.inner.set_computed_states(name, Some(state), before);
        true
    }

    // The vector recompute owns its large temporary frame and state buffers.
    // Keeping that fallback out of refresh leaves scalar appends no stack slot
    // or register spills for buffers that only a declined resume can use.
    #[cold]
    #[inline(never)]
    fn recompute_stale(
        &mut self,
        name: &str,
        node: &Ast,
        Stale { reach, vr, since, warm }: Stale,
        height: usize,
    ) -> PyResult<()> {
        let origin = self.inner.origin();
        let held = since >= origin;
        let lb = reach.lookback;
        let df = &self.inner;
        // A directive with no recursion and no position output anywhere (SMA, ROC,
        // `close > ma:20`, price transforms, CDL, …) depends only on a fixed trailing
        // window, so recomputing a window is exact and O(lookback): the window reaches
        // `lb + 1` rows behind the first stale row (one more than the lookback covers
        // indicators that emit from a partial first difference) and doubles that as
        // margin. The decision is structural (`evaluates_on_a_window`, the rule a resume
        // reads its operands by), never inferred from the values. It carries no state.
        let win = 2 * (lb + 1);
        if vr > win && volas_directive::lookback::evaluates_on_a_window(node) {
            let start = vr - win;
            // `slice_data` skips the per-cached-column ComputedMeta clone; the window is
            // read-only and discarded.
            let windowed =
                volas_directive::execute(&df.slice_data(start, height), node).map_err(value_err)?;
            let tail = windowed.slice(vr - start, windowed.len());
            self.inner
                .update_computed_tail(name, vr, &tail)
                .map_err(pyerr)?;
            return Ok(());
        }
        if reach.history.recursive() && !held {
            // The rows the recursion started on are gone: replay its checkpoint.
            // Replay may pass rows before the stale tail.
            let pending = self.pending_row(name, reach.history, 0);
            return self.replay_column(name, node, (vr, height), pending);
        }
        // Cold / warm-up / no resume: recompute the column and capture both states.
        // Inside warm-up (`vr < lb`) the whole column is rewritten, re-applying the
        // full-series warm-up mask to the early rows, so they are frozen correctly when
        // the resume takes over at the warm boundary; past it only the stale tail is
        // (the rows before are exact — and a head-dropped slice's carried head is not
        // reproducible from its own rows).
        let (recomputed, state, before, anchors) =
            volas_directive::exec::execute_for_cache(df, node).map_err(value_err)?;
        let from = if warm { vr } else { 0 };
        let tail = recomputed.slice(from, recomputed.len());
        self.inner
            .update_computed_tail(name, from, &tail)
            .map_err(pyerr)?;
        self.inner.set_computed_states(name, state, before);
        if since == origin {
            // a history computed from its start: the anchors it passed
            self.inner.keep_computed_anchors(name, anchors);
        }
        Ok(())
    }

    /// Refresh one cached column whose rows `[vr, height)` are stale over `led` — the
    /// live frame preceded by rows of its lead, which a resume or a
    /// recompute of row `vr` reads: a carried state resumes over it (reading the rows it
    /// needs before); otherwise a column that depends on a trailing window, or whose history
    /// the lead holds from its start, is recomputed over it — its rows' values are those of
    /// the whole history — and any other recursion replays its checkpoint (or raises).
    #[inline(never)]
    fn refresh_led(
        &mut self,
        name: &str,
        node: &Ast,
        led: &DataFrame,
        Stale {
            reach,
            vr,
            since,
            warm,
        }: Stale,
        height: usize,
    ) -> PyResult<()> {
        let off = led.height() - height;
        let mut row = vr;
        if warm && vr + off >= reach.resume && self.inner.computed_resume_state(name).is_some() {
            while row < height {
                let value = self.inner.resume_computed_value(name, |_, prev, _, next| {
                    volas_directive::exec::execute_resume_step(
                        led,
                        node,
                        prev,
                        (row + off, since),
                        next,
                    )
                }).map_err(pyerr)?;
                let Some(_) = value else { break };
                row += 1;
            }
            if row == height {
                return Ok(());
            }
            // No single-row step (or it declined): the `Vec` resume over the rows left — up
            // to the row before the last, then the last, so both states are left.
            let resumed = self
                .inner
                .computed_resume_state(name)
                .and_then(|(state, _)| {
                    use volas_directive::exec::execute_resume;
                    let (mut tail, mut before, mut at) = (None, None, row);
                    if at + 1 < height {
                        let (more, b) =
                            execute_resume(led, node, state, at + off, height - 1 + off, since)?;
                        (tail, before, at) = (Some(more), Some(b), height - 1);
                    }
                    let (one, last) = execute_resume(
                        led,
                        node,
                        before.as_deref().unwrap_or(state),
                        at + off,
                        height + off,
                        since,
                    )?;
                    let tail = match tail {
                        Some(mut tail) => {
                            tail.append(&one).ok()?;
                            tail
                        }
                        None => one,
                    };
                    Some((tail, last, before))
                });
            if let Some((tail, last, before)) = resumed {
                self.inner
                    .update_computed_tail(name, row, &tail)
                    .map_err(pyerr)?;
                match before {
                    None => self.inner.advance_computed_state(name, last),
                    Some(before) => self
                        .inner
                        .set_computed_states(name, Some(last), Some(before)),
                }
                return Ok(());
            }
        }
        if reach.history.recursive() && since < led.origin() {
            return self.replay_column(name, node, (row, height), None);
        }
        let (recomputed, state, before, anchors) =
            volas_directive::exec::execute_for_cache(led, node).map_err(value_err)?;
        if since == led.origin() {
            // a history computed from its start: the anchors it passed
            self.inner.keep_computed_anchors(name, anchors);
        }
        let from = if warm { row } else { 0 };
        let tail = recomputed.slice(off + from, recomputed.len());
        self.inner
            .continue_computed(name, from, &tail, state, before)
            .map_err(pyerr)
    }

    /// The absolute row whose checkpoint the next drop of a bounded frame's head keeps for
    /// cached recursion `name` (`history`), when it is not kept yet: the drop keeps the
    /// frame's last `capacity` rows of twice that, so its head is `capacity` rows on.
    fn pending_row(&self, name: &str, history: volas_core::History, from: usize) -> Option<usize> {
        let (Some(w), volas_core::History::Carried { operands }) = (&self.window, history) else {
            return None;
        };
        // Compare frame-relative rows first: an ordinary append is already past
        // this boundary and needs neither absolute-coordinate sums nor an owner read.
        let row = w.capacity - 1 + operands;
        if row < from || row >= self.inner.height() {
            return None;
        }
        let at = self.inner.origin() + row;
        (!self.inner.has_computed_pending(name, at)).then_some(at)
    }

    /// Keep the checkpoint of cached recursion `name` as of row `end - 1`, which its
    /// carried state now describes, for the next drop of the head (`whole`: the frame
    /// holds the history's start).
    #[inline(never)]
    fn keep_pending(&mut self, name: &str, node: &Ast, end: usize, whole: bool) {
        let state = self
            .inner
            .computed_resume_state(name)
            .map(|(s, _)| s.to_vec());
        let cp = state.and_then(|s| {
            volas_directive::exec::checkpoint(&self.inner, node, Some(s), end, whole)
        });
        self.inner.set_computed_pending(name, cp);
    }

    /// Resume cached column `name` from its carried state over its stale rows `[from,
    /// height)` (the `Vec` resume), leaving both states: several rows resume up to the
    /// last, then the last from the state before it — and, when they pass `pending` (the
    /// row the next drop of a bounded frame's head keeps), up to it first, keeping its
    /// checkpoint (`whole`: the frame holds the history's start). `false` (nothing
    /// written) when the column carries no state of row `from - 1` or a resume declines.
    #[inline(never)]
    fn resume_rows(
        &mut self,
        name: &str,
        node: &Ast,
        (from, height, since, whole): (usize, usize, usize, bool),
        pending: Option<usize>,
    ) -> PyResult<bool> {
        use volas_directive::exec::{checkpoint, execute_resume};
        let df = &self.inner;
        let split = pending
            .and_then(|p| p.checked_sub(df.origin()))
            .filter(|&t| t >= from && t + 1 < height);
        let resumed = self
            .inner
            .computed_resume_state(name)
            .and_then(|(state, _)| {
                // up to the pending row, when the rows pass it before the last
                let (mut tail, mut at, mut carried, kept) = match split {
                    Some(t) => {
                        let (head, at_t) = execute_resume(df, node, state, from, t + 1, since)?;
                        let kept = checkpoint(df, node, Some(at_t.clone()), t + 1, whole);
                        (Some(head), t + 1, Some(at_t), kept)
                    }
                    None => (None, from, None, None),
                };
                // then up to the last row, and the last
                if at + 1 < height {
                    let (more, before) = execute_resume(
                        df,
                        node,
                        carried.as_deref().unwrap_or(state),
                        at,
                        height - 1,
                        since,
                    )?;
                    match &mut tail {
                        Some(t) => t.append(&more).ok()?,
                        None => tail = Some(more),
                    }
                    (at, carried) = (height - 1, Some(before));
                }
                let (one, last) = execute_resume(
                    df,
                    node,
                    carried.as_deref().unwrap_or(state),
                    at,
                    height,
                    since,
                )?;
                let tail = match tail {
                    Some(mut t) => {
                        t.append(&one).ok()?;
                        t
                    }
                    None => one,
                };
                Some((tail, last, carried, kept))
            });
        let Some((tail, last, before, kept)) = resumed else {
            return Ok(false);
        };
        self.inner
            .update_computed_tail(name, from, &tail)
            .map_err(pyerr)?;
        match before {
            None => self.inner.advance_computed_state(name, last),
            Some(before) => self
                .inner
                .set_computed_states(name, Some(last), Some(before)),
        }
        if split.is_some() {
            self.inner.set_computed_pending(name, kept);
        } else if pending == Some(self.inner.origin() + height - 1) {
            self.keep_pending(name, node, height, whole);
        }
        Ok(true)
    }

    /// Recompute recursive column `name`, whose history starts before the frame's row 0,
    /// by replaying its checkpoint over the rows after it — up to the row before the last,
    /// then the last — leaving both states (and the checkpoint of `pending`, the row the
    /// next drop of the head keeps, when the rows pass it). The rows up to the checkpoint
    /// keep their values, so they must be ones the frame vouches for: still valid (the
    /// first `vr`), or hidden (before the visible rows, which no write reaches). Without
    /// such a checkpoint (a slice carries none; a bounded window keeps one for each
    /// declared recursion) the frame does not hold the history the values depend on, and
    /// the recompute raises.
    #[inline(never)]
    fn replay_column(
        &mut self,
        name: &str,
        node: &Ast,
        (vr, height): (usize, usize),
        pending: Option<usize>,
    ) -> PyResult<()> {
        let lost = || {
            PyValueError::new_err(format!(
                "cannot recompute '{name}' exactly: each of its values depends on every row \
                 before it, and this frame no longer holds the rows its history started on (a \
                 slice continues a recursive column only from its carried state, which a write \
                 discards; a windowed frame keeps the history of the directives declared in \
                 max_lookback). Recompute it on the frame that holds its whole history."
            ))
        };
        // over the frame's lead too, when it keeps one: an operand of the recursion reads
        // the rows before row 0 (a checkpoint at a selection's boundary replays from row 0)
        let base = &self.inner;
        let led =
            (self.inner.lead_rows() > 0).then(|| self.inner.led(base, self.inner.lead_rows()));
        let df = led.as_ref().unwrap_or(base);
        let off = df.height() - height;
        // an input the directive cannot read (an int64 past 2^53) is its own error — in the
        // rows the frame holds, or in those dropped without a checkpoint
        let fail = || match self.inner.computed_history_lost(name) {
            Some(e) => value_err(e.clone()),
            None => match volas_directive::execute(df, node) {
                Err(e) => value_err(e),
                Ok(_) => lost(),
            },
        };
        let origin = self.inner.origin();
        let vouched = vr.max(self.logical_range().0);
        let checkpoint = match self
            .inner
            .computed_checkpoint(name)
            .filter(|cp| cp.at < origin + vouched)
        {
            Some(cp) if cp.at + 1 >= origin => cp.clone(),
            // the nearest point of the history before row 0 a selection keeps (a checkpoint,
            // or the history's start, with the rows after it in its lead): replayed to the
            // row before row 0
            _ => volas_directive::checkpoint_at(&self.inner, name, origin - 1).ok_or_else(fail)?,
        };
        // the frame's first row the replay writes (the rows before it keep their values)
        let from = (checkpoint.at + 1).checked_sub(origin).ok_or_else(fail)?;
        // the replay's stops, in `df`'s rows: the pending row's, the row before the last, the last
        let stop = pending
            .and_then(|p| p.checked_sub(df.origin()))
            .filter(|&t| t >= from + off && t + 2 < df.height());
        let mut ends: Vec<usize> = stop.map(|t| t + 1).into_iter().collect();
        ends.extend([df.height() - 1, df.height()]);
        let mut tail: Option<volas_core::Column> = None;
        let mut points = Vec::with_capacity(ends.len());
        let mut at = checkpoint.clone();
        for end in ends {
            let (part, next) =
                volas_directive::exec::replay(df, node, &at, end).ok_or_else(fail)?;
            if !part.is_empty() {
                match &mut tail {
                    Some(tail) => tail.append(&part).map_err(pyerr)?,
                    None => tail = Some(part),
                }
            }
            points.push(next.clone());
            at = next;
        }
        let kept = pending.and_then(|p| points.iter().find(|cp| cp.at == p).cloned());
        let (last, before) = (
            points.pop().expect("the last row"),
            points.pop().expect("the row before"),
        );
        self.inner
            .continue_computed(
                name,
                from,
                &tail.expect("a replay advances rows"),
                last.state.map(|state| state.to_vec()),
                before.state.map(|state| state.to_vec()),
            )
            .map_err(pyerr)?;
        if kept.is_some() {
            self.inner.set_computed_pending(name, kept);
        }
        Ok(())
    }
}
