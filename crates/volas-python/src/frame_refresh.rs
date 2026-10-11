//! The cached-directive refresh: recomputing the stale tail of materialized
//! directive columns after an append. Includes the time-frame forming-row refresh
//! (`refresh_tf_column`), which recomputes the forming last row from the state
//! before it — kept in its own module so the refresh policy lives apart from the
//! frame's construction / access surface.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::DataFrame;

#[allow(unused_imports)]
use crate::*;

impl PyDataFrame {
    /// The carried resume states `(state, state_before_last)` for a recursive column
    /// freshly computed over `frame` (`col` is that computation). `state` describes
    /// `frame`'s last row. A time-frame frame also keeps the state before its last
    /// (forming) row, which the forming refresh recomputes that row from; there an
    /// explicit-`@series` column carries no state at all (it always recomputes). A
    /// plain frame keeps a single state.
    pub(crate) fn capture_resume_states(
        &self,
        frame: &DataFrame,
        node: &volas_directive::types::Ast,
        col: &volas_core::Column,
    ) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        if self.tf.is_none() {
            return (volas_directive::exec::initial_state(frame, node, col), None);
        }
        if !directive_uses_default_series(node) {
            return (None, None);
        }
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
        // A tf-fold frame's last row is the forming bar (its raw OHLCV is replaced as
        // bars of its period arrive), so its cached columns refresh through the
        // two-state path (`refresh_tf_column`) rather than the plain append resume.
        let tf = self.tf.is_some();
        let mut base: Option<DataFrame> = None;
        for (name, lb, vr) in stale {
            if tf && self.refresh_tf_column(&name, lb, vr, height)? {
                continue;
            }
            if self.inner.computed_resume_state(&name).is_some() {
                if height == vr + 1 {
                    if let Some(value) = volas_directive::exec::execute_resume_default_series_one(
                        &self.inner,
                        &name,
                        vr,
                    ) {
                        self.inner
                            .update_computed_f64_value(&name, vr, value)
                            .map_err(pyerr)?;
                        continue;
                    }
                    // Recursive single-row fast path (ema / smma / atr / natr / rsi / cmo):
                    // one step yields the row's value AND the advanced carried state, both
                    // written IN PLACE — no tail/state `Vec` allocation. Bit-identical to
                    // the `Vec` resume below (same shared `*_step` kernel). The state is
                    // borrowed straight off `self.inner` and dropped before the write.
                    let scalar = self.inner.computed_resume_state(&name).and_then(|(state, _)| {
                        volas_directive::exec::execute_resume_one(&self.inner, &name, state, vr)
                    });
                    if let Some(step) = scalar {
                        self.inner
                            .update_computed_f64_value(&name, vr, step.value)
                            .map_err(pyerr)?;
                        for (i, &s) in step.state().iter().enumerate() {
                            self.inner.update_computed_state_at(&name, i, s);
                        }
                        continue;
                    }
                }
                if let Some((tail, new_state)) =
                    volas_directive::exec::execute_resume_default_series(
                        &self.inner,
                        &name,
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
            let node = parse(&name).map_err(directive_err)?;
            // State-carry fast-path (additive): if this column carries a recursive
            // state, continue the recursion over only the new rows `[vr, height)` —
            // O(new rows), bit-identical to a full recompute — then refresh the carried
            // state. This is the high-performance append path for recursive indicators
            // (and continues correctly across a head-dropping slice, since the state is
            // self-contained and the resume never reads before `vr`). On `None` (no
            // resume kernel for this directive) we fall through to the existing
            // probe / full-recompute path unchanged — always correct.
            // Default-series resumes only read canonical input columns, so they can
            // skip building a non-computed base frame on the single-column append hot
            // path. Explicit series may reference stale computed columns, so those
            // still use the base-frame fallback below. (State borrowed off `self.inner`,
            // dropped before the tail write.)
            if directive_uses_default_series(&node) {
                let resumed = self.inner.computed_resume_state(&name).and_then(|(state, origin)| {
                    volas_directive::exec::execute_resume(&self.inner, &node, state, vr, origin)
                });
                if let Some((tail, new_state)) = resumed {
                    self.inner
                        .update_computed_tail(&name, vr, &tail)
                        .map_err(pyerr)?;
                    self.inner.set_computed_state(&name, Some(new_state));
                    continue;
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
                // Exclude computed columns by a direct membership test (FxHashMap
                // lookup) — no intermediate `HashSet` of cloned names.
                let real_names: Vec<String> = self
                    .inner
                    .names()
                    .iter()
                    .filter(|n| !self.inner.is_computed(n))
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
            let resumed = self.inner.computed_resume_state(&name).and_then(|(state, origin)| {
                volas_directive::exec::execute_resume(frame, &node, state, vr, origin)
            });
            if let Some((tail, new_state)) = resumed {
                self.inner
                    .update_computed_tail(&name, vr, &tail)
                    .map_err(pyerr)?;
                self.inner.set_computed_state(&name, Some(new_state));
                continue;
            }
            // A directive whose every command has finite memory and a position-free
            // output (SMA, ROC, price transforms, CDL, …) depends only on a fixed
            // trailing window, so recomputing a window is exact and O(lookback): the
            // window reaches `lb + 1` rows behind the first stale row (one more than the
            // lookback covers indicators that emit from a partial first difference) and
            // doubles that as margin. Anything recursive (EMA / Wilder / MACD / SAR /
            // cumulative OBV / HT) or position-valued (the index family) depends on the
            // whole prefix, so it recomputes from row 0 — O(n) but exact. The decision
            // is structural (`recomputes_on_a_window`), never inferred from the values.
            // (A slice that dropped its head only has the visible rows, so a stateful
            // indicator there cannot be continued past the missing history.)
            let win = 2 * (lb + 1);
            let (recomputed, off) =
                if vr > win && volas_directive::lookback::recomputes_on_a_window(&node) {
                    let start = vr - win;
                    // `slice_data` skips the per-cached-column ComputedMeta clone; the
                    // window is read-only and discarded, so no resume carry is needed.
                    let windowed =
                        volas_directive::exec::execute_refresh(&frame.slice_data(start, height), &node)
                            .map_err(value_err)?;
                    (windowed, vr - start)
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
            let (new_state, _) = self.capture_resume_states(frame, &node, &recomputed);
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

    /// Refresh one cached column on a time-frame frame. Returns `true` when it fully
    /// handled the column.
    ///
    /// Only **default-series** columns (the live-trading norm — `atr`, `ema`, `rsi`,
    /// …) are handled here; each carries two states, the state after the last row and
    /// the state before it. The common case — only the forming last row is stale —
    /// recomputes that row from the state before it and stores the advanced state as
    /// the state after it (a period rollover has already moved the previous state
    /// after into the state before, so the closed bar is never stepped twice). A cold /
    /// warm-up / multi-row-stale column recomputes the whole column and captures both
    /// states again. An explicit-`@series` column carries no state on a time-frame
    /// frame and falls through (`false`) to the full-recompute path.
    fn refresh_tf_column(
        &mut self,
        name: &str,
        lb: usize,
        vr: usize,
        height: usize,
    ) -> PyResult<bool> {
        let node = parse(name).map_err(directive_err)?;
        if !directive_uses_default_series(&node) {
            self.inner.set_computed_state(name, None);
            self.inner.set_computed_state_before_last(name, None);
            return Ok(false);
        }
        // Fast path: only the forming row is stale, it is PAST warm-up (`vr >= lb`), and
        // the state before it is known. The `vr >= lb` gate keeps the whole warm-up
        // region on the cold full recompute below: an indicator whose warm-up mask is
        // length-dependent (its short-frame output differs from the masked full-series
        // value) would otherwise freeze the incrementally-carried early rows and diverge
        // from a full recompute. The state is borrowed straight off `self.inner` and
        // dropped before each write.
        if vr + 1 == height && vr >= lb {
            // Recursive scalars FIRST (ema / smma / atr / natr / rsi / cmo): one step from
            // the state before the row yields its value and the state after it, both
            // written in place — no allocation.
            let scalar = self.inner.computed_state_before_last(name).and_then(|(state, _)| {
                volas_directive::exec::execute_resume_one_node(&self.inner, &node, state, vr)
            });
            if let Some(step) = scalar {
                self.inner
                    .update_computed_f64_value(name, vr, step.value)
                    .map_err(pyerr)?;
                self.inner.store_computed_state(name, step.state());
                return Ok(true);
            }
            // Stateless finite-memory (avgprice / medprice / mom / roc / …): the forming
            // row depends only on raw inputs — a parse-free scalar recompute; its carried
            // state is empty.
            if let Some(value) =
                volas_directive::exec::execute_resume_default_series_one(&self.inner, name, vr)
            {
                self.inner
                    .update_computed_f64_value(name, vr, value)
                    .map_err(pyerr)?;
                self.inner.store_computed_state(name, &[]);
                return Ok(true);
            }
            // Every other resume kernel: the forming row and the state after it.
            let resumed = self.inner.computed_state_before_last(name).and_then(|(state, origin)| {
                volas_directive::exec::execute_resume(&self.inner, &node, state, vr, origin)
            });
            if let Some((tail, new_state)) = resumed {
                self.inner
                    .update_computed_tail(name, vr, &tail)
                    .map_err(pyerr)?;
                self.inner.set_computed_state(name, Some(new_state));
                return Ok(true);
            }
        }
        // Cold / warm-up / multi-row-stale: recompute against the live frame (a
        // default-series command reads only raw OHLCV, never its own stale cache) and
        // rewrite the WHOLE column `[0, height)`, not just the stale tail. Rewriting from
        // row 0 re-applies the full-series warm-up mask to the early rows every cold run,
        // so the masked values are frozen correctly when the fast path takes over at the
        // warm boundary (`vr >= lb`). Then capture both states again.
        let recomputed =
            volas_directive::exec::execute_refresh(&self.inner, &node).map_err(value_err)?;
        self.inner
            .update_computed_tail(name, 0, &recomputed)
            .map_err(pyerr)?;
        let (after, before) = self.capture_resume_states(&self.inner, &node, &recomputed);
        self.inner.set_computed_state(name, after);
        self.inner.set_computed_state_before_last(name, before);
        Ok(true)
    }
}
