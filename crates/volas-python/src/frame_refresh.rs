//! The cached-directive refresh: recomputing the stale tail of materialized
//! directive columns after an append. Includes the anchor-preserving tf
//! forming-row resume (`refresh_forming_column`) — kept in its own module so the
//! refresh policy lives apart from the frame's construction / access surface.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::DataFrame;

#[allow(unused_imports)]
use crate::*;

impl PyDataFrame {
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
        // A tf-fold frame's last row is the OPEN forming bar (its raw OHLCV keeps
        // changing as fine bars fold in). For such a frame a recursive column's
        // carried state must stay anchored at the last CLOSED bar, never advance onto
        // the volatile forming row — so its refresh is routed through a dedicated,
        // anchor-preserving path rather than the standard append resume below.
        let forming = self.tf.as_ref().and_then(|t| t.open.as_ref()).is_some();
        let mut base: Option<DataFrame> = None;
        for (name, lb, vr) in stale {
            if forming && self.refresh_forming_column(&name, lb, vr, height)? {
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
                    // Recursive single-state single-row fast path (ema/smma): the new
                    // value IS the new state `[value]`, so write the value and update the
                    // state IN PLACE — no tail/state `Vec` allocation. Bit-identical to
                    // the `Vec` resume below (same shared `*_step` kernel). The state is
                    // borrowed straight off `self.inner` and dropped before the write.
                    let scalar = self.inner.computed_resume_state(&name).and_then(|(state, _)| {
                        volas_directive::exec::execute_resume_one(&self.inner, &name, state, vr)
                    });
                    if let Some(value) = scalar {
                        self.inner
                            .update_computed_f64_value(&name, vr, value)
                            .map_err(pyerr)?;
                        self.inner.update_computed_state_at(&name, 0, value);
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

    /// Anchor-preserving refresh of one cached column on a tf-fold (forming) frame.
    /// Returns `true` when it fully handled the column.
    ///
    /// Only **default-series** recursive columns (the live-trading norm — `atr`,
    /// `ema`, `rsi`, …) are anchored here: their carried state describes the last
    /// CLOSED bar (`height - 2`), never the volatile forming row. The common case —
    /// only the forming row (`height - 1`) is stale — resumes that single row from the
    /// anchor and writes its value WITHOUT advancing the state (the next fold re-forms
    /// the row; the period rollover finalises the anchor). A cold / multi-row-stale
    /// column recomputes its tail and re-derives the anchor as of the last closed bar.
    /// An explicit-`@series` column can't be cleanly anchored, so it drops any state
    /// and falls through (`false`) to the unchanged full path.
    fn refresh_forming_column(
        &mut self,
        name: &str,
        lb: usize,
        vr: usize,
        height: usize,
    ) -> PyResult<bool> {
        let node = parse(name).map_err(directive_err)?;
        if !directive_uses_default_series(&node) {
            // Can't anchor an explicit-series resume at the closed bar; keep the
            // pre-fold behaviour (full recompute, no carried state).
            self.inner.set_computed_state(name, None);
            return Ok(false);
        }
        // Fast path: only the open forming row is stale, it is PAST warm-up (`vr >= lb`),
        // and we hold an anchor — resume that single row and write its value, DISCARDING
        // the new state so the anchor stays at the last closed bar (`vr - 1`). The
        // `vr >= lb` gate keeps the whole warm-up region on the cold full-recompute below:
        // an indicator whose warm-up mask is length-dependent (its short-frame output
        // differs from the masked full-series value) would otherwise freeze the
        // incrementally-carried early rows and diverge from a full recompute. The anchor
        // state is borrowed straight off `self.inner` and dropped before each write.
        if vr + 1 == height && vr >= lb {
            // Stateless finite-memory (avgprice / medprice / mom / roc / …): the forming
            // row depends only on its own inputs, so a parse-free scalar recompute writes
            // the value with no carried state and no allocation.
            if let Some(value) =
                volas_directive::exec::execute_resume_default_series_one(&self.inner, name, vr)
            {
                self.inner
                    .update_computed_f64_value(name, vr, value)
                    .map_err(pyerr)?;
                return Ok(true);
            }
            // Single-state scalar resume (ema / smma / atr / natr / rsi / cmo …) — no
            // tail/state allocation.
            let scalar = self.inner.computed_resume_state(name).and_then(|(state, _)| {
                volas_directive::exec::execute_resume_one(&self.inner, name, state, vr)
            });
            if let Some(value) = scalar {
                self.inner
                    .update_computed_f64_value(name, vr, value)
                    .map_err(pyerr)?;
                return Ok(true);
            }
            // Every other recursive resume kernel with a scalar twin (the Wilder /
            // directional / cascade / volume / HT / … families): the single forming-row
            // value, allocation-free, discarding the new state to keep the anchor.
            let value = self.inner.computed_resume_state(name).and_then(|(state, origin)| {
                volas_directive::exec::execute_resume_value(&self.inner, &node, state, vr, origin)
            });
            if let Some(value) = value {
                self.inner
                    .update_computed_f64_value(name, vr, value)
                    .map_err(pyerr)?;
                return Ok(true);
            }
            // Remaining kernels without a scalar twin (index family, stochrsi `.d`):
            // the Vec-returning resume, discarding the new state.
            let resumed = self.inner.computed_resume_state(name).and_then(|(state, origin)| {
                volas_directive::exec::execute_resume(&self.inner, &node, state, vr, origin)
            });
            if let Some((tail, _new_state)) = resumed {
                self.inner
                    .update_computed_tail(name, vr, &tail)
                    .map_err(pyerr)?;
                return Ok(true);
            }
        }
        // Cold / warm-up / multi-row-stale: recompute against the live frame (a
        // default-series command reads only raw OHLCV, never its own stale cache) and
        // rewrite the WHOLE column `[0, height)`, not just the stale tail. Rewriting from
        // row 0 re-applies the full-series warm-up mask to the early rows every cold run,
        // so the masked values are frozen correctly when the fast path takes over at the
        // warm boundary (`vr >= lb`) — the carried early rows can no longer be a stale
        // short-frame (unmasked) output. Then re-anchor the state at the last CLOSED bar.
        let recomputed =
            volas_directive::exec::execute_refresh(&self.inner, &node).map_err(value_err)?;
        self.inner
            .update_computed_tail(name, 0, &recomputed)
            .map_err(pyerr)?;
        if height >= 1 {
            let anchor = self.inner.slice_data(0, height - 1);
            let st = volas_directive::exec::initial_state(&anchor, &node, &recomputed);
            self.inner.set_computed_state(name, st);
        }
        Ok(true)
    }
}
