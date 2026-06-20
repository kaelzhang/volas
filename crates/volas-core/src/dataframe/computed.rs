//! The materialized-directive (computed) column cache lifecycle on a
//! [`DataFrame`]: recording a cached directive result, tracking staleness
//! after an append, incrementally refreshing its tail, and invalidating it on
//! a manual write. The cache (`DataFrame::computed`) is private frame state;
//! these methods own its invariants.


use crate::column::Column;
use crate::error::{Result, VolasError};

use super::{ComputedMeta, DataFrame};

impl DataFrame {
    /// Record that column `name` is a materialized directive result (valid for
    /// all current rows).
    pub fn set_computed(&mut self, name: &str, lookback: usize) {
        self.computed.insert(
            name.to_string(),
            ComputedMeta {
                lookback,
                valid_rows: self.height,
                state: None,
                origin: 0,
            },
        );
    }

    /// Attach (or replace) the carried recursive [`ComputedMeta::state`] for a
    /// cached column, enabling an O(new-rows) append resume. No-op if `name` is
    /// not a tracked computed column.
    pub fn set_computed_state(&mut self, name: &str, state: Option<Vec<f64>>) {
        if let Some(meta) = self.computed.get_mut(name) {
            meta.state = state;
        }
    }

    /// Overwrite element `idx` of a cached column's carried state **in place** (no
    /// allocation) — the single-row resume fast path, where the new state reuses the
    /// existing fixed-size buffer. No-op if `name` is not computed, has no state, or
    /// `idx` is out of range.
    pub fn update_computed_state_at(&mut self, name: &str, idx: usize, val: f64) {
        if let Some(slot) = self
            .computed
            .get_mut(name)
            .and_then(|m| m.state.as_mut())
            .and_then(|s| s.get_mut(idx))
        {
            *slot = val;
        }
    }

    /// Whether any materialized directive column is stale (its `valid_rows` lags
    /// `height` after an `append`), i.e. a bulk read would see NaN until `fulfill`.
    pub fn has_stale_computed(&self) -> bool {
        self.computed.values().any(|m| m.valid_rows < self.height)
    }

    /// Borrow a cached column's carried resume state and its origin offset — the
    /// `(&[f64], usize)` the resume kernels read — **without cloning** the state
    /// `Vec`. Lets the live refresh / rollover-finalize loops continue the recursion
    /// straight off the stored state instead of snapshotting a per-column copy.
    /// `None` if `name` is untracked or carries no state.
    pub fn computed_resume_state(&self, name: &str) -> Option<(&[f64], usize)> {
        let meta = self.computed.get(name)?;
        Some((meta.state.as_deref()?, meta.origin))
    }

    /// Snapshot of the materialized-directive columns (`name`, meta).
    pub fn computed_columns(&self) -> Vec<(String, ComputedMeta)> {
        self.computed
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Stale materialized-directive columns as `(name, lookback, valid_rows)` — the
    /// small `Copy` cursor fields only, **never the carried `state` Vec**. The live
    /// fulfill loop drives off this owned name list (so it can take `&mut self` to
    /// write the refreshed tail) and borrows each column's state (and origin) on
    /// demand via [`computed_resume_state`](Self::computed_resume_state) just before
    /// resuming it, so no per-column `Vec<f64>` is deep-copied per fulfill.
    pub fn stale_computed_columns(&self, only: Option<&str>) -> Vec<(String, usize, usize)> {
        self.computed
            .iter()
            .filter(|(name, meta)| {
                meta.valid_rows < self.height && only.is_none_or(|target| target == name.as_str())
            })
            .map(|(name, meta)| (name.clone(), meta.lookback, meta.valid_rows))
            .collect()
    }

    /// Names of all materialized-directive columns.
    pub fn computed_names(&self) -> Vec<String> {
        self.computed.keys().cloned().collect()
    }

    /// Overwrite a computed column's rows `[from, from + tail.len())` in place
    /// (copy-on-write) with the recomputed `tail`, and mark it valid up to the
    /// full height. Directive results are F64 or Bool, so both are handled.
    /// O(tail.len()).
    pub fn update_computed_tail(&mut self, name: &str, from: usize, tail: &Column) -> Result<()> {
        let pos = self
            .name_to_idx
            .get(name)
            .copied()
            .ok_or_else(|| VolasError::ColumnNotFound(name.to_string()))?;
        match (&mut self.columns[pos], tail) {
            (Column::F64(arc), Column::F64(t)) => {
                let buf = arc.make_mut();
                for (i, &v) in t.iter().enumerate() {
                    if from + i < buf.len() {
                        buf[from + i] = v;
                    }
                }
            }
            (Column::Bool(arc, _), Column::Bool(t, _)) => {
                let buf = arc.make_mut();
                for (i, &v) in t.iter().enumerate() {
                    if from + i < buf.len() {
                        buf[from + i] = v;
                    }
                }
            }
            (col, t) => {
                return Err(VolasError::DType(format!(
                    "computed tail dtype {} does not match column \"{name}\" dtype {}",
                    t.dtype(),
                    col.dtype()
                )))
            }
        }
        if let Some(meta) = self.computed.get_mut(name) {
            meta.valid_rows = self.height;
        }
        Ok(())
    }

    /// Overwrite one F64 computed value and mark the computed column current.
    /// This avoids allocating a one-value tail column on the single-bar append path.
    pub fn update_computed_f64_value(&mut self, name: &str, row: usize, value: f64) -> Result<()> {
        let pos = self
            .name_to_idx
            .get(name)
            .copied()
            .ok_or_else(|| VolasError::ColumnNotFound(name.to_string()))?;
        match &mut self.columns[pos] {
            Column::F64(arc) => {
                let buf = arc.make_mut();
                if row < buf.len() {
                    buf[row] = value;
                }
            }
            col => {
                return Err(VolasError::DType(format!(
                    "computed scalar dtype F64 does not match column \"{name}\" dtype {}",
                    col.dtype()
                )))
            }
        }
        if let Some(meta) = self.computed.get_mut(name) {
            meta.valid_rows = self.height;
        }
        Ok(())
    }

    /// Drop the computed (cached-directive) status of the column at `col`, if any.
    fn drop_computed_at(&mut self, col: usize) {
        if let Some(name) = self.names.get(col) {
            self.computed.remove(name);
        }
    }

    /// Live tf-fold invalidation: only the **forming row** (the last, still-open
    /// period bar at `forming_row`) changed, so mark each cached column stale from
    /// `forming_row` on but **keep** its carried recursive state. The state stays
    /// anchored at the last CLOSED bar (`forming_row - 1`); `refresh_computed`
    /// resumes just the forming row from it (O(lookback)) without advancing the
    /// anchor, and the period rollover advances the anchor over the now-closed bar.
    /// Unlike [`invalidate_computed_on_write_at`] this neither drops the state nor
    /// zeroes `valid_rows`, so the live fold never forces a full O(buffer) recompute.
    pub(super) fn invalidate_computed_forming_row(&mut self, forming_row: usize) {
        for meta in self.computed.values_mut() {
            if meta.valid_rows > forming_row {
                meta.valid_rows = forming_row;
            }
        }
    }

    /// A user write to column `col` invalidates the directive cache: `col` loses any
    /// computed status, and every OTHER cached directive column is marked fully stale —
    /// it may have been derived from `col`, so it is recomputed on next access (a bulk
    /// read raises until `fulfill`, exactly like an append). Conservative (no per-column
    /// dependency tracking), but writes are rare relative to reads.
    pub(super) fn invalidate_computed_on_write_at(&mut self, col: usize) {
        self.drop_computed_at(col);
        for meta in self.computed.values_mut() {
            meta.valid_rows = 0;
            // The carried recursive state described row `valid_rows - 1`; resetting
            // `valid_rows` to 0 breaks that correspondence, so drop it. A later refresh
            // recomputes from scratch (correct) and repopulates the state.
            meta.state = None;
        }
    }

    /// Name-based variant for the `df[name] = value` whole-column replace path.
    pub fn invalidate_computed_on_write(&mut self, written_name: &str) {
        self.computed.remove(written_name);
        for meta in self.computed.values_mut() {
            meta.valid_rows = 0;
            meta.state = None;
        }
    }
}
