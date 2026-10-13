//! The materialized-directive (computed) column cache lifecycle on a
//! [`DataFrame`]: recording a cached directive result, tracking staleness
//! after an append, incrementally refreshing its tail, and invalidating it on
//! a manual write. The cache (`DataFrame::computed`) is private frame state;
//! these methods own its invariants:
//! - rows `[0, valid_rows)` hold valid values, and only a contiguous extension of
//!   them (a write starting at or before `valid_rows`, ending inside the frame) moves
//!   the cursor;
//! - a carried state is resumed only from the row it describes (`state_at`), which is
//!   the last valid row;
//! - warm-up is counted on the column's processed history (`since` to the last valid
//!   row, absolute positions), never on a slice's local row numbers.

use std::sync::Arc;

use crate::column::Column;
use crate::error::{Result, VolasError};

use super::{Anchors, ComputedMeta, DataFrame, Directive, Reach, STALE_BY_APPEND, STALE_BY_WRITE};

/// A stale cached column as the refresh reads it: its name, bound directive, reach,
/// valid-row cursor, and whether its processed history is past the warm-up
/// ([`ComputedMeta::warm`]).
/// `Name` defaults to `String`; shared snapshots use `Arc<str>` without copying names.
#[derive(Clone, Debug)]
pub struct StaleComputed<Name = String> {
    /// The column (= its canonical directive), owned or shared by the snapshot.
    pub name: Name,
    /// The bound directive it computes.
    pub directive: Directive,
    /// The rows the column reads around its own.
    pub reach: Reach,
    /// Rows `[0, valid_rows)` hold valid values.
    pub valid_rows: usize,
    /// The absolute position of the row the column's history starts at
    /// ([`ComputedMeta::since`]): before this frame's row 0 when the frame dropped it.
    pub since: usize,
    /// More than `lookback` rows of the column's history are processed.
    pub warm: bool,
}

fn stale_snapshot<Name>(name: Name, meta: &ComputedMeta, origin: usize) -> StaleComputed<Name> {
    StaleComputed {
        name,
        directive: Arc::clone(&meta.directive),
        reach: meta.reach,
        valid_rows: meta.valid_rows,
        since: meta.since,
        warm: meta.warm(origin),
    }
}

impl DataFrame {
    /// Re-derive `first_computed` after the set of cached columns changed.
    pub(super) fn note_computed(&mut self) {
        self.first_computed = self.computed.keys().filter_map(|k| self.name_to_idx.get(k.as_ref()).copied()).min();
        self.retrim_lead();
    }

    /// Record that column `name` is a materialized result of `directive` (valid for all
    /// current rows), its history starting at this frame's row 0, reading `reach` rows
    /// around its own.
    pub fn set_computed(&mut self, name: &str, directive: Directive, reach: Reach) {
        self.computed.insert(
            Arc::from(name),
            ComputedMeta {
                directive,
                reach,
                valid_rows: self.height,
                state: None,
                state_before_last: None,
                since: self.origin,
                state_at: 0,
                scratch: None,
                checkpoint: None,
                pending: None,
                anchors: Anchors::default(),
                lost: None,
                first_present: None,
            },
        );
        let (origin, height) = (self.origin, self.height);
        if let (Some(pos), Some(meta)) = (self.name_to_idx.get(name), self.computed.get_mut(name)) {
            meta.note_written(&self.columns[*pos], origin, 0, height);
        }
        self.note_computed();
        #[cfg(debug_assertions)]
        self.debug_check();
    }

    /// Attach (or replace) both carried states of a cached column: `state` as of its
    /// last valid row and `before_last` as of the row before it (see [`ComputedMeta`]).
    /// A column with no valid row carries none; one with a single valid row no state
    /// before it. No-op if `name` is not a tracked computed column.
    pub fn set_computed_states(
        &mut self,
        name: &str,
        state: Option<Vec<f64>>,
        before_last: Option<Vec<f64>>,
    ) {
        let origin = self.origin;
        if let Some(meta) = self.computed.get_mut(name) {
            match meta.last_valid(origin) {
                Some(at) => {
                    meta.state_at = at;
                    meta.state_before_last = before_last.filter(|_| at > origin && state.is_some()).map(Arc::from);
                    meta.state = state.map(Arc::from);
                }
                None => {
                    meta.state = None;
                    meta.state_before_last = None;
                }
            }
        }
    }

    /// Resume a cached column by ONE row — the row after its last valid one: `step` gets
    /// this frame, the carried state (as of the last valid row), the frame's origin and
    /// a buffer of the state's length, and returns the new row's value having written the
    /// state as of that row into the buffer. On `Some`, the states advance (the old state
    /// becomes [`ComputedMeta::state_before_last`]); on `None` they are untouched. The
    /// caller writes the value. The buffers are recycled, so a live column steps with no
    /// allocation. `None` if `name` is untracked or carries no state of its last valid row.
    pub fn step_computed_state(
        &mut self,
        name: &str,
        step: impl FnOnce(&DataFrame, &[f64], usize, &mut [f64]) -> Option<f64>,
    ) -> Option<f64> {
        let origin = self.origin;
        let meta = self.computed.get_mut(name)?;
        if meta.last_valid(origin) != Some(meta.state_at) {
            return None;
        }
        let prev = meta.state.take()?;
        let mut next = match meta.scratch.take() {
            Some(next) if next.len() == prev.len() => next,
            // The sized slice and its Arc header occupy one allocation. A successful
            // step publishes every word, so recycled buffers need no clearing.
            _ => std::iter::repeat_n(0.0, prev.len()).collect::<Arc<[f64]>>(),
        };
        let out = match Arc::get_mut(&mut next) {
            Some(writable) => step(self, &prev, origin, writable),
            None => step(self, &prev, origin, Arc::make_mut(&mut next)),
        };
        let meta = self.computed.get_mut(name)?;
        if out.is_some() {
            meta.scratch = meta.state_before_last.take();
            meta.state_before_last = Some(prev);
            meta.state = Some(next);
            meta.state_at += 1;
        } else {
            meta.state = Some(prev);
            meta.scratch = Some(next);
        }
        out
    }

    /// The states after a resume of ONE new row produced `state` (the state as of that
    /// row, now the last valid one): the old state becomes the state before it. No-op
    /// if `name` is untracked.
    pub fn advance_computed_state(&mut self, name: &str, state: Vec<f64>) {
        let origin = self.origin;
        if let Some(meta) = self.computed.get_mut(name) {
            let at = meta.last_valid(origin);
            // the old state is the one before only if it described the row before
            let before = meta.state.replace(Arc::from(state));
            meta.state_before_last = before.filter(|_| at == Some(meta.state_at + 1));
            match at {
                Some(at) => meta.state_at = at,
                None => meta.state = None, // LCOV_EXCL_LINE (called after a written row)
            }
        }
    }

    /// Whether any materialized directive column is stale (its `valid_rows` lags
    /// `height` after an `append`), i.e. a bulk read would see NaN until `fulfill`.
    pub fn has_stale_computed(&self) -> bool {
        self.computed.values().any(|m| m.valid_rows < self.height)
    }

    /// What left a cached column stale — "an append" or "a column write" — or `None`
    /// while every cached column is fresh.
    pub fn stale_cause(&self) -> Option<&'static str> {
        self.has_stale_computed().then_some(self.stale_by)
    }

    /// Borrow a cached column's carried resume state and the frame's origin — the
    /// `(&[f64], usize)` the resume kernels read — **without cloning** the state
    /// `Vec`. Lets the live refresh / rollover-finalize loops continue the recursion
    /// straight off the stored state instead of snapshotting a per-column copy.
    /// `None` if `name` is untracked or carries no state of its last valid row.
    pub fn computed_resume_state(&self, name: &str) -> Option<(&[f64], usize)> {
        let meta = self.computed.get(name)?;
        if meta.last_valid(self.origin) != Some(meta.state_at) {
            return None;
        }
        Some((meta.state.as_deref()?, self.origin))
    }

    /// The absolute position of the row cached column `name`'s history starts at
    /// ([`ComputedMeta::since`]), if it is one.
    pub fn computed_since(&self, name: &str) -> Option<usize> {
        Some(self.computed.get(name)?.since)
    }

    /// The bound directive the cached column computes.
    pub fn computed_directive(&self, name: &str) -> Option<&Directive> {
        self.computed.get(name).map(|meta| &meta.directive)
    }

    /// The absolute position of cached column `name`'s first present value — where its
    /// warm-up ended — if one has been computed.
    pub fn computed_first_present(&self, name: &str) -> Option<usize> {
        self.computed.get(name)?.first_present
    }

    /// Snapshot of the materialized-directive columns (`name`, meta).
    pub fn computed_columns(&self) -> Vec<(String, ComputedMeta)> {
        self.computed
            .iter()
            .map(|(k, v)| (k.as_ref().to_owned(), v.clone()))
            .collect()
    }

    /// The stale materialized-directive columns ([`StaleComputed`]: the small cursor
    /// fields only, **never the carried `state` Vec**) — all of them, or just `only`.
    /// The live fulfill loop drives off this owned list (so it can take `&mut self` to
    /// write the refreshed tail) and borrows each column's state (and origin) on demand
    /// via [`computed_resume_state`](Self::computed_resume_state) just before resuming
    /// it, so no per-column `Vec<f64>` is deep-copied per fulfill.
    pub fn stale_computed_columns(&self, only: Option<&str>) -> Vec<StaleComputed> {
        self.stale_metadata(only)
            .map(|(name, meta)| stale_snapshot(name.as_ref().to_owned(), meta, self.origin))
            .collect()
    }

    /// An owned stale-column snapshot with shared names. A single column needs no
    /// allocation: the first item is inline and only further items use a Vec. Like
    /// `stale_computed_columns`, it holds no frame borrow or carried state, so the
    /// frame may be refreshed while consuming it.
    pub fn shared_stale_computed_columns(&self, only: Option<&str>) -> impl Iterator<Item = StaleComputed<Arc<str>>> + use<> {
        let mut items = self.stale_metadata(only)
            .map(|(name, meta)| stale_snapshot(Arc::clone(name), meta, self.origin));
        let first = items.next();
        let rest: Vec<_> = items.collect();
        first.into_iter().chain(rest)
    }

    fn stale_metadata<'a>(&'a self, only: Option<&'a str>) -> impl Iterator<Item = (&'a Arc<str>, &'a ComputedMeta)> {
        self.computed.iter().filter(move |(name, meta)| {
            meta.valid_rows < self.height && only.is_none_or(|target| target == name.as_ref())
        })
    }

    /// Write rows `[from, from + len)` of the computed column `name` through `write`,
    /// then mark it valid through the last written row. The rows must lie inside the
    /// frame and, for a tracked column, start at or before its valid rows (a write past
    /// them would leave a gap of stale rows marked valid) — anything else is an error and
    /// nothing is written. A write from row 0 recomputed the column from this frame's own
    /// rows, so its history starts here. One lookup of each map (the live refresh path).
    fn write_computed(
        &mut self,
        name: &str,
        from: usize,
        len: usize,
        write: impl FnOnce(&mut Column) -> Result<()>,
    ) -> Result<()> {
        let pos = self
            .name_to_idx
            .get(name)
            .copied()
            .ok_or_else(|| VolasError::ColumnNotFound(name.to_string()))?;
        let end = from.checked_add(len).filter(|&end| end <= self.height).ok_or_else(|| {
            VolasError::Shape(format!(
                "cannot write {len} computed rows of \"{name}\" from row {from}: the frame \
                 has {} rows",
                self.height
            ))
        })?;
        let meta = self.computed.get_mut(name);
        if let Some(meta) = &meta {
            if from > meta.valid_rows {
                return Err(VolasError::Shape(format!(
                    "cannot write computed rows of \"{name}\" from row {from}: rows {} to \
                     {from} are not valid yet",
                    meta.valid_rows
                )));
            }
        }
        write(&mut self.columns[pos])?;
        if let Some(meta) = meta {
            meta.note_written(&self.columns[pos], self.origin, from, end);
            meta.valid_rows = meta.valid_rows.max(end);
            if from == 0 {
                meta.since = self.origin;
            }
        }
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }

    /// Overwrite a computed column's rows `[from, from + tail.len())` in place
    /// (copy-on-write) with the recomputed `tail`, and mark it valid through the last
    /// written row, under the rule of `write_computed`. A directive result of any dtype
    /// is handled ([`Column::write_rows`]), each row taking the tail's value and
    /// presence (a missing truth stays missing). O(tail.len()).
    pub fn update_computed_tail(&mut self, name: &str, from: usize, tail: &Column) -> Result<()> {
        self.write_computed(name, from, tail.len(), |col| {
            let dtype = col.dtype();
            col.write_rows(from, tail).map_err(|_| {
                VolasError::DType(format!(
                    "computed tail dtype {} does not match column \"{name}\" dtype {dtype}",
                    tail.dtype()
                ))
            })
        })
    }

    /// Overwrite one numeric or boolean computed value and mark the column valid through that row,
    /// under the same rule as [`update_computed_tail`](Self::update_computed_tail).
    /// This avoids allocating a one-value tail column on the single-bar append path.
    pub fn update_computed_value(&mut self, name: &str, row: usize, value: f64) -> Result<()> {
        self.write_computed(name, row, 1, |col| match col {
            Column::F64(buf) => {
                buf.set(row, value);
                Ok(())
            }
            Column::Bool(buf, valid) => {
                valid.set(row, buf.len(), !value.is_nan());
                buf.make_mut()[row] = value != 0.0;
                Ok(())
            }
            col => Err(VolasError::DType(format!(
                "computed scalar dtype F64 does not match column \"{name}\" dtype {}",
                col.dtype()
            ))),
        })
    }

    /// Drop the computed (cached-directive) status of the column at `col`, if any.
    fn drop_computed_at(&mut self, col: usize) {
        if let Some(name) = self.names.get(col) {
            self.computed.remove(name.as_str());
            self.note_computed();
        }
    }

    /// The last row changed in place (a time-frame forming bar folded or replaced, or a
    /// row appended with the last row's label): mark each cached column stale from
    /// `last_row` on but **keep** what it carries. A column that was valid through that
    /// row drops back by one: its state as of the row before (`state_before_last`)
    /// becomes its state, so the refresh recomputes just that row from it (O(1) for a
    /// recursive kernel). The dropped buffer is kept for the next step. Unlike
    /// [`invalidate_computed_on_write_at`] this never zeroes `valid_rows`, so the live
    /// path never forces a full O(buffer) recompute.
    pub(super) fn invalidate_computed_forming_row(&mut self, last_row: usize) {
        self.stale_by = STALE_BY_APPEND;
        for meta in self.computed.values_mut() {
            if meta.pending.as_ref().is_some_and(|p| p.at >= self.origin + last_row) {
                meta.pending = None; // it covers the replaced row
            }
            if meta.valid_rows > last_row {
                meta.valid_rows = last_row;
                let replaced = std::mem::replace(&mut meta.state, meta.state_before_last.take());
                if let Some(buf) = replaced {
                    meta.scratch = Some(buf);
                    meta.state_at = meta.state_at.saturating_sub(1);
                }
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
        self.stale_every_computed();
    }

    /// Name-based variant for the `df[name] = value` whole-column replace path.
    pub fn invalidate_computed_on_write(&mut self, written_name: &str) {
        self.computed.remove(written_name);
        self.note_computed();
        self.stale_every_computed();
    }

    /// Every cached column is stale from row 0, carrying nothing.
    fn stale_every_computed(&mut self) {
        self.stale_by = STALE_BY_WRITE;
        for meta in self.computed.values_mut() {
            meta.valid_rows = 0;
            // The carried recursive state described row `valid_rows - 1`; resetting
            // `valid_rows` to 0 breaks that correspondence, so drop it. A later refresh
            // recomputes from scratch (correct) and repopulates the state.
            meta.state = None;
            meta.state_before_last = None;
            meta.pending = None;
            meta.anchors = Anchors::default(); // the history's inputs changed
        }
    }
}
