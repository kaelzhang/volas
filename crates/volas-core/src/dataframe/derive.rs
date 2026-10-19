//! The one owner of what a frame derived from another keeps of it. Every derivation —
//! a row selection, a column projection, a relabelling, a cast, a time-zone change —
//! is one of these methods; the binding only maps its arguments onto them.
//!
//! - A **contiguous row selection** (the rows `[start, end)` in order, however it is
//!   written) continues the source's series: its row 0 keeps its absolute position
//!   ([`DataFrame::origin`]) and every cached directive column stays cached, with its
//!   values and with what continuing it needs — the carried states of the source's last
//!   valid row when the selection ends there (else none: a continuation computes them
//!   from what the selection keeps), the nearest checkpoint of a recursion's history
//!   before its first row (one the source keeps, one of its anchors, or the history's
//!   start — taken, not replayed, when it is within `ANCHOR_SPACING` rows; replayed
//!   through [`Replay`] otherwise), and the **lead**: the data rows before row 0 that
//!   recomputing its first rows reads, back to that checkpoint. Every recursion carries
//!   continuation state; the directive layer owns that state's representation.
//! - A **column projection** keeps the rows and the listed columns, each as data: a
//!   listed cached column's values, never continued (R40). A **column removal** keeps
//!   every other column as it is, cached ones continuing.
//! - Every **other derivation** — rows not contiguous in order, a cast, a rename, a new
//!   index, a time-zone change — starts a **new series**: its row 0 at position 0 and its
//!   data columns only. A cached column is dropped; requested again, it is computed from
//!   the new frame's own rows.
//!
//! A selection of rows by positions decides between the two structurally, on the
//! positions alone ([`DataFrame::select_rows`]).

use std::collections::HashMap;
use std::sync::Arc;

use crate::column::Column;
use crate::error::{Result, VolasError};
use crate::fxhash::FxHashMap;
use crate::index::{Index, IndexKind};

use super::{Checkpoint, CheckpointRef, ComputedMeta, DataFrame, History, Reach, ANCHOR_SPACING, STALE_BY_APPEND};

/// The directive layer's half of a contiguous selection: the history of a recursive
/// cached column carried by its state at a row the source frame keeps no point of it
/// near. The core knows where a column's history is; only the directive layer can replay
/// it.
pub trait Replay {
    /// For the recursive cached column `name` of `frame` (carried by its state): the
    /// checkpoint of its history as of the row before `frame`'s row `start` — replayed
    /// from the history `frame` holds of the column (the nearest anchor, its checkpoint
    /// or the history's start), reading the rows in place. `None` when it cannot be
    /// replayed.
    fn boundary(&self, frame: &DataFrame, name: &str, start: usize) -> Option<Checkpoint>;

    /// The continuation states as of rows `end - 1` and `end - 2`, lazily recovered
    /// from this cached column's history. The directive layer owns their format.
    fn continuation(&self, frame: &DataFrame, name: &str, end: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)>;
}

/// A replay that never has the history (core tests: a selection that needs it keeps the
/// column without a checkpoint).
#[cfg(test)]
pub(super) struct NoReplay;

#[cfg(test)]
impl Replay for NoReplay {
    fn continuation(&self, _: &DataFrame, _: &str, _: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> {
        None
    }

    fn boundary(&self, _: &DataFrame, _: &str, _: usize) -> Option<Checkpoint> {
        None
    }
}

impl Reach {
    /// The rows before a row that recomputing it — or resuming the column into it —
    /// reads at most: what a frame must hold, in its own rows or its lead, before its
    /// first row for that row to be recomputed exactly. A resume reads one row past the
    /// lookback of its command (the row its window drops), so at least `lookback + 1`.
    pub fn lead(&self) -> usize {
        self.continuation.max(self.resume).max(self.lookback + 1)
    }
}

impl DataFrame {
    /// Build a frame that **shares this frame's schema** (names + lookup, both
    /// `Arc`-cloned) over freshly derived `columns` / `index`, with no cached column and
    /// no lead — the shape every derivation starts from. Row 0 of the result is at
    /// absolute position `origin`.
    pub(super) fn same_schema(&self, columns: Vec<Column>, index: Index, origin: usize) -> DataFrame {
        let height = columns.first().map_or(index.len(), |c| c.len()); // no columns: the index
        DataFrame {
            names: Arc::clone(&self.names),
            name_to_idx: Arc::clone(&self.name_to_idx),
            columns,
            index: Arc::new(index),
            height,
            computed: FxHashMap::default(),
            first_computed: None,
            origin,
            stale_by: STALE_BY_APPEND,
            lead: None,
        }
    }

    /// The data columns (every column but the cached directive ones), by name, in order.
    pub fn data_names(&self) -> Vec<String> {
        self.names.iter().filter(|n| !self.computed.contains_key(n.as_str())).cloned().collect()
    }

    /// A new series of these rows: the data columns of `self` as they are (taken with
    /// `columns`, one per name in `names`), over `index`, its row 0 at position 0.
    fn new_series(names: Vec<String>, columns: Vec<Column>, index: Index) -> Result<DataFrame> {
        DataFrame::new(names, columns, Some(index))
    }

    /// The data columns' names and values.
    fn data_parts(&self) -> (Vec<String>, Vec<Column>) {
        let names = self.data_names();
        let columns = names.iter().map(|n| self.columns[self.name_to_idx[n]].clone()).collect();
        (names, columns)
    }

    // --- contiguous row selections ----------------------------------------------------

    /// The rows `[start, end)` as a continuation of this frame's series (see the module
    /// docs): every cached column stays cached; a recursion's end states are recovered
    /// lazily through `Replay::continuation` when the column first continues.
    ///
    /// The rows are a **value copy**, not a view into this frame's buffers: a selection is
    /// an independent frame, so selecting the recent tail of a long history does not pin
    /// the whole history alive. What it keeps of the rows before it is bounded by its
    /// cached columns' reach (the lead).
    pub fn rows(&self, start: usize, end: usize, replay: &dyn Replay) -> DataFrame {
        let start = start.min(self.height);
        let end = end.max(start).min(self.height);
        let mut computed = FxHashMap::with_capacity_and_hasher(self.computed.len(), Default::default());
        for (name, meta) in &self.computed {
            computed.insert(name.clone(), self.continued(name, meta, start, end, replay));
        }
        let mut copied_lead = None;
        let columns: Vec<Column> = if end < self.height && end > start && end - start <= ANCHOR_SPACING
            && self.width() > 1 && self.columns.iter().all(|c| matches!(c, Column::F64(_)))
        {
            // Short interior slices keep no source allocation alive. Pool their
            // copied values; a write materializes only the affected column's range.
            let rows = end - start;
            let lead_rows = self.copied_slab_lead_rows(start, &computed);
            let raw = self.width() - computed.len();
            if lead_rows > 0 && self.first_computed == Some(raw) {
                let (columns, lead) = self.copy_float_prefix_rows(start, end, raw, lead_rows);
                copied_lead = Some(lead);
                columns
            } else {
                let mut values = Vec::with_capacity(rows * self.width() + lead_rows * (self.width() - computed.len()));
                for col in &self.columns {
                    values.extend_from_slice(&col.as_f64().expect("checked floats")[start..end]);
                }
                let (owner, lead) = self.finish_copied_slab_lead(start, &computed, lead_rows, values);
                copied_lead = lead;
                self.columns.iter().enumerate().map(|(i, col)| {
                    let range = i * rows..(i + 1) * rows;
                    let buffer = crate::buffer::Buffer::from_shared_known_range(
                        Arc::clone(&owner), range, col.known_has_nan() == Some(false),
                        col.known_has_nonfinite() == Some(false),
                    ).expect("copied slab range");
                    Column::F64(buffer)
                }).collect()
            }
        } else {
            self.columns.iter().map(|c| c.slice(start, end)).collect()
        };
        let mut df = self.same_schema(columns, self.index.slice(start, end), self.origin + start);
        df.stale_by = self.stale_by;
        df.computed = computed;
        df.lead = copied_lead.or_else(|| self.lead_before(start, &df.computed));
        // Same schema and same cached names: preserve the memo directly. The lead
        // was already built to exactly the wanted reach, so no second trim is needed.
        df.first_computed = self.first_computed;
        #[cfg(debug_assertions)]
        df.debug_check();
        df
    }

    /// The last `rows` rows, as a bounded window keeps them when it drops its older ones:
    /// the contiguous selection ending at the last row (which never replays — every
    /// column's states describe its last valid row, which it keeps).
    pub fn retain_tail(&self, rows: usize, replay: &dyn Replay) -> DataFrame {
        self.rows(self.height.saturating_sub(rows), self.height, replay)
    }

    /// Cached column `name` (`meta`) as the selection `[start, end)` of this frame keeps
    /// it: its valid rows, its states as of its last valid row there (this frame's, when
    /// it is the same row), and the nearest point of its history before the rows it holds.
    fn continued(&self, name: &str, meta: &ComputedMeta, start: usize, end: usize, replay: &dyn Replay) -> ComputedMeta {
        let first = self.origin + start;
        let valid = meta.valid_rows.saturating_sub(start).min(end - start);
        let last = first + valid; // one past the selection's last valid row
        let own = meta.state.is_some()
            && meta.last_valid(self.origin) == Some(meta.state_at)
            && meta.state_at + 1 == last
            && valid > 0;
        // a checkpoint is replayed from over the rows after it: kept when the selection
        // holds them (its rows, whatever its output cursor says)
        let covers = |at: usize| (first..=first + (end - start)).contains(&(at + 1));
        let kept = meta.checkpoint.as_ref().filter(|cp| covers(cp.position())).cloned();
        // a recursion carried by its state whose history starts before the selection: what
        // every recompute of its rows replays from — the nearest point of its history
        // before the selection (a checkpoint of this frame's, one of its anchors, or the
        // history's start), kept with the rows after it in the lead, when it is within
        // `ANCHOR_SPACING` rows; else its checkpoint at the selection's boundary, replayed
        // now (forming the anchors it passes)
        let carried = matches!(meta.reach.history, History::Carried { .. }) && meta.since < first;
        let lazy = match carried && kept.is_none() {
            true => self.nearest_history(meta, first),
            false => None,
        };
        let boundary = match carried && kept.is_none() && lazy.is_none() {
            true => replay.boundary(self, name, start).map(CheckpointRef::from),
            false => None,
        };
        // the states as of its last valid row: this frame's when it is the same row; else
        // none — a continuation of the column computes them from what the selection keeps
        let (state, before) = match own {
            true => (meta.state.clone(), meta.state_before_last.clone()),
            false => (None, None),
        };
        // the state before describes the row before the last, which must be held (the
        // boundary checkpoint stands for the row before the first)
        let before = before.filter(|_| state.is_some() && last - 1 > first);
        ComputedMeta {
            directive: Arc::clone(&meta.directive),
            reach: meta.reach,
            valid_rows: valid,
            state,
            state_before_last: before,
            since: meta.since,
            state_at: last.saturating_sub(1),
            scratch: None,
            checkpoint: kept.or(lazy.flatten()).or(boundary),
            // the anchors of its history before its last row, where it agrees with this frame's
            anchors: meta.anchors.upto(last.saturating_sub(1)),
            pending: meta.pending.as_ref().filter(|cp| covers(cp.at)).cloned(),
            lost: meta.lost.clone(),
            first_present: meta.first_present,
        }
    }

    /// The nearest point of recursion `meta`'s history before absolute row `first` that a
    /// replay can start from over rows this frame holds (its own or its lead's), within
    /// `ANCHOR_SPACING` rows of `first`: `Some(Some(checkpoint))` — this frame's
    /// checkpoint or the nearest of its anchors, the later — or `Some(None)`, the
    /// history's start, when it has no checkpoint before `first`. `None` when the
    /// nearest is farther (or not held).
    fn nearest_history(&self, meta: &ComputedMeta, first: usize) -> Option<Option<CheckpointRef>> {
        let held = self.origin - self.lead_rows(); // the first row this frame holds
        let near = |from: usize| from >= held && first - from <= ANCHOR_SPACING;
        let own = meta.checkpoint.clone().filter(|cp| cp.position() < first);
        let nearest = match (own, meta.anchors.nearest(first - 1)) {
            (Some(a), Some(b)) => Some(if a.position() >= b.position() { a } else { b }),
            (a, b) => a.or(b),
        };
        match nearest {
            Some(cp) => near(cp.position() + 1).then_some(Some(cp)),
            None => near(meta.since).then_some(None),
        }
    }

    /// The rows `[start, end)` to READ (a reader's view of a frame's visible rows): their
    /// values, each cached column marked as one over its valid rows, and nothing that
    /// continues it — no state, checkpoint or lead. It is never appended to and never
    /// derived from: a derivation maps its rows onto this frame instead
    /// ([`rows`](Self::rows)).
    pub fn view_rows(&self, start: usize, end: usize) -> DataFrame {
        let start = start.min(self.height);
        let end = end.max(start).min(self.height);
        let columns: Vec<Column> = self.columns.iter().map(|c| c.slice(start, end)).collect();
        let mut df = self.same_schema(columns, self.index.slice(start, end), self.origin + start);
        df.stale_by = self.stale_by;
        df.computed.reserve(self.computed.len());
        for (name, meta) in &self.computed {
            let valid = meta.valid_rows.saturating_sub(start).min(end - start);
            let mut view = ComputedMeta { valid_rows: valid, ..meta.view() };
            view.state_at = (self.origin + start + valid).saturating_sub(1);
            df.computed.insert(name.clone(), view);
        }
        df.first_computed = self.first_computed;
        df
    }

    /// A `[start, end)` row slice that does **not** carry the cached-directive metadata —
    /// for a READ-ONLY derivation that is never appended to (a refresh probe, a
    /// row-select feeding a computation). The result's computed columns are plain data —
    /// correct values, but it MUST NOT be appended to (use [`rows`](Self::rows) for
    /// anything that continues).
    pub fn slice_data(&self, start: usize, end: usize) -> DataFrame {
        let start = start.min(self.height);
        let end = end.max(start).min(self.height);
        let columns: Vec<Column> = self.columns.iter().map(|c| c.slice(start, end)).collect();
        self.same_schema(columns, self.index.slice(start, end), self.origin + start)
    }

    // --- selections of rows by position --------------------------------------------------

    /// The rows at `positions`, decided structurally on the positions alone: rows that are
    /// contiguous and in order are the contiguous selection ([`rows`](Self::rows)); any
    /// other selection — rows out of order, repeated or with gaps, or none — is a new
    /// series of the data columns ([`take_data`](Self::take_data)).
    pub fn select_rows(&self, positions: &[usize], replay: &dyn Replay) -> DataFrame {
        match positions {
            [first, ..] if positions.windows(2).all(|w| w[1] == w[0] + 1) => {
                self.rows(*first, first + positions.len(), replay)
            }
            _ => self.take_data(positions),
        }
    }

    /// The rows selected by a boolean `mask` ([`select_rows`](Self::select_rows)).
    pub fn filter_mask(&self, mask: &[bool], replay: &dyn Replay) -> Result<DataFrame> {
        if mask.len() != self.height {
            return Err(VolasError::Shape(format!(
                "boolean mask length {} != frame height {}",
                mask.len(),
                self.height
            )));
        }
        let positions: Vec<usize> = mask.iter().enumerate().filter_map(|(i, &b)| b.then_some(i)).collect();
        Ok(self.select_rows(&positions, replay))
    }

    /// The rows at `positions` as a new series: the data columns only, row 0 at position
    /// 0.
    pub fn take_data(&self, positions: &[usize]) -> DataFrame {
        let (names, columns) = self.data_parts();
        let columns = columns.iter().map(|c| c.take(positions)).collect();
        DataFrame::new_series(names, columns, self.index.take(positions)).expect("taken alike")
    }

    /// Every column's values at `positions` (cached ones as values), row 0 at position 0
    /// — a read of the rows (a row, a column's values), never a frame that continues: it
    /// carries no cached column.
    pub fn take(&self, positions: &[usize]) -> DataFrame {
        let columns: Vec<Column> = self.columns.iter().map(|c| c.take(positions)).collect();
        self.same_schema(columns, self.index.take(positions), 0)
    }

    // --- column projections ----------------------------------------------------------------

    /// The columns `names`, in that order, over the same rows, each a data column: a
    /// listed cached column keeps its values as this frame shows them and is plain data
    /// from then on — never continued, even with its inputs alongside (R40); an unlisted
    /// one is dropped. This internal same-series data view retains computation
    /// coordinates; use `project` for a public new-series projection.
    pub fn select(&self, names: &[String]) -> Result<DataFrame> {
        let mut columns = Vec::with_capacity(names.len());
        let mut name_to_idx = FxHashMap::with_capacity_and_hasher(names.len(), Default::default());
        for (i, n) in names.iter().enumerate() {
            columns.push(self.column(n)?.clone());
            if name_to_idx.insert(n.clone(), i).is_some() {
                return Err(VolasError::Value(format!("duplicate column name \"{n}\"")));
            }
        }
        // the rows are the same: the index is shared, not copied
        let df = DataFrame {
            names: Arc::new(names.to_vec()),
            columns,
            name_to_idx: Arc::new(name_to_idx),
            index: Arc::clone(&self.index),
            height: self.height,
            computed: FxHashMap::default(),
            first_computed: None,
            origin: self.origin,
            stale_by: self.stale_by,
            lead: None,
        };
        #[cfg(debug_assertions)]
        df.debug_check();
        Ok(df)
    }

    /// A public column projection of rows `[start, end)`: the listed values are
    /// ordinary data in a new series, with their own computation coordinates.
    /// Internal same-series data views use `select` and keep their origin.
    pub fn project(&self, names: &[String], start: usize, end: usize) -> Result<DataFrame> {
        let start = start.min(self.height);
        let end = end.max(start).min(self.height);
        let mut projected = self.select(names)?;
        if start != 0 || end != self.height {
            projected.columns = projected.columns.iter().map(|c| c.slice(start, end)).collect();
            projected.index = Arc::new(self.index.slice(start, end));
            projected.height = end - start;
        }
        projected.origin = 0;
        projected.stale_by = STALE_BY_APPEND;
        #[cfg(debug_assertions)]
        projected.debug_check();
        Ok(projected)
    }

    /// This frame without the columns `names` (a name that is not a column is ignored):
    /// every other column as it is, over the same rows — a cached one still cached and
    /// continuing, with its states, checkpoint and anchors — and the lead trimmed to what
    /// the cached columns left read (a column removal, not a projection: see
    /// [`select`](Self::select)).
    pub fn drop_columns(&self, names: &[String]) -> Result<DataFrame> {
        let keep: Vec<String> = self.names.iter().filter(|n| !names.contains(n)).cloned().collect();
        let mut df = self.select(&keep)?;
        df.computed = keep.iter().filter_map(|n| self.computed.get_key_value(n.as_str()).map(|(key, m)| (Arc::clone(key), m.clone()))).collect();
        df.note_computed();
        if !df.computed.is_empty() {
            df.lead = self.lead.as_ref().map(|l| l.over(&df));
            df.retrim_lead();
        }
        #[cfg(debug_assertions)]
        df.debug_check();
        Ok(df)
    }

    // --- new series ------------------------------------------------------------------------

    /// The rename `mapping` takes effect as: its entries whose source is a data column of
    /// this frame (pandas ignores a name that is not a column; a cached column is not
    /// carried by a rename). The one resolution every part of a renamed frame follows.
    pub fn renaming(&self, mapping: &HashMap<String, String>) -> HashMap<String, String> {
        mapping
            .iter()
            .filter(|(from, _)| self.has_column(from) && !self.is_computed(from))
            .map(|(from, to)| (from.clone(), to.clone()))
            .collect()
    }

    /// Rename columns (pandas `rename(columns=...)`) — a new series of the data columns,
    /// renamed as [`renaming`](Self::renaming) resolves `mapping`.
    pub fn rename(&self, mapping: &HashMap<String, String>) -> Result<DataFrame> {
        let mapping = self.renaming(mapping);
        let (names, columns) = self.data_parts();
        let names = names.into_iter().map(|n| mapping.get(&n).cloned().unwrap_or(n)).collect();
        DataFrame::new_series(names, columns, (*self.index).clone())
    }

    /// Cast the named data columns to new dtypes (pandas `astype`) — a new series of the
    /// data columns; a name that is not one is a [`VolasError::ColumnNotFound`].
    pub fn astype(&self, mapping: &HashMap<String, crate::dtype::DType>) -> Result<DataFrame> {
        let (names, mut columns) = self.data_parts();
        for (name, dtype) in mapping {
            let pos = names
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| VolasError::ColumnNotFound(name.clone()))?;
            columns[pos] = columns[pos].cast(*dtype)?;
        }
        DataFrame::new_series(names, columns, (*self.index).clone())
    }

    /// Move data column `name` into the row index (pandas `set_index`) — a new series of
    /// the other data columns labelled by it ([`with_labels`](Self::with_labels)). The index
    /// records the column's name (pandas keeps it, so `reset_index` restores the column
    /// label).
    pub fn set_index(&self, name: &str) -> Result<DataFrame> {
        let labels = self.column(name)?.clone();
        let rest: Vec<String> = self.data_names().into_iter().filter(|n| n != name).collect();
        if rest.len() == self.data_names().len() {
            return Err(VolasError::ColumnNotFound(name.to_string())); // a cached column
        }
        self.select(&rest)?.with_labels(&labels, crate::tz::Tz::Naive, Some(name.to_string()))
    }

    /// The data columns re-labelled by `labels` (one per row; a datetime column's labels in
    /// zone `tz`, the index named `name`; pandas `DataFrame(df, index=...)`) — a new series.
    /// Every frame's labels are unique ([`Index::from_column_tz`]): a repeated label raises,
    /// consecutive or not (only the fine bars a cumulation is fed keep re-send semantics).
    pub fn with_labels(&self, labels: &Column, tz: crate::tz::Tz, name: Option<String>) -> Result<DataFrame> {
        if labels.len() != self.height {
            return Err(VolasError::Shape(format!("index length {} != frame height {}", labels.len(), self.height)));
        }
        let index = Index::from_column_tz(labels, tz)?.with_name(name);
        self.reindexed(index)
    }

    /// Move the row index into a leading column `label` (pandas `reset_index`) — or
    /// discard it (`None`) — under a fresh `RangeIndex`: a new series of the data columns.
    pub fn reset_index(&self, label: Option<&str>) -> Result<DataFrame> {
        let (mut names, mut columns) = self.data_parts();
        if let Some(label) = label {
            if names.iter().any(|n| n == label) {
                return Err(VolasError::Value(format!(
                    "reset_index: column name {label:?} already exists (would duplicate)"
                )));
            }
            names.insert(0, label.to_string());
            columns.insert(0, self.index.to_column());
        }
        DataFrame::new_series(names, columns, Index::range(self.height))
    }

    /// Change the DatetimeIndex's **display / matching** timezone without moving any
    /// instant (pandas `tz_convert`) — a new series of the data columns. Errors if the
    /// index is not a DatetimeIndex, or is naive (an unanchored wall clock has no zone to
    /// convert from: `tz_localize` it first, as pandas requires).
    pub fn tz_convert(&self, tz: crate::tz::Tz) -> Result<DataFrame> {
        match self.index.kind() {
            IndexKind::Datetime(_, cur) if cur.is_aware() => self.reindexed((*self.index).clone().with_tz(tz)),
            IndexKind::Datetime(..) => Err(VolasError::DType(
                "cannot tz_convert a tz-naive DatetimeIndex; use tz_localize to anchor it first".into(),
            )),
            _ => Err(VolasError::DType("tz_convert requires a DatetimeIndex".into())),
        }
    }

    /// Tag the DatetimeIndex's zone directly, without the naive-axis guard of
    /// [`Self::tz_convert`] — for importers (`from_pandas`) whose instants are ALREADY true
    /// UTC and arrive carrying their zone; not a user-facing API. A new series.
    pub fn set_index_tz(&self, tz: crate::tz::Tz) -> Result<DataFrame> {
        match self.index.kind() {
            IndexKind::Datetime(..) => self.reindexed((*self.index).clone().with_tz(tz)),
            _ => Err(VolasError::DType("set_index_tz requires a DatetimeIndex".into())),
        }
    }

    /// Reinterpret the index's **wall clock** as `tz` (pandas `tz_localize`): each instant
    /// is recomputed so the displayed wall clock is unchanged but now correct for `tz` — a
    /// new series of the data columns. Errors if the index is not a naive DatetimeIndex or
    /// a wall clock does not exist in `tz` (a DST gap).
    pub fn tz_localize(&self, tz: crate::tz::Tz) -> Result<DataFrame> {
        let (values, cur) = match self.index.kind() {
            IndexKind::Datetime(v, cur) => (v, cur),
            _ => return Err(VolasError::DType("tz_localize requires a DatetimeIndex".into())),
        };
        // Localize anchors an UNanchored wall clock; an already-aware axis must use
        // tz_convert (re-localizing would silently reinterpret instants).
        if cur.is_aware() {
            return Err(VolasError::DType(format!(
                "index is already tz-aware ({}); use tz_convert",
                cur.name()
            )));
        }
        // `Tz` owns the wall clock -> instant conversion (sub-second digits kept; a DST
        // gap / fold or an instant past the range raises); a NaT label stays NaT.
        let shifted = tz.localize_values(values, cur)?;
        self.reindexed(Index::datetime(shifted, tz).with_name(self.index.name().map(String::from)))
    }

    /// The data columns over `index` (same length): a new series.
    fn reindexed(&self, index: Index) -> Result<DataFrame> {
        let (names, columns) = self.data_parts();
        DataFrame::new_series(names, columns, index)
    }
}
