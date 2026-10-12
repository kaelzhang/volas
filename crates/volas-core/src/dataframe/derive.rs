//! The one owner of what a frame derived from another keeps of it. Every derivation —
//! a row selection, a column projection, a relabelling, a cast, a time-zone change —
//! is one of these methods; the binding only maps its arguments onto them.
//!
//! - A **contiguous row selection** (the rows `[start, end)` in order, however it is
//!   written) continues the source's series: its row 0 keeps its absolute position
//!   ([`DataFrame::origin`]) and every cached directive column stays cached, with its
//!   values and with what continuing it needs — the carried states moved to the
//!   selection's last row (replayed once, through [`Replay`], from the history the source
//!   holds when the selection ends before the source's last valid row), the checkpoint of
//!   its history when the selection holds the row after it, and the **lead**: the data
//!   rows before row 0 that recomputing its first rows reads. A column the selection
//!   cannot recompute from what it keeps (a recursion that carries no state, whose
//!   history started before the selection) keeps its values and raises when a refresh
//!   would recompute it: it is never restarted on the selection's own rows.
//! - A **column projection** keeps the rows: the listed columns, cached ones continuing.
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

use super::{Checkpoint, ComputedMeta, DataFrame, Reach, STALE_BY_APPEND};

/// The directive layer's half of a contiguous selection that ends before a recursive
/// cached column's last valid row: the column's carried states as of the selection's
/// last row, which the source frame does not carry. The core knows where a column's
/// history is; only the directive layer can replay it.
pub trait Replay {
    /// The carried states of the recursive cached column `name` of `frame` as of
    /// `frame`'s row `end - 1` and as of the row before it (`None` for that one when
    /// there is no state of it), replayed from the history `frame` holds of the column:
    /// its rows from the column's history start, or its checkpoint. `None` when `frame`
    /// holds neither, or the column carries no state.
    fn states(&self, frame: &DataFrame, name: &str, end: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)>;
}

/// The data rows just before a frame's row 0 that its cached columns read (see the
/// module docs): the rows `[from, to)` of `rows`, a frame of data columns whose row `to`
/// would be the frame's row 0. A lead that is most of its source's rows shares the
/// source's buffers; a shorter one is a copy of just its rows, so a selection never pins
/// more than twice the rows its lead keeps.
#[derive(Clone, Debug)]
pub(super) struct Lead {
    rows: Arc<DataFrame>,
    from: usize,
    to: usize,
}

impl Lead {
    /// How many rows the lead keeps.
    fn len(&self) -> usize {
        self.to - self.from
    }

    /// The last `n` (at most [`len`](Self::len)) rows of column `name` in `dtype` — the
    /// dtype the column has now. They are missing when the lead does not hold the column
    /// (one added after it: no column whose history starts before the frame reads it), or
    /// when its values there do not convert to `dtype` exactly (a column replaced by one
    /// of another dtype): never a value the history did not have.
    fn tail(&self, name: &str, n: usize, dtype: crate::DType) -> Column {
        self.rows
            .column(name)
            .ok()
            .and_then(|c| c.slice(self.to - n, self.to).convert_lossless(dtype).ok())
            .unwrap_or_else(|| Column::na_of(dtype, n))
    }

    /// This lead over the data columns `names` it holds (a projection).
    fn select(&self, names: &[String]) -> Lead {
        let kept: Vec<String> = names.iter().filter(|n| self.rows.has_column(n)).cloned().collect();
        Lead { rows: Arc::new(self.rows.select(&kept).expect("held columns")), ..*self }
    }

    #[cfg(debug_assertions)]
    pub(super) fn debug_check(&self, origin: usize) {
        assert!(self.rows.computed.is_empty() && self.rows.lead.is_none(), "a lead holds data rows only");
        assert!(self.from <= self.to && self.to <= self.rows.height, "the lead's rows are its frame's");
        assert_eq!(self.rows.origin + self.to, origin, "the lead ends just before row 0");
    }
}

/// A replay that never has the history (core tests: a selection that needs it keeps the
/// column without a state).
#[cfg(test)]
pub(super) struct NoReplay;

#[cfg(test)]
impl Replay for NoReplay {
    fn states(&self, _: &DataFrame, _: &str, _: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> {
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
        self.names.iter().filter(|n| !self.computed.contains_key(*n)).cloned().collect()
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
    /// docs): every cached column stays cached; `replay` replays a recursion's states
    /// when the selection ends before its last valid row.
    ///
    /// The rows are a **value copy**, not a view into this frame's buffers: a selection is
    /// an independent frame, so selecting the recent tail of a long history does not pin
    /// the whole history alive. What it keeps of the rows before it is bounded by its
    /// cached columns' reach (the lead).
    pub fn rows(&self, start: usize, end: usize, replay: &dyn Replay) -> DataFrame {
        let start = start.min(self.height);
        let end = end.max(start).min(self.height);
        let columns: Vec<Column> = self.columns.iter().map(|c| c.slice(start, end)).collect();
        let mut df = self.same_schema(columns, self.index.slice(start, end), self.origin + start);
        df.stale_by = self.stale_by;
        for (name, meta) in &self.computed {
            df.computed.insert(name.clone(), self.continued(name, meta, start, end, replay));
        }
        df.lead = self.lead_before(start, &df.computed);
        df.note_computed();
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
    /// it: its valid rows, its states as of its last valid row there (this frame's, or
    /// replayed), and the checkpoints that cover rows before ones it holds.
    fn continued(&self, name: &str, meta: &ComputedMeta, start: usize, end: usize, replay: &dyn Replay) -> ComputedMeta {
        let first = self.origin + start;
        let valid = meta.valid_rows.saturating_sub(start).min(end - start);
        let last = first + valid; // one past the selection's last valid row
        let own = meta.state.is_some() && meta.last_valid(self.origin) == Some(meta.state_at);
        let (state, before) = if valid == 0 {
            (None, None)
        } else if own && meta.state_at + 1 == last {
            (meta.state.clone(), meta.state_before_last.clone())
        } else if meta.reach.history.recursive() && meta.state.is_some() {
            // ends before the column's last valid row: replayed once from its history (a
            // recursion that carries no state has none to replay)
            replay.states(self, name, start + valid).map_or((None, None), |(s, b)| (Some(s), b))
        } else {
            (None, None) // a window recomputes; no state needed
        };
        // the state before describes the row before the last, which must be held
        let before = before.filter(|_| state.is_some() && last - 1 > first);
        // a checkpoint is replayed from over the rows after it: kept when they are held
        let covers = |cp: &Checkpoint, rows: usize| (first..=first + rows).contains(&(cp.at + 1));
        ComputedMeta {
            reach: meta.reach,
            valid_rows: valid,
            state,
            state_before_last: before,
            since: meta.since,
            state_at: last.saturating_sub(1),
            scratch: Vec::new(),
            checkpoint: meta.checkpoint.clone().filter(|cp| covers(cp, valid)),
            pending: meta.pending.clone().filter(|cp| covers(cp, end - start)),
            lost: meta.lost.clone(),
            first_present: meta.first_present,
        }
    }

    /// The lead of the selection starting at row `start` whose cached columns are
    /// `computed`: the data rows before it that they read to recompute its first rows —
    /// for each column whose history starts before it, the fewer of its
    /// [`lead`](Reach::lead) and its history's rows there, or, for a recursion that
    /// carries no state, its whole history there (it is computed over all of it) — taken
    /// from this frame's rows before `start`, then its own lead. `None` when no column
    /// reads before it.
    fn lead_before(&self, start: usize, computed: &FxHashMap<String, ComputedMeta>) -> Option<Lead> {
        let first = self.origin + start;
        let wanted = computed
            .values()
            .filter(|m| m.since < first)
            .map(|m| match m.reach.history.recursive() && m.state.is_none() && m.checkpoint.is_none() {
                true => first - m.since, // computed over all of it
                false => m.reach.lead().min(first - m.since),
            })
            .max()
            .unwrap_or(0);
        let own = wanted.min(start);
        let held = self.lead.as_ref().map_or(0, Lead::len);
        let older = (wanted - own).min(held);
        if own + older == 0 {
            return None;
        }
        let names = self.data_names();
        if older == 0 && 2 * own >= self.height {
            // most of this frame's rows: share its buffers rather than copy them
            let rows = Arc::new(self.select(&names).expect("data columns"));
            return Some(Lead { rows, from: start - own, to: start });
        }
        let columns = names
            .iter()
            .map(|name| {
                let col = &self.columns[self.name_to_idx[name]];
                let mut out = match &self.lead {
                    Some(lead) if older > 0 => lead.tail(name, older, col.dtype()),
                    _ => col.slice(0, 0),
                };
                out.append(&col.slice(start - own, start)).expect("same dtype");
                out
            })
            .collect();
        let len = own + older;
        let rows = DataFrame::new(names, columns, Some(Index::range(len)))
            .expect("equal-length data columns")
            .with_origin(first - len);
        Some(Lead { rows: Arc::new(rows), from: 0, to: len })
    }

    /// How many data rows before row 0 this frame keeps in its lead (see the module docs).
    pub fn lead_rows(&self) -> usize {
        self.lead.as_ref().map_or(0, Lead::len)
    }

    /// `frame` — this frame's rows over some of its data columns (the frame a cached
    /// column recomputes over) — preceded by the last `rows` rows of this frame's lead
    /// (all of it when it holds fewer): the frame a recompute that reads before row 0 runs
    /// over, its row 0 at the absolute position of the first lead row it takes. A column
    /// the lead does not hold is missing on its rows. Its labels are positions (no
    /// computation reads labels).
    pub fn led(&self, frame: &DataFrame, rows: usize) -> DataFrame {
        let Some(lead) = self.lead.as_ref().filter(|_| rows > 0) else {
            return frame.clone();
        };
        let take = rows.min(lead.len());
        let columns = frame
            .names
            .iter()
            .zip(&frame.columns)
            .map(|(name, col)| {
                let mut out = lead.tail(name, take, col.dtype());
                out.append(col).expect("same dtype");
                out
            })
            .collect();
        DataFrame::new(frame.names.to_vec(), columns, Some(Index::range(take + frame.height)))
            .expect("equal-length columns")
            .with_origin(frame.origin - take)
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

    /// The columns `names`, in that order, over the same rows: the same series (its
    /// origin), a listed cached column staying cached as it is, and the lead of the listed
    /// data columns.
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
        let mut df = DataFrame {
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
        df.computed = names
            .iter()
            .filter_map(|n| self.computed.get(n).map(|m| (n.clone(), m.clone())))
            .collect();
        df.note_computed();
        if !df.computed.is_empty() {
            df.lead = self.lead.as_ref().map(|l| l.select(&df.data_names()));
        }
        #[cfg(debug_assertions)]
        df.debug_check();
        Ok(df)
    }

    // --- new series ------------------------------------------------------------------------

    /// Rename columns (pandas `rename(columns=...)`) — a new series of the data columns
    /// (a name in `mapping` that is not one is ignored, as pandas does).
    pub fn rename(&self, mapping: &HashMap<String, String>) -> Result<DataFrame> {
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
            IndexKind::Datetime(v, cur) => (v, *cur),
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
        let shifted = values
            .iter()
            .map(|&ns| match ns {
                crate::datetime::NAT => Ok(ns),
                _ => tz.localize_from(ns, cur),
            })
            .collect::<Result<Vec<i64>>>()?;
        self.reindexed(Index::datetime(shifted, tz).with_name(self.index.name().map(String::from)))
    }

    /// The data columns over `index` (same length): a new series.
    fn reindexed(&self, index: Index) -> Result<DataFrame> {
        let (names, columns) = self.data_parts();
        DataFrame::new_series(names, columns, index)
    }
}
