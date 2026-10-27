//! DataFrame: ordered, named columns sharing a single row index.


use crate::column::ColumnData;
use std::sync::Arc;

use crate::column::Column;
use crate::fxhash::FxHashMap;
use crate::error::{Result, VolasError};
use crate::index::Index;
use crate::series::Series;

/// The rows a cached directive column reads around its own, fixed per directive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reach {
    /// The directive's lookback (warm-up rows).
    pub lookback: usize,
    /// The valid rows before a new one that resuming the column's carried state reads
    /// (0 when its framed state owns every kernel tail): a frame holding fewer
    /// recomputes instead. This counts raw frame rows, never rows saved in the state.
    pub resume: usize,
    /// The rows a contiguous slice must keep for the column to stay cached (at least
    /// `lookback`): fewer cannot continue it exactly.
    pub continuation: usize,
    /// How a value depends on the rows before it ([`History`]): a recursion anywhere
    /// makes it depend on every row before, so a slice continues the column exactly only
    /// from its carried state, and a frame that dropped its history's head recomputes it
    /// only from a [`Checkpoint`].
    pub history: History,
}

/// The directive a cached column computes, bound by the directive engine and opaque to
/// the core: kept in the column's metadata — carried through clones and contiguous
/// slices with it — so a refresh continues exactly what was cached instead of re-reading
/// the column's name.
pub type Directive = Arc<dyn std::any::Any + Send + Sync>;

/// Metadata for a materialized (cached) directive column: the directive that
/// produced it, its lookback, and how many leading rows currently hold valid
/// values. After an `append`, the new rows are stale (NaN) and `valid_rows` lags
/// `height` until `fulfill` recomputes the tail.
#[derive(Debug)]
#[repr(align(16))]
pub struct ComputedMeta {
    /// The bound directive the column computes (its canonical name is the column's).
    pub directive: Directive,
    /// The rows the column reads around its own: lookback, resume and continuation.
    pub reach: Reach,
    /// Rows `[0, valid_rows)` currently hold valid values.
    pub valid_rows: usize,
    /// Carried recursive state for an O(new-rows) append resume: a small,
    /// fixed-size per-indicator vector capturing the internal recursive state as
    /// of the last valid row (`valid_rows - 1`), so an `append`/`fulfill` can
    /// continue the recursion over only the new rows, bit-identical to a fresh
    /// full recompute. `None` when the directive has no resume implementation (it
    /// then falls back to the correct full recompute) or the state is unknown
    /// (e.g. after a slice that did not reach the parent's `valid_rows`).
    pub state: Option<Arc<[f64]>>,
    /// The carried state as of the row before the last valid one (`valid_rows - 2`):
    /// the state that row was resumed from. When the last row is replaced in place (a
    /// row appended with the last row's label, a time-frame forming bar), it becomes
    /// [`state`](Self::state) again, so the replacement is recomputed in one step from
    /// it — never re-stepping or recomputing the rows before. A resume of one new row
    /// moves the old `state` here. `None` when unknown (a replaced row then recomputes
    /// the column, capturing both states again).
    pub state_before_last: Option<Arc<[f64]>>,
    /// The absolute position (see [`DataFrame::origin`]) of the first row the column's
    /// history starts at — the row its warm-up counts from. Its rows `[since, origin +
    /// valid_rows)` have been processed, so warm-up is judged on that count, never on
    /// a slice's local row number.
    pub since: usize,
    /// The absolute position of the row [`state`](Self::state) describes (the one
    /// before it for `state_before_last`). A state is only ever resumed from that
    /// row: one whose recorded row is not the last valid row is refused.
    state_at: usize,
    /// A spare buffer the one-row resume writes the next state into, recycled between
    /// the two states so a live column steps with no allocation.
    scratch: Option<Arc<[f64]>>,
    /// The history of a recursion before the frame's row 0, once the frame dropped the
    /// rows it started on ([`Checkpoint`]): a recompute replays from it. Shared: a
    /// checkpoint never changes once formed.
    checkpoint: Option<CheckpointRef>,
    /// The checkpoint the next drop of the head will keep, captured as the refresh passed
    /// its row (so the drop need not replay to it); dropped by any change to the rows it
    /// covers.
    /// Kept outside hot metadata: most columns have no pending boundary.
    pending: Option<CheckpointRef>,
    /// Checkpoints of the history at regular rows ([`Anchors`]), formed by replays.
    anchors: Anchors,
    /// Why no checkpoint of the history could be formed when the frame dropped the rows
    /// it read (an input outside the computation's domain): a recompute reports it.
    lost: Option<crate::error::VolasError>,
    /// The absolute position of the column's first present (non-missing) value among the
    /// rows written so far — where its warm-up ended; `None` until one is written.
    first_present: Option<usize>,
}

impl ComputedMeta {
    /// This column as a reader's view keeps it (`DataFrame::view_rows`): its reach, where
    /// its history starts and its first present value — nothing that continues it.
    fn view(&self) -> ComputedMeta {
        ComputedMeta {
            directive: Arc::clone(&self.directive),
            reach: self.reach,
            valid_rows: self.valid_rows,
            state: None,
            state_before_last: None,
            since: self.since,
            state_at: self.state_at,
            scratch: None,
            checkpoint: None,
            pending: None,
            anchors: Anchors::default(),
            lost: None,
            first_present: self.first_present,
        }
    }

    /// Note that rows `[from, end)` (frame rows, row 0 at absolute position `origin`) of
    /// `col` were written: the first present value among them is the column's first when
    /// none was known before them. Scans only while the first value is unknown or being
    /// rewritten, so a live column pays nothing per row once warmed.
    fn note_written(&mut self, col: &Column, origin: usize, from: usize, end: usize) {
        if self.first_present.is_some_and(|f| f < origin + from) {
            return;
        }
        let found = (from..end).find(|&i| col.is_valid(i)).map(|i| origin + i);
        if found.is_some() || self.first_present.is_some_and(|f| f < origin + end) {
            self.first_present = found;
        }
    }

    /// The absolute position of the column's last valid row, if it has one.
    fn last_valid(&self, origin: usize) -> Option<usize> {
        (origin + self.valid_rows).checked_sub(1).filter(|_| self.valid_rows > 0)
    }

    /// Whether the column is past its warm-up: more than `lookback` rows of its
    /// history are processed.
    pub fn warm(&self, origin: usize) -> bool {
        origin + self.valid_rows >= self.since + self.reach.lookback
    }
}

/// A 2-D, column-oriented, time-indexed table. All columns share one index and
/// have equal length (`height`).
#[derive(Debug)]
pub struct DataFrame {
    // Schema (names + lookup) is `Arc`-shared so a frame clone / same-schema
    // derivation (slice / take / mask / astype) is an O(1) refcount bump, not a
    // rebuild of the name strings + hash map (copy-on-write on mutation).
    names: Arc<Vec<String>>,
    columns: Vec<Column>,
    // These internal name maps use FxHash (not the default SipHash): their keys are
    // internal column / directive names, and they are hit on the live-append hot path.
    name_to_idx: Arc<FxHashMap<String, usize>>,
    index: Arc<Index>,
    height: usize,
    /// Materialized directive columns (name -> meta). Tracked so `fulfill` can
    /// incrementally recompute their tail after an append. Carried through
    /// `clone` / `append`; dropped by shape-changing ops (slice/select/…), where
    /// the columns become plain data.
    computed: FxHashMap<Arc<str>, ComputedMeta>,
    /// The position of the first cached directive column (`None` without one) — derived
    /// from `computed` (`note_computed` keeps it), so whether the leading columns hold a
    /// cached one is O(1) on the per-bar append path.
    first_computed: Option<usize>,
    /// The absolute position of row 0 in the series this frame continues: `0` for a
    /// new frame; a contiguous slice (a bounded frame dropping its head included) adds
    /// its start, so every row keeps the position it has in the full series. A
    /// derivation over the same rows (columns, names, dtypes, index) keeps it; one
    /// that selects rows otherwise (`take`, a mask) starts a new series at `0`.
    /// Sliding-window computations re-anchor their running sums at absolute positions
    /// (see `volas_compute`), so a value depends only on the rows near it, the same in
    /// the frame and in the full series; the index family (maxindex / minindex /
    /// minmaxindex) emits absolute positions.
    origin: usize,
    /// What last left cached columns stale ("an append", "a column write"), for the
    /// read gate's message; meaningful only while one is stale.
    stale_by: &'static str,
    /// The data rows just before row 0 that the cached columns read to recompute their
    /// first rows (see [`lead_rows`](Self::lead_rows)): a contiguous selection that drops
    /// the rows its columns' windows reach into keeps them here. `None` with none.
    lead: Option<lead::Lead>,
}

impl Clone for DataFrame {
    fn clone(&self) -> Self {
        // Copies share their immutable prefix, never future publications.
        let computed = self.computed.iter().map(|(name, meta)| {
            (Arc::clone(name), meta.forked(self.origin))
        }).collect();
        Self {
            names: Arc::clone(&self.names), columns: self.columns.clone(),
            name_to_idx: Arc::clone(&self.name_to_idx), index: Arc::clone(&self.index),
            height: self.height, computed, first_computed: self.first_computed,
            origin: self.origin, stale_by: self.stale_by, lead: self.lead.clone(),
        }
    }
}

/// The cause an append (new rows, a replaced last row, a folded forming bar) leaves.
const STALE_BY_APPEND: &str = "an append";
/// The cause a write to a column leaves.
const STALE_BY_WRITE: &str = "a column write";

/// The layout every column a frame admits must hold (its validity covers its values,
/// [`Column::check_layout`]) — a column assembled from its public parts may carry a
/// mask of another length, and no frame entry point takes one.
fn admit_layout(name: &str, col: &Column) -> Result<()> {
    match col.check_layout() {
        Err(VolasError::Shape(why)) => Err(VolasError::Shape(format!("column \"{name}\": {why}"))),
        other => other,
    }
}

/// Recover the first repeated name only after the successful-path count check failed.
#[cold]
fn duplicate_name(names: &[String]) -> VolasError {
    let mut seen = std::collections::HashSet::with_capacity(names.len());
    let name = names.iter().find(|name| !seen.insert(name.as_str())).expect("name count found a duplicate");
    VolasError::Value(format!("duplicate column name \"{name}\""))
}

impl DataFrame {
    /// Construct a frame from parallel `names` / `columns`, validating shape.
    pub fn new(names: Vec<String>, columns: Vec<Column>, index: Option<Index>) -> Result<Self> {
        Self::admit(names, columns, index, None)
    }

    /// Admit columns with an explicit row count and a checked positional index.
    /// The row count also preserves zero-column Arrow tables.
    pub fn from_columns(names: Vec<String>, columns: Vec<Column>, height: usize) -> Result<Self> {
        Self::admit(names, columns, None, Some(height))
    }

    fn admit(names: Vec<String>, columns: Vec<Column>, index: Option<Index>, height: Option<usize>) -> Result<Self> {
        if names.len() != columns.len() {
            return Err(VolasError::Shape(format!(
                "{} names but {} columns",
                names.len(),
                columns.len()
            )));
        }
        // a frame of no columns is as tall as its index (pandas `df[[]]`)
        let extent = columns.first().map(Column::extent);
        let height = height.or_else(|| extent.map(|e| e.len())).or_else(|| index.as_ref().map(Index::len)).unwrap_or(0);
        for (pos, (n, c)) in names.iter().zip(&columns).enumerate() {
            let len = if pos == 0 { extent.unwrap().len() } else { c.len() };
            if len != height {
                return Err(VolasError::Shape(format!(
                    "column \"{}\" has length {} but frame height is {}",
                    n,
                    len,
                    height
                )));
            }
            admit_layout(n, c)?;
        }
        let index = match index {
            Some(ix) => {
                ix.validate()?;
                if ix.len() != height {
                    return Err(VolasError::Shape(format!(
                        "index length {} != frame height {}",
                        ix.len(),
                        height
                    )));
                }
                ix
            }
            None => match extent { Some(extent) => Index::for_extent(extent), None => Index::positional(height)? },
        };
        let mut name_to_idx = FxHashMap::with_capacity_and_hasher(names.len(), Default::default());
        for (i, n) in names.iter().enumerate() {
            name_to_idx.insert(n.clone(), i);
        }
        if name_to_idx.len() != names.len() {
            return Err(duplicate_name(&names));
        }
        let df = DataFrame {
            names: Arc::new(names),
            columns,
            name_to_idx: Arc::new(name_to_idx),
            index: Arc::new(index),
            height,
            computed: FxHashMap::default(),
            first_computed: None,
            origin: 0,
            stale_by: STALE_BY_APPEND,
            lead: None,
        };
        #[cfg(debug_assertions)]
        df.debug_check();
        Ok(df)
    }

    /// Debug-build structural self-check, run at the end of every mutation (compiled
    /// out of release builds): names, columns and the name map agree; every column,
    /// its validity and the index span the frame's height; every computed column
    /// exists and its valid-row cursor stays within the height.
    #[cfg(debug_assertions)]
    pub(crate) fn debug_check(&self) {
        assert_eq!(self.names.len(), self.columns.len(), "names and columns differ in count");
        assert_eq!(self.name_to_idx.len(), self.names.len(), "the name map has stale entries");
        for (i, (name, col)) in self.names.iter().zip(&self.columns).enumerate() {
            assert_eq!(self.name_to_idx.get(name), Some(&i), "name map entry of {name:?}");
            assert_eq!(col.len(), self.height, "column {name:?} length differs from the height");
            col.debug_check(name);
        }
        assert_eq!(self.index.len(), self.height, "index length differs from the height");
        let first = self.computed.keys().filter_map(|k| self.name_to_idx.get(k.as_ref()).copied()).min();
        assert_eq!(self.first_computed, first, "first cached column position is stale");
        for (name, meta) in &self.computed {
            assert!(self.name_to_idx.contains_key(name.as_ref()), "computed {name:?} is not a column");
            assert!(meta.since <= self.origin, "computed {name:?} history starts after row 0");
            assert!(meta.valid_rows <= self.height, "computed {name:?} cursor past the height");
            // a carried state describes a valid row of THIS frame (never one before it)
            if meta.state.is_some() {
                let rows = self.origin..self.origin + meta.valid_rows;
                assert!(rows.contains(&meta.state_at), "computed {name:?} state outside its rows");
            }
            if meta.state_before_last.is_some() {
                assert!(meta.state_at > self.origin, "computed {name:?} earlier state before row 0");
            }
        }
        if let Some(lead) = &self.lead {
            lead.debug_check();
        }
    }

    /// The absolute position of row 0 in the series this frame continues (see the
    /// field).
    pub fn origin(&self) -> usize {
        self.origin
    }

    /// This frame, its row 0 at absolute position `origin` — for a derivation that
    /// keeps the rows of a frame through [`DataFrame::new`].
    pub fn with_origin(mut self, origin: usize) -> Self {
        self.origin = origin;
        self
    }

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Column names in order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The `Arc`-shared name vector. Pointer-stable across row-only mutations
    /// (`append` / forming-row folds), so a caller can validate "schema unchanged"
    /// with an O(1) [`Arc::ptr_eq`] instead of an element-wise name comparison.
    pub fn names_arc(&self) -> &Arc<Vec<String>> {
        &self.names
    }

    /// The shared row index.
    pub fn index(&self) -> &Arc<Index> {
        &self.index
    }

    /// Columns in order.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Position of a column by name.
    pub fn column_pos(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    /// Whether a column exists.
    pub fn has_column(&self, name: &str) -> bool {
        self.name_to_idx.contains_key(name)
    }

    /// Whether `name` is a cached directive (computed) column rather than plain data.
    /// Plain columns are supplied per bar; computed ones are derived and refreshed by
    /// `fulfill`. A read-only metadata lookup — off every compute / append hot path.
    pub fn is_computed(&self, name: &str) -> bool {
        self.computed.contains_key(name)
    }

    /// Borrow a column by name.
    pub fn column(&self, name: &str) -> Result<&Column> {
        self.column_pos(name)
            .map(|i| &self.columns[i])
            .ok_or_else(|| VolasError::ColumnNotFound(name.to_string()))
    }

    /// Extract a column as a [`Series`] sharing this frame's index.
    pub fn series(&self, name: impl Into<String>) -> Result<Series> {
        let name = name.into();
        let col = self.column(&name)?.clone();
        Ok(Series::from_frame_parts(Some(name), col, Arc::clone(&self.index)))
    }

    /// Extract a column by position, carrying the frame's admitted shape.
    /// Panics if `position` is outside the column axis, like indexing `columns()`.
    #[inline]
    pub fn series_at(&self, position: usize) -> Series {
        Series::from_frame_parts(Some(self.names[position].clone()), self.columns[position].clone(), Arc::clone(&self.index))
    }

    /// Extract one column's contiguous rows, clamped to the frame's height.
    /// The same admitted extent supplies both values and labels, without first
    /// constructing a full Series or rechecking their equal length.
    pub fn series_rows(&self, name: impl Into<String>, start: usize, end: usize) -> Result<Series> {
        let name = name.into();
        let col = self.column(&name)?;
        let start = start.min(self.height);
        let end = end.max(start).min(self.height);
        Ok(Series::from_frame_parts(Some(name), col.slice(start, end), Arc::new(self.index.slice(start, end))))
    }

    /// Add a new column or replace an existing one (must match `height`, unless
    /// the frame currently has no columns). Replacing a column invalidates the
    /// directive cache (see `invalidate_computed_on_write`); adding one leaves it.
    pub fn set_column(&mut self, name: &str, col: Column) -> Result<()> {
        self.install_column(name, col, false)
    }

    /// Install values and computed identity in one schema transaction.
    pub fn set_computed_column(&mut self, name: &str, col: Column, directive: Directive, reach: Reach) -> Result<()> {
        self.install_column(name, col, true)?;
        self.record_computed(name, directive, reach);
        Ok(())
    }

    fn install_column(&mut self, name: &str, col: Column, computed: bool) -> Result<()> {
        admit_layout(name, &col)?;
        if self.columns.is_empty() {
            self.height = col.len();
            if self.index.len() != self.height {
                self.index = Arc::new(Index::new(None, crate::IndexKind::Range { start: 0, len: self.height })?);
            }
        } else if col.len() != self.height {
            return Err(VolasError::Shape(format!(
                "new column \"{}\" has length {} but frame height is {}",
                name,
                col.len(),
                self.height
            )));
        }
        match self.column_pos(name) {
            Some(i) => {
                // The rows before row 0 keep their values: a dtype that cannot hold them
                // exactly is refused here, before anything is written.
                self.check_lead_dtype(name, &col)?;
                // A replaced column may feed any cached directive: `name` loses its own
                // computed status and every other cache goes stale (one rule for every
                // writer — `df[name] = …`, `astype`, …).
                // Retire any reusable schema proof before changing the dtype or
                // turning a cached column into data.
                if self.columns[i].dtype() != col.dtype() || self.computed.contains_key(name) != computed {
                    Arc::make_mut(&mut self.names);
                }
                self.columns[i] = col;
                if computed {
                    // The same computed role survives. Other dependencies still
                    // become stale when its installed values are replaced.
                    self.stale_every_computed();
                } else {
                    self.computed.remove(name);
                    self.note_computed();
                    self.stale_every_computed();
                }
            }
            None => {
                Arc::make_mut(&mut self.name_to_idx).insert(name.to_string(), self.columns.len());
                Arc::make_mut(&mut self.names).push(name.to_string());
                self.columns.push(col);
            }
        }
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }

    /// Value equality (pandas `DataFrame.equals`): same column names + order,
    /// same index *labels*, and value-equal columns (`NaN == NaN`). The index
    /// *name* is metadata and is ignored, matching pandas (`.equals` ignores it).
    pub fn equals(&self, other: &DataFrame) -> bool {
        self.height == other.height
            && self.names == other.names
            && self.index.label_eq(&other.index)
            && self
                .columns
                .iter()
                .zip(&other.columns)
                .all(|(a, b)| a.equals(b))
    }

    /// Flatten to a row-major (C-order) 2-D `f64` buffer for NumPy export,
    /// returning `(data, height, width)`. Float64 inputs are borrowed; other
    /// dtypes use the validity-aware conversion, so a missing cell (int/bool NA, datetime NaT,
    /// str) exports as `NaN` — not the raw placeholder — matching the 1-D
    /// `Series` export and `pandas` `Int64.to_numpy()`.
    pub fn to_row_major_f64(&self) -> (Vec<f64>, usize, usize) {
        let h = self.height;
        let w = self.columns.len();
        let mut out = vec![0.0f64; h * w];
        for (j, c) in self.columns.iter().enumerate() {
            let col = match c {
                Column { data: ColumnData::F64(v) } => std::borrow::Cow::Borrowed(v.as_slice()),
                _ => std::borrow::Cow::Owned(c.to_f64_vec()),
            };
            for i in 0..h {
                out[i * w + j] = col[i];
            }
        }
        (out, h, w)
    }

    /// Flatten to a row-major (C-order) 2-D `i64` buffer for an **exact** integer
    /// (or `datetime64[ns]`) NumPy export, returning `(data, height, width)`. A
    /// datetime column contributes its raw epoch-ns — so sub-2⁵³ ns and `NaT`
    /// (which stays `i64::MIN`, the datetime64 sentinel) survive, unlike the
    /// `to_row_major_f64` channel — and an `i64` column its exact value (no
    /// float round-trip past 2⁵³). A float column truncates toward zero. A `str`
    /// column has no integer meaning; the export boundary rejects it before
    /// calling this, so it contributes a `0` placeholder it never reaches.
    pub fn to_row_major_i64(&self) -> (Vec<i64>, usize, usize) {
        let h = self.height;
        let w = self.columns.len();
        let mut out = vec![0i64; h * w];
        for (j, c) in self.columns.iter().enumerate() {
            for i in 0..h {
                out[i * w + j] = match c {
                    Column { data: ColumnData::Datetime(v) } => v[i],
                    Column { data: ColumnData::I64(v, _) } => v[i],
                    Column { data: ColumnData::I32(v, _) } => v[i] as i64,
                    Column { data: ColumnData::Bool(v, _) } => v[i] as i64,
                    Column { data: ColumnData::F64(v) } => v[i] as i64,
                    Column { data: ColumnData::F32(v) } => v[i] as i64,
                    Column { data: ColumnData::Str(..) } => 0,
                };
            }
        }
        (out, h, w)
    }
}

mod fold_input;
pub use fold_input::{AdmittedAppend, AdmittedBar, AdmittedBars, CheckpointFold, FoldLayout};
mod admit;
pub use admit::{FoldedRow, FormingFold};
pub use admit::Retained;
mod cells;
pub use cells::{CellWrite, CellWrites};
mod derive;
pub use derive::Replay;
mod lead;
mod computed;
pub use computed::StaleComputed;
mod state_words;
pub use state_words::StateWords;
#[doc(hidden)]
pub use state_words::StateWordArena;
mod history;
pub use history::{Anchors, Checkpoint, CheckpointRef, History, ANCHOR_SPACING};
#[doc(hidden)]
pub use history::ANCHOR_CHUNK_POINTS;

#[cfg(test)]
pub(crate) use derive::NoReplay;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_nan_fact;
#[cfg(test)]
mod tests_cells;
#[cfg(test)]
mod tests_history;
#[cfg(test)]
mod tests_derive;
