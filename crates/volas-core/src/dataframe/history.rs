//! What a cached directive column's values depend on before its own row, and the
//! [`Checkpoint`] a frame keeps of a recursion's history once it no longer holds the
//! rows the history started on (a bounded window that dropped its head).
//!
//! Every directive has one of three history classes, owned by the command spec: its
//! values depend on a trailing window (class 1), on a running accumulator over a trailing
//! window (class 2), or on every row before — a recursion, a cumulative line, an unbounded
//! count (class 3). A bounded window computes class 1 whenever its margin holds the
//! window, and classes 2 and 3 only when they are declared. A class-1 or class-2 column,
//! and a class-3 count whose values are a function of its window, are recomputed exactly
//! by any frame holding the window. A recursion depends on every row before: a frame
//! that dropped the rows its history started on recomputes it only from a checkpoint —
//! the state as of a row before the rows it keeps, plus the inputs a resume reads before
//! its first row — and **replays** forward from it. The checkpoint's content is produced
//! and consumed by the directive layer (it knows what a command reads); this module owns
//! where it lives and how it moves with the column.

use std::sync::Arc;

use crate::column::Column;
use crate::error::{Result, VolasError};

use super::{DataFrame, StateWords};

/// How a cached column's values depend on the rows before them: its history class and,
/// for class 3, how a frame keeps it (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum History {
    /// Class 1: a trailing window of [`Reach::lookback`](super::Reach::lookback) rows,
    /// no running accumulator.
    Window,
    /// Class 2: a running accumulator over a trailing window (its sums are exact, so a
    /// value is a function of the window).
    Accumulator,
    /// Class 3, a count over a trailing window (`increase`): declared only, and kept by
    /// recomputing its window.
    Counted,
    /// Class 3, kept by the directive's continuation state (Own or Chain).
    Carried {
        /// Frame rows the continuation reads before a new row, beyond its carried tails.
        operands: usize,
    },
}

impl History {
    /// The history class: 1, 2 or 3.
    pub fn class(self) -> u8 {
        match self {
            History::Window => 1,
            History::Accumulator => 2,
            History::Counted | History::Carried { .. } => 3,
        }
    }

    /// Whether a value depends on every row before it (a recursion anywhere): only a
    /// carried state or the whole history reproduces it.
    pub fn recursive(self) -> bool {
        matches!(self, History::Carried { .. })
    }
}

/// The history of a recursive cached column before a row a frame keeps: everything a
/// replay of the column from the row after [`at`](Self::at) reads besides the frame's
/// own rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    /// The absolute position of the last row the checkpoint covers; a replay continues
    /// from the row after it.
    pub at: usize,
    /// The column's carried state as of that row (as of its last complete row up to
    /// it); `None` while the history up to it is too short to define one (inside the
    /// warm-up), when [`raw`](Self::raw) holds every raw row of it instead.
    pub state: Option<StateWords>,
    /// Raw frame columns from the history's start while the continuation is warming
    /// up. The directive owns their meaning; core only stores them.
    pub raw: Vec<(String, Column)>,
}

/// A shared immutable checkpoint, either a single point or one point of a batch.
/// Batch points share one allocation; selecting one does not allocate or copy words.
#[derive(Clone)]
pub struct CheckpointRef {
    points: Arc<CheckpointBank>,
    slot: usize,
    cached_position: usize,
}

#[derive(Debug)]
enum CheckpointBank {
    One(Checkpoint),
    Many(Vec<Checkpoint>),
}

impl CheckpointBank {
    fn point(&self, slot: usize) -> &Checkpoint {
        match self { Self::One(point) => point, Self::Many(points) => &points[slot] }
    }
}

impl CheckpointRef {
    /// Immutable row metadata, usable without touching the cold checkpoint bank.
    pub(super) fn position(&self) -> usize { self.cached_position }

    /// Obtain an owned checkpoint, sharing its opaque state words.
    pub fn into_owned(self) -> Checkpoint {
        match Arc::try_unwrap(self.points) {
            Ok(CheckpointBank::One(point)) => point,
            Ok(CheckpointBank::Many(mut points)) => points.swap_remove(self.slot),
            Err(points) => match points.as_ref() {
                CheckpointBank::One(point) => point.clone(),
                CheckpointBank::Many(points) => points[self.slot].clone(),
            },
        }
    }
}

impl From<Checkpoint> for CheckpointRef {
    fn from(point: Checkpoint) -> Self {
        Self { cached_position: point.at, points: Arc::new(CheckpointBank::One(point)), slot: 0 }
    }
}

impl From<Arc<Checkpoint>> for CheckpointRef {
    fn from(point: Arc<Checkpoint>) -> Self {
        Self::from(Arc::unwrap_or_clone(point))
    }
}

impl std::ops::Deref for CheckpointRef {
    type Target = Checkpoint;
    fn deref(&self) -> &Checkpoint {
        self.points.point(self.slot)
    }
}

impl std::fmt::Debug for CheckpointRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::ops::Deref::deref(self).fmt(f)
    }
}

impl PartialEq for CheckpointRef {
    fn eq(&self, other: &Self) -> bool {
        std::ops::Deref::deref(self) == std::ops::Deref::deref(other)
    }
}

/// The spacing of a recursion's [`Anchors`]: one every this many rows.
pub const ANCHOR_SPACING: usize = 512;

/// Checkpoints of a recursion's history at regular rows (absolute rows `k ·
/// ANCHOR_SPACING - 1`), formed by the computation or the replay that passes them and kept
/// with the column: a later replay of its history to a row starts at the nearest one
/// before it, so a selection's boundary costs at most the spacing however long the
/// history is. A memo of the history's inputs: a write to them drops every anchor;
/// anchors are never formed at a frame's last row, the one a re-sent row replaces.
///
/// Frames derived from one another share one store while their histories agree: a copy
/// shares all of it, a selection the rows before its last — it
/// never reads or forms an anchor past them, where its rows may come to differ.
#[derive(Clone, Debug)]
pub struct Anchors {
    held: Arc<std::sync::Mutex<AnchorStore>>,
    /// The rows of the store this frame's history agrees with: those before this one.
    upto: usize,
}

/// Immutable batches have one shared bank, not one Arc handle per point. The
/// grid bound and chunk ends are published together under the store lock.
#[derive(Debug)]
struct AnchorChunk {
    points: Arc<CheckpointBank>,
    start: usize,
    end: usize,
}

#[derive(Debug)]
enum AnchorStore {
    Dense { first: usize, chunks: Vec<AnchorChunk>, len: usize },
    Sparse(Vec<CheckpointRef>),
}

impl Default for AnchorStore {
    fn default() -> Self { Self::Sparse(Vec::new()) }
}

impl AnchorStore {
    fn len(&self) -> usize {
        match self { Self::Dense { len, .. } => *len, Self::Sparse(points) => points.len() }
    }

    fn selected(&self, slot: usize) -> CheckpointRef {
        match self {
            Self::Dense { first, chunks, .. } => {
                let chunk_at = chunks.partition_point(|chunk| chunk.end <= slot);
                let base = chunk_at.checked_sub(1).map_or(0, |i| chunks[i].end);
                let chunk = &chunks[chunk_at];
                // A valid global slot is backed by an actual grid checkpoint:
                // publication proved this row exists, so its arithmetic cannot overflow.
                CheckpointRef { points: Arc::clone(&chunk.points), slot: chunk.start + slot - base,
                    cached_position: first + slot * ANCHOR_SPACING }
            }
            Self::Sparse(points) => points[slot].clone(),
        }
    }

    fn append(&mut self, point: CheckpointRef, count: usize) {
        if let Self::Dense { chunks, len, .. } = self {
            *len += count;
            chunks.push(AnchorChunk { points: point.points, start: point.slot, end: *len });
        }
    }

    /// Sparse or overlapping public writes alone materialize the per-point handles.
    /// Regular publication and live grid append never flatten previous batches.
    fn sparse(&mut self) -> &mut Vec<CheckpointRef> {
        if matches!(self, Self::Dense { .. }) {
            let points = (0..self.len()).map(|slot| self.selected(slot)).collect();
            *self = Self::Sparse(points);
        }
        match self { Self::Sparse(points) => points, Self::Dense { .. } => unreachable!("converted above") }
    }
}

fn dense_grid(points: &[Checkpoint]) -> Option<usize> {
    let first = points.first()?.at;
    (first % ANCHOR_SPACING == ANCHOR_SPACING - 1
        && points.windows(2).all(|pair| pair[0].at.checked_add(ANCHOR_SPACING) == Some(pair[1].at)))
        .then_some(first)
}

impl Default for Anchors {
    fn default() -> Anchors {
        Anchors { held: Arc::default(), upto: usize::MAX }
    }
}

impl Anchors {
    /// The anchor at the latest row at or before absolute row `at`.
    pub fn nearest(&self, at: usize) -> Option<CheckpointRef> {
        let at = at.min(self.upto.checked_sub(1)?);
        let held = self.held.lock().expect("anchors lock");
        match &*held {
            AnchorStore::Dense { first, len, .. } => {
                let slot = at.checked_sub(*first)? / ANCHOR_SPACING;
                Some(held.selected(slot.min(len - 1)))
            }
            AnchorStore::Sparse(points) => {
                let after = points.partition_point(|cp| cp.position() <= at);
                after.checked_sub(1).map(|i| held.selected(i))
            }
        }
    }

    /// Keep `checkpoint` as an anchor (one per row, in row order) — when its row is one
    /// this frame shares the store for.
    pub fn keep(&self, checkpoint: impl Into<CheckpointRef>) {
        let checkpoint = checkpoint.into();
        if checkpoint.position() >= self.upto {
            return;
        }
        let mut held = self.held.lock().expect("anchors lock");
        if held.len() == 0 && checkpoint.position() % ANCHOR_SPACING == ANCHOR_SPACING - 1 {
            *held = AnchorStore::Dense { first: checkpoint.position(), chunks: Vec::new(), len: 0 };
            held.append(checkpoint, 1);
            return;
        }
        if let AnchorStore::Dense { first, len, .. } = &*held {
            let distance = checkpoint.position().checked_sub(*first);
            if distance.is_some_and(|d| d % ANCHOR_SPACING == 0 && d / ANCHOR_SPACING < *len) {
                return; // a duplicate keeps the existing point
            }
            if len.checked_mul(ANCHOR_SPACING).and_then(|d| first.checked_add(d)) == Some(checkpoint.position()) {
                held.append(checkpoint, 1);
                return;
            }
        }
        let points = held.sparse();
        let at = points.partition_point(|cp| cp.position() < checkpoint.position());
        if points.get(at).is_none_or(|cp| cp.position() != checkpoint.position()) {
            points.insert(at, checkpoint);
        }
    }

    /// Keep checkpoints published by one computing pass under one lock. The normal
    /// ascending, unique batch replaces an empty store or appends directly; sparse,
    /// unordered and duplicate batches merge while preserving existing checkpoints.
    pub fn keep_all(&self, checkpoints: Vec<Checkpoint>) {
        let mut incoming: Vec<_> = checkpoints.into_iter().filter(|cp| cp.at < self.upto).collect();
        if incoming.is_empty() {
            return;
        }
        if !incoming.windows(2).all(|pair| pair[0].at <= pair[1].at) {
            incoming.sort_by_key(|cp| cp.at);
        }
        incoming.dedup_by_key(|cp| cp.at);
        let count = incoming.len();
        let grid = dense_grid(&incoming);
        let cached_position = incoming[0].at;
        let points = Arc::new(CheckpointBank::Many(incoming));
        let batch = CheckpointRef { points, slot: 0, cached_position };
        let mut held = self.held.lock().expect("anchors lock");
        if let Some(first) = grid {
            if held.len() == 0 {
                *held = AnchorStore::Dense { first, chunks: Vec::new(), len: 0 };
                held.append(batch, count);
                return;
            }
            if let AnchorStore::Dense { first: old, len, .. } = &*held {
                if len.checked_mul(ANCHOR_SPACING).and_then(|d| old.checked_add(d)) == Some(first) {
                    held.append(batch, count);
                    return;
                }
            }
        }
        let mut incoming: Vec<_> = (0..count)
            .map(|slot| {
                let cached_position = batch.points.point(slot).at;
                CheckpointRef { points: Arc::clone(&batch.points), slot, cached_position }
            }).collect();
        let held = held.sparse();
        if held.is_empty() {
            *held = incoming;
            return;
        }
        if held.last().is_some_and(|cp| cp.position() < incoming[0].position()) {
            held.append(&mut incoming);
            return;
        }
        let mut merged = Vec::with_capacity(held.len() + incoming.len());
        let mut old = std::mem::take(held).into_iter().peekable();
        let mut new = incoming.into_iter().peekable();
        while let (Some(a), Some(b)) = (old.peek(), new.peek()) {
            match a.position().cmp(&b.position()) {
                std::cmp::Ordering::Less => merged.push(old.next().expect("peeked")),
                std::cmp::Ordering::Greater => merged.push(new.next().expect("peeked")),
                std::cmp::Ordering::Equal => {
                    merged.push(old.next().expect("peeked"));
                    new.next();
                }
            }
        }
        merged.extend(old);
        merged.extend(new);
        *held = merged;
    }

    /// These anchors for a frame whose history agrees with this one's on the rows before
    /// absolute row `upto` only (a selection's).
    pub(super) fn upto(&self, upto: usize) -> Anchors {
        Anchors { held: Arc::clone(&self.held), upto: self.upto.min(upto) }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        let held = self.held.lock().expect("anchors lock");
        match &*held {
            AnchorStore::Dense { first, len, .. } => self.upto.checked_sub(1).and_then(|at| at.checked_sub(*first))
                .map_or(0, |d| (d / ANCHOR_SPACING + 1).min(*len)),
            AnchorStore::Sparse(points) => points.partition_point(|cp| cp.position() < self.upto),
        }
    }
}

impl PartialEq for Anchors {
    fn eq(&self, other: &Anchors) -> bool {
        Arc::ptr_eq(&self.held, &other.held) && self.upto == other.upto
    }
}

impl DataFrame {
    /// The anchors of cached column `name`'s history ([`Anchors`]), if it is cached.
    pub fn computed_anchors(&self, name: &str) -> Option<Anchors> {
        Some(self.computed.get(name)?.anchors.clone())
    }

    /// Keep `anchors` as anchors of cached column `name`'s history ([`Anchors`]): the
    /// checkpoints a computation over its whole history passed. No-op for an untracked
    /// column.
    pub fn keep_computed_anchors(&self, name: &str, anchors: Vec<Checkpoint>) {
        if let Some(meta) = self.computed.get(name) {
            meta.anchors.keep_all(anchors);
        }
    }

    /// The checkpoint of cached column `name`'s history, if it keeps one.
    pub fn computed_checkpoint(&self, name: &str) -> Option<&Checkpoint> {
        self.computed.get(name)?.checkpoint.as_deref()
    }

    /// Keep `checkpoint` as cached column `name`'s history — or, when none could be
    /// formed, the error that prevented it, if any (`Err`). A checkpoint covers rows before
    /// the ones the frame keeps after the coming drop, so it is set before it. No-op for an
    /// untracked column.
    pub fn set_computed_checkpoint(&mut self, name: &str, checkpoint: std::result::Result<Checkpoint, Option<VolasError>>) {
        if let Some(meta) = self.computed.get_mut(name) {
            (meta.checkpoint, meta.lost) = match checkpoint {
                Ok(cp) => (Some(CheckpointRef::from(cp)), None),
                Err(why) => (None, why),
            };
        }
    }

    /// Why cached column `name` keeps no checkpoint of its history, if an error prevented
    /// it.
    pub fn computed_history_lost(&self, name: &str) -> Option<&VolasError> {
        self.computed.get(name)?.lost.as_ref()
    }

    /// Keep `checkpoint` as the one cached column `name`'s history will need at the next
    /// drop of the head (see [`take_computed_pending`](Self::take_computed_pending)); a
    /// change to the rows it covers drops it. No-op for an untracked column.
    pub fn set_computed_pending(&mut self, name: &str, checkpoint: Option<Checkpoint>) {
        if let Some(meta) = self.computed.get_mut(name) {
            meta.pending = checkpoint;
        }
    }

    /// The pending checkpoint of cached column `name` as of absolute row `at`, taken out,
    /// if it has one there; one as of another row stays.
    pub fn take_computed_pending(&mut self, name: &str, at: usize) -> Option<Checkpoint> {
        let meta = self.computed.get_mut(name)?;
        meta.pending.take_if(|p| p.at == at)
    }

    /// Each cached column's name, history class and the absolute row its history starts
    /// at — what a bounded frame's drop of its head reads, without copying the states.
    pub fn computed_histories(&self) -> Vec<(String, History, usize)> {
        self.computed.iter().map(|(n, m)| (n.as_ref().to_owned(), m.reach.history, m.since)).collect()
    }

    /// Whether cached column `name` keeps a pending checkpoint as of absolute row `at`.
    pub fn has_computed_pending(&self, name: &str, at: usize) -> bool {
        self.computed.get(name).and_then(|m| m.pending.as_ref()).is_some_and(|p| p.at == at)
    }

    /// A replay continued cached column `name` from its checkpoint: rows `[from,
    /// from + tail.len())` — through the last row — take `tail`, the column is valid
    /// through the last row and carries `state` (as of the last row; none inside the
    /// warm-up) and `before_last` (as of the row before). The history still starts where it did: the rows before
    /// `from` belong to it and keep their values (a bounded frame's hidden history, which
    /// it never shows). An untracked column, or a tail that does not end on the last row,
    /// is an error and nothing is written.
    pub fn continue_computed(
        &mut self,
        name: &str,
        from: usize,
        tail: &Column,
        state: Option<Vec<f64>>,
        before_last: Option<Vec<f64>>,
    ) -> Result<()> {
        let pos = self
            .column_pos(name)
            .filter(|_| self.computed.contains_key(name))
            .ok_or_else(|| VolasError::ColumnNotFound(name.to_string()))?;
        if from.checked_add(tail.len()) != Some(self.height) {
            return Err(VolasError::Shape(format!(
                "a replay of \"{name}\" from row {from} wrote {} rows of {}",
                tail.len(),
                self.height
            )));
        }
        self.columns[pos].write_rows(from, tail)?;
        let origin = self.origin;
        let height = self.height;
        let meta = self.computed.get_mut(name).expect("tracked above");
        meta.note_written(&self.columns[pos], origin, from, height);
        meta.valid_rows = height;
        meta.state_at = origin + height - 1;
        meta.state_before_last = before_last.filter(|_| height > 1 && state.is_some()).map(Arc::from);
        meta.state = state.map(Arc::from);
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }
}
