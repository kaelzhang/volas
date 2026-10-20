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

use super::{ComputedMeta, DataFrame, StateWords};

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
/// Batch points share one allocation; looking one up does not allocate or copy words.
/// A frame derivation bounds its retained point and words through this owner.
#[derive(Clone)]
pub struct CheckpointRef {
    points: Arc<CheckpointBank>,
    slot: usize,
    cached_position: usize,
}

#[derive(Debug)]
enum CheckpointBank {
    One(Checkpoint),
    Many { points: Vec<Checkpoint>, words: usize },
}

impl Checkpoint {
    fn words(&self) -> usize {
        self.raw.iter().fold(self.state.as_ref().map_or(0, |state| state.len()),
            |words, (_, column)| words.saturating_add(column.len()))
    }
}

impl CheckpointBank {
    fn point(&self, slot: usize) -> &Checkpoint {
        match self { Self::One(point) => point, Self::Many { points, .. } => &points[slot] }
    }
    fn slice(&self) -> &[Checkpoint] {
        match self { Self::One(point) => std::slice::from_ref(point), Self::Many { points, .. } => points }
    }
    fn words(&self) -> usize {
        match self { Self::One(point) => point.words(), Self::Many { words, .. } => *words }
    }

}

impl CheckpointRef {
    /// Immutable row metadata, usable without touching the cold checkpoint bank.
    pub(super) fn position(&self) -> usize { self.cached_position }

    /// Obtain an owned checkpoint, sharing its opaque state words.
    pub fn into_owned(self) -> Checkpoint {
        match Arc::try_unwrap(self.points) {
            Ok(CheckpointBank::One(point)) => point,
            Ok(CheckpointBank::Many { mut points, .. }) => points.swap_remove(self.slot),
            Err(points) => match points.as_ref() {
                CheckpointBank::One(point) => point.clone(),
                CheckpointBank::Many { points, .. } => points[self.slot].clone(),
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

#[cfg(test)]
thread_local! { static BANK_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

impl std::ops::Deref for CheckpointRef {
    type Target = Checkpoint;
    fn deref(&self) -> &Checkpoint {
        #[cfg(test)]
        BANK_READS.with(|reads| reads.set(reads.get() + 1));
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
/// Maximum checkpoints retained by one shared bank.
pub const ANCHOR_CHUNK_POINTS: usize = 128;
const ANCHOR_CHUNK_WORDS: usize = 2 * super::StateWordArena::WORDS;

/// Checkpoints of a recursion's history at regular rows (absolute rows `k ·
/// ANCHOR_SPACING - 1`), formed by the computation or the replay that passes them and kept
/// with the column: a later replay of its history to a row starts at the nearest one
/// before it, so a selection's boundary costs at most the spacing however long the
/// history is. A memo of the history's inputs: a write to them drops every anchor;
/// anchors are never formed at a frame's last row, the one a re-sent row replaces.
///
/// A frame copy or row selection owns only the anchors on its rows, before its
/// last valid row, sharing only the immutable chunks intersecting that range.
/// Anchor handles of that same frame share its later publications.
/// Its boundary checkpoint separately owns the history before its first row.
#[derive(Debug, Default)]
pub struct Anchors {
    held: std::sync::OnceLock<Arc<std::sync::Mutex<AnchorStore>>>,
    frozen: Option<AnchorSnapshot>,
}

/// Immutable batches have one shared bank, not one Arc handle per point. The
/// grid bound and chunk ends are published together under the store lock.
#[derive(Clone, Debug)]
struct AnchorChunk {
    points: Arc<CheckpointBank>,
    start: usize,
    end: usize,
}

/// Dense and sparse histories have the same chunk ownership. Only their row
/// lookup differs: a proved regular grid uses arithmetic; sparse rows use binary search.
#[derive(Clone, Debug, Default)]
struct AnchorStore {
    grid: Option<usize>,
    chunks: Vec<AnchorChunk>,
    len: usize,
    /// Points per full chunk; zero when the layout is not proved uniform.
    stride: u8,
}

impl AnchorStore {
    fn index(&self) -> AnchorIndex<'_> { AnchorIndex { grid: self.grid, chunks: &self.chunks, len: self.len, stride: self.stride } }
    fn selected(&self, slot: usize) -> CheckpointRef { self.index().selected(slot) }
    fn position(&self, slot: usize) -> usize { self.index().position(slot) }
    fn before(&self, at: usize) -> usize { self.index().before(at) }

    fn append(&mut self, point: CheckpointRef) {
        let first = point.position();
        self.grid = if self.len == 0 {
            (first % ANCHOR_SPACING == ANCHOR_SPACING - 1).then_some(first)
        } else {
            self.grid.filter(|old| self.len.checked_mul(ANCHOR_SPACING).and_then(|d| old.checked_add(d)) == Some(first))
        };
        let base = self.chunks.len().checked_sub(2).map_or(0, |i| self.chunks[i].end);
        if let Some(last) = self.chunks.last_mut().filter(|last| last.start == 0) {
            if let Some(bank) = Arc::get_mut(&mut last.points) {
                let physical = bank.slice().len();
                let weight = point.words();
                if physical == last.end - base && physical < ANCHOR_CHUNK_POINTS
                    && bank.words().saturating_add(weight) <= ANCHOR_CHUNK_WORDS
                {
                    if let CheckpointBank::One(_) = bank {
                        let words = bank.words();
                        let capacity = (ANCHOR_CHUNK_WORDS / words.max(1)).clamp(2, ANCHOR_CHUNK_POINTS);
                        let old = std::mem::replace(bank, CheckpointBank::Many { points: Vec::with_capacity(capacity), words });
                        if let (CheckpointBank::One(old), CheckpointBank::Many { points, .. }) = (old, &mut *bank) { points.push(old); }
                    }
                    if let CheckpointBank::Many { points, words } = bank {
                        *words = words.saturating_add(weight);
                        points.push(point.into_owned());
                    }
                    if physical + 1 > self.stride as usize { self.stride = 0; }
                    self.len += 1;
                    last.end = self.len;
                    return;
                }
            }
        }
        if self.stride != 0 && !self.len.is_multiple_of(self.stride as usize) { self.stride = 0; }
        self.len += 1;
        self.chunks.push(AnchorChunk { points: point.points, start: point.slot, end: self.len });
    }

    fn from_points(points: Vec<Checkpoint>) -> Self {
        let grid = dense_grid(&points);
        let len = points.len();
        let mut chunks = Vec::new();
        let mut source = points.into_iter().peekable();
        let mut end = 0;
        while let Some(first) = source.next() {
            let mut words = first.words();
            let remaining = source.len() + 1;
            let (bank, count) = if remaining == 1 || words > ANCHOR_CHUNK_WORDS {
                (CheckpointBank::One(first), 1)
            } else {
                let capacity = (ANCHOR_CHUNK_WORDS / words.max(1)).clamp(1, ANCHOR_CHUNK_POINTS).min(remaining);
                let mut points = Vec::with_capacity(capacity);
                points.push(first);
                while points.len() < ANCHOR_CHUNK_POINTS {
                    let Some(next) = source.peek() else { break; };
                    let weight = next.words();
                    if words.saturating_add(weight) > ANCHOR_CHUNK_WORDS { break; }
                    words = words.saturating_add(weight);
                    points.push(source.next().expect("peeked"));
                }
                let count = points.len();
                (CheckpointBank::Many { points, words }, count)
            };
            end += count;
            chunks.push(AnchorChunk { points: Arc::new(bank), start: 0, end });
        }
        // Inspect only this newly published pass. Every non-tail chunk must
        // have the same span before an index may replace search with division.
        let width = chunks.first().map_or(0, |chunk| chunk.end);
        let regular = chunks.iter().enumerate().all(|(i, chunk)| chunk.end == ((i + 1) * width).min(len));
        let stride = if regular { width as u8 } else { 0 };
        Self { grid, chunks, len, stride }
    }

    /// Re-index an overlapping sparse merge without copying checkpoints or words.
    fn from_handles(points: Vec<CheckpointRef>) -> Self {
        let mut store = Self::default();
        for point in points {
            let first = point.position();
            store.grid = if store.len == 0 {
                (first % ANCHOR_SPACING == ANCHOR_SPACING - 1).then_some(first)
            } else { store.grid.filter(|old| store.len.checked_mul(ANCHOR_SPACING).and_then(|d| old.checked_add(d)) == Some(first)) };
            let base = store.chunks.len().checked_sub(2).map_or(0, |i| store.chunks[i].end);
            if let Some(last) = store.chunks.last_mut() {
                if Arc::ptr_eq(&last.points, &point.points) && last.start + last.end - base == point.slot {
                    store.len += 1;
                    last.end = store.len;
                    continue;
                }
            }
            store.len += 1;
            store.chunks.push(AnchorChunk { points: point.points, start: point.slot, end: store.len });
        }
        store
    }

    fn extend(&mut self, mut incoming: Self) {
        if self.len == 0 { *self = incoming; return; }
        if self.stride != 0 && (!self.len.is_multiple_of(self.stride as usize)
            || incoming.stride != self.stride && !(incoming.chunks.len() == 1 && incoming.len <= self.stride as usize))
        { self.stride = 0; }
        self.grid = self.grid.filter(|first| self.len.checked_mul(ANCHOR_SPACING).and_then(|d| first.checked_add(d)).is_some_and(|next| incoming.grid == Some(next)));
        for chunk in &mut incoming.chunks { chunk.end += self.len; }
        self.len += incoming.len;
        self.chunks.append(&mut incoming.chunks);
    }

}

/// A single-bank selection needs only an Arc and two checked byte offsets.
/// Larger immutable indices stay behind one Arc, never inside every cached meta.
#[derive(Clone, Debug)]
struct BankRange {
    points: Arc<CheckpointBank>,
    start: u8,
    len: u8,
}

impl BankRange {
    fn points(&self) -> &[Checkpoint] {
        &self.points.slice()[self.start as usize..self.start as usize + self.len as usize]
    }
}

#[derive(Clone, Debug)]
enum AnchorSnapshot {
    Dense(BankRange),
    Sparse(BankRange),
    Many(Arc<AnchorStore>),
}

impl AnchorSnapshot {
    fn one(dense: bool, points: Arc<CheckpointBank>, start: usize, len: usize) -> Self {
        debug_assert!(start + len <= ANCHOR_CHUNK_POINTS && len > 0);
        let range = BankRange { points, start: start as u8, len: len as u8 };
        if dense { Self::Dense(range) } else { Self::Sparse(range) }
    }

    fn from_store(mut store: AnchorStore) -> Self {
        if store.chunks.len() == 1 {
            let chunk = store.chunks.pop().expect("one chunk");
            Self::one(store.grid.is_some(), chunk.points, chunk.start, store.len)
        } else { Self::Many(Arc::new(store)) }
    }

    fn len(&self) -> usize {
        match self { Self::Dense(r) | Self::Sparse(r) => r.len as usize, Self::Many(store) => store.len }
    }
    fn position(&self, slot: usize) -> usize {
        match self {
            Self::Dense(r) => r.points()[0].at + slot * ANCHOR_SPACING,
            Self::Sparse(r) => r.points()[slot].at,
            Self::Many(store) => store.position(slot),
        }
    }
    fn before(&self, at: usize) -> usize {
        match self {
            Self::Dense(r) => at.saturating_sub(r.points()[0].at).div_ceil(ANCHOR_SPACING).min(r.len as usize),
            Self::Sparse(r) => r.points().partition_point(|point| point.at < at),
            Self::Many(store) => store.before(at),
        }
    }
    fn nearest(&self, at: usize) -> Option<CheckpointRef> {
        if let Self::Many(store) = self { return store.index().nearest(at); }
        let after = if at == usize::MAX { self.len() } else { self.before(at + 1) };
        let slot = after.checked_sub(1)?;
        let (Self::Dense(range) | Self::Sparse(range)) = self else { unreachable!() };
        Some(CheckpointRef { points: Arc::clone(&range.points), slot: range.start as usize + slot, cached_position: self.position(slot) })
    }
    fn bounded(&self, first: usize, upto: usize) -> Option<Self> {
        if let Self::Many(store) = self {
            return if first <= store.position(0) && upto > store.position(store.len - 1) {
                Some(self.clone())
            } else { store.index().range(first, upto) };
        }
        let start = self.before(first);
        let end = self.before(upto).max(start);
        if start == end { return None; }
        let (Self::Dense(range) | Self::Sparse(range)) = self else { unreachable!() };
        let range = BankRange { points: Arc::clone(&range.points), start: (range.start as usize + start) as u8, len: (end - start) as u8 };
        Some(if matches!(self, Self::Dense(_)) { Self::Dense(range) } else { Self::Sparse(range) })
    }
    fn into_store(self) -> AnchorStore {
        let dense = matches!(&self, Self::Dense(_));
        match self {
            Self::Many(store) => Arc::unwrap_or_clone(store),
            Self::Dense(range) | Self::Sparse(range) => {
                let first = range.points()[0].at;
                let len = range.len as usize;
                let grid = dense.then_some(first);
                AnchorStore { grid, len, stride: 0, chunks: vec![AnchorChunk { points: range.points, start: range.start as usize, end: len }] }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct AnchorIndex<'a> {
    grid: Option<usize>,
    chunks: &'a [AnchorChunk],
    len: usize,
    /// Points per full chunk; zero when the layout is not proved uniform.
    stride: u8,
}

#[cfg(test)]
thread_local! { static TAIL_INDEX_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

impl AnchorIndex<'_> {
    fn locate(&self, slot: usize) -> (&AnchorChunk, usize) {
        // A proved uniform span identifies the bank without reading an unrelated
        // tail descriptor. Irregular live tails keep their constant-time check.
        if self.stride != 0 {
            let at = slot / self.stride as usize;
            let chunk = &self.chunks[at];
            return (chunk, chunk.start + slot - at * self.stride as usize);
        }
        // Live tails read the newest checkpoint. Chunking must not turn that
        // lookup into a search through every earlier publication chunk.
        #[cfg(test)]
        TAIL_INDEX_READS.with(|reads| reads.set(reads.get() + 1));
        let last = self.chunks.len() - 1;
        let base = last.checked_sub(1).map_or(0, |i| self.chunks[i].end);
        if slot >= base {
            let chunk = &self.chunks[last];
            return (chunk, chunk.start + slot - base);
        }
        let at = self.chunks.partition_point(|chunk| chunk.end <= slot);
        let base = at.checked_sub(1).map_or(0, |i| self.chunks[i].end);
        let chunk = &self.chunks[at];
        (chunk, chunk.start + slot - base)
    }

    #[inline]
    fn position(&self, slot: usize) -> usize {
        if let Some(first) = self.grid { return first + slot * ANCHOR_SPACING; }
        self.sparse_position(slot)
    }

    #[cold]
    #[inline(never)]
    fn sparse_position(&self, slot: usize) -> usize {
        let (chunk, slot) = self.locate(slot);
        chunk.points.point(slot).at
    }

    #[inline]
    fn selected(&self, slot: usize) -> CheckpointRef {
        let (chunk, local) = self.locate(slot);
        CheckpointRef { points: Arc::clone(&chunk.points), slot: local, cached_position: self.position(slot) }
    }

    fn before(&self, at: usize) -> usize {
        if let Some(first) = self.grid {
            return at.saturating_sub(first).div_ceil(ANCHOR_SPACING).min(self.len);
        }
        let (mut low, mut high) = (0, self.len);
        while low < high {
            let mid = low + (high - low) / 2;
            if self.position(mid) < at { low = mid + 1; } else { high = mid; }
        }
        low
    }

    fn range(&self, first: usize, upto: usize) -> Option<AnchorSnapshot> {
        let start = self.before(first);
        let end = self.before(upto).max(start);
        if start == end { return None; }
        Some(self.range_slots(start, end))
    }

    fn range_slots(&self, start: usize, end: usize) -> AnchorSnapshot {
        debug_assert!(start < end);
        let (begin, finish) = if self.stride != 0 {
            (start / self.stride as usize, (end - 1) / self.stride as usize + 1)
        } else {
            (self.chunks.partition_point(|chunk| chunk.end <= start), self.chunks.partition_point(|chunk| chunk.end < end) + 1)
        };
        let mut base = begin.checked_sub(1).map_or(0, |i| self.chunks[i].end);
        if finish == begin + 1 {
            let chunk = &self.chunks[begin];
            return AnchorSnapshot::one(self.grid.is_some(), Arc::clone(&chunk.points),
                chunk.start + start - base, end - start);
        }
        let chunks = self.chunks[begin..finish].iter().map(|chunk| {
            let selected = AnchorChunk { points: Arc::clone(&chunk.points),
                start: chunk.start + start.saturating_sub(base), end: chunk.end.min(end) - start };
            base = chunk.end;
            selected
        }).collect();
        AnchorSnapshot::from_store(AnchorStore {
            grid: self.grid.map(|first| first + start * ANCHOR_SPACING), chunks, len: end - start,
            stride: if self.stride != 0 && start.is_multiple_of(self.stride as usize) { self.stride } else { 0 },
        })
    }

    fn nearest(&self, at: usize) -> Option<CheckpointRef> {
        let after = if at == usize::MAX { self.len } else { self.before(at + 1) };
        after.checked_sub(1).map(|slot| self.selected(slot))
    }
}

/// Combining the immutable selection and later publications is normally an
/// ordered concatenation of chunks. Only overlapping sparse public writes merge.
#[inline]
fn combine(a: Option<AnchorSnapshot>, b: Option<AnchorSnapshot>) -> Option<AnchorSnapshot> {
    let (a, b) = match (a, b) { (Some(a), Some(b)) => (a, b), (a, b) => return a.or(b) };
    Some(merge_snapshots(a, b))
}

#[cold]
#[inline(never)]
fn merge_snapshots(a: AnchorSnapshot, b: AnchorSnapshot) -> AnchorSnapshot {
    let mut left = a.into_store();
    let right = b.into_store();
    if left.position(left.len - 1) < right.position(0) {
        left.extend(right);
    } else {
        let mut points: Vec<_> = (0..left.len).map(|slot| left.selected(slot)).chain((0..right.len).map(|slot| right.selected(slot))).collect();
        points.sort_by_key(CheckpointRef::position);
        points.dedup_by_key(|point| point.position());
        left = AnchorStore::from_handles(points);
    }
    AnchorSnapshot::from_store(left)
}

fn dense_grid(points: &[Checkpoint]) -> Option<usize> {
    let first = points.first()?.at;
    (first % ANCHOR_SPACING == ANCHOR_SPACING - 1
        && points.windows(2).all(|pair| pair[0].at.checked_add(ANCHOR_SPACING) == Some(pair[1].at)))
        .then_some(first)
}

impl Clone for Anchors {
    fn clone(&self) -> Self {
        // A handle obtained before first publication must still observe it. Keep
        // this memo shared for copies; empty row derivations need no allocation.
        Self { held: std::sync::OnceLock::from(Arc::clone(self.held.get_or_init(Arc::default))), frozen: self.frozen.clone() }
    }
}

impl Anchors {
    /// The anchor at the latest row at or before absolute row `at`.
    pub fn nearest(&self, at: usize) -> Option<CheckpointRef> {
        let frozen = self.frozen.as_ref().and_then(|snapshot| snapshot.nearest(at));
        let Some(store) = self.held.get() else { return frozen; };
        let held = store.lock().expect("anchors lock");
        let published = held.index().nearest(at);
        match (frozen, published) {
            (Some(a), Some(b)) => Some(if a.position() >= b.position() { a } else { b }),
            (a, b) => a.or(b),
        }
    }

    /// Keep a checkpoint. Ordered live appends touch only the open tail chunk.
    pub fn keep(&self, checkpoint: impl Into<CheckpointRef>) {
        let point = checkpoint.into();
        if self.frozen_contains(point.position()) { return; }
        let mut held = self.held.get_or_init(Arc::default).lock().expect("anchors lock");
        let slot = held.before(point.position());
        if slot == held.len { held.append(point); return; }
        if held.position(slot) == point.position() { return; }
        let mut points: Vec<_> = (0..held.len).map(|slot| held.selected(slot)).collect();
        points.insert(slot, point);
        *held = AnchorStore::from_handles(points);
    }

    /// Publish one pass in bounded chunks. Regular and sparse ordered appends
    /// never scan old history; only overlapping writes merge their sorted points.
    pub fn keep_all(&self, checkpoints: Vec<Checkpoint>) {
        let mut incoming = checkpoints;
        if incoming.is_empty() { return; }
        if !incoming.windows(2).all(|pair| pair[0].at <= pair[1].at) { incoming.sort_by_key(|cp| cp.at); }
        incoming.dedup_by_key(|cp| cp.at);
        if self.frozen.as_ref().is_some_and(|snapshot| snapshot.position(snapshot.len() - 1) >= incoming[0].at) {
            incoming.retain(|point| !self.frozen_contains(point.at));
            if incoming.is_empty() { return; }
        }
        if incoming.len() == 1 { self.keep(incoming.pop().expect("one checkpoint")); return; }
        let incoming = AnchorStore::from_points(incoming);
        let mut held = self.held.get_or_init(Arc::default).lock().expect("anchors lock");
        if held.len == 0 || held.position(held.len - 1) < incoming.position(0) {
            held.extend(incoming);
            return;
        }
        let mut merged = Vec::with_capacity(held.len + incoming.len);
        let mut old = (0..held.len).peekable();
        let mut new = (0..incoming.len).peekable();
        while let (Some(&a), Some(&b)) = (old.peek(), new.peek()) {
            match held.position(a).cmp(&incoming.position(b)) {
                std::cmp::Ordering::Less => { merged.push(held.selected(a)); old.next(); }
                std::cmp::Ordering::Greater => { merged.push(incoming.selected(b)); new.next(); }
                std::cmp::Ordering::Equal => { merged.push(held.selected(a)); old.next(); new.next(); }
            }
        }
        merged.extend(old.map(|slot| held.selected(slot)));
        merged.extend(new.map(|slot| incoming.selected(slot)));
        *held = AnchorStore::from_handles(merged);
    }

    fn frozen_contains(&self, at: usize) -> bool {
        self.frozen.as_ref().is_some_and(|snapshot| {
            let slot = snapshot.before(at);
            slot < snapshot.len() && snapshot.position(slot) == at
        })
    }

    /// Select only intersecting immutable chunks. Publication stays lazy: a
    /// short selection needs neither a new mutex nor a heap chunk index.
    pub(super) fn bounded(&self, first: usize, upto: usize) -> Anchors {
        self.selection(first, upto, None).0
    }

    /// A row derivation reads its boundary and retained range from one publication
    /// snapshot, under one lock. It need not reopen the same history for each.
    #[inline(always)]
    pub(super) fn selection(&self, first: usize, upto: usize, at: Option<usize>) -> (Anchors, Option<CheckpointRef>) {
        let frozen = self.frozen.as_ref().and_then(|snapshot| snapshot.bounded(first, upto));
        let mut nearest = at.and_then(|at| self.frozen.as_ref().and_then(|snapshot| snapshot.nearest(at)));
        let published = self.held.get().and_then(|store| {
            let held = store.lock().expect("anchors lock");
            let index = held.index();
            let start = index.before(first);
            let boundary = at.and_then(|at| {
                if first.checked_sub(1) == Some(at) {
                    start.checked_sub(1).map(|slot| index.selected(slot))
                } else { index.nearest(at) }
            });
            if let Some(point) = boundary {
                if nearest.as_ref().is_none_or(|old| old.position() < point.position()) { nearest = Some(point); }
            }
            // No second position lookup when there is no anchor on the selected
            // rows. This is the usual short-tail / short-interior derivation.
            let end = if start == index.len || index.position(start) >= upto { start }
                else { index.before(upto).max(start) };
            (start < end).then(|| index.range_slots(start, end))
        });
        (Anchors { held: Default::default(), frozen: combine(frozen, published) }, nearest)
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        let frozen = self.frozen.as_ref().map_or(0, |snapshot| snapshot.len());
        let Some(store) = self.held.get() else { return frozen; };
        let held = store.lock().expect("anchors lock");
        frozen + held.len
    }

}

impl PartialEq for Anchors {
    fn eq(&self, other: &Anchors) -> bool {
        match (self.held.get(), other.held.get()) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => std::ptr::eq(self, other),
            _ => false,
        }
    }
}

impl ComputedMeta {
    fn record_checkpoint(&mut self, checkpoint: std::result::Result<CheckpointRef, Option<VolasError>>) {
        (self.checkpoint, self.lost) = match checkpoint {
            Ok(cp) => (Some(cp), None), Err(why) => (None, why),
        };
    }

    fn with_anchors(&self, anchors: Anchors) -> Self {
        Self {
            directive: Arc::clone(&self.directive), reach: self.reach, valid_rows: self.valid_rows,
            state: self.state.clone(), state_before_last: self.state_before_last.clone(),
            since: self.since, state_at: self.state_at, scratch: self.scratch.clone(),
            checkpoint: self.checkpoint.clone(), pending: self.pending.clone(), anchors,
            lost: self.lost.clone(), first_present: self.first_present,
        }
    }

    /// Frame copies share today's immutable prefix and publish independently.
    pub(super) fn forked(&self, origin: usize) -> Self {
        let upto = origin.saturating_add(self.valid_rows).saturating_sub(1);
        self.with_anchors(self.anchors.bounded(0, upto))
    }
}

impl Clone for ComputedMeta {
    fn clone(&self) -> Self { self.with_anchors(self.anchors.clone()) }
}

impl DataFrame {
    /// The anchors of cached column `name`'s history ([`Anchors`]), if it is cached.
    pub fn computed_anchors(&self, name: &str) -> Option<Anchors> {
        self.computed_anchors_ref(name).cloned()
    }

    /// Borrow this frame's history owner for replay, without materializing a
    /// publication memo merely to read an immutable selection.
    #[doc(hidden)]
    pub fn computed_anchors_ref(&self, name: &str) -> Option<&Anchors> {
        Some(&self.computed.get(name)?.anchors)
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
        self.computed_checkpoint_ref(name).map(std::ops::Deref::deref)
    }

    /// Borrow the existing checkpoint owner; replay need not re-box its value.
    #[doc(hidden)]
    pub fn computed_checkpoint_ref(&self, name: &str) -> Option<&CheckpointRef> {
        self.computed.get(name)?.checkpoint.as_ref()
    }

    /// Keep `checkpoint` as cached column `name`'s history — or, when none could be
    /// formed, the error that prevented it, if any (`Err`). A checkpoint covers rows before
    /// the ones the frame keeps after the coming drop, so it is set before it. No-op for an
    /// untracked column.
    pub fn set_computed_checkpoint(&mut self, name: &str, checkpoint: std::result::Result<Checkpoint, Option<VolasError>>) {
        if let Some(meta) = self.computed.get_mut(name) {
            meta.record_checkpoint(checkpoint.map(CheckpointRef::from));
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
            meta.pending = checkpoint.map(CheckpointRef::from);
        }
    }

    /// The pending checkpoint of cached column `name` as of absolute row `at`, taken out,
    /// if it has one there; one as of another row stays.
    pub fn take_computed_pending(&mut self, name: &str, at: usize) -> Option<Checkpoint> {
        let meta = self.computed.get_mut(name)?;
        meta.pending.take_if(|p| p.position() == at).map(CheckpointRef::into_owned)
    }

    /// Promote the captured boundary at `at` into retained history.
    #[doc(hidden)]
    pub fn promote_computed_pending(&mut self, name: &str, at: usize) -> bool {
        let Some(meta) = self.computed.get_mut(name) else { return false; };
        let Some(cp) = meta.pending.take_if(|p| p.position() == at) else { return false; };
        meta.record_checkpoint(Ok(cp));
        true
    }

    /// Each cached column's name, history class and the absolute row its history starts
    /// at — what a bounded frame's drop of its head reads, without copying the states.
    pub fn computed_histories(&self) -> Vec<(String, History, usize)> {
        self.computed.iter().map(|(n, m)| (n.as_ref().to_owned(), m.reach.history, m.since)).collect()
    }

    /// Whether cached column `name` keeps a pending checkpoint as of absolute row `at`.
    pub fn has_computed_pending(&self, name: &str, at: usize) -> bool {
        self.computed.get(name).and_then(|m| m.pending.as_ref()).is_some_and(|p| p.position() == at)
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

#[cfg(test)]
mod chunk_index_tests {
    use super::*;
    fn point(i: usize, width: usize) -> Checkpoint {
        Checkpoint { at: (i + 1) * ANCHOR_SPACING - 1, state: Some(vec![i as f64; width].into()), raw: vec![] }
    }

    #[test]
    fn a_uniform_index_does_not_read_an_unrelated_tail_to_locate_a_bank() {
        let store = AnchorStore::from_points((0..300).map(|i| point(i, 1)).collect());
        assert_ne!(store.stride, 0);
        TAIL_INDEX_READS.with(|reads| reads.set(0));
        for slot in [0, 127, 128, 299] {
            assert_eq!(store.selected(slot).state.as_ref().unwrap()[0], slot as f64);
        }
        assert_eq!(TAIL_INDEX_READS.with(std::cell::Cell::get), 0,
            "the cached uniform span already identifies the target bank");
    }

    #[test]
    fn uniform_chunks_cache_their_span_and_shared_partial_tails_invalidate_it() {
        let mut store = AnchorStore::from_points((0..300).map(|i| point(i, 1)).collect());
        assert_eq!(store.stride as usize, ANCHOR_CHUNK_POINTS);
        assert_eq!(store.index().stride, store.stride);
        let held = store.selected(299);
        store.append(point(300, 1).into());
        assert_eq!(store.stride, 0);
        assert_eq!(held.state.as_ref().unwrap()[0], 299.);
        assert_eq!(store.selected(300).state.as_ref().unwrap()[0], 300.);
    }

    #[test]
    fn variable_widths_clipping_and_partial_batch_extensions_invalidate_the_span() {
        assert_eq!(AnchorStore::from_points(vec![]).stride, 0);
        let mixed = AnchorStore::from_points((0..300).map(|i| point(i, if i < 128 { 1 } else { 4097 })).collect());
        assert_eq!(mixed.stride, 0);
        let mut store = AnchorStore::from_points((0..256).map(|i| point(i, 1)).collect());
        store.extend(AnchorStore::from_points((256..300).map(|i| point(i, 1)).collect()));
        assert_eq!(store.stride as usize, ANCHOR_CHUNK_POINTS);
        let AnchorSnapshot::Many(clipped) = store.index().range_slots(96, 290) else { panic!("multiple chunks") };
        assert_eq!(clipped.stride, 0);
        assert_eq!(clipped.selected(0).state.as_ref().unwrap()[0], 96.);
        store.extend(AnchorStore::from_points((300..344).map(|i| point(i, 1)).collect()));
        assert_eq!(store.stride, 0);
        assert_eq!(store.selected(343).state.as_ref().unwrap()[0], 343.);
    }
}

#[cfg(test)]
mod storage_edge_tests {
    use super::*;

    fn point(i: usize, width: usize, sparse: bool) -> Checkpoint {
        Checkpoint { at: if sparse { i * 513 + 11 } else { (i + 1) * ANCHOR_SPACING - 1 },
            state: Some(vec![i as f64; width].into()), raw: vec![] }
    }

    #[test]
    fn an_unshared_full_tail_opens_a_new_bank_at_both_capacity_limits() {
        for (count, width) in [(ANCHOR_CHUNK_POINTS, 1), (6, 1200)] {
            let mut store = AnchorStore::from_points((0..count).map(|i| point(i, width, false)).collect());
            let old = std::ptr::from_ref(store.chunks[0].points.point(count - 1));
            store.append(point(count, width, false).into());
            assert_eq!(store.chunks.len(), 2);
            assert_eq!(std::ptr::from_ref(&*store.selected(count - 1)), old);
            assert_eq!(store.selected(count).state.as_ref().unwrap()[0], count as f64);
        }
    }

    #[test]
    fn collapsing_a_single_bank_index_preserves_its_offset_and_identity() {
        for sparse in [false, true] {
            let parent = AnchorStore::from_points((0..8).map(|i| point(i, 1, sparse)).collect());
            let first = parent.selected(3);
            let selected = AnchorStore::from_handles((3..5).map(|i| parent.selected(i)).collect());
            let snapshot = AnchorSnapshot::from_store(selected);
            assert_eq!(snapshot.len(), 2);
            assert!(snapshot.nearest(first.position() - 1).is_none());
            assert!(std::ptr::eq(&*snapshot.nearest(first.position()).unwrap(), &*first));
            assert_eq!(snapshot.nearest(usize::MAX).unwrap().position(), parent.position(4));
        }
    }
}

#[cfg(test)]
mod pending_read_cost_tests {
    use super::*;

    #[test]
    fn testing_a_pending_position_never_reads_its_cold_bank() {
        let mut frame = DataFrame::new(vec!["ema".into()], vec![Column::f64(vec![1.0; 4])], None).unwrap();
        frame.set_computed("ema", Arc::new(()), super::super::Reach {
            lookback: 2, resume: 0, continuation: 2, history: History::Carried { operands: 0 },
        });
        frame.set_computed_pending("ema", Some(Checkpoint { at: 2, state: Some(vec![42.0].into()), raw: vec![] }));
        BANK_READS.with(|reads| reads.set(0));
        assert!(frame.has_computed_pending("ema", 2));
        assert!(!frame.has_computed_pending("ema", 3));
        assert!(!frame.has_computed_pending("missing", 2));
        let selected = frame.rows(0, 4, &super::super::derive::NoReplay);
        assert!(selected.has_computed_pending("ema", 2));
        frame.invalidate_computed_forming_row(3);
        assert!(frame.has_computed_pending("ema", 2));
        assert_eq!(BANK_READS.with(std::cell::Cell::get), 0, "pending positions belong to hot immutable metadata");
    }
}
