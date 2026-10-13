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

use super::DataFrame;

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
    /// Class 3, a recursion carried by the state of its top command alone, whose `@`
    /// operands are of classes 1–2 or counts and read `operands` rows before a row: a
    /// frame that dropped its history's head keeps it exactly with a [`Checkpoint`].
    Carried {
        /// The rows an operand reads before its own (the largest lookback among them).
        operands: usize,
    },
    /// Class 3 in a form no carried state covers (a recursion inside an operand or a
    /// composite, or a recursive command without a carried state): a frame that does not
    /// hold the whole history cannot recompute it.
    Unkept,
}

impl History {
    /// The history class: 1, 2 or 3.
    pub fn class(self) -> u8 {
        match self {
            History::Window => 1,
            History::Accumulator => 2,
            History::Counted | History::Carried { .. } | History::Unkept => 3,
        }
    }

    /// Whether a value depends on every row before it (a recursion anywhere): only a
    /// carried state or the whole history reproduces it.
    pub fn recursive(self) -> bool {
        matches!(self, History::Carried { .. } | History::Unkept)
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
    /// warm-up), when [`inputs`](Self::inputs) hold every complete row of it instead.
    pub state: Option<Vec<f64>>,
    /// Each input slot the command reads, with its values on the last complete rows up
    /// to [`at`](Self::at), oldest first: as many as a resume reads before its first
    /// row (fewer when the history has fewer) — or, without a state, all of them.
    pub inputs: Vec<(usize, Vec<f64>)>,
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
    held: Arc<std::sync::Mutex<Vec<Arc<Checkpoint>>>>,
    /// The rows of the store this frame's history agrees with: those before this one.
    upto: usize,
}

impl Default for Anchors {
    fn default() -> Anchors {
        Anchors { held: Arc::default(), upto: usize::MAX }
    }
}

impl Anchors {
    /// The anchor at the latest row at or before absolute row `at`.
    pub fn nearest(&self, at: usize) -> Option<Arc<Checkpoint>> {
        let at = at.min(self.upto.checked_sub(1)?);
        let held = self.held.lock().expect("anchors lock");
        let after = held.partition_point(|cp| cp.at <= at); // held in row order
        after.checked_sub(1).map(|i| held[i].clone())
    }

    /// Keep `checkpoint` as an anchor (one per row, in row order) — when its row is one
    /// this frame shares the store for.
    pub fn keep(&self, checkpoint: impl Into<Arc<Checkpoint>>) {
        let checkpoint = checkpoint.into();
        if checkpoint.at >= self.upto {
            return;
        }
        let mut held = self.held.lock().expect("anchors lock");
        let at = held.partition_point(|cp| cp.at < checkpoint.at);
        if held.get(at).is_none_or(|cp| cp.at != checkpoint.at) {
            held.insert(at, checkpoint);
        }
    }

    /// These anchors for a frame whose history agrees with this one's on the rows before
    /// absolute row `upto` only (a selection's).
    pub(super) fn upto(&self, upto: usize) -> Anchors {
        Anchors { held: Arc::clone(&self.held), upto: self.upto.min(upto) }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        let held = self.held.lock().expect("anchors lock");
        held.partition_point(|cp| cp.at < self.upto)
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
            anchors.into_iter().for_each(|cp| meta.anchors.keep(cp));
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
                Ok(cp) => (Some(Arc::new(cp)), None),
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
        self.computed.iter().map(|(n, m)| (n.clone(), m.reach.history, m.since)).collect()
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
        meta.state_before_last = before_last.filter(|_| height > 1 && state.is_some());
        meta.state = state;
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }
}
