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
    /// Class 3, kept by the directive's continuation state (Own or Chain).
    Carried {
        /// Frame rows the continuation reads before a new row, beyond its carried tails.
        operands: usize,
    },
    /// Class 3 with a recursive command that has no continuation: a frame that does not
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
    /// warm-up), when [`raw`](Self::raw) holds every raw row of it instead.
    pub state: Option<Vec<f64>>,
    /// Raw frame columns from the history's start while the continuation is warming
    /// up. The directive owns their meaning; core only stores them.
    pub raw: Vec<(String, Column)>,
}

impl DataFrame {
    /// The checkpoint of cached column `name`'s history, if it keeps one.
    pub fn computed_checkpoint(&self, name: &str) -> Option<&Checkpoint> {
        self.computed.get(name)?.checkpoint.as_ref()
    }

    /// Keep `checkpoint` as cached column `name`'s history — or, when none could be
    /// formed, the error that prevented it, if any (`Err`). A checkpoint covers rows before
    /// the ones the frame keeps after the coming drop, so it is set before it. No-op for an
    /// untracked column.
    pub fn set_computed_checkpoint(&mut self, name: &str, checkpoint: std::result::Result<Checkpoint, Option<VolasError>>) {
        if let Some(meta) = self.computed.get_mut(name) {
            (meta.checkpoint, meta.lost) = match checkpoint {
                Ok(cp) => (Some(cp), None),
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
