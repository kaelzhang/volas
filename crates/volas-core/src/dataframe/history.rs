//! What a cached directive column's values depend on before its own row, and the
//! [`Checkpoint`] a frame keeps of a recursion's history once it no longer holds the
//! rows the history started on (a bounded window that dropped its head).
//!
//! A column whose values depend on a trailing window only ([`History::Window`]) is
//! recomputed exactly by any frame holding that window. A recursion depends on every row
//! before: a frame that dropped the rows its history started on recomputes it only from a
//! checkpoint — the state as of a row before the rows it keeps, plus the inputs a resume
//! reads before its first row — and **replays** forward from it. The checkpoint's
//! content is produced and consumed by the directive layer (it knows what a command
//! reads); this module owns where it lives and how it moves with the column.

use crate::column::Column;
use crate::error::{Result, VolasError};

use super::DataFrame;

/// How a cached column's values depend on the rows before them (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum History {
    /// A trailing window of [`Reach::lookback`](super::Reach::lookback) rows only: no
    /// recursion anywhere, so any frame holding those rows computes a value exactly.
    Window,
    /// Every row before, through the carried state of its recursive top command alone,
    /// whose `@` operands are history-independent and read `operands` rows before a row:
    /// a frame that dropped its history's head keeps it exactly with a [`Checkpoint`].
    Carried {
        /// The rows an operand reads before its own (the largest lookback among them).
        operands: usize,
    },
    /// Every row before, in a form no carried state covers (a recursion inside an
    /// operand or a composite, or a recursive command without a carried state): a frame
    /// that does not hold the whole history cannot recompute it.
    Unkept,
}

impl History {
    /// Whether a value depends on every row before it (a recursion anywhere).
    pub fn recursive(self) -> bool {
        self != History::Window
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

impl DataFrame {
    /// The checkpoint of cached column `name`'s history, if it keeps one.
    pub fn computed_checkpoint(&self, name: &str) -> Option<&Checkpoint> {
        self.computed.get(name)?.checkpoint.as_ref()
    }

    /// Keep `checkpoint` as cached column `name`'s history (`None`: it keeps none). A
    /// checkpoint covers rows before the ones the frame keeps after the coming drop, so it
    /// is set before it. No-op for an untracked column.
    pub fn set_computed_checkpoint(&mut self, name: &str, checkpoint: Option<Checkpoint>) {
        if let Some(meta) = self.computed.get_mut(name) {
            meta.checkpoint = checkpoint;
        }
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
        meta.valid_rows = height;
        meta.state_at = origin + height - 1;
        meta.state_before_last = before_last.filter(|_| height > 1 && state.is_some());
        meta.state = state;
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }
}
