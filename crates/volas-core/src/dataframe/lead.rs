//! The **lead** of a frame: the data rows just before its row 0 that its cached columns
//! read to recompute their first rows. A contiguous selection keeps it (see `derive`):
//! each cached column whose history starts before the selection needs the fewer of its
//! [`Reach::lead`](super::Reach::lead) and its history's rows there — and a recursion the
//! rows after the point of its history it replays from: its checkpoint, or its whole
//! history when it keeps none (a recursion carried by its state whose start is near, or one
//! that carries no state, which it is computed over).
//!
//! The lead is history: it is never written. Its values keep the dtype they had; when a
//! data column is later replaced by one of another dtype, a recompute reads both in their
//! lossless common dtype (never rounded, never made missing), and a replacement whose
//! dtype cannot hold the lead's values exactly is refused before it is written.

use std::sync::Arc;

use crate::column::Column;
use crate::error::{Result, VolasError};
use crate::fxhash::FxHashMap;
use crate::index::Index;

use super::{ComputedMeta, DataFrame, History};

/// The rows `[from, to)` of the data columns of a frame's schema — the schema of the
/// frame it was taken for (`names` / `name_to_idx`, shared with it), one column per name,
/// `None` for a cached column's — whose row `to` would be the frame's row 0. A lead that
/// is most of its source's rows shares the source's buffers; a shorter one is a copy of
/// just its rows, so a lead never pins more than twice the rows it keeps.
#[derive(Clone, Debug)]
pub(super) struct Lead {
    names: Arc<Vec<String>>,
    name_to_idx: Arc<FxHashMap<String, usize>>,
    columns: Vec<Option<Column>>,
    /// The rows each held column has.
    height: usize,
    from: usize,
    to: usize,
}

/// `head` followed by `tail` in their lossless common dtype: the same dtype, or a wider
/// numeric one both convert to exactly (a float32 or an integer into float64, an int32
/// into int64). Anything else is a [`VolasError::DType`] — never a rounded or missing
/// value.
fn join(head: Column, tail: &Column) -> Result<Column> {
    let (a, b) = (head.dtype(), tail.dtype());
    let mut out = match a == b {
        true => head,
        false => {
            let numeric = |c: &Column| c.dtype().is_numeric() || matches!(c, Column::Bool(..));
            if !numeric(&head) || !numeric(tail) {
                return Err(VolasError::DType(format!("{a} and {b} values have no common dtype")));
            }
            head.convert_lossless(crate::numeric::binary_supertype(a, b))?
        }
    };
    let tail = tail.convert_lossless(out.dtype())?;
    out.append(&tail)?;
    Ok(out)
}

/// How many rows before absolute row `first` the cached columns `computed` read (see the
/// module docs).
fn wanted(computed: &FxHashMap<String, ComputedMeta>, first: usize) -> usize {
    computed
        .values()
        .filter(|m| m.since < first)
        .map(|m| match replays_from(m) {
            // the rows after the point it replays from, and what their recompute reads before
            Some(from) if from < first => (first - from + m.reach.lead()).min(first - m.since),
            _ => m.reach.lead().min(first - m.since),
        })
        .max()
        .unwrap_or(0)
}

/// The absolute row a recompute of recursion `m` replays from, when it is a recursion
/// that a frame not holding its history keeps by a point of it: the row after its
/// checkpoint — or its history's start, which a recursion carried by its state with no
/// checkpoint, or one carrying no state, is computed over.
fn replays_from(m: &ComputedMeta) -> Option<usize> {
    match (m.reach.history, &m.checkpoint) {
        (History::Carried { .. }, Some(cp)) => Some(cp.at + 1),
        (History::Carried { .. }, None) => Some(m.since),
        (History::Unkept, _) if m.state.is_none() => Some(m.since),
        _ => None,
    }
}

impl Lead {
    /// How many rows the lead keeps.
    pub(super) fn len(&self) -> usize {
        self.to - self.from
    }

    /// Column `name` as the lead holds it (all its rows), if it does.
    fn column(&self, name: &str) -> Option<&Column> {
        self.columns[*self.name_to_idx.get(name)?].as_ref()
    }

    /// The last `n` (at most [`len`](Self::len)) rows of column `name` as the lead holds
    /// them (in the dtype they had), or `n` missing cells of `dtype` when it does not
    /// hold the column (one added after it: no column whose history starts before the
    /// frame reads it).
    fn tail(&self, name: &str, n: usize, dtype: crate::DType) -> Column {
        match self.column(name) {
            Some(c) => c.slice(self.to - n, self.to),
            None => Column::na_of(dtype, n),
        }
    }

    /// This lead over the data columns of `frame` (a projection of the frame it was taken
    /// for): those it holds, in `frame`'s schema.
    pub(super) fn select(&self, frame: &DataFrame) -> Lead {
        let held = |n: &String| self.column(n).filter(|_| !frame.computed.contains_key(n)).cloned();
        let columns = frame.names.iter().map(held).collect();
        Lead { names: Arc::clone(&frame.names), name_to_idx: Arc::clone(&frame.name_to_idx), columns, ..*self }
    }

    /// This lead keeping its last `rows` rows only, copied out of the buffers they would
    /// no longer be most of; `None` for none.
    fn trimmed(&self, rows: usize) -> Option<Lead> {
        let rows = rows.min(self.len());
        if rows == 0 {
            return None;
        }
        let from = self.to - rows;
        if 2 * rows >= self.height {
            return Some(Lead { from, ..self.clone() });
        }
        let columns = self.columns.iter().map(|c| c.as_ref().map(|c| c.slice(from, self.to))).collect();
        Some(Lead { columns, height: rows, from: 0, to: rows, ..self.clone() })
    }

    #[cfg(debug_assertions)]
    pub(super) fn debug_check(&self) {
        assert_eq!(self.columns.len(), self.names.len(), "one lead column per name");
        assert!(self.columns.iter().flatten().all(|c| c.len() == self.height), "the lead's columns are alike");
        assert!(self.from <= self.to && self.to <= self.height, "the lead's rows are its columns'");
    }
}

impl DataFrame {
    /// The lead of the selection starting at row `start` whose cached columns are
    /// `computed`: the rows they read before it (see the module docs), taken from this
    /// frame's rows before `start`, then its own lead. `None` when no column reads
    /// before it.
    pub(super) fn lead_before(&self, start: usize, computed: &FxHashMap<String, ComputedMeta>) -> Option<Lead> {
        let first = self.origin + start;
        let wanted = wanted(computed, first);
        let own = wanted.min(start);
        let held = self.lead.as_ref().map_or(0, Lead::len);
        let older = (wanted - own).min(held);
        if own + older == 0 {
            return None;
        }
        let data = |i: usize| !computed.contains_key(&self.names[i]);
        let schema = (Arc::clone(&self.names), Arc::clone(&self.name_to_idx));
        if older == 0 && 2 * own >= self.height {
            // most of this frame's rows: share its buffers rather than copy them
            let columns = self.columns.iter().enumerate().map(|(i, c)| data(i).then(|| c.clone())).collect();
            let (names, name_to_idx) = schema;
            return Some(Lead { names, name_to_idx, columns, height: self.height, from: start - own, to: start });
        }
        let columns = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, col)| {
                data(i).then(|| {
                    let own_rows = col.slice(start - own, start);
                    match &self.lead {
                        Some(lead) if older > 0 => join(lead.tail(&self.names[i], older, col.dtype()), &own_rows)
                            .expect("checked when the column was written"),
                        _ => own_rows,
                    }
                })
            })
            .collect();
        let len = own + older;
        let (names, name_to_idx) = schema;
        Some(Lead { names, name_to_idx, columns, height: len, from: 0, to: len })
    }

    /// The lead trimmed to what the cached columns read now — after the set of cached
    /// columns, or what they carry, changed (a column dropped, a checkpoint formed): the
    /// rows no column reads any longer are released.
    pub(super) fn retrim_lead(&mut self) {
        if let Some(lead) = &self.lead {
            let wanted = wanted(&self.computed, self.origin);
            if wanted < lead.len() {
                self.lead = lead.trimmed(wanted);
            }
        }
    }

    /// Whether column `name` may take `col` in place of its values: the lead's rows of
    /// it (the history before row 0, which no write changes) must read with `col`'s
    /// values in a lossless common dtype — else the write is refused before it is made.
    pub(super) fn check_lead_dtype(&self, name: &str, col: &Column) -> Result<()> {
        let Some(lead) = &self.lead else { return Ok(()) };
        if lead.column(name).is_none() || lead.len() == 0 {
            return Ok(());
        }
        join(lead.tail(name, lead.len(), col.dtype()), &col.slice(0, 0)).map(|_| ()).map_err(|e| {
            VolasError::DType(format!(
                "cannot replace column \"{name}\" with {} values: this frame's cached columns read \
                 its rows before the frame's first row, which hold {} values ({e})",
                col.dtype(),
                lead.column(name).map(|c| c.dtype().to_string()).unwrap_or_default()
            ))
        })
    }

    /// How many data rows before row 0 this frame keeps in its lead (see the module docs).
    pub fn lead_rows(&self) -> usize {
        self.lead.as_ref().map_or(0, Lead::len)
    }

    /// `frame` — this frame's rows over some of its data columns (the frame a cached
    /// column recomputes over) — preceded by the last `rows` rows of this frame's lead
    /// (all of it when it holds fewer): the frame a recompute that reads before row 0 runs
    /// over, its row 0 at the absolute position of the first lead row it takes. A column
    /// the lead does not hold is missing on its rows; one whose dtype changed reads in the
    /// lossless common dtype. Its labels are positions (no computation reads labels).
    pub fn led(&self, frame: &DataFrame, rows: usize) -> DataFrame {
        let Some(lead) = self.lead.as_ref().filter(|_| rows > 0) else {
            return frame.clone();
        };
        let take = rows.min(lead.len());
        let columns = frame
            .names
            .iter()
            .zip(&frame.columns)
            .map(|(name, col)| join(lead.tail(name, take, col.dtype()), col).expect("checked when the column was written"))
            .collect();
        DataFrame::new(frame.names.to_vec(), columns, Some(Index::range(take + frame.height)))
            .expect("equal-length columns")
            .with_origin(frame.origin - take)
    }
}
