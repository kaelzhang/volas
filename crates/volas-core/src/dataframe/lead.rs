//! The **lead** of a frame: the data rows just before its row 0 that its cached columns
//! read to recompute their first rows. A contiguous selection keeps it (see `derive`):
//! each cached column whose history starts before the selection needs the fewer of its
//! [`Reach::lead`](super::Reach::lead) and its history's rows there — and a recursion the
//! rows after the point of its history it replays from: its checkpoint, or its whole
//! history when it keeps no checkpoint (a recursion whose history starts nearby).
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
/// frame it was taken for (`name_to_idx`, shared with it), one column per name,
/// `None` for a cached column's — whose row `to` would be the frame's row 0. A lead that
/// is most of its source's rows shares the source's buffers; a shorter one is a copy of
/// just its rows. Packed float columns may share the copied lead rows and selected
/// visible rows in one bounded slab; it never pins the source history. Removing
/// columns may retain the other copied ranges until the lead is copied again.
#[derive(Clone, Debug)]
pub(super) struct Lead {
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
fn wanted(computed: &FxHashMap<Arc<str>, ComputedMeta>, first: usize) -> usize {
    computed
        .values()
        .filter(|m| m.since < first)
        .map(|m| match (m.reach.history, replays_from(m)) {
            // Checkpoint state already owns every kernel tail and chain ring. Only
            // stateless operands read raw rows before its next row (`operands`).
            (History::Carried { operands }, Some(from)) => {
                first.saturating_sub(from.saturating_sub(operands)).min(first - m.since)
            }
            _ => m.reach.lead().min(first - m.since),
        })
        .max()
        .unwrap_or(0)
}

/// The absolute row a recompute of recursion `m` replays from, when it is a recursion
/// that a frame not holding its history keeps by a point of it: the row after its
/// checkpoint — or its history's start when no checkpoint has formed yet.
fn replays_from(m: &ComputedMeta) -> Option<usize> {
    match (m.reach.history, &m.checkpoint) {
        (History::Carried { .. }, Some(cp)) => Some(cp.position() + 1),
        (History::Carried { .. }, None) => Some(m.since),
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

    /// This lead over the data columns of `frame` (the frame it was taken for, less some
    /// columns): those it holds, in `frame`'s schema.
    pub(super) fn over(&self, frame: &DataFrame) -> Lead {
        let held = |n: &String| self.column(n).filter(|_| !frame.computed.contains_key(n.as_str())).cloned();
        let columns = frame.names.iter().map(held).collect();
        Lead { name_to_idx: Arc::clone(&frame.name_to_idx), columns, ..*self }
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
        assert_eq!(self.columns.len(), self.name_to_idx.len(), "one lead column per name");
        assert!(self.columns.iter().flatten().all(|c| c.len() == self.height), "the lead's columns are alike");
        assert!(self.from <= self.to && self.to <= self.height, "the lead's rows are its columns'");
    }
}

impl DataFrame {
    /// A short float selection may copy its visible rows and raw lead into one
    /// immutable allocation. Older lead or mostly-shared history follows `lead_before`.
    pub(super) fn copied_slab_lead_rows(&self, start: usize, computed: &FxHashMap<Arc<str>, ComputedMeta>) -> usize {
        let rows = wanted(computed, self.origin + start);
        if rows > 0 && rows <= start && rows < self.height - rows && computed.len() < self.width() {
            rows
        } else {
            0
        }
    }

    /// Raw inputs precede the cached suffix: copy each input's adjacent lead and
    /// visible rows together. Every column still has its own COW range and NaN fact.
    pub(super) fn copy_float_prefix_rows(&self, start: usize, end: usize, raw: usize, lead_rows: usize) -> (Vec<Column>, Lead) {
        let rows = end - start;
        let mut values = Vec::with_capacity(rows * self.width() + lead_rows * raw);
        for (i, col) in self.columns.iter().enumerate() {
            let from = if i < raw { start - lead_rows } else { start };
            values.extend_from_slice(&col.as_f64().expect("checked floats")[from..end]);
        }
        let owner = Arc::new(values);
        let mut columns = Vec::with_capacity(self.width());
        let mut history = Vec::with_capacity(self.width());
        for (i, col) in self.columns.iter().enumerate() {
            let clean = col.known_has_nan() == Some(false);
            let finite = col.known_has_nonfinite() == Some(false);
            let buffer = |range| {
                let values = crate::buffer::Buffer::from_shared_known_range(
                    Arc::clone(&owner), range, clean, finite,
                ).expect("copied prefix slab range");
                Column::F64(values)
            };
            let from = if i < raw { i * (rows + lead_rows) } else { i * rows + raw * lead_rows };
            let visible = if i < raw { from + lead_rows } else { from };
            columns.push(buffer(visible..visible + rows));
            history.push((i < raw).then(|| buffer(from..from + lead_rows)));
        }
        let lead = Lead { name_to_idx: Arc::clone(&self.name_to_idx), columns: history, height: lead_rows, from: 0, to: lead_rows };
        (columns, lead)
    }

    /// Finish the bounded copied slab, preserving the visible prefix and adding only
    /// raw operand rows. The caller checked that all columns are float64.
    pub(super) fn finish_copied_slab_lead(
        &self, start: usize, computed: &FxHashMap<Arc<str>, ComputedMeta>, rows: usize, mut values: Vec<f64>,
    ) -> (Arc<Vec<f64>>, Option<Lead>) {
        if rows == 0 {
            return (Arc::new(values), None);
        }
        let data_count = self.width() - computed.len();
        let raw_prefix = self.first_computed == Some(data_count);
        let data = |i: usize| if raw_prefix { i < data_count } else {
            !computed.contains_key(self.names[i].as_str())
        };
        let mut offset = values.len();
        for (i, col) in self.columns.iter().enumerate() {
            if data(i) {
                values.extend_from_slice(&col.as_f64().expect("checked floats")[start - rows..start]);
            }
        }
        let owner = Arc::new(values);
        let columns = self.columns.iter().enumerate().map(|(i, col)| data(i).then(|| {
            let range = offset..offset + rows;
            offset += rows;
            let buffer = crate::buffer::Buffer::from_shared_known_range(
                        Arc::clone(&owner), range, col.known_has_nan() == Some(false),
                        col.known_has_nonfinite() == Some(false),
                    ).expect("copied lead slab range");
            Column::F64(buffer)
        })).collect();
        let lead = Lead { name_to_idx: Arc::clone(&self.name_to_idx), columns, height: rows, from: 0, to: rows };
        (owner, Some(lead))
    }

    /// The lead of the selection starting at row `start` whose cached columns are
    /// `computed`: the rows they read before it (see the module docs), taken from this
    /// frame's rows before `start`, then its own lead. `None` when no column reads
    /// before it.
    pub(super) fn lead_before(&self, start: usize, computed: &FxHashMap<Arc<str>, ComputedMeta>) -> Option<Lead> {
        let first = self.origin + start;
        let wanted = wanted(computed, first);
        let own = wanted.min(start);
        let held = self.lead.as_ref().map_or(0, Lead::len);
        let older = (wanted - own).min(held);
        if own + older == 0 {
            return None;
        }
        let data_count = self.width().saturating_sub(computed.len());
        // When every raw column precedes the cached suffix, its count and the
        // memoized first cached position agree. No per-column key lookup is needed.
        let raw_prefix = self.first_computed == Some(data_count);
        let data = |i: usize| if raw_prefix { i < data_count } else {
            self.first_computed.is_none_or(|first| i < first)
                || !computed.contains_key(self.names[i].as_str())
        };
        let name_to_idx = Arc::clone(&self.name_to_idx);
        if older == 0 && 2 * own >= self.height {
            // most of this frame's rows: share its buffers rather than copy them
            let columns = self.columns.iter().enumerate().map(|(i, c)| data(i).then(|| c.clone())).collect();
            return Some(Lead { name_to_idx, columns, height: self.height, from: start - own, to: start });
        }
        if older == 0 && data_count > 1
            && self.columns.iter().all(|col| matches!(col, Column::F64(_)))
        {
            // One bounded copy for the raw float lead, not a separate Vec and Arc
            // per column. The visible columns remain independent value copies.
            let mut values = Vec::with_capacity(own * data_count);
            for (i, col) in self.columns.iter().enumerate() {
                if data(i) {
                    values.extend_from_slice(&col.as_f64().expect("checked floats")[start - own..start]);
                }
            }
            let owner = Arc::new(values);
            let mut offset = 0;
            let columns = self.columns.iter().enumerate().map(|(i, _)| {
                data(i).then(|| {
                    let from = offset;
                    offset += own;
                    let buffer = crate::buffer::Buffer::from_shared_range(Arc::clone(&owner), from..offset)
                        .expect("copied slab range");
                    Column::F64(buffer)
                })
            }).collect();
            return Some(Lead { name_to_idx, columns, height: own, from: 0, to: own });
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
        Some(Lead { name_to_idx, columns, height: len, from: 0, to: len })
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

    /// Validate incoming cells against the dtype they will share with the historical
    /// lead. Every numeric widening is exact by dtype except int64 into float64;
    /// only that pair scans present cells, with no conversion or allocation.
    pub(super) fn check_lead_values(&self, name: &str, col: &Column, from: usize, end: usize) -> Result<()> {
        let Column::I64(values, valid) = col else { return Ok(()) };
        let Some(held) = self.lead.as_ref().filter(|lead| lead.len() > 0).and_then(|lead| lead.column(name)) else {
            return Ok(());
        };
        if !held.dtype().is_float() {
            return Ok(());
        }
        for row in from..end {
            if valid.is_valid(row) && crate::numeric::i64_to_f64_exact(values[row]).is_none() {
                return Err(lead_inexact(name, values[row]));
            }
        }
        Ok(())
    }

    /// Check a fold through the column owner, then validate its already computed
    /// integer aggregate against the historical lead. No second pass or allocation.
    pub(super) fn check_fold_with_lead<'a>(
        &self, col: usize, dst: &Column, (row, op, restart): (usize, crate::CombineOp, bool),
        cells: impl IntoIterator<Item = (&'a Column, usize)>,
    ) -> Result<()> {
        let value = dst.checked_fold_value(row, op, restart, cells)?;
        if let (Column::I64(..), Some(value), Some(lead)) = (dst, value, &self.lead) {
            if lead.len() > 0 && crate::numeric::i64_to_f64_exact(value).is_none() {
                let name = &self.names[col];
                if lead.column(name).is_some_and(|held| held.dtype().is_float()) {
                    return Err(lead_inexact(name, value));
                }
            }
        }
        Ok(())
    }

    /// Whether column `name` may take `col` in place of its values: the lead's rows of
    /// it (the history before row 0, which no write changes) must read with `col`'s
    /// values in a lossless common dtype — else the write is refused before it is made.
    pub(super) fn check_lead_dtype(&self, name: &str, col: &Column) -> Result<()> {
        let Some(lead) = &self.lead else { return Ok(()) };
        if lead.column(name).is_none() || lead.len() == 0 {
            return Ok(());
        }
        self.check_lead_values(name, col, 0, col.len())?;
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

fn lead_inexact(name: &str, value: i64) -> VolasError {
    VolasError::DType(format!(
        "cannot write column \"{name}\": {value} is not exactly representable as float64, \
         the common dtype of its new values and its rows before the frame's first row"
    ))
}
