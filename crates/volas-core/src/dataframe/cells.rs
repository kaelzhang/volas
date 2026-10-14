//! Cell writes into a [`DataFrame`]: the one owner of "write values into cells in
//! place" — every positional, label and mask write (`iloc` / `loc` / `iat` / `at`
//! assignment, `df[mask] = v`) lands through it.
//!
//! A write runs in two phases, as rows entering the frame do ([`super::admit`]).
//! [`DataFrame::check_writes`] checks the whole batch first — every column and row in
//! range, every source of a length the rows take and converted losslessly to its
//! column's dtype ([`Column::assignable`]) — and hands back the checked batch holding
//! the frame it was checked against, so it can land nowhere else. [`CellWrites::land`]
//! then writes it in place, by steps that cannot fail. So a refused batch leaves the
//! frame exactly as it was, and a landed one costs O(cells written): each column's
//! buffer is written where it lies (copied once only while shared, copy-on-write), never
//! rebuilt.

use crate::column::Column;
use crate::error::{Result, VolasError};

use super::DataFrame;

/// One cell write: `values` into column `col` at the row positions `rows` — one value
/// (broadcast to every row) or one per row; a later repeat of a row wins.
#[derive(Clone, Debug)]
pub struct CellWrite {
    /// The column position.
    pub col: usize,
    /// The row positions written.
    pub rows: Vec<usize>,
    /// The values written: length 1 (broadcast) or `rows.len()`.
    pub values: Column,
}

/// A batch of [`CellWrite`]s checked against a frame ([`DataFrame::check_writes`]),
/// holding that frame: each write selects at least one row, and its values are of its
/// column's dtype. It lands on that frame alone ([`land`](Self::land)), which cannot
/// fail; dropped unlanded, it leaves the frame as it was.
#[derive(Debug)]
pub struct CellWrites<'a> {
    frame: &'a mut DataFrame,
    writes: Vec<CellWrite>,
}

impl CellWrites<'_> {
    /// The frame, as it is before the writes land.
    pub fn frame(&self) -> &DataFrame {
        self.frame
    }

    /// Whether these writes may change row `row` of the frame as a bar of its data
    /// columns: they write a cell of that row, or a cached column — which they turn
    /// into data. When they cannot, the row's data cells and the frame's data columns
    /// stay as they are. O(cells written).
    pub fn may_change_data_row(&self, row: usize) -> bool {
        let frame = self.frame();
        self.writes.iter().any(|w| frame.is_computed(&frame.names[w.col]) || w.rows.contains(&row))
    }

    /// Row `row` of the frame as the writes leave it, before they land: a one-row frame
    /// of the columns that are data after them — a written cached column becomes data —
    /// at the row's label. O(columns + cells written): the rest of the frame is never
    /// touched.
    pub fn data_row_after(&self, row: usize) -> DataFrame {
        let frame = self.frame();
        let written = |j: usize| self.writes.iter().any(|w| w.col == j);
        let keep: Vec<usize> = (0..frame.columns.len())
            .filter(|&j| !frame.is_computed(&frame.names[j]) || written(j))
            .collect();
        let mut cells: Vec<Column> = keep.iter().map(|&j| frame.columns[j].slice(row, row + 1)).collect();
        for w in &self.writes {
            let Some(k) = w.rows.iter().rposition(|&r| r == row) else { continue };
            let at = keep.binary_search(&w.col).expect("a written column is kept");
            let value = match w.values.len() {
                1 => w.values.clone(),
                _ => w.values.slice(k, k + 1),
            };
            cells[at].scatter_assignable(&[0], &value);
        }
        frame.row_frame(&keep, cells, row, row + 1)
    }
}

impl CellWrites<'_> {
    /// Land the writes in place on their frame — the second phase of a cell write;
    /// cannot fail. Writing in place keeps every cached column's identity: a written
    /// cached column **drops its computed status** (it becomes plain data, so a later
    /// `fulfill` can never silently clobber the override), and every other one goes
    /// stale.
    pub fn land(self) {
        let frame = self.frame;
        for w in self.writes {
            frame.columns[w.col].scatter_assignable(&w.rows, &w.values);
            frame.invalidate_computed_on_write_at(w.col);
        }
        #[cfg(debug_assertions)]
        frame.debug_check();
    }
}

impl DataFrame {
    /// Check `writes` against this frame — the first phase of a cell write (see the
    /// module doc). In order, for each write: its column is in range, its values are
    /// one or one per row, its rows are in range, and its values convert to the
    /// column's dtype ([`Column::assignable`]); the first failure is the error. A write
    /// selecting no row changes nothing: it is dropped unchecked.
    pub fn check_writes(&mut self, writes: Vec<CellWrite>) -> Result<CellWrites<'_>> {
        let mut checked = Vec::with_capacity(writes.len());
        for CellWrite { col, rows, values } in writes {
            if rows.is_empty() {
                continue;
            }
            let Some(column) = self.columns.get(col) else {
                return Err(VolasError::Shape(format!(
                    "column position {col} is out of range (width {})",
                    self.columns.len()
                )));
            };
            let n = rows.len();
            if values.len() != 1 && values.len() != n {
                return Err(VolasError::Shape(format!(
                    "cannot assign {} values to {n} selected rows",
                    values.len()
                )));
            }
            if let Some(&p) = rows.iter().find(|&&p| p >= self.height) {
                return Err(VolasError::Shape(format!(
                    "row position {p} is out of range (height {})",
                    self.height
                )));
            }
            let values = column.assignable(&values)?;
            self.check_lead_values(&self.names[col], &values, 0, values.len())?;
            checked.push(CellWrite { col, rows, values });
        }
        Ok(CellWrites { frame: self, writes: checked })
    }

    /// Rows `[start, end)` as bars of this frame's data columns (every column but the
    /// cached ones), at their labels — a time-frame frame's forming row (or none, when
    /// the frame is empty) as its fold reads it. O(columns × rows).
    pub fn data_rows(&self, start: usize, end: usize) -> DataFrame {
        let keep: Vec<usize> = (0..self.columns.len()).filter(|&j| !self.is_computed(&self.names[j])).collect();
        let cells = keep.iter().map(|&j| self.columns[j].slice(start, end)).collect();
        self.row_frame(&keep, cells, start, end)
    }

    /// A frame of the columns at positions `keep` holding `cells`, the rows
    /// `[start, end)` at their labels.
    fn row_frame(&self, keep: &[usize], cells: Vec<Column>, start: usize, end: usize) -> DataFrame {
        let names = keep.iter().map(|&j| self.names[j].clone()).collect();
        DataFrame::new(names, cells, Some(self.index.slice(start, end)))
            .expect("the rows of distinct columns of one frame")
    }
}

#[cfg(test)]
impl DataFrame {
    /// One checked-then-landed write (the two phases at once).
    pub(crate) fn write_one(&mut self, col: usize, rows: &[usize], values: &Column) -> Result<()> {
        self.check_writes(vec![CellWrite { col, rows: rows.to_vec(), values: values.clone() }])?.land();
        Ok(())
    }
}
