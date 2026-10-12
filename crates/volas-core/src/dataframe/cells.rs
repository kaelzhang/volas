//! Cell writes into a [`DataFrame`]: the one owner of "write values into cells in
//! place" — every positional, label and mask write (`iloc` / `loc` / `iat` / `at`
//! assignment, `df[mask] = v`) lands through it.
//!
//! A write runs in two phases, as rows entering the frame do ([`super::admit`]).
//! [`DataFrame::check_writes`] checks the whole batch first — every column and row in
//! range, every source of a length the rows take and converted losslessly to its
//! column's dtype ([`Column::assignable`]). [`DataFrame::write_cells`] then lands the
//! checked batch in place, by steps that cannot fail. So a refused batch leaves the
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

/// A batch of [`CellWrite`]s checked against a frame ([`DataFrame::check_writes`]):
/// each selects at least one row, and its values are of its column's dtype. Lands
/// on that frame with [`DataFrame::write_cells`], which cannot fail.
#[derive(Debug)]
pub struct CellWrites {
    writes: Vec<CellWrite>,
    /// The `(height, width)` of the frame the batch was checked against.
    shape: (usize, usize),
}

impl CellWrites {
    /// Whether these writes may change row `row` of `frame` (the frame they were
    /// checked against) as a bar of its data columns: they write a cell of that row, or
    /// a cached column — which they turn into data. When they cannot, the row's data
    /// cells and the frame's data columns stay as they are. O(cells written).
    pub fn may_change_data_row(&self, frame: &DataFrame, row: usize) -> bool {
        debug_assert_eq!(self.shape, (frame.height, frame.columns.len()), "checked on another frame");
        self.writes.iter().any(|w| frame.is_computed(&frame.names[w.col]) || w.rows.contains(&row))
    }

    /// Row `row` of `frame` (the frame these writes were checked against) as the
    /// writes leave it: a one-row frame of the columns that are data after them — a
    /// written cached column becomes data — at the row's label. O(columns + cells
    /// written): the rest of the frame is never touched.
    pub fn data_row_after(&self, frame: &DataFrame, row: usize) -> DataFrame {
        debug_assert_eq!(self.shape, (frame.height, frame.columns.len()), "checked on another frame");
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
        frame.row_frame(&keep, cells, row)
    }
}

impl DataFrame {
    /// Check `writes` against this frame — the first phase of a cell write (see the
    /// module doc). In order, for each write: its column is in range, its values are
    /// one or one per row, its rows are in range, and its values convert to the
    /// column's dtype ([`Column::assignable`]); the first failure is the error. A write
    /// selecting no row changes nothing: it is dropped unchecked.
    pub fn check_writes(&self, writes: Vec<CellWrite>) -> Result<CellWrites> {
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
            checked.push(CellWrite { col, rows, values });
        }
        Ok(CellWrites { writes: checked, shape: (self.height, self.columns.len()) })
    }

    /// Land checked `writes` in place — the second phase of a cell write; cannot fail.
    /// Writing in place keeps every cached column's identity: a written cached column
    /// **drops its computed status** (it becomes plain data, so a later `fulfill` can
    /// never silently clobber the override), and every other one goes stale.
    pub fn write_cells(&mut self, writes: CellWrites) {
        debug_assert_eq!(writes.shape, (self.height, self.columns.len()), "checked on another frame");
        for w in writes.writes {
            self.columns[w.col].scatter_assignable(&w.rows, &w.values);
            self.invalidate_computed_on_write_at(w.col);
        }
        #[cfg(debug_assertions)]
        self.debug_check();
    }

    /// Row `row` as a bar of this frame's data columns (every column but the cached
    /// ones): a one-row frame at the row's label. O(columns).
    pub fn data_row(&self, row: usize) -> DataFrame {
        let keep: Vec<usize> = (0..self.columns.len()).filter(|&j| !self.is_computed(&self.names[j])).collect();
        let cells = keep.iter().map(|&j| self.columns[j].slice(row, row + 1)).collect();
        self.row_frame(&keep, cells, row)
    }

    /// A one-row frame of the columns at positions `keep` holding `cells`, at row
    /// `row`'s label.
    fn row_frame(&self, keep: &[usize], cells: Vec<Column>, row: usize) -> DataFrame {
        let names = keep.iter().map(|&j| self.names[j].clone()).collect();
        DataFrame::new(names, cells, Some(self.index.slice(row, row + 1)))
            .expect("one cell per distinct column of one frame")
    }
}

#[cfg(test)]
impl DataFrame {
    /// One checked-then-landed write (the two phases at once).
    pub(crate) fn write_one(&mut self, col: usize, rows: &[usize], values: &Column) -> Result<()> {
        let writes = self.check_writes(vec![CellWrite { col, rows: rows.to_vec(), values: values.clone() }])?;
        self.write_cells(writes);
        Ok(())
    }
}
