//! Immutable grouping proof: counts, nonempty groups and every position are
//! checked once, then shared by all columns of the same height.
use super::*;

type GroupResult<T> = std::result::Result<T, (usize, VolasError)>;

/// Checked row groups bound to a source height. Its immutable positions cannot
/// change between validation and publication or lie about the output count.
pub struct RowGroups<'a> {
    source_len: usize,
    rows: Groups<'a>,
}
enum Groups<'a> {
    Nested(&'a [Vec<usize>]),
    Flat(&'a [usize], &'a [usize]),
}
impl<'a> RowGroups<'a> {
    /// Validate nonempty groups and all source positions.
    pub fn new(groups: &'a [Vec<usize>], source_len: usize) -> GroupResult<Self> {
        validate(groups.iter().map(Vec::as_slice), source_len)?;
        Ok(Self { source_len, rows: Groups::Nested(groups) })
    }

    /// Validate flat positions and group offsets, including both endpoints.
    /// Empty grouping is represented by empty positions and offsets `[0]`.
    pub fn from_offsets(positions: &'a [usize], offsets: &'a [usize], source_len: usize) -> GroupResult<Self> {
        if offsets.first() != Some(&0) || offsets.last() != Some(&positions.len()) {
            return Err(group_error(0));
        }
        for (i, w) in offsets.windows(2).enumerate() {
            if w[0] >= w[1] || w[1] > positions.len() { return Err(group_error(i)); }
        }
        validate(offsets.windows(2).map(|w| &positions[w[0]..w[1]]), source_len)?;
        Ok(Self { source_len, rows: Groups::Flat(positions, offsets) })
    }

    /// Retain an admitted prefix without scanning its positions again.
    pub fn prefix(&self, count: usize) -> Self {
        let rows = match self.rows {
            Groups::Nested(rows) => Groups::Nested(&rows[..count.min(rows.len())]),
            Groups::Flat(rows, offsets) => {
                let count = count.min(offsets.len() - 1);
                Groups::Flat(&rows[..offsets[count]], &offsets[..=count])
            }
        };
        Self { source_len: self.source_len, rows }
    }

    /// Reduce into a fresh, correctly sized column.
    pub fn reduce(&self, source: &Column, op: CombineOp) -> GroupResult<Column> {
        self.reduce_with_spare(source, op, 0, 0)
    }

    /// Reduce with capacity for subsequent rows and UTF-8 payload bytes.
    pub fn reduce_with_spare(&self, source: &Column, op: CombineOp, spare_rows: usize, spare_bytes: usize) -> GroupResult<Column> {
        if source.len() != self.source_len { return Err(group_error(0)); }
        match self.rows {
            Groups::Nested(rows) => source.reduce_groups_with_spare(op, rows.iter().map(Vec::as_slice), spare_rows, spare_bytes),
            Groups::Flat(rows, offsets) => source.reduce_groups_with_spare(op, offsets.windows(2).map(|w| &rows[w[0]..w[1]]), spare_rows, spare_bytes),
        }
    }
}
fn validate<'a>(groups: impl Iterator<Item = &'a [usize]>, source_len: usize) -> GroupResult<()> {
    for (i, rows) in groups.enumerate() {
        if rows.is_empty() || rows.iter().any(|&row| row >= source_len) { return Err(group_error(i)); }
    }
    Ok(())
}
#[cold]
fn group_error(i: usize) -> (usize, VolasError) {
    (i, VolasError::Shape("invalid aggregation group or source height".into()))
}
