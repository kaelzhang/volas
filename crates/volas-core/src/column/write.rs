//! Target-bound fold admissions. Raw cell/reduction commits stay crate-private,
//! so the core frame owner can reuse its own complete-plan validation.
use super::*;

/// A checked cell fold bound to its exact destination and immutable source.
pub struct FoldCell<'a> {
    target: &'a mut Column,
    source: &'a Column,
    row: usize,
    source_row: usize,
    op: CombineOp,
    mode: CellMode,
}
enum CellMode { Init, Combine, Reduced(bool) }
impl FoldCell<'_> {
    /// Commit the admitted operation without validating a second destination.
    pub fn land(self) -> Result<()> {
        match self.mode {
            CellMode::Init => self.target.init_at(self.row, self.op, self.source, self.source_row),
            CellMode::Combine => self.target.combine_at(self.row, self.op, self.source, self.source_row),
            CellMode::Reduced(missing) => self.target.combine_reduced_at(self.row, self.op, self.source, self.source_row, missing),
        }
    }
}

/// A complete checked reduction, holding its destination until publication.
pub struct Reduction<'a> {
    target: &'a mut Column,
    source: &'a Column,
    row: usize,
    op: CombineOp,
    rows: ReductionRows<'a>,
}
/// Immutable reduction positions, retained unchanged until the checked commit.
pub enum ReductionRows<'a> {
    /// A contiguous owned range.
    Range(std::ops::Range<usize>),
    /// Borrowed positions, which cannot change during admission and commit.
    Positions(&'a [usize]),
}
impl From<std::ops::Range<usize>> for ReductionRows<'_> {
    fn from(rows: std::ops::Range<usize>) -> Self { Self::Range(rows) }
}
impl<'a> From<&'a [usize]> for ReductionRows<'a> {
    fn from(rows: &'a [usize]) -> Self { Self::Positions(rows) }
}
impl<'a, const N: usize> From<&'a [usize; N]> for ReductionRows<'a> {
    fn from(rows: &'a [usize; N]) -> Self { Self::Positions(rows) }
}
impl Reduction<'_> {
    /// Publish the admitted reduction on the original destination.
    pub fn land(self) -> Result<()> {
        match self.rows {
            ReductionRows::Range(rows) => self.target.reduce_into(self.row, self.op, self.source, rows),
            ReductionRows::Positions(rows) => self.target.reduce_into(self.row, self.op, self.source, rows.iter().copied()),
        }
    }
}

impl Column {
    /// Check all bounds, conversions and arithmetic before a cell fold. `restart`
    /// initializes from the source instead of continuing the destination value.
    pub fn check_fold_cell<'a>(&'a mut self, row: usize, op: CombineOp, source: &'a Column, source_row: usize, restart: bool) -> Result<FoldCell<'a>> {
        self.check_fold(row, op, restart, [(source, source_row)])?;
        Ok(FoldCell { target: self, source, row, source_row, op, mode: if restart { CellMode::Init } else { CellMode::Combine } })
    }

    /// Check a reduction-prefix continuation with its existing missing-input fact.
    pub fn check_reduced_cell<'a>(&'a mut self, row: usize, op: CombineOp, source: &'a Column, source_row: usize, missing: bool) -> Result<FoldCell<'a>> {
        self.check_fold(row, op, false, [(source, source_row)])?;
        Ok(FoldCell { target: self, source, row, source_row, op, mode: CellMode::Reduced(missing) })
    }

    /// Check an immutable selection completely before reducing any cell.
    pub fn check_reduction<'a>(&'a mut self, row: usize, op: CombineOp, source: &'a Column, rows: impl Into<ReductionRows<'a>>) -> Result<Reduction<'a>> {
        let rows = rows.into();
        match &rows {
            ReductionRows::Range(rows) => self.check_fold(row, op, true, rows.clone().map(|r| (source, r)))?,
            ReductionRows::Positions(rows) => self.check_fold(row, op, true, rows.iter().map(|&r| (source, r)))?,
        }
        Ok(Reduction { target: self, source, row, op, rows })
    }
}
