//! The aggregation fold: a period's values reduced into one cell, in place. The live
//! time-frame fold (the forming bar's update) and the batch reduce (`cumulate`) both
//! run these kernels — [`Column::init_at`] with the period's first value, then
//! [`Column::combine_at`] with each later one — so a fold equals the batch
//! aggregation exactly, for every dtype.
//!
//! The rules, held here once:
//! - support ([`CombineOp::supports`], contract #24): `first` / `last` any dtype,
//!   `max` / `min` the numeric dtypes and datetime, `sum` the numeric dtypes; each
//!   keeps its column's dtype;
//! - missing values (contract #23): an aggregate is missing as soon as any of the
//!   period's values is — `first` / `last` / `max` / `min` / `sum` alike, never a
//!   skipped value, never `±inf`, whatever payload a missing cell holds;
//! - an integer `sum` is checked (contract #25): an overflow raises;
//! - a source of another dtype converts to the cell's dtype losslessly first
//!   (contract #20) — never truncated or saturated.

use super::*;

/// An aggregation as an in-place single-cell fold ([`Column::combine_at`]): fold one
/// incoming value into a forming aggregate without re-reducing the period or
/// allocating. `first` keeps, `last` replaces, the rest combine associatively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineOp {
    /// Keep the destination unchanged (the period's `first`).
    Keep,
    /// Overwrite with the source value (the period's `last`).
    Replace,
    /// Running maximum.
    Max,
    /// Running minimum.
    Min,
    /// Running sum.
    Sum,
}

impl CombineOp {
    /// The aggregation support table (contract #24), owned here once: `first` /
    /// `last` (`Keep` / `Replace`) take any dtype; `max` / `min` the four numeric
    /// dtypes and datetime; `sum` the four numeric dtypes only (a sum of instants is
    /// not an instant). Each keeps its column's dtype.
    pub fn supports(self, dtype: DType) -> bool {
        match self {
            CombineOp::Keep | CombineOp::Replace => true,
            CombineOp::Max | CombineOp::Min => dtype.is_numeric() || dtype == DType::Datetime,
            CombineOp::Sum => dtype.is_numeric(),
        }
    }

    /// The aggregator's name (`first` / `last` / `max` / `min` / `sum`).
    pub fn name(self) -> &'static str {
        match self {
            CombineOp::Keep => "first",
            CombineOp::Replace => "last",
            CombineOp::Max => "max",
            CombineOp::Min => "min",
            CombineOp::Sum => "sum",
        }
    }

    /// The error of an aggregation the support table rejects.
    pub fn unsupported(self, dtype: DType) -> VolasError {
        VolasError::DType(format!("cannot {} a {dtype} column", self.name()))
    }
}

impl Column {
    /// Set cell `row` to the aggregate of the single value `src[src_row]` under
    /// `op`: that value with its presence (a one-value period's aggregate is the
    /// value). [`combine_at`](Self::combine_at) then folds each later value in. A
    /// `Str` cell is written in place only as the column's last row (the forming
    /// row; a [`VolasError::Shape`] error elsewhere).
    pub fn init_at(&mut self, row: usize, op: CombineOp, src: &Column, src_row: usize) -> Result<()> {
        if src.dtype() != self.dtype() || !op.supports(self.dtype()) {
            return self.fold_converted(row, op, src, src_row, true);
        }
        if !src.is_valid(src_row) {
            return self.set_missing_at(row);
        }
        self.write_present(row, src, src_row)
    }

    /// Fold `src[src_row]` into cell `row` in place, per `op` — the allocation-free
    /// forming-bar update of the live time-frame fold: a missing cell stays missing,
    /// a missing source makes it missing, and otherwise `Keep` (period `first`)
    /// leaves the cell, `Replace` (period `last`) takes the value, and `Max` / `Min`
    /// / `Sum` combine. Writes through `Buffer::make_mut`, so it is O(1) on a
    /// uniquely-owned buffer. A `Str` cell folds in place only as the last row.
    pub fn combine_at(
        &mut self,
        row: usize,
        op: CombineOp,
        src: &Column,
        src_row: usize,
    ) -> Result<()> {
        if src.dtype() != self.dtype() || !op.supports(self.dtype()) {
            return self.fold_converted(row, op, src, src_row, false);
        }
        if !self.is_valid(row) {
            return Ok(()); // a missing aggregate stays missing
        }
        if !src.is_valid(src_row) {
            return self.set_missing_at(row);
        }
        let r = src_row;
        match (op, self, src) {
            (CombineOp::Keep, ..) => {}
            (CombineOp::Replace, dst, src) => dst.write_present(row, src, r)?,
            // `make_mut` forgets the column's NaN fact; a resume then checks its few
            // rows directly (tracking it here would grow this per-bar hot path).
            (op, Column::F64(buf), Column::F64(s)) => {
                let b = buf.make_mut();
                b[row] = fold_float(op, b[row], s[r]);
            }
            (op, Column::F32(buf), Column::F32(s)) => {
                let b = buf.make_mut();
                b[row] = fold_float(op, b[row], s[r]);
            }
            (op, Column::I64(buf, _), Column::I64(s, _)) => fold_int(buf, row, op, s[r], DType::I64)?,
            (op, Column::I32(buf, _), Column::I32(s, _)) => fold_int(buf, row, op, s[r], DType::I32)?,
            (op, Column::Datetime(buf), Column::Datetime(s)) => {
                let b = buf.make_mut();
                b[row] = if op == CombineOp::Max { b[row].max(s[r]) } else { b[row].min(s[r]) };
            }
            _ => unreachable!("the support table admits no other pair"), // LCOV_EXCL_LINE
        }
        Ok(())
    }

    /// Write the present value `src[r]` (same dtype) into cell `row`, present.
    #[inline]
    fn write_present(&mut self, row: usize, src: &Column, r: usize) -> Result<()> {
        let n = self.len();
        match (self, src) {
            (Column::F64(buf), Column::F64(s)) => buf.set(row, s[r]),
            (Column::F32(buf), Column::F32(s)) => buf.set(row, s[r]),
            (Column::Datetime(buf), Column::Datetime(s)) => buf.make_mut()[row] = s[r],
            (Column::I64(buf, val), Column::I64(s, _)) => {
                buf.make_mut()[row] = s[r];
                val.set(row, n, true);
            }
            (Column::I32(buf, val), Column::I32(s, _)) => {
                buf.make_mut()[row] = s[r];
                val.set(row, n, true);
            }
            (Column::Bool(buf, val), Column::Bool(s, _)) => {
                buf.make_mut()[row] = s[r];
                val.set(row, n, true);
            }
            (Column::Str(buf, val), Column::Str(s, _)) => {
                if row + 1 != n {
                    return Err(str_not_last(row, n));
                }
                buf.set_last(s.get(r));
                val.set(row, n, true);
            }
            _ => unreachable!("the caller matched the dtypes"), // LCOV_EXCL_LINE
        }
        Ok(())
    }

    /// Mark cell `row` missing in the dtype's own representation (`NaN` / `NaT`,
    /// else a validity hole) — the rare branch of the fold, kept off its hot path.
    #[cold]
    #[inline(never)]
    fn set_missing_at(&mut self, row: usize) -> Result<()> {
        let n = self.len();
        match self {
            Column::F64(a) => a.set(row, f64::NAN),
            Column::F32(a) => a.set(row, f32::NAN),
            Column::Datetime(a) => a.make_mut()[row] = i64::MIN,
            Column::Str(_, _) if row + 1 != n => return Err(str_not_last(row, n)),
            Column::Bool(_, v) | Column::I64(_, v) | Column::I32(_, v) | Column::Str(_, v) => {
                v.set(row, n, false)
            }
        }
        Ok(())
    }

    /// Off the live path: the support table's error for an unsupported `op`, else
    /// `src[src_row]` converted losslessly to this column's dtype (contract #20) and
    /// then initialized (`init`) or combined into cell `row`.
    #[cold]
    #[inline(never)]
    fn fold_converted(&mut self, row: usize, op: CombineOp, src: &Column, src_row: usize, init: bool) -> Result<()> {
        if !op.supports(self.dtype()) {
            return Err(op.unsupported(self.dtype()));
        }
        let cell = src
            .slice(src_row, src_row + 1)
            .convert_lossless(self.dtype())
            .map_err(|e| self.append_error(src, e))?;
        if init {
            self.init_at(row, op, &cell, 0)
        } else {
            self.combine_at(row, op, &cell, 0)
        }
    }
}

/// The error of a `Str` fold into a cell other than the last row (a string column
/// is rewritten in place only at its end).
#[cold]
#[inline(never)]
fn str_not_last(row: usize, n: usize) -> VolasError {
    VolasError::Shape(format!("a str cell folds in place only at the last row ({row} of {n})"))
}

/// `Max` / `Min` / `Sum` of two present floats.
#[inline]
fn fold_float<T: Numeric + std::ops::Add<Output = T>>(op: CombineOp, acc: T, x: T) -> T {
    match op {
        CombineOp::Max => if x > acc { x } else { acc },
        CombineOp::Min => if x < acc { x } else { acc },
        _ => acc + x, // Sum (Keep / Replace never reach here)
    }
}

/// `Max` / `Min` / `Sum` of two present integers in `i64`, checked: a result past
/// `dtype` is its overflow error.
#[inline]
fn fold_int<T: crate::buffer::Element + Copy + Into<i64> + TryFrom<i64>>(
    buf: &mut Buffer<T>,
    row: usize,
    op: CombineOp,
    x: T,
    dtype: DType,
) -> Result<()> {
    let (acc, x): (i64, i64) = (buf[row].into(), x.into());
    let v = match op {
        CombineOp::Max => Some(acc.max(x)),
        CombineOp::Min => Some(acc.min(x)),
        _ => acc.checked_add(x), // Sum (Keep / Replace never reach here)
    };
    match v.and_then(|v| T::try_from(v).ok()) {
        Some(v) => buf.make_mut()[row] = v,
        None => return Err(fold_overflow(dtype)),
    }
    Ok(())
}

/// The overflow error of an integer fold — off the per-bar hot path.
#[cold]
#[inline(never)]
fn fold_overflow(dtype: DType) -> VolasError {
    crate::numeric::overflow(dtype, "a time-frame fold")
}
