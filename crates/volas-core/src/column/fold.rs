//! The aggregation fold: a period's values reduced into one cell, in place. The live
//! time-frame fold folds each new bar in ([`Column::combine_at`], after
//! [`Column::init_at`] for the period's first value); a whole period — the batch
//! `cumulate` and a re-sent bar's re-aggregation — reduces in one pass per column
//! ([`Column::reduce_into`]). Both apply the same steps, so a fold equals the batch
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

    /// Whether this is an endpoint (`first` / `last`) rather than a combination.
    fn is_endpoint(self) -> bool {
        matches!(self, CombineOp::Keep | CombineOp::Replace)
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
    /// uniquely-owned buffer. One dtype dispatch per call; another source dtype or
    /// an unsupported `op` takes the cold path. A `Str` cell folds in place only as
    /// the last row.
    pub fn combine_at(
        &mut self,
        row: usize,
        op: CombineOp,
        src: &Column,
        src_row: usize,
    ) -> Result<()> {
        let r = src_row;
        match (&mut *self, src) {
            // `make_mut` forgets the column's NaN fact; a resume then checks its few
            // rows directly (tracking it here would grow this per-bar hot path).
            (Column::F64(buf), Column::F64(s)) => {
                if let Some(v) = float_step(op, buf[row], s[r]) {
                    buf.make_mut()[row] = v;
                }
            }
            (Column::F32(buf), Column::F32(s)) => {
                if let Some(v) = float_step(op, buf[row], s[r]) {
                    buf.make_mut()[row] = v;
                }
            }
            (Column::I64(buf, val), Column::I64(s, sv)) => {
                int_step(buf, val, row, op, s[r], sv.is_valid(r), DType::I64)?
            }
            (Column::I32(buf, val), Column::I32(s, sv)) => {
                int_step(buf, val, row, op, s[r], sv.is_valid(r), DType::I32)?
            }
            (Column::Datetime(buf), Column::Datetime(s)) if op != CombineOp::Sum => {
                if let Some(v) = instant_step(op, buf[row], s[r]) {
                    buf.make_mut()[row] = v;
                }
            }
            (Column::Bool(..), Column::Bool(..)) | (Column::Str(..), Column::Str(..)) if op.is_endpoint() => {
                if !self.is_valid(row) {
                    return Ok(()); // a missing aggregate stays missing
                }
                if !src.is_valid(r) {
                    return self.set_missing_at(row);
                }
                if op == CombineOp::Replace {
                    self.write_present(row, src, r)?;
                }
            }
            _ => return self.fold_converted(row, op, src, src_row, false),
        }
        Ok(())
    }

    /// Reduce `src`'s `rows` (a whole period, in order) into cell `row` under `op`
    /// in one pass — the batch reduce and a re-sent bar's re-aggregation. The dtype
    /// is dispatched once for the column; the result is exactly
    /// [`init_at`](Self::init_at) with the first row and
    /// [`combine_at`](Self::combine_at) with each later one. An empty `rows` leaves
    /// the cell unchanged.
    pub fn reduce_into(
        &mut self,
        row: usize,
        op: CombineOp,
        src: &Column,
        rows: impl IntoIterator<Item = usize>,
    ) -> Result<()> {
        let mut rows = rows.into_iter();
        let Some(first) = rows.next() else { return Ok(()) };
        if src.dtype() != self.dtype() || !op.supports(self.dtype()) {
            self.init_at(row, op, src, first)?;
            return rows.try_for_each(|r| self.combine_at(row, op, src, r));
        }
        if op.is_endpoint() {
            // first / last: the endpoint cell, missing unless every row is present
            // (presence read once per dtype, not dispatched per row)
            let (present, last) = match src {
                Column::F64(s) => endpoint_scan(first, rows, |r| !s[r].is_nan()),
                Column::F32(s) => endpoint_scan(first, rows, |r| !s[r].is_nan()),
                Column::Datetime(s) => endpoint_scan(first, rows, |r| s[r] != i64::MIN),
                Column::I64(_, v) | Column::I32(_, v) | Column::Bool(_, v) | Column::Str(_, v)
                    if !v.has_nulls() =>
                {
                    endpoint_scan(first, rows, |_| true)
                }
                Column::I64(_, v) | Column::I32(_, v) | Column::Bool(_, v) | Column::Str(_, v) => {
                    endpoint_scan(first, rows, |r| v.is_valid(r))
                }
            };
            if !present {
                return self.set_missing_at(row);
            }
            return self.write_present(row, src, if op == CombineOp::Keep { first } else { last });
        }
        match (&mut *self, src) {
            (Column::F64(buf), Column::F64(s)) => buf.set(row, float_reduce(op, s, first, rows)),
            (Column::F32(buf), Column::F32(s)) => buf.set(row, float_reduce(op, s, first, rows)),
            (Column::I64(buf, val), Column::I64(s, sv)) => {
                let v = int_reduce(op, s, sv, first, rows, DType::I64)?;
                write_int(buf, val, row, v)
            }
            (Column::I32(buf, val), Column::I32(s, sv)) => {
                let v = int_reduce(op, s, sv, first, rows, DType::I32)?;
                write_int(buf, val, row, v)
            }
            (Column::Datetime(buf), Column::Datetime(s)) => {
                let mut acc = s[first];
                for x in std::iter::once(first).chain(rows).map(|r| s[r]) {
                    if x == i64::MIN {
                        return self.set_missing_at(row);
                    }
                    acc = if op == CombineOp::Max { acc.max(x) } else { acc.min(x) };
                }
                buf.make_mut()[row] = acc;
            }
            _ => unreachable!("the support table admits no other pair"), // LCOV_EXCL_LINE
        }
        Ok(())
    }

    /// Write the present value `src[r]` (same dtype) into cell `row`, present.
    fn write_present(&mut self, row: usize, src: &Column, r: usize) -> Result<()> {
        let n = self.len();
        match (self, src) {
            (Column::F64(buf), Column::F64(s)) => buf.set(row, s[r]),
            (Column::F32(buf), Column::F32(s)) => buf.set(row, s[r]),
            (Column::Datetime(buf), Column::Datetime(s)) => buf.make_mut()[row] = s[r],
            (Column::I64(buf, val), Column::I64(s, _)) => write_int(buf, val, row, Some(s[r])),
            (Column::I32(buf, val), Column::I32(s, _)) => write_int(buf, val, row, Some(s[r])),
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

/// Whether every row (`first` then `rest`) is present, and the last row.
#[inline]
fn endpoint_scan(first: usize, rest: impl Iterator<Item = usize>, valid: impl Fn(usize) -> bool) -> (bool, usize) {
    let mut present = valid(first);
    let mut last = first;
    for r in rest {
        present &= valid(r);
        last = r;
    }
    (present, last)
}

/// One float fold step: the new aggregate, or `None` when it stays as it is. A
/// missing (`NaN`) aggregate stays missing; a missing value makes it missing.
#[inline]
fn float_step<T: Numeric + std::ops::Add<Output = T>>(op: CombineOp, acc: T, x: T) -> Option<T> {
    if acc.is_missing() {
        return None;
    }
    if x.is_missing() {
        return Some(x);
    }
    match op {
        CombineOp::Keep => None,
        CombineOp::Replace => Some(x),
        _ => Some(float_combine(op, acc, x)),
    }
}

/// `Max` / `Min` / `Sum` of two present floats.
#[inline]
fn float_combine<T: Numeric + std::ops::Add<Output = T>>(op: CombineOp, acc: T, x: T) -> T {
    match op {
        CombineOp::Max => if x > acc { x } else { acc },
        CombineOp::Min => if x < acc { x } else { acc },
        _ => acc + x, // Sum (endpoints never reach here)
    }
}

/// `Max` / `Min` / `Sum` of `s` over `first` then `rest` — `NaN` at the first
/// missing value.
#[inline]
fn float_reduce<T: Numeric + std::ops::Add<Output = T>>(
    op: CombineOp,
    s: &[T],
    first: usize,
    rest: impl Iterator<Item = usize>,
) -> T {
    let mut acc = s[first];
    if acc.is_missing() {
        return acc;
    }
    for x in rest.map(|r| s[r]) {
        if x.is_missing() {
            return x;
        }
        acc = float_combine(op, acc, x);
    }
    acc
}

/// One instant fold step (`Keep` / `Replace` / `Max` / `Min`), `NaT` missing.
#[inline]
fn instant_step(op: CombineOp, acc: i64, x: i64) -> Option<i64> {
    if acc == i64::MIN {
        return None;
    }
    if x == i64::MIN {
        return Some(x);
    }
    match op {
        CombineOp::Keep => None,
        CombineOp::Replace => Some(x),
        CombineOp::Max => Some(acc.max(x)),
        _ => Some(acc.min(x)), // Min (a datetime has no Sum)
    }
}

/// `Max` / `Min` / `Sum` of two present integers in `i64`, checked: a result past
/// `dtype` is its overflow error.
#[inline]
fn int_combine<T: Copy + Into<i64> + TryFrom<i64>>(op: CombineOp, acc: T, x: T, dtype: DType) -> Result<T> {
    let (a, b): (i64, i64) = (acc.into(), x.into());
    let v = match op {
        CombineOp::Max => Some(a.max(b)),
        CombineOp::Min => Some(a.min(b)),
        _ => a.checked_add(b), // Sum (endpoints never reach here)
    };
    v.and_then(|v| T::try_from(v).ok()).ok_or_else(|| fold_overflow(dtype))
}

/// One integer fold step into cell `row` (`x` present when `x_valid`).
#[inline]
fn int_step<T: crate::buffer::Element + Copy + Into<i64> + TryFrom<i64>>(
    buf: &mut Buffer<T>,
    val: &mut Validity,
    row: usize,
    op: CombineOp,
    x: T,
    x_valid: bool,
    dtype: DType,
) -> Result<()> {
    if !val.is_valid(row) {
        return Ok(()); // a missing aggregate stays missing
    }
    if !x_valid {
        write_int(buf, val, row, None);
        return Ok(());
    }
    match op {
        CombineOp::Keep => {}
        CombineOp::Replace => buf.make_mut()[row] = x,
        _ => buf.make_mut()[row] = int_combine(op, buf[row], x, dtype)?,
    }
    Ok(())
}

/// `Max` / `Min` / `Sum` of `s` over `first` then `rest`, `None` at the first
/// missing value.
#[inline]
fn int_reduce<T: crate::buffer::Element + Copy + Into<i64> + TryFrom<i64>>(
    op: CombineOp,
    s: &[T],
    sv: &Validity,
    first: usize,
    rest: impl Iterator<Item = usize>,
    dtype: DType,
) -> Result<Option<T>> {
    let dense = !sv.has_nulls();
    if !(dense || sv.is_valid(first)) {
        return Ok(None);
    }
    let mut acc = s[first];
    for r in rest {
        if !(dense || sv.is_valid(r)) {
            return Ok(None);
        }
        acc = int_combine(op, acc, s[r], dtype)?;
    }
    Ok(Some(acc))
}

/// Write an integer aggregate into cell `row`: its value, or a validity hole.
#[inline]
fn write_int<T: crate::buffer::Element + Copy>(buf: &mut Buffer<T>, val: &mut Validity, row: usize, v: Option<T>) {
    let n = buf.len();
    match v {
        Some(v) => {
            buf.make_mut()[row] = v;
            val.set(row, n, true);
        }
        None => val.set(row, n, false),
    }
}

/// The overflow error of an integer fold — off the per-bar hot path.
#[cold]
#[inline(never)]
fn fold_overflow(dtype: DType) -> VolasError {
    crate::numeric::overflow(dtype, "a time-frame fold")
}
