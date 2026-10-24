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
//!   skipped value, non-finite inputs included, whatever payload a missing cell holds;
//! - an integer `sum` is checked (contract #25): an overflow raises;
//! - a source of another dtype converts to the cell's dtype losslessly first
//!   (contract #20) — never truncated or saturated.

use super::*;
use crate::buffer::Float;

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
    /// Whether this cell is a missing aggregation input. Raw storage validity is
    /// unchanged; floating inputs use the shared non-finite computation rule.
    #[inline]
    pub fn is_missing_input(&self, row: usize) -> bool {
        match self {
            Column::F64(s) => s[row].is_missing_value(),
            Column::F32(s) => s[row].is_missing_value(),
            _ => !self.is_valid(row),
        }
    }

    /// Reduce ordered, nonempty groups into one typed output buffer. Each group is
    /// a slice of source row positions (duplicates can already have been removed).
    /// The caller owns grouping; the fold and its exact arithmetic stay here. An
    /// error carries its group number so column-wise callers can preserve the
    /// period-then-column error priority without materializing period frames.
    /// The repeatable group iterator also supplies a density estimate, so a shared
    /// finite-input fact is learned once without scanning a sparse source.
    pub fn reduce_groups<'a>(
        &self,
        op: CombineOp,
        groups: impl ExactSizeIterator<Item = &'a [usize]> + Clone,
    ) -> std::result::Result<Column, (usize, VolasError)> {
        self.reduce_groups_with_spare(op, groups, 0, 0)
    }

    /// The grouped fold with capacity for a following append. `spare_bytes` is
    /// the additional UTF-8 payload size; fixed-width columns only use `spare_rows`.
    pub fn reduce_groups_with_spare<'a>(
        &self,
        op: CombineOp,
        groups: impl ExactSizeIterator<Item = &'a [usize]> + Clone,
        spare_rows: usize,
        spare_bytes: usize,
    ) -> std::result::Result<Column, (usize, VolasError)> {
        if !op.supports(self.dtype()) {
            return Err((0, op.unsupported(self.dtype())));
        }
        let finite = match self {
            Column::F64(s) => finite_group_inputs(s, &groups),
            Column::F32(s) => finite_group_inputs(s, &groups),
            _ => false,
        };
        if op.is_endpoint() {
            return Ok(self.endpoint_columns(op, groups, spare_rows, spare_bytes, finite));
        }
        Ok(match self {
            Column::F64(s) => Column::f64(collect_with_spare(groups.map(|r| float_reduce(op, s, r[0], r[1..].iter().copied(), finite)), spare_rows)),
            Column::F32(s) => Column::f32(collect_with_spare(groups.map(|r| float_reduce(op, s, r[0], r[1..].iter().copied(), finite)), spare_rows)),
            Column::I64(s, sv) => {
                let (values, valid) = int_groups(op, s, sv, groups, DType::I64, spare_rows)?;
                Column::i64_masked(values, valid)
            }
            Column::I32(s, sv) => {
                let (values, valid) = int_groups(op, s, sv, groups, DType::I32, spare_rows)?;
                Column::i32_masked(values, valid)
            }
            Column::Datetime(s) => Column::datetime(collect_with_spare(groups.map(|r| instant_reduce(op, s, r[0], r[1..].iter().copied())), spare_rows)),
            _ => unreachable!("the support table admits no other pair"), // LCOV_EXCL_LINE
        })
    }

    /// Endpoints write directly to the final typed values. Only strings need a
    /// position pass to learn the exact UTF-8 byte capacity before gathering.
    fn endpoint_columns<'a>(
        &self, op: CombineOp, groups: impl ExactSizeIterator<Item = &'a [usize]>,
        spare: usize, bytes: usize, finite: bool,
    ) -> Column {
        match self {
            Column::F64(s) => {
                let (values, _, missing) = float_endpoint_values(op, groups, s, spare, finite);
                let mut out = Column::f64(values);
                // Match the original builder's learned NaN fact as well as its bits.
                if let (Some(row), Column::F64(v)) = (missing, &mut out) { v.set(row, f64::NAN); }
                out
            }
            Column::F32(s) => {
                let (values, _, missing) = float_endpoint_values(op, groups, s, spare, finite);
                let mut out = Column::f32(values);
                if let (Some(row), Column::F32(v)) = (missing, &mut out) { v.set(row, f32::NAN); }
                out
            }
            Column::Datetime(s) => {
                let (values, _, _) = endpoint_values(op, groups, s, |r| s[r] != i64::MIN, |_| i64::MIN, spare);
                Column::datetime(values)
            }
            Column::I64(s, v) => {
                let (values, valid, _) = endpoint_values(op, groups, s, |r| v.is_valid(r), |x| x, spare);
                Column::i64_masked(values, valid)
            }
            Column::I32(s, v) => {
                let (values, valid, _) = endpoint_values(op, groups, s, |r| v.is_valid(r), |x| x, spare);
                Column::i32_masked(values, valid)
            }
            Column::Bool(s, v) => {
                let (values, valid, _) = endpoint_values(op, groups, s, |r| v.is_valid(r), |x| x, spare);
                Column::bool_masked(values, valid)
            }
            Column::Str(_, v) => {
                let (positions, valid) = endpoint_groups(op, groups, |r| v.is_valid(r));
                let mut out = self.take_with_spare(&positions, spare, bytes);
                if let Column::Str(_, v) = &mut out { *v = valid; }
                out
            }
        }
    }

    /// Set cell `row` to the aggregate of the single value `src[src_row]` under
    /// `op`: that value with its presence (a one-value period's aggregate is the
    /// value). [`combine_at`](Self::combine_at) then folds each later value in. A
    /// `Str` cell is written in place only as the column's last row (the forming
    /// row; a [`VolasError::Shape`] error elsewhere).
    pub fn init_at(&mut self, row: usize, op: CombineOp, src: &Column, src_row: usize) -> Result<()> {
        if src.dtype() != self.dtype() || !op.supports(self.dtype()) {
            return self.fold_converted(row, op, src, src_row, true);
        }
        if src.is_missing_input(src_row) {
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
            (Column::F64(buf), Column::F64(s)) => float_fold(buf, row, op, s[r]),
            (Column::F32(buf), Column::F32(s)) => float_fold(buf, row, op, s[r]),
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

    /// Continue a batch reduction from its prefix aggregate. `had_missing`
    /// distinguishes a reduction stopped by a non-finite input from an IEEE
    /// intermediate produced by finite inputs, which must still visit later inputs. Endpoints use
    /// the same canonical missing value as `reduce_into`. This preserves even NaN
    /// payloads and signed zeros when a cached prefix replaces a full re-reduction.
    pub fn combine_reduced_at(&mut self, row: usize, op: CombineOp, src: &Column, r: usize, had_missing: bool) -> Result<()> {
        if op.is_endpoint() && (!self.is_valid(row) || src.is_missing_input(r)) {
            return self.set_missing_at(row);
        }
        if !op.is_endpoint() {
            match (&mut *self, src) {
                (Column::F64(a), Column::F64(b)) => {
                    if !had_missing { a.set(row, if b[r].is_missing_value() { missing_float(b[r]) } else { float_combine(op, a[row], b[r]).unwrap_or(a[row]) }); }
                    return Ok(());
                }
                (Column::F32(a), Column::F32(b)) => {
                    if !had_missing { a.set(row, if b[r].is_missing_value() { missing_float(b[r]) } else { float_combine(op, a[row], b[r]).unwrap_or(a[row]) }); }
                    return Ok(());
                }
                _ => {}
            }
        }
        self.combine_at(row, op, src, r)
    }

    /// Check, writing nothing, that folding `cells` — source `(column, row)` pairs in
    /// fold order — into cell `row` under `op` succeeds: from the cell's value
    /// ([`combine_at`](Self::combine_at) each), or with `restart` from the first cell
    /// ([`init_at`](Self::init_at) then `combine_at` — a period's re-reduction). The
    /// fold's failures, as the fold itself raises them: an unsupported `op`, a source
    /// value that does not convert losslessly to this column's dtype, an integer
    /// overflow, a `Str` cell other than the last row. A multi-column fold checks every
    /// column with this before its first write, so a rejected fold changes nothing.
    #[inline(never)]
    pub fn check_fold<'c>(
        &self,
        row: usize,
        op: CombineOp,
        restart: bool,
        cells: impl IntoIterator<Item = (&'c Column, usize)>,
    ) -> Result<()> {
        self.checked_fold_value(row, op, restart, cells).map(|_| ())
    }

    /// The integer aggregate already calculated while checking a fold. Frame
    /// admission reuses it to validate historical dtype compatibility before writing.
    #[inline(never)]
    pub(crate) fn checked_fold_value<'c>(
        &self, row: usize, op: CombineOp, restart: bool,
        cells: impl IntoIterator<Item = (&'c Column, usize)>,
    ) -> Result<Option<i64>> {
        let dtype = self.dtype();
        if !op.supports(dtype) {
            return Err(op.unsupported(dtype));
        }
        if matches!(self, Column::Str(..)) && row + 1 != self.len() {
            return Err(str_not_last(row, self.len()));
        }
        // the integer aggregate (`None` once missing); only an integer fold can overflow
        let mut acc = match self {
            Column::I64(buf, val) => val.is_valid(row).then(|| buf[row]),
            Column::I32(buf, val) => val.is_valid(row).then(|| i64::from(buf[row])),
            _ => None,
        };
        let integer = matches!(dtype, DType::I64 | DType::I32);
        let mut first = restart;
        for (src, r) in cells {
            let converted;
            let (src, r) = if src.dtype() == dtype {
                (src, r)
            } else {
                converted = src.slice(r, r + 1).convert_lossless(dtype).map_err(|e| self.append_error(src, e))?;
                (&converted, 0)
            };
            if integer {
                let x = src.is_valid(r).then(|| src.get_i64(r));
                acc = match (first, acc, x) {
                    (true, _, x) => x,
                    (false, Some(a), Some(b)) => match op {
                        CombineOp::Keep => Some(a),
                        CombineOp::Replace => Some(b),
                        _ if dtype == DType::I32 => {
                            Some(i64::from(int_combine(op, a as i32, b as i32, dtype)?))
                        }
                        _ => Some(int_combine(op, a, b, dtype)?),
                    },
                    _ => None, // a missing aggregate stays missing; a missing value makes it
                };
            }
            first = false;
        }
        Ok(acc)
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
                Column::F64(s) => float_endpoint_scan(s, first, rows),
                Column::F32(s) => float_endpoint_scan(s, first, rows),
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
            return self.write_present(row, src, endpoint_position(op, first, last, true));
        }
        match (&mut *self, src) {
            (Column::F64(buf), Column::F64(s)) => {
                let finite = finite_inputs(s, rows.size_hint().0.saturating_add(1));
                buf.set(row, float_reduce(op, s, first, rows, finite));
            },
            (Column::F32(buf), Column::F32(s)) => {
                let finite = finite_inputs(s, rows.size_hint().0.saturating_add(1));
                buf.set(row, float_reduce(op, s, first, rows, finite));
            },
            (Column::I64(buf, val), Column::I64(s, sv)) => {
                let v = int_reduce(op, s, sv, first, rows, DType::I64)?;
                write_int(buf, val, row, v)
            }
            (Column::I32(buf, val), Column::I32(s, sv)) => {
                let v = int_reduce(op, s, sv, first, rows, DType::I32)?;
                write_int(buf, val, row, v)
            }
            (Column::Datetime(buf), Column::Datetime(s)) => {
                buf.make_mut()[row] = instant_reduce(op, s, first, rows);
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

fn endpoint_groups<'a>(op: CombineOp, groups: impl ExactSizeIterator<Item = &'a [usize]>, present: impl Fn(usize) -> bool) -> (Vec<usize>, Validity) {
    let n = groups.len();
    let mut valid = Validity::default();
    let positions = groups.enumerate().map(|(i, rows)| {
        let first = rows[0];
        let (present, last) = endpoint_scan(first, rows[1..].iter().copied(), &present);
        valid.set(i, n, present);
        endpoint_position(op, first, last, present)
    }).collect();
    (positions, valid)
}

fn int_groups<'a, T: crate::buffer::Element + Copy + Into<i64> + TryFrom<i64>>(
    op: CombineOp,
    s: &[T],
    sv: &Validity,
    groups: impl ExactSizeIterator<Item = &'a [usize]>,
    dtype: DType,
    spare: usize,
) -> std::result::Result<(Vec<T>, Validity), (usize, VolasError)> {
    let n = groups.len();
    let mut valid = Validity::default();
    let mut values = Vec::with_capacity(reserved_capacity::<T>(n, spare));
    for (i, rows) in groups.enumerate() {
        let value = int_reduce(op, s, sv, rows[0], rows[1..].iter().copied(), dtype).map_err(|e| (i, e))?;
        valid.set(i, n, value.is_some());
        values.push(value.unwrap_or(s[rows[0]]));
    }
    Ok((values, valid))
}

fn instant_reduce(op: CombineOp, s: &[i64], first: usize, rest: impl Iterator<Item = usize>) -> i64 {
    let mut acc = s[first];
    for x in std::iter::once(first).chain(rest).map(|r| s[r]) {
        if x == i64::MIN { return i64::MIN; }
        acc = if op == CombineOp::Max { acc.max(x) } else { acc.min(x) };
    }
    acc
}

/// The error of a `Str` fold into a cell other than the last row (a string column
/// is rewritten in place only at its end).
#[cold]
#[inline(never)]
fn str_not_last(row: usize, n: usize) -> VolasError {
    VolasError::Shape(format!("a str cell folds in place only at the last row ({row} of {n})"))
}

/// Literal endpoint selection, shared by the dense and missing-aware paths.
#[inline]
fn endpoint_position(op: CombineOp, first: usize, last: usize, present: bool) -> usize {
    if present && op == CombineOp::Replace { last } else { first }
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

/// Storage sentinels and computation inputs have different domains. Keep
/// Numeric's IEEE/storage semantics; delegate only input classification here.
trait FloatInput: Numeric + Float {
    const NAN: Self;
}

impl FloatInput for f64 { const NAN: Self = f64::NAN; }
impl FloatInput for f32 { const NAN: Self = f32::NAN; }

/// Reuse the buffer owner's fact. Only dense selections may trigger its SIMD
/// scan: at most two source rows per selected row; sparse selections remain O(k).
fn finite_inputs<T: Float>(s: &Buffer<T>, selected: usize) -> bool {
    match s.known_has_nonfinite() {
        Some(has_missing) => !has_missing,
        None if selected >= s.len().div_ceil(2) => !s.has_nonfinite(),
        None => false,
    }
}

fn finite_group_inputs<'a, T: Float, G: Iterator<Item = &'a [usize]> + Clone>(s: &Buffer<T>, groups: &G) -> bool {
    let selected = if s.known_has_nonfinite().is_some() { 0 }
        else { groups.clone().fold(0usize, |n, rows| n.saturating_add(rows.len())) };
    finite_inputs(s, selected)
}

fn float_endpoint_scan<T: Float>(s: &Buffer<T>, first: usize, rest: impl Iterator<Item = usize>) -> (bool, usize) {
    if finite_inputs(s, rest.size_hint().0.saturating_add(1)) {
        endpoint_scan(first, rest, |r| { let _ = s[r]; true })
    } else {
        endpoint_scan(first, rest, |r| !s[r].is_missing_value())
    }
}

fn float_endpoint_values<'a, T: FloatInput>(
    op: CombineOp, groups: impl ExactSizeIterator<Item = &'a [usize]>,
    s: &Buffer<T>, spare: usize, finite: bool,
) -> (Vec<T>, Validity, Option<usize>) {
    if finite {
        let values = collect_with_spare(groups.map(|rows| {
            let first = rows[0];
            let last = rows[rows.len() - 1];
            let len = s.len();
            let in_bounds = rows.iter().fold(true, |ok, &r| ok & (r < len));
            assert!(in_bounds, "aggregation row is outside the column");
            s[endpoint_position(op, first, last, true)]
        }), spare);
        (values, Validity::default(), None)
    } else {
        endpoint_values(op, groups, s, |r| !s[r].is_missing_value(), |_| T::NAN, spare)
    }
}

/// Preserve a NaN input's payload, but normalize infinities to missing output.
#[cold]
#[inline(never)]
fn missing_float<T: FloatInput>(x: T) -> T {
    if x.is_missing() { x } else { T::NAN }
}

/// Complete the rare input write outside the finite fold, preserving a missing
/// aggregate's bits and a NaN input's payload.
#[cold]
#[inline(never)]
fn write_missing_float<T: FloatInput>(buf: &mut Buffer<T>, row: usize, x: T) {
    if !buf[row].is_missing() { buf.set(row, missing_float(x)); }
}

/// Read and write the destination through one checked mutable cell. `Keep`
/// avoids copy-on-write entirely; non-finite inputs stay on the cold write path.
#[inline]
fn float_fold<T: FloatInput + std::ops::Add<Output = T>>(buf: &mut Buffer<T>, row: usize, op: CombineOp, x: T) {
    if x.is_missing_value() {
        write_missing_float(buf, row, x);
    } else if op == CombineOp::Keep {
        let _ = buf[row]; // validate the destination even when no write is needed
    } else {
        let cell = &mut buf.make_mut()[row];
        if let Some(v) = float_step(op, *cell, x) { *cell = v; }
    }
}

/// One float fold step for a finite input. A NaN aggregate stays missing;
/// `Keep` leaves its buffer untouched.
#[inline]
fn float_step<T: FloatInput + std::ops::Add<Output = T>>(op: CombineOp, acc: T, x: T) -> Option<T> {
    if acc.is_missing() { return None; }
    match op {
        CombineOp::Keep => None,
        CombineOp::Replace => Some(x),
        _ => float_combine(op, acc, x),
    }
}

/// Arithmetic on two present floats (ordered extrema, IEEE sum). Endpoints
/// dispatch separately, so they never evaluate unrelated arithmetic. An unchanged
/// extremum needs no write; batch reduction keeps its accumulator in that case.
#[inline]
fn float_combine<T: Numeric + std::ops::Add<Output = T>>(op: CombineOp, acc: T, x: T) -> Option<T> {
    match op {
        CombineOp::Max => if x > acc { Some(x) } else { None },
        CombineOp::Min => if x < acc { Some(x) } else { None },
        _ => Some(acc + x), // Sum (endpoints never reach here)
    }
}

/// `Max` / `Min` / `Sum` of `s` over `first` then `rest` — `NaN` at the first
/// missing value.
#[inline]
fn float_reduce<T: FloatInput + std::ops::Add<Output = T>>(
    op: CombineOp,
    s: &[T],
    first: usize,
    rest: impl Iterator<Item = usize>,
    finite: bool,
) -> T {
    if !finite { return float_reduce_missing(op, s, first, rest); }
    match op {
        CombineOp::Keep => float_reduce_by(s, first, rest, |a, _| a),
        CombineOp::Replace => float_reduce_by(s, first, rest, |_, x| x),
        CombineOp::Max => float_reduce_by(s, first, rest, |a, x| float_combine(CombineOp::Max, a, x).unwrap_or(a)),
        CombineOp::Min => float_reduce_by(s, first, rest, |a, x| float_combine(CombineOp::Min, a, x).unwrap_or(a)),
        CombineOp::Sum => float_reduce_by(s, first, rest, |a, x| float_combine(CombineOp::Sum, a, x).unwrap_or(a)),
    }
}

/// Dispatch the operation once; the loop retains the scalar IEEE order.
#[inline]
fn float_reduce_by<T: Copy>(s: &[T], first: usize, rest: impl Iterator<Item = usize>, combine: impl Fn(T, T) -> T) -> T {
    let mut acc = s[first];
    for r in rest { acc = combine(acc, s[r]); }
    acc
}

/// Non-finite repair stays outside the all-finite numerical loop.
#[cold]
#[inline(never)]
fn float_reduce_missing<T: FloatInput + std::ops::Add<Output = T>>(
    op: CombineOp, s: &[T], first: usize, rest: impl Iterator<Item = usize>,
) -> T {
    let mut acc = s[first];
    if acc.is_missing_value() {
        return missing_float(acc);
    }
    for x in rest.map(|r| s[r]) {
        if x.is_missing_value() {
            return missing_float(x);
        }
        acc = float_combine(op, acc, x).unwrap_or(acc);
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

/// Materialize an exact-length iterator once, retaining spare append capacity.
pub(super) fn collect_with_spare<T>(values: impl ExactSizeIterator<Item = T>, spare: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(reserved_capacity::<T>(values.len(), spare));
    out.extend(values);
    out
}

/// Keep the normal Vec growth minimum for a buffer that will continue appending.
fn reserved_capacity<T>(len: usize, spare: usize) -> usize {
    let capacity = len + spare;
    if spare == 0 { capacity } else { capacity.max(if std::mem::size_of::<T>() == 1 { 8 } else { 4 }) }
}

fn endpoint_values<'a, T: Copy>(
    op: CombineOp, groups: impl ExactSizeIterator<Item = &'a [usize]>, source: &[T],
    present: impl Fn(usize) -> bool, absent: impl Fn(T) -> T, spare: usize,
) -> (Vec<T>, Validity, Option<usize>) {
    let n = groups.len();
    let mut values = Vec::with_capacity(reserved_capacity::<T>(n, spare));
    let mut valid = Validity::default();
    let mut missing = None;
    for (i, rows) in groups.enumerate() {
        let first = rows[0];
        let (present, last) = endpoint_scan(first, rows[1..].iter().copied(), &present);
        let row = endpoint_position(op, first, last, present);
        valid.set(i, n, present);
        values.push(if present { source[row] } else { missing.get_or_insert(i); absent(source[row]) });
    }
    (values, valid, missing)
}

#[cfg(test)]
mod float_operation_tests {
    use super::{float_step, float_reduce, CombineOp};

    #[test]
    fn literal_endpoints_and_ordered_ties_keep_their_zero_bits() {
        let negative = 1_u64 << 63;
        let cases = [
            (CombineOp::Keep, [negative, 0]),
            (CombineOp::Replace, [0, negative]),
            (CombineOp::Max, [negative, 0]),
            (CombineOp::Min, [negative, 0]),
            (CombineOp::Sum, [0, 0]),
        ];
        for (op, expected) in cases {
            assert_eq!([
                float_reduce(op, &[-0.0_f64, 0.0], 0, std::iter::once(1), true).to_bits(),
                float_reduce(op, &[0.0_f64, -0.0], 0, std::iter::once(1), true).to_bits(),
            ], expected, "{op:?} reduced float64");
            assert_eq!([
                float_reduce(op, &[-0.0_f32, 0.0], 0, std::iter::once(1), true).to_bits(),
                float_reduce(op, &[0.0_f32, -0.0], 0, std::iter::once(1), true).to_bits(),
            ], expected.map(|bits| (bits >> 32) as u32), "{op:?} reduced float32");
            assert_eq!([
                float_step(op, -0.0_f64, 0.0).unwrap_or(-0.0).to_bits(),
                float_step(op, 0.0_f64, -0.0).unwrap_or(0.0).to_bits(),
            ], expected, "{op:?} float64");
            assert_eq!([
                float_step(op, -0.0_f32, 0.0).unwrap_or(-0.0).to_bits(),
                float_step(op, 0.0_f32, -0.0).unwrap_or(0.0).to_bits(),
            ], expected.map(|bits| (bits >> 32) as u32), "{op:?} float32");
        }
    }
}

#[cfg(test)]
mod aggregation_fact_tests {
    use super::*;

    #[test]
    fn dense_groups_learn_finite_inputs_without_scanning_sparse_columns() {
        for col in [Column::f64(vec![1.; 100]), Column::f32(vec![1.; 100])] {
            let known = Column::known_has_nonfinite;
            assert_eq!(known(&col), None);
            let sparse = [vec![0, 99]];
            let out = col.reduce_groups(CombineOp::Sum, sparse.iter().map(Vec::as_slice)).unwrap();
            assert_eq!(out.get_f64(0), 2.);
            assert_eq!(known(&col), None, "a sparse selection must not scan the source");
            let dense = [(0..100).collect::<Vec<_>>()];
            let out = col.reduce_groups(CombineOp::Sum, dense.iter().map(Vec::as_slice)).unwrap();
            assert_eq!(out.get_f64(0), 100.);
            assert_eq!(known(&col), Some(false), "dense groups reuse one finite-input fact");
        }
    }

    #[test]
    fn a_nonfinite_value_outside_the_groups_does_not_poison_selected_inputs() {
        for col in [Column::f64(vec![1., 2., f64::INFINITY, 4.]),
                    Column::f32(vec![1., 2., f32::INFINITY, 4.])] {
            let groups = [vec![0, 1], vec![3]];
            let out = col.reduce_groups(CombineOp::Sum, groups.iter().map(Vec::as_slice)).unwrap();
            assert_eq!(out.get_f64(0), 3.);
            assert_eq!(out.get_f64(1), 4.);
        }
    }
}

#[cfg(test)]
mod endpoint_bounds_tests {
    use super::*;

    #[test]
    #[should_panic]
    fn finite_endpoints_validate_the_middle_positions_too() {
        let col = Column::f64(vec![1.; 100]);
        let dense = [(0..100).collect::<Vec<_>>()];
        col.reduce_groups(CombineOp::Sum, dense.iter().map(Vec::as_slice)).unwrap();
        let invalid = [vec![0, 1000, 1]];
        col.reduce_groups(CombineOp::Keep, invalid.iter().map(Vec::as_slice)).unwrap();
    }
}
