//! `Column` arithmetic, comparison, logical, and reduction operations.

use super::*;

impl Scalar {
    /// The dtype the value is carried in.
    pub fn dtype(&self) -> DType {
        match self {
            Scalar::F64(_) => DType::F64,
            Scalar::F32(_) => DType::F32,
            Scalar::I64(_) => DType::I64,
            Scalar::I32(_) => DType::I32,
            Scalar::Bool(_) => DType::Bool,
        }
    }
}

impl Column {
    /// The single equal-length check every two-column operation calls before any
    /// element access: returns the shared length, or a [`VolasError::Shape`] when
    /// the operands differ — never a truncated result, a panic, or an
    /// out-of-bounds read.
    pub(crate) fn require_same_len(&self, other: &Column) -> Result<usize> {
        let (n, m) = (self.len(), other.len());
        if n == m {
            Ok(n)
        } else {
            Err(VolasError::Shape(format!(
                "operands have unequal lengths: {n} vs {m}"
            )))
        }
    }

    /// Combined validity of `self` and `other` (present only where both are) —
    /// the missing-value rule for a dtype-preserving binary op.
    fn combined_nulls(&self, other: &Column) -> Validity {
        let a = self.validity().cloned().unwrap_or_default();
        let b = other.validity().cloned().unwrap_or_default();
        a.and(&b)
    }

    /// Error if this column is not numeric (`float` / `int` / `bool` — bool acts as
    /// 0/1). The guard for the numeric-only operations (arithmetic, reductions): a
    /// `Str` / `Datetime` column would funnel through `to_f64_vec` to `NaN` (str) or
    /// a quantized epoch (datetime), which the API contract (C4) forbids as a silent
    /// lossy conversion.
    pub fn require_numeric(&self) -> Result<()> {
        if self.dtype().is_numeric() || matches!(self, Column { data: ColumnData::Bool(..) }) {
            Ok(())
        } else {
            Err(VolasError::DType(format!(
                "a numeric operation is not supported on a {} column",
                self.dtype()
            )))
        }
    }

    /// Binary `+ - *` against `other`, dtype-preserving via [`binary_supertype`]
    /// (`int ∘ int → i64`, else f64). `bool ∘ bool` is logical, matching pandas:
    /// `+` is OR, `*` is AND, `-` is an error (numpy disallows bool subtraction);
    /// `bool ∘ number` promotes (bool acts as 0/1). Integer ops are checked: an
    /// overflow is a [`VolasError::Overflow`] (contract #25; pandas wraps). Unequal
    /// lengths are a [`VolasError::Shape`] error.
    pub fn binary(&self, other: &Column, op: BinOp) -> Result<Column> {
        self.require_same_len(other)?;
        self.require_numeric()?;
        other.require_numeric()?;
        match (self, other) {
            (Column { data: ColumnData::F64(a) }, Column { data: ColumnData::F64(b) }) => return Ok(Column::f64(float_slice_kernel(a, b, op))),
            (Column { data: ColumnData::F32(a) }, Column { data: ColumnData::F32(b) }) => return Ok(Column::f32(float_slice_kernel(a, b, op))),
            _ => {}
        }
        if let (Column { data: ColumnData::Bool(a, av) }, Column { data: ColumnData::Bool(b, bv) }) = (self, other) {
            let nulls = av.and(bv);
            return match op {
                BinOp::Add => Ok(Column::bool_masked(
                    a.iter().zip(b.iter()).map(|(&x, &y)| x || y).collect(),
                    nulls,
                )),
                BinOp::Mul => Ok(Column::bool_masked(
                    a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect(),
                    nulls,
                )),
                BinOp::Sub => Err(VolasError::DType(
                    "the `-` operator is not supported for bool columns (use `^`)".into(),
                )),
            };
        }
        match binary_supertype(self.dtype(), other.dtype()) {
            DType::I64 => {
                let nulls = self.combined_nulls(other);
                let out = binary_kernel(&self.as_i64_vec()?, &other.as_i64_vec()?, op, &nulls)?;
                Ok(Column::i64_masked(out, nulls))
            }
            DType::I32 => {
                let nulls = self.combined_nulls(other);
                let out = binary_kernel(&self.as_i32_vec()?, &other.as_i32_vec()?, op, &nulls)?;
                Ok(Column::i32_masked(out, nulls))
            }
            _ => Ok(Column::f64(float_kernel(
                &self.to_f64_domain("arithmetic")?,
                &other.to_f64_domain("arithmetic")?,
                op,
            ))),
        }
    }

    /// True division `self / other` (pandas `/`): always float. `bool / bool` is
    /// an error, matching pandas (division is not defined on bool).
    pub fn div(&self, other: &Column) -> Result<Column> {
        self.require_same_len(other)?;
        self.require_numeric()?;
        other.require_numeric()?;
        match (self, other) {
            (Column { data: ColumnData::F64(a) }, Column { data: ColumnData::F64(b) }) => {
                return Ok(Column::f64(float_output(a, b, |x, y| x / y)));
            }
            (Column { data: ColumnData::F32(a) }, Column { data: ColumnData::F32(b) }) => {
                // True division has always widened BOTH operands before dividing.
                return Ok(Column::f64(float_output(a, b, |x, y| x as f64 / y as f64)));
            }
            _ => {}
        }
        if matches!((self, other), (Column { data: ColumnData::Bool(_, _) }, Column { data: ColumnData::Bool(_, _) })) {
            return Err(VolasError::DType(
                "division is not supported between two bool columns".into(),
            ));
        }
        let (a, b) = (self.to_f64_domain("`/`")?, other.to_f64_domain("`/`")?);
        Ok(Column::f64(
            a.iter().zip(&b).map(|(&x, &y)| x / y).collect(),
        ))
    }

    /// Floor division `self // other` (pandas `//`), dtype-preserving like pandas:
    /// `int // int` stays integer (the supertype, exact even past 2^53) **unless** a
    /// present divisor is `0`, in which case the whole result is float64 (`inf` /
    /// `-inf` / `nan`) — exactly how pandas computes it; anything involving a float
    /// is float64. NA in either operand propagates. `bool // bool` is an error, like
    /// true division.
    pub fn floordiv(&self, other: &Column) -> Result<Column> {
        let n = self.require_same_len(other)?;
        self.require_numeric()?;
        other.require_numeric()?;
        if matches!((self, other), (Column { data: ColumnData::Bool(_, _) }, Column { data: ColumnData::Bool(_, _) })) {
            return Err(VolasError::DType(
                "floor division is not supported between two bool columns".into(),
            ));
        }
        let st = binary_supertype(self.dtype(), other.dtype());
        // A present divisor of 0 forces pandas' float path (`inf` / `-inf` / `nan`).
        let int_path = matches!(st, DType::I64 | DType::I32)
            && !(0..n).any(|i| other.is_valid(i) && other.get_f64(i) == 0.0);
        if int_path {
            let (a, b) = (self.as_i64_vec()?, other.as_i64_vec()?);
            let present = |i: usize| self.is_valid(i) && other.is_valid(i);
            // exact i64 floor division (the divisor is non-zero where present);
            // NA positions hold a 0 placeholder, masked by the result validity.
            // `MIN // -1` overflows its dtype (contract #25), at i64 or at i32.
            let fail = || overflow(st, "`//`");
            let out: Vec<i64> = (0..n)
                .map(|i| if present(i) { ifloordiv(a[i], b[i]).ok_or_else(fail) } else { Ok(0) })
                .collect::<Result<_>>()?;
            let val = Validity::from_valid_iter(n, (0..n).map(present));
            // i32 // i32 stays i32; any i64 operand widens to i64.
            Ok(if st == DType::I32 {
                let narrow = out.iter().map(|&x| i32::try_from(x).map_err(|_| fail()));
                Column::i32_masked(narrow.collect::<Result<_>>()?, val)
            } else {
                Column::i64_masked(out, val)
            })
        } else {
            let (a, b) = (self.to_f64_domain("`//`")?, other.to_f64_domain("`//`")?);
            Ok(Column::f64(
                a.iter().zip(&b).map(|(&x, &y)| (x / y).floor()).collect(),
            ))
        }
    }

    /// Element-wise comparison (pandas `== != < <= > >=`) producing a **non-nullable
    /// bool** mask (decision ②): `str` / `datetime` / `bool` compare by their native
    /// value (NOT through the `to_f64_vec` funnel, which maps `str` to `NaN` and
    /// loses `datetime` precision), numeric kinds compare exactly (#26). A missing slot
    /// (either side NA) follows IEEE — `!=` is `true`, every other op `false`,
    /// matching the existing numeric comparison. Comparing two incompatible kinds
    /// (e.g. `str` vs a number) is a [`VolasError::DType`]; unequal lengths are a
    /// [`VolasError::Shape`].
    pub fn compare(&self, other: &Column, op: CmpOp) -> Result<Column> {
        match (self, other) {
            (Column { data: ColumnData::F64(a) }, Column { data: ColumnData::F64(b) }) => {
                self.require_same_len(other)?;
                Ok(Column::bool(compare_floats(a, b, op)))
            }
            (Column { data: ColumnData::F32(a) }, Column { data: ColumnData::F32(b) }) => {
                self.require_same_len(other)?;
                Ok(Column::bool(compare_floats(a, b, op)))
            }
            _ => self.map_order_rows(other, |o| op.matches_optional(o)).map(Column::bool),
        }
    }

    /// The exact order of `self[i]` against `other[i]` for every row, `None` where
    /// either side is missing — the one comparison owner: `str` lexically,
    /// `datetime` by instant, `bool` with `false < true`, and numeric (or bool vs
    /// numeric) pairs EXACTLY (contract #26): two integer-like sides as `i64`, an
    /// integer against a float by exact order (no float64 round trip, so
    /// `2^53 + 1 > 2^53`), float against float by value. [`compare`](Self::compare)
    /// maps it to IEEE booleans; a directive comparison keeps `None` as a missing
    /// truth (contract #27). Two incompatible kinds are a [`VolasError::DType`];
    /// unequal lengths a [`VolasError::Shape`].
    pub fn order_rows(&self, other: &Column) -> Result<Vec<Option<Ordering>>> {
        self.map_order_rows(other, |o| o)
    }

    /// Dispatch the comparison once and write its final representation directly.
    /// Both IEEE masks and nullable directive orders use this comparison owner.
    fn map_order_rows<T>(&self, other: &Column, map: impl Fn(Option<Ordering>) -> T) -> Result<Vec<T>> {
        let n = self.require_same_len(other)?;
        let numeric_like = |c: &Column| c.dtype().is_numeric() || matches!(c, Column { data: ColumnData::Bool(..) });
        Ok(match (self, other) {
            (Column { data: ColumnData::F64(a) }, Column { data: ColumnData::F64(b) }) => mapped_orders(n, &map, |i| a[i].partial_cmp(&b[i])),
            (Column { data: ColumnData::F32(a) }, Column { data: ColumnData::F32(b) }) => mapped_orders(n, &map, |i| a[i].partial_cmp(&b[i])),
            (Column { data: ColumnData::Str(a, av) }, Column { data: ColumnData::Str(b, bv) }) => mapped_orders(n, &map, |i| {
                // SAFETY: `i < n` and `require_same_len` proved both columns have length `n`.
                (av.is_valid(i) && bv.is_valid(i))
                    .then(|| unsafe { a.get_unchecked(i).cmp(b.get_unchecked(i)) })
            }),
            (Column { data: ColumnData::Datetime(a) }, Column { data: ColumnData::Datetime(b) }) => {
                mapped_orders(n, &map, |i| (a[i] != i64::MIN && b[i] != i64::MIN).then(|| a[i].cmp(&b[i])))
            }
            (Column { data: ColumnData::Bool(a, av) }, Column { data: ColumnData::Bool(b, bv) }) => {
                mapped_orders(n, &map, |i| (av.is_valid(i) && bv.is_valid(i)).then(|| a[i].cmp(&b[i])))
            }
            (a, b) if numeric_like(a) && numeric_like(b) => match (a.is_float_like(), b.is_float_like()) {
                (true, true) => {
                    mapped_orders(n, &map, |i| a.get_f64(i).partial_cmp(&b.get_f64(i)))
                }
                (false, false) => mapped_orders(n, &map, |i| {
                    (a.is_valid(i) && b.is_valid(i)).then(|| a.get_i64(i).cmp(&b.get_i64(i)))
                }),
                (false, true) => {
                    mapped_orders(n, &map, |i| a.is_valid(i).then(|| cmp_i64_f64(a.get_i64(i), b.get_f64(i))).flatten())
                }
                (true, false) => mapped_orders(n, &map, |i| {
                    b.is_valid(i)
                        .then(|| cmp_i64_f64(b.get_i64(i), a.get_f64(i)).map(Ordering::reverse))
                        .flatten()
                }),
            },
            _ => {
                return Err(VolasError::DType(format!(
                    "cannot compare a {} column with a {} column",
                    self.dtype(),
                    other.dtype()
                )))
            }
        })
    }

    /// A column of per-column reduction results (a DataFrame `sum` / `min` / …),
    /// each part `(dtype it contributes, value)`. The column's dtype is the
    /// supertype of the parts (all int64 → int64, int32 → int32, all bool → bool, any float →
    /// float), and every value converts to it losslessly (contract #20 / #26): an
    /// integer past the float's exact range raises rather than rounding. A `NaN`
    /// value (an all-missing reduction) becomes the target's missing cell.
    pub fn from_scalars(parts: &[(DType, Scalar)]) -> Result<Column> {
        let target = if !parts.is_empty() && parts.iter().all(|p| p.0 == DType::Bool) {
            DType::Bool // bool min / max stay bool (pandas)
        } else {
            parts.iter().map(|p| p.0).reduce(binary_supertype).unwrap_or(DType::F64)
        };
        let mut out = Column::na_of(target, 0);
        for &(_, s) in parts {
            let cell = match s {
                Scalar::F64(x) if x.is_nan() => Column::na_of(target, 1),
                Scalar::F64(x) => Column::f64(vec![x]),
                Scalar::F32(x) => Column::f32(vec![x]),
                Scalar::I64(x) => Column::i64(vec![x]),
                Scalar::I32(x) => Column::i32(vec![x]),
                Scalar::Bool(b) => Column::bool(vec![b]),
            };
            out.append(&cell.convert_lossless(target)?)?;
        }
        Ok(out)
    }

    /// Whether the values are floats (`F64` / `F32`) rather than integer-like.
    fn is_float_like(&self) -> bool {
        self.dtype().is_float()
    }

    /// Whether the values are monotonically non-decreasing (`increasing`) or
    /// non-increasing (pandas `is_monotonic_*`), by each dtype's own order through
    /// [`compare`](Self::compare) — exact for int64, lexical for str, by instant for
    /// datetime. Any missing cell makes it false (pandas).
    pub fn is_monotonic(&self, increasing: bool) -> bool {
        let n = self.len();
        if self.has_missing() {
            return false;
        }
        if n < 2 {
            return true;
        }
        let op = if increasing { CmpOp::Ge } else { CmpOp::Le };
        let ok = self.slice(1, n).compare(&self.slice(0, n - 1), op).expect("a column compares with itself");
        ok.as_bool().expect("compare yields bool").iter().all(|&b| b)
    }

    /// Three-valued logical `and` / `or` / `xor` (pandas `&` / `|` / `^`), Kleene
    /// semantics over [`truth_at`](Self::truth_at): a present `false` makes `and`
    /// false and a present `true` makes `or` true even when the other side is
    /// missing; otherwise a missing operand (a NaN, an int / bool NA) yields NA —
    /// never a present value (contract #27). Unequal lengths are a
    /// [`VolasError::Shape`] error.
    pub fn logical(&self, other: &Column, op: BoolOp) -> Result<Column> {
        let n = self.require_same_len(other)?;
        Ok(Column::bool_from_options(
            (0..n).map(|i| match (op, self.truth_at(i), other.truth_at(i)) {
                (BoolOp::And, Some(false), _) | (BoolOp::And, _, Some(false)) => Some(false),
                (BoolOp::Or, Some(true), _) | (BoolOp::Or, _, Some(true)) => Some(true),
                (BoolOp::And, Some(a), Some(b)) => Some(a && b),
                (BoolOp::Or, Some(a), Some(b)) => Some(a || b),
                (BoolOp::Xor, Some(a), Some(b)) => Some(a ^ b),
                _ => None,
            }),
        ))
    }

    /// Logical NOT (pandas `~`) over [`truth_at`](Self::truth_at): NA in -> NA out.
    pub fn not(&self) -> Column {
        let n = self.len();
        Column::bool_from_options((0..n).map(|i| self.truth_at(i).map(|a| !a)))
    }

    /// The truth value of element `i`, or `None` when it is missing — the one
    /// validity-aware boolean conversion (contract #27) behind the logical ops,
    /// `bool()` of a one-element Series, masks and `any` / `all`. A bool reads its
    /// value, a number `x != 0`, a str `!s.is_empty()`, a datetime `ns != 0`
    /// (Python truthiness); a missing cell (NaN / NA / NaT) is never read as a
    /// value, whatever payload it carries.
    pub fn truth_at(&self, i: usize) -> Option<bool> {
        if !self.is_valid(i) {
            return None;
        }
        Some(match self {
            Column { data: ColumnData::Bool(v, _) } => v[i],
            Column { data: ColumnData::Str(v, _) } => !v.get(i).is_empty(),
            other => other.get_f64(i) != 0.0,
        })
    }

    /// Sum, dtype-preserving (pandas): a float column -> float, int/bool -> i64
    /// (bool counts trues). Computed natively (i64 in i64, exact past 2^53) and
    /// checked: an int64 overflow is an error (contract #25).
    pub fn sum(&self) -> Result<Scalar> {
        Ok(match self {
            Column { data: ColumnData::F64(v) } => Scalar::F64(stats::sum(v.as_slice())?),
            Column { data: ColumnData::F32(v) } => Scalar::F32(stats::sum(v.as_slice())?),
            Column { data: ColumnData::I64(v, val) } => Scalar::I64(sum_valid(v, val)?),
            // int32 / bool sum promotes to int64 (pandas / numpy accumulator)
            Column { data: ColumnData::I32(v, val) } => Scalar::I64(sum_valid(&widen_i64(v), val)?),
            Column { data: ColumnData::Bool(v, val) } => Scalar::I64(sum_valid(&widen_i64(v), val)?),
            other => Scalar::F64(stats::sum(&other.to_f64_vec())?),
        })
    }

    /// Product, dtype-preserving (float -> float, int/bool -> i64), checked like
    /// [`sum`](Self::sum).
    pub fn prod(&self) -> Result<Scalar> {
        Ok(match self {
            Column { data: ColumnData::F64(v) } => Scalar::F64(stats::prod(v.as_slice())?),
            Column { data: ColumnData::F32(v) } => Scalar::F32(stats::prod(v.as_slice())?),
            Column { data: ColumnData::I64(v, val) } => Scalar::I64(prod_valid(v, val)?),
            Column { data: ColumnData::I32(v, val) } => Scalar::I64(prod_valid(&widen_i64(v), val)?),
            Column { data: ColumnData::Bool(v, val) } => Scalar::I64(prod_valid(&widen_i64(v), val)?),
            other => Scalar::F64(stats::prod(&other.to_f64_vec())?),
        })
    }

    /// Minimum (`want_max = false`) / maximum, dtype-preserving (float -> float,
    /// int -> i64, bool -> bool). Empty / all-missing -> `F64(NaN)`.
    pub fn extreme(&self, want_max: bool) -> Scalar {
        match self {
            Column { data: ColumnData::I64(v, val) } => match extreme_valid(v, val, want_max) {
                Some(x) => Scalar::I64(x),
                None => Scalar::F64(f64::NAN),
            },
            Column { data: ColumnData::I32(v, val) } => match extreme_valid(v, val, want_max) {
                Some(x) => Scalar::I32(x),
                None => Scalar::F64(f64::NAN),
            },
            Column { data: ColumnData::F32(v) } => {
                Scalar::F32(stats::extreme(v.as_slice(), want_max).unwrap_or(f32::NAN))
            }
            Column { data: ColumnData::Bool(v, val) } => {
                // min = all (AND) / max = any (OR), over present values; none -> NaN
                let present = |i: usize| val.is_valid(i);
                if !(0..v.len()).any(present) {
                    Scalar::F64(f64::NAN)
                } else if want_max {
                    Scalar::Bool((0..v.len()).any(|i| present(i) && v[i]))
                } else {
                    Scalar::Bool((0..v.len()).all(|i| !present(i) || v[i]))
                }
            }
            Column { data: ColumnData::F64(v) } => {
                Scalar::F64(stats::extreme(v.as_slice(), want_max).unwrap_or(f64::NAN))
            }
            other => Scalar::F64(stats::extreme(&other.to_f64_vec(), want_max).unwrap_or(f64::NAN)),
        }
    }
}


/// Keep both closures statically dispatched so homogeneous comparisons can
/// vectorize, without an indirect call per cell or an intermediate order vector.
#[inline]
fn mapped_orders<T>(
    n: usize,
    map: &impl Fn(Option<Ordering>) -> T,
    key: impl Fn(usize) -> Option<Ordering>,
) -> Vec<T> {
    (0..n).map(key).map(map).collect()
}

/// Native IEEE predicates for homogeneous floats. Select the operator once and
/// write the bools directly, without computing a three-way order per value.
/// The caller has checked the slice lengths.
fn compare_floats<T: PartialOrd>(a: &[T], b: &[T], op: CmpOp) -> Vec<bool> {
    let pairs = || a.iter().zip(b);
    match op {
        CmpOp::Eq => pairs().map(|(x, y)| x == y).collect(),
        CmpOp::Ne => pairs().map(|(x, y)| x != y).collect(),
        CmpOp::Lt => pairs().map(|(x, y)| x < y).collect(),
        CmpOp::Le => pairs().map(|(x, y)| x <= y).collect(),
        CmpOp::Gt => pairs().map(|(x, y)| x > y).collect(),
        CmpOp::Ge => pairs().map(|(x, y)| x >= y).collect(),
    }
}
