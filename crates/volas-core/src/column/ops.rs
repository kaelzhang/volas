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
        if self.dtype().is_numeric() || matches!(self, Column::Bool(..)) {
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
        if let (Column::Bool(a, av), Column::Bool(b, bv)) = (self, other) {
            let nulls = av.and(bv);
            return match op {
                BinOp::Add => Ok(Column::bool_with(
                    a.iter().zip(b.iter()).map(|(&x, &y)| x || y).collect(),
                    nulls,
                )),
                BinOp::Mul => Ok(Column::bool_with(
                    a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect(),
                    nulls,
                )),
                BinOp::Sub => Err(VolasError::DType(
                    "the `-` operator is not supported for bool columns (use `^`)".into(),
                )),
            };
        }
        let dense = Validity::dense();
        match binary_supertype(self.dtype(), other.dtype()) {
            DType::I64 => {
                let nulls = self.combined_nulls(other);
                let out = binary_kernel(&self.as_i64_vec()?, &other.as_i64_vec()?, op, &nulls)?;
                Ok(Column::i64_with(out, nulls))
            }
            DType::I32 => {
                let nulls = self.combined_nulls(other);
                let out = binary_kernel(&self.as_i32_vec()?, &other.as_i32_vec()?, op, &nulls)?;
                Ok(Column::i32_with(out, nulls))
            }
            DType::F32 => Ok(Column::f32(binary_kernel(
                &self.to_f32_vec(),
                &other.to_f32_vec(),
                op,
                &dense,
            )?)),
            _ => Ok(Column::f64(binary_kernel(
                &self.to_f64_domain("arithmetic")?,
                &other.to_f64_domain("arithmetic")?,
                op,
                &dense,
            )?)),
        }
    }

    /// True division `self / other` (pandas `/`): always float. `bool / bool` is
    /// an error, matching pandas (division is not defined on bool).
    pub fn div(&self, other: &Column) -> Result<Column> {
        self.require_same_len(other)?;
        self.require_numeric()?;
        other.require_numeric()?;
        if matches!((self, other), (Column::Bool(_, _), Column::Bool(_, _))) {
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
        if matches!((self, other), (Column::Bool(_, _), Column::Bool(_, _))) {
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
                Column::i32_with(narrow.collect::<Result<_>>()?, val)
            } else {
                Column::i64_with(out, val)
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
        let n = self.require_same_len(other)?;
        let numeric_like = |c: &Column| c.dtype().is_numeric() || matches!(c, Column::Bool(..));
        let out = match (self, other) {
            (Column::Str(a, av), Column::Str(b, bv)) => cmp_typed(n, op, |i| {
                // SAFETY: `i < n` and `require_same_len` proved both columns have length `n`.
                (av.is_valid(i) && bv.is_valid(i))
                    .then(|| unsafe { a.get_unchecked(i).cmp(b.get_unchecked(i)) })
            }),
            (Column::Datetime(a), Column::Datetime(b)) => cmp_typed(n, op, |i| {
                (a[i] != i64::MIN && b[i] != i64::MIN).then(|| a[i].cmp(&b[i]))
            }),
            (Column::Bool(a, av), Column::Bool(b, bv)) => cmp_typed(n, op, |i| {
                (av.is_valid(i) && bv.is_valid(i)).then(|| a[i].cmp(&b[i]))
            }),
            // numeric (and bool-vs-numeric) compare EXACTLY (contract #26): two
            // integer-like sides as i64, an integer against a float by exact order
            // (no float64 round trip, so 2^53 + 1 > 2^53); only float vs float is
            // IEEE. A missing side follows IEEE (`!=` true, else false).
            (a, b) if numeric_like(a) && numeric_like(b) => match (a.is_float_like(), b.is_float_like()) {
                (true, true) => {
                    let (x, y) = (a.to_f64_vec(), b.to_f64_vec());
                    (0..n).map(|i| op.matches_f64(x[i], y[i])).collect()
                }
                (false, false) => cmp_typed(n, op, |i| {
                    (a.is_valid(i) && b.is_valid(i)).then(|| a.get_i64(i).cmp(&b.get_i64(i)))
                }),
                (false, true) => cmp_typed(n, op, |i| {
                    a.is_valid(i).then(|| cmp_i64_f64(a.get_i64(i), b.get_f64(i))).flatten()
                }),
                (true, false) => cmp_typed(n, op, |i| {
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
        };
        Ok(Column::bool(out))
    }

    /// A column of per-column reduction results (a DataFrame `sum` / `min` / …),
    /// each part `(dtype it contributes, value)`. The column's dtype is the
    /// supertype of the parts (all int64 → int64, int32 → int32, any float →
    /// float), and every value converts to it losslessly (contract #20 / #26): an
    /// integer past the float's exact range raises rather than rounding. A `NaN`
    /// value (an all-missing reduction) becomes the target's missing cell.
    pub fn from_scalars(parts: &[(DType, Scalar)]) -> Result<Column> {
        let target = parts.iter().map(|p| p.0).reduce(binary_supertype).unwrap_or(DType::F64);
        let mut out = Column::na_of(target, 0);
        for &(_, s) in parts {
            let cell = match s {
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
        if self.null_count() > 0 {
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
    /// semantics: a present `false` makes `and` false and a present `true` makes
    /// `or` true even when the other side is missing; otherwise a missing operand
    /// yields NA. A non-bool operand is read as `x != 0` (present). Unequal lengths
    /// are a [`VolasError::Shape`] error.
    pub fn logical(&self, other: &Column, op: BoolOp) -> Result<Column> {
        let n = self.require_same_len(other)?;
        Ok(from_option_bools(
            n,
            (0..n).map(|i| {
                let (a, ap) = self.bool_at(i);
                let (b, bp) = other.bool_at(i);
                match op {
                    BoolOp::And => {
                        if (ap && !a) || (bp && !b) {
                            Some(false)
                        } else if ap && bp {
                            Some(true)
                        } else {
                            None
                        }
                    }
                    BoolOp::Or => {
                        if (ap && a) || (bp && b) {
                            Some(true)
                        } else if ap && bp {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    BoolOp::Xor => (ap && bp).then_some(a ^ b),
                }
            }),
        ))
    }

    /// Logical NOT (pandas `~`), propagating missing (NA in -> NA out). A non-bool
    /// column is read as `x != 0` first.
    pub fn not(&self) -> Column {
        let n = self.len();
        from_option_bools(
            n,
            (0..n).map(|i| {
                let (a, ap) = self.bool_at(i);
                ap.then_some(!a)
            }),
        )
    }

    /// The value (`x != 0` for a non-bool) and presence of element `i`, for the
    /// logical ops. A non-bool is always present (its missing-as-bool question is
    /// the comparison policy: a missing value compares/reads `false`-y).
    fn bool_at(&self, i: usize) -> (bool, bool) {
        match self {
            Column::Bool(v, val) => (v[i], val.is_valid(i)),
            _ => (self.get_f64(i) != 0.0, true),
        }
    }

    /// Sum, dtype-preserving (pandas): a float column -> float, int/bool -> i64
    /// (bool counts trues). Computed natively (i64 in i64, exact past 2^53) and
    /// checked: an int64 overflow is an error (contract #25).
    pub fn sum(&self) -> Result<Scalar> {
        Ok(match self {
            Column::F64(v) => Scalar::F64(stats::sum(v.as_slice())?),
            Column::F32(v) => Scalar::F32(stats::sum(v.as_slice())?),
            Column::I64(v, val) => Scalar::I64(sum_valid(v, val)?),
            // int32 / bool sum promotes to int64 (pandas / numpy accumulator)
            Column::I32(v, val) => Scalar::I64(sum_valid(&widen_i64(v), val)?),
            Column::Bool(v, val) => Scalar::I64(sum_valid(&widen_i64(v), val)?),
            other => Scalar::F64(stats::sum(&other.to_f64_vec())?),
        })
    }

    /// Product, dtype-preserving (float -> float, int/bool -> i64), checked like
    /// [`sum`](Self::sum).
    pub fn prod(&self) -> Result<Scalar> {
        Ok(match self {
            Column::F64(v) => Scalar::F64(stats::prod(v.as_slice())?),
            Column::F32(v) => Scalar::F32(stats::prod(v.as_slice())?),
            Column::I64(v, val) => Scalar::I64(prod_valid(v, val)?),
            Column::I32(v, val) => Scalar::I64(prod_valid(&widen_i64(v), val)?),
            Column::Bool(v, val) => Scalar::I64(prod_valid(&widen_i64(v), val)?),
            other => Scalar::F64(stats::prod(&other.to_f64_vec())?),
        })
    }

    /// Minimum (`want_max = false`) / maximum, dtype-preserving (float -> float,
    /// int -> i64, bool -> bool). Empty / all-missing -> `F64(NaN)`.
    pub fn extreme(&self, want_max: bool) -> Scalar {
        match self {
            Column::I64(v, val) => match extreme_valid(v, val, want_max) {
                Some(x) => Scalar::I64(x),
                None => Scalar::F64(f64::NAN),
            },
            Column::I32(v, val) => match extreme_valid(v, val, want_max) {
                Some(x) => Scalar::I32(x),
                None => Scalar::F64(f64::NAN),
            },
            Column::F32(v) => {
                Scalar::F32(stats::extreme(v.as_slice(), want_max).unwrap_or(f32::NAN))
            }
            Column::Bool(v, val) => {
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
            Column::F64(v) => {
                Scalar::F64(stats::extreme(v.as_slice(), want_max).unwrap_or(f64::NAN))
            }
            other => Scalar::F64(stats::extreme(&other.to_f64_vec(), want_max).unwrap_or(f64::NAN)),
        }
    }
}
