//! `Column` arithmetic, comparison, logical, and reduction operations.

use super::*;

impl Column {
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
    /// `bool ∘ number` promotes (bool acts as 0/1). Wrapping int ops match pandas
    /// overflow. Equal lengths assumed.
    pub fn binary(&self, other: &Column, op: BinOp) -> Result<Column> {
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
        match binary_supertype(self.dtype(), other.dtype()) {
            DType::I64 => Ok(Column::i64_with(
                binary_kernel(&self.as_i64_vec()?, &other.as_i64_vec()?, op),
                self.combined_nulls(other),
            )),
            DType::I32 => Ok(Column::i32_with(
                binary_kernel(&self.as_i32_vec()?, &other.as_i32_vec()?, op),
                self.combined_nulls(other),
            )),
            DType::F32 => Ok(Column::f32(binary_kernel(
                &self.to_f32_vec(),
                &other.to_f32_vec(),
                op,
            ))),
            _ => Ok(Column::f64(binary_kernel(
                &self.to_f64_vec(),
                &other.to_f64_vec(),
                op,
            ))),
        }
    }

    /// True division `self / other` (pandas `/`): always float. `bool / bool` is
    /// an error, matching pandas (division is not defined on bool).
    pub fn div(&self, other: &Column) -> Result<Column> {
        self.require_numeric()?;
        other.require_numeric()?;
        if matches!((self, other), (Column::Bool(_, _), Column::Bool(_, _))) {
            return Err(VolasError::DType(
                "division is not supported between two bool columns".into(),
            ));
        }
        let (a, b) = (self.to_f64_vec(), other.to_f64_vec());
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
        self.require_numeric()?;
        other.require_numeric()?;
        if matches!((self, other), (Column::Bool(_, _), Column::Bool(_, _))) {
            return Err(VolasError::DType(
                "floor division is not supported between two bool columns".into(),
            ));
        }
        let n = self.len();
        let st = binary_supertype(self.dtype(), other.dtype());
        // A present divisor of 0 forces pandas' float path (`inf` / `-inf` / `nan`).
        let int_path = matches!(st, DType::I64 | DType::I32)
            && !(0..n).any(|i| other.is_valid(i) && other.get_f64(i) == 0.0);
        if int_path {
            let (a, b) = (self.as_i64_vec()?, other.as_i64_vec()?);
            let present = |i: usize| self.is_valid(i) && other.is_valid(i);
            // exact i64 floor division (the divisor is non-zero where present);
            // NA positions hold a 0 placeholder, masked by the result validity.
            let out: Vec<i64> = (0..n)
                .map(|i| if present(i) { ifloordiv(a[i], b[i]) } else { 0 })
                .collect();
            let val = Validity::from_valid_iter(n, (0..n).map(present));
            // i32 // i32 stays i32 (the result fits); any i64 operand widens to i64.
            Ok(if st == DType::I32 {
                Column::i32_with(out.iter().map(|&x| x as i32).collect(), val)
            } else {
                Column::i64_with(out, val)
            })
        } else {
            let (a, b) = (self.to_f64_vec(), other.to_f64_vec());
            Ok(Column::f64(
                a.iter().zip(&b).map(|(&x, &y)| (x / y).floor()).collect(),
            ))
        }
    }

    /// Element-wise comparison (pandas `== != < <= > >=`) producing a **non-nullable
    /// bool** mask (decision ②): `str` / `datetime` / `bool` compare by their native
    /// value (NOT through the `to_f64_vec` funnel, which maps `str` to `NaN` and
    /// loses `datetime` precision), numeric kinds compare as `f64`. A missing slot
    /// (either side NA) follows IEEE — `!=` is `true`, every other op `false`,
    /// matching the existing numeric comparison. Comparing two incompatible kinds
    /// (e.g. `str` vs a number) is a [`VolasError::DType`]. Equal lengths assumed.
    pub fn compare(&self, other: &Column, op: CmpOp) -> Result<Column> {
        let n = self.len();
        let numeric_like = |c: &Column| c.dtype().is_numeric() || matches!(c, Column::Bool(..));
        let out = match (self, other) {
            (Column::Str(a, av), Column::Str(b, bv)) => cmp_typed(n, op, |i| {
                // SAFETY: `i < n` and both columns have length `n` (equal lengths assumed).
                (av.is_valid(i) && bv.is_valid(i))
                    .then(|| unsafe { a.get_unchecked(i).cmp(b.get_unchecked(i)) })
            }),
            (Column::Datetime(a), Column::Datetime(b)) => cmp_typed(n, op, |i| {
                (a[i] != i64::MIN && b[i] != i64::MIN).then(|| a[i].cmp(&b[i]))
            }),
            (Column::Bool(a, av), Column::Bool(b, bv)) => cmp_typed(n, op, |i| {
                (av.is_valid(i) && bv.is_valid(i)).then(|| a[i].cmp(&b[i]))
            }),
            // numeric (and bool-vs-numeric) compare as f64 — NaN follows IEEE here
            // (`==` false, `!=` true), the established mask policy.
            (a, b) if numeric_like(a) && numeric_like(b) => {
                let (x, y) = (a.to_f64_vec(), b.to_f64_vec());
                (0..n).map(|i| op.matches_f64(x[i], y[i])).collect()
            }
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

    /// Three-valued logical `and` / `or` / `xor` (pandas `&` / `|` / `^`), Kleene
    /// semantics: a present `false` makes `and` false and a present `true` makes
    /// `or` true even when the other side is missing; otherwise a missing operand
    /// yields NA. A non-bool operand is read as `x != 0` (present). Equal lengths.
    pub fn logical(&self, other: &Column, op: BoolOp) -> Column {
        let n = self.len();
        from_option_bools(
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
                    BoolOp::Xor => (ap && bp).then(|| a ^ b),
                }
            }),
        )
    }

    /// Logical NOT (pandas `~`), propagating missing (NA in -> NA out). A non-bool
    /// column is read as `x != 0` first.
    pub fn not(&self) -> Column {
        let n = self.len();
        from_option_bools(
            n,
            (0..n).map(|i| {
                let (a, ap) = self.bool_at(i);
                ap.then(|| !a)
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
    /// (bool counts trues). Computed natively (i64 in i64, exact past 2^53).
    pub fn sum(&self) -> Scalar {
        match self {
            Column::F64(v) => Scalar::F64(stats::sum(v.as_slice())),
            Column::F32(v) => Scalar::F32(stats::sum(v.as_slice())),
            Column::I64(v, val) => Scalar::I64(sum_valid(v, val)),
            // int32 / bool sum promotes to int64 (pandas / numpy accumulator)
            Column::I32(v, val) => Scalar::I64(sum_valid(&widen_i64(v), val)),
            Column::Bool(v, val) => Scalar::I64(sum_valid(&widen_i64(v), val)),
            other => Scalar::F64(stats::sum(&other.to_f64_vec())),
        }
    }

    /// Product, dtype-preserving (float -> float, int/bool -> i64).
    pub fn prod(&self) -> Scalar {
        match self {
            Column::F64(v) => Scalar::F64(stats::prod(v.as_slice())),
            Column::F32(v) => Scalar::F32(stats::prod(v.as_slice())),
            Column::I64(v, val) => Scalar::I64(prod_valid(v, val)),
            Column::I32(v, val) => Scalar::I64(prod_valid(&widen_i64(v), val)),
            Column::Bool(v, val) => Scalar::I64(prod_valid(&widen_i64(v), val)),
            other => Scalar::F64(stats::prod(&other.to_f64_vec())),
        }
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
