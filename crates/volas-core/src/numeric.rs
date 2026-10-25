//! The numeric element types a column can compute over (`f64`, `i64`), behind a
//! single trait so the cumulative / arithmetic / clip / select kernels are
//! written once and monomorphised per dtype (zero-cost, no f64 round-trip).
//!
//! Missing values use the f64 `NaN` sentinel — `i64` has none, so `is_missing`
//! is always false there — matching pandas 3.0's default (non-nullable) dtypes.
//! Integer arithmetic is checked (`checked_*`): an overflowing result is an
//! [`overflow`] error in every build profile (contract #25), never a wrap (pandas /
//! NumPy) and never a debug-only panic.

#[cfg(test)]
use crate::column::ColumnData;
use crate::column::Column;
use crate::dtype::DType;
use crate::error::VolasError;

/// A numeric column element type. Implemented for `f64` and `i64`; adding `i32` /
/// `f32` / `u64` later is just another `impl` plus a dispatch arm.
pub trait Numeric: Copy + PartialOrd {
    /// The column dtype backed by this element type.
    const DTYPE: DType;
    /// Additive identity (cumsum seed).
    const ZERO: Self;
    /// Multiplicative identity (cumprod seed).
    const ONE: Self;

    /// Wrap a computed buffer back into a `Column` of this element's dtype.
    fn into_column(values: Vec<Self>) -> Column;

    /// Whether this element is the missing sentinel (`NaN` for f64; never i64).
    fn is_missing(self) -> bool;
    /// Widen to f64 (lossless for both supported types within range).
    fn to_f64(self) -> f64;
    /// Narrow from f64 losslessly, or `None` when the value would not fit
    /// (non-integral / out of range / `NaN` for i64). Powers the keep-vs-promote
    /// decision in clip / where / mask / assignment.
    fn try_from_f64(x: f64) -> Option<Self>;

    /// Checked add: `None` when an integer result overflows the type (contract #25:
    /// integer overflow raises, identically in debug and release). A float never
    /// overflows here — IEEE gives `±inf` — so the float impls are always `Some`.
    fn checked_add(self, other: Self) -> Option<Self>;
    /// Checked subtract (see [`checked_add`](Self::checked_add)).
    fn checked_sub(self, other: Self) -> Option<Self>;
    /// Checked multiply (see [`checked_add`](Self::checked_add)).
    fn checked_mul(self, other: Self) -> Option<Self>;
    /// Checked absolute value: `abs(i64::MIN)` has no `i64` result, so it is `None`.
    fn checked_abs(self) -> Option<Self>;
}

/// The error every checked integer kernel raises when a result leaves its dtype's
/// range (contract #25) — pandas / NumPy wrap silently instead.
pub fn overflow(dtype: DType, op: &str) -> VolasError {
    VolasError::Overflow(format!(
        "{dtype} overflow in {op}; integer results never wrap — cast to float64 first \
         if an approximate result is acceptable"
    ))
}

impl Numeric for f64 {
    const DTYPE: DType = DType::F64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn into_column(values: Vec<Self>) -> Column {
        Column::f64(values)
    }
    #[inline]
    fn is_missing(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn try_from_f64(x: f64) -> Option<Self> {
        Some(x)
    }
    #[inline]
    fn checked_add(self, other: Self) -> Option<Self> {
        Some(self + other)
    }
    #[inline]
    fn checked_sub(self, other: Self) -> Option<Self> {
        Some(self - other)
    }
    #[inline]
    fn checked_mul(self, other: Self) -> Option<Self> {
        Some(self * other)
    }
    #[inline]
    fn checked_abs(self) -> Option<Self> {
        Some(self.abs())
    }
}

impl Numeric for f32 {
    const DTYPE: DType = DType::F32;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn into_column(values: Vec<Self>) -> Column {
        Column::f32(values)
    }
    #[inline]
    fn is_missing(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn try_from_f64(x: f64) -> Option<Self> {
        Some(x as f32) // f32 absorbs any float (with rounding), like f64
    }
    #[inline]
    fn checked_add(self, other: Self) -> Option<Self> {
        Some(self + other)
    }
    #[inline]
    fn checked_sub(self, other: Self) -> Option<Self> {
        Some(self - other)
    }
    #[inline]
    fn checked_mul(self, other: Self) -> Option<Self> {
        Some(self * other)
    }
    #[inline]
    fn checked_abs(self) -> Option<Self> {
        Some(self.abs())
    }
}

impl Numeric for i32 {
    const DTYPE: DType = DType::I32;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline]
    fn into_column(values: Vec<Self>) -> Column {
        Column::i32(values)
    }
    #[inline]
    fn is_missing(self) -> bool {
        false
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn try_from_f64(x: f64) -> Option<Self> {
        if x.is_finite() && x.fract() == 0.0 && x >= i32::MIN as f64 && x <= i32::MAX as f64 {
            Some(x as i32)
        } else {
            None
        }
    }
    #[inline]
    fn checked_add(self, other: Self) -> Option<Self> {
        i32::checked_add(self, other)
    }
    #[inline]
    fn checked_sub(self, other: Self) -> Option<Self> {
        i32::checked_sub(self, other)
    }
    #[inline]
    fn checked_mul(self, other: Self) -> Option<Self> {
        i32::checked_mul(self, other)
    }
    #[inline]
    fn checked_abs(self) -> Option<Self> {
        i32::checked_abs(self)
    }
}

impl Numeric for i64 {
    const DTYPE: DType = DType::I64;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline]
    fn into_column(values: Vec<Self>) -> Column {
        Column::i64(values)
    }
    #[inline]
    fn is_missing(self) -> bool {
        false
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn try_from_f64(x: f64) -> Option<Self> {
        // Lossless only: finite, integral, and within [-2^63, 2^63). `i64::MAX as
        // f64` rounds up to 2^63, so the upper bound must be exclusive.
        if x.is_finite() && x.fract() == 0.0 && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&x) {
            Some(x as i64)
        } else {
            None
        }
    }
    #[inline]
    fn checked_add(self, other: Self) -> Option<Self> {
        i64::checked_add(self, other)
    }
    #[inline]
    fn checked_sub(self, other: Self) -> Option<Self> {
        i64::checked_sub(self, other)
    }
    #[inline]
    fn checked_mul(self, other: Self) -> Option<Self> {
        i64::checked_mul(self, other)
    }
    #[inline]
    fn checked_abs(self) -> Option<Self> {
        i64::checked_abs(self)
    }
}

// --- dtype promotion (single source of truth for pandas 3.0 result dtypes) ---

/// Result dtype of a binary arithmetic op (`+` `-` `*`) by operand *type*
/// (numpy-style promotion): same numeric dtype stays (`f32+f32→f32`,
/// `i32+i32→i32`); same kind widens (`f32+f64→f64`, `i32+i64→i64`); mixed
/// int+float → `f64` (safe); bool counts as `i32`. Type-based, matching pandas
/// (`int * 2.0 → float64`); the value-based rule is [`fits`] (clip / where /
/// assign). (`/` is always f64.)
pub fn binary_supertype(a: DType, b: DType) -> DType {
    use DType::{Bool, F32, F64, I32, I64};
    if a == b && matches!(a, F64 | F32 | I64 | I32) {
        return a;
    }
    let int_like = |d| matches!(d, I64 | I32 | Bool);
    if a.is_float() || b.is_float() {
        if a.is_float() && b.is_float() {
            // both float -> the wider one
            if a == F64 || b == F64 { F64 } else { F32 }
        } else {
            F64 // int + float -> f64
        }
    } else if int_like(a) && int_like(b) {
        // both int (bool counts) -> the wider one (bool -> i32)
        if a == I64 || b == I64 { I64 } else { I32 }
    } else {
        F64
    }
}

// --- int64 <-> float exactness (the one owner of the range / exactness rule) ---
//
// Two distinct questions, both answered only here:
// - STORAGE (contract #20): may this integer be stored as a float without changing
//   its value? Yes exactly when it round-trips (`i64_to_f64_exact`), e.g. 2^60 does,
//   2^53 + 1 does not.
// - MATH (contract #26): may this integer enter float64 arithmetic? Only inside the
//   contiguous exact range |x| <= 2^53 (`in_f64_domain`), where every neighbour is
//   representable too; past it a float64 result is not a function of the exact
//   input, so the operation raises.

/// The float64 integer range: every integer with `|x| <= 2^53` is exactly
/// representable, and so are all its neighbours (contract #26).
pub const F64_EXACT_INT: u64 = 1 << 53;

/// Whether integer `x` may enter float64 math (`|x| <= 2^53`, contract #26).
#[inline]
pub fn in_f64_domain(x: i64) -> bool {
    x.unsigned_abs() <= F64_EXACT_INT
}

/// The error a float64-domain operation raises for an integer past 2^53.
pub fn outside_f64_domain(x: i64, op: &str) -> VolasError {
    VolasError::Overflow(format!(
        "{op}: the int64 value {x} is outside the float64-exact range |x| <= 2^53, so \
         a float64 result would not be exact"
    ))
}

/// `x` as an `f64` when the conversion is exact (it round-trips), else `None`
/// (contract #20). `2^63` is the one float whose saturating cast back reads as
/// `i64::MAX`, so it is excluded explicitly.
#[inline]
pub fn i64_to_f64_exact(x: i64) -> Option<f64> {
    let f = x as f64;
    (f != 9_223_372_036_854_775_808.0 && f as i64 == x).then_some(f)
}

/// `x` as an `f32` when the conversion is exact, else `None` (contract #20).
#[inline]
pub fn i64_to_f32_exact(x: i64) -> Option<f32> {
    let f = x as f32;
    (f != 9_223_372_036_854_775_808.0 && f as i64 == x).then_some(f)
}

/// A float64 narrowed to float32: precision rounds to the nearest float32 (the
/// float dtype's own semantics), but a finite value past the float32 range is
/// `None` — it would become `±inf` (contract #20). `NaN` / `±inf` carry over.
#[inline]
pub fn f64_to_f32_in_range(x: f64) -> Option<f32> {
    let f = x as f32;
    (f.is_finite() || !x.is_finite()).then_some(f)
}

/// Exact order of an integer against a float, `None` when the float is `NaN`:
/// no float64 round trip, so `2^53 + 1 > 2^53` (contract #26; NumPy compares in
/// float64 and calls them equal).
pub fn cmp_i64_f64(a: i64, b: f64) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering::{Greater, Less};
    const TWO_63: f64 = 9_223_372_036_854_775_808.0;
    if b.is_nan() {
        None
    } else if b >= TWO_63 {
        Some(Less)
    } else if b < -TWO_63 {
        Some(Greater)
    } else {
        // b in [-2^63, 2^63): its integral part is an exact i64, its fraction exact
        let t = b.trunc();
        Some(a.cmp(&(t as i64)).then_with(|| 0.0.partial_cmp(&(b - t)).expect("finite")))
    }
}

/// Whether the scalar `x` fits `target` losslessly — the keep-vs-promote test for
/// clip / where / mask / assignment (e.g. `0.0` fits `int64`, `2.5` does not).
pub fn fits(target: DType, x: f64) -> bool {
    match target {
        DType::F64 | DType::F32 => true, // any float fits a float dtype (rounding)
        DType::I64 => i64::try_from_f64(x).is_some(),
        DType::I32 => i32::try_from_f64(x).is_some(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_element_semantics() {
        assert_eq!(f64::DTYPE, DType::F64);
        assert!(f64::NAN.is_missing() && !1.0_f64.is_missing());
        assert_eq!(2.5_f64.to_f64(), 2.5);
        assert_eq!(f64::try_from_f64(2.5), Some(2.5)); // f64 always fits
        assert_eq!(Numeric::checked_add(3.0_f64, 4.0), Some(7.0));
        assert_eq!(Numeric::checked_abs(-2.0_f64), Some(2.0));
    }

    #[test]
    fn i64_element_semantics() {
        assert_eq!(i64::DTYPE, DType::I64);
        assert!(!5_i64.is_missing()); // i64 has no missing
        assert_eq!(5_i64.to_f64(), 5.0);
        assert_eq!(3_i32.to_f64(), 3.0);
        // lossless narrowing
        assert_eq!(i64::try_from_f64(5.0), Some(5));
        assert_eq!(i64::try_from_f64(2.5), None); // non-integral
        assert_eq!(i64::try_from_f64(f64::NAN), None); // NaN
        assert_eq!(i64::try_from_f64(1e30), None); // out of range
        assert_eq!(i64::try_from_f64(9.223372036854776e18), None); // 2^63, exclusive
        // checked: an overflowing result is None, never a wrap
        assert_eq!(Numeric::checked_add(i64::MAX, 1), None);
        assert_eq!(Numeric::checked_abs(i64::MIN), None);
        assert_eq!(Numeric::checked_mul(3_i64, 4), Some(12));
        assert_eq!(Numeric::checked_sub(7_i64, 10), Some(-3));
        assert!(matches!(overflow(DType::I64, "sum"), VolasError::Overflow(m) if m.starts_with("int64 overflow in sum")));
    }

    #[test]
    fn f32_i32_element_semantics() {
        // f32: NaN sentinel, lossless to_f64, always-fits try_from (rounding)
        assert_eq!(f32::DTYPE, DType::F32);
        assert!(f32::NAN.is_missing() && !1.5_f32.is_missing());
        assert_eq!(2.5_f32.to_f64(), 2.5);
        assert_eq!(f32::try_from_f64(2.5), Some(2.5_f32));
        assert_eq!(f32::try_from_f64(0.1).map(|x| x.is_finite()), Some(true)); // rounds, still fits
        assert_eq!(Numeric::checked_add(3.0_f32, 4.0), Some(7.0));
        assert_eq!(Numeric::checked_abs(-2.0_f32), Some(2.0));
        assert!(matches!(f32::into_column(vec![1.0_f32]), Column { data: ColumnData::F32(_) }));
        // i32: no missing, checked, range-checked try_from
        assert_eq!(i32::DTYPE, DType::I32);
        assert!(!5_i32.is_missing());
        assert_eq!(i32::try_from_f64(5.0), Some(5));
        assert_eq!(i32::try_from_f64(2.5), None); // non-integral
        assert_eq!(i32::try_from_f64(3e9), None); // out of i32 range
        assert_eq!(Numeric::checked_add(i32::MAX, 1), None); // never wraps
        assert_eq!(Numeric::checked_abs(i32::MIN), None);
        assert_eq!(Numeric::checked_sub(7_i32, 10), Some(-3));
        assert_eq!(Numeric::checked_mul(3_i32, 4), Some(12));
        assert!(matches!(i32::into_column(vec![1_i32]), Column { data: ColumnData::I32(_, _) }));
    }

    #[test]
    fn promotion_rules() {
        use DType::{Bool, F32, F64, I32, I64, Utf8};
        // same numeric dtype stays
        assert_eq!(binary_supertype(I64, I64), I64);
        assert_eq!(binary_supertype(F32, F32), F32);
        assert_eq!(binary_supertype(I32, I32), I32);
        // same kind widens
        assert_eq!(binary_supertype(F32, F64), F64);
        assert_eq!(binary_supertype(I32, I64), I64);
        // mixed int + float -> f64
        assert_eq!(binary_supertype(I64, F64), F64);
        assert_eq!(binary_supertype(I32, F32), F64);
        // bool counts as i32 (its arithmetic is special-cased elsewhere)
        assert_eq!(binary_supertype(Bool, I32), I32);
        assert_eq!(binary_supertype(Bool, I64), I64);
        assert_eq!(binary_supertype(I64, Utf8), F64); // non-numeric operand -> f64
        // value-based fits (clip / where / assign)
        assert!(fits(I64, 3.0) && !fits(I64, 2.5));
        assert!(fits(I32, 3.0) && !fits(I32, 1e20)); // out of i32 range
        assert!(fits(F64, 2.5) && fits(F32, 2.5)); // any float fits a float dtype
        assert!(!fits(Bool, 1.0));
    }
}
