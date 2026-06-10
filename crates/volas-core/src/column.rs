//! Column: a single typed, contiguous buffer.
//!
//! Each variant holds its buffer behind an `Arc`, so a `Column` — and the
//! `DataFrame` / `Series` that contain it — is cheap to clone: cloning shares the
//! buffer (an O(1) refcount bump, not an O(n) copy). Mutation (`append`) is
//! copy-on-write via `Arc::make_mut`: it grows the `Vec` in place when the buffer
//! is uniquely owned, and copies only when a view (another `Series`, a zero-copy
//! export) is still alive. `F64` columns use `NaN` for missing values (matching
//! stock-pandas / pandas semantics).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use crate::datetime;
use crate::dtype::DType;
use crate::error::{Result, VolasError};
use crate::numeric::{binary_supertype, fits, Numeric};
use crate::stats;
use crate::validity::Validity;

/// Run a numeric kernel over a column's element type, monomorphised per dtype
/// (`F64` / `I64`) with no f64 round-trip. `$slice` is bound to the typed slice;
/// `$body` must produce a `Column` (via [`Numeric::into_column`]). `Bool` is
/// handled per-op by the caller (pandas treats it inconsistently — `cumsum -> int`
/// but `abs/cummax -> bool`); a `Bool` / `Str` / `Datetime` column here is an error.
macro_rules! numeric_dispatch {
    ($col:expr, $slice:ident => $body:expr) => {
        match $col {
            Column::F64(buf) => {
                let $slice: &[f64] = buf.as_slice();
                Ok($body)
            }
            Column::F32(buf) => {
                let $slice: &[f32] = buf.as_slice();
                Ok($body)
            }
            Column::I64(buf, _) => {
                let $slice: &[i64] = buf.as_slice();
                Ok($body)
            }
            Column::I32(buf, _) => {
                let $slice: &[i32] = buf.as_slice();
                Ok($body)
            }
            other => Err(VolasError::DType(format!(
                "expected a numeric column, got {}",
                other.dtype()
            ))),
        }
    };
}

/// A typed, contiguous column of values. The buffer is `Arc`-shared (cheap clone)
/// and mutated copy-on-write.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    /// 64-bit floats; `NaN` denotes missing.
    F64(Arc<Vec<f64>>),
    /// 32-bit floats (narrow storage); `NaN` denotes missing.
    F32(Arc<Vec<f32>>),
    /// Booleans (comparison / signal results); `Validity` marks missing cells.
    Bool(Arc<Vec<bool>>, Validity),
    /// 64-bit signed integers; `Validity` marks missing cells.
    I64(Arc<Vec<i64>>, Validity),
    /// 32-bit signed integers (narrow storage); `Validity` marks missing cells.
    I32(Arc<Vec<i32>>, Validity),
    /// UTF-8 strings; `Validity` marks missing cells.
    Str(Arc<Vec<String>>, Validity),
    /// Datetimes as i64 nanoseconds since the Unix epoch (UTC-naive).
    Datetime(Arc<Vec<i64>>),
}

/// The result of a dtype-preserving scalar reduction ([`Column::sum`] etc.),
/// carrying the value in its pandas result dtype so the binding can box it as the
/// matching numpy scalar (`np.int64` / `np.float64` / `np.bool_`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    /// A float64 result (`mean`, an f64 column's `sum`/`min`/`max`, …).
    F64(f64),
    /// A float32 result (an f32 column's `sum`/`min`/`max`).
    F32(f32),
    /// An int64 result (an i64/bool column's `sum`/`prod`, an i64 `min`/`max`).
    I64(i64),
    /// An int32 result (an i32 column's `sum`/`prod`/`min`/`max`).
    I32(i32),
    /// A boolean result (a bool column's `min`/`max`).
    Bool(bool),
}

/// A dtype-preserving binary arithmetic op for [`Column::binary`] (pandas
/// `+ - *`). True division is always float, so it is [`Column::div`], not here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
}

/// A three-valued logical op for [`Column::logical`] (pandas `&` / `|` / `^` on
/// bool columns, Kleene semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    /// Logical AND (a present `false` short-circuits to `false`).
    And,
    /// Logical OR (a present `true` short-circuits to `true`).
    Or,
    /// Logical XOR (missing if either operand is missing).
    Xor,
}

/// An element-wise comparison op for [`Column::compare`] (pandas `== != < <= > >=`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    /// Whether `ordering` (the result of comparing two **present** values)
    /// satisfies this op.
    fn matches(self, o: Ordering) -> bool {
        match self {
            CmpOp::Eq => o == Ordering::Equal,
            CmpOp::Ne => o != Ordering::Equal,
            CmpOp::Lt => o == Ordering::Less,
            CmpOp::Le => o != Ordering::Greater,
            CmpOp::Gt => o == Ordering::Greater,
            CmpOp::Ge => o != Ordering::Less,
        }
    }

    /// The op applied to two `f64`s (the numeric path); `NaN` follows IEEE — `!=`
    /// is `true`, every other op `false`.
    fn matches_f64(self, a: f64, b: f64) -> bool {
        match self {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
        }
    }
}

mod cast;
mod ops;
mod order;
mod transform;

impl Column {
    /// Build an `F64` column.
    pub fn f64(v: Vec<f64>) -> Column {
        Column::F64(Arc::new(v))
    }

    /// Build an `F32` column.
    pub fn f32(v: Vec<f32>) -> Column {
        Column::F32(Arc::new(v))
    }

    /// Build an `I32` column (all values present).
    pub fn i32(v: Vec<i32>) -> Column {
        Column::I32(Arc::new(v), Validity::dense())
    }

    /// Build a `Bool` column (all values present).
    pub fn bool(v: Vec<bool>) -> Column {
        Column::Bool(Arc::new(v), Validity::dense())
    }

    /// Build an `I64` column (all values present).
    pub fn i64(v: Vec<i64>) -> Column {
        Column::I64(Arc::new(v), Validity::dense())
    }

    /// Build an `I64` column with an explicit validity (missing-aware).
    pub fn i64_with(v: Vec<i64>, validity: Validity) -> Column {
        Column::I64(Arc::new(v), validity)
    }

    /// Build an `I32` column with an explicit validity (missing-aware).
    pub fn i32_with(v: Vec<i32>, validity: Validity) -> Column {
        Column::I32(Arc::new(v), validity)
    }

    /// Build a `Bool` column with an explicit validity (missing-aware).
    pub fn bool_with(v: Vec<bool>, validity: Validity) -> Column {
        Column::Bool(Arc::new(v), validity)
    }

    /// Build a `Str` column (all values present).
    pub fn str(v: Vec<String>) -> Column {
        Column::Str(Arc::new(v), Validity::dense())
    }

    /// Build a `Str` column with an explicit validity (missing-aware).
    pub fn str_with(v: Vec<String>, validity: Validity) -> Column {
        Column::Str(Arc::new(v), validity)
    }

    /// Build a `Datetime` column (epoch nanoseconds).
    pub fn datetime(v: Vec<i64>) -> Column {
        Column::Datetime(Arc::new(v))
    }

    /// An all-missing column of `dtype` with `len` rows — the dtype-preserving
    /// default `other` for `where` / `mask` (so a kept value stays in its dtype
    /// and a replaced one becomes NA, never an f64-funnel NaN).
    pub fn na_of(dtype: DType, len: usize) -> Column {
        let all_na = || Validity::from_valid_iter(len, std::iter::repeat(false).take(len));
        match dtype {
            DType::F64 => Column::f64(vec![f64::NAN; len]),
            DType::F32 => Column::f32(vec![f32::NAN; len]),
            DType::I64 => Column::i64_with(vec![0; len], all_na()),
            DType::I32 => Column::i32_with(vec![0; len], all_na()),
            DType::Bool => Column::bool_with(vec![false; len], all_na()),
            DType::Utf8 => Column::str_with(vec![String::new(); len], all_na()),
            DType::Datetime => Column::datetime(vec![i64::MIN; len]),
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Column::F64(v) => v.len(),
            Column::F32(v) => v.len(),
            Column::Bool(v, _) => v.len(),
            Column::I64(v, _) => v.len(),
            Column::I32(v, _) => v.len(),
            Column::Str(v, _) => v.len(),
            Column::Datetime(v) => v.len(),
        }
    }

    /// Whether the column has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The logical dtype of the column.
    pub fn dtype(&self) -> DType {
        match self {
            Column::F64(_) => DType::F64,
            Column::F32(_) => DType::F32,
            Column::Bool(_, _) => DType::Bool,
            Column::I64(_, _) => DType::I64,
            Column::I32(_, _) => DType::I32,
            Column::Str(_, _) => DType::Utf8,
            Column::Datetime(_) => DType::Datetime,
        }
    }

    /// Whether the value at `i` is present (not `volas.NA`). Each dtype reads its
    /// natural missing representation: a float `NaN`, an int/bool validity mask,
    /// or a datetime `NaT` (`i64::MIN`). `i` is assumed in bounds.
    pub fn is_valid(&self, i: usize) -> bool {
        match self {
            Column::F64(v) => !v[i].is_nan(),
            Column::F32(v) => !v[i].is_nan(),
            Column::Bool(_, val)
            | Column::I64(_, val)
            | Column::I32(_, val)
            | Column::Str(_, val) => val.is_valid(i),
            Column::Datetime(v) => v[i] != i64::MIN,
        }
    }

    /// Count of missing (`volas.NA`) values.
    pub fn null_count(&self) -> usize {
        match self {
            Column::F64(v) => v.iter().filter(|x| x.is_nan()).count(),
            Column::F32(v) => v.iter().filter(|x| x.is_nan()).count(),
            Column::Bool(_, val)
            | Column::I64(_, val)
            | Column::I32(_, val)
            | Column::Str(_, val) => val.null_count(),
            Column::Datetime(v) => v.iter().filter(|&&x| x == i64::MIN).count(),
        }
    }

    /// Attach a validity to a (nullable) column produced by a value-only kernel,
    /// so an element-wise transform carries the input's missing positions through.
    /// A float/str/datetime column is returned unchanged (its missing lives in the
    /// values, not a side mask).
    fn with_validity(self, validity: Validity) -> Column {
        match self {
            Column::I64(v, _) => Column::I64(v, validity),
            Column::I32(v, _) => Column::I32(v, validity),
            // bool transforms build their column with the mask directly; float /
            // str / datetime carry missing in the values, so leave them unchanged.
            other => other,
        }
    }

    /// The validity of a nullable (int / bool) column, else `None` (a float column
    /// carries its missing in `NaN`, not a mask).
    fn validity(&self) -> Option<&Validity> {
        match self {
            Column::Bool(_, val)
            | Column::I64(_, val)
            | Column::I32(_, val)
            | Column::Str(_, val) => Some(val),
            _ => None,
        }
    }

    /// Borrow the underlying `f64` slice, if this is an `F64` column.
    pub fn as_f64(&self) -> Option<&[f64]> {
        if let Column::F64(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `bool` slice, if this is a `Bool` column.
    pub fn as_bool(&self) -> Option<&[bool]> {
        if let Column::Bool(v, _) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `i64` slice, if this is an `I64` column.
    pub fn as_i64(&self) -> Option<&[i64]> {
        if let Column::I64(v, _) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `String` slice, if this is a `Str` column.
    pub fn as_str(&self) -> Option<&[String]> {
        if let Column::Str(v, _) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying epoch-ns slice, if this is a `Datetime` column.
    pub fn as_datetime(&self) -> Option<&[i64]> {
        if let Column::Datetime(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Materialize the values as `f64` (`bool` -> 0.0/1.0, `i64` / `datetime` ->
    /// as f64, `str` -> NaN). Used to feed indicator kernels, which operate on
    /// `f64`.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Column::F64(v) => v.to_vec(),
            Column::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Column::Bool(v, val) => mask_f64(v.iter().map(|&b| if b { 1.0 } else { 0.0 }), val),
            Column::I64(v, val) => mask_f64(v.iter().map(|&i| i as f64), val),
            Column::I32(v, val) => mask_f64(v.iter().map(|&i| i as f64), val),
            Column::Str(v, _) => vec![f64::NAN; v.len()],
            Column::Datetime(v) => v
                .iter()
                .map(|&i| if i == i64::MIN { f64::NAN } else { i as f64 })
                .collect(),
        }
    }

    /// Value at position `i` coerced to `f64`, **ignoring validity** (a missing
    /// cell reads its raw placeholder). Used only by the logical-op coercion in
    /// `bool_at`; NumPy export goes through the validity-aware `to_f64_vec`.
    pub fn get_f64(&self, i: usize) -> f64 {
        match self {
            Column::F64(v) => v[i],
            Column::F32(v) => v[i] as f64,
            Column::Bool(v, _) => {
                if v[i] {
                    1.0
                } else {
                    0.0
                }
            }
            Column::I64(v, _) => v[i] as f64,
            Column::I32(v, _) => v[i] as f64,
            Column::Str(_, _) => f64::NAN,
            Column::Datetime(v) => v[i] as f64,
        }
    }

    /// The column as `bool` values (a `Bool` column directly; else an error).
    fn as_bool_vec(&self) -> Result<Vec<bool>> {
        match self {
            Column::Bool(v, _) => Ok(v.to_vec()),
            other => Err(VolasError::DType(format!(
                "expected a bool column, got {}",
                other.dtype()
            ))),
        }
    }

    /// The column as `i64` values: an `I64` column directly (exact), otherwise via
    /// lossless narrowing (errors if any value is non-integral / out of range).
    /// Used by the int paths of `select` / `binary`, where the caller has already
    /// established the values fit.
    fn as_i64_vec(&self) -> Result<Vec<i64>> {
        match self {
            Column::I64(v, _) => Ok(v.to_vec()),
            _ => self
                .to_f64_vec()
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if self.is_valid(i) {
                        i64::try_from_f64(x).ok_or_else(|| {
                            VolasError::DType(format!("value {x} does not fit int64"))
                        })
                    } else {
                        Ok(0) // placeholder for a missing value (masked by the result validity)
                    }
                })
                .collect(),
        }
    }

    /// The column's `String` values (a `Str` column directly); errors otherwise.
    /// Used by `select` so a str `where` / `mask` keeps its values dtype-preserving.
    fn as_str_vec(&self) -> Result<Vec<String>> {
        match self {
            Column::Str(v, _) => Ok((**v).clone()),
            other => Err(VolasError::DType(format!(
                "cannot select a {} column as str",
                other.dtype()
            ))),
        }
    }

    /// The column's epoch-ns values (a `Datetime` column directly); errors otherwise.
    fn as_datetime_vec(&self) -> Result<Vec<i64>> {
        match self {
            Column::Datetime(v) => Ok((**v).clone()),
            other => Err(VolasError::DType(format!(
                "cannot select a {} column as datetime",
                other.dtype()
            ))),
        }
    }

    /// The column as `i32` values: an `I32` column directly, a `Bool` as 0/1,
    /// otherwise via lossless narrowing (errors if out of range / non-integral).
    fn as_i32_vec(&self) -> Result<Vec<i32>> {
        match self {
            Column::I32(v, _) => Ok(v.to_vec()),
            Column::Bool(v, _) => Ok(v.iter().map(|&b| b as i32).collect()),
            _ => self
                .to_f64_vec()
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if self.is_valid(i) {
                        i32::try_from_f64(x).ok_or_else(|| {
                            VolasError::DType(format!("value {x} does not fit int32"))
                        })
                    } else {
                        Ok(0) // placeholder for a missing value (masked by the result validity)
                    }
                })
                .collect(),
        }
    }

    /// The column as `f32` values (an `F32` column directly; else converted).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Column::F32(v) => v.to_vec(),
            _ => self.to_f64_vec().iter().map(|&x| x as f32).collect(),
        }
    }

    /// Render each value as a `String` (for `astype(str)`).
    fn to_string_vec(&self) -> Vec<String> {
        match self {
            Column::Str(v, _) => v.to_vec(),
            Column::F64(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::F32(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::I64(v, _) => v.iter().map(|x| x.to_string()).collect(),
            Column::I32(v, _) => v.iter().map(|x| x.to_string()).collect(),
            Column::Bool(v, _) => v
                .iter()
                .map(|&b| if b { "True" } else { "False" }.to_string())
                .collect(),
            Column::Datetime(v) => v.iter().map(|&ns| datetime::format_ns(ns)).collect(),
        }
    }
}

/// Collect a value iterator to `f64`, mapping missing positions to `NaN` so the
/// f64-funnel reductions (mean / std / median …) skip them. Dense ⇒ a plain
/// collect with no per-element validity check, so the indicator feed path stays
/// unchanged.
fn mask_f64(vals: impl Iterator<Item = f64>, validity: &Validity) -> Vec<f64> {
    if validity.has_nulls() {
        vals.enumerate()
            .map(|(i, x)| if validity.is_valid(i) { x } else { f64::NAN })
            .collect()
    } else {
        vals.collect()
    }
}

/// A bool / i32 column widened to `i64` (for the i64-accumulator reductions).
fn widen_i64<T: Copy + Into<i64>>(v: &[T]) -> Vec<i64> {
    v.iter().map(|&x| x.into()).collect()
}

/// Collect `Option<bool>`s into a `Bool` column: `None` -> `volas.NA`.
fn from_option_bools(n: usize, it: impl Iterator<Item = Option<bool>>) -> Column {
    let (mut values, mut valid) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for o in it {
        values.push(o.unwrap_or(false));
        valid.push(o.is_some());
    }
    Column::bool_with(values, Validity::from_valid_iter(n, valid))
}

/// Shift `v` by `n` (pandas `shift`), filling vacated cells with `fill`; the kept
/// region moves as a single `memcpy`.
fn shift_fill<T: Copy>(v: &[T], n: isize, fill: T) -> Vec<T> {
    let len = v.len();
    let mut out = vec![fill; len];
    if n >= 0 {
        let n = (n as usize).min(len);
        out[n..].copy_from_slice(&v[..len - n]);
    } else {
        let n = ((-n) as usize).min(len);
        out[..len - n].copy_from_slice(&v[n..]);
    }
    out
}

/// `x[i] - x[i-n]` (pandas `diff`) in one pass; the shift-gap cells are `missing`.
fn diff_kernel<T: Copy + std::ops::Sub<Output = T>>(v: &[T], n: isize, missing: T) -> Vec<T> {
    let len = v.len();
    let mut out = vec![missing; len];
    if n >= 0 {
        let k = n as usize;
        for i in k..len {
            out[i] = v[i] - v[i - k];
        }
    } else {
        let k = (-n) as usize;
        for i in 0..len.saturating_sub(k) {
            out[i] = v[i] - v[i + k];
        }
    }
    out
}

/// Extend `av` (validity of an `alen`-long column) with `bv` (a `blen`-long
/// column's validity) so it stays aligned after an `append`. Dense + dense is a
/// no-op (the result is still fully present).
fn append_validity(av: &mut Validity, alen: usize, bv: &Validity, blen: usize) {
    if !av.has_nulls() && !bv.has_nulls() {
        return;
    }
    let flags: Vec<bool> = (0..alen)
        .map(|i| av.is_valid(i))
        .chain((0..blen).map(|i| bv.is_valid(i)))
        .collect();
    *av = Validity::from_valid_iter(alen + blen, flags);
}

/// Sum of the present values in their element type (skip `volas.NA`). Dense ⇒
/// the plain kernel (no per-element validity check).
fn sum_valid<T: Numeric>(v: &[T], val: &Validity) -> T {
    if !val.has_nulls() {
        return stats::sum(v);
    }
    v.iter()
        .enumerate()
        .filter(|(i, _)| val.is_valid(*i))
        .fold(T::ZERO, |a, (_, &x)| a.wrapping_add(x))
}

/// Product of the present values (skip `volas.NA`). Dense ⇒ the plain kernel.
fn prod_valid<T: Numeric>(v: &[T], val: &Validity) -> T {
    if !val.has_nulls() {
        return stats::prod(v);
    }
    v.iter()
        .enumerate()
        .filter(|(i, _)| val.is_valid(*i))
        .fold(T::ONE, |a, (_, &x)| a.wrapping_mul(x))
}

/// Min / max of the present values (skip `volas.NA`); `None` when none present.
fn extreme_valid<T: Numeric>(v: &[T], val: &Validity, want_max: bool) -> Option<T> {
    if !val.has_nulls() {
        return stats::extreme(v, want_max);
    }
    let mut it = v
        .iter()
        .enumerate()
        .filter(|(i, _)| val.is_valid(*i))
        .map(|(_, &x)| x);
    let first = it.next()?;
    Some(if want_max {
        it.fold(first, |a, x| if x > a { x } else { a })
    } else {
        it.fold(first, |a, x| if x < a { x } else { a })
    })
}

/// Running cumulative `op` over present values; a missing position gets
/// `placeholder` (masked by the carried validity, so its value is irrelevant).
fn cum_valid<T: Copy>(v: &[T], val: &Validity, placeholder: T, op: impl Fn(T, T) -> T) -> Vec<T> {
    let mut acc: Option<T> = None;
    v.iter()
        .enumerate()
        .map(|(i, &x)| {
            if val.is_valid(i) {
                let next = acc.map_or(x, |a| op(a, x));
                acc = Some(next);
                next
            } else {
                placeholder
            }
        })
        .collect()
}

/// Dtype-preserving cumulative for a numeric column: the dense kernel when fully
/// present, else a skip-and-propagate fold carrying the input validity through.
fn cum<T: Numeric>(
    v: &[T],
    val: &Validity,
    dense: fn(&[T]) -> Vec<T>,
    op: fn(T, T) -> T,
) -> Column {
    let out = if val.has_nulls() {
        cum_valid(v, val, T::ZERO, op)
    } else {
        dense(v)
    };
    T::into_column(out).with_validity(val.clone())
}

/// Banker's (half-to-even) round of `x` to `decimals` places; `NaN` stays `NaN`.
fn round_f64(x: f64, decimals: i32) -> f64 {
    if x.is_nan() {
        return x;
    }
    let f = 10f64.powi(decimals);
    (x * f).round_ties_even() / f
}

/// Round an int to `decimals` places: identity for `decimals >= 0`; for negative
/// `decimals`, banker's round to the nearest `10^|decimals|` multiple (integer
/// arithmetic, exact for all i64 — no f64 round-trip).
fn round_i64(x: i64, decimals: i32) -> i64 {
    if decimals >= 0 {
        return x;
    }
    let factor = match 10i64.checked_pow(decimals.unsigned_abs()) {
        Some(f) => f,
        None => return 0, // 10^k beyond i64 -> everything rounds to 0
    };
    let q = x.div_euclid(factor);
    let r = x.rem_euclid(factor); // 0..factor
    let half = factor / 2; // factor = 10^k is even, so this is exact
    let up = r > half || (r == half && q.rem_euclid(2) != 0); // tie -> even multiple
    if up { q + 1 } else { q }.wrapping_mul(factor)
}

/// Clamp each element to `[lo, hi]` (either optional); missing passes through.
/// Bounds are narrowed to `T` losslessly (the caller only stays in an int dtype
/// when the bounds fit it).
fn clip_vec<T: Numeric>(v: &[T], lo: Option<f64>, hi: Option<f64>) -> Vec<T> {
    let lo = lo.and_then(T::try_from_f64);
    let hi = hi.and_then(T::try_from_f64);
    v.iter()
        .map(|&x| {
            if x.is_missing() {
                return x;
            }
            let mut y = x;
            if let Some(l) = lo {
                if y < l {
                    y = l;
                }
            }
            if let Some(h) = hi {
                if y > h {
                    y = h;
                }
            }
            y
        })
        .collect()
}

/// Build a non-nullable bool mask from a per-index ordering of present pairs: a
/// present pair (`Some(ord)`) applies `op`; a missing slot (`None`) follows IEEE,
/// so only `!=` is `true`. Backs the typed (`str` / `datetime` / `bool`) arms of
/// [`Column::compare`].
fn cmp_typed(n: usize, op: CmpOp, key: impl Fn(usize) -> Option<Ordering>) -> Vec<bool> {
    (0..n)
        .map(|i| match key(i) {
            Some(ord) => op.matches(ord),
            None => op == CmpOp::Ne,
        })
        .collect()
}

/// Integer floor division (toward negative infinity, unlike Rust's truncating
/// `/`), wrapping on overflow like numpy. The divisor is non-zero (callers route
/// a zero divisor through the float path).
fn ifloordiv(a: i64, b: i64) -> i64 {
    let q = a.wrapping_div(b);
    let r = a.wrapping_rem(b);
    if r != 0 && (r < 0) != (b < 0) {
        q - 1
    } else {
        q
    }
}

/// A hashable key for a float value, or `None` for `NaN` (which groups as NA).
/// `-0.0` is canonicalized to `+0.0` so the two zeros count as one value.
fn float_key(x: f64) -> Option<u64> {
    if x.is_nan() {
        None
    } else if x == 0.0 {
        Some(0)
    } else {
        Some(x.to_bits())
    }
}

/// Group `0..len` by `key(i)` (a `None` key is the single NA group), returning
/// `(first_index, count, is_na)` per distinct value in order of first appearance.
fn group_by<K: Eq + std::hash::Hash>(
    len: usize,
    key: impl Fn(usize) -> Option<K>,
) -> Vec<(usize, usize, bool)> {
    let mut order: Vec<(usize, usize, bool)> = Vec::new();
    let mut seen: HashMap<K, usize> = HashMap::new();
    let mut na: Option<usize> = None;
    for i in 0..len {
        match key(i) {
            None => match na {
                Some(g) => order[g].1 += 1,
                None => {
                    na = Some(order.len());
                    order.push((i, 1, true));
                }
            },
            Some(k) => match seen.get(&k) {
                Some(&g) => order[g].1 += 1,
                None => {
                    seen.insert(k, order.len());
                    order.push((i, 1, false));
                }
            },
        }
    }
    order
}

/// Running OR (`or = true`, backs bool `cummax`) / running AND (`cummin`).
fn bool_running(v: &[bool], or: bool) -> Vec<bool> {
    let mut acc = !or; // OR seeds false; AND seeds true
    v.iter()
        .map(|&b| {
            acc = if or { acc || b } else { acc && b };
            acc
        })
        .collect()
}

/// Clamp a bool column (pandas `clip`): a `True` lower bound forces all true, a
/// `False` upper bound forces all false, otherwise unchanged (bool has only the
/// two values, so the bounds decide the whole column).
fn clip_bool(v: &[bool], lo: Option<f64>, hi: Option<f64>) -> Vec<bool> {
    let force_true = lo.map_or(false, |x| x != 0.0);
    let force_false = hi.map_or(false, |x| x == 0.0);
    v.iter()
        .map(|&b| {
            if force_false {
                false
            } else if force_true {
                true
            } else {
                b
            }
        })
        .collect()
}

/// Element-wise `a ∘ b` for `Add` / `Sub` / `Mul` (wrapping, pandas overflow).
fn binary_kernel<T: Numeric>(a: &[T], b: &[T], op: BinOp) -> Vec<T> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
        })
        .collect()
}

#[cfg(test)]
mod tests;
