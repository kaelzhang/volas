//! Column: a single typed, contiguous buffer.
//!
//! Each variant holds its buffer behind an `Arc`, so a `Column` — and the
//! `DataFrame` / `Series` that contain it — is cheap to clone: cloning shares the
//! buffer (an O(1) refcount bump, not an O(n) copy). Mutation (`append`) is
//! copy-on-write via `Arc::make_mut`: it grows the `Vec` in place when the buffer
//! is uniquely owned, and copies only when a view (another `Series`, a zero-copy
//! export) is still alive. `F64` columns use `NaN` for missing values (matching
//! stock-pandas / pandas semantics).

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

/// A scalar to assign into a column (boolean-mask / positional assignment). Kept
/// distinct from a plain `f64` so a real `bool` can be told from a number — they
/// fit different dtypes (pandas rejects a number into a bool column, and an
/// integral number stays in an int column while a real bool does not change it).
#[derive(Clone, Copy, Debug)]
pub enum SetVal {
    /// A boolean scalar.
    Bool(bool),
    /// A numeric scalar (an integral, finite value can stay in an int column).
    Num(f64),
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
            Column::Bool(_, val) | Column::I64(_, val) | Column::I32(_, val) | Column::Str(_, val) => {
                val.is_valid(i)
            }
            Column::Datetime(v) => v[i] != i64::MIN,
        }
    }

    /// Count of missing (`volas.NA`) values.
    pub fn null_count(&self) -> usize {
        match self {
            Column::F64(v) => v.iter().filter(|x| x.is_nan()).count(),
            Column::F32(v) => v.iter().filter(|x| x.is_nan()).count(),
            Column::Bool(_, val) | Column::I64(_, val) | Column::I32(_, val) | Column::Str(_, val) => {
                val.null_count()
            }
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
            Column::Bool(_, val) | Column::I64(_, val) | Column::I32(_, val) | Column::Str(_, val) => {
                Some(val)
            }
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
            Column::Datetime(v) => {
                v.iter().map(|&i| if i == i64::MIN { f64::NAN } else { i as f64 }).collect()
            }
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

    /// A contiguous `[start, end)` slice (a fresh buffer).
    pub fn slice(&self, start: usize, end: usize) -> Column {
        match self {
            Column::F64(v) => Column::f64(v[start..end].to_vec()),
            Column::F32(v) => Column::f32(v[start..end].to_vec()),
            Column::Bool(v, val) => Column::bool_with(v[start..end].to_vec(), val.slice(start, end)),
            Column::I64(v, val) => Column::i64_with(v[start..end].to_vec(), val.slice(start, end)),
            Column::I32(v, val) => Column::i32_with(v[start..end].to_vec(), val.slice(start, end)),
            Column::Str(v, val) => Column::str_with(v[start..end].to_vec(), val.slice(start, end)),
            Column::Datetime(v) => Column::datetime(v[start..end].to_vec()),
        }
    }

    /// Gather the given positions into a new column (fancy indexing).
    pub fn take(&self, idx: &[usize]) -> Column {
        match self {
            Column::F64(v) => Column::f64(idx.iter().map(|&i| v[i]).collect()),
            Column::F32(v) => Column::f32(idx.iter().map(|&i| v[i]).collect()),
            Column::Bool(v, val) => Column::bool_with(idx.iter().map(|&i| v[i]).collect(), val.take(idx)),
            Column::I64(v, val) => Column::i64_with(idx.iter().map(|&i| v[i]).collect(), val.take(idx)),
            Column::I32(v, val) => Column::i32_with(idx.iter().map(|&i| v[i]).collect(), val.take(idx)),
            Column::Str(v, val) => Column::str_with(idx.iter().map(|&i| v[i].clone()).collect(), val.take(idx)),
            Column::Datetime(v) => Column::datetime(idx.iter().map(|&i| v[i]).collect()),
        }
    }

    /// Append another column of the same dtype, copy-on-write (grows in place
    /// when the buffer is uniquely owned).
    pub fn append(&mut self, other: &Column) -> Result<()> {
        match (self, other) {
            (Column::F64(a), Column::F64(b)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::F32(a), Column::F32(b)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::Bool(a, av), Column::Bool(b, bv)) => {
                append_validity(av, a.len(), bv, b.len());
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::I64(a, av), Column::I64(b, bv)) => {
                append_validity(av, a.len(), bv, b.len());
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::I32(a, av), Column::I32(b, bv)) => {
                append_validity(av, a.len(), bv, b.len());
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::Str(a, av), Column::Str(b, bv)) => {
                append_validity(av, a.len(), bv, b.len());
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::Datetime(a), Column::Datetime(b)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (s, o) => Err(VolasError::DType(format!(
                "cannot append a {} column onto a {} column",
                o.dtype(),
                s.dtype()
            ))),
        }
    }

    /// Extend a stale computed column with placeholder missing values, avoiding a
    /// temporary one-row [`Column`] allocation on the live append path.
    pub fn append_missing(&mut self, len: usize) -> Result<()> {
        match self {
            Column::F64(v) => {
                Arc::make_mut(v).extend(std::iter::repeat(f64::NAN).take(len));
                Ok(())
            }
            Column::F32(v) => {
                Arc::make_mut(v).extend(std::iter::repeat(f32::NAN).take(len));
                Ok(())
            }
            // The refresh path overwrites these placeholder rows on recompute, so a
            // dense `false` keeps the validity simple (no lingering NA to clear).
            Column::Bool(v, _) => {
                Arc::make_mut(v).extend(std::iter::repeat(false).take(len));
                Ok(())
            }
            other => Err(VolasError::DType(format!(
                "column type {} has no missing-value placeholder",
                other.dtype()
            ))),
        }
    }

    /// Extend a **plain** (non-computed) column with `len` genuine missing values,
    /// dtype-preserving: float / datetime use their in-band sentinel (`NaN` /
    /// `NaT`), while int / bool / str grow the validity bitmap with `len` invalid
    /// bits. Used when a column is absent from an appended frame; a cached
    /// directive instead uses the cheaper [`append_missing`] placeholder, which
    /// `fulfill` overwrites.
    pub fn append_na(&mut self, len: usize) {
        let old = self.len();
        let na_validity =
            |val: &Validity| Validity::from_valid_iter(old + len, (0..old + len).map(|i| i < old && val.is_valid(i)));
        match self {
            Column::F64(v) => Arc::make_mut(v).extend(std::iter::repeat(f64::NAN).take(len)),
            Column::F32(v) => Arc::make_mut(v).extend(std::iter::repeat(f32::NAN).take(len)),
            Column::Datetime(v) => Arc::make_mut(v).extend(std::iter::repeat(i64::MIN).take(len)),
            Column::I64(v, val) => {
                *val = na_validity(val);
                Arc::make_mut(v).extend(std::iter::repeat(0).take(len));
            }
            Column::I32(v, val) => {
                *val = na_validity(val);
                Arc::make_mut(v).extend(std::iter::repeat(0).take(len));
            }
            Column::Bool(v, val) => {
                *val = na_validity(val);
                Arc::make_mut(v).extend(std::iter::repeat(false).take(len));
            }
            Column::Str(v, val) => {
                *val = na_validity(val);
                Arc::make_mut(v).extend(std::iter::repeat(String::new()).take(len));
            }
        }
    }

    /// Parse this column into a [`Column::Datetime`] (epoch ns). `Str` cells are
    /// parsed via [`datetime::parse_ns`]; an already-`Datetime` column is shared
    /// back (cheap). Errors on an unparseable cell or an unsupported dtype.
    pub fn to_datetime(&self) -> Result<Column> {
        self.to_datetime_tz(crate::tz::Tz::Utc)
    }

    /// Parse a string column to a UTC `Datetime` column, interpreting **naive**
    /// strings in `tz` (offset-aware strings are absolute; an existing `Datetime`
    /// column is already UTC and returned as-is). The tz is then attached to the
    /// *index* (storage stays UTC).
    pub fn to_datetime_tz(&self, tz: crate::tz::Tz) -> Result<Column> {
        match self {
            Column::Datetime(_) => Ok(self.clone()),
            Column::Str(v, val) => {
                let mut out = Vec::with_capacity(v.len());
                for (i, s) in v.iter().enumerate() {
                    // A missing (NA) string cell parses to NaT, not the "" placeholder
                    // (which would fail) — matching the float/int epoch NA paths.
                    if !val.is_valid(i) {
                        out.push(i64::MIN);
                        continue;
                    }
                    let ns = datetime::parse_ns_in_tz(s, tz).ok_or_else(|| {
                        VolasError::Value(format!("could not parse datetime {s:?}"))
                    })?;
                    out.push(ns);
                }
                Ok(Column::datetime(out))
            }
            other => Err(VolasError::DType(format!(
                "cannot parse a {} column as datetime",
                other.dtype()
            ))),
        }
    }

    /// Interpret a numeric (epoch) column as a UTC `Datetime` column, scaling by
    /// `unit` (`"s"` / `"ms"` / `"us"` / `"ns"`). Float epochs are **truncated** to
    /// the whole `unit` (matching a NumPy / pandas `astype('datetime64[unit]')`
    /// cast). The robust ingestion path for exchange APIs that return numeric
    /// timestamps.
    pub fn epoch_to_datetime(&self, unit: &str) -> Result<Column> {
        self.epoch_to_datetime_with(unit, |x| datetime::epoch_to_ns(x as i64, unit))
    }

    /// Like [`epoch_to_datetime`](Self::epoch_to_datetime) but **rounds** float
    /// epochs to the nearest nanosecond, preserving sub-`unit` fractions (matching
    /// `pandas.to_datetime(..., unit=...)`). Identical for integer columns.
    pub fn epoch_to_datetime_rounded(&self, unit: &str) -> Result<Column> {
        self.epoch_to_datetime_with(unit, |x| datetime::epoch_to_ns_f64(x, unit))
    }

    /// Shared epoch → `Datetime` conversion; `f64_to_ns` chooses how float epochs
    /// map to nanoseconds (truncate vs round). Integers always scale exactly.
    fn epoch_to_datetime_with(
        &self,
        unit: &str,
        f64_to_ns: impl Fn(f64) -> Option<i64>,
    ) -> Result<Column> {
        match self {
            // A missing input maps to `NaT` (i64::MIN), not 1970 / an error: read
            // the i64 validity bit, and treat a float `NaN` as missing.
            Column::I64(v, val) => v
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if !val.is_valid(i) {
                        Ok(i64::MIN)
                    } else {
                        datetime::epoch_to_ns(x, unit).ok_or_else(|| {
                            VolasError::Value(format!(
                            "could not convert epoch with unit {unit:?}: unknown unit or value out of nanosecond range"
                        ))
                        })
                    }
                })
                .collect::<Result<Vec<_>>>()
                .map(Column::datetime),
            Column::F64(v) => v
                .iter()
                .map(|&x| {
                    if x.is_nan() {
                        Ok(i64::MIN)
                    } else {
                        f64_to_ns(x).ok_or_else(|| {
                            VolasError::Value(format!(
                            "could not convert epoch with unit {unit:?}: unknown unit or value out of nanosecond range"
                        ))
                        })
                    }
                })
                .collect::<Result<Vec<_>>>()
                .map(Column::datetime),
            other => Err(VolasError::DType(format!(
                "cannot read a {} column as an epoch timestamp (need int64 / float64)",
                other.dtype()
            ))),
        }
    }

    /// Cast to another dtype (best-effort, pandas `astype`-like). A no-op when
    /// already the target dtype.
    pub fn cast(&self, to: DType) -> Result<Column> {
        if self.dtype() == to {
            return Ok(self.clone());
        }
        // An int/bool source to an int/bool target keeps its missing cells and
        // converts present values exactly (no f64 round-trip; range-checked).
        if matches!(self, Column::I64(..) | Column::I32(..) | Column::Bool(..))
            && matches!(to, DType::I64 | DType::I32 | DType::Bool)
        {
            return self.cast_int_bool(to);
        }
        match to {
            DType::F64 => Ok(Column::f64(self.to_f64_vec())),
            DType::I64 => match self {
                Column::F64(v) => {
                    // pandas raises (IntCastingNaNError) rather than silently
                    // turning NaN -> 0 / inf -> i64::MAX, which corrupts data.
                    if let Some(x) = v.iter().copied().find(|x| !x.is_finite()) {
                        return Err(VolasError::Value(format!(
                            "cannot convert non-finite value ({x}) to int64 (NaN / inf); \
                             fill or drop it first"
                        )));
                    }
                    Ok(Column::i64(v.iter().map(|&x| x as i64).collect()))
                }
                // int/bool sources are handled by `cast_int_bool` above.
                Column::Datetime(v) => Ok(Column::i64(v.to_vec())),
                other => Err(VolasError::DType(format!(
                    "cannot cast a {} column to int64",
                    other.dtype()
                ))),
            },
            DType::Bool => match self {
                Column::F64(v) => Ok(Column::bool(v.iter().map(|&x| x != 0.0).collect())),
                // int sources are handled by `cast_int_bool` above.
                other => Err(VolasError::DType(format!(
                    "cannot cast a {} column to bool",
                    other.dtype()
                ))),
            },
            DType::F32 => Ok(Column::f32(self.to_f64_vec().iter().map(|&x| x as f32).collect())),
            DType::I32 => self
                .to_f64_vec()
                .iter()
                .map(|&x| {
                    i32::try_from_f64(x).ok_or_else(|| {
                        VolasError::Value(format!(
                            "cannot convert {x} to int32 (non-finite / non-integral / out of range)"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(Column::i32),
            // int/bool sources carry their validity into the rendered strings.
            DType::Utf8 => {
                Ok(Column::str_with(self.to_string_vec(), self.validity().cloned().unwrap_or_default()))
            }
            DType::Datetime => self.to_datetime(),
        }
    }

    /// Cast an int/bool column to another int/bool dtype, carrying its validity (a
    /// missing cell stays missing) and converting present values exactly (no f64
    /// round-trip). Only a present, out-of-range value (a large `i64` into `i32`)
    /// errors; a missing cell never does.
    fn cast_int_bool(&self, to: DType) -> Result<Column> {
        let validity = self.validity().cloned().unwrap_or_default();
        let narrow_i32 = |v: &[i64]| -> Result<Vec<i32>> {
            v.iter()
                .enumerate()
                .map(|(i, &x)| {
                    if validity.is_valid(i) {
                        i32::try_from(x).map_err(|_| {
                            VolasError::Value(format!("cannot convert {x} to int32 (out of range)"))
                        })
                    } else {
                        Ok(0)
                    }
                })
                .collect()
        };
        match (self, to) {
            (Column::I64(v, _), DType::I32) => Ok(Column::i32_with(narrow_i32(v)?, validity)),
            (Column::I64(v, _), DType::Bool) => Ok(Column::bool_with(v.iter().map(|&x| x != 0).collect(), validity)),
            (Column::I32(v, _), DType::I64) => Ok(Column::i64_with(v.iter().map(|&x| x as i64).collect(), validity)),
            (Column::I32(v, _), DType::Bool) => Ok(Column::bool_with(v.iter().map(|&x| x != 0).collect(), validity)),
            (Column::Bool(v, _), DType::I64) => Ok(Column::i64_with(v.iter().map(|&b| b as i64).collect(), validity)),
            (Column::Bool(v, _), DType::I32) => Ok(Column::i32_with(v.iter().map(|&b| b as i32).collect(), validity)),
            _ => unreachable!("same-dtype is handled in cast"), // LCOV_EXCL_LINE
        }
    }

    /// Assign a scalar at the given positions, following pandas 3.0's in-place
    /// dtype rules: keep the column dtype when the value fits losslessly; upcast
    /// an int column to float for `NaN`; reject a lossy write (a non-integral
    /// number into an int column, or a number into a bool column) with a `DType`
    /// error — surfaces as `TypeError`, like pandas' `LossySetitemError`.
    /// `positions` are assumed in bounds (callers validate the mask / index).
    pub fn set_scalar_at(&self, positions: &[usize], value: SetVal) -> Result<Column> {
        let len = self.len();
        match self {
            // A float column absorbs any value (with rounding for f32); NaN is its
            // in-band missing, so no separate validity is needed.
            Column::F64(v) => Ok(set_float_at(v, positions, value)),
            Column::F32(v) => Ok(set_float_at(v, positions, value)),
            // An int column keeps its dtype AND its existing validity bitmap: a real
            // (integral) value marks those positions present, a NaN marks them
            // missing (NA, *not* an f64 upcast), a lossy value errors. Other rows'
            // NA is preserved (the bug fix: a scalar write used to drop the mask).
            Column::I64(v, val) => match value {
                SetVal::Num(x) if x.is_nan() => {
                    Ok(Column::i64_with(v.to_vec(), validity_set(val, positions, false, len)))
                }
                SetVal::Num(x) => match i64::try_from_f64(x) {
                    Some(iv) => Ok(Column::i64_with(set_each(v, positions, iv), validity_set(val, positions, true, len))),
                    None => Err(VolasError::DType(format!("Invalid value '{x}' for dtype 'int64'"))),
                },
                SetVal::Bool(b) => {
                    Ok(Column::i64_with(set_each(v, positions, b as i64), validity_set(val, positions, true, len)))
                }
            },
            Column::I32(v, val) => match value {
                SetVal::Num(x) if x.is_nan() => {
                    Ok(Column::i32_with(v.to_vec(), validity_set(val, positions, false, len)))
                }
                SetVal::Num(x) => match i32::try_from_f64(x) {
                    Some(iv) => Ok(Column::i32_with(set_each(v, positions, iv), validity_set(val, positions, true, len))),
                    None => Err(VolasError::DType(format!("Invalid value '{x}' for dtype 'int32'"))),
                },
                SetVal::Bool(b) => {
                    Ok(Column::i32_with(set_each(v, positions, b as i32), validity_set(val, positions, true, len)))
                }
            },
            Column::Bool(v, val) => match value {
                SetVal::Bool(b) => {
                    Ok(Column::bool_with(set_each(v, positions, b), validity_set(val, positions, true, len)))
                }
                SetVal::Num(x) if x.is_nan() => {
                    Ok(Column::bool_with(v.to_vec(), validity_set(val, positions, false, len)))
                }
                SetVal::Num(x) => Err(VolasError::DType(format!("Invalid value '{x}' for dtype 'bool'"))),
            },
            other => Err(VolasError::DType(format!(
                "cannot assign a scalar into a {} column",
                other.dtype()
            ))),
        }
    }

    /// Write a string scalar into a `Str` column at `positions`, preserving the
    /// existing validity (the written cells become present). Errors for a
    /// non-string column. Mirrors `set_scalar_at` for the string case (which a
    /// numeric `SetVal` cannot represent).
    pub fn set_str_scalar_at(&self, positions: &[usize], s: &str) -> Result<Column> {
        match self {
            Column::Str(v, val) => {
                let len = v.len();
                let mut nv = (**v).clone();
                for &i in positions {
                    nv[i] = s.to_string();
                }
                Ok(Column::str_with(nv, validity_set(val, positions, true, len)))
            }
            other => Err(VolasError::DType(format!(
                "cannot assign a string into a {} column",
                other.dtype()
            ))),
        }
    }

    // --- dtype-preserving numeric transforms (pandas 3.0) ---------------------
    // Each dispatches the kernel over the column's element type so an int column
    // stays int and computes natively (no f64 round-trip). A non-numeric column
    // is a `DType` error.

    /// Cumulative sum (pandas `cumsum`, skipna), dtype-preserving (`bool` -> int64
    /// counting trues). Missing values are skipped and stay missing (NA in → NA out).
    pub fn cumsum(&self) -> Result<Column> {
        match self {
            Column::Bool(v, val) => {
                let iv = widen_i64(v);
                let out = if val.has_nulls() {
                    cum_valid(&iv, val, 0, i64::wrapping_add)
                } else {
                    stats::cumsum(&iv)
                };
                Ok(Column::i64_with(out, val.clone()))
            }
            Column::I64(v, val) => Ok(cum(v, val, stats::cumsum, i64::wrapping_add)),
            Column::I32(v, val) => Ok(cum(v, val, stats::cumsum, i32::wrapping_add)),
            _ => numeric_dispatch!(self, v => Numeric::into_column(stats::cumsum(v))),
        }
    }
    /// Cumulative maximum (pandas `cummax`), dtype-preserving (`bool` running OR).
    /// Missing values are skipped and stay missing.
    pub fn cummax(&self) -> Result<Column> {
        match self {
            Column::Bool(v, val) => {
                let out = if val.has_nulls() {
                    cum_valid(v, val, false, |a, b| a || b)
                } else {
                    bool_running(v, true)
                };
                Ok(Column::bool_with(out, val.clone()))
            }
            Column::I64(v, val) => Ok(cum(v, val, stats::cummax, |a, x| if x > a { x } else { a })),
            Column::I32(v, val) => Ok(cum(v, val, stats::cummax, |a, x| if x > a { x } else { a })),
            _ => numeric_dispatch!(self, v => Numeric::into_column(stats::cummax(v))),
        }
    }
    /// Cumulative minimum (pandas `cummin`), dtype-preserving (`bool` running AND).
    /// Missing values are skipped and stay missing.
    pub fn cummin(&self) -> Result<Column> {
        match self {
            Column::Bool(v, val) => {
                let out = if val.has_nulls() {
                    cum_valid(v, val, true, |a, b| a && b)
                } else {
                    bool_running(v, false)
                };
                Ok(Column::bool_with(out, val.clone()))
            }
            Column::I64(v, val) => Ok(cum(v, val, stats::cummin, |a, x| if x < a { x } else { a })),
            Column::I32(v, val) => Ok(cum(v, val, stats::cummin, |a, x| if x < a { x } else { a })),
            _ => numeric_dispatch!(self, v => Numeric::into_column(stats::cummin(v))),
        }
    }
    /// Cumulative product (pandas `cumprod`), dtype-preserving (`bool` -> int64).
    /// Missing values are skipped and stay missing.
    pub fn cumprod(&self) -> Result<Column> {
        match self {
            Column::Bool(v, val) => {
                let iv = widen_i64(v);
                let out = if val.has_nulls() {
                    cum_valid(&iv, val, 1, i64::wrapping_mul)
                } else {
                    stats::cumprod(&iv)
                };
                Ok(Column::i64_with(out, val.clone()))
            }
            Column::I64(v, val) => Ok(cum(v, val, stats::cumprod, i64::wrapping_mul)),
            Column::I32(v, val) => Ok(cum(v, val, stats::cumprod, i32::wrapping_mul)),
            _ => numeric_dispatch!(self, v => Numeric::into_column(stats::cumprod(v))),
        }
    }

    /// Element-wise absolute value (pandas `abs`), dtype-preserving. `abs` of a
    /// missing (`NaN`) stays missing; `abs(i64::MIN)` wraps to `i64::MIN` (pandas);
    /// a `bool` column is unchanged (`abs(bool) == bool`).
    pub fn abs(&self) -> Result<Column> {
        if matches!(self, Column::Bool(_, _)) {
            return Ok(self.clone());
        }
        let validity = self.validity().cloned().unwrap_or_default();
        let out = numeric_dispatch!(self, v => Numeric::into_column(stats::abs(v)))?;
        Ok(out.with_validity(validity))
    }

    /// Round to `decimals` places (pandas `round`), dtype-preserving: banker's
    /// (half-to-even) for floats, and for ints an identity at `decimals >= 0` or a
    /// banker's round to the nearest power-of-ten multiple at negative `decimals`.
    pub fn round(&self, decimals: i32) -> Result<Column> {
        match self {
            Column::F64(v) => Ok(Column::f64(v.iter().map(|&x| round_f64(x, decimals)).collect())),
            Column::F32(v) => {
                Ok(Column::f32(v.iter().map(|&x| round_f64(x as f64, decimals) as f32).collect()))
            }
            Column::I64(v, val) => {
                Ok(Column::i64_with(v.iter().map(|&x| round_i64(x, decimals)).collect(), val.clone()))
            }
            Column::I32(v, val) => Ok(Column::i32_with(
                v.iter().map(|&x| round_i64(x as i64, decimals) as i32).collect(),
                val.clone(),
            )),
            Column::Bool(_, _) => Ok(self.clone()), // round(bool) == bool (pandas no-op)
            other => Err(VolasError::DType(format!("cannot round a {} column", other.dtype()))),
        }
    }

    /// Clamp to `[lower, upper]` (either bound optional), pandas `clip`. Stays in
    /// the column dtype when every present bound fits it losslessly; otherwise (an
    /// int column with a non-integral bound) promotes to float, matching pandas.
    pub fn clip(&self, lower: Option<f64>, upper: Option<f64>) -> Result<Column> {
        match self {
            // bool stays bool (pandas): a True lower bound forces all true, a
            // False upper bound forces all false, otherwise unchanged.
            Column::Bool(v, val) => return Ok(Column::bool_with(clip_bool(v, lower, upper), val.clone())),
            Column::F64(_) | Column::F32(_) | Column::I64(_, _) | Column::I32(_, _) => {}
            other => return Err(VolasError::DType(format!("cannot clip a {} column", other.dtype()))),
        }
        let bound_fits = |b: Option<f64>| b.map_or(true, |x| fits(self.dtype(), x));
        // Stay in dtype when every present bound fits it losslessly (a float dtype
        // always does); an int column with a non-integral bound promotes to float.
        let stay = self.dtype().is_float() || (bound_fits(lower) && bound_fits(upper));
        if stay {
            let validity = self.validity().cloned().unwrap_or_default();
            numeric_dispatch!(self, v => Numeric::into_column(clip_vec(v, lower, upper)))
                .map(|c| c.with_validity(validity))
        } else {
            // int -> f64 promotion: to_f64_vec already maps a missing value to NaN
            Ok(Column::f64(clip_vec(&self.to_f64_vec(), lower, upper)))
        }
    }

    /// Shift values by `n` periods (pandas `shift`), dtype-preserving. Vacated
    /// cells become missing (`NaN` for float, `volas.NA` for int/bool, `NaT` for
    /// datetime); a shifted-in value keeps its own missingness. A `str` column has
    /// no missing value of its own, so it degrades to an all-missing float column.
    pub fn shift(&self, n: isize) -> Column {
        let len = self.len();
        // Validity of the result: a cell is present when its source cell exists and
        // was itself present. Only the nullable (int/bool) variants build it; the
        // value buffers shift with a single `memcpy` via `shift_fill`.
        let nulls = || {
            Validity::from_valid_iter(
                len,
                (0..len).map(|i| {
                    let s = i as isize - n;
                    s >= 0 && (s as usize) < len && self.is_valid(s as usize)
                }),
            )
        };
        match self {
            Column::F64(v) => Column::f64(shift_fill(v, n, f64::NAN)),
            Column::F32(v) => Column::f32(shift_fill(v, n, f32::NAN)),
            Column::I64(v, _) => Column::i64_with(shift_fill(v, n, 0), nulls()),
            Column::I32(v, _) => Column::i32_with(shift_fill(v, n, 0), nulls()),
            Column::Bool(v, _) => Column::bool_with(shift_fill(v, n, false), nulls()),
            Column::Datetime(v) => Column::datetime(shift_fill(v, n, i64::MIN)),
            Column::Str(v, _) => {
                let vals = (0..len)
                    .map(|i| {
                        let s = i as isize - n;
                        if s >= 0 && (s as usize) < len {
                            v[s as usize].clone()
                        } else {
                            String::new() // placeholder, masked by the gap validity
                        }
                    })
                    .collect();
                Column::str_with(vals, nulls())
            }
        }
    }

    /// Discrete difference `x[i] - x[i-n]` (pandas `diff`), dtype-preserving: the
    /// first `n` cells (the shift gap) are missing. A `bool` / `str` / `datetime`
    /// column differences in f64 (no defined subtraction of its own).
    pub fn diff(&self, n: isize) -> Result<Column> {
        match self {
            Column::F64(v) => Ok(Column::f64(diff_kernel(v, n, f64::NAN))),
            Column::F32(v) => Ok(Column::f32(diff_kernel(v, n, f32::NAN))),
            Column::Bool(_, _) | Column::Str(_, _) | Column::Datetime(_) => {
                let (a, b) = (self.to_f64_vec(), self.shift(n).to_f64_vec());
                Ok(Column::f64(a.iter().zip(&b).map(|(&x, &y)| x - y).collect()))
            }
            // int stays int with an NA gap and NA-propagating subtraction.
            _ => self.binary(&self.shift(n), BinOp::Sub),
        }
    }

    /// Replace missing cells with the constant `value` (pandas `fillna`),
    /// dtype-preserving when `value` fits the dtype, else promoting an int column
    /// to float (a non-integral fill).
    pub fn fillna(&self, value: f64) -> Result<Column> {
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        let len = self.len();
        Ok(match self {
            Column::F64(v) => Column::f64(v.iter().map(|&x| if x.is_nan() { value } else { x }).collect()),
            Column::F32(v) => {
                Column::f32(v.iter().map(|&x| if x.is_nan() { value as f32 } else { x }).collect())
            }
            Column::I64(v, val) => match i64::try_from_f64(value) {
                Some(iv) => Column::i64((0..len).map(|i| if val.is_valid(i) { v[i] } else { iv }).collect()),
                None => Column::f64((0..len).map(|i| if val.is_valid(i) { v[i] as f64 } else { value }).collect()),
            },
            Column::I32(v, val) => match i32::try_from_f64(value) {
                Some(iv) => Column::i32((0..len).map(|i| if val.is_valid(i) { v[i] } else { iv }).collect()),
                None => Column::f64((0..len).map(|i| if val.is_valid(i) { v[i] as f64 } else { value }).collect()),
            },
            // a 0/1 fill keeps bool; a non-0/1 fill promotes the (numeric-family)
            // bool column to float.
            Column::Bool(v, val) if value == 0.0 || value == 1.0 => {
                Column::bool((0..len).map(|i| if val.is_valid(i) { v[i] } else { value != 0.0 }).collect())
            }
            Column::Bool(..) => {
                Column::f64(self.to_f64_vec().iter().map(|&x| if x.is_nan() { value } else { x }).collect())
            }
            // A numeric fill cannot apply to a non-numeric column: volas has no
            // `object` dtype to hold a mixed string/number or datetime/number
            // column, and the old f64 funnel silently turned valid strings into the
            // fill and lost the datetime dtype. Reject it.
            Column::Str(..) | Column::Datetime(..) => {
                return Err(VolasError::DType(format!(
                    "cannot fill a {} column with the numeric value {value} (volas has \
                     no object dtype); drop or select the missing rows instead",
                    self.dtype(),
                )))
            }
        })
    }

    /// Forward-fill (`forward = true`, pandas `ffill`) or backward-fill (`bfill`)
    /// missing cells from the nearest present value in that direction,
    /// dtype-preserving; leading / trailing cells with no source stay missing.
    pub fn fill_dir(&self, forward: bool) -> Column {
        if self.null_count() == 0 {
            return self.clone();
        }
        let len = self.len();
        // For each position, the source index of the value to carry in, or `None`.
        let mut src = vec![None; len];
        let mut last: Option<usize> = None;
        for k in 0..len {
            let i = if forward { k } else { len - 1 - k };
            if self.is_valid(i) {
                last = Some(i);
            }
            src[i] = last;
        }
        let validity = Validity::from_valid_iter(len, src.iter().map(|s| s.is_some()));
        match self {
            Column::F64(v) => Column::f64(src.iter().map(|s| s.map_or(f64::NAN, |j| v[j])).collect()),
            Column::F32(v) => Column::f32(src.iter().map(|s| s.map_or(f32::NAN, |j| v[j])).collect()),
            Column::I64(v, _) => Column::i64_with(src.iter().map(|s| s.map_or(0, |j| v[j])).collect(), validity),
            Column::I32(v, _) => Column::i32_with(src.iter().map(|s| s.map_or(0, |j| v[j])).collect(), validity),
            Column::Bool(v, _) => {
                Column::bool_with(src.iter().map(|s| s.map_or(false, |j| v[j])).collect(), validity)
            }
            Column::Datetime(v) => {
                Column::datetime(src.iter().map(|s| s.map_or(i64::MIN, |j| v[j])).collect())
            }
            // str carries missing too: gather the carried value (empty placeholder
            // for an unfilled cell, which `validity` then marks NA), like int/bool.
            Column::Str(v, _) => Column::str_with(
                src.iter().map(|s| s.map_or_else(String::new, |j| v[j].clone())).collect(),
                validity,
            ),
        }
    }

    /// `where` / `mask` core: pick `self` where `cond` is true, else `other`,
    /// producing `target` dtype (the caller resolves keep-vs-promote so the fill's
    /// value/type is accounted for). Picks i64 natively when `target` is `I64`
    /// (no f64 round-trip, so large ints stay exact). Equal lengths assumed.
    pub fn select(&self, cond: &[bool], other: &Column, target: DType) -> Result<Column> {
        match target {
            DType::I64 => Ok(Column::i64_with(
                stats::select(cond, &self.as_i64_vec()?, &other.as_i64_vec()?),
                self.select_nulls(cond, other),
            )),
            DType::I32 => Ok(Column::i32_with(
                stats::select(cond, &self.as_i32_vec()?, &other.as_i32_vec()?),
                self.select_nulls(cond, other),
            )),
            DType::F32 => Ok(Column::f32(stats::select(cond, &self.to_f32_vec(), &other.to_f32_vec()))),
            // bool ∘ bool stays bool (pandas keeps a bool result when the fill is
            // also bool); `bool` isn't `Numeric`, so the pick is inlined.
            DType::Bool => {
                let (a, b) = (self.as_bool_vec()?, other.as_bool_vec()?);
                Ok(Column::bool_with(
                    (0..cond.len()).map(|i| if cond[i] { a[i] } else { b[i] }).collect(),
                    self.select_nulls(cond, other),
                ))
            }
            // str / datetime stay in their dtype (a default `other` of NA keeps the
            // kept values and marks the rest missing) instead of an f64 funnel that
            // turned every kept string into NaN.
            DType::Utf8 => {
                let (a, b) = (self.as_str_vec()?, other.as_str_vec()?);
                Ok(Column::str_with(
                    (0..cond.len()).map(|i| if cond[i] { a[i].clone() } else { b[i].clone() }).collect(),
                    self.select_nulls(cond, other),
                ))
            }
            DType::Datetime => {
                let (a, b) = (self.as_datetime_vec()?, other.as_datetime_vec()?);
                Ok(Column::datetime(
                    (0..cond.len()).map(|i| if cond[i] { a[i] } else { b[i] }).collect(),
                ))
            }
            _ => Ok(Column::f64(stats::select(cond, &self.to_f64_vec(), &other.to_f64_vec()))),
        }
    }

    /// Picked validity for `select`: position `i` is present when the *chosen*
    /// side (`self` where `cond`, else `other`) is present there.
    fn select_nulls(&self, cond: &[bool], other: &Column) -> Validity {
        Validity::from_valid_iter(
            cond.len(),
            (0..cond.len()).map(|i| if cond[i] { self.is_valid(i) } else { other.is_valid(i) }),
        )
    }

    /// Combined validity of `self` and `other` (present only where both are) —
    /// the missing-value rule for a dtype-preserving binary op.
    fn combined_nulls(&self, other: &Column) -> Validity {
        let a = self.validity().cloned().unwrap_or_default();
        let b = other.validity().cloned().unwrap_or_default();
        a.and(&b)
    }

    /// The column as `bool` values (a `Bool` column directly; else an error).
    fn as_bool_vec(&self) -> Result<Vec<bool>> {
        match self {
            Column::Bool(v, _) => Ok(v.to_vec()),
            other => Err(VolasError::DType(format!("expected a bool column, got {}", other.dtype()))),
        }
    }

    /// Binary `+ - *` against `other`, dtype-preserving via [`binary_supertype`]
    /// (`int ∘ int → i64`, else f64). `bool ∘ bool` is logical, matching pandas:
    /// `+` is OR, `*` is AND, `-` is an error (numpy disallows bool subtraction);
    /// `bool ∘ number` promotes (bool acts as 0/1). Wrapping int ops match pandas
    /// overflow. Equal lengths assumed.
    pub fn binary(&self, other: &Column, op: BinOp) -> Result<Column> {
        if let (Column::Bool(a, av), Column::Bool(b, bv)) = (self, other) {
            let nulls = av.and(bv);
            return match op {
                BinOp::Add => {
                    Ok(Column::bool_with(a.iter().zip(b.iter()).map(|(&x, &y)| x || y).collect(), nulls))
                }
                BinOp::Mul => {
                    Ok(Column::bool_with(a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect(), nulls))
                }
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
            DType::F32 => Ok(Column::f32(binary_kernel(&self.to_f32_vec(), &other.to_f32_vec(), op))),
            _ => Ok(Column::f64(binary_kernel(&self.to_f64_vec(), &other.to_f64_vec(), op))),
        }
    }

    /// True division `self / other` (pandas `/`): always float. `bool / bool` is
    /// an error, matching pandas (division is not defined on bool).
    pub fn div(&self, other: &Column) -> Result<Column> {
        if matches!((self, other), (Column::Bool(_, _), Column::Bool(_, _))) {
            return Err(VolasError::DType(
                "division is not supported between two bool columns".into(),
            ));
        }
        let (a, b) = (self.to_f64_vec(), other.to_f64_vec());
        Ok(Column::f64(a.iter().zip(&b).map(|(&x, &y)| x / y).collect()))
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
            Column::F64(v) => Scalar::F64(stats::extreme(v.as_slice(), want_max).unwrap_or(f64::NAN)),
            other => {
                Scalar::F64(stats::extreme(&other.to_f64_vec(), want_max).unwrap_or(f64::NAN))
            }
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
                        i64::try_from_f64(x)
                            .ok_or_else(|| VolasError::DType(format!("value {x} does not fit int64")))
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
            other => Err(VolasError::DType(format!("cannot select a {} column as str", other.dtype()))),
        }
    }

    /// The column's epoch-ns values (a `Datetime` column directly); errors otherwise.
    fn as_datetime_vec(&self) -> Result<Vec<i64>> {
        match self {
            Column::Datetime(v) => Ok((**v).clone()),
            other => Err(VolasError::DType(format!("cannot select a {} column as datetime", other.dtype()))),
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
                        i32::try_from_f64(x)
                            .ok_or_else(|| VolasError::DType(format!("value {x} does not fit int32")))
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

    /// Value equality where `NaN == NaN` (pandas `equals` semantics), unlike the
    /// derived `PartialEq` (which uses IEEE `NaN != NaN`).
    pub fn equals(&self, other: &Column) -> bool {
        match (self, other) {
            (Column::F64(a), Column::F64(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
            }
            (Column::F32(a), Column::F32(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
            }
            _ => self == other,
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
    let flags: Vec<bool> =
        (0..alen).map(|i| av.is_valid(i)).chain((0..blen).map(|i| bv.is_valid(i))).collect();
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
    let mut it = v.iter().enumerate().filter(|(i, _)| val.is_valid(*i)).map(|(_, &x)| x);
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
fn cum<T: Numeric>(v: &[T], val: &Validity, dense: fn(&[T]) -> Vec<T>, op: fn(T, T) -> T) -> Column {
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

/// Write a scalar into a float column at `positions` (any value fits; f32 rounds).
fn set_float_at<T: Numeric>(v: &[T], positions: &[usize], value: SetVal) -> Column {
    let x = match value {
        SetVal::Num(x) => T::try_from_f64(x).unwrap_or(T::ZERO), // float try_from is always Some
        SetVal::Bool(b) => {
            if b {
                T::ONE
            } else {
                T::ZERO
            }
        }
    };
    let mut nv = v.to_vec();
    for &i in positions {
        nv[i] = x;
    }
    T::into_column(nv)
}

/// Copy `v` and write `x` into each of `positions` (scalar assignment).
fn set_each<T: Copy>(v: &[T], positions: &[usize], x: T) -> Vec<T> {
    let mut nv = v.to_vec();
    for &i in positions {
        nv[i] = x;
    }
    nv
}

/// The validity after a scalar write: every other row keeps its existing validity;
/// `positions` become present (`present = true`) or missing (a NA write). A dense
/// column stays dense when the write only sets values present (no allocation in
/// the common case).
fn validity_set(val: &Validity, positions: &[usize], present: bool, len: usize) -> Validity {
    if present && !val.has_nulls() {
        return Validity::dense();
    }
    let mut flags: Vec<bool> = (0..len).map(|i| val.is_valid(i)).collect();
    for &i in positions {
        flags[i] = present;
    }
    Validity::from_valid_iter(len, flags)
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
        .map(|&b| if force_false { false } else if force_true { true } else { b })
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
mod tests {
    use super::*;

    #[test]
    fn datetime_column_basics() {
        let c = Column::datetime(vec![10, 20, 30]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.dtype(), DType::Datetime);
        assert_eq!(c.as_datetime().unwrap(), &[10, 20, 30]);
        assert_eq!(c.get_f64(1), 20.0);
        assert_eq!(c.to_f64_vec(), vec![10.0, 20.0, 30.0]);
        assert_eq!(c.slice(1, 3), Column::datetime(vec![20, 30]));
        assert_eq!(c.take(&[2, 0]), Column::datetime(vec![30, 10]));
    }

    #[test]
    fn append_is_copy_on_write() {
        // A shared view must not see a later append (CoW), but an unshared column
        // grows in place.
        let mut a = Column::f64(vec![1.0, 2.0]);
        let view = a.clone(); // shares the Arc buffer
        a.append(&Column::f64(vec![3.0])).unwrap();
        assert_eq!(a.as_f64().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(view.as_f64().unwrap(), &[1.0, 2.0]); // view unchanged
    }

    #[test]
    fn datetime_append_same_dtype_only() {
        let mut a = Column::datetime(vec![1]);
        a.append(&Column::datetime(vec![2, 3])).unwrap();
        assert_eq!(a, Column::datetime(vec![1, 2, 3]));
        assert!(a.append(&Column::i64(vec![4])).is_err());
    }

    #[test]
    fn to_datetime_parses_strings() {
        let c = Column::str(vec!["2020-01-01".into(), "2020-01-02 03:04:05".into()]);
        let dt = c.to_datetime().unwrap();
        assert_eq!(dt.dtype(), DType::Datetime);
        assert_eq!(dt.len(), 2);
        // idempotent on an already-datetime column
        assert_eq!(dt.to_datetime().unwrap(), dt);
    }

    #[test]
    fn to_datetime_errors() {
        assert!(Column::str(vec!["not-a-date".into()])
            .to_datetime()
            .is_err());
        assert!(Column::i64(vec![1, 2]).to_datetime().is_err());
    }

    #[test]
    fn cast_between_dtypes_and_errors() {
        // no-op when already the target dtype
        let f = Column::f64(vec![1.0, 2.0]);
        assert_eq!(f.cast(DType::F64).unwrap(), f);

        // -> F64 (incl. the Str -> NaN arm of to_f64_vec)
        assert_eq!(
            Column::i64(vec![3]).cast(DType::F64).unwrap(),
            Column::f64(vec![3.0])
        );
        assert_eq!(
            Column::bool(vec![true, false]).cast(DType::F64).unwrap(),
            Column::f64(vec![1.0, 0.0])
        );
        let from_str = Column::str(vec!["a".into(), "b".into()])
            .cast(DType::F64)
            .unwrap();
        assert_eq!(from_str.dtype(), DType::F64);
        assert!(from_str.to_f64_vec().iter().all(|x| x.is_nan()));

        // -> I64 (F64 / Bool / Datetime; Str errors)
        assert_eq!(
            Column::f64(vec![2.9]).cast(DType::I64).unwrap(),
            Column::i64(vec![2])
        );
        assert_eq!(
            Column::bool(vec![true]).cast(DType::I64).unwrap(),
            Column::i64(vec![1])
        );
        assert_eq!(
            Column::datetime(vec![5]).cast(DType::I64).unwrap(),
            Column::i64(vec![5])
        );
        assert!(Column::str(vec!["x".into()]).cast(DType::I64).is_err());

        // -> Bool (F64 / I64; Str errors)
        assert_eq!(
            Column::f64(vec![0.0, 1.5]).cast(DType::Bool).unwrap(),
            Column::bool(vec![false, true])
        );
        assert_eq!(
            Column::i64(vec![0, 2]).cast(DType::Bool).unwrap(),
            Column::bool(vec![false, true])
        );
        assert!(Column::str(vec!["x".into()]).cast(DType::Bool).is_err());

        // -> Utf8 (every source variant of to_string_vec)
        assert_eq!(
            Column::f64(vec![1.5]).cast(DType::Utf8).unwrap(),
            Column::str(vec!["1.5".into()])
        );
        assert_eq!(
            Column::i64(vec![7]).cast(DType::Utf8).unwrap(),
            Column::str(vec!["7".into()])
        );
        assert_eq!(
            Column::bool(vec![true, false]).cast(DType::Utf8).unwrap(),
            Column::str(vec!["True".into(), "False".into()])
        );
        let dt_str = Column::datetime(vec![0]).cast(DType::Utf8).unwrap();
        assert_eq!(dt_str.dtype(), DType::Utf8);
        assert_eq!(dt_str.len(), 1);

        // -> Datetime (delegates to to_datetime)
        assert_eq!(
            Column::str(vec!["2020-01-01".into()])
                .cast(DType::Datetime)
                .unwrap()
                .dtype(),
            DType::Datetime
        );
    }

    #[test]
    fn equals_treats_nan_as_equal() {
        let a = Column::f64(vec![1.0, f64::NAN]);
        let b = Column::f64(vec![1.0, f64::NAN]);
        assert!(a.equals(&b)); // NaN == NaN here ...
        assert_ne!(a, b); // ... but derived PartialEq says NaN != NaN
        assert!(!a.equals(&Column::f64(vec![1.0]))); // length mismatch
        assert!(Column::i64(vec![1, 2]).equals(&Column::i64(vec![1, 2]))); // non-F64 fallback
        assert!(!Column::i64(vec![1]).equals(&Column::str(vec!["1".into()]))); // dtype mismatch
    }

    #[test]
    fn typed_accessors_reject_wrong_variant() {
        let f = Column::f64(vec![1.0]);
        assert!(f.as_bool().is_none());
        assert!(f.as_i64().is_none());
        assert!(f.as_str().is_none());
        assert!(f.as_datetime().is_none());
        assert!(Column::bool(vec![true]).as_f64().is_none());
        assert!(Column::f64(vec![]).is_empty());
    }

    #[test]
    fn per_variant_get_slice_take() {
        // get_f64 across the Bool / I64 / Str / F64 arms
        assert_eq!(Column::f64(vec![2.5]).get_f64(0), 2.5);
        assert_eq!(Column::bool(vec![true, false]).get_f64(0), 1.0);
        assert_eq!(Column::i64(vec![5]).get_f64(0), 5.0);
        assert!(Column::str(vec!["x".into()]).get_f64(0).is_nan());

        // slice / take across Bool / I64 / Str
        assert_eq!(
            Column::bool(vec![true, false, true]).slice(1, 3),
            Column::bool(vec![false, true])
        );
        assert_eq!(
            Column::i64(vec![1, 2, 3]).take(&[2, 0]),
            Column::i64(vec![3, 1])
        );
        assert_eq!(
            Column::str(vec!["a".into(), "b".into(), "c".into()]).take(&[1, 2]),
            Column::str(vec!["b".into(), "c".into()])
        );
        assert_eq!(
            Column::str(vec!["a".into(), "b".into()]).slice(0, 1),
            Column::str(vec!["a".into()])
        );

        // to_f64_vec Bool / I64 arms
        assert_eq!(Column::bool(vec![true, false]).to_f64_vec(), vec![1.0, 0.0]);
        assert_eq!(Column::i64(vec![3, 4]).to_f64_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn bool_get_false_branch_and_bool_append() {
        assert_eq!(Column::bool(vec![true, false]).get_f64(1), 0.0); // the `else { 0.0 }` arm
        let mut a = Column::bool(vec![true]);
        a.append(&Column::bool(vec![false, true])).unwrap();
        assert_eq!(a.as_bool().unwrap(), &[true, false, true]);
    }

    #[test]
    fn epoch_to_datetime_and_to_string_vec() {
        // epoch_to_datetime over int64 and float64 epochs; non-numeric dtypes error.
        assert!(Column::i64(vec![1, 2]).epoch_to_datetime("s").is_ok());
        assert!(Column::f64(vec![1.0, 2.0]).epoch_to_datetime("s").is_ok());
        assert!(Column::bool(vec![true]).epoch_to_datetime("s").is_err());
        // epoch_to_datetime_rounded preserves a fractional second; integers agree.
        assert_eq!(
            Column::f64(vec![1.5])
                .epoch_to_datetime_rounded("s")
                .unwrap(),
            Column::datetime(vec![1_500_000_000])
        );
        assert_eq!(
            Column::f64(vec![2.0]).epoch_to_datetime("s").unwrap(),
            Column::datetime(vec![2_000_000_000])
        );
        assert_eq!(
            Column::i64(vec![3]).epoch_to_datetime_rounded("s").unwrap(),
            Column::datetime(vec![3_000_000_000])
        );
        // the error closure on each numeric arm fires on an unknown unit
        assert!(Column::i64(vec![1]).epoch_to_datetime("weeks").is_err());
        assert!(Column::f64(vec![1.0]).epoch_to_datetime("weeks").is_err());
        assert!(Column::f64(vec![1.0])
            .epoch_to_datetime_rounded("weeks")
            .is_err());
        assert!(Column::bool(vec![true])
            .epoch_to_datetime_rounded("s")
            .is_err());
        // A missing epoch maps to NaT (i64::MIN), not 1970 / an error: a float NaN
        // and an int64 NA-bit both yield NaT, in both the truncating and rounded
        // variants; a present value still converts, and a bad unit still errors.
        let fnan = Column::f64(vec![f64::NAN, 2.0]).epoch_to_datetime("s").unwrap();
        assert!(!fnan.is_valid(0) && fnan.is_valid(1) && fnan.null_count() == 1);
        let fnan_r = Column::f64(vec![f64::NAN, 1.5]).epoch_to_datetime_rounded("s").unwrap();
        assert!(!fnan_r.is_valid(0) && fnan_r.is_valid(1));
        let ina = Column::i64_with(vec![0, 100], Validity::from_valid_iter(2, [false, true]))
            .epoch_to_datetime("s")
            .unwrap();
        assert!(!ina.is_valid(0) && ina.is_valid(1)); // NA-bit -> NaT, not epoch 0 -> 1970
        let ina_r = Column::i64_with(vec![5, 0], Validity::from_valid_iter(2, [true, false]))
            .epoch_to_datetime_rounded("s")
            .unwrap();
        assert!(ina_r.is_valid(0) && !ina_r.is_valid(1));
        // an all-NA float column is all NaT
        assert_eq!(
            Column::f64(vec![f64::NAN, f64::NAN]).epoch_to_datetime("s").unwrap().null_count(),
            2
        );
        // to_string_vec renders each supported dtype.
        assert_eq!(
            Column::str(vec!["a".into()]).to_string_vec(),
            vec!["a".to_string()]
        );
        assert_eq!(
            Column::f64(vec![1.5]).to_string_vec(),
            vec!["1.5".to_string()]
        );
        assert_eq!(Column::i64(vec![3]).to_string_vec(), vec!["3".to_string()]);
    }

    #[test]
    fn set_scalar_at_follows_pandas_dtype_rules() {
        use SetVal::{Bool, Num};
        // F64 stays F64 for a number or a bool.
        let f = Column::f64(vec![1.0, 2.0, 3.0]);
        assert_eq!(f.set_scalar_at(&[1], Num(9.0)).unwrap(), Column::f64(vec![1.0, 9.0, 3.0]));
        assert_eq!(f.set_scalar_at(&[0], Bool(false)).unwrap(), Column::f64(vec![0.0, 2.0, 3.0]));
        // I64 keeps int for an integral number or a bool.
        let i = Column::i64(vec![1, 2, 3]);
        assert_eq!(i.set_scalar_at(&[2], Num(0.0)).unwrap(), Column::i64(vec![1, 2, 0]));
        assert_eq!(i.set_scalar_at(&[0], Bool(false)).unwrap(), Column::i64(vec![0, 2, 3]));
        // I64 + NaN keeps int64, marking that cell NA (the NA model; no float upcast).
        let na = i.set_scalar_at(&[1], Num(f64::NAN)).unwrap();
        assert_eq!(na.dtype(), DType::I64);
        assert!(na.is_valid(0) && !na.is_valid(1) && na.is_valid(2));
        // I64 + a non-integral number is lossy -> error.
        assert!(i.set_scalar_at(&[0], Num(2.5)).is_err());
        // Bool keeps bool for a bool; a number into bool is lossy -> error.
        let b = Column::bool(vec![true, false]);
        assert_eq!(b.set_scalar_at(&[1], Bool(true)).unwrap(), Column::bool(vec![true, true]));
        assert!(b.set_scalar_at(&[0], Num(0.0)).is_err());
        // A scalar into a str column is unsupported -> error.
        assert!(Column::str(vec!["a".into()]).set_scalar_at(&[0], Num(1.0)).is_err());
    }

    #[test]
    fn cumulatives_preserve_dtype() {
        // i64 stays i64, computed natively
        assert_eq!(Column::i64(vec![1, 2, 3, 4]).cumsum().unwrap(), Column::i64(vec![1, 3, 6, 10]));
        assert_eq!(Column::i64(vec![3, 1, 4, 1]).cummax().unwrap(), Column::i64(vec![3, 3, 4, 4]));
        assert_eq!(Column::i64(vec![3, 1, 4, 1]).cummin().unwrap(), Column::i64(vec![3, 1, 1, 1]));
        assert_eq!(Column::i64(vec![1, 2, 3]).cumprod().unwrap(), Column::i64(vec![1, 2, 6]));
        // f64 keeps NaN in place (compare with equals: NaN == NaN)
        assert!(Column::f64(vec![1.0, f64::NAN, 2.0, 4.0]).cumsum().unwrap()
            .equals(&Column::f64(vec![1.0, f64::NAN, 3.0, 7.0])));
        assert!(Column::f64(vec![1.0, f64::NAN, 4.0, 2.0]).cummax().unwrap()
            .equals(&Column::f64(vec![1.0, f64::NAN, 4.0, 4.0])));
        assert!(Column::f64(vec![3.0, f64::NAN, 1.0]).cummin().unwrap()
            .equals(&Column::f64(vec![3.0, f64::NAN, 1.0])));
        assert!(Column::f64(vec![2.0, f64::NAN, 3.0]).cumprod().unwrap()
            .equals(&Column::f64(vec![2.0, f64::NAN, 6.0])));
        // bool is treated as i64 (pandas bool.cumsum -> int64); str -> error
        assert_eq!(Column::bool(vec![true, false, true]).cumsum().unwrap(), Column::i64(vec![1, 1, 2]));
        assert!(Column::str(vec!["a".into()]).cumsum().is_err());
    }

    #[test]
    fn abs_preserves_dtype_and_wraps() {
        assert!(Column::f64(vec![-1.0, f64::NAN, 2.0]).abs().unwrap()
            .equals(&Column::f64(vec![1.0, f64::NAN, 2.0])));
        // abs(i64::MIN) wraps to i64::MIN (pandas / numpy)
        assert_eq!(Column::i64(vec![-3, 4, i64::MIN]).abs().unwrap(), Column::i64(vec![3, 4, i64::MIN]));
    }

    #[test]
    fn round_preserves_dtype() {
        // f64 banker's, NaN passthrough
        assert!(Column::f64(vec![0.5, 1.5, 2.5, f64::NAN]).round(0).unwrap()
            .equals(&Column::f64(vec![0.0, 2.0, 2.0, f64::NAN])));
        // i64 identity at decimals>=0; banker's-to-multiple at negative decimals
        assert_eq!(Column::i64(vec![7, 8]).round(0).unwrap(), Column::i64(vec![7, 8]));
        assert_eq!(Column::i64(vec![15, 25, 35, 45, 5]).round(-1).unwrap(), Column::i64(vec![20, 20, 40, 40, 0]));
        assert_eq!(Column::i64(vec![16, 13]).round(-1).unwrap(), Column::i64(vec![20, 10])); // r>half / r<half
        assert_eq!(Column::i64(vec![-15, -25]).round(-1).unwrap(), Column::i64(vec![-20, -20])); // negative
        assert_eq!(Column::i64(vec![123]).round(-25).unwrap(), Column::i64(vec![0])); // 10^25 overflows -> 0
        assert_eq!(Column::bool(vec![true, false]).round(0).unwrap(), Column::bool(vec![true, false])); // bool no-op
        assert!(Column::str(vec!["a".into()]).round(0).is_err());
    }

    #[test]
    fn clip_preserves_dtype_or_promotes() {
        use DType::{F64, I64};
        // f64: both bounds, lo-only, hi-only, no bounds, NaN passthrough
        assert!(Column::f64(vec![-1.0, 1.0, 3.0, f64::NAN]).clip(Some(0.0), Some(2.0)).unwrap()
            .equals(&Column::f64(vec![0.0, 1.0, 2.0, f64::NAN])));
        assert_eq!(Column::f64(vec![-1.0, 5.0]).clip(Some(0.0), None).unwrap(), Column::f64(vec![0.0, 5.0]));
        assert_eq!(Column::f64(vec![-1.0, 5.0]).clip(None, Some(2.0)).unwrap(), Column::f64(vec![-1.0, 2.0]));
        assert_eq!(Column::f64(vec![1.0, 5.0]).clip(None, None).unwrap(), Column::f64(vec![1.0, 5.0]));
        // i64 with integral bounds stays int
        assert_eq!(Column::i64(vec![1, 5, 9]).clip(Some(2.0), Some(8.0)).unwrap(), Column::i64(vec![2, 5, 8]));
        // i64 with a non-integral bound promotes to float (pandas)
        let p = Column::i64(vec![1, 5, 9]).clip(Some(2.5), None).unwrap();
        assert_eq!(p.dtype(), F64);
        assert_eq!(p, Column::f64(vec![2.5, 5.0, 9.0]));
        let _ = I64;
        // bool stays bool: clip(F,T) no-op, clip(T,T) forces true, clip(F,F) forces false
        assert_eq!(Column::bool(vec![true, false]).clip(Some(0.0), Some(1.0)).unwrap(),
                   Column::bool(vec![true, false]));
        assert_eq!(Column::bool(vec![true, false]).clip(Some(1.0), Some(1.0)).unwrap(),
                   Column::bool(vec![true, true]));
        assert_eq!(Column::bool(vec![true, false]).clip(Some(0.0), Some(0.0)).unwrap(),
                   Column::bool(vec![false, false]));
        assert_eq!(Column::bool(vec![true, false]).clip(None, None).unwrap(),
                   Column::bool(vec![true, false]));
        assert!(Column::str(vec!["a".into()]).clip(None, None).is_err());
    }

    #[test]
    fn select_picks_in_target_dtype() {
        let cond = [true, false, true];
        let a = Column::i64(vec![1, 2, 3]);
        // target I64: other is i64 (direct) and f64-integral (lossless narrow)
        assert_eq!(a.select(&cond, &Column::i64(vec![10, 20, 30]), DType::I64).unwrap(),
                   Column::i64(vec![1, 20, 3]));
        assert_eq!(a.select(&cond, &Column::f64(vec![10.0, 20.0, 30.0]), DType::I64).unwrap(),
                   Column::i64(vec![1, 20, 3]));
        // target F64
        assert_eq!(Column::f64(vec![1.0, 2.0, 3.0])
            .select(&cond, &Column::f64(vec![10.0, 20.0, 30.0]), DType::F64).unwrap(),
            Column::f64(vec![1.0, 20.0, 3.0]));
        // as_i64_vec error: target I64 but a value is non-integral
        assert!(a.select(&cond, &Column::f64(vec![1.5, 2.0, 3.0]), DType::I64).is_err());
    }

    #[test]
    fn binary_and_div_dtype() {
        use DType::{F64, I64};
        let a = Column::i64(vec![5, 7]);
        let b = Column::i64(vec![2, 3]);
        assert_eq!(a.binary(&b, BinOp::Add).unwrap(), Column::i64(vec![7, 10]));
        assert_eq!(a.binary(&b, BinOp::Sub).unwrap(), Column::i64(vec![3, 4]));
        assert_eq!(a.binary(&b, BinOp::Mul).unwrap(), Column::i64(vec![10, 21]));
        // int + float -> f64
        let r = a.binary(&Column::f64(vec![2.0, 3.0]), BinOp::Add).unwrap();
        assert_eq!(r.dtype(), F64);
        assert_eq!(r, Column::f64(vec![7.0, 10.0]));
        // wrapping overflow matches pandas int64
        assert_eq!(Column::i64(vec![i64::MAX]).binary(&Column::i64(vec![1]), BinOp::Add).unwrap(),
                   Column::i64(vec![i64::MIN]));
        // div is always float
        assert_eq!(a.div(&b).unwrap().dtype(), F64);
        assert_eq!(a.div(&b).unwrap(), Column::f64(vec![2.5, 7.0 / 3.0]));
        let _ = I64;
    }

    #[test]
    fn reductions_carry_result_dtype() {
        use Scalar::{Bool as SB, F64, I64};
        // sum / prod: float -> F64; int / bool -> I64; non-numeric -> F64 (f64 fallback)
        assert_eq!(Column::f64(vec![1.0, f64::NAN, 2.0]).sum(), F64(3.0));
        assert_eq!(Column::i64(vec![1, 2, 3]).sum(), I64(6));
        assert_eq!(Column::bool(vec![true, false, true]).sum(), I64(2));
        assert!(matches!(Column::str(vec!["a".into()]).sum(), F64(_)));
        assert_eq!(Column::f64(vec![2.0, 3.0]).prod(), F64(6.0));
        assert_eq!(Column::i64(vec![2, 3, 4]).prod(), I64(24));
        assert_eq!(Column::bool(vec![true, true]).prod(), I64(1));
        assert!(matches!(Column::str(vec!["a".into()]).prod(), F64(_)));
        // min / max keep dtype: int -> I64, bool -> Bool, float -> F64
        assert_eq!(Column::i64(vec![3, 1, 2]).extreme(false), I64(1));
        assert_eq!(Column::i64(vec![3, 1, 2]).extreme(true), I64(3));
        assert_eq!(Column::bool(vec![true, false, true]).extreme(false), SB(false)); // AND
        assert_eq!(Column::bool(vec![true, false, true]).extreme(true), SB(true)); // OR
        assert_eq!(Column::f64(vec![3.0, 1.0]).extreme(false), F64(1.0));
        assert!(matches!(Column::str(vec!["a".into()]).extreme(true), F64(_)));
        // empty / all-missing extreme -> NaN (F64)
        assert!(matches!(Column::i64(vec![]).extreme(false), F64(x) if x.is_nan()));
        assert!(matches!(Column::bool(vec![]).extreme(true), F64(x) if x.is_nan()));
        assert!(matches!(Column::f64(vec![]).extreme(true), F64(x) if x.is_nan()));
    }

    #[test]
    fn f32_i32_columns() {
        use Scalar::{F32, I32, I64};
        let f = Column::f32(vec![1.5, 2.5, 3.5]);
        let i = Column::i32(vec![3, 1, 4]);
        // storage basics
        assert_eq!((f.dtype(), i.dtype(), f.len()), (DType::F32, DType::I32, 3));
        assert_eq!(f.to_f64_vec(), vec![1.5, 2.5, 3.5]);
        assert_eq!(i.get_f64(0), 3.0);
        assert_eq!(f.slice(0, 2), Column::f32(vec![1.5, 2.5]));
        assert_eq!(i.take(&[2, 0]), Column::i32(vec![4, 3]));
        assert_eq!(Column::i64(vec![1, 2]).to_f32_vec(), vec![1.0_f32, 2.0]);
        assert_eq!(i.to_string_vec(), vec!["3", "1", "4"]);
        assert!(Column::f32(vec![f32::NAN]).equals(&Column::f32(vec![f32::NAN]))); // NaN == NaN
        let mut a = Column::f32(vec![1.0]);
        a.append(&Column::f32(vec![2.0])).unwrap();
        assert_eq!(a, Column::f32(vec![1.0, 2.0]));
        // cast
        assert_eq!(Column::f64(vec![1.5]).cast(DType::F32).unwrap(), Column::f32(vec![1.5]));
        assert_eq!(Column::f64(vec![3.0]).cast(DType::I32).unwrap(), Column::i32(vec![3]));
        assert!(Column::f64(vec![2.5]).cast(DType::I32).is_err()); // non-integral
        assert!(Column::f64(vec![3e9]).cast(DType::I32).is_err()); // out of range
        assert_eq!(f.cast(DType::F64).unwrap(), Column::f64(vec![1.5, 2.5, 3.5]));
        // reductions: f32 -> F32; i32 sum -> I64 (promotes), min -> I32
        assert_eq!(f.sum(), F32(7.5));
        assert_eq!(f.extreme(false), F32(1.5));
        assert_eq!(i.sum(), I64(8));
        assert_eq!(i.prod(), I64(12));
        assert_eq!(i.extreme(true), I32(4));
        // round / clip preserve dtype
        assert_eq!(Column::f32(vec![1.4, 2.6]).round(0).unwrap(), Column::f32(vec![1.0, 3.0]));
        assert_eq!(i.round(-1).unwrap().dtype(), DType::I32);
        assert_eq!(f.clip(Some(2.0), Some(3.0)).unwrap(), Column::f32(vec![2.0, 2.5, 3.0]));
        assert_eq!(i.clip(Some(2.0), Some(3.0)).unwrap().dtype(), DType::I32);
        // binary: same-dtype preserves
        assert_eq!(f.binary(&f, BinOp::Add).unwrap(), Column::f32(vec![3.0, 5.0, 7.0]));
        assert_eq!(i.binary(&i, BinOp::Add).unwrap(), Column::i32(vec![6, 2, 8]));
        // select (where/mask) in f32 / i32 target
        let cond = [true, false, true];
        assert_eq!(f.select(&cond, &Column::f32(vec![0.0, 0.0, 0.0]), DType::F32).unwrap(),
                   Column::f32(vec![1.5, 0.0, 3.5]));
        assert_eq!(i.select(&cond, &Column::i32(vec![0, 0, 0]), DType::I32).unwrap(),
                   Column::i32(vec![3, 0, 4]));
        // assignment: f32 writes; i32 keeps the dtype (a NaN write marks the cell
        // NA, no float upcast), rejects a lossy value
        assert_eq!(f.set_scalar_at(&[1], SetVal::Num(9.0)).unwrap(), Column::f32(vec![1.5, 9.0, 3.5]));
        assert_eq!(i.set_scalar_at(&[1], SetVal::Bool(true)).unwrap(), Column::i32(vec![3, 1, 4]));
        assert_eq!(i.set_scalar_at(&[1], SetVal::Num(9.0)).unwrap(), Column::i32(vec![3, 9, 4]));
        let i_na = i.set_scalar_at(&[0], SetVal::Num(f64::NAN)).unwrap();
        assert!(i_na.dtype() == DType::I32 && !i_na.is_valid(0) && i_na.is_valid(1));
        assert!(i.set_scalar_at(&[0], SetVal::Num(2.5)).is_err());
        assert_eq!(f.set_scalar_at(&[0], SetVal::Bool(false)).unwrap(), Column::f32(vec![0.0, 2.5, 3.5]));
        // remaining f32/i32 arms (both directions of slice/take, the other reductions,
        // sub/mul kernels through the trait, bool->i32, append/append_missing)
        assert_eq!(f.get_f64(0), 1.5);
        assert_eq!(i.slice(1, 3), Column::i32(vec![1, 4]));
        assert_eq!(f.take(&[2, 0]), Column::f32(vec![3.5, 1.5]));
        assert_eq!(f.to_string_vec(), vec!["1.5", "2.5", "3.5"]);
        assert_eq!(f.prod(), F32(13.125));
        assert!(matches!(Column::i32(vec![]).extreme(false), Scalar::F64(x) if x.is_nan()));
        assert_eq!(f.binary(&f, BinOp::Sub).unwrap(), Column::f32(vec![0.0, 0.0, 0.0]));
        assert_eq!(f.binary(&f, BinOp::Mul).unwrap(), Column::f32(vec![2.25, 6.25, 12.25]));
        assert_eq!(i.binary(&i, BinOp::Sub).unwrap(), Column::i32(vec![0, 0, 0]));
        assert_eq!(i.binary(&i, BinOp::Mul).unwrap(), Column::i32(vec![9, 1, 16]));
        assert_eq!(Column::bool(vec![true, false, true]).binary(&i, BinOp::Add).unwrap(),
                   Column::i32(vec![4, 1, 5]));
        // as_i32_vec f64 fallback (lossless narrow + lossy error)
        assert_eq!(i.select(&cond, &Column::f64(vec![0.0, 0.0, 0.0]), DType::I32).unwrap(),
                   Column::i32(vec![3, 0, 4]));
        assert!(i.select(&cond, &Column::f64(vec![2.5, 0.0, 0.0]), DType::I32).is_err());
        assert_eq!(f.set_scalar_at(&[0], SetVal::Bool(true)).unwrap(), Column::f32(vec![1.0, 2.5, 3.5]));
        let mut ii = Column::i32(vec![1]);
        ii.append(&Column::i32(vec![2])).unwrap();
        assert_eq!(ii, Column::i32(vec![1, 2]));
        assert!(Column::f32(vec![1.0]).append(&Column::i32(vec![1])).is_err());
        let mut fm = Column::f32(vec![1.0]);
        fm.append_missing(2).unwrap();
        assert!(matches!(&fm, Column::F32(v) if v.len() == 3 && v[1].is_nan()));
    }

    #[test]
    fn bool_matches_pandas() {
        let b = || Column::bool(vec![true, false, true]);
        let c = Column::bool(vec![true, true, false]);
        // cumsum / cumprod -> int64 (counts / product)
        assert_eq!(b().cumsum().unwrap(), Column::i64(vec![1, 1, 2]));
        assert_eq!(b().cumprod().unwrap(), Column::i64(vec![1, 0, 0]));
        // cummax / cummin -> bool (running OR / AND)
        assert_eq!(b().cummax().unwrap(), Column::bool(vec![true, true, true]));
        assert_eq!(b().cummin().unwrap(), Column::bool(vec![true, false, false]));
        // abs -> bool (identity)
        assert_eq!(b().abs().unwrap(), b());
        // + is OR, * is AND, - is an error
        assert_eq!(b().binary(&c, BinOp::Add).unwrap(), Column::bool(vec![true, true, true]));
        assert_eq!(b().binary(&c, BinOp::Mul).unwrap(), Column::bool(vec![true, false, false]));
        assert!(b().binary(&c, BinOp::Sub).is_err());
        // bool / bool -> error; bool ∘ number promotes (bool acts as 0/1)
        assert!(b().div(&c).is_err());
        assert_eq!(b().binary(&Column::i64(vec![1, 1, 1]), BinOp::Add).unwrap(), Column::i64(vec![2, 1, 2]));
        let f = b().binary(&Column::f64(vec![1.0, 1.0, 1.0]), BinOp::Add).unwrap();
        assert_eq!(f.dtype(), DType::F64);
        // where/mask with a bool fill stays bool (Column::select Bool target)
        let cond = [true, false, true];
        assert_eq!(
            b().select(&cond, &Column::bool(vec![false, false, false]), DType::Bool).unwrap(),
            Column::bool(vec![true, false, true])
        );
        assert!(Column::i64(vec![1]).as_bool_vec().is_err()); // non-bool -> error
    }

    // --- NA (validity) behaviour ---------------------------------------------

    fn na_i64(vals: &[i64], present: &[bool]) -> Column {
        Column::i64_with(vals.to_vec(), Validity::from_valid_iter(present.len(), present.iter().copied()))
    }

    /// Assert a column's present/missing pattern and present values (NA -> NaN).
    fn assert_na(c: &Column, expected: &[f64]) {
        let got = c.to_f64_vec();
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected) {
            if e.is_nan() {
                assert!(g.is_nan(), "expected NA, got {g}");
            } else {
                assert_eq!(g, e);
            }
        }
    }

    #[test]
    fn na_is_valid_and_null_count() {
        let c = na_i64(&[1, 0, 3], &[true, false, true]);
        assert!(c.is_valid(0) && !c.is_valid(1) && c.is_valid(2));
        assert_eq!(c.null_count(), 1);
        // float -> NaN, datetime -> NaT (i64::MIN), str -> never missing
        let f = Column::f64(vec![1.0, f64::NAN]);
        assert!(!f.is_valid(1) && f.null_count() == 1);
        let f32c = Column::f32(vec![1.0, f32::NAN]);
        assert!(!f32c.is_valid(1) && f32c.null_count() == 1);
        let d = Column::datetime(vec![i64::MIN, 5]);
        assert!(!d.is_valid(0) && d.is_valid(1) && d.null_count() == 1);
        let s = Column::str(vec!["a".into()]);
        assert!(s.is_valid(0) && s.null_count() == 0);
        // to_f64_vec maps NA -> NaN
        assert_na(&c, &[1.0, f64::NAN, 3.0]);
    }

    #[test]
    fn na_reductions_skip_missing() {
        let c = na_i64(&[1, 0, 3], &[true, false, true]); // 1, NA, 3
        assert_eq!(c.sum(), Scalar::I64(4));
        assert_eq!(c.prod(), Scalar::I64(3));
        assert_eq!(c.extreme(false), Scalar::I64(1));
        assert_eq!(c.extreme(true), Scalar::I64(3));
        // all-NA int -> NaN
        assert!(matches!(na_i64(&[0, 0], &[false, false]).extreme(true), Scalar::F64(x) if x.is_nan()));
        // i32 / bool skip NA, promote to i64
        let i = Column::i32_with(vec![5, 0, 7], Validity::from_valid_iter(3, [true, false, true]));
        assert_eq!(i.sum(), Scalar::I64(12));
        assert_eq!(i.extreme(false), Scalar::I32(5));
        let b = Column::bool_with(vec![true, false, false], Validity::from_valid_iter(3, [true, false, true]));
        assert_eq!(b.sum(), Scalar::I64(1)); // present trues: pos0
        assert_eq!(b.extreme(true), Scalar::Bool(true)); // any present
        assert_eq!(b.extreme(false), Scalar::Bool(false)); // all present (true && false)
        assert!(matches!(
            Column::bool_with(vec![false], Validity::from_valid_iter(1, [false])).extreme(true),
            Scalar::F64(x) if x.is_nan()
        ));
        // the f64-funnel (mean / std / …) skips NA because to_f64_vec maps it to
        // NaN: here the present values are 1 and 3.
        let present: Vec<f64> = c.to_f64_vec().into_iter().filter(|x| !x.is_nan()).collect();
        assert_eq!(present, vec![1.0, 3.0]);
    }

    #[test]
    fn na_cumulatives_propagate() {
        assert_na(&na_i64(&[1, 0, 3], &[true, false, true]).cumsum().unwrap(), &[1.0, f64::NAN, 4.0]);
        assert_na(&na_i64(&[2, 0, 3], &[true, false, true]).cumprod().unwrap(), &[2.0, f64::NAN, 6.0]);
        assert_na(&na_i64(&[3, 0, 1], &[true, false, true]).cummax().unwrap(), &[3.0, f64::NAN, 3.0]);
        assert_na(&na_i64(&[3, 0, 1], &[true, false, true]).cummin().unwrap(), &[3.0, f64::NAN, 1.0]);
        // i32
        let i = Column::i32_with(vec![1, 0, 3], Validity::from_valid_iter(3, [true, false, true]));
        assert_na(&i.cumsum().unwrap(), &[1.0, f64::NAN, 4.0]);
        assert_na(&i.cummax().unwrap(), &[1.0, f64::NAN, 3.0]);
        assert_na(&i.cummin().unwrap(), &[1.0, f64::NAN, 1.0]);
        assert_na(&i.cumprod().unwrap(), &[1.0, f64::NAN, 3.0]);
        // bool: cumsum/cumprod -> i64; cummax = running OR, cummin = running AND
        let b = Column::bool_with(vec![true, false, false], Validity::from_valid_iter(3, [true, false, true]));
        assert_eq!(b.cumsum().unwrap().dtype(), DType::I64);
        assert_na(&b.cumsum().unwrap(), &[1.0, f64::NAN, 1.0]);
        assert_na(&b.cumprod().unwrap(), &[1.0, f64::NAN, 0.0]);
        assert_na(&b.cummax().unwrap(), &[1.0, f64::NAN, 1.0]);
        assert_na(&b.cummin().unwrap(), &[1.0, f64::NAN, 0.0]);
    }

    #[test]
    fn na_elementwise_propagate() {
        assert_na(&na_i64(&[-1, 0, -3], &[true, false, true]).abs().unwrap(), &[1.0, f64::NAN, 3.0]);
        assert_na(&na_i64(&[12, 0, 28], &[true, false, true]).round(-1).unwrap(), &[10.0, f64::NAN, 30.0]);
        assert_na(&na_i64(&[1, 0, 9], &[true, false, true]).clip(Some(2.0), Some(8.0)).unwrap(), &[2.0, f64::NAN, 8.0]);
        // a non-integral bound promotes int -> float; NA stays NA (NaN)
        let promoted = na_i64(&[1, 0, 9], &[true, false, true]).clip(Some(2.5), None).unwrap();
        assert_eq!(promoted.dtype(), DType::F64);
        assert_na(&promoted, &[2.5, f64::NAN, 9.0]);
        // i32 round keeps i32 + validity
        let r = Column::i32_with(vec![28, 0, 12], Validity::from_valid_iter(3, [true, false, true])).round(-1).unwrap();
        assert_eq!(r.dtype(), DType::I32);
        assert_na(&r, &[30.0, f64::NAN, 10.0]);
    }

    #[test]
    fn na_binary_and_select_propagate() {
        // x ∘ NA = NA: combined validity (present only where both present)
        let a = na_i64(&[1, 0, 3], &[true, false, true]);
        let b = na_i64(&[10, 20, 0], &[true, true, false]);
        assert_na(&a.binary(&b, BinOp::Add).unwrap(), &[11.0, f64::NAN, f64::NAN]);
        // bool ∘ bool keeps bool, combined validity
        let bt = Column::bool_with(vec![true, false, true], Validity::from_valid_iter(3, [true, false, true]));
        let bf = Column::bool_with(vec![false, true, true], Validity::from_valid_iter(3, [true, true, false]));
        assert_na(&bt.binary(&bf, BinOp::Add).unwrap(), &[1.0, f64::NAN, f64::NAN]); // OR
        // bool ∘ int promotes to int (validity()'s Bool arm)
        assert_na(&bt.binary(&Column::i64(vec![1, 1, 1]), BinOp::Add).unwrap(), &[2.0, f64::NAN, 2.0]);
        // division funnels through f64 (NA -> NaN automatically)
        assert_na(&a.div(&Column::f64(vec![2.0, 2.0, 2.0])).unwrap(), &[0.5, f64::NAN, 1.5]);
        // select carries the chosen side's validity
        let other = na_i64(&[9, 9, 9], &[true, true, true]);
        let r = a.select(&[true, true, false], &other, DType::I64).unwrap();
        assert_na(&r, &[1.0, f64::NAN, 9.0]);
        // bool select target carries validity too
        let rb = bt.select(&[true, true, false], &Column::bool(vec![false, false, false]), DType::Bool).unwrap();
        assert_eq!(rb.dtype(), DType::Bool);
        assert_na(&rb, &[1.0, f64::NAN, 0.0]);
        // as_i32_vec's NA-placeholder branch: an i32 target with an f64 fill whose
        // NaN is selected becomes NA (value 0, masked).
        let i32c = Column::i32(vec![1, 2, 3]);
        let r32 = i32c.select(&[true, false, false], &Column::f64(vec![9.0, f64::NAN, 9.0]), DType::I32).unwrap();
        assert_eq!(r32.dtype(), DType::I32);
        assert_na(&r32, &[1.0, f64::NAN, 9.0]);
    }

    #[test]
    fn na_shift_and_diff() {
        // float gap = NaN
        assert_na(&Column::f64(vec![1.0, 2.0, 3.0]).shift(1), &[f64::NAN, 1.0, 2.0]);
        assert_na(&Column::f32(vec![1.0, 2.0, 3.0]).shift(1), &[f64::NAN, 1.0, 2.0]);
        // int / bool keep their dtype with an NA gap (PDEP-16 alignment)
        let i = Column::i64(vec![1, 2, 3]);
        assert_eq!(i.shift(1).dtype(), DType::I64);
        assert_na(&i.shift(1), &[f64::NAN, 1.0, 2.0]);
        assert_na(&i.shift(-1), &[2.0, 3.0, f64::NAN]);
        assert_na(&Column::i32(vec![1, 2, 3]).shift(1), &[f64::NAN, 1.0, 2.0]);
        let b = Column::bool(vec![true, false, true]);
        assert_eq!(b.shift(1).dtype(), DType::Bool);
        assert_na(&b.shift(1), &[f64::NAN, 1.0, 0.0]);
        // datetime gap = NaT; str degrades to an all-missing float column
        assert_na(&Column::datetime(vec![10, 20, 30]).shift(1), &[f64::NAN, 10.0, 20.0]);
        assert_na(&Column::str(vec!["a".into(), "b".into()]).shift(1), &[f64::NAN, f64::NAN]);
        // shift carries a pre-existing NA; shift(0) is identity; beyond len -> all NA
        assert_na(&na_i64(&[1, 0, 3], &[true, false, true]).shift(1), &[f64::NAN, 1.0, f64::NAN]);
        assert_na(&i.shift(0), &[1.0, 2.0, 3.0]);
        assert_na(&i.shift(5), &[f64::NAN, f64::NAN, f64::NAN]);

        // diff: int keeps int + NA gap, float stays float, bool/datetime -> f64
        assert_eq!(Column::i64(vec![1, 3, 6]).diff(1).unwrap().dtype(), DType::I64);
        assert_na(&Column::i64(vec![1, 3, 6]).diff(1).unwrap(), &[f64::NAN, 2.0, 3.0]);
        assert_na(&Column::f64(vec![1.0, 3.0, 6.0]).diff(1).unwrap(), &[f64::NAN, 2.0, 3.0]);
        assert_na(&Column::f32(vec![1.0, 3.0, 6.0]).diff(1).unwrap(), &[f64::NAN, 2.0, 3.0]);
        // negative-n diff (the backward branch of diff_kernel)
        assert_na(&Column::f64(vec![1.0, 3.0, 6.0]).diff(-1).unwrap(), &[-2.0, -3.0, f64::NAN]);
        let bd = Column::bool(vec![true, false, true]).diff(1).unwrap();
        assert_eq!(bd.dtype(), DType::F64);
        assert_na(&bd, &[f64::NAN, -1.0, 1.0]);
        assert_na(&Column::datetime(vec![10, 25, 30]).diff(1).unwrap(), &[f64::NAN, 15.0, 5.0]);
    }

    #[test]
    fn na_take_slice_append_preserve_validity() {
        let c = na_i64(&[1, 0, 3, 0, 5], &[true, false, true, false, true]); // 1,NA,3,NA,5
        assert_na(&c.slice(1, 4), &[f64::NAN, 3.0, f64::NAN]);
        assert_na(&c.take(&[4, 1, 0]), &[5.0, f64::NAN, 1.0]);
        // dense slice / take stay dense (the validity fast path)
        assert_eq!(Column::i64(vec![1, 2, 3]).slice(0, 2).null_count(), 0);
        assert_eq!(Column::i64(vec![1, 2, 3]).take(&[2, 0]).null_count(), 0);
        // append concatenates validity (NA ++ NA)
        let mut a = na_i64(&[1, 0], &[true, false]);
        a.append(&na_i64(&[0, 4], &[false, true])).unwrap();
        assert_na(&a, &[1.0, f64::NAN, f64::NAN, 4.0]);
        // dense ++ dense stays dense
        let mut d = Column::i64(vec![1, 2]);
        d.append(&Column::i64(vec![3])).unwrap();
        assert_eq!(d.null_count(), 0);
        // append_missing keeps the new bool rows dense `false` (refresh placeholder)
        let mut b = Column::bool(vec![true, false]);
        b.append_missing(2).unwrap();
        assert_eq!(b.null_count(), 0);
        assert_na(&b, &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn na_fillna() {
        // keep dtype when the fill fits; promote int -> float on a non-integral fill
        let c = na_i64(&[1, 0, 3], &[true, false, true]);
        assert_eq!(c.fillna(9.0).unwrap().dtype(), DType::I64);
        assert_na(&c.fillna(9.0).unwrap(), &[1.0, 9.0, 3.0]);
        assert_eq!(c.fillna(2.5).unwrap().dtype(), DType::F64);
        assert_na(&c.fillna(2.5).unwrap(), &[1.0, 2.5, 3.0]);
        let i32c = Column::i32_with(vec![1, 0, 3], Validity::from_valid_iter(3, [true, false, true]));
        assert_eq!(i32c.fillna(9.0).unwrap().dtype(), DType::I32);
        assert_na(&i32c.fillna(9.0).unwrap(), &[1.0, 9.0, 3.0]);
        assert_eq!(i32c.fillna(2.5).unwrap().dtype(), DType::F64);
        // bool: a 0/1 fill keeps bool, else promote to float
        let bc = Column::bool_with(vec![true, false, false], Validity::from_valid_iter(3, [true, false, true]));
        assert_eq!(bc.fillna(1.0).unwrap().dtype(), DType::Bool);
        assert_na(&bc.fillna(1.0).unwrap(), &[1.0, 1.0, 0.0]);
        assert_eq!(bc.fillna(5.0).unwrap().dtype(), DType::F64);
        // float fill; a dense column is cloned unchanged
        assert_na(&Column::f64(vec![1.0, f64::NAN]).fillna(0.0).unwrap(), &[1.0, 0.0]);
        assert_na(&Column::f32(vec![1.0, f32::NAN]).fillna(0.0).unwrap(), &[1.0, 0.0]);
        assert_eq!(Column::i64(vec![1, 2]).fillna(9.0).unwrap(), Column::i64(vec![1, 2]));
        // a numeric fill on a non-numeric column (str / datetime) is rejected, not
        // silently funneled through f64 (which corrupted strings / lost the dtype)
        assert!(Column::datetime(vec![i64::MIN, 20]).fillna(9.0).is_err());
        assert!(Column::str_with(
            vec!["a".into(), String::new()],
            Validity::from_valid_iter(2, [true, false])
        )
        .fillna(0.0)
        .is_err());
    }

    #[test]
    fn set_scalar_at_preserves_validity() {
        // a scalar write keeps every other row's NA (regression: it used to return
        // a dense column and silently turn pre-existing NA into 0 / false)
        let c = na_i64(&[1, 0, 3], &[true, false, true]); // 1, NA, 3
        let r = c.set_scalar_at(&[0], SetVal::Num(9.0)).unwrap();
        assert_eq!(r.dtype(), DType::I64);
        assert!(r.is_valid(0) && !r.is_valid(1) && r.is_valid(2)); // 9, NA, 3
        // writing NaN marks the position NA and keeps int (no f64 upcast)
        let r2 = c.set_scalar_at(&[2], SetVal::Num(f64::NAN)).unwrap();
        assert_eq!(r2.dtype(), DType::I64);
        assert!(r2.is_valid(0) && !r2.is_valid(1) && !r2.is_valid(2));
        // bool keeps its validity too
        let b = Column::bool_with(vec![true, false, false], Validity::from_valid_iter(3, [true, false, true]));
        let rb = b.set_scalar_at(&[0], SetVal::Bool(false)).unwrap();
        assert!(rb.dtype() == DType::Bool && rb.is_valid(0) && !rb.is_valid(1) && rb.is_valid(2));
        // a dense column stays dense (no validity allocated) on a real write
        assert_eq!(Column::i64(vec![1, 2, 3]).set_scalar_at(&[0], SetVal::Num(9.0)).unwrap().null_count(), 0);
        // a bool fill into an int column converts (pandas), keeping validity
        let rib = c.set_scalar_at(&[1], SetVal::Bool(true)).unwrap();
        assert!(rib.dtype() == DType::I64 && rib.is_valid(1));
    }

    #[test]
    fn na_fill_dir() {
        let c = na_i64(&[0, 2, 0, 0, 5], &[false, true, false, false, true]); // NA,2,NA,NA,5
        assert_na(&c.fill_dir(true), &[f64::NAN, 2.0, 2.0, 2.0, 5.0]); // ffill: leading NA stays
        assert_na(&c.fill_dir(false), &[2.0, 2.0, 5.0, 5.0, 5.0]); // bfill: trailing filled
        let i32c = Column::i32_with(vec![7, 0, 0], Validity::from_valid_iter(3, [true, false, false]));
        assert_na(&i32c.fill_dir(true), &[7.0, 7.0, 7.0]);
        let bc = Column::bool_with(vec![true, false, false], Validity::from_valid_iter(3, [true, false, false]));
        assert_na(&bc.fill_dir(true), &[1.0, 1.0, 1.0]);
        assert_na(&Column::f64(vec![1.0, f64::NAN, 3.0]).fill_dir(true), &[1.0, 1.0, 3.0]);
        assert_na(&Column::f32(vec![1.0, f32::NAN]).fill_dir(true), &[1.0, 1.0]);
        assert_na(&Column::datetime(vec![i64::MIN, 20]).fill_dir(false), &[20.0, 20.0]);
        assert_eq!(Column::i64(vec![1, 2]).fill_dir(true), Column::i64(vec![1, 2])); // dense clone
        assert_eq!(Column::str(vec!["a".into()]).fill_dir(true), Column::str(vec!["a".into()]));
        // str + NA carries values directionally like every other dtype (regression:
        // the Str arm used to `unreachable!()` and panic on a missing cell).
        let sc = Column::str_with(
            vec!["x".into(), String::new(), String::new(), "z".into()],
            Validity::from_valid_iter(4, [true, false, false, true]),
        ); // "x", NA, NA, "z"
        assert_eq!(
            sc.fill_dir(true), // ffill -> x, x, x, z (no holes left -> dense)
            Column::str(vec!["x".into(), "x".into(), "x".into(), "z".into()])
        );
        assert_eq!(
            sc.fill_dir(false), // bfill -> x, z, z, z
            Column::str(vec!["x".into(), "z".into(), "z".into(), "z".into()])
        );
        // a leading gap stays NA on ffill (nothing to carry in)
        let ff = Column::str_with(
            vec![String::new(), "a".into()],
            Validity::from_valid_iter(2, [false, true]),
        )
        .fill_dir(true);
        assert!(!ff.is_valid(0) && ff.is_valid(1)); // NA, "a"
        // a fully-missing str column ffills to itself (all still NA)
        let allna = Column::str_with(
            vec![String::new(), String::new()],
            Validity::from_valid_iter(2, [false, false]),
        );
        assert_eq!(allna.fill_dir(true).null_count(), 2);
    }

    #[test]
    fn append_na_pads_dtype_preserving() {
        // a plain column padded on append keeps its dtype and marks the new rows NA
        let mut i = Column::i64(vec![1, 2]);
        i.append_na(2);
        assert_eq!(i.dtype(), DType::I64); // no upcast to float
        assert!(i.is_valid(0) && i.is_valid(1) && !i.is_valid(2) && !i.is_valid(3));
        let mut i32c = Column::i32(vec![7]);
        i32c.append_na(1);
        assert!(i32c.is_valid(0) && !i32c.is_valid(1) && i32c.dtype() == DType::I32);
        let mut b = Column::bool(vec![true]);
        b.append_na(1);
        assert!(b.is_valid(0) && !b.is_valid(1) && b.dtype() == DType::Bool);
        let mut s = Column::str(vec!["a".into()]);
        s.append_na(1);
        assert!(s.is_valid(0) && !s.is_valid(1) && s.dtype() == DType::Utf8);
        let mut d = Column::datetime(vec![100]);
        d.append_na(1);
        assert!(d.is_valid(0) && !d.is_valid(1)); // NaT sentinel
        let mut f = Column::f64(vec![1.0]);
        f.append_na(1);
        assert!(f.is_valid(0) && !f.is_valid(1)); // NaN in-band
        let mut f32c = Column::f32(vec![1.0]);
        f32c.append_na(1);
        assert!(f32c.is_valid(0) && !f32c.is_valid(1)); // f32 NaN in-band
        // an existing hole is preserved, the appended rows are added as NA
        let mut i2 = Column::i64_with(vec![1, 0], Validity::from_valid_iter(2, [true, false]));
        i2.append_na(1);
        assert!(i2.is_valid(0) && !i2.is_valid(1) && !i2.is_valid(2) && i2.null_count() == 2);
    }

    #[test]
    fn na_logical_kleene_and_not() {
        let b = Column::bool_with(vec![true, false, false], Validity::from_valid_iter(3, [true, true, false])); // T,F,NA
        let t = Column::bool(vec![true, true, true]);
        let f = Column::bool(vec![false, false, false]);
        // Kleene AND: NA & True = NA, NA & False = False
        assert_na(&b.logical(&t, BoolOp::And), &[1.0, 0.0, f64::NAN]);
        assert_na(&b.logical(&f, BoolOp::And), &[0.0, 0.0, 0.0]);
        // Kleene OR: NA | True = True, NA | False = NA
        assert_na(&b.logical(&t, BoolOp::Or), &[1.0, 1.0, 1.0]);
        assert_na(&b.logical(&f, BoolOp::Or), &[1.0, 0.0, f64::NAN]);
        // XOR: missing if either is missing
        assert_na(&b.logical(&t, BoolOp::Xor), &[0.0, 1.0, f64::NAN]);
        // NOT propagates NA
        assert_na(&b.not(), &[0.0, 1.0, f64::NAN]);
        // a non-bool operand reads as x != 0 (present), so it never injects NA
        let i = Column::i64(vec![1, 0, 5]);
        assert_na(&i.not(), &[0.0, 1.0, 0.0]);
        assert_na(&b.logical(&i, BoolOp::And), &[1.0, 0.0, f64::NAN]);
    }

    #[test]
    fn na_cast_int_bool_carries_validity() {
        let i = na_i64(&[1, 0, 3], &[true, false, true]); // 1, NA, 3
        assert_eq!(i.cast(DType::I32).unwrap().dtype(), DType::I32);
        assert_na(&i.cast(DType::I32).unwrap(), &[1.0, f64::NAN, 3.0]);
        assert_na(&i.cast(DType::Bool).unwrap(), &[1.0, f64::NAN, 1.0]);
        assert_na(&i.cast(DType::F64).unwrap(), &[1.0, f64::NAN, 3.0]); // int+NA -> float (NaN)
        let i32c = Column::i32_with(vec![1, 0, 5], Validity::from_valid_iter(3, [true, false, true]));
        assert_eq!(i32c.cast(DType::I64).unwrap().dtype(), DType::I64);
        assert_na(&i32c.cast(DType::I64).unwrap(), &[1.0, f64::NAN, 5.0]);
        assert_na(&i32c.cast(DType::Bool).unwrap(), &[1.0, f64::NAN, 1.0]);
        let b = Column::bool_with(vec![true, false, false], Validity::from_valid_iter(3, [true, false, true]));
        assert_na(&b.cast(DType::I64).unwrap(), &[1.0, f64::NAN, 0.0]);
        assert_na(&b.cast(DType::I32).unwrap(), &[1.0, f64::NAN, 0.0]);
        // a present out-of-range i64 -> i32 errors (a missing one never does)
        assert!(Column::i64(vec![3_000_000_000]).cast(DType::I32).is_err());
    }

    #[test]
    fn na_str_carries_validity() {
        let s = Column::str_with(
            vec!["a".into(), String::new(), "c".into()],
            Validity::from_valid_iter(3, [true, false, true]),
        ); // a, NA, c
        assert!(s.is_valid(0) && !s.is_valid(1) && s.is_valid(2) && s.null_count() == 1);
        // shift keeps str with an NA gap: [NA, a, NA] (the trailing NA was already missing)
        let sh = s.shift(1);
        assert_eq!(sh.dtype(), DType::Utf8);
        assert!(!sh.is_valid(0) && sh.is_valid(1) && !sh.is_valid(2));
        assert_eq!(sh.as_str().unwrap()[1], "a");
        // slice / take carry validity
        assert!(!s.slice(1, 3).is_valid(0) && s.slice(1, 3).is_valid(1));
        assert!(!s.take(&[1, 0]).is_valid(0) && s.take(&[1, 0]).is_valid(1));
        // append concatenates validity
        let mut a = Column::str_with(vec!["x".into(), String::new()], Validity::from_valid_iter(2, [true, false]));
        a.append(&Column::str(vec!["y".into()])).unwrap();
        assert!(a.len() == 3 && a.is_valid(0) && !a.is_valid(1) && a.is_valid(2));
        // int + NA -> str carries the missing cell
        let i = na_i64(&[1, 0, 3], &[true, false, true]).cast(DType::Utf8).unwrap();
        assert_eq!(i.dtype(), DType::Utf8);
        assert!(!i.is_valid(1) && i.null_count() == 1);
    }
}
