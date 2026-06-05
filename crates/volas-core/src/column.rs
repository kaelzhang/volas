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

/// A typed, contiguous column of values. The buffer is `Arc`-shared (cheap clone)
/// and mutated copy-on-write.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    /// 64-bit floats; `NaN` denotes missing.
    F64(Arc<Vec<f64>>),
    /// Booleans (comparison / signal results).
    Bool(Arc<Vec<bool>>),
    /// 64-bit signed integers.
    I64(Arc<Vec<i64>>),
    /// UTF-8 strings.
    Str(Arc<Vec<String>>),
    /// Datetimes as i64 nanoseconds since the Unix epoch (UTC-naive).
    Datetime(Arc<Vec<i64>>),
}

impl Column {
    /// Build an `F64` column.
    pub fn f64(v: Vec<f64>) -> Column {
        Column::F64(Arc::new(v))
    }
    /// Build a `Bool` column.
    pub fn bool(v: Vec<bool>) -> Column {
        Column::Bool(Arc::new(v))
    }
    /// Build an `I64` column.
    pub fn i64(v: Vec<i64>) -> Column {
        Column::I64(Arc::new(v))
    }
    /// Build a `Str` column.
    pub fn str(v: Vec<String>) -> Column {
        Column::Str(Arc::new(v))
    }
    /// Build a `Datetime` column (epoch nanoseconds).
    pub fn datetime(v: Vec<i64>) -> Column {
        Column::Datetime(Arc::new(v))
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Column::F64(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::I64(v) => v.len(),
            Column::Str(v) => v.len(),
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
            Column::Bool(_) => DType::Bool,
            Column::I64(_) => DType::I64,
            Column::Str(_) => DType::Utf8,
            Column::Datetime(_) => DType::Datetime,
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
        if let Column::Bool(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `i64` slice, if this is an `I64` column.
    pub fn as_i64(&self) -> Option<&[i64]> {
        if let Column::I64(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Borrow the underlying `String` slice, if this is a `Str` column.
    pub fn as_str(&self) -> Option<&[String]> {
        if let Column::Str(v) = self {
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
            Column::Bool(v) => v.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
            Column::I64(v) => v.iter().map(|&i| i as f64).collect(),
            Column::Str(v) => vec![f64::NAN; v.len()],
            Column::Datetime(v) => v.iter().map(|&i| i as f64).collect(),
        }
    }

    /// Value at position `i` coerced to `f64` (for NumPy 2-D export).
    pub fn get_f64(&self, i: usize) -> f64 {
        match self {
            Column::F64(v) => v[i],
            Column::Bool(v) => {
                if v[i] {
                    1.0
                } else {
                    0.0
                }
            }
            Column::I64(v) => v[i] as f64,
            Column::Str(_) => f64::NAN,
            Column::Datetime(v) => v[i] as f64,
        }
    }

    /// A contiguous `[start, end)` slice (a fresh buffer).
    pub fn slice(&self, start: usize, end: usize) -> Column {
        match self {
            Column::F64(v) => Column::f64(v[start..end].to_vec()),
            Column::Bool(v) => Column::bool(v[start..end].to_vec()),
            Column::I64(v) => Column::i64(v[start..end].to_vec()),
            Column::Str(v) => Column::str(v[start..end].to_vec()),
            Column::Datetime(v) => Column::datetime(v[start..end].to_vec()),
        }
    }

    /// Gather the given positions into a new column (fancy indexing).
    pub fn take(&self, idx: &[usize]) -> Column {
        match self {
            Column::F64(v) => Column::f64(idx.iter().map(|&i| v[i]).collect()),
            Column::Bool(v) => Column::bool(idx.iter().map(|&i| v[i]).collect()),
            Column::I64(v) => Column::i64(idx.iter().map(|&i| v[i]).collect()),
            Column::Str(v) => Column::str(idx.iter().map(|&i| v[i].clone()).collect()),
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
            (Column::Bool(a), Column::Bool(b)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::I64(a), Column::I64(b)) => {
                Arc::make_mut(a).extend_from_slice(b);
                Ok(())
            }
            (Column::Str(a), Column::Str(b)) => {
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
            Column::Str(v) => {
                let mut out = Vec::with_capacity(v.len());
                for s in v.iter() {
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
    /// `unit` (`"s"` / `"ms"` / `"us"` / `"ns"`). The robust ingestion path for
    /// exchange APIs that return numeric timestamps.
    pub fn epoch_to_datetime(&self, unit: &str) -> Result<Column> {
        let to_ns = |x: i64| {
            datetime::epoch_to_ns(x, unit)
                .ok_or_else(|| VolasError::Value(format!("invalid epoch unit {unit:?} or overflow")))
        };
        match self {
            Column::I64(v) => v.iter().map(|&x| to_ns(x)).collect::<Result<Vec<_>>>().map(Column::datetime),
            Column::F64(v) => v
                .iter()
                .map(|&x| to_ns(x as i64))
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
        match to {
            DType::F64 => Ok(Column::f64(self.to_f64_vec())),
            DType::I64 => match self {
                Column::F64(v) => Ok(Column::i64(v.iter().map(|&x| x as i64).collect())),
                Column::Bool(v) => Ok(Column::i64(v.iter().map(|&b| b as i64).collect())),
                Column::Datetime(v) => Ok(Column::i64(v.to_vec())),
                other => Err(VolasError::DType(format!(
                    "cannot cast a {} column to int64",
                    other.dtype()
                ))),
            },
            DType::Bool => match self {
                Column::F64(v) => Ok(Column::bool(v.iter().map(|&x| x != 0.0).collect())),
                Column::I64(v) => Ok(Column::bool(v.iter().map(|&x| x != 0).collect())),
                other => Err(VolasError::DType(format!(
                    "cannot cast a {} column to bool",
                    other.dtype()
                ))),
            },
            DType::Utf8 => Ok(Column::str(self.to_string_vec())),
            DType::Datetime => self.to_datetime(),
        }
    }

    /// Render each value as a `String` (for `astype(str)`).
    fn to_string_vec(&self) -> Vec<String> {
        match self {
            Column::Str(v) => v.to_vec(),
            Column::F64(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::I64(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::Bool(v) => v
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
            _ => self == other,
        }
    }
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
        assert!(Column::str(vec!["not-a-date".into()]).to_datetime().is_err());
        assert!(Column::i64(vec![1, 2]).to_datetime().is_err());
    }

    #[test]
    fn cast_between_dtypes_and_errors() {
        // no-op when already the target dtype
        let f = Column::f64(vec![1.0, 2.0]);
        assert_eq!(f.cast(DType::F64).unwrap(), f);

        // -> F64 (incl. the Str -> NaN arm of to_f64_vec)
        assert_eq!(Column::i64(vec![3]).cast(DType::F64).unwrap(), Column::f64(vec![3.0]));
        assert_eq!(
            Column::bool(vec![true, false]).cast(DType::F64).unwrap(),
            Column::f64(vec![1.0, 0.0])
        );
        let from_str = Column::str(vec!["a".into(), "b".into()]).cast(DType::F64).unwrap();
        assert_eq!(from_str.dtype(), DType::F64);
        assert!(from_str.to_f64_vec().iter().all(|x| x.is_nan()));

        // -> I64 (F64 / Bool / Datetime; Str errors)
        assert_eq!(Column::f64(vec![2.9]).cast(DType::I64).unwrap(), Column::i64(vec![2]));
        assert_eq!(Column::bool(vec![true]).cast(DType::I64).unwrap(), Column::i64(vec![1]));
        assert_eq!(Column::datetime(vec![5]).cast(DType::I64).unwrap(), Column::i64(vec![5]));
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
        assert_eq!(Column::f64(vec![1.5]).cast(DType::Utf8).unwrap(), Column::str(vec!["1.5".into()]));
        assert_eq!(Column::i64(vec![7]).cast(DType::Utf8).unwrap(), Column::str(vec!["7".into()]));
        assert_eq!(
            Column::bool(vec![true, false]).cast(DType::Utf8).unwrap(),
            Column::str(vec!["True".into(), "False".into()])
        );
        let dt_str = Column::datetime(vec![0]).cast(DType::Utf8).unwrap();
        assert_eq!(dt_str.dtype(), DType::Utf8);
        assert_eq!(dt_str.len(), 1);

        // -> Datetime (delegates to to_datetime)
        assert_eq!(
            Column::str(vec!["2020-01-01".into()]).cast(DType::Datetime).unwrap().dtype(),
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
        assert_eq!(Column::bool(vec![true, false, true]).slice(1, 3), Column::bool(vec![false, true]));
        assert_eq!(Column::i64(vec![1, 2, 3]).take(&[2, 0]), Column::i64(vec![3, 1]));
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
        // to_string_vec renders each supported dtype.
        assert_eq!(Column::str(vec!["a".into()]).to_string_vec(), vec!["a".to_string()]);
        assert_eq!(Column::f64(vec![1.5]).to_string_vec(), vec!["1.5".to_string()]);
        assert_eq!(Column::i64(vec![3]).to_string_vec(), vec!["3".to_string()]);
    }
}
