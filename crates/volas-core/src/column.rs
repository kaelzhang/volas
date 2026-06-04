//! Column: a single typed, contiguous buffer.
//!
//! v1 stores values directly in a `Vec<T>`; `F64` columns use `NaN` for missing
//! values (warm-up regions, gaps), matching stock-pandas / pandas semantics.

use crate::datetime;
use crate::dtype::DType;
use crate::error::{Result, VolasError};

/// A typed, contiguous column of values.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    /// 64-bit floats; `NaN` denotes missing.
    F64(Vec<f64>),
    /// Booleans (comparison / signal results).
    Bool(Vec<bool>),
    /// 64-bit signed integers.
    I64(Vec<i64>),
    /// UTF-8 strings.
    Str(Vec<String>),
    /// Datetimes as i64 nanoseconds since the Unix epoch (UTC-naive).
    Datetime(Vec<i64>),
}

impl Column {
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
            Some(v)
        } else {
            None
        }
    }

    /// Borrow the underlying `bool` slice, if this is a `Bool` column.
    pub fn as_bool(&self) -> Option<&[bool]> {
        if let Column::Bool(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Borrow the underlying `i64` slice, if this is an `I64` column.
    pub fn as_i64(&self) -> Option<&[i64]> {
        if let Column::I64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Borrow the underlying `String` slice, if this is a `Str` column.
    pub fn as_str(&self) -> Option<&[String]> {
        if let Column::Str(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Borrow the underlying epoch-ns slice, if this is a `Datetime` column.
    pub fn as_datetime(&self) -> Option<&[i64]> {
        if let Column::Datetime(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Materialize the values as `f64` (`bool` -> 0.0/1.0, `i64` / `datetime` ->
    /// as f64, `str` -> NaN). Used to feed indicator kernels, which operate on
    /// `f64`.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Column::F64(v) => v.clone(),
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

    /// A contiguous `[start, end)` slice (copying).
    pub fn slice(&self, start: usize, end: usize) -> Column {
        match self {
            Column::F64(v) => Column::F64(v[start..end].to_vec()),
            Column::Bool(v) => Column::Bool(v[start..end].to_vec()),
            Column::I64(v) => Column::I64(v[start..end].to_vec()),
            Column::Str(v) => Column::Str(v[start..end].to_vec()),
            Column::Datetime(v) => Column::Datetime(v[start..end].to_vec()),
        }
    }

    /// Gather the given positions into a new column (fancy indexing).
    pub fn take(&self, idx: &[usize]) -> Column {
        match self {
            Column::F64(v) => Column::F64(idx.iter().map(|&i| v[i]).collect()),
            Column::Bool(v) => Column::Bool(idx.iter().map(|&i| v[i]).collect()),
            Column::I64(v) => Column::I64(idx.iter().map(|&i| v[i]).collect()),
            Column::Str(v) => Column::Str(idx.iter().map(|&i| v[i].clone()).collect()),
            Column::Datetime(v) => Column::Datetime(idx.iter().map(|&i| v[i]).collect()),
        }
    }

    /// Append another column of the same dtype in place.
    pub fn append(&mut self, other: &Column) -> Result<()> {
        match (self, other) {
            (Column::F64(a), Column::F64(b)) => {
                a.extend_from_slice(b);
                Ok(())
            }
            (Column::Bool(a), Column::Bool(b)) => {
                a.extend_from_slice(b);
                Ok(())
            }
            (Column::I64(a), Column::I64(b)) => {
                a.extend_from_slice(b);
                Ok(())
            }
            (Column::Str(a), Column::Str(b)) => {
                a.extend_from_slice(b);
                Ok(())
            }
            (Column::Datetime(a), Column::Datetime(b)) => {
                a.extend_from_slice(b);
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
    /// parsed via [`datetime::parse_ns`]; an already-`Datetime` column is
    /// returned unchanged. Errors on an unparseable cell or an unsupported dtype.
    pub fn to_datetime(&self) -> Result<Column> {
        match self {
            Column::Datetime(_) => Ok(self.clone()),
            Column::Str(v) => {
                let mut out = Vec::with_capacity(v.len());
                for s in v {
                    let ns = datetime::parse_ns(s).ok_or_else(|| {
                        VolasError::Value(format!("could not parse datetime {s:?}"))
                    })?;
                    out.push(ns);
                }
                Ok(Column::Datetime(out))
            }
            other => Err(VolasError::DType(format!(
                "cannot parse a {} column as datetime",
                other.dtype()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datetime_column_basics() {
        let c = Column::Datetime(vec![10, 20, 30]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.dtype(), DType::Datetime);
        assert_eq!(c.as_datetime().unwrap(), &[10, 20, 30]);
        assert_eq!(c.get_f64(1), 20.0);
        assert_eq!(c.to_f64_vec(), vec![10.0, 20.0, 30.0]);
        assert_eq!(c.slice(1, 3), Column::Datetime(vec![20, 30]));
        assert_eq!(c.take(&[2, 0]), Column::Datetime(vec![30, 10]));
    }

    #[test]
    fn datetime_append_same_dtype_only() {
        let mut a = Column::Datetime(vec![1]);
        a.append(&Column::Datetime(vec![2, 3])).unwrap();
        assert_eq!(a, Column::Datetime(vec![1, 2, 3]));
        assert!(a.append(&Column::I64(vec![4])).is_err());
    }

    #[test]
    fn to_datetime_parses_strings() {
        let c = Column::Str(vec!["2020-01-01".into(), "2020-01-02 03:04:05".into()]);
        let dt = c.to_datetime().unwrap();
        assert_eq!(dt.dtype(), DType::Datetime);
        assert_eq!(dt.len(), 2);
        // idempotent on an already-datetime column
        assert_eq!(dt.to_datetime().unwrap(), dt);
    }

    #[test]
    fn to_datetime_errors() {
        assert!(Column::Str(vec!["not-a-date".into()]).to_datetime().is_err());
        assert!(Column::I64(vec![1, 2]).to_datetime().is_err());
    }
}
