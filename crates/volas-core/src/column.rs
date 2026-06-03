//! Column: a single typed, contiguous buffer.
//!
//! v1 stores values directly in a `Vec<T>`; `F64` columns use `NaN` for missing
//! values (warm-up regions, gaps), matching stock-pandas / pandas semantics.

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
}

impl Column {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Column::F64(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::I64(v) => v.len(),
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

    /// Materialize the values as `f64` (`bool` -> 0.0/1.0, `i64` -> as f64).
    /// Used to feed indicator kernels, which operate on `f64`.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Column::F64(v) => v.clone(),
            Column::Bool(v) => v.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
            Column::I64(v) => v.iter().map(|&i| i as f64).collect(),
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
        }
    }

    /// A contiguous `[start, end)` slice (copying).
    pub fn slice(&self, start: usize, end: usize) -> Column {
        match self {
            Column::F64(v) => Column::F64(v[start..end].to_vec()),
            Column::Bool(v) => Column::Bool(v[start..end].to_vec()),
            Column::I64(v) => Column::I64(v[start..end].to_vec()),
        }
    }

    /// Gather the given positions into a new column (fancy indexing).
    pub fn take(&self, idx: &[usize]) -> Column {
        match self {
            Column::F64(v) => Column::F64(idx.iter().map(|&i| v[i]).collect()),
            Column::Bool(v) => Column::Bool(idx.iter().map(|&i| v[i]).collect()),
            Column::I64(v) => Column::I64(idx.iter().map(|&i| v[i]).collect()),
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
            (s, o) => Err(VolasError::DType(format!(
                "cannot append a {} column onto a {} column",
                o.dtype(),
                s.dtype()
            ))),
        }
    }
}
