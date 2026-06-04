//! Index: the row labels shared by a frame and the series drawn from it.

use crate::column::Column;
use crate::error::{Result, VolasError};

/// Row labels. Defaults to an implicit `0..n` range; a `Datetime` index is the
/// common OHLCV case (i64 nanoseconds since the Unix epoch).
#[derive(Clone, Debug, PartialEq)]
pub enum Index {
    /// Implicit `0..n` integer labels.
    Range(usize),
    /// Explicit integer labels.
    Int64(Vec<i64>),
    /// Datetime labels as i64 nanoseconds since the Unix epoch.
    Datetime(Vec<i64>),
}

impl Index {
    /// Build an index from a column (for `set_index`): a `Datetime` column
    /// becomes a `DatetimeIndex`, an `I64` column an `Int64Index`. Other dtypes
    /// are not supported as an index in v1 (string / float indexes are a
    /// documented future refinement).
    pub fn from_column(col: &Column) -> Result<Index> {
        match col {
            Column::Datetime(v) => Ok(Index::Datetime(v.to_vec())),
            Column::I64(v) => Ok(Index::Int64(v.to_vec())),
            other => Err(VolasError::DType(format!(
                "cannot use a {} column as an index (only datetime / int64)",
                other.dtype()
            ))),
        }
    }

    /// Number of labels.
    pub fn len(&self) -> usize {
        match self {
            Index::Range(n) => *n,
            Index::Int64(v) => v.len(),
            Index::Datetime(v) => v.len(),
        }
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Materialize the labels as `i64`.
    pub fn to_i64_labels(&self) -> Vec<i64> {
        match self {
            Index::Range(n) => (0..*n as i64).collect(),
            Index::Int64(v) => v.clone(),
            Index::Datetime(v) => v.clone(),
        }
    }

    /// A `[start, end)` slice.
    pub fn slice(&self, start: usize, end: usize) -> Index {
        match self {
            Index::Range(_) => Index::Range(end.saturating_sub(start)),
            Index::Int64(v) => Index::Int64(v[start..end].to_vec()),
            Index::Datetime(v) => Index::Datetime(v[start..end].to_vec()),
        }
    }

    /// Gather the given positions.
    pub fn take(&self, idx: &[usize]) -> Index {
        match self {
            Index::Range(_) => Index::Int64(idx.iter().map(|&i| i as i64).collect()),
            Index::Int64(v) => Index::Int64(idx.iter().map(|&i| v[i]).collect()),
            Index::Datetime(v) => Index::Datetime(idx.iter().map(|&i| v[i]).collect()),
        }
    }

    /// Concatenate two indexes (extending labels).
    pub fn append(&self, other: &Index) -> Index {
        match (self, other) {
            (Index::Range(a), Index::Range(b)) => Index::Range(a + b),
            (Index::Datetime(a), Index::Datetime(b)) => {
                let mut v = a.clone();
                v.extend_from_slice(b);
                Index::Datetime(v)
            }
            _ => {
                let mut v = self.to_i64_labels();
                v.extend(other.to_i64_labels());
                Index::Int64(v)
            }
        }
    }

    /// Position of the first label exactly equal to `value`.
    pub fn position_of(&self, value: i64) -> Option<usize> {
        match self {
            Index::Range(n) => {
                if value >= 0 && (value as usize) < *n {
                    Some(value as usize)
                } else {
                    None
                }
            }
            Index::Int64(v) | Index::Datetime(v) => v.iter().position(|&x| x == value),
        }
    }

    /// `[start, end)` positions covering the inclusive label range `[lo, hi]`
    /// (ascending labels; pandas `.loc` slice semantics). Either bound may be
    /// `None` for open-ended.
    pub fn label_slice(&self, lo: Option<i64>, hi: Option<i64>) -> (usize, usize) {
        let labels = self.to_i64_labels();
        let start = match lo {
            Some(lo) => labels.iter().position(|&x| x >= lo).unwrap_or(labels.len()),
            None => 0,
        };
        let end = match hi {
            Some(hi) => labels.iter().rposition(|&x| x <= hi).map(|p| p + 1).unwrap_or(0),
            None => labels.len(),
        };
        (start, end.max(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_datetime_and_int_columns() {
        assert_eq!(
            Index::from_column(&Column::datetime(vec![5, 6])).unwrap(),
            Index::Datetime(vec![5, 6])
        );
        assert_eq!(
            Index::from_column(&Column::i64(vec![1, 2])).unwrap(),
            Index::Int64(vec![1, 2])
        );
    }

    #[test]
    fn from_unsupported_column_errors() {
        assert!(Index::from_column(&Column::f64(vec![1.0])).is_err());
        assert!(Index::from_column(&Column::str(vec!["x".into()])).is_err());
    }
}
