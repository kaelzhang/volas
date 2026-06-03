//! Index: the row labels shared by a frame and the series drawn from it.

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
}
