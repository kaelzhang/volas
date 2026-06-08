//! Series: a single named column plus the (shared) index it was drawn from.

use std::sync::Arc;

use crate::column::Column;
use crate::dtype::DType;
use crate::index::Index;

/// A one-dimensional, named, indexed column.
///
/// The `data.len() == index.len()` invariant is enforced at construction (see
/// [`Series::new`]); always build through `new` / `from_f64` rather than the
/// fields. (Full field privatization with accessors is a documented follow-up —
/// the read sites are pervasive and the construction check covers the realistic
/// way the invariant could break.)
#[derive(Clone, Debug)]
pub struct Series {
    /// Optional column name.
    pub name: Option<String>,
    /// The column values.
    pub data: Column,
    /// The shared row index.
    pub index: Arc<Index>,
}

impl Series {
    /// Build a series from its parts. The data length must equal the index
    /// length (checked in debug builds).
    pub fn new(name: Option<String>, data: Column, index: Arc<Index>) -> Self {
        debug_assert_eq!(
            data.len(),
            index.len(),
            "Series data length != index length"
        );
        Series { name, data, index }
    }

    /// Build a float series with an implicit range index.
    pub fn from_f64(name: Option<String>, values: Vec<f64>) -> Self {
        let n = values.len();
        Series::new(name, Column::f64(values), Arc::new(Index::range(n)))
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the series is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The series dtype.
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_has_range_index_and_query_methods() {
        let s = Series::from_f64(Some("x".into()), vec![1.0, 2.0, 3.0]);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert_eq!(s.dtype(), DType::F64);
        assert_eq!(s.name.as_deref(), Some("x"));
        assert_eq!(s.index.len(), 3);
    }

    #[test]
    fn new_preserves_parts_and_empty_is_empty() {
        let s = Series::new(None, Column::i64(vec![7, 8]), Arc::new(Index::range(2)));
        assert_eq!(s.dtype(), DType::I64);
        assert_eq!(s.len(), 2);
        assert!(s.name.is_none());

        let empty = Series::from_f64(None, vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }
}
