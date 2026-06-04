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
        Series::new(name, Column::f64(values), Arc::new(Index::Range(n)))
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
