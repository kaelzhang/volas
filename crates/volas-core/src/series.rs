//! Series: a single named column plus the (shared) index it was drawn from.

use std::sync::Arc;

use crate::column::Column;
use crate::dtype::DType;
use crate::index::Index;

/// A one-dimensional, named, indexed column.
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
    /// Build a series from its parts.
    pub fn new(name: Option<String>, data: Column, index: Arc<Index>) -> Self {
        Series { name, data, index }
    }

    /// Build a float series with an implicit range index.
    pub fn from_f64(name: Option<String>, values: Vec<f64>) -> Self {
        let n = values.len();
        Series {
            name,
            data: Column::F64(values),
            index: Arc::new(Index::Range(n)),
        }
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
