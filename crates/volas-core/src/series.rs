//! Series: a single named column plus the (shared) index it was drawn from.

use std::sync::Arc;

use crate::column::Column;
use crate::dtype::DType;
use crate::error::{Result, VolasError};
use crate::index::Index;

/// A one-dimensional, named, indexed column with opaque structure.
///
/// [`Series::new`] checks that values and labels have equal lengths. Parts are
/// available through read-only accessors; external callers cannot replace them.
#[derive(Clone, Debug)]
pub struct Series {
    name: Option<String>,
    data: Column,
    index: Arc<Index>,
}

impl Series {
    /// Build a series from equally sized data and a checked index.
    #[inline]
    pub fn new(name: Option<String>, data: Column, index: Arc<Index>) -> Result<Self> {
        let data_len = data.len();
        let index_len = index.len();
        if data_len != index_len {
            return Err(shape_error(data_len, index_len));
        }
        Ok(Self { name, data, index })
    }

    /// Assemble parts whose equal length is already owned by this crate's frame.
    #[inline]
    pub(crate) fn from_frame_parts(name: Option<String>, data: Column, index: Arc<Index>) -> Self {
        Self { name, data, index }
    }

    /// Optional column name.
    pub fn name(&self) -> Option<&str> { self.name.as_deref() }

    /// Borrow the column values.
    pub fn data(&self) -> &Column { &self.data }

    /// Borrow the shared checked row index.
    pub fn index(&self) -> &Arc<Index> { &self.index }

    /// Consume the series and take its column values.
    pub fn into_data(self) -> Column { self.data }

    /// Return a renamed series, preserving its checked values and labels.
    pub fn with_name(&self, name: Option<String>) -> Self {
        Self { name, data: self.data.clone(), index: Arc::clone(&self.index) }
    }

    /// Build a float series with an implicit range index.
    pub fn from_f64(name: Option<String>, values: Vec<f64>) -> Result<Self> {
        Self::from_column(name, Column::f64(values))
    }

    /// Admit a column with a fresh checked positional index. Its length supplies
    /// both parts of the shape. Valid physical column storage also proves the
    /// positional labels fit i64, so neither fact needs revalidation.
    #[inline]
    pub fn from_column(name: Option<String>, data: Column) -> Result<Self> {
        let index = Arc::new(Index::for_column(&data));
        Ok(Self { name, data, index })
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

    /// Write values without changing the series length or labels.
    ///
    /// A single value broadcasts; otherwise values must cover the selected rows.
    /// Invalid positions or a lossy conversion leave the series unchanged.
    pub fn scatter(&mut self, positions: &[usize], values: &Column) -> Result<()> {
        self.data.scatter(positions, values)
    }

    /// Rows where `mask` is `true`, as a new Series (pandas `s[bool_mask]`).
    /// Surviving index labels are preserved (a `RangeIndex` becomes `Int64`).
    pub fn filter_mask(&self, mask: &[bool]) -> Result<Series> {
        if mask.len() != self.len() {
            return Err(VolasError::Shape(format!(
                "boolean mask length {} != series length {}",
                mask.len(),
                self.len()
            )));
        }
        let idx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();
        let index = Arc::new(self.index.take(&idx)?);
        Series::new(self.name.clone(), self.data.take(&idx), index)
    }
}

/// Formatting an invalid shape must not enlarge the successful constructor path.
#[cold]
#[inline(never)]
fn shape_error(data_len: usize, index_len: usize) -> VolasError {
    VolasError::Shape(format!("Series data length {data_len} != index length {index_len}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_has_range_index_and_query_methods() {
        let s = Series::from_f64(Some("x".into()), vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert_eq!(s.dtype(), DType::F64);
        assert_eq!(s.name.as_deref(), Some("x"));
        assert_eq!(s.index.len(), 3);
    }

    #[test]
    fn new_preserves_parts_and_empty_is_empty() {
        let s = Series::new(None, Column::i64(vec![7, 8]), Arc::new(Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap())).unwrap();
        assert_eq!(s.dtype(), DType::I64);
        assert_eq!(s.len(), 2);
        assert!(s.name().is_none());
        assert_eq!(s.data().as_i64(), Some(&[7, 8][..]));
        assert_eq!(s.index().len(), 2);
        let renamed = s.with_name(Some("next".into()));
        assert_eq!(renamed.name(), Some("next"));
        assert_eq!(renamed.into_data(), s.data);

        let empty = Series::from_f64(None, vec![]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn checked_constructor_rejects_mismatched_parts() {
        let result = Series::new(None, Column::i64(vec![1]), Arc::new(Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap()));
        assert!(matches!(result, Err(VolasError::Shape(message)) if message == "Series data length 1 != index length 2"));
    }

    #[test]
    fn scatter_preserves_structure_and_rejects_invalid_writes() {
        let mut series = Series::from_f64(Some("v".into()), vec![1.0, 2.0]).unwrap();
        let index = Arc::clone(series.index());
        let values_pointer = series.data().as_f64().unwrap().as_ptr();
        series.scatter(&[0], &Column::f64(vec![3.0])).unwrap();
        assert_eq!(series.data().as_f64().unwrap().as_ptr(), values_pointer);
        assert_eq!(series.data().to_f64_vec(), vec![3.0, 2.0]);
        assert!(Arc::ptr_eq(&index, series.index()));
        assert!(series.scatter(&[2], &Column::f64(vec![9.0])).is_err());
        assert!(series.scatter(&[0, 1], &Column::f64(vec![])).is_err());
        assert!(series.scatter(&[0], &Column::str(vec!["bad".into()])).is_err());
        assert_eq!(series.data().to_f64_vec(), vec![3.0, 2.0]);
    }

    #[test]
    fn filter_mask_keeps_true_rows_and_labels() {
        let s = Series::new(
            Some("v".into()),
            Column::f64(vec![10.0, 20.0, 30.0]),
            Arc::new(Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap()),
        ).unwrap();
        let sub = s.filter_mask(&[false, true, true]).unwrap();
        assert_eq!(sub.data.to_f64_vec(), vec![20.0, 30.0]);
        // RangeIndex -> the surviving original labels (1, 2)
        assert_eq!(sub.index.label_at(0), Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap().label_at(1));
        assert!(s.filter_mask(&[true, false]).is_err()); // length mismatch
    }
}
