//! Typed export dispatch belongs to Column. Consumers receive only the selected
//! buffer, never a second Column descriptor to match or move.
use super::*;

/// A consumer of a column's typed storage. Numeric buffers are transferred on
/// consuming export; validity stays read-only.
pub trait ColumnExport {
    /// The consumer's result, including its own boundary errors.
    type Output;
    /// Float64 storage.
    fn f64(self, values: Buffer<f64>) -> Self::Output;
    /// Float32 storage.
    fn f32(self, values: Buffer<f32>) -> Self::Output;
    /// Integer storage and its checked validity.
    fn i64(self, values: Buffer<i64>, validity: &Validity) -> Self::Output;
    /// Narrow integer storage and its checked validity.
    fn i32(self, values: Buffer<i32>, validity: &Validity) -> Self::Output;
    /// Boolean storage and its checked validity.
    fn bool(self, values: Buffer<bool>, validity: &Validity) -> Self::Output;
    /// String storage and its checked validity.
    fn str(self, values: &StrBuffer, validity: &Validity) -> Self::Output;
    /// Epoch nanosecond storage.
    fn datetime(self, values: Buffer<i64>) -> Self::Output;
}

/// A typed consumer borrowing immutable column storage without temporary ownership.
pub trait ColumnBorrowExport {
    /// The consumer's result, including its own boundary errors.
    type Output;
    /// Float64 storage.
    fn f64(self, values: &Buffer<f64>) -> Self::Output;
    /// Float32 storage.
    fn f32(self, values: &Buffer<f32>) -> Self::Output;
    /// Integer storage and its checked validity.
    fn i64(self, values: &Buffer<i64>, validity: &Validity) -> Self::Output;
    /// Narrow integer storage and its checked validity.
    fn i32(self, values: &Buffer<i32>, validity: &Validity) -> Self::Output;
    /// Boolean storage and its checked validity.
    fn bool(self, values: &Buffer<bool>, validity: &Validity) -> Self::Output;
    /// String storage and its checked validity.
    fn str(self, values: &StrBuffer, validity: &Validity) -> Self::Output;
    /// Epoch nanosecond storage.
    fn datetime(self, values: &Buffer<i64>) -> Self::Output;
}

impl Column {
    /// Consume the column with one type dispatch. A unique numeric buffer moves
    /// directly to the consumer, without copying its values.
    #[inline]
    pub fn into_export<E: ColumnExport>(self, export: E) -> E::Output {
        if let ColumnData::F64(v) = self.data { return export.f64(v); }
        self.into_export_other(export)
    }

    #[cold]
    fn into_export_other<E: ColumnExport>(self, export: E) -> E::Output {
        match self.data {
            ColumnData::F64(_) => unreachable!("float64 exported before the cold dispatch"), // LCOV_EXCL_LINE
            ColumnData::F32(v) => export.f32(v),
            ColumnData::I64(v, valid) => export.i64(v, &valid),
            ColumnData::I32(v, valid) => export.i32(v, &valid),
            ColumnData::Bool(v, valid) => export.bool(v, &valid),
            ColumnData::Str(v, valid) => export.str(&v, &valid),
            ColumnData::Datetime(v) => export.datetime(v),
        }
    }

    /// Borrow the column with one type dispatch. Numeric buffers share their
    /// immutable allocation; a later write uses copy-on-write as usual.
    #[inline]
    pub fn export<E: ColumnBorrowExport>(&self, export: E) -> E::Output {
        match &self.data {
            ColumnData::F64(v) => export.f64(v),
            ColumnData::F32(v) => export.f32(v),
            ColumnData::I64(v, valid) => export.i64(v, valid),
            ColumnData::I32(v, valid) => export.i32(v, valid),
            ColumnData::Bool(v, valid) => export.bool(v, valid),
            ColumnData::Str(v, valid) => export.str(v, valid),
            ColumnData::Datetime(v) => export.datetime(v),
        }
    }
}
