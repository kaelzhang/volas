//! Read-only storage access and checked adoption of interop buffers.

use super::*;

/// A borrowed view of a column's typed values and missing-value mask.
/// Views cannot alter a column or construct a new storage pairing.
#[derive(Clone, Copy, Debug)]
pub enum ColumnView<'a> {
    /// Float64 values; NaN denotes missing.
    F64(&'a Buffer<f64>),
    /// Float32 values; NaN denotes missing.
    F32(&'a Buffer<f32>),
    /// Boolean values and their validity.
    Bool(&'a Buffer<bool>, &'a Validity),
    /// Int64 values and their validity.
    I64(&'a Buffer<i64>, &'a Validity),
    /// Int32 values and their validity.
    I32(&'a Buffer<i32>, &'a Validity),
    /// UTF-8 values and their validity.
    Str(&'a StrBuffer, &'a Validity),
    /// Epoch nanoseconds; NaT denotes missing.
    Datetime(&'a Buffer<i64>),
}

impl Column {
    /// Inspect the physical storage without exposing mutable access.
    #[inline]
    pub fn view(&self) -> ColumnView<'_> {
        match &self.data {
            ColumnData::F64(v) => ColumnView::F64(v),
            ColumnData::F32(v) => ColumnView::F32(v),
            ColumnData::Bool(v, valid) => ColumnView::Bool(v, valid),
            ColumnData::I64(v, valid) => ColumnView::I64(v, valid),
            ColumnData::I32(v, valid) => ColumnView::I32(v, valid),
            ColumnData::Str(v, valid) => ColumnView::Str(v, valid),
            ColumnData::Datetime(v) => ColumnView::Datetime(v),
        }
    }

    /// Adopt a float64 buffer without copying it.
    pub fn from_f64_buffer(values: Buffer<f64>) -> Self {
        Self { data: ColumnData::F64(values) }
    }

    /// Adopt a float32 buffer without copying it.
    pub fn from_f32_buffer(values: Buffer<f32>) -> Self {
        Self { data: ColumnData::F32(values) }
    }

    /// Adopt an int64 buffer and validate its missing-value mask.
    pub fn from_i64_buffer(values: Buffer<i64>, validity: Validity) -> Result<Self> {
        Self { data: ColumnData::I64(values, validity) }.checked_layout()
    }

    /// Adopt an int32 buffer and validate its missing-value mask.
    pub fn from_i32_buffer(values: Buffer<i32>, validity: Validity) -> Result<Self> {
        Self { data: ColumnData::I32(values, validity) }.checked_layout()
    }

    /// Adopt a boolean buffer and validate its missing-value mask.
    pub fn from_bool_buffer(values: Buffer<bool>, validity: Validity) -> Result<Self> {
        Self { data: ColumnData::Bool(values, validity) }.checked_layout()
    }

    /// Adopt a UTF-8 buffer and validate its missing-value mask.
    pub fn from_str_buffer(values: StrBuffer, validity: Validity) -> Result<Self> {
        Self { data: ColumnData::Str(values, validity) }.checked_layout()
    }

    /// Adopt epoch-nanosecond values without copying them.
    pub fn from_datetime_buffer(values: Buffer<i64>) -> Self {
        Self { data: ColumnData::Datetime(values) }
    }

    /// Take ownership of float64 values, converting other dtypes when needed.
    /// A uniquely owned float64 buffer is moved without copying.
    #[inline]
    pub fn into_f64_vec(self) -> Vec<f64> {
        match self.data {
            ColumnData::F64(values) => values.into_vec(),
            _ => self.to_f64_vec(),
        }
    }

    /// Take ownership of int64 values, rejecting dtypes requiring conversion.
    #[inline]
    pub fn into_i64_vec(self) -> Result<Vec<i64>> {
        match self.data {
            ColumnData::I64(values, _) => Ok(values.into_vec()),
            _ => self.as_i64_vec(),
        }
    }

    /// Take ownership of boolean values, rejecting dtypes requiring conversion.
    #[inline]
    pub fn into_bool_vec(self) -> Result<Vec<bool>> {
        match self.data {
            ColumnData::Bool(values, _) => Ok(values.into_vec()),
            _ => self.as_bool_vec(),
        }
    }
}
