//! The Arrow **C-Stream interface** for a frame of columns: produce / consume an
//! `FFI_ArrowArrayStream` (one `RecordBatch`), backing the `__arrow_c_stream__`
//! protocol so a whole DataFrame crosses the boundary in one handshake.

use std::ffi::c_void;
use std::sync::Arc;

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{ArrowError, Field, Schema};
use arrow_select::concat::concat_batches;
use volas_core::Column;

use crate::{column_from_arrow, column_to_arrow};

/// Named columns as a one-batch Arrow C-Stream (the DataFrame export). Every column's
/// data buffer is shared with Arrow (no copy); the stream yields a single `RecordBatch`.
pub fn columns_to_c_stream(
    names: &[String],
    cols: &[Column],
) -> Result<FFI_ArrowArrayStream, ArrowError> {
    let arrays: Vec<ArrayRef> = cols.iter().map(column_to_arrow).collect();
    let fields: Vec<Field> = names
        .iter()
        .zip(&arrays)
        .map(|(n, a)| Field::new(n, a.data_type().clone(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Ok(FFI_ArrowArrayStream::new(Box::new(reader)))
}

/// Consume an Arrow C-Stream capsule payload into `(names, columns)` (the DataFrame
/// import). All batches are concatenated, so a multi-chunk producer (a pyarrow `Table`)
/// lands as one column each.
///
/// # Safety
/// `stream` must be the live `"arrow_array_stream"` capsule payload, not yet consumed.
pub unsafe fn columns_from_c_stream(
    stream: *mut c_void,
) -> Result<(Vec<String>, Vec<Column>), ArrowError> {
    let stream = std::ptr::replace(stream as *mut FFI_ArrowArrayStream, FFI_ArrowArrayStream::empty());
    let reader = ArrowArrayStreamReader::try_new(stream)?;
    let schema = reader.schema();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let batches = reader.collect::<Result<Vec<RecordBatch>, ArrowError>>()?;
    let batch = concat_batches(&schema, &batches)?;
    let cols = batch
        .columns()
        .iter()
        .map(column_from_arrow)
        .collect::<Result<Vec<Column>, ArrowError>>()?;
    Ok((names, cols))
}
