//! The Arrow **C-Stream interface** for a frame of columns: produce / consume an
//! `FFI_ArrowArrayStream` (one `RecordBatch`), backing the `__arrow_c_stream__`
//! protocol so a whole DataFrame crosses the boundary in one handshake.

use std::ffi::{c_int, c_void, CStr};
use std::sync::Arc;

use arrow_array::ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, RecordBatchOptions, StructArray};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use arrow_select::concat::concat_batches;
use volas_core::Column;

use crate::ffi::{check_schema, validate_imported};
use crate::{column_from_arrow, column_to_arrow};

/// Named columns as a one-batch Arrow C-Stream (the DataFrame export). Every column's
/// data buffer is shared with Arrow (no copy); the stream yields a single `RecordBatch`.
pub fn columns_to_c_stream(
    names: &[String],
    cols: &[Column],
) -> Result<FFI_ArrowArrayStream, ArrowError> {
    frame_to_c_stream(names, cols, cols.first().map_or(0, Column::len))
}

/// Export a frame with an explicit row count, including zero-column frames.
pub fn frame_to_c_stream(
    names: &[String],
    cols: &[Column],
    height: usize,
) -> Result<FFI_ArrowArrayStream, ArrowError> {
    // Checked here, never inside the stream's `extern "C"` callbacks (a panic there
    // aborts the process): one name per column, none holding a NUL byte (the C
    // interface carries names as NUL-terminated strings).
    if names.len() != cols.len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "{} names for {} columns",
            names.len(),
            cols.len()
        )));
    }
    if let Some(bad) = names.iter().find(|n| n.contains('\0')) {
        return Err(ArrowError::InvalidArgumentError(format!(
            "column name {bad:?} contains a NUL byte, which the Arrow C interface cannot carry"
        )));
    }
    let arrays: Vec<ArrayRef> = cols.iter().map(column_to_arrow).collect();
    let fields: Vec<Field> = names
        .iter()
        .zip(&arrays)
        .map(|(n, a)| Field::new(n, a.data_type().clone(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new_with_options(
        schema.clone(), arrays, &RecordBatchOptions::new().with_row_count(Some(height)),
    )?;
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
    frame_from_c_stream(stream).map(|(names, cols, _)| (names, cols))
}

/// Consume a frame stream, preserving its row count independently of its columns.
///
/// # Safety
/// `stream` must be a live, not-yet-consumed Arrow C-Stream capsule payload.
pub unsafe fn frame_from_c_stream(
    stream: *mut c_void,
) -> Result<(Vec<String>, Vec<Column>, usize), ArrowError> {
    let mut stream =
        std::ptr::replace(stream as *mut FFI_ArrowArrayStream, FFI_ArrowArrayStream::empty());
    // Drive the stream by hand rather than through arrow-rs's reader, which builds each
    // batch from the producer's buffers unchecked: every batch is validated
    // (`validate_imported`) before any of its buffers is read.
    let (Some(get_schema), Some(get_next), Some(_)) =
        (stream.get_schema, stream.get_next, stream.release)
    else {
        return Err(ArrowError::CDataInterface("input stream is already released".into()));
    };
    let mut ffi_schema = FFI_ArrowSchema::empty();
    let rc = get_schema(&mut stream, &mut ffi_schema);
    if rc != 0 {
        return Err(stream_error(&mut stream, rc));
    }
    check_schema(&ffi_schema as *const FFI_ArrowSchema as *const c_void)?;
    let schema = Arc::new(Schema::try_from(&ffi_schema)?);
    let row_type = DataType::Struct(schema.fields().clone());
    let mut batches = Vec::new();
    let mut height = 0usize;
    loop {
        let mut array = FFI_ArrowArray::empty();
        let rc = get_next(&mut stream, &mut array);
        if rc != 0 {
            return Err(stream_error(&mut stream, rc));
        }
        if array.is_released() {
            break; // end of stream
        }
        let data = from_ffi_and_data_type(array, row_type.clone())?;
        if data.is_empty() {
            // A zero-row batch contributes nothing and is never read (a zero-length
            // string slice may carry its parent's offset, which arrow-rs's import
            // pairs with an empty value buffer).
            continue;
        }
        validate_imported(&data)?;
        height = height.checked_add(data.len()).ok_or_else(|| {
            ArrowError::InvalidArgumentError("stream row count overflow".into())
        })?;
        batches.push(RecordBatch::from(StructArray::from(data)));
    }
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let batch = concat_batches(&schema, &batches)?;
    let cols = batch
        .columns()
        .iter()
        .map(column_from_arrow)
        .collect::<Result<Vec<Column>, ArrowError>>()?;
    Ok((names, cols, height))
}

/// The producer's error for a failed stream call (`get_last_error`, else the code).
unsafe fn stream_error(stream: &mut FFI_ArrowArrayStream, rc: c_int) -> ArrowError {
    let msg = stream
        .get_last_error
        .map(|f| f(stream))
        .filter(|p| !p.is_null())
        .map(|p| CStr::from_ptr(p).to_string_lossy().into_owned());
    ArrowError::CDataInterface(msg.unwrap_or_else(|| format!("stream error code {rc}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_column_stream_keeps_rows_in_both_directions() {
        let mut stream = frame_to_c_stream(&[], &[], 3).unwrap();
        // SAFETY: a fresh stream owned by this test.
        let (names, cols, height) = unsafe {
            frame_from_c_stream(&mut stream as *mut _ as *mut c_void)
        }
        .unwrap();
        assert!(names.is_empty() && cols.is_empty());
        assert_eq!(height, 3);
    }

    #[test]
    fn frame_stream_rejects_row_count_mismatching_its_columns() {
        assert!(frame_to_c_stream(&["x".into()], &[Column::i64(vec![1])], 2).is_err());
    }

    #[test]
    fn zero_column_stream_checks_total_row_count_overflow() {
        let schema = Arc::new(Schema::empty());
        let batch = RecordBatch::try_new_with_options(
            schema.clone(), vec![],
            &RecordBatchOptions::new().with_row_count(Some(i64::MAX as usize)),
        )
        .unwrap();
        let reader = RecordBatchIterator::new(
            vec![Ok(batch.clone()), Ok(batch.clone()), Ok(batch)], schema,
        );
        let mut stream = FFI_ArrowArrayStream::new(Box::new(reader));
        // SAFETY: a fresh stream owned by this test, each batch length fits int64.
        let error = unsafe {
            frame_from_c_stream(&mut stream as *mut _ as *mut c_void)
        }
        .unwrap_err();
        assert!(error.to_string().contains("row count overflow"));
    }
}
