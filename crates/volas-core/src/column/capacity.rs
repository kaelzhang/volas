//! Storage growth belongs to Column. Frame commits only after every column has
//! admitted value, offset, payload and validity capacity, including missing pads.
use super::*;

#[cold]
pub(crate) fn capacity_error() -> VolasError { VolasError::Overflow("column growth capacity overflow".into()) }
#[cold]
pub(crate) fn allocation_error(error: std::collections::TryReserveError) -> VolasError {
    VolasError::Overflow(format!("cannot reserve column growth: {error}"))
}
// Mask work belongs only to nullable storage; float/datetime growth never
// dispatches the source's validity or reads its cell presence.
#[inline]
fn missing(source: Option<&Column>, cell: Option<usize>) -> bool {
    source.is_none_or(|s| cell.map_or_else(|| s.validity().is_some_and(Validity::has_nulls), |row| !s.is_valid(row)))
}

fn reserve_nullable<T: crate::buffer::Element + Clone>(buffer: &mut Buffer<T>, validity: &mut Validity, added: usize, source: Option<&Column>, cell: Option<usize>) -> Result<()> {
    buffer.try_reserve(added)?;
    validity.prepare_growth(buffer.len(), added, missing(source, cell))
}

impl Column {
    /// Prepare COW and allocation before index/replacement/cache publication.
    /// Values and input facts stay unchanged, including on a reservation error.
    pub(crate) fn prepare_growth(&mut self, added: usize, source: Option<&Column>, cell: Option<usize>, replaces: bool) -> Result<()> {
        match &mut self.data {
            ColumnData::F64(b) => b.try_reserve(added)?,
            ColumnData::F32(b) => b.try_reserve(added)?,
            ColumnData::Datetime(b) => b.try_reserve(added)?,
            ColumnData::Bool(b, v) => reserve_nullable(b, v, added, source, cell)?,
            ColumnData::I64(b, v) => reserve_nullable(b, v, added, source, cell)?,
            ColumnData::I32(b, v) => reserve_nullable(b, v, added, source, cell)?,
            ColumnData::Str(b, v) => {
                // A replacement can change payload size as well as its suffix.
                let bytes = match source {
                    Some(Column { data: ColumnData::Str(s, _) }) => cell.map_or_else(|| s.buffers().1.len(), |row| s.get(row).len()),
                    _ => 0,
                };
                // The frame's admitted replacement removes its previous tail.
                // Reserve net payload growth, not bytes that will be truncated.
                let removed = if replaces { b.get(b.len() - 1).len() } else { 0 };
                b.prepare_growth(added, bytes.saturating_sub(removed))?;
                v.prepare_growth(b.len(), added, missing(source, cell))?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish_growth(&mut self) {
        match &mut self.data {
            ColumnData::Bool(_, v) | ColumnData::I64(_, v) | ColumnData::I32(_, v) | ColumnData::Str(_, v) => v.finish_growth(),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn impossible_shared_buffer_and_bitmap_growth_return_errors_without_writes() {
        let mut buffer = Buffer::from_vec(vec![1i64]);
        let before = buffer.clone();
        assert!(matches!(buffer.try_reserve(usize::MAX), Err(VolasError::Overflow(_))));
        assert_eq!(buffer, before);
        let mut validity = Validity::dense();
        assert!(matches!(validity.prepare_growth(usize::MAX, 1, true), Err(VolasError::Overflow(_))));
        assert_eq!(validity, Validity::dense());
    }
}
