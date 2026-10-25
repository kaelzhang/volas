//! Building a `Column`: the dense constructors, the missing-aware ones (checked
//! against the layout invariant, or derived from optional cells so it always holds),
//! and the one boundary for values that arrive with a separate missing mask.

use super::*;

impl Column {
    /// Build an `F64` column.
    pub fn f64(v: Vec<f64>) -> Column {
        Column { data: ColumnData::F64(Buffer::from_vec(v)) }
    }

    /// Build an `F32` column.
    pub fn f32(v: Vec<f32>) -> Column {
        Column { data: ColumnData::F32(Buffer::from_vec(v)) }
    }

    /// Build an `I32` column (all values present).
    pub fn i32(v: Vec<i32>) -> Column {
        Column { data: ColumnData::I32(Buffer::from_vec(v), Validity::dense()) }
    }

    /// Build a `Bool` column (all values present).
    pub fn bool(v: Vec<bool>) -> Column {
        Column { data: ColumnData::Bool(Buffer::from_vec(v), Validity::dense()) }
    }

    /// Build an `I64` column (all values present).
    pub fn i64(v: Vec<i64>) -> Column {
        Column { data: ColumnData::I64(Buffer::from_vec(v), Validity::dense()) }
    }

    /// Build an `I64` column with an explicit validity (missing-aware). The validity
    /// must mask exactly `v.len()` values, else a [`VolasError::Shape`] error — the
    /// published constructor checks the layout invariant every reader relies on.
    pub fn i64_with(v: Vec<i64>, validity: Validity) -> Result<Column> {
        Column::i64_masked(v, validity).checked_layout()
    }

    /// Build an `I32` column with an explicit validity, checked like
    /// [`i64_with`](Self::i64_with).
    pub fn i32_with(v: Vec<i32>, validity: Validity) -> Result<Column> {
        Column::i32_masked(v, validity).checked_layout()
    }

    /// Build a `Bool` column with an explicit validity, checked like
    /// [`i64_with`](Self::i64_with).
    pub fn bool_with(v: Vec<bool>, validity: Validity) -> Result<Column> {
        Column::bool_masked(v, validity).checked_layout()
    }

    /// Build a `Str` column (all values present).
    pub fn str(v: Vec<String>) -> Column {
        Column { data: ColumnData::Str(StrBuffer::from_vec(v), Validity::dense()) }
    }

    /// Build a `Str` column with an explicit validity, checked like
    /// [`i64_with`](Self::i64_with).
    pub fn str_with(v: Vec<String>, validity: Validity) -> Result<Column> {
        Column::str_masked(v, validity).checked_layout()
    }

    /// Build an `I64` column from optional cells (`None` is `volas.NA`). The validity
    /// is derived from the cells, so the layout always holds.
    pub fn i64_from_options(cells: impl IntoIterator<Item = Option<i64>>) -> Column {
        let (values, validity) = split_options(cells, 0);
        Column::i64_masked(values, validity)
    }

    /// Build a `Bool` column from optional cells (`None` is `volas.NA`).
    pub fn bool_from_options(cells: impl IntoIterator<Item = Option<bool>>) -> Column {
        let (values, validity) = split_options(cells, false);
        Column::bool_masked(values, validity)
    }

    /// Build a `Str` column from optional cells (`None` is `volas.NA`).
    pub fn str_from_options(cells: impl IntoIterator<Item = Option<String>>) -> Column {
        let (values, validity) = split_options(cells, String::new());
        Column::str_masked(values, validity)
    }

    /// This column with every cell where `missing[i]` holds made missing, in the
    /// column's own representation (`NaN` / `NaT` in-band, an int / bool / str
    /// validity hole) — the one boundary for values that arrive with a separate
    /// missing mask (a NumPy masked array, an interop `(values, mask)` pair). A mask
    /// of another length is a [`VolasError::Shape`] error, never a partial or
    /// out-of-bounds read.
    pub fn with_missing_mask(&self, missing: &[bool]) -> Result<Column> {
        let n = self.len();
        if missing.len() != n {
            return Err(VolasError::Shape(format!(
                "missing mask has {} entries but the column has {n}",
                missing.len()
            )));
        }
        Ok(self.missing_unless(|i| !missing[i]))
    }

    // The crate's own kernels derive each validity from the values they build (the
    // same length by construction), so they assemble the column directly; every
    // column a frame receives is checked again by `DataFrame::new`.

    pub(crate) fn i64_masked(v: Vec<i64>, validity: Validity) -> Column {
        Column { data: ColumnData::I64(Buffer::from_vec(v), validity) }
    }

    pub(crate) fn i32_masked(v: Vec<i32>, validity: Validity) -> Column {
        Column { data: ColumnData::I32(Buffer::from_vec(v), validity) }
    }

    pub(crate) fn bool_masked(v: Vec<bool>, validity: Validity) -> Column {
        Column { data: ColumnData::Bool(Buffer::from_vec(v), validity) }
    }

    pub(crate) fn str_masked(v: Vec<String>, validity: Validity) -> Column {
        Column { data: ColumnData::Str(StrBuffer::from_vec(v), validity) }
    }

    /// The layout invariant of a nullable column: its validity masks exactly its
    /// values (a dense validity always does). A [`VolasError::Shape`] error names
    /// both lengths — a mask of another length would read past its end or count
    /// missing values the column does not have. Float / datetime columns carry
    /// missing in-band and always pass.
    pub fn check_layout(&self) -> Result<()> {
        match self.validity() {
            Some(val) if !val.covers(self.len()) => Err(VolasError::Shape(format!(
                "validity mask covers {} values but the column has {}",
                val.bitmap().map_or(0, |bm| bm.len()),
                self.len()
            ))),
            _ => Ok(()),
        }
    }

    /// `self` when its layout holds ([`check_layout`](Self::check_layout)).
    pub(super) fn checked_layout(self) -> Result<Column> {
        self.check_layout().map(|()| self)
    }

    /// Build a `Datetime` column (epoch nanoseconds).
    pub fn datetime(v: Vec<i64>) -> Column {
        Column { data: ColumnData::Datetime(Buffer::from_vec(v)) }
    }

    /// An all-missing column of `dtype` with `len` rows — the dtype-preserving
    /// default `other` for `where` / `mask` (so a kept value stays in its dtype
    /// and a replaced one becomes NA, never an f64-funnel NaN).
    pub fn na_of(dtype: DType, len: usize) -> Column {
        let all_na = || Validity::from_valid_iter(len, std::iter::repeat_n(false, len));
        match dtype {
            DType::F64 => Column::f64(vec![f64::NAN; len]),
            DType::F32 => Column::f32(vec![f32::NAN; len]),
            DType::I64 => Column::i64_masked(vec![0; len], all_na()),
            DType::I32 => Column::i32_masked(vec![0; len], all_na()),
            DType::Bool => Column::bool_masked(vec![false; len], all_na()),
            DType::Utf8 => Column::str_masked(vec![String::new(); len], all_na()),
            DType::Datetime => Column::datetime(vec![i64::MIN; len]),
        }
    }

}

/// Split optional cells into values (`placeholder` under a `None`) and the validity
/// they imply — the two always have the same length.
fn split_options<T>(cells: impl IntoIterator<Item = Option<T>>, placeholder: T) -> (Vec<T>, Validity)
where
    T: Clone,
{
    let cells = cells.into_iter();
    let (mut values, mut valid) = (Vec::with_capacity(cells.size_hint().0), Vec::new());
    for cell in cells {
        valid.push(cell.is_some());
        values.push(cell.unwrap_or_else(|| placeholder.clone()));
    }
    let validity = Validity::from_valid_iter(values.len(), valid);
    (values, validity)
}
