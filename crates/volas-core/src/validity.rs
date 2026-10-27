//! Validity: an optional, packed missing-value mask for nullable column
//! variants (`I64` / `I32` / `Bool`).
//!
//! `volas.NA` is one logical missing value, but each dtype stores it in its
//! natural place: a float column uses `NaN` in-band (no mask), so only the
//! integer and boolean variants carry a [`Validity`]. A [`Validity`] is `None`
//! in the common dense case — every value present, **zero allocation** — and a
//! packed [`Bitmap`] (1 bit per element, **set = present**) only once a missing
//! value actually appears. This is the Arrow / pandas-nullable physical layout,
//! kept behind a uniform `is_valid` interface so the rest of the library never
//! sees the per-dtype storage difference.

use std::sync::Arc;
use crate::error::{Result, VolasError};

/// A packed validity mask: 1 bit per element, **set = present (valid)**, clear =
/// missing (`volas.NA`). Bits are stored LSB-first in `u64` words; bits beyond
/// `len` are kept clear so a popcount over every word counts exactly the present
/// values. The count of missing values is kept as the bits change, so reading it —
/// and a single-cell write — is O(1), never a popcount of the whole mask.
/// Raw bitmap mutation is private to the owning crate.
///
/// ```compile_fail
/// use volas_core::Bitmap;
/// let mut bitmap = Bitmap::from_valid_iter(2, [false, false]);
/// bitmap.set(63, true);
/// ```
#[derive(Clone, Debug)]
pub struct Bitmap {
    words: Vec<u64>,
    len: usize,
    /// The number of clear bits among the first `len`.
    nulls: usize,
    // A Column transaction may reserve a first mask before writing its hole.
    // Keep that capacity across replacement writes until append or cancellation.
    prepared: bool,
}

/// Words needed to hold `len` bits.
fn words_for(len: usize) -> usize {
    len.div_ceil(64)
}

impl Bitmap {
    /// An all-present mask of `len` bits (every value valid).
    pub fn all_valid(len: usize) -> Bitmap {
        let mut words = vec![u64::MAX; words_for(len)];
        // Clear the unused high bits of the final word so popcount == len.
        if let Some(last) = words.last_mut() {
            let used = len % 64;
            if used != 0 {
                *last = (1u64 << used) - 1;
            }
        }
        Bitmap { words, len, nulls: 0, prepared: false }
    }

    /// Build from an iterator of `valid` flags (true = present).
    pub fn from_valid_iter(len: usize, valid: impl IntoIterator<Item = bool>) -> Bitmap {
        let mut bm = Bitmap { words: vec![0; words_for(len)], len, nulls: 0, prepared: false };
        for (i, v) in valid.into_iter().enumerate().take(len) {
            if v {
                bm.words[i / 64] |= 1u64 << (i % 64);
            }
        }
        bm.nulls = len - present_bits(&bm.words);
        bm
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the mask covers no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether element `i` is present (valid). `i` is assumed in bounds.
    pub fn get(&self, i: usize) -> bool {
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Mark element `i` present (`valid = true`) or missing, keeping the missing
    /// count. `i` in bounds.
    pub(crate) fn set(&mut self, i: usize, valid: bool) {
        if self.get(i) == valid {
            return;
        }
        let (w, bit) = (i / 64, 1u64 << (i % 64));
        if valid {
            self.words[w] |= bit;
            self.nulls -= 1;
        } else {
            self.words[w] &= !bit;
            self.nulls += 1;
        }
    }

    /// Count of missing values (clear bits among the first `len`) — O(1).
    pub fn null_count(&self) -> usize {
        self.nulls
    }

    /// Grow by `count` missing (clear) bits — amortized O(count / 64): the words
    /// past `len` are already clear.
    fn grow_missing(&mut self, count: usize) {
        self.len += count;
        self.nulls += count;
        self.words.resize(words_for(self.len), 0);
    }
}

/// Set bits across `words`, accumulated in `usize` — a bitmap may hold more than
/// `u32::MAX` rows, so a narrower accumulator would overflow.
fn present_bits<'a>(words: impl IntoIterator<Item = &'a u64>) -> usize {
    words.into_iter().map(|w| w.count_ones() as usize).sum()
}

impl PartialEq for Bitmap {
    /// Equal when the same length and the same present/missing pattern (unused
    /// high bits are always clear, so a plain word comparison is exact).
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.words == other.words
    }
}

/// An optional validity mask. `None` (the dense default) means every value is
/// present with no allocation; `Some` carries a [`Bitmap`] with at least one
/// missing value. The invariant "`Some` ⇒ has a missing value" is upheld by the
/// constructors so equality and `has_nulls` stay cheap and canonical.
/// External writers use the checked [`set`](Self::set) operation; structural
/// growth is controlled by the owning column.
///
/// ```compile_fail
/// use volas_core::Validity;
/// let mut validity = Validity::from_valid_iter(2, [false, false]);
/// validity.extend_missing(63, 1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Validity(Option<Arc<Bitmap>>);

impl Validity {
    /// A fully-present (dense) validity — no allocation.
    pub fn dense() -> Validity {
        Validity(None)
    }

    /// Wrap a bitmap, collapsing an all-present one back to dense so the dense
    /// fast path is always taken when there are no missing values.
    pub fn from_bitmap(bm: Bitmap) -> Validity {
        if bm.null_count() == 0 {
            Validity(None)
        } else {
            Validity(Some(Arc::new(bm)))
        }
    }

    /// Build from an iterator of `valid` flags; dense when all are present.
    pub fn from_valid_iter(len: usize, valid: impl IntoIterator<Item = bool>) -> Validity {
        Validity::from_bitmap(Bitmap::from_valid_iter(len, valid))
    }

    /// Whether element `i` is present. Dense ⇒ always true. `i` in bounds.
    pub fn is_valid(&self, i: usize) -> bool {
        match &self.0 {
            None => true,
            Some(bm) => bm.get(i),
        }
    }

    /// Set element `i`'s presence in place (one cell). Dense stays dense when
    /// marking present; a bitmap is materialized only when a first missing value is
    /// introduced. `len` is the column length, needed to size that fresh bitmap.
    /// Rejects an out-of-bounds index or a bitmap of another length without changing the mask.
    pub fn set(&mut self, i: usize, len: usize, valid: bool) -> Result<()> {
        if !self.covers(len) {
            return Err(VolasError::Shape(format!("validity mask does not cover {len} values")));
        }
        if i >= len {
            return Err(VolasError::Index(format!("validity index {i} out of bounds for length {len}")));
        }
        self.set_in_bounds(i, len, valid);
        Ok(())
    }

    /// Update a cell whose bounds and mask length the owning column established.
    pub(crate) fn set_in_bounds(&mut self, i: usize, len: usize, valid: bool) {
        match &mut self.0 {
            None => {
                if !valid {
                    let mut bm = Bitmap::all_valid(len);
                    bm.set(i, false);
                    self.0 = Some(Arc::new(bm));
                }
            }
            Some(arc) => {
                let bm = Arc::make_mut(arc);
                // Filling the last hole empties the mask, so it re-collapses to
                // dense — preserving the "Some ⇒ has a missing value" invariant that
                // `has_nulls` / `null_count` / `PartialEq` / the NA-aware export paths
                // rely on. The bitmap keeps its missing count, so this is O(1).
                bm.set(i, valid);
                if bm.null_count() == 0 && !bm.prepared {
                    self.0 = None;
                }
            }
        }
    }

    /// Reserve a possible first missing bit, keeping all existing bits intact.
    /// An all-present prepared bitmap is private to the column transaction.
    pub(crate) fn prepare_growth(&mut self, len: usize, added: usize, missing: bool) -> Result<()> {
        if self.0.is_none() && !missing { return Ok(()); }
        let total = len.checked_add(added).ok_or_else(crate::column::capacity::capacity_error)?;
        let words = words_for(total);
        if let Some(bitmap) = self.0.as_mut().and_then(Arc::get_mut) {
            bitmap.words.try_reserve(words - bitmap.words.len()).map_err(crate::column::capacity::allocation_error)?;
            bitmap.prepared = true;
            return Ok(());
        }
        let mut storage = Vec::new();
        storage.try_reserve(crate::capacity::growth_capacity(words_for(len), words))
            .map_err(crate::column::capacity::allocation_error)?;
        let (nulls, old_len) = if let Some(bitmap) = &self.0 {
            storage.extend_from_slice(&bitmap.words);
            (bitmap.nulls, bitmap.len)
        } else {
            storage.resize(words_for(len), u64::MAX);
            if !len.is_multiple_of(64) { *storage.last_mut().unwrap() = (1u64 << (len % 64)) - 1; }
            (0, len)
        };
        self.0 = Some(Arc::new(Bitmap { words: storage, len: old_len, nulls, prepared: true }));
        Ok(())
    }

    pub(crate) fn finish_growth(&mut self) {
        if let Some(b) = self.0.as_mut().and_then(Arc::get_mut) { b.prepared = false; }
        if self.0.as_ref().is_some_and(|b| b.nulls == 0) { self.0 = None; }
    }

    /// Append prepared validity bits without a temporary per-row flags vector.
    pub(crate) fn append_from(&mut self, len: usize, source: &Self, rows: std::ops::Range<usize>) {
        if self.0.is_none() && !source.has_nulls() { return; }
        let count = rows.len();
        if self.0.is_none() { self.0 = Some(Arc::new(Bitmap::all_valid(len))); }
        let bitmap = Arc::make_mut(self.0.as_mut().unwrap());
        bitmap.grow_missing(count);
        for (offset, row) in rows.enumerate() { if source.is_valid(row) { bitmap.set(len + offset, true); } }
        self.finish_growth();
    }

    /// Grow the mask of a `len`-value column by `count` missing values, in place:
    /// amortized O(count) on a uniquely-owned bitmap; a dense mask materializes its
    /// bitmap once (O(len / 64)). No-op for `count == 0`.
    pub(crate) fn extend_missing(&mut self, len: usize, count: usize) {
        if count == 0 {
            return;
        }
        match &mut self.0 {
            Some(arc) => {
                debug_assert_eq!(arc.len(), len, "the mask covers the column");
                Arc::make_mut(arc).grow_missing(count)
            }
            None => {
                let mut bm = Bitmap::all_valid(len);
                bm.grow_missing(count);
                self.0 = Some(Arc::new(bm));
            }
        }
        self.finish_growth();
    }

    /// Whether this validity masks exactly `len` values: a dense validity fits every
    /// length; a bitmap must cover `len` bits — a shorter one would read past its
    /// end, a longer one would count missing values the column does not have.
    pub fn covers(&self, len: usize) -> bool {
        self.0.as_ref().is_none_or(|bm| bm.len() == len)
    }

    /// Whether any value is missing (cheap: dense ⇒ false).
    pub fn has_nulls(&self) -> bool {
        self.0.is_some()
    }

    /// Number of missing values (dense ⇒ 0).
    pub fn null_count(&self) -> usize {
        self.0.as_ref().map_or(0, |bm| bm.null_count())
    }

    /// Borrow the backing bitmap, if any (`None` when dense).
    pub fn bitmap(&self) -> Option<&Bitmap> {
        self.0.as_deref()
    }

    /// Debug-build layout check (compiled out of release builds): a bitmap covers
    /// exactly the `len` values it masks, and `Some` holds at least one missing value
    /// (the canonical form `has_nulls` / equality / the exports rely on).
    #[cfg(debug_assertions)]
    pub(crate) fn debug_check(&self, len: usize, column: &str) {
        if let Some(bm) = &self.0 {
            assert_eq!(bm.len(), len, "validity of {column:?} covers the wrong length");
            assert!(bm.null_count() > 0, "validity of {column:?} is a bitmap without a hole");
        }
    }

    /// Combined validity of two equal-length columns: present only where **both**
    /// are present — the missing-value rule for a binary op (`x ∘ NA = NA`).
    pub fn and(&self, other: &Validity) -> Validity {
        match (&self.0, &other.0) {
            (None, None) => Validity::dense(),
            // One side dense ⇒ the other's holes win (a `Some` already has a hole).
            (Some(a), None) | (None, Some(a)) => Validity(Some(Arc::clone(a))),
            (Some(a), Some(b)) => {
                let n = a.len();
                Validity::from_valid_iter(n, (0..n).map(|i| a.get(i) && b.get(i)))
            }
        }
    }

    /// Gather present-flags at `idx` (fancy indexing / dropna). Dense stays dense.
    pub fn take(&self, idx: &[usize]) -> Validity {
        match &self.0 {
            None => Validity::dense(),
            Some(bm) => Validity::from_valid_iter(idx.len(), idx.iter().map(|&i| bm.get(i))),
        }
    }

    /// Present-flags over `[start, end)` (slicing). Dense stays dense.
    pub fn slice(&self, start: usize, end: usize) -> Validity {
        match &self.0 {
            None => Validity::dense(),
            Some(bm) => Validity::from_valid_iter(end - start, (start..end).map(|i| bm.get(i))),
        }
    }
}

impl PartialEq for Validity {
    /// Dense compares equal to an all-present bitmap (the constructors keep
    /// `Some` non-trivial, so in practice this reduces to comparing patterns).
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            // Constructors collapse all-present to `None`, so a `Some` here has a
            // missing value and can never equal a dense `None`.
            (Some(_), None) | (None, Some(_)) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_bit_count_holds_past_u32_max() {
        // 2^26 full words = 2^32 present bits (a 2^32-row all-valid bitmap), without
        // allocating the 512 MiB bitmap itself.
        let words = std::iter::repeat_n(&u64::MAX, 1 << 26);
        assert_eq!(present_bits(words), 1usize << 32);
    }

    #[test]
    fn checked_validity_writes_preserve_counts_and_reject_invalid_positions() {
        let mut validity = Validity::from_valid_iter(2, [false, false]);
        let before = validity.clone();
        assert_eq!(validity.set(63, 2, true), Err(VolasError::Index(
            "validity index 63 out of bounds for length 2".into()
        )));
        assert_eq!(validity, before);
        assert_eq!(validity.set(0, 3, true), Err(VolasError::Shape(
            "validity mask does not cover 3 values".into()
        )));
        assert_eq!(validity, before);
        let column = crate::Column::i64_with(vec![1, 2], validity).unwrap();
        assert_eq!(column.count(), (0..2).filter(|&i| column.is_valid(i)).count());
        assert_eq!(column.null_count(), 2);

        let mut dense = Validity::dense();
        for (position, length) in [(0, 0), (3, 3)] {
            assert!(matches!(dense.set(position, length, false), Err(VolasError::Index(_))));
            assert!(dense.bitmap().is_none());
        }
        let mut changed = before.clone();
        changed.set(0, 2, true).unwrap();
        assert_eq!((changed.null_count(), before.null_count()), (1, 2));
        assert!(changed.is_valid(0) && !before.is_valid(0));
    }

    #[test]
    fn validity_set_cell_in_place() {
        let mut v = Validity::dense();
        v.set(1, 3, true).unwrap(); // dense + still present -> stays dense (no allocation)
        assert!(!v.has_nulls());
        v.set(1, 3, false).unwrap(); // first missing value -> materialize a bitmap
        assert!(v.has_nulls());
        assert!(!v.is_valid(1) && v.is_valid(0));
        v.set(1, 3, true).unwrap(); // Some(bitmap) branch -> back to present
        assert!(v.is_valid(1));
        // Clearing the LAST hole re-collapses to dense (C1): the `Some ⇒ has a
        // missing value` invariant holds, so has_nulls/null_count stay canonical.
        assert!(!v.has_nulls() && v.bitmap().is_none() && v.null_count() == 0);
    }

    #[test]
    fn validity_set_collapses_only_on_last_hole() {
        // Two holes: filling one stays Some (still a hole); filling the second
        // re-collapses to dense. A no-op set-present must not collapse spuriously.
        let mut v = Validity::from_valid_iter(4, [true, false, true, false]);
        assert_eq!(v.null_count(), 2);
        v.set(3, 4, true).unwrap(); // fill one hole -> still Some (idx 1 missing)
        assert!(v.has_nulls() && v.null_count() == 1);
        v.set(0, 4, true).unwrap(); // no-op set-present (already valid) -> stays Some
        assert!(v.has_nulls() && v.null_count() == 1);
        v.set(1, 4, true).unwrap(); // fill the last hole -> collapse to dense
        assert!(!v.has_nulls() && v.bitmap().is_none());
    }

    #[test]
    fn bitmap_get_set_and_null_count() {
        let mut bm = Bitmap::all_valid(70); // spans two words
        assert_eq!(bm.len(), 70);
        assert!(!bm.is_empty());
        assert_eq!(bm.null_count(), 0);
        assert!(bm.get(0) && bm.get(69));
        bm.set(3, false);
        bm.set(64, false);
        assert!(!bm.get(3) && !bm.get(64));
        assert_eq!(bm.null_count(), 2);
        bm.set(3, true);
        assert!(bm.get(3));
        assert_eq!(bm.null_count(), 1);
    }

    #[test]
    fn bitmap_keeps_its_missing_count_as_bits_change() {
        let mut bm = Bitmap::from_valid_iter(130, (0..130).map(|i| i % 7 != 0));
        assert_eq!(bm.null_count(), 19);
        bm.set(0, false); // already missing: unchanged
        bm.set(1, true); // already present: unchanged
        assert_eq!(bm.null_count(), 19);
        bm.set(0, true);
        bm.set(129, false);
        bm.set(128, false);
        assert_eq!(bm.null_count(), 20);
        bm.grow_missing(70);
        assert_eq!((bm.len(), bm.null_count()), (200, 90));
        assert_eq!(bm.null_count(), bm.len() - present_bits(&bm.words));
    }

    #[test]
    fn bitmap_null_count_past_u32() {
        // 2^32 present bits (512 MiB): the popcount must not wrap a 32-bit
        // accumulator (release reported 2^32 nulls, debug panicked).
        let n = 1usize << 32;
        let mut bm = Bitmap::all_valid(n + 1);
        assert_eq!(bm.null_count(), 0);
        bm.set(n, false);
        assert_eq!(bm.null_count(), 1);
    }

    #[test]
    fn bitmap_empty_and_full_word_boundary() {
        assert!(Bitmap::all_valid(0).is_empty());
        // exactly 64 bits: no partial-word masking, popcount == len
        assert_eq!(Bitmap::all_valid(64).null_count(), 0);
    }

    #[test]
    fn bitmap_from_valid_iter_and_eq() {
        let a = Bitmap::from_valid_iter(3, [true, false, true]);
        let b = Bitmap::from_valid_iter(3, [true, false, true]);
        assert_eq!(a, b);
        assert_eq!(a.null_count(), 1);
        assert!(!a.get(1));
        assert_ne!(a, Bitmap::from_valid_iter(3, [true, true, true]));
        assert_ne!(a, Bitmap::from_valid_iter(2, [true, false])); // length differs
    }

    #[test]
    fn validity_dense_is_all_valid() {
        let v = Validity::dense();
        assert!(!v.has_nulls());
        assert_eq!(v.null_count(), 0);
        assert!(v.is_valid(0) && v.is_valid(999));
        assert!(v.bitmap().is_none());
    }

    #[test]
    fn validity_collapses_all_present_to_dense() {
        // an all-present bitmap becomes dense (None)
        let v = Validity::from_valid_iter(3, [true, true, true]);
        assert!(!v.has_nulls());
        assert!(v.bitmap().is_none());
        // a bitmap with a hole stays Some
        let v = Validity::from_valid_iter(3, [true, false, true]);
        assert!(v.has_nulls());
        assert_eq!(v.null_count(), 1);
        assert!(!v.is_valid(1) && v.is_valid(0));
        assert!(v.bitmap().is_some());
    }

    #[test]
    fn validity_from_bitmap_and_default() {
        assert_eq!(Validity::default(), Validity::dense());
        let v = Validity::from_bitmap(Bitmap::all_valid(4));
        assert!(!v.has_nulls()); // collapsed to dense
    }

    #[test]
    fn validity_eq() {
        let dense = Validity::dense();
        let holed = Validity::from_valid_iter(2, [true, false]);
        assert_eq!(dense, Validity::dense());
        assert_eq!(holed, Validity::from_valid_iter(2, [true, false]));
        assert_ne!(dense, holed);
        assert_ne!(holed, dense);
        assert_ne!(holed, Validity::from_valid_iter(2, [false, true]));
    }

    #[test]
    fn validity_and_combines_holes() {
        let dense = Validity::dense();
        let a = Validity::from_valid_iter(3, [true, false, true]);
        let b = Validity::from_valid_iter(3, [true, true, false]);
        assert_eq!(dense.and(&dense), Validity::dense()); // both dense
        assert_eq!(dense.and(&a), a); // dense ∧ holed -> holed
        assert_eq!(a.and(&dense), a); // holed ∧ dense -> holed
        // hole from either side wins
        assert_eq!(a.and(&b), Validity::from_valid_iter(3, [true, false, false]));
    }
}

#[cfg(test)]
mod growth_spare_tests {
    use super::*;
    #[test]
    fn shared_and_first_masks_keep_the_next_words_spare() {
        let sparse = Validity::from_valid_iter(512, (0..512).map(|i| i != 0));
        let snapshot = sparse.clone();
        for mut mask in [sparse, Validity::dense()] {
            let nulls = mask.null_count();
            mask.prepare_growth(512, 1, true).unwrap();
            assert!(mask.bitmap().unwrap().words.capacity() >= words_for(577), "mask growth must retain the next word's spare");
            mask.extend_missing(512, 1);
            assert_eq!(mask.null_count(), nulls + 1);
            assert!(!mask.is_valid(512));
        }
        assert_eq!(snapshot.bitmap().unwrap().len(), 512);
        assert_eq!(snapshot.null_count(), 1);
    }
}
