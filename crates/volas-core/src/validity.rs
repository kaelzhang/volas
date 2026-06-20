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

/// A packed validity mask: 1 bit per element, **set = present (valid)**, clear =
/// missing (`volas.NA`). Bits are stored LSB-first in `u64` words; bits beyond
/// `len` are kept clear so a popcount over every word counts exactly the present
/// values.
#[derive(Clone, Debug)]
pub struct Bitmap {
    words: Vec<u64>,
    len: usize,
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
        Bitmap { words, len }
    }

    /// Build from an iterator of `valid` flags (true = present).
    pub fn from_valid_iter(len: usize, valid: impl IntoIterator<Item = bool>) -> Bitmap {
        let mut bm = Bitmap { words: vec![0; words_for(len)], len };
        for (i, v) in valid.into_iter().enumerate().take(len) {
            if v {
                bm.words[i / 64] |= 1u64 << (i % 64);
            }
        }
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

    /// Mark element `i` present (`valid = true`) or missing. `i` in bounds.
    pub fn set(&mut self, i: usize, valid: bool) {
        let (w, bit) = (i / 64, 1u64 << (i % 64));
        if valid {
            self.words[w] |= bit;
        } else {
            self.words[w] &= !bit;
        }
    }

    /// Count of missing values (clear bits among the first `len`).
    pub fn null_count(&self) -> usize {
        let present: u32 = self.words.iter().map(|w| w.count_ones()).sum();
        self.len - present as usize
    }
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
    /// Backs the single-cell write in [`crate::Column::combine_at`].
    pub fn set(&mut self, i: usize, len: usize, valid: bool) {
        match &mut self.0 {
            None => {
                if !valid {
                    let mut bm = Bitmap::all_valid(len);
                    bm.set(i, false);
                    self.0 = Some(Arc::new(bm));
                }
            }
            Some(arc) => Arc::make_mut(arc).set(i, valid),
        }
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
    fn validity_set_cell_in_place() {
        let mut v = Validity::dense();
        v.set(1, 3, true); // dense + still present -> stays dense (no allocation)
        assert!(!v.has_nulls());
        v.set(1, 3, false); // first missing value -> materialize a bitmap
        assert!(v.has_nulls());
        assert!(!v.is_valid(1) && v.is_valid(0));
        v.set(1, 3, true); // Some(bitmap) branch -> back to present
        assert!(v.is_valid(1));
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
