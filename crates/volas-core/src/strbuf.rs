//! [`StrBuffer`]: the Arrow-native columnar string layout — one contiguous UTF-8
//! [`Buffer<u8>`] plus an `n+1` [`Buffer<i64>`] of offsets (Arrow `LargeUtf8`).
//!
//! This replaces an `Arc<Vec<String>>` (one heap allocation per cell, pointer-
//! chasing scans): a `StrBuffer` is a single allocation, cache-friendly to build
//! and scan, and byte-compatible with Arrow for zero-copy interop. Cell access is
//! a slice of the data buffer (`get` / `iter`). Missing cells are tracked by the
//! column's separate `Validity`; an NA cell carries an empty (or placeholder) span
//! here, exactly like the old `String::new()` placeholder.

use std::fmt;

use crate::buffer::Buffer;
use crate::error::{Result, VolasError};
use crate::validity::Validity;

/// A contiguous UTF-8 string column buffer (Arrow `LargeUtf8`: i64 offsets).
#[derive(Clone)]
pub struct StrBuffer {
    /// `len + 1` monotonic byte offsets into `data`, always **zero-based**:
    /// `offsets[0] == 0` and `offsets[len] == data.len()`, and every span
    /// `offsets[i]..offsets[i + 1]` is valid UTF-8. A volas-built buffer holds this
    /// by construction; foreign buffers enter only through
    /// [`try_from_buffers`](StrBuffer::try_from_buffers), which checks it (an Arrow
    /// import of a *sliced* array is re-based to 0 by the caller first). The
    /// unchecked accessors (`get` / `get_unchecked` / `iter`) and `extend` rely on it.
    offsets: Buffer<i64>,
    /// Concatenated UTF-8 bytes of every cell.
    data: Buffer<u8>,
}

impl StrBuffer {
    /// Build from owned `String`s.
    #[inline]
    pub fn from_vec(v: Vec<String>) -> Self {
        v.into_iter().collect()
    }

    /// Zero-copy from foreign Arrow `LargeUtf8`-layout buffers, **validated**: the
    /// offsets must be zero-based, monotonic and end at `data.len()`, and every
    /// **present** cell (per `validity`, which must cover the same length) must be
    /// valid UTF-8 on its own span — otherwise a `Value` error, never a buffer the
    /// unchecked accessors would misread. A **missing** cell's bytes are unspecified
    /// (the Arrow format allows anything there), so when one is not valid UTF-8 the
    /// buffer is repacked with every missing cell as an empty span; a buffer that is
    /// already sound stays a zero-copy borrow.
    pub fn try_from_buffers(
        offsets: Buffer<i64>,
        data: Buffer<u8>,
        validity: &Validity,
    ) -> Result<Self> {
        let invalid = |what: String| VolasError::Value(format!("invalid string buffer: {what}"));
        let off = offsets.as_slice();
        let bytes = data.as_slice();
        let Some(&first) = off.first() else {
            return Err(invalid("offsets must hold len + 1 entries (got none)".into()));
        };
        let len = off.len() - 1;
        if let Some(bm) = validity.bitmap() {
            if bm.len() != len {
                return Err(invalid(format!(
                    "validity covers {} cells but the offsets describe {len}",
                    bm.len()
                )));
            }
        }
        if first != 0 {
            return Err(invalid(format!("offsets must start at 0 (got {first})")));
        }
        if let Some(i) = off.windows(2).position(|w| w[0] > w[1]) {
            return Err(invalid(format!(
                "offsets must be monotonic (offset {i} = {} > offset {} = {})",
                off[i],
                i + 1,
                off[i + 1]
            )));
        }
        if off[len] as u64 != bytes.len() as u64 {
            return Err(invalid(format!(
                "offsets must end at the data length {} (got {})",
                bytes.len(),
                off[len]
            )));
        }
        // Fast path: the whole buffer is UTF-8 and every offset sits on a char
        // boundary, so every span is UTF-8 (the arrow-rs check).
        if let Ok(text) = std::str::from_utf8(bytes) {
            if off.iter().all(|&o| text.is_char_boundary(o as usize)) {
                return Ok(StrBuffer { offsets, data });
            }
        }
        // Slow path — some span is not UTF-8 (were all of them, their concatenation
        // would have passed above). A present cell's must be; a missing cell's bytes
        // are dropped: repack with every missing cell as an empty span.
        let mut b = StrBufferBuilder::with_capacity(len);
        for i in 0..len {
            if validity.is_valid(i) {
                let cell = std::str::from_utf8(&bytes[off[i] as usize..off[i + 1] as usize])
                    .map_err(|_| invalid(format!("cell {i} is not valid UTF-8")))?;
                b.push(cell);
            } else {
                b.push("");
            }
        }
        Ok(b.finish())
    }

    /// The raw offset / data buffers (for a zero-copy Arrow export).
    #[inline]
    pub fn buffers(&self) -> (&Buffer<i64>, &Buffer<u8>) {
        (&self.offsets, &self.data)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cell `i` as `&str` (a borrow into the data buffer). Bounds-checked.
    #[inline]
    pub fn get(&self, i: usize) -> &str {
        let off = self.offsets.as_slice();
        let (s, e) = (off[i] as usize, off[i + 1] as usize);
        // SAFETY: every span is valid UTF-8 — the structural invariant every
        // constructor establishes (`try_from_buffers` checks foreign buffers).
        unsafe { std::str::from_utf8_unchecked(&self.data.as_slice()[s..e]) }
    }

    /// Cell `i` as `&str`, **without** bounds checks — the zero-overhead accessor for
    /// the internal hot kernels (compare / sort / scan), mirroring arrow-rs
    /// `GenericStringArray::value_unchecked`.
    ///
    /// # Safety
    /// `i < self.len()`. The offsets are a structural invariant — monotonic with
    /// `offsets[len] == data.len()` — so `i < len` makes both offset reads and the
    /// `[s..e]` data span in bounds; only the redundant checks are elided.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> &str {
        let off = self.offsets.as_slice();
        let (s, e) = (*off.get_unchecked(i) as usize, *off.get_unchecked(i + 1) as usize);
        std::str::from_utf8_unchecked(self.data.as_slice().get_unchecked(s..e))
    }

    /// Iterate the cells as `&str`. The two backing slices are resolved **once** (a
    /// single `Buffer` match each, hoisted out of the walk), then indexed without
    /// bounds checks — so a sequential scan is one tight loop, matching a `&[String]`
    /// walk with no per-element overhead.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &str> + '_ {
        let off = self.offsets.as_slice();
        let data = self.data.as_slice();
        // The resolved offsets already carry the admitted cell count.
        (0..off.len() - 1).map(move |i| {
            // SAFETY: `i < len` ⟹ `i + 1 <= len < off.len()`, and offsets are monotonic
            // with `off[len] == data.len()`, so `off[i]..off[i + 1]` is in bounds of `data`
            // and valid UTF-8 (the invariant every constructor establishes).
            unsafe {
                let s = *off.get_unchecked(i) as usize;
                let e = *off.get_unchecked(i + 1) as usize;
                std::str::from_utf8_unchecked(data.get_unchecked(s..e))
            }
        })
    }

    /// Materialize owned `String`s (only where a `Vec<String>` is genuinely needed).
    pub fn to_vec(&self) -> Vec<String> {
        self.iter().map(String::from).collect()
    }

    /// A contiguous sub-range `[start, end)` as a fresh `StrBuffer` (re-packs the
    /// bytes; `Buffer` borrowing of a sub-span is a future refinement).
    pub fn slice(&self, start: usize, end: usize) -> StrBuffer {
        (start..end).map(|i| self.get(i)).collect()
    }

    pub(crate) fn prepare_growth(&mut self, rows: usize, bytes: usize) -> Result<()> {
        self.offsets.try_reserve(rows)?;
        self.data.try_reserve(bytes)
    }

    /// Append the cells of `items` **in place** — the offset and data buffers grow via
    /// copy-on-write rather than being rebuilt, so a live row-by-row append stays
    /// amortised `O(1)` per cell instead of `O(n)` (which made repeated append `O(n²)`).
    /// A borrowed (or shared) buffer materialises once on the first append.
    pub fn extend<S: AsRef<str>>(&mut self, items: impl IntoIterator<Item = S>) {
        let data = self.data.make_mut();
        let offsets = self.offsets.make_mut();
        for s in items {
            data.extend_from_slice(s.as_ref().as_bytes());
            offsets.push(data.len() as i64);
        }
    }
}

impl StrBuffer {
    /// Overwrite the last cell with `s` in place (its bytes are the tail of the data
    /// buffer, so only they are rewritten; the offsets stay zero-based and end at the
    /// data's length). No-op on an empty buffer.
    pub fn set_last(&mut self, s: &str) {
        let n = self.len();
        if n == 0 {
            return;
        }
        let start = self.offsets.as_slice()[n - 1] as usize;
        let data = self.data.make_mut();
        data.truncate(start);
        data.extend_from_slice(s.as_bytes());
        let end = data.len() as i64;
        self.offsets.make_mut()[n] = end;
    }
}

/// Incremental builder (append / scatter paths). Every constructor seeds the
/// leading `0` offset, so each `push` records exactly one cell.
pub struct StrBufferBuilder {
    offsets: Vec<i64>,
    data: Vec<u8>,
}

impl Default for StrBufferBuilder {
    fn default() -> Self {
        StrBufferBuilder::with_capacity(0)
    }
}

impl StrBufferBuilder {
    pub fn with_capacity(n: usize) -> Self {
        Self::with_capacities(n, 0)
    }
    /// Reserve row offsets and UTF-8 bytes for a known-size output.
    pub fn with_capacities(n: usize, bytes: usize) -> Self {
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        StrBufferBuilder { offsets, data: Vec::with_capacity(bytes) }
    }
    #[inline]
    pub fn push(&mut self, s: &str) {
        self.data.extend_from_slice(s.as_bytes());
        self.offsets.push(self.data.len() as i64);
    }
    pub fn finish(self) -> StrBuffer {
        StrBuffer {
            offsets: Buffer::from_vec(self.offsets),
            data: Buffer::from_vec(self.data),
        }
    }
}

impl PartialEq for StrBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl fmt::Debug for StrBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<S: AsRef<str>> FromIterator<S> for StrBuffer {
    /// The single construction path: concatenate the cells' UTF-8 into one `data`
    /// buffer, recording the running byte boundary in `offsets` (`n + 1` entries).
    fn from_iter<I: IntoIterator<Item = S>>(it: I) -> Self {
        let mut offsets = vec![0i64];
        let mut data = Vec::new();
        for s in it {
            data.extend_from_slice(s.as_ref().as_bytes());
            offsets.push(data.len() as i64);
        }
        StrBuffer {
            offsets: Buffer::from_vec(offsets),
            data: Buffer::from_vec(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_access_and_emptiness() {
        let sb = StrBuffer::from_vec(vec!["a".into(), "".into(), "cd".into()]);
        assert_eq!(sb.len(), 3);
        assert!(!sb.is_empty());
        assert_eq!(sb.get(2), "cd");
        // SAFETY: 0 < len.
        assert_eq!(unsafe { sb.get_unchecked(1) }, "");
        assert_eq!(sb.iter().collect::<Vec<_>>(), ["a", "", "cd"]);
        assert_eq!(sb.to_vec(), vec!["a".to_string(), "".into(), "cd".into()]);
        assert!(StrBuffer::from_vec(vec![]).is_empty());
    }

    #[test]
    fn buffers_round_trip_zero_copy() {
        // The Arrow bridge path: take the raw offset/data buffers out and rebuild.
        let sb = StrBuffer::from_vec(vec!["xy".into(), "z".into()]);
        let (offsets, data) = sb.buffers();
        let rebuilt =
            StrBuffer::try_from_buffers(offsets.clone(), data.clone(), &Validity::dense()).unwrap();
        assert_eq!(rebuilt, sb);
        assert_eq!(rebuilt.slice(1, 2), StrBuffer::from_vec(vec!["z".into()]));
    }

    #[test]
    fn builder_default_finishes_empty() {
        // A `Default` builder that never pushes must still yield valid `[0]` offsets.
        let empty = StrBufferBuilder::default().finish();
        assert!(empty.is_empty());
        let mut b = StrBufferBuilder::with_capacity(2);
        b.push("p");
        b.push("qr");
        assert_eq!(b.finish(), StrBuffer::from_vec(vec!["p".into(), "qr".into()]));
    }

    fn parts(offsets: Vec<i64>, data: &[u8]) -> (Buffer<i64>, Buffer<u8>) {
        (Buffer::from_vec(offsets), Buffer::from_vec(data.to_vec()))
    }

    #[test]
    fn try_from_buffers_rejects_every_broken_invariant() {
        let dense = Validity::dense();
        let cases: [(Vec<i64>, &[u8], &str); 6] = [
            (vec![], b"", "len + 1"),
            (vec![1, 2], b"ab", "start at 0"),
            (vec![0, 3, 1, 3], b"abc", "monotonic"),
            (vec![0, 2], b"abc", "end at the data length"),
            (vec![0, 2], b"\xff\xfe", "cell 0 is not valid UTF-8"),
            (vec![0, 1, 2], "é".as_bytes(), "cell 0 is not valid UTF-8"), // split char
        ];
        for (off, data, msg) in cases {
            let (o, d) = parts(off, data);
            let err = StrBuffer::try_from_buffers(o, d, &dense).unwrap_err();
            assert!(err.to_string().contains(msg), "{err}");
        }
        // a validity mask of the wrong length is rejected too
        let (o, d) = parts(vec![0, 1], b"a");
        let short = Validity::from_valid_iter(2, [true, false]);
        let err = StrBuffer::try_from_buffers(o, d, &short).unwrap_err();
        assert!(err.to_string().contains("validity covers 2 cells"), "{err}");
    }

    #[test]
    fn try_from_buffers_normalizes_non_utf8_missing_cells() {
        // a missing cell's bytes are unspecified: repacked as an empty span
        let validity = Validity::from_valid_iter(3, [false, true, false]);
        let (o, d) = parts(vec![0, 2, 3, 3], b"\xff\xfea");
        let sb = StrBuffer::try_from_buffers(o, d, &validity).unwrap();
        assert_eq!(sb.to_vec(), vec![String::new(), "a".into(), String::new()]);
        // a sound buffer with a (UTF-8) missing payload keeps its bytes zero-copy
        let validity = Validity::from_valid_iter(2, [false, true]);
        let (o, d) = parts(vec![0, 1, 3], "xé".as_bytes());
        let ptr = d.as_ptr();
        let sb = StrBuffer::try_from_buffers(o, d, &validity).unwrap();
        assert_eq!(sb.buffers().1.as_ptr(), ptr);
        assert_eq!(sb.get(1), "é");
        // a missing-cell span that splits a char is not UTF-8 alone: normalized
        let validity = Validity::from_valid_iter(2, [false, true]);
        let (o, d) = parts(vec![0, 1, 2], "é".as_bytes());
        let sb = StrBuffer::try_from_buffers(o, d, &validity);
        assert!(sb.is_err(), "the present cell 1 holds half a char");
    }

    #[test]
    fn default_builder_keeps_every_pushed_value() {
        let mut b = StrBufferBuilder::default();
        b.push("p");
        b.push("qr");
        assert_eq!(b.finish().to_vec(), vec!["p".to_string(), "qr".into()]);
    }

    #[test]
    fn debug_lists_cells() {
        let sb = StrBuffer::from_vec(vec!["a".into(), "b".into()]);
        assert_eq!(format!("{sb:?}"), r#"["a", "b"]"#);
    }
}
