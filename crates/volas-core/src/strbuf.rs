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

/// A contiguous UTF-8 string column buffer (Arrow `LargeUtf8`: i64 offsets).
#[derive(Clone)]
pub struct StrBuffer {
    /// `len + 1` monotonic byte offsets into `data`, always **zero-based**:
    /// `offsets[0] == 0` and `offsets[len] == data.len()`. A volas-built buffer is
    /// normalized by construction; an Arrow import of a *sliced* array (whose
    /// offsets are absolute, with a non-zero first offset) is re-based to 0 on the
    /// way in (`from_buffers` callers trim `data` to the live span and subtract the
    /// base), so this invariant holds for every `StrBuffer` — the unchecked
    /// accessors (`get_unchecked` / `iter`) and `extend` rely on it.
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

    /// Zero-copy from foreign Arrow `LargeUtf8` buffers (offsets + data + their guard).
    #[inline]
    pub fn from_buffers(offsets: Buffer<i64>, data: Buffer<u8>) -> Self {
        StrBuffer { offsets, data }
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
        // SAFETY: the data is built from valid UTF-8 (`from_iter`) or imported from
        // an Arrow Utf8 array (UTF-8 by spec), and offsets are on char boundaries.
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
        (0..self.len()).map(move |i| {
            // SAFETY: `i < len` ⟹ `i + 1 <= len < off.len()`, and offsets are monotonic
            // with `off[len] == data.len()`, so `off[i]..off[i + 1]` is in bounds of `data`.
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

/// Incremental builder (append / scatter paths).
#[derive(Default)]
pub struct StrBufferBuilder {
    offsets: Vec<i64>,
    data: Vec<u8>,
}

impl StrBufferBuilder {
    pub fn with_capacity(n: usize) -> Self {
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        StrBufferBuilder { offsets, data: Vec::new() }
    }
    #[inline]
    pub fn push(&mut self, s: &str) {
        self.data.extend_from_slice(s.as_bytes());
        self.offsets.push(self.data.len() as i64);
    }
    pub fn finish(mut self) -> StrBuffer {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
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
        let rebuilt = StrBuffer::from_buffers(offsets.clone(), data.clone());
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

    #[test]
    fn debug_lists_cells() {
        let sb = StrBuffer::from_vec(vec!["a".into(), "b".into()]);
        assert_eq!(format!("{sb:?}"), r#"["a", "b"]"#);
    }
}
