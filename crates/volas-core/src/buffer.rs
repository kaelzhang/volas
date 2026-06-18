//! [`Buffer<T>`]: a typed, contiguous value buffer that is either **owned**
//! (`Arc<Vec<T>>`, the volas-computed case) or **borrows foreign memory**
//! zero-copy (e.g. an Arrow / NumPy buffer kept alive by a guard).
//!
//! Reads go through a slice uniformly ([`Deref`] to `[T]`), so every kernel —
//! which already takes `&[T]` — is unaffected by where the bytes live. Mutation is
//! copy-on-write: a borrowed (or shared-owned) buffer materialises to a uniquely
//! owned `Vec` on the first write ([`Buffer::make_mut`]). This is the foundation
//! for zero-copy ingest, zero-copy slicing, and zero-copy export.

use std::any::Any;
use std::fmt;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

/// A typed contiguous buffer: owned or a zero-copy borrow of foreign memory.
pub enum Buffer<T> {
    /// volas-owned, `Arc`-shared (cheap clone, copy-on-write mutation).
    Owned(Arc<Vec<T>>),
    /// A zero-copy view of foreign memory (Arrow / NumPy). `guard` keeps the
    /// foreign allocation alive for exactly as long as this buffer does.
    Borrowed {
        ptr: NonNull<T>,
        len: usize,
        guard: Arc<dyn Any + Send + Sync>,
    },
}

// SAFETY: `Borrowed` is only constructed (via `from_foreign`) with a `Send + Sync`
// guard that owns the pointed-to memory; the elements are `T`, so the buffer is
// `Send`/`Sync` exactly when `T` is.
unsafe impl<T: Send + Sync> Send for Buffer<T> {}
unsafe impl<T: Send + Sync> Sync for Buffer<T> {}

impl<T> Buffer<T> {
    /// An owned buffer from a `Vec` (the volas-computed path).
    #[inline]
    pub fn from_vec(v: Vec<T>) -> Self {
        Buffer::Owned(Arc::new(v))
    }

    /// A zero-copy borrow of foreign memory.
    ///
    /// # Safety
    /// `ptr` must point to `len` initialised, contiguous `T` that stay valid and
    /// immutable for as long as `guard` is alive; `guard` must own that allocation.
    #[inline]
    pub unsafe fn from_foreign(ptr: NonNull<T>, len: usize, guard: Arc<dyn Any + Send + Sync>) -> Self {
        Buffer::Borrowed { ptr, len, guard }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Buffer::Owned(a) => a.len(),
            Buffer::Borrowed { len, .. } => *len,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            Buffer::Owned(a) => a.as_slice(),
            // SAFETY: the invariant of `from_foreign` (guard keeps `len` valid `T` alive).
            Buffer::Borrowed { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(ptr.as_ptr(), *len)
            },
        }
    }
}

impl<T: Clone> Buffer<T> {
    /// Mutable owned access, copy-on-write: materialises a borrowed buffer (or an
    /// aliased owned one) to a uniquely-owned `Vec` first, then hands out `&mut`.
    #[inline]
    pub fn make_mut(&mut self) -> &mut Vec<T> {
        if matches!(self, Buffer::Borrowed { .. }) {
            let owned = self.as_slice().to_vec();
            *self = Buffer::Owned(Arc::new(owned));
        }
        match self {
            Buffer::Owned(a) => Arc::make_mut(a),
            Buffer::Borrowed { .. } => unreachable!("materialised to Owned above"), // LCOV_EXCL_LINE
        }
    }

    /// Consume into an owned `Vec` — no copy when uniquely owned, else one copy.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Buffer::Owned(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
            Buffer::Borrowed { .. } => self.as_slice().to_vec(),
        }
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Clone> Clone for Buffer<T> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Buffer::Owned(a) => Buffer::Owned(Arc::clone(a)),
            Buffer::Borrowed { ptr, len, guard } => Buffer::Borrowed {
                ptr: *ptr,
                len: *len,
                guard: Arc::clone(guard),
            },
        }
    }
}

impl<T: PartialEq> PartialEq for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl<T> From<Vec<T>> for Buffer<T> {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        Buffer::from_vec(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Borrowed` buffer that points into `v`'s data, with `v` (as an `Arc`) as
    /// the guard keeping it alive — the shape an Arrow import produces.
    fn borrowed(v: Vec<f64>) -> Buffer<f64> {
        let arc = Arc::new(v);
        let ptr = NonNull::new(arc.as_slice().as_ptr() as *mut f64).unwrap();
        let len = arc.len();
        // SAFETY: `arc` owns the `len` f64 and is moved in as the guard.
        unsafe { Buffer::from_foreign(ptr, len, arc) }
    }

    #[test]
    fn owned_basics() {
        let b = Buffer::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(b.len(), 3);
        assert!(!b.is_empty());
        assert_eq!(b.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(&*b, &[1.0, 2.0, 3.0]); // Deref
        assert_eq!(Buffer::<f64>::from_vec(vec![]).is_empty(), true);
        assert_eq!(Buffer::from(vec![1.0]).into_vec(), vec![1.0]); // From + into_vec (no copy)
        assert_eq!(format!("{:?}", b), "[1.0, 2.0, 3.0]"); // Debug
        assert_eq!(b, b.clone()); // Clone + PartialEq (Owned)
    }

    #[test]
    fn borrowed_reads_zero_copy() {
        let b = borrowed(vec![1.0, 2.0, 3.0]);
        assert_eq!(b.len(), 3);
        assert!(!b.is_empty());
        assert_eq!(b.as_slice(), &[1.0, 2.0, 3.0]);
        let c = b.clone(); // Clone (Borrowed): shares the guard, no copy
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(b, c); // PartialEq across two borrowed views
        assert_eq!(format!("{:?}", b), "[1.0, 2.0, 3.0]");
    }

    #[test]
    fn borrowed_make_mut_materialises_cow() {
        let mut b = borrowed(vec![1.0, 2.0]);
        assert!(matches!(b, Buffer::Borrowed { .. }));
        b.make_mut().push(3.0); // copy-on-write: becomes Owned
        assert!(matches!(b, Buffer::Owned(_)));
        assert_eq!(b.as_slice(), &[1.0, 2.0, 3.0]);
        // an owned buffer's make_mut grows in place
        b.make_mut().push(4.0);
        assert_eq!(b.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn borrowed_into_vec_copies() {
        assert_eq!(borrowed(vec![7.0, 8.0]).into_vec(), vec![7.0, 8.0]);
        // owned-but-shared into_vec clones (try_unwrap fails)
        let a = Buffer::from_vec(vec![5.0]);
        let _alias = a.clone();
        assert_eq!(a.into_vec(), vec![5.0]);
    }
}
