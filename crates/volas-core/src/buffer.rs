//! [`Buffer<T>`]: a typed, contiguous value buffer that is either **owned**
//! (`Arc<Vec<T>>`, the volas-computed case) or **holds a guarded range**
//! zero-copy (e.g. an Arrow / NumPy buffer, or immutable copied slice slab).
//!
//! Reads go through a slice uniformly ([`Deref`] to `[T]`), so every kernel —
//! which already takes `&[T]` — is unaffected by where the bytes live. Mutation is
//! copy-on-write: a borrowed (or shared-owned) buffer materialises to a uniquely
//! owned `Vec` on the first write ([`Buffer::make_mut`]). This is the foundation
//! for zero-copy ingest, zero-copy slicing, and zero-copy export.
//!
//! A float buffer also remembers whether it holds a NaN ([`Buffer::has_nan`]): the
//! fact is computed on the first query and kept until the values change. It lives in
//! the buffer value, not in the shared allocation — the bytes behind an `Arc` never
//! change while shared, and every write goes through `&mut self` — so a clone carries
//! the fact, and [`Buffer::make_mut`] (the one way to write) forgets it. The tracked
//! writes ([`Buffer::append`], [`Buffer::set`]) keep it current in O(1) instead.
//!
//! The fact is only ever remembered for bytes volas owns. A zero-copy view of foreign
//! memory (an Arrow import) belongs to a producer that may still write it, so it is
//! scanned on every query and never remembered; its first write copies it into an
//! owned buffer, which may then remember. What volas lends out zero-copy (DLPack,
//! Arrow) is read-only to the consumer, so owned bytes cannot change behind it.

use std::any::Any;
use std::fmt;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Where a buffer's bytes live.
enum Data<T> {
    /// volas-owned, `Arc`-shared (cheap clone, copy-on-write mutation).
    Owned(Arc<Vec<T>>),
    /// A guarded zero-copy range of foreign or immutable Rust-owned memory.
    Borrowed {
        ptr: NonNull<T>,
        len: usize,
        guard: Arc<dyn Any + Send + Sync>,
    },
}

/// Whether the values hold a NaN: not yet known, known not to, or known to.
const UNKNOWN: u8 = 0;
const NO_NAN: u8 = 1;
const SOME_NAN: u8 = 2;
// Unknown facts over checked immutable Rust-owned ranges may be remembered.
const IMMUTABLE_UNKNOWN: u8 = 3;

/// Where a buffer keeps its NaN fact: an atomic fact for a float element (with an unknown
/// immutable-range marker; a query
/// records it through `&self` on a `Sync` buffer; every thread derives the same value
/// from the same immutable bytes, so `Relaxed` suffices), nothing at all for an
/// element that cannot be NaN.
pub trait NanFact: Send + Sync {
    #[doc(hidden)]
    fn new(state: u8) -> Self;
    #[doc(hidden)]
    fn get(&self) -> u8;
    #[doc(hidden)]
    fn set(&self, state: u8);
}

impl NanFact for () {
    #[inline]
    fn new(_: u8) -> Self {}
    #[inline]
    fn get(&self) -> u8 {
        UNKNOWN
    }
    #[inline]
    fn set(&self, _: u8) {}
}

impl NanFact for AtomicU8 {
    #[inline]
    fn new(state: u8) -> Self {
        AtomicU8::new(state)
    }
    #[inline]
    fn get(&self) -> u8 {
        self.load(Ordering::Relaxed)
    }
    #[inline]
    fn set(&self, state: u8) {
        self.store(state, Ordering::Relaxed);
    }
}

/// A buffer element, with where its NaN fact lives (see [`NanFact`]).
pub trait Element {
    #[doc(hidden)]
    type Nan: NanFact;
}

macro_rules! element {
    ($nan:ty: $($t:ty),*) => { $(impl Element for $t { type Nan = $nan; })* };
}
element!(AtomicU8: f64, f32);
element!((): i64, i32, u8, bool);

/// A typed contiguous buffer: owned or a zero-copy borrow of foreign memory.
pub struct Buffer<T: Element> {
    data: Data<T>,
    /// The memoized NaN fact (`UNKNOWN` / `NO_NAN` / `SOME_NAN`) of a float buffer.
    nan: T::Nan,
}

// SAFETY: `Borrowed` is only constructed (via `from_foreign`) with a `Send + Sync`
// guard that owns the pointed-to memory; the elements are `T`, so the buffer is
// `Send`/`Sync` exactly when `T` is (the NaN fact is `Send + Sync` by its bound).
unsafe impl<T: Element + Send + Sync> Send for Buffer<T> {}
unsafe impl<T: Element + Send + Sync> Sync for Buffer<T> {}

impl<T: Element> Buffer<T> {
    fn with(data: Data<T>, nan: u8) -> Self {
        Buffer {
            data,
            nan: T::Nan::new(nan),
        }
    }

    /// An owned buffer from a `Vec` (the volas-computed path).
    #[inline]
    pub fn from_vec(v: Vec<T>) -> Self {
        Buffer::with(Data::Owned(Arc::new(v)), UNKNOWN)
    }

    /// A zero-copy borrow of foreign memory.
    ///
    /// # Safety
    /// `ptr` must point to `len` initialised, contiguous `T` that stay valid and
    /// immutable for as long as `guard` is alive; `guard` must own that allocation.
    #[inline]
    pub unsafe fn from_foreign(
        ptr: NonNull<T>,
        len: usize,
        guard: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Buffer::with(Data::Borrowed { ptr, len, guard }, UNKNOWN)
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.data {
            Data::Owned(a) => a.len(),
            Data::Borrowed { len, .. } => *len,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match &self.data {
            Data::Owned(a) => a.as_slice(),
            // SAFETY: `from_foreign` or checked `from_shared_range` gives a guard
            // that keeps these `len` initialized, immutable values alive.
            Data::Borrowed { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(ptr.as_ptr(), *len)
            },
        }
    }

    /// Pointer to the first element (valid for [`len`](Self::len) reads).
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    /// Whether the bytes are a guarded zero-copy view.
    #[inline]
    pub fn is_borrowed(&self) -> bool {
        matches!(self.data, Data::Borrowed { .. })
    }

    #[inline]
    fn nan_state(&self) -> u8 {
        self.nan.get()
    }

    #[inline]
    fn set_nan_state(&self, state: u8) {
        self.nan.set(state);
    }
}

impl<T: Element + Send + Sync + 'static> Buffer<T> {
    /// A checked zero-copy range of an immutable Rust-owned allocation. The guard
    /// owns the Vec; any external `Arc::make_mut` copies while this view exists.
    /// These immutable ranges remember their own NaN fact and materialize only their
    /// selected range on a write. Empty ranges are valid; invalid ranges are rejected.
    pub fn from_shared_range(owner: Arc<Vec<T>>, range: std::ops::Range<usize>) -> Option<Self> {
        let values = owner.get(range)?;
        // A slice reference always has a non-null aligned pointer, including when empty.
        let ptr = NonNull::new(values.as_ptr() as *mut T).expect("slice pointer");
        let len = values.len();
        Some(Buffer::with(Data::Borrowed { ptr, len, guard: owner }, IMMUTABLE_UNKNOWN))
    }

    /// Preserve the fact from a NaN-free source when copying one selected range.
    pub(crate) fn from_shared_nan_free_range(owner: Arc<Vec<T>>, range: std::ops::Range<usize>) -> Option<Self> {
        let buffer = Self::from_shared_range(owner, range)?;
        buffer.set_nan_state(NO_NAN);
        Some(buffer)
    }

    /// An owner handle that keeps this buffer's allocation alive — the keep-alive a
    /// zero-copy foreign export (Arrow / DLPack) hands the consumer alongside the raw
    /// [`as_ptr`](Self::as_ptr) pointer, so the bytes outlive this `Buffer`. Cloning is
    /// `O(1)` (an `Arc` bump) and exposes no volas-internal type to the export crate.
    #[inline]
    pub fn keepalive(&self) -> Arc<dyn Any + Send + Sync> {
        match &self.data {
            Data::Owned(a) => a.clone(),
            Data::Borrowed { guard, .. } => guard.clone(),
        }
    }
}

impl<T: Element + Clone> Buffer<T> {
    /// Mutable owned access, copy-on-write: materialises a borrowed buffer (or an
    /// aliased owned one) to a uniquely-owned `Vec` first, then hands out `&mut`.
    /// The values may change, so the NaN fact is forgotten.
    #[inline]
    pub fn make_mut(&mut self) -> &mut Vec<T> {
        self.set_nan_state(UNKNOWN);
        if self.is_borrowed() {
            let owned = self.as_slice().to_vec();
            self.data = Data::Owned(Arc::new(owned));
        }
        match &mut self.data {
            Data::Owned(a) => Arc::make_mut(a),
            Data::Borrowed { .. } => unreachable!("materialised to Owned above"), // LCOV_EXCL_LINE
        }
    }

    /// Consume into an owned `Vec` — no copy when uniquely owned, else one copy.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        match self.data {
            Data::Owned(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
            Data::Borrowed { .. } => self.as_slice().to_vec(),
        }
    }

    /// The rows `[start, end)` as a fresh owned buffer. A slice of a buffer known to
    /// hold no NaN holds none either.
    pub fn slice(&self, start: usize, end: usize) -> Buffer<T> {
        let nan = if self.nan_state() == NO_NAN {
            NO_NAN
        } else {
            UNKNOWN
        };
        Buffer::with(
            Data::Owned(Arc::new(self.as_slice()[start..end].to_vec())),
            nan,
        )
    }
}

/// A float element: one that can be NaN.
pub trait Float: Element + Copy {
    /// Whether this value is NaN.
    fn is_nan_value(self) -> bool;
}

impl Float for f64 {
    #[inline]
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
}

impl Float for f32 {
    #[inline]
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
}

/// Whether any value of `x` is NaN — branch-free over fixed chunks so the scan
/// vectorises.
fn scan_nan<T: Float>(x: &[T]) -> bool {
    let mut chunks = x.chunks_exact(8);
    for c in &mut chunks {
        if c.iter().fold(false, |acc, &v| acc | v.is_nan_value()) {
            return true;
        }
    }
    chunks.remainder().iter().any(|&v| v.is_nan_value())
}

impl<T: Float> Buffer<T> {
    /// Whether any value is NaN. Scans once, then remembers until the values change —
    /// except for a zero-copy view of foreign memory, which its owner may write behind
    /// volas's back: that is scanned on every query, never remembered.
    pub fn has_nan(&self) -> bool {
        let state = self.nan_state();
        match state {
            NO_NAN => false,
            SOME_NAN => true,
            _ => {
                let found = scan_nan(self.as_slice());
                if state == IMMUTABLE_UNKNOWN || !self.is_borrowed() {
                    self.set_nan_state(if found { SOME_NAN } else { NO_NAN });
                }
                found
            }
        }
    }

    /// The NaN fact if it is already known — never scans.
    pub fn known_has_nan(&self) -> Option<bool> {
        match self.nan_state() {
            NO_NAN => Some(false),
            SOME_NAN => Some(true),
            _ => None,
        }
    }

    /// Copy foreign bytes in before a tracked write, learning their NaN fact from the
    /// copy: it is volas's own memory now, so the fact holds until the next write.
    #[inline]
    fn own(&mut self) {
        if self.is_borrowed() {
            self.copy_in();
        }
    }

    #[cold]
    #[inline(never)]
    fn copy_in(&mut self) {
        let owned = self.as_slice().to_vec();
        let state = if scan_nan(&owned) { SOME_NAN } else { NO_NAN };
        self.data = Data::Owned(Arc::new(owned));
        self.set_nan_state(state);
    }

    /// Append `other`'s values (copy-on-write), keeping the NaN fact in O(`other`): it
    /// holds a NaN once either part does, and is otherwise what it was. (The states are
    /// ordered `UNKNOWN < NO_NAN < SOME_NAN`, so that is one `max`.)
    pub fn append(&mut self, other: &[T]) {
        self.own();
        let added = if scan_nan(other) { SOME_NAN } else { UNKNOWN };
        let state = self.nan_state().max(added);
        self.make_mut().extend_from_slice(other);
        self.set_nan_state(state);
    }

    /// Append `len` copies of `value` (copy-on-write), keeping the NaN fact.
    pub fn append_repeated(&mut self, value: T, len: usize) {
        self.own();
        let added = if len > 0 && value.is_nan_value() {
            SOME_NAN
        } else {
            UNKNOWN
        };
        let state = self.nan_state().max(added);
        self.make_mut().extend(std::iter::repeat_n(value, len));
        self.set_nan_state(state);
    }

    /// Overwrite the value at `i` (copy-on-write), keeping the NaN fact where one
    /// write decides it: a NaN written means some NaN; a number written over a
    /// NaN-free buffer keeps it NaN-free; overwriting an old NaN may remove the
    /// last one, so only that case forgets a known NaN.
    pub fn set(&mut self, i: usize, value: T) {
        self.scatter(&[i], |_| value);
    }

    /// Overwrite the value at each of `positions` with `value(k)` for the `k`-th of
    /// them (copy-on-write; a later repeat of a position wins), in O(`positions`) on a
    /// uniquely-owned buffer, keeping the NaN fact where the writes decide it, as
    /// [`set`](Self::set) does for one — judged by the values the written positions
    /// hold afterwards, never by a value a later repeat replaced: a written position
    /// left NaN means some NaN; numbers left over a NaN-free buffer keep it NaN-free;
    /// a known NaN is kept unless the writes cover an old NaN and might remove
    /// the last one.
    pub fn scatter(&mut self, positions: &[usize], value: impl Fn(usize) -> T) {
        self.own();
        let before = self.nan_state();
        let w = self.make_mut();
        // Inspect the old cells before any write: a repeated position may hold a
        // temporary NaN which a later write removes, without touching an old NaN.
        let covered_nan = before == SOME_NAN
            && positions.iter().any(|&p| w[p].is_nan_value());
        for (k, &p) in positions.iter().enumerate() {
            w[p] = value(k);
        }
        let left_nan = positions.iter().any(|&p| w[p].is_nan_value());
        self.set_nan_state(if left_nan {
            SOME_NAN
        } else if covered_nan {
            UNKNOWN
        } else {
            before
        });
    }
}

impl<T: Element> Deref for Buffer<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Element + Clone> Clone for Buffer<T> {
    /// Shares the bytes (an `Arc` bump) and carries the NaN fact.
    #[inline]
    fn clone(&self) -> Self {
        let data = match &self.data {
            Data::Owned(a) => Data::Owned(Arc::clone(a)),
            Data::Borrowed { ptr, len, guard } => Data::Borrowed {
                ptr: *ptr,
                len: *len,
                guard: Arc::clone(guard),
            },
        };
        Buffer::with(data, self.nan_state())
    }
}

impl<T: Element + PartialEq> PartialEq for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Element + fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl<T: Element> From<Vec<T>> for Buffer<T> {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        Buffer::from_vec(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_ranges_keep_only_their_values_independent_of_owner_writes() {
        let mut owner = Arc::new(vec![9.0, 1.0, 2.0, 8.0]);
        let mut range = Buffer::from_shared_range(Arc::clone(&owner), 1..3).unwrap();
        let old = Arc::downgrade(&owner);
        Arc::make_mut(&mut owner)[1] = 7.0;
        assert_eq!(range.as_slice(), &[1.0, 2.0]);
        assert!(!range.has_nan());
        assert_eq!(range.known_has_nan(), Some(false));
        range.set(0, 3.0);
        assert!(!range.is_borrowed());
        assert_eq!(range.as_slice(), &[3.0, 2.0]);
        assert_eq!(owner.as_slice(), &[9.0, 7.0, 2.0, 8.0]);
        assert!(old.upgrade().is_none());
    }

    #[test]
    fn shared_ranges_validate_bounds_and_keep_exports_alive() {
        let owner = Arc::new(vec![1.0, f64::NAN]);
        assert!(Buffer::from_shared_range(Arc::clone(&owner), 2..2).unwrap().is_empty());
        assert!(Buffer::from_shared_range(Arc::clone(&owner), 2..3).is_none());
        let reversed = std::ops::Range { start: 1, end: 0 };
        assert!(Buffer::from_shared_range(Arc::clone(&owner), reversed).is_none());
        let range = Buffer::from_shared_range(Arc::clone(&owner), 1..2).unwrap();
        assert!(range.has_nan());
        assert_eq!(range.known_has_nan(), Some(true));
        let weak = Arc::downgrade(&owner);
        let export = range.keepalive();
        drop(owner);
        drop(range);
        assert!(weak.upgrade().unwrap()[1].is_nan());
        drop(export);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn shared_range_nan_facts_describe_only_the_selected_range() {
        let owner = Arc::new(vec![f64::NAN, 1.0, 2.0]);
        let missing = Buffer::from_shared_range(Arc::clone(&owner), 0..1).unwrap();
        let clean = Buffer::from_shared_range(Arc::clone(&owner), 1..3).unwrap();
        assert_eq!(missing.known_has_nan(), None);
        assert_eq!(clean.known_has_nan(), None);
        assert!(missing.has_nan());
        assert!(!clean.has_nan());
        assert_eq!(missing.clone().known_has_nan(), Some(true));
        assert_eq!(clean.clone().known_has_nan(), Some(false));
        let known = Buffer::from_shared_nan_free_range(owner, 1..2).unwrap();
        assert_eq!(known.known_has_nan(), Some(false));
    }

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
        assert!(Buffer::<f64>::from_vec(vec![]).is_empty());
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
        assert!(c.is_borrowed());
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(b, c); // PartialEq across two borrowed views
        assert_eq!(format!("{:?}", b), "[1.0, 2.0, 3.0]");
    }

    #[test]
    fn borrowed_make_mut_materialises_cow() {
        let mut b = borrowed(vec![1.0, 2.0]);
        assert!(b.is_borrowed());
        b.make_mut().push(3.0); // copy-on-write: becomes Owned
        assert!(!b.is_borrowed());
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

    // --- the NaN fact: a stale `false` would hide a missing row, so every write
    // --- path must leave it either right or unknown.

    const NAN: f64 = f64::NAN;

    #[test]
    fn the_nan_fact_is_computed_once_and_remembered() {
        let b = Buffer::from_vec(vec![1.0; 20]);
        assert_eq!(b.known_has_nan(), None);
        assert!(!b.has_nan());
        assert_eq!(b.known_has_nan(), Some(false));
        let mut v = vec![1.0; 20];
        v[17] = NAN;
        let c = Buffer::from_vec(v);
        assert!(c.has_nan());
        assert_eq!(c.known_has_nan(), Some(true));
        assert!(c.has_nan()); // remembered
        assert!(borrowed(vec![1.0, NAN]).has_nan());
        assert!(!Buffer::from_vec(vec![f32::INFINITY]).has_nan()); // an infinity is a value
        assert!(Buffer::from_vec(vec![f32::NAN]).has_nan());
    }

    /// Foreign bytes can change behind the buffer, so their fact is never remembered —
    /// and once copied in by a write, the owned copy may remember again.
    #[test]
    fn a_borrowed_buffer_never_remembers() {
        let arc = Arc::new(vec![1.0, 2.0]);
        let ptr = NonNull::new(arc.as_ptr() as *mut f64).unwrap();
        // SAFETY: `arc` owns the 2 f64 and is the guard.
        let b = unsafe { Buffer::from_foreign(ptr, 2, arc.clone()) };
        assert!(!b.has_nan());
        assert_eq!(b.known_has_nan(), None);
        // the producer writes its bytes: the next query sees it
        // SAFETY: test-only stand-in for a producer writing the shared allocation.
        unsafe { *ptr.as_ptr() = NAN };
        assert!(b.has_nan());
        assert_eq!(b.clone().known_has_nan(), None);
        assert_eq!(b.slice(1, 2).known_has_nan(), None);
        let mut owned = b.clone();
        owned.append(&[3.0]); // copied in: owned now, its fact taken from the copy
        assert!(!owned.is_borrowed());
        assert_eq!(owned.known_has_nan(), Some(true));
    }

    /// A tracked write copies foreign bytes in and learns their fact from the copy —
    /// volas's own memory now — so an imported column appended live gets the clean
    /// fast path.
    #[test]
    fn a_tracked_write_learns_the_fact_of_the_foreign_bytes_it_copies() {
        let mut a = borrowed(vec![1.0, 2.0]);
        a.append(&[3.0]);
        assert_eq!(a.known_has_nan(), Some(false));
        let mut r = borrowed(vec![1.0]);
        r.append_repeated(2.0, 2);
        assert_eq!(r.known_has_nan(), Some(false));
        let mut s = borrowed(vec![1.0, 2.0]);
        s.set(0, 5.0);
        assert_eq!(s.known_has_nan(), Some(false));
        let mut n = borrowed(vec![NAN, 2.0]);
        n.set(0, 5.0); // the copy held the NaN being overwritten: forget it
        assert_eq!(n.known_has_nan(), None);
        assert!(!n.has_nan());
        let mut m = borrowed(vec![NAN]);
        m.append(&[1.0]);
        assert_eq!(m.known_has_nan(), Some(true));
    }

    #[test]
    fn make_mut_forgets_the_fact() {
        let mut b = Buffer::from_vec(vec![1.0, 2.0]);
        assert!(!b.has_nan());
        b.make_mut()[0] = NAN;
        assert_eq!(b.known_has_nan(), None);
        assert!(b.has_nan());
        // copy-on-write of a borrowed buffer, too
        let mut f = borrowed(vec![1.0]);
        assert!(!f.has_nan());
        f.make_mut().push(NAN);
        assert!(f.has_nan());
    }

    #[test]
    fn a_clone_carries_the_fact_and_a_write_to_one_side_does_not_leak() {
        let a = Buffer::from_vec(vec![1.0, 2.0]);
        assert!(!a.has_nan());
        let mut b = a.clone();
        assert_eq!(b.known_has_nan(), Some(false));
        b.set(0, NAN); // copy-on-write: `a` keeps its bytes and its fact
        assert!(b.has_nan());
        assert!(!a.has_nan());
        assert_eq!(a.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn append_keeps_the_fact() {
        let mut b = Buffer::from_vec(vec![1.0]);
        b.append(&[2.0]);
        assert_eq!(b.known_has_nan(), None); // unknown stays unknown without a NaN
        assert!(!b.has_nan());
        b.append(&[3.0, 4.0]);
        assert_eq!(b.known_has_nan(), Some(false));
        b.append(&[NAN]);
        assert_eq!(b.known_has_nan(), Some(true));
        b.append(&[5.0]);
        assert_eq!(b.known_has_nan(), Some(true));
        let mut u = Buffer::from_vec(vec![1.0]);
        u.append(&[NAN]);
        assert_eq!(u.known_has_nan(), Some(true));
    }

    #[test]
    fn append_repeated_keeps_the_fact() {
        let mut b = Buffer::from_vec(vec![1.0]);
        assert!(!b.has_nan());
        b.append_repeated(2.0, 3);
        assert_eq!(b.known_has_nan(), Some(false));
        b.append_repeated(NAN, 0);
        assert_eq!(b.known_has_nan(), Some(false));
        b.append_repeated(NAN, 2);
        assert_eq!(b.known_has_nan(), Some(true));
        let mut u = Buffer::from_vec(vec![1.0]);
        u.append_repeated(2.0, 1);
        assert_eq!(u.known_has_nan(), None);
        let mut s = Buffer::from_vec(vec![NAN]);
        assert!(s.has_nan());
        s.append_repeated(1.0, 1);
        assert_eq!(s.known_has_nan(), Some(true));
    }

    #[test]
    fn set_keeps_the_fact_where_one_write_decides_it() {
        let mut b = Buffer::from_vec(vec![1.0, 2.0]);
        assert!(!b.has_nan());
        b.set(1, 3.0);
        assert_eq!(b.known_has_nan(), Some(false));
        b.set(0, NAN);
        assert_eq!(b.known_has_nan(), Some(true));
        b.set(0, 1.0); // the only NaN may be gone: forget, never claim either way
        assert_eq!(b.known_has_nan(), None);
        assert!(!b.has_nan());
        let mut u = Buffer::from_vec(vec![NAN, 1.0]);
        u.set(1, 2.0); // unknown stays unknown
        assert_eq!(u.known_has_nan(), None);
        assert!(u.has_nan());
    }

    #[test]
    fn scatter_writes_many_cells_in_place_keeping_the_fact() {
        let mut b = Buffer::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(!b.has_nan());
        let at = b.as_ptr();
        b.scatter(&[3, 0, 3], |k| [7.0, 8.0, 9.0][k]); // a later repeat wins
        assert_eq!((b.as_slice(), b.as_ptr()), (&[8.0, 2.0, 3.0, 9.0][..], at));
        assert_eq!(b.known_has_nan(), Some(false));
        b.scatter(&[1, 2], |k| if k == 0 { NAN } else { 0.0 });
        assert_eq!(b.known_has_nan(), Some(true));
        b.scatter(&[1], |_| 1.0); // the only NaN may be gone: forget
        assert_eq!(b.known_has_nan(), None);
        let mut f = borrowed(vec![1.0, NAN]); // foreign bytes are copied in first
        f.scatter(&[0], |_| 5.0);
        assert!(!f.is_borrowed());
        assert_eq!(f.known_has_nan(), Some(true));
        assert!(f.has_nan());
    }

    #[test]
    fn scatter_judges_the_fact_by_the_values_a_repeated_position_keeps() {
        // a NaN written and then overwritten at the same position leaves no NaN
        let mut b = Buffer::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(!b.has_nan());
        b.scatter(&[1, 1], |k| [NAN, 5.0][k]);
        assert_eq!(b.as_slice(), &[1.0, 5.0, 3.0]);
        assert_eq!(b.known_has_nan(), Some(false));
        // the other order keeps the NaN
        b.scatter(&[2, 2], |k| [6.0, NAN][k]);
        assert_eq!(b.known_has_nan(), Some(true));
        // over a buffer holding a NaN, numbers written may have removed the last one
        b.scatter(&[2, 0, 2], |k| [NAN, 7.0, 8.0][k]);
        assert_eq!(b.known_has_nan(), None);
        assert!(!b.has_nan());
    }

    #[test]
    fn a_slice_of_a_nan_free_buffer_is_nan_free() {
        let b = Buffer::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(b.slice(0, 2).known_has_nan(), None);
        assert!(!b.has_nan());
        let s = b.slice(1, 3);
        assert_eq!(s.as_slice(), &[2.0, 3.0]);
        assert_eq!(s.known_has_nan(), Some(false));
        let n = Buffer::from_vec(vec![NAN, 1.0]);
        assert!(n.has_nan());
        let t = n.slice(1, 2); // the NaN may be outside the slice
        assert_eq!(t.known_has_nan(), None);
        assert!(!t.has_nan());
    }
}
