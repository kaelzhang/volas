//! Immutable words whose encoding belongs entirely to the directive owner.
//! A batch may store many states in one allocation; core only owns their checked
//! ranges and shared lifetime, never the meaning of any word.

use std::fmt;
use std::ops::{Deref, Range};
use std::sync::Arc;

#[derive(Clone)]
pub struct StateWords {
    words: Arc<Vec<f64>>,
    range: Range<usize>,
}

impl StateWords {
    /// Publish a checked range of shared words. Oversized external arenas are
    /// bounded here; native passes use the chunk arena directly. Invalid ranges
    /// are rejected before a value can be constructed through this safe API.
    pub fn from_shared(words: Arc<Vec<f64>>, range: Range<usize>) -> Option<Self> {
        if range.start > range.end || range.end > words.len() { return None; }
        // External shared arenas enter the same bounded ownership model at
        // publication. The native builder writes directly into these chunks.
        if words.capacity() > StateWordArena::WORDS.max(range.len()) {
            return Some(words[range].to_vec().into());
        }
        Some(Self { words, range })
    }
}

impl From<Arc<Vec<f64>>> for StateWords {
    fn from(words: Arc<Vec<f64>>) -> Self {
        let end = words.len();
        Self::from_shared(words, 0..end).expect("whole state")
    }
}

impl From<Vec<f64>> for StateWords {
    fn from(words: Vec<f64>) -> Self {
        Self::from(Arc::new(words))
    }
}

impl Deref for StateWords {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        &self.words[self.range.clone()]
    }
}

impl fmt::Debug for StateWords {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl PartialEq for StateWords {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_words_preserve_ranges_values_and_lifetime() {
        let arena = Arc::new(vec![1.0, -0.0, f64::from_bits(0x7ff8000000000123), 4.0]);
        let selected = StateWords::from_shared(Arc::clone(&arena), 1..3).unwrap();
        assert_eq!(selected.as_ptr(), arena[1..].as_ptr());
        let clone = selected.clone();
        drop(arena);
        drop(selected);
        assert_eq!(clone[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(clone[1].to_bits(), 0x7ff8000000000123);
        let owned = StateWords::from(vec![2.0, 3.0]);
        assert_eq!(format!("{owned:?}"), "[2.0, 3.0]");
        assert_eq!(owned, StateWords::from(vec![2.0, 3.0]));
    }

    #[test]
    fn safe_shared_words_reject_invalid_ranges_and_allow_empty_ones() {
        let arena = Arc::new(vec![1.0, 2.0]);
        assert!(StateWords::from_shared(Arc::clone(&arena), 1..3).is_none());
        assert!(StateWords::from_shared(Arc::clone(&arena), Range { start: 2, end: 1 }).is_none());
        assert!(StateWords::from_shared(arena, 2..2).unwrap().is_empty());
    }
}

/// A publication pass's opaque word arena. Completed chunks are immutable after
/// `finish`; no state crosses a chunk boundary. A single wide state is its own chunk.
#[doc(hidden)]
#[derive(Default)]
pub struct StateWordArena {
    chunks: Vec<Vec<f64>>,
    ranges: Vec<(usize, Range<usize>)>,
}

impl StateWordArena {
    pub(super) const WORDS: usize = 4096;

    /// Append one state into the open tail. `width` is a reservation hint: warm
    /// encodings may grow. The writer may only append; failure publishes nothing.
    pub fn write(&mut self, width: usize, write: impl FnOnce(&mut Vec<f64>) -> bool) -> bool {
        if self.chunks.last().is_none_or(|tail| !tail.is_empty() && tail.len() + width > Self::WORDS) {
            // Reserve whole frames, not padding that no complete state can use.
            // A wide state owns one chunk; ordinary chunks stay within WORDS.
            let width = width.max(1);
            let capacity = width * (Self::WORDS / width).max(1);
            self.chunks.push(Vec::with_capacity(capacity));
        }
        let slot = self.chunks.len() - 1;
        let tail = &mut self.chunks[slot];
        let start = tail.len();
        if !write(tail) {
            tail.truncate(start);
            if tail.capacity() > Self::WORDS { tail.shrink_to_fit(); }
            return false;
        }
        let end = tail.len();
        if end > Self::WORDS || tail.capacity() > Self::WORDS {
            self.finish_growth(slot, start);
        } else { self.ranges.push((slot, start..end)); }
        true
    }

    /// A wider warm frame can exceed its seed-width hint. Repair only the open
    /// chunk before publication; completed chunks and derived frames never copy.
    #[cold]
    #[inline(never)]
    fn finish_growth(&mut self, slot: usize, start: usize) {
        let tail = &mut self.chunks[slot];
        if start > 0 && tail.len() > Self::WORDS {
            let words = tail.split_off(start);
            if tail.capacity() > Self::WORDS { tail.shrink_to_fit(); }
            let end = words.len();
            let slot = self.chunks.len();
            self.chunks.push(words);
            self.ranges.push((slot, 0..end));
        } else {
            tail.shrink_to_fit();
            self.ranges.push((slot, start..tail.len()));
        }
    }

    /// States in publication order, before their chunks become shared.
    pub fn iter(&self) -> impl Iterator<Item = &[f64]> {
        self.ranges.iter().map(|(slot, range)| &self.chunks[*slot][range.clone()])
    }

    /// Seal the pass's chunks without copying any words.
    pub fn finish(self) -> Vec<StateWords> {
        let chunks: Vec<_> = self.chunks.into_iter().map(Arc::new).collect();
        self.ranges.into_iter().map(|(slot, range)| StateWords {
            words: Arc::clone(&chunks[slot]), range,
        }).collect()
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn publication_chunks_share_words_and_release_unreachable_chunks() {
        let mut arena = StateWordArena::default();
        let bits = [(-0.0f64).to_bits(), 0x7ff8000000000123];
        let half = StateWordArena::WORDS / 2;
        for k in 0..5 {
            assert!(arena.write(half, |words| {
                words.extend((0..half).map(|i| f64::from_bits(bits[(i + k) % 2])));
                true
            }));
        }
        assert_eq!(arena.iter().count(), 5);
        let states = arena.finish();
        assert!(Arc::ptr_eq(&states[0].words, &states[1].words));
        assert!(!Arc::ptr_eq(&states[1].words, &states[2].words));
        let unreachable = Arc::downgrade(&states[0].words);
        let kept = states[2].clone();
        let pointer = kept.as_ptr();
        drop(states);
        assert!(unreachable.upgrade().is_none());
        assert_eq!(kept.as_ptr(), pointer);
        assert_eq!(kept[0].to_bits(), bits[0]);
        assert_eq!(kept[1].to_bits(), bits[1]);
    }

    #[test]
    fn wide_states_failed_publications_and_external_capacity_are_bounded() {
        assert!(StateWordArena::default().finish().is_empty());
        let mut arena = StateWordArena::default();
        assert!(!arena.write(1, |words| { words.push(99.0); false }));
        assert!(arena.write(10_000, |words| { words.resize(words.len() + 10_000, 7.0); true }));
        assert!(arena.write(1, |words| { words.push(-0.0); true }));
        let states = arena.finish();
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].len(), 10_000);
        assert_eq!(states[1][0].to_bits(), (-0.0f64).to_bits());
        assert!(!Arc::ptr_eq(&states[0].words, &states[1].words));
        let mut excess = Vec::with_capacity(1_048_576);
        excess.extend([-0.0, f64::from_bits(0x7ff8000000000123)]);
        let excess = Arc::new(excess);
        let weak = Arc::downgrade(&excess);
        let state = StateWords::from(excess);
        assert!(weak.upgrade().is_none());
        assert_eq!(state[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(state[1].to_bits(), 0x7ff8000000000123);
    }
}

#[cfg(test)]
mod growing_tests {
    use super::*;

    #[test]
    fn growing_warm_frames_never_overfill_a_shared_chunk() {
        let mut arena = StateWordArena::default();
        let scale = StateWordArena::WORDS / 4096;
        let sizes = [2400 * scale, 1200 * scale, 1200 * scale, 6000 * scale, 3600 * scale, 1];
        for width in sizes {
            arena.write(1, |words| { words.resize(words.len() + width, -0.0); true });
        }
        // A rejected oversized frame cannot leave excess capacity behind a
        // previous successful publication in the same tail.
        assert!(!arena.write(1, |words| { words.reserve(100_000); false }));
        let states = arena.finish();
        for state in &states {
            assert!(state.words.len() <= StateWordArena::WORDS || state.range == (0..state.words.len()));
            assert!(state.words.capacity() <= StateWordArena::WORDS.max(state.len()));
            assert!(state.iter().all(|word| word.to_bits() == (-0.0f64).to_bits()));
        }
        assert!(Arc::ptr_eq(&states[0].words, &states[1].words));
        assert!(!Arc::ptr_eq(&states[1].words, &states[2].words));
        assert_eq!(states.iter().map(|s| s.len()).collect::<Vec<_>>(), sizes);
    }
}
