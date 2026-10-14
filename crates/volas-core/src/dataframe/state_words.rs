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
    /// Keep a checked range of shared words. An invalid range is rejected before
    /// a value can be constructed through this safe API.
    pub fn from_shared(words: Arc<Vec<f64>>, range: Range<usize>) -> Option<Self> {
        (range.start <= range.end && range.end <= words.len()).then_some(Self { words, range })
    }
}

impl From<Arc<Vec<f64>>> for StateWords {
    fn from(words: Arc<Vec<f64>>) -> Self {
        let end = words.len();
        Self { words, range: 0..end }
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
