//! Expansion storage: common sums stay inline; precision is never capacity-bound.
use std::ops::{Deref, DerefMut};

const INLINE: usize = 8;

#[derive(Clone, Debug)]
pub(super) struct Partials {
    inline: [f64; INLINE],
    len: usize,
    spill: Option<Vec<f64>>,
}
impl Default for Partials {
    fn default() -> Self {
        Self {
            inline: [0.0; INLINE],
            len: 0,
            spill: None,
        }
    }
}
impl Deref for Partials {
    type Target = [f64];
    #[inline]
    fn deref(&self) -> &[f64] {
        match &self.spill {
            Some(v) => v,
            None => &self.inline[..self.len],
        }
    }
}
impl DerefMut for Partials {
    #[inline]
    fn deref_mut(&mut self) -> &mut [f64] {
        match &mut self.spill {
            Some(v) => v,
            None => &mut self.inline[..self.len],
        }
    }
}
impl Partials {
    #[inline]
    pub(super) fn push(&mut self, x: f64) {
        if let Some(v) = &mut self.spill {
            v.push(x);
        } else if self.len < INLINE {
            self.inline[self.len] = x;
            self.len += 1;
        } else {
            let mut v = Vec::with_capacity(INLINE * 2);
            v.extend_from_slice(&self.inline);
            v.push(x);
            self.spill = Some(v);
        }
    }
    #[inline]
    pub(super) fn truncate(&mut self, len: usize) {
        if let Some(v) = &mut self.spill {
            v.truncate(len);
        } else {
            self.len = self.len.min(len);
        }
    }
    #[inline]
    pub(super) fn clear(&mut self) {
        self.truncate(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spill_preserves_every_word_and_reuses_its_capacity() {
        let mut p = Partials::default();
        p.truncate(2);
        for i in 0..INLINE {
            p.push(i as f64);
        }
        assert!(p.spill.is_none());
        p.push(-0.0);
        assert!(p.spill.is_some());
        assert_eq!(p[INLINE].to_bits(), (-0.0f64).to_bits());
        assert_eq!(
            p.clone()[..INLINE],
            (0..INLINE).map(|i| i as f64).collect::<Vec<_>>()
        );
        p[0] = -1.0;
        assert_eq!(p[0], -1.0);
        p.truncate(INLINE);
        p.clear();
        p.push(9.0);
        p.push(10.0);
        assert_eq!(&*p, &[9.0, 10.0]);
        assert!(p.spill.is_some());
    }
    #[test]
    fn exact_expansions_can_exceed_the_inline_capacity() {
        let words: Vec<_> = (0..16).map(|i| 2.0f64.powi(-1000 + 60 * i)).collect();
        let mut exact = super::super::Exact::default();
        for &w in &words {
            exact.add(w);
        }
        assert!(exact.partials.spill.is_some());
        assert_eq!(exact.partials.len(), words.len());
        let a = exact.to_csum();
        let mut reverse = super::super::Exact::default();
        for &w in words.iter().rev() {
            reverse.add(w);
        }
        assert_eq!(a, reverse.to_csum());
        for &w in &words {
            exact.add(-w);
        }
        assert_eq!(exact.to_csum().value().to_bits(), 0.0f64.to_bits());
    }
}
