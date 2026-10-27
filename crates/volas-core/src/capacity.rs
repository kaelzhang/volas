//! Spare retained by a clone-then-grow Vec, before any transaction publication.
//! Byte capacity and allocation errors still belong to Vec::try_reserve.
#[inline]
pub(crate) fn growth_capacity(held: usize, needed: usize) -> usize {
    if needed > held { needed.max(held.saturating_mul(2)) } else { needed }
}
