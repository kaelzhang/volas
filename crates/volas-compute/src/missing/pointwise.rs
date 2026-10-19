//! Admission folded into a pointwise arithmetic pass, with no preliminary scan.
use super::any_of;

/// Exact finite-input classification, packed into four SIMD lanes per register.
/// The signless upper word of every NaN or infinity is at least infinity's word;
/// every finite value's word is smaller, including subnormals and signed zero.
#[derive(Default)]
pub(crate) struct Finiteness {
    largest: u32,
}

impl Finiteness {
    #[inline(always)]
    pub(crate) fn record(&mut self, value: f64) {
        let word = ((value.to_bits() >> 32) as u32) & 0x7fff_ffff;
        self.largest = self.largest.max(word);
    }

    #[inline(always)]
    pub(crate) fn all_finite(&self) -> bool {
        self.largest < (f64::INFINITY.to_bits() >> 32) as u32
    }
}

/// Compute terms in IEEE order. An unknown input's finite check is an OR reduction
/// alongside the value writes, so it needs neither an extra pass nor per-row result
/// masks. Only a pass with a non-finite witness checks its terms afterwards.
#[inline(always)]
pub(crate) fn map_terms<const K: usize>(
    len: usize,
    warmup: usize,
    complete: bool,
    reads: impl Fn(usize) -> [f64; K],
    value: impl Fn(usize, [f64; K]) -> f64,
) -> Vec<f64> {
    map_terms_impl::<false, false, K>(len, warmup, complete, reads, |i, inputs| {
        // Any NaN or infinity makes this witness non-finite. Finite inputs
        // may overflow it too: the cold check distinguishes that and preserves
        // the IEEE value. Its additions vectorise with the maths.
        let witness = inputs.into_iter().reduce(|a, b| a + b).unwrap_or(0.0);
        (value(i, inputs), witness)
    })
}

/// As `map_terms`, with a witness shared with the indicator's own arithmetic.
/// It must be non-finite whenever any original input is missing; a false positive
/// merely enters the exact cold check and never changes a finite-input result.
#[inline(always)]
pub(crate) fn map_terms_with<const K: usize>(
    len: usize,
    warmup: usize,
    complete: bool,
    reads: impl Fn(usize) -> [f64; K],
    value_and_witness: impl Fn(usize, [f64; K]) -> (f64, f64),
) -> Vec<f64> {
    map_terms_impl::<true, false, K>(len, warmup, complete, reads, value_and_witness)
}

/// Shared-arithmetic witnesses reduced in packed upper words. This retains the
/// same exact cold repair while avoiding per-lane boolean masks in the value loop.
#[inline(always)]
pub(crate) fn map_terms_packed<const K: usize>(
    len: usize,
    warmup: usize,
    complete: bool,
    reads: impl Fn(usize) -> [f64; K],
    value_and_witness: impl Fn(usize, [f64; K]) -> (f64, f64),
) -> Vec<f64> {
    map_terms_impl::<false, true, K>(len, warmup, complete, reads, value_and_witness)
}

// A maximum of absolute IEEE bit patterns retains NaN as well as infinity.
// The shared-arithmetic witness uses this vector reduction to avoid narrowing
// per-lane predicates to bools; ordinary input sums retain their existing loop.
#[inline(always)]
fn map_terms_impl<const MAXIMUM: bool, const PACKED: bool, const K: usize>(
    len: usize,
    warmup: usize,
    complete: bool,
    reads: impl Fn(usize) -> [f64; K],
    value_and_witness: impl Fn(usize, [f64; K]) -> (f64, f64),
) -> Vec<f64> {
    // This explicit bound also proves every trailing read's index nonnegative.
    if warmup >= len {
        return vec![f64::NAN; len];
    }
    let start = warmup;
    let mut out = Vec::with_capacity(len);
    out.extend(std::iter::repeat_n(f64::NAN, start));
    let mut missing = false;
    let mut maximum = 0_u64;
    let mut packed = Finiteness::default();
    if complete {
        out.extend((start..len).map(|i| value_and_witness(i, reads(i)).0));
    } else {
        out.extend((start..len).map(|i| {
            let (value, witness) = value_and_witness(i, reads(i));
            if PACKED {
                packed.record(witness);
            } else if MAXIMUM {
                maximum = maximum.max(witness.abs().to_bits());
            } else {
                missing |= !witness.is_finite();
            }
            value
        }));
    }
    if if PACKED {
        !packed.all_finite()
    } else if MAXIMUM {
        maximum >= f64::INFINITY.to_bits()
    } else {
        missing
    } {
        repair_terms(&mut out, start, reads);
    }
    out
}

#[cold]
#[inline(never)]
pub(crate) fn repair_terms<const K: usize>(
    out: &mut [f64],
    start: usize,
    reads: impl Fn(usize) -> [f64; K],
) {
    repair_slots(out.len(), start, reads, |i, value| out[i] = value);
}

/// Repair written slots before an output buffer publishes or checks its values.
#[cold]
#[inline(never)]
pub(crate) fn repair_slots<const K: usize>(
    len: usize,
    start: usize,
    reads: impl Fn(usize) -> [f64; K],
    mut write: impl FnMut(usize, f64),
) {
    for i in start..len {
        if any_of(&reads(i)) {
            write(i, f64::NAN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Finiteness;

    #[test]
    fn packed_finiteness_matches_the_missing_owner_for_every_ieee_class() {
        for sign in [0, 1_u64 << 63] {
            for exponent in 0..=0x7ff_u64 {
                for mantissa in [0, 1, (1_u64 << 52) - 1] {
                    let value = f64::from_bits(sign | (exponent << 52) | mantissa);
                    let mut fact = Finiteness::default();
                    assert!(fact.all_finite());
                    fact.record(value);
                    assert_eq!(!fact.all_finite(), crate::missing::is_missing(value));
                    fact.record(1.0);
                    assert_eq!(!fact.all_finite(), crate::missing::is_missing(value));
                }
            }
        }
    }
}
