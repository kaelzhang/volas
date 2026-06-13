//! Single-write indicator output buffers.
//!
//! An indicator builds an `n`-long `f64` column: a NaN warm-up prefix, then the
//! valid region. The fast way to fill it writes each slot exactly once. The old
//! `Vec::with_capacity(n)` + `unsafe set_len(n)` did that, but left a window where
//! a missed write would read uninitialized memory — silent garbage in a live
//! signal (owner decision D2 retired it). The safe-but-slower replacement,
//! `vec![f64::NAN; n]` then overwrite, writes the valid region twice: a redundant
//! NaN memset over `[lookback, n)` plus the real value.
//!
//! [`build_f64`] gets the single-write speed back without the hazard. It hands the
//! caller `n` uninitialized slots as `&mut [MaybeUninit<f64>]` (so the type system
//! forces an explicit `.write()` per slot) and `set_len`s only after the closure
//! returns. In debug / test builds it first poisons every slot and then asserts
//! none survived, so a future code path that forgets a slot aborts the test
//! deterministically — regardless of what the allocator happened to hand out.
//! Release builds compile the poison + assert away (`debug_assertions` off), so the
//! generated code is identical to the retired `set_len` pattern (proven by a
//! release disassembly: same instruction count, the only delta being where the
//! length store lands).
//!
//! Pair this with `MallocPreScribble=1` (macOS) / `MALLOC_PERTURB_` (glibc) when
//! running the suite, and `cargo miri test` in CI, for three independent,
//! luck-free guards against an unwritten slot.

use std::mem::MaybeUninit;

/// A NaN with a recognisable payload — never a legitimate indicator value (warm-up
/// is the canonical `f64::NAN`, `0x7ff8_0000_0000_0000`; outputs are finite or that
/// NaN), so a surviving poison slot is unambiguously an unwritten one.
#[cfg(debug_assertions)]
const POISON: f64 = f64::from_bits(0x7ff8_0000_dead_beef);

/// Build an `n`-element `f64` indicator output with one write per slot. `fill`
/// receives all `n` slots as `&mut [MaybeUninit<f64>]` and **must write every one**
/// (warm-up NaN + valid region); on return the whole buffer is initialised.
///
/// Single-write (no prefill-then-overwrite double pass), and no `set_len`-on-uninit
/// hazard: debug builds poison the slots and assert all were overwritten.
#[inline]
pub(crate) fn build_f64(n: usize, fill: impl FnOnce(&mut [MaybeUninit<f64>])) -> Vec<f64> {
    let mut v = Vec::<f64>::with_capacity(n);
    {
        // `with_capacity(n)` guarantees `capacity >= n`; take exactly the first `n`.
        let slots = &mut v.spare_capacity_mut()[..n];
        #[cfg(debug_assertions)]
        for s in slots.iter_mut() {
            s.write(POISON);
        }
        fill(slots);
        #[cfg(debug_assertions)]
        for (i, s) in slots.iter().enumerate() {
            // SAFETY: poisoned just above, so initialised to read back in debug.
            let bits = unsafe { s.assume_init() }.to_bits();
            assert!(
                bits != POISON.to_bits(),
                "indicator output slot {i} of {n} left unwritten (build_f64 contract)"
            );
        }
    }
    // SAFETY: `fill` wrote every slot in `[0, n)` (enforced in debug by the poison
    // sweep above; release relies on the same contract, which the kernels uphold).
    unsafe {
        v.set_len(n);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_f64_writes_every_slot() {
        let out = build_f64(5, |s| {
            for (i, slot) in s.iter_mut().enumerate() {
                slot.write(i as f64 * 2.0);
            }
        });
        assert_eq!(out, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn build_f64_warmup_then_valid() {
        let out = build_f64(4, |s| {
            s[0].write(f64::NAN);
            s[1].write(f64::NAN);
            s[2].write(10.0);
            s[3].write(20.0);
        });
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!(&out[2..], &[10.0, 20.0]);
    }

    #[test]
    #[should_panic(expected = "left unwritten")]
    #[cfg(debug_assertions)]
    fn build_f64_catches_a_missed_slot() {
        // Deliberately skip slot 1 — the poison survives, so debug aborts.
        let _ = build_f64(3, |s| {
            s[0].write(1.0);
            s[2].write(3.0);
        });
    }
}
