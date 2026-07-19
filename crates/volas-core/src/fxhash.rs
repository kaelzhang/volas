//! A fast hasher for the library's INTERNAL string maps — the column-name index
//! and the directive cache. It is not suitable for security-sensitive keys because
//! these maps use internal column / directive names, never adversarial input, so the standard
//! library's SipHash (which trades speed for DoS resistance) is pure overhead. These
//! maps are hit on the live-append hot path — every `column_pos` / `column` lookup
//! and every directive refresh — where SipHash dominated the per-bar cost; FxHash
//! (the hasher rustc itself uses) is several times faster on short keys.
//!
//! This is ONLY for internal-keyed maps. A map over user-supplied keys keeps the
//! default SipHash.

use std::hash::{BuildHasherDefault, Hasher};

/// A [`std::collections::HashMap`] using [`FxHasher`].
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// FxHash: fold each machine word of the key into the state with a rotate-xor-multiply.
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            self.add(u64::from_le_bytes(bytes[..8].try_into().unwrap()));
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            self.add(u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64);
            bytes = &bytes[4..];
        }
        if bytes.len() >= 2 {
            self.add(u16::from_le_bytes(bytes[..2].try_into().unwrap()) as u64);
            bytes = &bytes[2..];
        }
        if let Some(&b) = bytes.first() {
            self.add(b as u64);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash(s: &str) -> u64 {
        let mut h = FxHasher::default();
        s.hash(&mut h);
        h.finish()
    }

    #[test]
    fn fxhash_distinguishes_keys_across_lengths() {
        // exercises every chunk branch: 8-byte ("volume_x"=8), 4, 2, 1-byte tails.
        let keys = ["", "a", "lo", "low", "open", "close", "volume", "volume_xy", "macd.signal"];
        let hashes: Vec<u64> = keys.iter().map(|k| hash(k)).collect();
        // distinct keys -> distinct hashes (no collisions in this small set)
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(hashes[i], hashes[j], "collision: {:?} vs {:?}", keys[i], keys[j]);
            }
        }
        // deterministic
        assert_eq!(hash("close"), hash("close"));
    }

    #[test]
    fn fxhashmap_works_as_a_map() {
        let mut m: FxHashMap<String, usize> = FxHashMap::default();
        m.insert("open".into(), 0);
        m.insert("close".into(), 3);
        assert_eq!(m.get("open"), Some(&0));
        assert_eq!(m.get("close"), Some(&3));
        assert_eq!(m.get("missing"), None);
    }
}
