//! Non-finite numeric inputs: the shared predicate and scan for storage and compute.

/// Whether `x` is a missing computation input: NaN or either infinity.
#[inline]
pub fn is_missing(x: f64) -> bool {
    !x.is_finite()
}

/// Whether any input is missing. Scan 32 rows, then eight rows, then the scalar tail.
/// Shared by window guards and recursive input completeness checks.
#[inline(never)]
#[allow(clippy::needless_return)] // mutually exclusive architecture implementations
pub fn any_missing(x: &[f64]) -> bool {
    let mut chunks = x.chunks_exact(32);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on aarch64. Each unaligned two-lane load is
    // inside its initialized 32-row chunk or eight-row remainder chunk.
    unsafe {
        use std::arch::aarch64::*;
        let inf = vdupq_n_f64(f64::INFINITY);
        for c in &mut chunks {
            let p = c.as_ptr();
            let m0 = vcaltq_f64(vld1q_f64(p), inf);
            let m1 = vcaltq_f64(vld1q_f64(p.add(2)), inf);
            let m2 = vcaltq_f64(vld1q_f64(p.add(4)), inf);
            let m3 = vcaltq_f64(vld1q_f64(p.add(6)), inf);
            let m4 = vcaltq_f64(vld1q_f64(p.add(8)), inf);
            let m5 = vcaltq_f64(vld1q_f64(p.add(10)), inf);
            let m6 = vcaltq_f64(vld1q_f64(p.add(12)), inf);
            let m7 = vcaltq_f64(vld1q_f64(p.add(14)), inf);
            let m8 = vcaltq_f64(vld1q_f64(p.add(16)), inf);
            let m9 = vcaltq_f64(vld1q_f64(p.add(18)), inf);
            let m10 = vcaltq_f64(vld1q_f64(p.add(20)), inf);
            let m11 = vcaltq_f64(vld1q_f64(p.add(22)), inf);
            let m12 = vcaltq_f64(vld1q_f64(p.add(24)), inf);
            let m13 = vcaltq_f64(vld1q_f64(p.add(26)), inf);
            let m14 = vcaltq_f64(vld1q_f64(p.add(28)), inf);
            let m15 = vcaltq_f64(vld1q_f64(p.add(30)), inf);
            let finite = vandq_u64(
                vandq_u64(
                    vandq_u64(vandq_u64(m0, m1), vandq_u64(m2, m3)),
                    vandq_u64(vandq_u64(m4, m5), vandq_u64(m6, m7)),
                ),
                vandq_u64(
                    vandq_u64(vandq_u64(m8, m9), vandq_u64(m10, m11)),
                    vandq_u64(vandq_u64(m12, m13), vandq_u64(m14, m15)),
                ),
            );
            if vminvq_u32(vreinterpretq_u32_u64(finite)) == 0 {
                return true;
            }
        }
        let mut remainder = chunks.remainder().chunks_exact(8);
        for c in &mut remainder {
            let p = c.as_ptr();
            let m0 = vcaltq_f64(vld1q_f64(p), inf);
            let m1 = vcaltq_f64(vld1q_f64(p.add(2)), inf);
            let m2 = vcaltq_f64(vld1q_f64(p.add(4)), inf);
            let m3 = vcaltq_f64(vld1q_f64(p.add(6)), inf);
            let finite = vandq_u64(vandq_u64(m0, m1), vandq_u64(m2, m3));
            if vminvq_u32(vreinterpretq_u32_u64(finite)) == 0 {
                return true;
            }
        }
        return remainder.remainder().iter().any(|&v| is_missing(v));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is mandatory on x86_64. Each unaligned two-lane load is
    // inside its initialized 32-row chunk or eight-row remainder chunk.
    unsafe {
        use std::arch::x86_64::*;
        let inf = _mm_set1_pd(f64::INFINITY);
        let sign = _mm_set1_pd(-0.0);
        for c in &mut chunks {
            let p = c.as_ptr();
            let m0 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p)), inf);
            let m1 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(2))), inf);
            let m2 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(4))), inf);
            let m3 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(6))), inf);
            let m4 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(8))), inf);
            let m5 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(10))), inf);
            let m6 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(12))), inf);
            let m7 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(14))), inf);
            let m8 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(16))), inf);
            let m9 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(18))), inf);
            let m10 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(20))), inf);
            let m11 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(22))), inf);
            let m12 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(24))), inf);
            let m13 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(26))), inf);
            let m14 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(28))), inf);
            let m15 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(30))), inf);
            let finite = _mm_and_pd(
                _mm_and_pd(
                    _mm_and_pd(_mm_and_pd(m0, m1), _mm_and_pd(m2, m3)),
                    _mm_and_pd(_mm_and_pd(m4, m5), _mm_and_pd(m6, m7)),
                ),
                _mm_and_pd(
                    _mm_and_pd(_mm_and_pd(m8, m9), _mm_and_pd(m10, m11)),
                    _mm_and_pd(_mm_and_pd(m12, m13), _mm_and_pd(m14, m15)),
                ),
            );
            if _mm_movemask_pd(finite) != 3 {
                return true;
            }
        }
        let mut remainder = chunks.remainder().chunks_exact(8);
        for c in &mut remainder {
            let p = c.as_ptr();
            let m0 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p)), inf);
            let m1 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(2))), inf);
            let m2 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(4))), inf);
            let m3 = _mm_cmplt_pd(_mm_andnot_pd(sign, _mm_loadu_pd(p.add(6))), inf);
            let finite = _mm_and_pd(_mm_and_pd(m0, m1), _mm_and_pd(m2, m3));
            if _mm_movemask_pd(finite) != 3 {
                return true;
            }
        }
        return remainder.remainder().iter().any(|&v| is_missing(v));
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for c in &mut chunks {
            if c.iter().fold(false, |acc, &v| acc | is_missing(v)) {
                return true;
            }
        }
        chunks.remainder().iter().any(|&v| is_missing(v))
    }
}


/// Native f32 version of the shared non-finite input scan.
#[inline(never)]
#[allow(clippy::needless_return)] // mutually exclusive architecture implementations
pub fn any_missing_f32(x: &[f32]) -> bool {
    let mut chunks = x.chunks_exact(32);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: SIMD is mandatory on this architecture; every unaligned four-lane
    // load stays inside its initialized 32-row or eight-row chunk.
    unsafe {
        use std::arch::aarch64::*;
        let inf = vdupq_n_f32(f32::INFINITY);
        for c in &mut chunks {
            let p = c.as_ptr();
            let m0 = vcaltq_f32(vld1q_f32(p), inf);
            let m1 = vcaltq_f32(vld1q_f32(p.add(4)), inf);
            let m2 = vcaltq_f32(vld1q_f32(p.add(8)), inf);
            let m3 = vcaltq_f32(vld1q_f32(p.add(12)), inf);
            let m4 = vcaltq_f32(vld1q_f32(p.add(16)), inf);
            let m5 = vcaltq_f32(vld1q_f32(p.add(20)), inf);
            let m6 = vcaltq_f32(vld1q_f32(p.add(24)), inf);
            let m7 = vcaltq_f32(vld1q_f32(p.add(28)), inf);
            let finite = vandq_u32(vandq_u32(vandq_u32(m0, m1), vandq_u32(m2, m3)), vandq_u32(vandq_u32(m4, m5), vandq_u32(m6, m7)));
            if vminvq_u32(finite) == 0 {
                return true;
            }
        }
        let mut remainder = chunks.remainder().chunks_exact(8);
        for c in &mut remainder {
            let p = c.as_ptr();
            let m0 = vcaltq_f32(vld1q_f32(p), inf);
            let m1 = vcaltq_f32(vld1q_f32(p.add(4)), inf);
            let finite = vandq_u32(m0, m1);
            if vminvq_u32(finite) == 0 {
                return true;
            }
        }
        return remainder.remainder().iter().any(|&v| is_missing_f32(v));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SIMD is mandatory on this architecture; every unaligned four-lane
    // load stays inside its initialized 32-row or eight-row chunk.
    unsafe {
        use std::arch::x86_64::*;
        let inf = _mm_set1_ps(f32::INFINITY);
        let sign = _mm_set1_ps(-0.0);
        for c in &mut chunks {
            let p = c.as_ptr();
            let m0 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p)), inf);
            let m1 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p.add(4))), inf);
            let m2 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p.add(8))), inf);
            let m3 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p.add(12))), inf);
            let m4 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p.add(16))), inf);
            let m5 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p.add(20))), inf);
            let m6 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p.add(24))), inf);
            let m7 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p.add(28))), inf);
            let finite = _mm_and_ps(_mm_and_ps(_mm_and_ps(m0, m1), _mm_and_ps(m2, m3)), _mm_and_ps(_mm_and_ps(m4, m5), _mm_and_ps(m6, m7)));
            if _mm_movemask_ps(finite) != 15 {
                return true;
            }
        }
        let mut remainder = chunks.remainder().chunks_exact(8);
        for c in &mut remainder {
            let p = c.as_ptr();
            let m0 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p)), inf);
            let m1 = _mm_cmplt_ps(_mm_andnot_ps(sign, _mm_loadu_ps(p.add(4))), inf);
            let finite = _mm_and_ps(m0, m1);
            if _mm_movemask_ps(finite) != 15 {
                return true;
            }
        }
        return remainder.remainder().iter().any(|&v| is_missing_f32(v));
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for c in &mut chunks {
            if c.iter().fold(false, |acc, &v| acc | is_missing_f32(v)) {
                return true;
            }
        }
        chunks.remainder().iter().any(|&v| is_missing_f32(v))
    }
}

/// Whether an f32 is a missing computation input.
#[inline]
pub fn is_missing_f32(x: f32) -> bool {
    !x.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_scans_cover_every_lane_and_remainder() {
        for len in 0..=70 {
            let mut single = vec![f32::MAX; len];
            let mut double = vec![f64::MAX; len];
            assert!(!any_missing_f32(&single));
            assert!(!any_missing(&double));
            for lane in 0..len {
                for bad in [f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    single[lane] = bad;
                    double[lane] = bad as f64;
                    assert!(any_missing_f32(&single), "len={len}, lane={lane}, bad={bad}");
                    assert!(any_missing(&double), "len={len}, lane={lane}, bad={bad}");
                }
                single[lane] = f32::MAX;
                double[lane] = f64::MAX;
            }
        }
        for finite in [0.0, -0.0, f32::MIN_POSITIVE, -f32::MAX] {
            assert!(!is_missing_f32(finite));
            assert!(!any_missing_f32(&[finite; 70]));
        }
    }
}
