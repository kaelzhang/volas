use super::*;

#[test]
fn a_compensated_sum_forgets_a_spike() {
    let mut s = Csum::default();
    s.add(0.1);
    s.add(1e300);
    s.sub(1e300);
    assert_eq!(s.value(), 0.1);
    assert!(s.is_exact());
    let mut st = [0.0; 2];
    s.write_state(&mut st);
    assert_eq!(Csum::from_state(&st), Some(s));
    assert_eq!(Csum::from_state(&[1.0]), None);
    s.add(f64::INFINITY);
    assert!(s.value().is_nan());
    assert!(!s.is_exact());
}

#[test]
fn a_flat_window_has_its_value_as_mean() {
    for c in [0.1, 100.1, 1234.5678, 1e-7 / 3.0, 7.0 / 3.0] {
        for n in 1..60 {
            let s = Csum::of(std::iter::repeat_n(c, n));
            assert_eq!(s.mean(n as f64), c, "{c} x{n}");
        }
    }
    // the mean does not depend on how the words split the sum
    let a = Csum { sum: 3.0, comp: 0.5, exact: true };
    let b = Csum { sum: 3.5, comp: 0.0, exact: true };
    assert_eq!(a.mean(7.0), b.mean(7.0));
    let mut p = Csum::default();
    p.add_prod(3.0, 0.1);
    p.sub_sum(&Csum::of([0.1, 0.1, 0.1]));
    assert_eq!(p.value(), 0.0);
    p.add_sum(&Csum::of([1.0]));
    p.sub_prod(2.0, 0.5);
    assert_eq!(p.value(), 0.0);
    assert!(p.is_exact());
}

#[test]
fn anchors_fall_on_multiples_of_the_period() {
    for (from, period) in [(0, 3), (5, 3), (6, 3), (7, 1), (11, 4)] {
        let mut a = Anchors::from(from, period);
        for pos in from..from + 20 {
            assert_eq!(a.tick(), anchors_at(pos, period), "{from} {period} {pos}");
        }
    }
    assert!(!anchors_at(5, 0));
}

/// Two outliers of different magnitudes overflow the compensation word: the check
/// sees it, and the window re-summed is exact again.
#[test]
fn a_rounded_compensation_is_detected() {
    let mut s = Csum::default();
    for x in [1.1, 1e35, 1e20, 1.3, 2.7] {
        s.add(x);
    }
    for x in [1.1, 1e35, 1e20] {
        s.sub(x);
    }
    assert!(!s.is_exact());
    assert_ne!(s.value(), 4.0);
    let w = Csum::of([1.3, 2.7, 3.1, 4.9]);
    assert!(w.is_exact());
    assert_eq!(w.value(), 12.0);
    // an inexact sum is carried with a NaN low word and read back inexact
    let mut st = [0.0; 2];
    s.write_state(&mut st);
    assert!(st[1].is_nan());
    assert!(!Csum::from_state(&st).unwrap().is_exact());
    // the inexactness of a sum spreads to one it is added to
    let mut t = Csum::default();
    t.add_sum(&s);
    assert!(!t.is_exact());
    let mut t = Csum::default();
    t.sub_sum(&s);
    assert!(!t.is_exact());
}

/// An independent exact reference: terms `m·2^e` with `e ∈ [-60, 10]` summed as `i128`
/// multiples of `2⁻⁶⁰`, then rounded once (`as f64` rounds to nearest, ties to even).
fn reference(terms: &[f64]) -> f64 {
    let q = 2f64.powi(60);
    let total: i128 = terms
        .iter()
        .map(|&x| {
            let scaled = x * q;
            assert_eq!(scaled.fract(), 0.0, "{x} is not a multiple of 2^-60");
            scaled as i128
        })
        .sum();
    total as f64 / q
}

/// A deterministic multi-magnitude series: mantissas of 53 bits, exponents spread over
/// multiples of `2⁻⁶⁰` below `2⁶³`, both signs.
fn terms(seed: u64, n: usize) -> Vec<f64> {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let m = (x >> 11) as f64; // < 2^53
            let e = ((x >> 3) % 71) as i32 - 60; // m·2^e: multiples of 2^-60 below 2^63
            let sign = if x & 1 == 0 { 1.0 } else { -1.0 };
            sign * m * 2f64.powi(e)
        })
        .collect()
}

#[test]
fn exact_sums_are_correctly_rounded() {
    for seed in 0..400 {
        let t = terms(seed, 1 + (seed as usize % 37));
        let want = reference(&t);
        let mut e = Exact::default();
        for &x in &t {
            e.add(x);
        }
        assert_eq!(e.to_csum().value().to_bits(), want.to_bits(), "seed {seed}");
        let c = Csum::of(t.iter().copied());
        assert_eq!(c.value().to_bits(), want.to_bits(), "seed {seed}");
        assert_eq!(e.to_csum().value().to_bits(), want.to_bits(), "seed {seed}");
    }
}

/// A sliding sum over a multi-magnitude series, re-summed whenever it goes inexact,
/// reads every window's correctly rounded sum.
#[test]
fn a_sliding_sum_re_summed_when_inexact_is_exact() {
    let mut resums = 0;
    for seed in 0..60 {
        let t = terms(seed + 1000, 300);
        for w in [1, 2, 3, 7, 20] {
            let mut s = Csum::default();
            for i in 0..t.len() {
                s.add(t[i]);
                if i >= w {
                    s.sub(t[i - w]);
                }
                if !s.is_exact() && i + 1 >= w {
                    resums += 1;
                    s = Csum::of(t[i + 1 - w..=i].iter().copied());
                }
                if i + 1 >= w {
                    let want = reference(&t[i + 1 - w..=i]);
                    assert_eq!(s.value().to_bits(), want.to_bits(), "seed {seed} w {w} row {i}");
                }
            }
        }
    }
    assert!(resums > 1000, "{resums}");
}

#[test]
fn of_rows_resums_only_the_inexact_sums() {
    let rows = [[1e35, 1.0], [1e20, 2.0], [1.3, 3.0], [-1e35, 4.0], [-1e20, 5.0]];
    let [a, b] = Csum::of_rows(rows);
    assert_eq!(a.value(), 1.3);
    assert!(a.is_exact());
    assert_eq!(b.value(), 15.0);
}

#[test]
fn an_exact_sum_needing_three_words_is_read_correctly_but_marked_inexact() {
    // 2^120 + 2^60 + 1: three words; the canonical pair keeps the correctly rounded
    // value and the rounded remainder.
    let t = [2f64.powi(120), 2f64.powi(60), 1.0];
    let c = Csum::of(t);
    assert!(!c.is_exact());
    assert_eq!(c.value(), 2f64.powi(120));
    assert_eq!(c.comp, 2f64.powi(60) + 1.0);
    // the mean of a canonical pair is a function of the sum: equal to one built from
    // the same sum in another order
    let d = Csum::of([1.0, 2f64.powi(60), 2f64.powi(120)]);
    assert_eq!(c.mean(3.0).to_bits(), d.mean(3.0).to_bits());
}

#[test]
fn the_remainder_is_nudged_off_a_tie_that_would_round_away() {
    // hi = 1 + 2^-52 (odd), remainder just below half an ulp: rounded to nearest it
    // would be exactly half an ulp, a tie that rounds hi + lo up to an even neighbour.
    let ulp = 2f64.powi(-52);
    let t = [1.0 + ulp, ulp / 2.0, -ulp * 2f64.powi(-60)];
    let c = Csum::of(t);
    assert_eq!(c.value(), 1.0 + ulp);
    assert!(c.comp < ulp / 2.0);
    assert!(!c.is_exact());
}

#[test]
fn half_even_ties_follow_fsum() {
    // a tie between 1 and 1 + 2^-52 broken upwards by a lower word
    let ulp = 2f64.powi(-52);
    let t = [1.0, ulp / 2.0, ulp * 2f64.powi(-60)];
    assert_eq!(Csum::of(t).value(), 1.0 + ulp);
    // the same tie unbroken goes to the even neighbour, and broken downwards stays
    assert_eq!(Csum::of([1.0, ulp / 2.0]).value(), 1.0);
    assert_eq!(Csum::of([1.0 + ulp, ulp / 2.0, -ulp * 2f64.powi(-60)]).value(), 1.0 + ulp);
    let mut e = Exact::default();
    for x in [-1.0, -ulp / 2.0, -ulp * 2f64.powi(-60)] {
        e.add(x);
    }
    assert_eq!(e.to_csum().value(), -1.0 - ulp);
}

#[test]
fn non_finite_terms_sum_like_fsum() {
    let sum = |t: &[f64]| Csum::of(t.iter().copied()).value();
    assert_eq!(sum(&[1.0, f64::INFINITY, 2.0]), f64::INFINITY);
    assert_eq!(sum(&[1.0, f64::NEG_INFINITY]), f64::NEG_INFINITY);
    assert!(sum(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
    assert!(sum(&[1.0, f64::NAN]).is_nan());
    let mut e = Exact::default();
    e.add(f64::INFINITY);
    assert_eq!(e.to_csum().value(), f64::INFINITY);
    assert!(!e.to_csum().is_exact());
    assert_eq!(Exact::default().to_csum().value(), 0.0);
    assert_eq!(Exact::default().to_csum(), Csum::default());
}

#[test]
fn an_overflowing_window_is_summed_at_a_scale() {
    let big = f64::MAX;
    // the true sums are finite though every pairwise total overflows
    assert_eq!(Csum::of([big, big, -big]).value(), big);
    assert_eq!(Csum::of([big, big, -big, -big, 3.0]).value(), 3.0);
    assert_eq!(Csum::of([1e308, 1e308, -1e308, 0.5]).value(), 1e308);
    // a sum past the threshold is infinite
    let c = Csum::of([big, big]);
    assert_eq!(c.value(), f64::INFINITY);
    assert!(!c.is_exact());
    assert_eq!(Csum::of([-big, -big]).value(), f64::NEG_INFINITY);
    // a huge term entering an expansion already scaled, and the scaled pair
    let mut e = Exact::default();
    for x in [big, big, -big, 1.0, -big / 2.0] {
        e.add(x);
    }
    let c = e.to_csum();
    assert!(!c.is_exact());
    assert_eq!(c.value(), big / 2.0);
    // words already held are scaled with it (a subnormal one drops out: below the
    // scaled grid, the documented residue)
    let mut e = Exact::default();
    for x in [5e-324, 1.0, big, -big] {
        e.add(x);
    }
    assert_eq!(e.to_csum().value(), 1.0);
    // a scaled expansion whose sum is past the threshold
    let mut e = Exact::default();
    for x in [big, big, 1.0] {
        e.add(x);
    }
    assert_eq!(e.to_csum().value(), f64::INFINITY);
}

#[test]
fn a_stream_mean_re_sums_an_inexact_window() {
    let mut m = StreamMean::<2>::new();
    let mut got = Vec::new();
    for x in [1e35, 1e20, 1.5, 2.5, 3.5] {
        got.push(m.push(x, false));
    }
    assert_eq!(got[3], Some(2.0));
    assert_eq!(got[4], Some(3.0));
    assert!(m.finite());
}
