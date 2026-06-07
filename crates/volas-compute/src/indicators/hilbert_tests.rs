use super::*;

#[test]
fn ht_core_short_input_returns_warmup_only() {
    // Inputs at or below the WMA warm-up never enter the main loop (covers the
    // early return); every bar stays at the HtBar default.
    for n in [0usize, 1, CORE_START, CORE_START - 1] {
        let price: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let bars = ht_core(&price);
        assert_eq!(bars.len(), n);
        assert!(bars.iter().all(|b| b.smooth_period == 0.0 && b.q1 == 0.0));
    }
}

#[test]
fn dcphase_zero_imaginary_carry_branch() {
    // The `imagPart == 0` carry branch (a verbatim port of TA-Lib's defensive
    // path) only runs when the cosine-weighted sum cancels exactly while the
    // sine-weighted sum does not. Force it with DCPeriodInt == 2 over two equal
    // buffer values: cos(0)+cos(π) = 1 + (-1) = 0 exactly, while
    // sin(0)+sin(π) = sin(π) ≈ 1.2e-16 ≠ 0, so `real_part` keeps the sign of the
    // (positive / negative) prices and the ±90 nudge is exercised.
    assert_eq!(
        (TWO_PI / 2.0).cos(),
        -1.0,
        "cos(π) must be exactly -1 for the cancellation"
    );

    let mut up = DcPhase::new();
    let _ = up.push(5.0, 2.0); // warm the buffer; smooth_period 2.0 -> DCPeriodInt 2
    let p_up = up.push(5.0, 2.0); // window [5, 5] -> imag == 0, real > 0 (+90)
    assert!(p_up.is_finite());

    let mut down = DcPhase::new();
    let _ = down.push(-5.0, 2.0);
    let p_down = down.push(-5.0, 2.0); // window [-5, -5] -> imag == 0, real < 0 (-90)
    assert!(p_down.is_finite());

    // imag == 0 AND real == 0 (DCPeriodInt 0 -> empty window): the carry block is
    // entered but neither ±90 nudge fires (the fall-through).
    let p_flat = up.push(5.0, 0.4); // smooth_period 0.4 -> DCPeriodInt 0
    assert!(p_flat.is_finite());
}

/// A non-degenerate ~250-bar series so every HT output is well past its lookback
/// (a sine carrier + slow drift, like the parity fixtures).
fn series(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            100.0 + 0.05 * t + 6.0 * (t * 0.30).sin() + 2.0 * (t * 0.11).cos()
        })
        .collect()
}

/// Compare two slices for exact bit equality (NaN == NaN), the state-carry oracle.
fn assert_bits(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
            "{what}: bar {i}: resume {x:?} != full {y:?}",
        );
    }
}

/// Tolerant resume-vs-full check for the **windowed-average** outputs
/// (TRENDLINE / TRENDMODE): the full pass sums each DC-period window via O(n)
/// price prefix sums while the per-bar resume keeps the exact O(period) rescan,
/// so the two agree only to a **non-accumulating ~1e-12** (the same window, a
/// different summation order). The recursive core they share stays bit-exact
/// (`assert_bits` above); these outputs are independently pinned to TA-Lib.
fn assert_near(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let ok = (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-9 + 1e-9 * y.abs();
        assert!(ok, "{what}: bar {i}: resume {x:?} !~= full {y:?}");
    }
}

/// Every HT resume, fed the carried state of a full compute over `price[..from]`,
/// must reproduce the tail of a full compute over all of `price` — bit-for-bit.
/// This is the kernel-level twin of `test_mutation_parity`'s append/slice oracle.
#[test]
fn ht_resume_is_bit_identical_to_full() {
    let price = series(250);
    // `from` values spanning a fresh resume just past lookback, an even and an odd
    // bar (the core alternates), and a generic large offset.
    for &from in &[64usize, 65, 100, 200, 249] {
        let head = &price[..from];

        // DCPERIOD / PHASOR (core-only state).
        let st = ht_core_state(head).unwrap();
        let (v, _) = ht_dcperiod_resume(&price, from, &st).unwrap();
        assert_bits(&v, &ht_dcperiod(&price)[from..], "dcperiod");
        let (v, _) = ht_phasor_resume(&price, false, from, &st).unwrap();
        assert_bits(&v, &ht_phasor(&price).0[from..], "phasor.inphase");
        let (v, _) = ht_phasor_resume(&price, true, from, &st).unwrap();
        assert_bits(&v, &ht_phasor(&price).1[from..], "phasor.quadrature");

        // DCPHASE / SINE (core + DC-phase ring).
        let st = ht_dcphase_state(head).unwrap();
        let (v, _) = ht_dcphase_resume(&price, from, &st).unwrap();
        assert_bits(&v, &ht_dcphase(&price)[from..], "dcphase");
        let st = ht_sine_state(head).unwrap();
        let (v, _) = ht_sine_resume(&price, false, from, &st).unwrap();
        assert_bits(&v, &ht_sine(&price).0[from..], "sine.sine");
        let (v, _) = ht_sine_resume(&price, true, from, &st).unwrap();
        assert_bits(&v, &ht_sine(&price).1[from..], "sine.leadsine");

        // TRENDLINE / TRENDMODE (core + iTrend triple [+ DC-phase / counters]).
        let st = ht_trendline_state(head).unwrap();
        let (v, _) = ht_trendline_resume(&price, from, &st).unwrap();
        assert_near(&v, &ht_trendline(&price)[from..], "trendline");
        let st = ht_trendmode_state(head).unwrap();
        let (v, _) = ht_trendmode_resume(&price, from, &st).unwrap();
        assert_near(&v, &ht_trendmode(&price)[from..], "trendmode");

        // MAMA / FAMA (core + [mama, fama, prev_phase]).
        let st = mama_state(head, 0.5, 0.05).unwrap();
        let (v, _) = mama_resume(&price, 0.5, 0.05, false, from, &st).unwrap();
        assert_bits(&v, &mama(&price, 0.5, 0.05).0[from..], "mama");
        let (v, _) = mama_resume(&price, 0.5, 0.05, true, from, &st).unwrap();
        assert_bits(&v, &mama(&price, 0.5, 0.05).1[from..], "mama.fama");
    }
}

/// Chained resumes (append one bar at a time, threading the returned state) must
/// also stay bit-identical — the live-tick pattern, where each fulfill resumes
/// from the previous fulfill's carried state.
#[test]
fn ht_resume_chains_bit_identical() {
    let price = series(160);
    let full = ht_dcphase(&price);
    let mut state = ht_dcphase_state(&price[..150]).unwrap();
    // Collect each single-bar chained result, then compare the whole tail with the
    // bit-exact oracle (which reads both operands unconditionally on the happy path).
    let mut chained = Vec::with_capacity(price.len() - 150);
    for from in 150..price.len() {
        let (v, st) = ht_dcphase_resume(&price[..from + 1], from, &state).unwrap();
        assert_eq!(v.len(), 1);
        chained.push(v[0]);
        state = st;
    }
    assert_bits(&chained, &full[150..], "chained dcphase");
}

/// Resume guards: at/under the core warm-up every HT resume declines (`None` →
/// caller falls back); the price-windowed trendline / trendmode additionally
/// decline before a full dominant-cycle window is visible.
#[test]
fn ht_resume_declines_below_warmup() {
    let price = series(120);
    let st = ht_core_state(&price).unwrap();
    assert!(ht_dcperiod_resume(&price, CORE_START, &st).is_none());
    assert!(ht_phasor_resume(&price, false, CORE_START, &st).is_none());
    let st = ht_dcphase_state(&price).unwrap();
    assert!(ht_dcphase_resume(&price, CORE_START, &st).is_none());
    assert!(ht_sine_resume(&price, true, CORE_START, &st).is_none());
    // Trendline / trendmode need `from >= SMOOTH_PRICE_SIZE` for the raw-price window.
    let st = ht_trendline_state(&price).unwrap();
    assert!(ht_trendline_resume(&price, SMOOTH_PRICE_SIZE - 1, &st).is_none());
    let st = ht_trendmode_state(&price).unwrap();
    assert!(ht_trendmode_resume(&price, SMOOTH_PRICE_SIZE - 1, &st).is_none());
    // Under-warm-up series carry no state at all.
    let tiny = series(CORE_START);
    assert!(ht_core_state(&tiny).is_none());
    assert!(ht_dcphase_state(&tiny).is_none());
    assert!(ht_trendline_state(&tiny).is_none());
    assert!(ht_trendmode_state(&tiny).is_none());
    assert!(mama_state(&tiny, 0.5, 0.05).is_none());
}

/// Defensive state-length guards: every HT resume bails (`None`) when handed a state
/// shorter than its layout needs — a truncated `HtCore` (`deserialize`), or a core that
/// deserializes but lacks the trailing per-output rings/scalars. `from` is past each
/// resume's warm-up so the length check (not the warm-up check) is what declines.
#[test]
fn ht_resume_declines_on_short_state() {
    let price = series(120);

    // Too short for even `HtCore::deserialize` (< CORE_STATE_LEN) -> setup `?` (line 318)
    // propagates None. `from > CORE_START` so the from-guard passes first.
    let stub = vec![0.0; CORE_STATE_LEN - 1];
    assert!(ht_dcperiod_resume(&price, 20, &stub).is_none());

    // A bare core (exactly CORE_STATE_LEN) deserializes, but the per-output tail is
    // missing, so each family's length check declines.
    let core_only = vec![0.0; CORE_STATE_LEN];
    // DCPHASE / SINE need CORE_STATE_LEN + DCPHASE_STATE_LEN (lines 679 / 710).
    assert!(ht_dcphase_resume(&price, 20, &core_only).is_none());
    assert!(ht_sine_resume(&price, false, 20, &core_only).is_none());
    // TRENDLINE needs CORE_STATE_LEN + 3 (line 788); `from >= SMOOTH_PRICE_SIZE` so the
    // warm-up guard passes and the length guard is what fires.
    assert!(ht_trendline_resume(&price, SMOOTH_PRICE_SIZE, &core_only).is_none());
    // TRENDMODE needs CORE_STATE_LEN + DCPHASE_STATE_LEN + 5 (line 898).
    assert!(ht_trendmode_resume(&price, SMOOTH_PRICE_SIZE, &core_only).is_none());
    // MAMA / FAMA need CORE_STATE_LEN + 3 (line 998).
    assert!(mama_resume(&price, 0.5, 0.05, false, 20, &core_only).is_none());
}
