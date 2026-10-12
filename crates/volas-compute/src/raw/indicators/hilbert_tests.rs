use super::*;

#[test]
fn ht_core_short_input_returns_warmup_only() {
    // Inputs at or below the WMA warm-up never enter the main loop (covers the
    // early return); every bar stays at the HtBar default.
    for (n, start) in [(0usize, PERIOD_START), (1, PERIOD_START), (PERIOD_START, PERIOD_START), (PHASE_START, PHASE_START)] {
        let price: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let bars = ht_core(&price, start);
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
        ((1.0 * 1f64.atan() * 8.0) / 2.0).cos(),
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

/// A resume that starts under an output's warm-up mask (reachable once missing rows
/// shift the gathered bar index down) emits NaN there, as the full compute does, and
/// the full values past it.
#[test]
fn ht_resume_under_the_mask_is_nan_like_the_full_compute() {
    let price = series(120);
    let from = 20; // past PERIOD_START, under LB_PERIOD
    let head = &price[..from];
    let st = ht_core_state(head).unwrap();
    let mut tail = Vec::new();
    ht_dcperiod_resume(&price, from, &st, &mut tail).unwrap();
    assert_bits(&tail, &ht_dcperiod(&price)[from..], "dcperiod");
    let mut tail = Vec::new();
    ht_phasor_resume(&price, true, from, &st, &mut tail).unwrap();
    assert_bits(&tail, &ht_phasor(&price).1[from..], "phasor");
    let st = mama_state(head, 0.5, 0.05).unwrap();
    let mut tail = Vec::new();
    mama_resume(&price, 0.5, 0.05, false, from, &st, &mut tail).unwrap();
    assert_bits(&tail, &mama(&price, 0.5, 0.05).0[from..], "mama");
    let from = 40; // past PHASE_START, under LB_PHASE
    let st = ht_dcphase_state(&price[..from]).unwrap();
    let mut tail = Vec::new();
    ht_dcphase_resume(&price, from, &st, &mut tail).unwrap();
    assert_bits(&tail, &ht_dcphase(&price)[from..], "dcphase");
    let mut tail = Vec::new();
    ht_sine_resume(&price, true, from, &st, &mut tail).unwrap();
    assert_bits(&tail, &ht_sine(&price).1[from..], "sine");
    // the trendline family's raw-price window needs `from >= SMOOTH_PRICE_SIZE`
    let from = 55; // under LB_PHASE
    let st = ht_trendline_state(&price[..from]).unwrap();
    let mut tail = Vec::new();
    ht_trendline_resume(&price, from, &st, &mut tail).unwrap();
    assert_bits(&tail, &ht_trendline(&price)[from..], "trendline");
    let st = ht_trendmode_state(&price[..from]).unwrap();
    let mut tail = Vec::new();
    ht_trendmode_resume(&price, from, &st, &mut tail).unwrap();
    assert_bits(&tail, &ht_trendmode(&price)[from..], "trendmode");
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
        let mut v = Vec::new();
        let _ = ht_dcperiod_resume(&price, from, &st, &mut v).unwrap();
        assert_bits(&v, &ht_dcperiod(&price)[from..], "dcperiod");
        let mut v = Vec::new();
        let _ = ht_phasor_resume(&price, false, from, &st, &mut v).unwrap();
        assert_bits(&v, &ht_phasor(&price).0[from..], "phasor.inphase");
        let mut v = Vec::new();
        let _ = ht_phasor_resume(&price, true, from, &st, &mut v).unwrap();
        assert_bits(&v, &ht_phasor(&price).1[from..], "phasor.quadrature");

        // DCPHASE / SINE (core + DC-phase ring).
        let st = ht_dcphase_state(head).unwrap();
        let mut v = Vec::new();
        let _ = ht_dcphase_resume(&price, from, &st, &mut v).unwrap();
        assert_bits(&v, &ht_dcphase(&price)[from..], "dcphase");
        let st = ht_sine_state(head).unwrap();
        let mut v = Vec::new();
        let _ = ht_sine_resume(&price, false, from, &st, &mut v).unwrap();
        assert_bits(&v, &ht_sine(&price).0[from..], "sine.sine");
        let mut v = Vec::new();
        let _ = ht_sine_resume(&price, true, from, &st, &mut v).unwrap();
        assert_bits(&v, &ht_sine(&price).1[from..], "sine.leadsine");

        // TRENDLINE / TRENDMODE (core + iTrend triple [+ DC-phase / counters]).
        let st = ht_trendline_state(head).unwrap();
        let mut v = Vec::new();
        let _ = ht_trendline_resume(&price, from, &st, &mut v).unwrap();
        assert_bits(&v, &ht_trendline(&price)[from..], "trendline");
        let st = ht_trendmode_state(head).unwrap();
        let mut v = Vec::new();
        let _ = ht_trendmode_resume(&price, from, &st, &mut v).unwrap();
        assert_bits(&v, &ht_trendmode(&price)[from..], "trendmode");

        // MAMA / FAMA (core + [mama, fama, prev_phase]).
        let st = mama_state(head, 0.5, 0.05).unwrap();
        let mut v = Vec::new();
        let _ = mama_resume(&price, 0.5, 0.05, false, from, &st, &mut v).unwrap();
        assert_bits(&v, &mama(&price, 0.5, 0.05).0[from..], "mama");
        let mut v = Vec::new();
        let _ = mama_resume(&price, 0.5, 0.05, true, from, &st, &mut v).unwrap();
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
        let mut v = Vec::new();
        let st = ht_dcphase_resume(&price[..from + 1], from, &state, &mut v).unwrap();
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
    assert!(ht_dcperiod_resume(&price, PERIOD_START, &st, &mut Vec::new()).is_none());
    assert!(ht_phasor_resume(&price, false, PERIOD_START, &st, &mut Vec::new()).is_none());
    let st = ht_dcphase_state(&price).unwrap();
    assert!(ht_dcphase_resume(&price, PHASE_START, &st, &mut Vec::new()).is_none());
    assert!(ht_sine_resume(&price, true, PHASE_START, &st, &mut Vec::new()).is_none());
    // Trendline / trendmode need `from >= SMOOTH_PRICE_SIZE` for the raw-price window.
    let st = ht_trendline_state(&price).unwrap();
    assert!(ht_trendline_resume(&price, SMOOTH_PRICE_SIZE - 1, &st, &mut Vec::new()).is_none());
    let st = ht_trendmode_state(&price).unwrap();
    assert!(ht_trendmode_resume(&price, SMOOTH_PRICE_SIZE - 1, &st, &mut Vec::new()).is_none());
    // Under-warm-up series carry no state at all.
    let tiny = series(PERIOD_START);
    assert!(ht_core_state(&tiny).is_none());
    assert!(mama_state(&tiny, 0.5, 0.05).is_none());
    // The phase family's recurrence starts later, at PHASE_START.
    let tiny = series(PHASE_START);
    assert!(ht_dcphase_state(&tiny).is_none());
    assert!(ht_trendline_state(&tiny).is_none());
    assert!(ht_trendmode_state(&tiny).is_none());
}

/// Defensive state-length guards: every HT resume bails (`None`) when handed a state
/// shorter than its layout needs — a truncated `HtCore` (`deserialize`), or a core that
/// deserializes but lacks the trailing per-output rings/scalars. `from` is past each
/// resume's warm-up so the length check (not the warm-up check) is what declines.
#[test]
fn ht_resume_declines_on_short_state() {
    let price = series(120);

    // Too short for even `HtCore::deserialize` (< CORE_STATE_LEN) -> setup `?`
    // propagates None. `from > PERIOD_START` so the from-guard passes first.
    let stub = vec![0.0; CORE_STATE_LEN - 1];
    assert!(ht_dcperiod_resume(&price, 20, &stub, &mut Vec::new()).is_none());

    // A bare core (exactly CORE_STATE_LEN) deserializes, but the per-output tail is
    // missing, so each family's length check declines.
    let core_only = vec![0.0; CORE_STATE_LEN];
    // DCPHASE / SINE need CORE_STATE_LEN + DCPHASE_STATE_LEN.
    assert!(ht_dcphase_resume(&price, 40, &core_only, &mut Vec::new()).is_none());
    assert!(ht_sine_resume(&price, false, 40, &core_only, &mut Vec::new()).is_none());
    // TRENDLINE needs CORE_STATE_LEN + 3; `from >= SMOOTH_PRICE_SIZE` so the
    // warm-up guard passes and the length guard is what fires.
    assert!(ht_trendline_resume(&price, SMOOTH_PRICE_SIZE, &core_only, &mut Vec::new()).is_none());
    // TRENDMODE needs exactly CORE_STATE_LEN + DCPHASE_STATE_LEN + 3 + 4.
    assert!(ht_trendmode_resume(&price, SMOOTH_PRICE_SIZE, &core_only, &mut Vec::new()).is_none());
    // MAMA / FAMA need CORE_STATE_LEN + 3.
    assert!(mama_resume(&price, 0.5, 0.05, false, 20, &core_only, &mut Vec::new()).is_none());
}

/// A carried state no computation wrote — a ring cursor or DC-phase index out of its
/// ring, a parity not 0 / 1 — is refused by every resume, never used to index.
#[test]
fn a_forged_state_is_refused() {
    let price = series(120);
    let head = &price[..80];
    let forge = |mut st: Vec<f64>, at: usize, v: f64| {
        st[at] = v;
        st
    };
    let core = ht_core_state(head).unwrap();
    for (at, v) in [(3, 3.0), (3, 1e9), (3, -1.0), (3, 0.5), (3, f64::NAN), (4, 0.5), (4, 2.0)] {
        let st = forge(core.clone(), at, v);
        assert!(ht_dcperiod_resume(&price, 80, &st, &mut Vec::new()).is_none(), "cursor {at}={v}");
        assert!(ht_phasor_resume(&price, true, 80, &st, &mut Vec::new()).is_none());
    }
    let mama_st = forge(mama_state(head, 0.5, 0.05).unwrap(), 3, 7.0);
    assert!(mama_resume(&price, 0.5, 0.05, false, 80, &mama_st, &mut Vec::new()).is_none());
    let tl = forge(ht_trendline_state(head).unwrap(), 3, 4.0);
    assert!(ht_trendline_resume(&price, 80, &tl, &mut Vec::new()).is_none());
    // the DC-phase ring index sits right after the core and the phase
    let dc = ht_dcphase_state(head).unwrap();
    for v in [50.0, 1e12, -2.0, 1.5] {
        let st = forge(dc.clone(), CORE_STATE_LEN + 1, v);
        assert!(ht_dcphase_resume(&price, 80, &st, &mut Vec::new()).is_none(), "dc idx {v}");
        assert!(ht_sine_resume(&price, false, 80, &st, &mut Vec::new()).is_none());
    }
    let tm = forge(ht_trendmode_state(head).unwrap(), CORE_STATE_LEN + 1, 99.0);
    assert!(ht_trendmode_resume(&price, 80, &tm, &mut Vec::new()).is_none());
    // a well-formed state still resumes
    assert!(ht_dcphase_resume(&price, 80, &dc, &mut Vec::new()).is_some());
}
