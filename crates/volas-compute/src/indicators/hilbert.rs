//! Hilbert-transform machinery and every indicator derived from it.
//!
//! TA-Lib's "Cycle Indicators" group (`HT_DCPERIOD`, `HT_DCPHASE`, `HT_PHASOR`,
//! `HT_SINE`, `HT_TRENDMODE`) and two overlap studies (`MAMA`/`FAMA`,
//! `HT_TRENDLINE`) all run one identical core: a 4-bar weighted moving-average
//! price smoother feeding a quadrature Hilbert transform (the `a=0.0962`,
//! `b=0.5769` 7-tap), a homodyne discriminator that estimates the dominant cycle
//! `period`, and a smoothed period. They differ only in (a) where output begins
//! (lookback 32 vs 63) and (b) the per-bar value they emit.
//!
//! We run the shared core once ([`ht_core`]) and each public function derives its
//! output from it, porting the recurrences verbatim from `ta_HT_*.c` / `ta_MAMA.c`
//! so results are bit-identical to TA-Lib 0.6.4. Warm-up entries are `NaN`.

mod engine;
use engine::*;

/// `HT_DCPERIOD`: the smoothed dominant-cycle period.
pub fn ht_dcperiod(price: &[f64]) -> Vec<f64> {
    let n = price.len();
    if n <= LB_PERIOD {
        return vec![f64::NAN; n];
    }
    let mut out = Vec::with_capacity(n);
    out.resize(LB_PERIOD, f64::NAN);
    let mut core = HtCoreState::seed(price);
    for i in CORE_START..LB_PERIOD {
        let _ = core.step(price, i);
    }
    for i in LB_PERIOD..n {
        // TA-Lib's HT_DCPERIOD writes the smoothed period directly from the
        // Hilbert loop; streaming here avoids materialising the shared HtBar vector.
        out.push(core.step(price, i).smooth_period);
    }
    out
}

/// `HT_PHASOR`: the in-phase and quadrature components `(inPhase, quadrature)`.
pub fn ht_phasor(price: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = price.len();
    let mut inphase = vec![f64::NAN; n];
    let mut quad = vec![f64::NAN; n];
    if n <= CORE_START {
        return (inphase, quad);
    }
    let mut core = HtCoreState::seed(price);
    for i in CORE_START..n {
        // PHASOR is also a direct TA-Lib Hilbert-loop output; keep both lines in
        // the same pass but skip the full-history HtBar scratch vector.
        let b = core.step(price, i);
        if i >= LB_PERIOD {
            inphase[i] = b.i1;
            quad[i] = b.q1;
        }
    }
    (inphase, quad)
}

/// Single-line `HT_PHASOR` used by directive execution. TA-Lib exposes both
/// arrays, but a directive requests one line at a time; avoid allocating and
/// writing the sibling line on that hot path.
pub fn ht_phasor_line(price: &[f64], quadrature: bool) -> Vec<f64> {
    let n = price.len();
    let mut out = vec![f64::NAN; n];
    if n <= CORE_START {
        return out;
    }
    let mut core = HtCoreState::seed(price);
    for i in CORE_START..n {
        let b = core.step(price, i);
        if i >= LB_PERIOD {
            out[i] = if quadrature { b.q1 } else { b.i1 };
        }
    }
    out
}

/// `HT_DCPHASE`: the dominant-cycle phase in degrees.
pub fn ht_dcphase(price: &[f64]) -> Vec<f64> {
    let core = ht_core(price);
    let phase = dcphase_all(&core);
    let mut out = vec![f64::NAN; price.len()];
    if price.len() > LB_PHASE {
        out[LB_PHASE..].copy_from_slice(&phase[LB_PHASE..price.len()]);
    }
    out
}

/// `HT_SINE`: `(sine, leadSine)` where `sine = sin(DCPhase)` and
/// `leadSine = sin(DCPhase + 45°)`.
pub fn ht_sine(price: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let core = ht_core(price);
    let phase = dcphase_all(&core);
    let n = price.len();
    let mut sine = vec![f64::NAN; n];
    let mut lead = vec![f64::NAN; n];
    for i in LB_PHASE..n {
        sine[i] = (phase[i] * DEG2RAD).sin();
        lead[i] = ((phase[i] + 45.0) * DEG2RAD).sin();
    }
    (sine, lead)
}

/// `HT_TRENDLINE`: the instantaneous trendline (a dominant-cycle-window average).
pub fn ht_trendline(price: &[f64]) -> Vec<f64> {
    let n = price.len();
    let mut out = vec![f64::NAN; n];
    if n <= CORE_START {
        return out;
    }
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0);
    for &p in price {
        prefix.push(prefix[prefix.len() - 1] + p);
    }
    let mut core = HtCoreState::seed(price);
    let (mut it1, mut it2, mut it3) = (0.0f64, 0.0f64, 0.0f64);
    for today in CORE_START..n {
        let b = core.step(price, today);
        // Keep the full-compute prefix-sum average while streaming the Hilbert core:
        // TA-Lib's iTrend recurrence is per-bar, but only bars >= LB_PHASE are visible.
        let dc = (b.smooth_period + 0.5) as usize;
        let lo = (today + 1).saturating_sub(dc);
        let avg = (prefix[today + 1] - prefix[lo]) / dc as f64;
        if today >= LB_PHASE {
            out[today] = (4.0 * avg + 3.0 * it1 + 2.0 * it2 + it3) / 10.0;
        }
        it3 = it2;
        it2 = it1;
        it1 = avg;
    }
    out
}

/// `HT_TRENDMODE`: `1.0` when the market is trending, `0.0` when cycling.
pub fn ht_trendmode(price: &[f64]) -> Vec<f64> {
    let core = ht_core(price);
    let phase = dcphase_all(&core);
    let trend_line = trendline_all(&core, price);
    let n = price.len();
    let mut out = vec![f64::NAN; n];
    let mut days_in_trend: i32 = 0;
    // Assume trend; demote to cycle on a fresh SineWave crossing, when too few days
    // have elapsed since the last crossing, or when the phase advances at roughly the
    // dominant-cycle rate; a strong price/trendline divergence forces trend mode. The
    // per-bar decision lives in `trendmode_decide` (shared with the resume).
    for i in CORE_START..n {
        let trend = trendmode_decide(
            phase[i],
            phase[i - 1],
            core[i].smooth_period,
            core[i].smoothed,
            trend_line[i],
            &mut days_in_trend,
        );
        if i >= LB_PHASE {
            out[i] = trend as f64;
        }
    }
    out
}

/// `MAMA`/`FAMA`: the MESA Adaptive Moving Average and its following adaptive MA,
/// returned as `(mama, fama)`. `fast_limit`/`slow_limit` bound the adaptive alpha
/// (TA-Lib defaults 0.5 / 0.05).
pub fn mama(price: &[f64], fast_limit: f64, slow_limit: f64) -> (Vec<f64>, Vec<f64>) {
    let n = price.len();
    let mut mama_out = vec![f64::NAN; n];
    let mut fama_out = vec![f64::NAN; n];
    if n <= CORE_START {
        return (mama_out, fama_out);
    }
    let mut mama = 0.0f64;
    let mut fama = 0.0f64;
    let mut prev_phase = 0.0f64;
    let mut core = HtCoreState::seed(price);
    for i in CORE_START..n {
        // TA-Lib advances the Hilbert core and MAMA recurrence in one pass; keeping
        // that order avoids the temporary `HtBar` vector without changing the
        // per-bar state transition used by parity and resume.
        let b = core.step(price, i);
        mama_step(
            &b,
            price[i],
            fast_limit,
            slow_limit,
            &mut mama,
            &mut fama,
            &mut prev_phase,
        );
        if i >= LB_PERIOD {
            mama_out[i] = mama;
            fama_out[i] = fama;
        }
    }
    (mama_out, fama_out)
}

/// Single-line `MAMA`/`FAMA` used by directive execution. The recurrence still
/// updates both MAMA accumulators exactly like TA-Lib; it only skips the sibling
/// output vector that the caller will discard.
pub fn mama_line(price: &[f64], fast_limit: f64, slow_limit: f64, want_fama: bool) -> Vec<f64> {
    let n = price.len();
    let mut out = vec![f64::NAN; n];
    if n <= CORE_START {
        return out;
    }
    let mut mama = 0.0f64;
    let mut fama = 0.0f64;
    let mut prev_phase = 0.0f64;
    let mut core = HtCoreState::seed(price);
    for i in CORE_START..n {
        let b = core.step(price, i);
        mama_step(
            &b,
            price[i],
            fast_limit,
            slow_limit,
            &mut mama,
            &mut fama,
            &mut prev_phase,
        );
        if i >= LB_PERIOD {
            out[i] = if want_fama { fama } else { mama };
        }
    }
    out
}

// --- state-carry: resume the Hilbert core (additive; fallback stays correct) ----
//
// Every HT output's whole history compresses into the shared [`HtCoreState`]
// (`CORE_STATE_LEN` words) plus a tiny per-output tail (a DC-phase ring, the
// trendline `iTrend` triple, MAMA/FAMA accumulators, …). `*_final_state` captures
// that state after a full compute; `*_resume` reconstructs it and steps the
// recurrence over only the new bars `[from, n)`, emitting values bit-identical to a
// fresh full recompute (the per-bar math is the single shared [`HtCoreState::step`]
// and the same post-pass body). A resume at or before the core warm-up
// (`from <= CORE_START`) — or, for the price-windowed trendline / trendmode, before
// a full dominant-cycle window is visible (`from < SMOOTH_PRICE_SIZE`) — returns
// `None`, so the caller transparently keeps the correct full-recompute fallback.

/// Capture [`HtCoreState`] after a full core pass over `price` (state as of bar
/// `n-1`). `None` for a series at/under the warm-up (no resumable state). Shared by
/// the core-only outputs (DCPERIOD / PHASOR) whose state is exactly the core.
fn ht_core_final_state(price: &[f64]) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= CORE_START {
        return None;
    }
    let mut core = HtCoreState::seed(price);
    for today in CORE_START..n {
        let _ = core.step(price, today);
    }
    Some(core.serialize())
}

/// Reconstruct [`HtCoreState`] for a resume at `from` (state as of bar `from-1`),
/// stepping no bars. `None` when `from <= CORE_START` (the carried state would
/// predate the warm-up) or the state buffer is too short.
fn ht_core_resume_setup(state: &[f64], from: usize) -> Option<HtCoreState> {
    if from <= CORE_START {
        return None;
    }
    HtCoreState::deserialize(state)
}

/// Final state for `HT_DCPERIOD` / `HT_PHASOR` (both read only the core's per-bar
/// outputs): exactly [`HtCoreState`].
pub fn ht_core_state(price: &[f64]) -> Option<Vec<f64>> {
    ht_core_final_state(price)
}

/// Resume `HT_DCPERIOD` over `[from, n)`: step the core, emit `smooth_period`.
/// Every emitted bar is past the `LB_PERIOD` mask (`from >= valid_rows > LB_PERIOD`).
pub fn ht_dcperiod_resume(
    price: &[f64],
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut core = ht_core_resume_setup(state, from)?;
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        out.push(core.step(price, today).smooth_period);
    }
    Some((out, core.serialize()))
}

/// Resume `HT_PHASOR` over `[from, n)`, emitting the in-phase (`quad == false`) or
/// quadrature (`quad == true`) component.
pub fn ht_phasor_resume(
    price: &[f64],
    quad: bool,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut core = ht_core_resume_setup(state, from)?;
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        out.push(if quad { b.q1 } else { b.i1 });
    }
    Some((out, core.serialize()))
}

/// Final state for `HT_DCPHASE` / `HT_SINE`: the core followed by the [`DcPhase`]
/// accumulator (`DCPHASE_STATE_LEN` words). `None` under the warm-up.
fn ht_dcphase_final_state(price: &[f64]) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= CORE_START {
        return None;
    }
    let mut core = HtCoreState::seed(price);
    let mut dc = DcPhase::new();
    let mut v = Vec::with_capacity(CORE_STATE_LEN + DCPHASE_STATE_LEN);
    for today in CORE_START..n {
        let b = core.step(price, today);
        let _ = dc.push(b.smoothed, b.smooth_period);
    }
    v.extend(core.serialize());
    dc.push_state(&mut v);
    Some(v)
}

/// Final state for `HT_DCPHASE` (alias used by the dispatch table).
pub fn ht_dcphase_state(price: &[f64]) -> Option<Vec<f64>> {
    ht_dcphase_final_state(price)
}

/// Resume `HT_DCPHASE` over `[from, n)`: step the core, push each smoothed bar
/// through the carried [`DcPhase`], emit the phase (all past the `LB_PHASE` mask).
pub fn ht_dcphase_resume(
    price: &[f64],
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut core = ht_core_resume_setup(state, from)?;
    let mut off = CORE_STATE_LEN;
    if state.len() < CORE_STATE_LEN + DCPHASE_STATE_LEN {
        return None;
    }
    let mut dc = DcPhase::from_state(state, &mut off);
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        out.push(dc.push(b.smoothed, b.smooth_period));
    }
    let mut v = core.serialize();
    dc.push_state(&mut v);
    Some((out, v))
}

/// Final state for `HT_SINE` — identical to [`ht_dcphase_state`] (sine/leadSine are
/// pure functions of the DC phase, carrying no extra state).
pub fn ht_sine_state(price: &[f64]) -> Option<Vec<f64>> {
    ht_dcphase_final_state(price)
}

/// Resume `HT_SINE` over `[from, n)`, emitting `sin(phase)` (`lead == false`) or
/// `sin(phase + 45°)` (`lead == true`).
pub fn ht_sine_resume(
    price: &[f64],
    lead: bool,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut core = ht_core_resume_setup(state, from)?;
    let mut off = CORE_STATE_LEN;
    if state.len() < CORE_STATE_LEN + DCPHASE_STATE_LEN {
        return None;
    }
    let mut dc = DcPhase::from_state(state, &mut off);
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        let phase = dc.push(b.smoothed, b.smooth_period);
        out.push(if lead {
            ((phase + 45.0) * DEG2RAD).sin()
        } else {
            (phase * DEG2RAD).sin()
        });
    }
    let mut v = core.serialize();
    dc.push_state(&mut v);
    Some((out, v))
}

/// Final state for `HT_TRENDLINE`: the core followed by the `iTrend` triple
/// `[it1, it2, it3]`. The price-window average reads raw `price[today-j]`
/// (`j < dcPeriod <= SMOOTH_PRICE_SIZE`), in range on a resume guarded by
/// `from >= SMOOTH_PRICE_SIZE`. `None` under the warm-up.
fn ht_trendline_final_state(price: &[f64]) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= CORE_START {
        return None;
    }
    let mut core = HtCoreState::seed(price);
    let (mut it1, mut it2, mut it3) = (0.0f64, 0.0f64, 0.0f64);
    for today in CORE_START..n {
        let b = core.step(price, today);
        let avg = trendline_avg(price, today, b.smooth_period);
        it3 = it2;
        it2 = it1;
        it1 = avg;
    }
    let mut v = core.serialize();
    v.push(it1);
    v.push(it2);
    v.push(it3);
    Some(v)
}

/// Final state for `HT_TRENDLINE` (alias used by the dispatch table).
pub fn ht_trendline_state(price: &[f64]) -> Option<Vec<f64>> {
    ht_trendline_final_state(price)
}

/// One trendline bar's dominant-cycle-window raw-price average (the inner sum of
/// [`trendline_all`], factored out so the full compute, `*_final_state`, and resume
/// share one definition).
#[inline]
fn trendline_avg(price: &[f64], today: usize, smooth_period: f64) -> f64 {
    let dc_period_int = (smooth_period + 0.5) as usize;
    let mut sum = 0.0f64;
    for j in 0..dc_period_int {
        if today >= j {
            sum += price[today - j];
        }
    }
    if dc_period_int > 0 {
        sum / dc_period_int as f64
    } else {
        0.0
    }
}

/// Resume `HT_TRENDLINE` over `[from, n)`: step the core, form the windowed average,
/// advance the `iTrend` recurrence, emit `(4·avg + 3·it1 + 2·it2 + it3)/10`. Guarded
/// by `from >= SMOOTH_PRICE_SIZE` so the raw-price window is fully in range (and the
/// dominant-cycle window is complete) on a head-dropping slice.
pub fn ht_trendline_resume(
    price: &[f64],
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from < SMOOTH_PRICE_SIZE {
        return None;
    }
    let mut core = ht_core_resume_setup(state, from)?;
    if state.len() < CORE_STATE_LEN + 3 {
        return None;
    }
    let (mut it1, mut it2, mut it3) = (
        state[CORE_STATE_LEN],
        state[CORE_STATE_LEN + 1],
        state[CORE_STATE_LEN + 2],
    );
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        let avg = trendline_avg(price, today, b.smooth_period);
        out.push((4.0 * avg + 3.0 * it1 + 2.0 * it2 + it3) / 10.0);
        it3 = it2;
        it2 = it1;
        it1 = avg;
    }
    let mut v = core.serialize();
    v.push(it1);
    v.push(it2);
    v.push(it3);
    Some((out, v))
}

/// One `HT_TRENDMODE` bar's trend/cycle decision, given this bar's DC phase / the
/// previous bar's DC phase, the smoothed price + period, the trendline value, and
/// the running `days_in_trend` (mutated). Factored from [`ht_trendmode`]'s loop so
/// the full compute and resume share one definition.
#[inline]
fn trendmode_decide(
    dc: f64,
    prev_dc: f64,
    smooth_period: f64,
    smoothed: f64,
    trend_line: f64,
    days_in_trend: &mut i32,
) -> i32 {
    let sine = (dc * DEG2RAD).sin();
    let lead = ((dc + 45.0) * DEG2RAD).sin();
    let prev_sine = (prev_dc * DEG2RAD).sin();
    let prev_lead = ((prev_dc + 45.0) * DEG2RAD).sin();
    let sp = smooth_period;
    let mut trend = 1i32;
    if (sine > lead && prev_sine <= prev_lead) || (sine < lead && prev_sine >= prev_lead) {
        *days_in_trend = 0;
        trend = 0;
    }
    *days_in_trend += 1;
    if (*days_in_trend as f64) < 0.5 * sp {
        trend = 0;
    }
    let dphase = dc - prev_dc;
    if sp != 0.0 && dphase > 0.67 * 360.0 / sp && dphase < 1.5 * 360.0 / sp {
        trend = 0;
    }
    if trend_line != 0.0 && ((smoothed - trend_line) / trend_line).abs() >= 0.015 {
        trend = 1;
    }
    trend
}

/// Final state for `HT_TRENDMODE`: core ‖ [`DcPhase`] ‖ `iTrend` triple ‖
/// `[days_in_trend, prev_dc_phase]`. The carried `DcPhase.phase` already equals
/// `phase[n-1]`, but the explicit `prev_dc_phase` keeps the resume's first-bar
/// `phase[from-1]` read self-contained. `None` under the warm-up.
fn ht_trendmode_final_state(price: &[f64]) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= CORE_START {
        return None;
    }
    let mut core = HtCoreState::seed(price);
    let mut dc = DcPhase::new();
    let (mut it1, mut it2, mut it3) = (0.0f64, 0.0f64, 0.0f64);
    let mut days_in_trend: i32 = 0;
    let mut prev_phase = 0.0f64;
    for today in CORE_START..n {
        let b = core.step(price, today);
        let phase = dc.push(b.smoothed, b.smooth_period);
        let avg = trendline_avg(price, today, b.smooth_period);
        let trend_line = (4.0 * avg + 3.0 * it1 + 2.0 * it2 + it3) / 10.0;
        let _ = trendmode_decide(
            phase,
            prev_phase,
            b.smooth_period,
            b.smoothed,
            trend_line,
            &mut days_in_trend,
        );
        it3 = it2;
        it2 = it1;
        it1 = avg;
        prev_phase = phase;
    }
    let mut v = core.serialize();
    dc.push_state(&mut v);
    v.push(it1);
    v.push(it2);
    v.push(it3);
    v.push(days_in_trend as f64);
    v.push(prev_phase);
    Some(v)
}

/// Final state for `HT_TRENDMODE` (alias used by the dispatch table).
pub fn ht_trendmode_state(price: &[f64]) -> Option<Vec<f64>> {
    ht_trendmode_final_state(price)
}

/// Resume `HT_TRENDMODE` over `[from, n)`. Steps the core, reconstructs the DC phase +
/// trendline, runs the shared decision with the carried `days_in_trend`, and
/// emits `0.0`/`1.0`. Guarded by `from >= SMOOTH_PRICE_SIZE` (trendline window).
pub fn ht_trendmode_resume(
    price: &[f64],
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from < SMOOTH_PRICE_SIZE {
        return None;
    }
    let mut core = ht_core_resume_setup(state, from)?;
    let tail = CORE_STATE_LEN + DCPHASE_STATE_LEN;
    if state.len() < tail + 5 {
        return None;
    }
    let mut off = CORE_STATE_LEN;
    let mut dc = DcPhase::from_state(state, &mut off);
    let (mut it1, mut it2, mut it3) = (state[tail], state[tail + 1], state[tail + 2]);
    let mut days_in_trend = state[tail + 3] as i32;
    let mut prev_phase = state[tail + 4];
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        let phase = dc.push(b.smoothed, b.smooth_period);
        let avg = trendline_avg(price, today, b.smooth_period);
        let trend_line = (4.0 * avg + 3.0 * it1 + 2.0 * it2 + it3) / 10.0;
        let trend = trendmode_decide(
            phase,
            prev_phase,
            b.smooth_period,
            b.smoothed,
            trend_line,
            &mut days_in_trend,
        );
        out.push(trend as f64);
        it3 = it2;
        it2 = it1;
        it1 = avg;
        prev_phase = phase;
    }
    let mut v = core.serialize();
    dc.push_state(&mut v);
    v.push(it1);
    v.push(it2);
    v.push(it3);
    v.push(days_in_trend as f64);
    v.push(prev_phase);
    Some((out, v))
}

/// Final state for `MAMA`/`FAMA`: core ‖ `[mama, fama, prev_phase]`. The adaptive
/// `alpha` is a pure function of the (carried) core's `i1`/`q1`, so only the two
/// running averages and the previous phase are carried. `None` under the warm-up.
fn mama_final_state(price: &[f64], fast_limit: f64, slow_limit: f64) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= CORE_START {
        return None;
    }
    let mut core = HtCoreState::seed(price);
    let (mut mama, mut fama, mut prev_phase) = (0.0f64, 0.0f64, 0.0f64);
    for today in CORE_START..n {
        let b = core.step(price, today);
        mama_step(
            &b,
            price[today],
            fast_limit,
            slow_limit,
            &mut mama,
            &mut fama,
            &mut prev_phase,
        );
    }
    let mut v = core.serialize();
    v.push(mama);
    v.push(fama);
    v.push(prev_phase);
    Some(v)
}

/// Final state for `MAMA`/`FAMA` (alias used by the dispatch table).
pub fn mama_state(price: &[f64], fast_limit: f64, slow_limit: f64) -> Option<Vec<f64>> {
    mama_final_state(price, fast_limit, slow_limit)
}

/// Advance the MAMA/FAMA recurrence by one bar (the verbatim body of [`mama`]'s
/// loop), mutating `mama`/`fama`/`prev_phase`. Shared by the full compute,
/// `*_final_state`, and the resume so continuation is bit-identical.
#[inline]
fn mama_step(
    b: &HtBar,
    price_today: f64,
    fast_limit: f64,
    slow_limit: f64,
    mama: &mut f64,
    fama: &mut f64,
    prev_phase: &mut f64,
) {
    let phase = if b.i1 != 0.0 {
        (b.q1 / b.i1).atan() * RAD2DEG
    } else {
        0.0
    };
    let mut delta = *prev_phase - phase;
    *prev_phase = phase;
    if delta < 1.0 {
        delta = 1.0;
    }
    let alpha = if delta > 1.0 {
        (fast_limit / delta).max(slow_limit)
    } else {
        fast_limit
    };
    *mama = alpha * price_today + (1.0 - alpha) * *mama;
    let half = 0.5 * alpha;
    *fama = half * *mama + (1.0 - half) * *fama;
}

/// Resume `MAMA`/`FAMA` over `[from, n)`, emitting `mama` (`want_fama == false`) or
/// `fama` (`want_fama == true`); both are past the `LB_PERIOD` mask.
pub fn mama_resume(
    price: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    want_fama: bool,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut core = ht_core_resume_setup(state, from)?;
    if state.len() < CORE_STATE_LEN + 3 {
        return None;
    }
    let (mut mama, mut fama, mut prev_phase) = (
        state[CORE_STATE_LEN],
        state[CORE_STATE_LEN + 1],
        state[CORE_STATE_LEN + 2],
    );
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        mama_step(
            &b,
            price[today],
            fast_limit,
            slow_limit,
            &mut mama,
            &mut fama,
            &mut prev_phase,
        );
        out.push(if want_fama { fama } else { mama });
    }
    let mut v = core.serialize();
    v.push(mama);
    v.push(fama);
    v.push(prev_phase);
    Some((out, v))
}


#[cfg(test)]
#[path = "hilbert_tests.rs"]
mod hilbert_tests;
