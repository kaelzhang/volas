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
    let mut core = HtCoreState::seed(price, PERIOD_START);
    for i in PERIOD_START..LB_PERIOD {
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
    if n <= PERIOD_START {
        return (inphase, quad);
    }
    let mut core = HtCoreState::seed(price, PERIOD_START);
    for i in PERIOD_START..n {
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
    if n <= PERIOD_START {
        return out;
    }
    let mut core = HtCoreState::seed(price, PERIOD_START);
    #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
    for i in PERIOD_START..n {
        let b = core.step(price, i);
        if i >= LB_PERIOD {
            out[i] = if quadrature { b.q1 } else { b.i1 };
        }
    }
    out
}

/// `HT_DCPHASE`: the dominant-cycle phase in degrees.
pub fn ht_dcphase(price: &[f64]) -> Vec<f64> {
    let core = ht_core(price, PHASE_START);
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
    let core = ht_core(price, PHASE_START);
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
    if n <= PHASE_START {
        return out;
    }
    let mut core = HtCoreState::seed(price, PHASE_START);
    let mut itrend = ITrend::default();
    for (today, slot) in out.iter_mut().enumerate().skip(PHASE_START) {
        let b = core.step(price, today);
        // TA-Lib's iTrend recurrence is per-bar, but only bars >= LB_PHASE are visible.
        let v = itrend.push(price, today, b.smooth_period);
        if today >= LB_PHASE {
            *slot = v;
        }
    }
    out
}

/// `HT_TRENDMODE`: `1.0` when the market is trending, `0.0` when cycling.
pub fn ht_trendmode(price: &[f64]) -> Vec<f64> {
    let n = price.len();
    let mut out = vec![f64::NAN; n];
    if n <= PHASE_START {
        return out;
    }
    let mut run = TrendModeRun::seed(price);
    for (today, slot) in out.iter_mut().enumerate().skip(PHASE_START) {
        let trend = run.step(price, today);
        if today >= LB_PHASE {
            *slot = trend;
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
    if n <= PERIOD_START {
        return (mama_out, fama_out);
    }
    let mut mama = 0.0f64;
    let mut fama = 0.0f64;
    let mut prev_phase = 0.0f64;
    let mut core = HtCoreState::seed(price, PERIOD_START);
    for i in PERIOD_START..n {
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
    if n <= PERIOD_START {
        return out;
    }
    let mut mama = 0.0f64;
    let mut fama = 0.0f64;
    let mut prev_phase = 0.0f64;
    let mut core = HtCoreState::seed(price, PERIOD_START);
    for i in PERIOD_START..n {
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
// (`from <=` the family's first recurrence bar) — or, for the price-windowed trendline / trendmode, before
// a full dominant-cycle window is visible (`from < SMOOTH_PRICE_SIZE`) — returns
// `None`, so the caller transparently keeps the correct full-recompute fallback.

/// Capture [`HtCoreState`] after a full core pass over `price` (state as of bar
/// `n-1`). `None` for a series at/under the warm-up (no resumable state). Shared by
/// the core-only outputs (DCPERIOD / PHASOR) whose state is exactly the core.
fn ht_core_final_state(price: &[f64]) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= PERIOD_START {
        return None;
    }
    let mut core = HtCoreState::seed(price, PERIOD_START);
    for today in PERIOD_START..n {
        let _ = core.step(price, today);
    }
    Some(core.serialize())
}

/// A resumed bar's output under its full-compute mask: `NaN` below `mask`, else `v`.
/// The bar index is that of the complete (gathered) rows, so missing rows can shift
/// a resumed bar under the mask that a gap-free history would have passed.
#[inline]
fn masked(today: usize, mask: usize, v: f64) -> f64 {
    if today >= mask {
        v
    } else {
        f64::NAN
    }
}

/// Reconstruct [`HtCoreState`] for a resume at `from` (state as of bar `from-1`),
/// stepping no bars. `None` when `from <= start` (the family's first recurrence bar:
/// the carried state would predate the warm-up) or the state buffer is too short.
fn ht_core_resume_setup(state: &[f64], from: usize, start: usize) -> Option<HtCoreState> {
    if from <= start {
        return None;
    }
    HtCoreState::deserialize(state)
}

/// Final state for `HT_DCPERIOD` / `HT_PHASOR` (both read only the core's per-bar
/// outputs): exactly [`HtCoreState`].
pub fn ht_core_state(price: &[f64]) -> Option<Vec<f64>> {
    ht_core_final_state(price)
}

/// Resume `HT_DCPERIOD` over `[from, n)`: step the core, emit `smooth_period`,
/// masked under `LB_PERIOD` like the full compute (a bar index there is reachable
/// once missing rows shift the warm-up).
pub fn ht_dcperiod_resume(
    price: &[f64],
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut core = ht_core_resume_setup(state, from, PERIOD_START)?;
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let p = core.step(price, today).smooth_period;
        out.push(masked(today, LB_PERIOD, p));
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
    let mut core = ht_core_resume_setup(state, from, PERIOD_START)?;
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        out.push(masked(today, LB_PERIOD, if quad { b.q1 } else { b.i1 }));
    }
    Some((out, core.serialize()))
}

/// Final state for `HT_DCPHASE` / `HT_SINE`: the core followed by the [`DcPhase`]
/// accumulator (`DCPHASE_STATE_LEN` words). `None` under the warm-up.
fn ht_dcphase_final_state(price: &[f64]) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= PHASE_START {
        return None;
    }
    let mut core = HtCoreState::seed(price, PHASE_START);
    let mut dc = DcPhase::new();
    let mut v = Vec::with_capacity(CORE_STATE_LEN + DCPHASE_STATE_LEN);
    for today in PHASE_START..n {
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
/// through the carried [`DcPhase`], emit the phase (masked under `LB_PHASE`).
pub fn ht_dcphase_resume(
    price: &[f64],
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut core = ht_core_resume_setup(state, from, PHASE_START)?;
    let mut off = CORE_STATE_LEN;
    if state.len() < CORE_STATE_LEN + DCPHASE_STATE_LEN {
        return None;
    }
    let mut dc = DcPhase::from_state(state, &mut off)?;
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        out.push(masked(today, LB_PHASE, dc.push(b.smoothed, b.smooth_period)));
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
    let mut core = ht_core_resume_setup(state, from, PHASE_START)?;
    let mut off = CORE_STATE_LEN;
    if state.len() < CORE_STATE_LEN + DCPHASE_STATE_LEN {
        return None;
    }
    let mut dc = DcPhase::from_state(state, &mut off)?;
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        let phase = dc.push(b.smoothed, b.smooth_period);
        let v = if lead {
            ((phase + 45.0) * DEG2RAD).sin()
        } else {
            (phase * DEG2RAD).sin()
        };
        out.push(masked(today, LB_PHASE, v));
    }
    let mut v = core.serialize();
    dc.push_state(&mut v);
    Some((out, v))
}

/// Final state for `HT_TRENDLINE`: the core followed by the [`ITrend`] triple. The
/// price-window average reads raw `price[today-j]` (`j < dcPeriod <= SMOOTH_PRICE_SIZE`),
/// in range on a resume guarded by `from >= SMOOTH_PRICE_SIZE`. `None` under the warm-up.
fn ht_trendline_final_state(price: &[f64]) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= PHASE_START {
        return None;
    }
    let mut core = HtCoreState::seed(price, PHASE_START);
    let mut itrend = ITrend::default();
    for today in PHASE_START..n {
        let b = core.step(price, today);
        let _ = itrend.push(price, today, b.smooth_period);
    }
    let mut v = core.serialize();
    itrend.push_state(&mut v);
    Some(v)
}

/// Final state for `HT_TRENDLINE` (alias used by the dispatch table).
pub fn ht_trendline_state(price: &[f64]) -> Option<Vec<f64>> {
    ht_trendline_final_state(price)
}

/// Resume `HT_TRENDLINE` over `[from, n)`: step the core and the carried [`ITrend`].
/// Guarded by `from >= SMOOTH_PRICE_SIZE` so the raw-price window is fully in range
/// (and the dominant-cycle window is complete) on a head-dropping slice.
pub fn ht_trendline_resume(
    price: &[f64],
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from < SMOOTH_PRICE_SIZE {
        return None;
    }
    let mut core = ht_core_resume_setup(state, from, PHASE_START)?;
    let mut off = CORE_STATE_LEN;
    let mut itrend = ITrend::from_state(state, &mut off)?;
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        out.push(masked(today, LB_PHASE, itrend.push(price, today, b.smooth_period)));
    }
    let mut v = core.serialize();
    itrend.push_state(&mut v);
    Some((out, v))
}

/// `HT_TRENDMODE`'s trend / cycle decision state: the days since the last SineWave
/// crossing, the previous bar's DC phase, and the previous bar's sine / lead sine —
/// carried as values, as TA-Lib carries them (both start at `0`, so the first
/// recurrence bar compares against `0 / 0`, not against `sin(0°) / sin(45°)`).
#[derive(Clone, Copy, Default)]
struct TrendMode {
    days_in_trend: i32,
    prev_phase: f64,
    sine: f64,
    lead: f64,
}

impl TrendMode {
    const STATE_LEN: usize = 4;

    /// One bar's decision from its DC phase, smoothed period, smoothed price and
    /// trendline (TA-Lib `HT_TRENDMODE`). Assume trend; demote to cycle on a fresh
    /// SineWave crossing, when too few days have elapsed since the last crossing, or
    /// when the phase advances at roughly the dominant-cycle rate; a strong
    /// price / trendline divergence forces trend mode.
    #[inline]
    fn decide(&mut self, phase: f64, smooth_period: f64, smoothed: f64, trend_line: f64) -> i32 {
        let (prev_sine, prev_lead) = (self.sine, self.lead);
        self.sine = (phase * DEG2RAD).sin();
        self.lead = ((phase + 45.0) * DEG2RAD).sin();
        let (sine, lead, sp) = (self.sine, self.lead, smooth_period);
        let mut trend = 1i32;
        if (sine > lead && prev_sine <= prev_lead) || (sine < lead && prev_sine >= prev_lead) {
            self.days_in_trend = 0;
            trend = 0;
        }
        self.days_in_trend += 1;
        if (self.days_in_trend as f64) < 0.5 * sp {
            trend = 0;
        }
        let dphase = phase - self.prev_phase;
        self.prev_phase = phase;
        if sp != 0.0 && dphase > 0.67 * 360.0 / sp && dphase < 1.5 * 360.0 / sp {
            trend = 0;
        }
        if trend_line != 0.0 && ((smoothed - trend_line) / trend_line).abs() >= 0.015 {
            trend = 1;
        }
        trend
    }
}

/// The whole `HT_TRENDMODE` recurrence — core, [`DcPhase`], [`ITrend`] and
/// [`TrendMode`] — stepped one bar at a time by the full compute, the final state and
/// the resume alike. Serialised as core ‖ `DcPhase` ‖ `ITrend` ‖
/// `[days_in_trend, prev_phase, sine, lead]`.
struct TrendModeRun {
    core: HtCoreState,
    dc: DcPhase,
    itrend: ITrend,
    mode: TrendMode,
}

impl TrendModeRun {
    const STATE_LEN: usize = CORE_STATE_LEN + DCPHASE_STATE_LEN + ITrend::STATE_LEN + TrendMode::STATE_LEN;

    /// The recurrence positioned at bar [`PHASE_START`]; requires `price.len() > PHASE_START`.
    fn seed(price: &[f64]) -> Self {
        TrendModeRun {
            core: HtCoreState::seed(price, PHASE_START),
            dc: DcPhase::new(),
            itrend: ITrend::default(),
            mode: TrendMode::default(),
        }
    }

    /// Bar `today`'s trend mode (`1.0` / `0.0`), advancing every part.
    #[inline]
    fn step(&mut self, price: &[f64], today: usize) -> f64 {
        let b = self.core.step(price, today);
        let phase = self.dc.push(b.smoothed, b.smooth_period);
        let trend_line = self.itrend.push(price, today, b.smooth_period);
        self.mode.decide(phase, b.smooth_period, b.smoothed, trend_line) as f64
    }

    fn serialize(&self) -> Vec<f64> {
        let mut v = self.core.serialize();
        self.dc.push_state(&mut v);
        self.itrend.push_state(&mut v);
        let m = &self.mode;
        v.extend_from_slice(&[m.days_in_trend as f64, m.prev_phase, m.sine, m.lead]);
        v
    }

    /// Inverse of [`serialize`](Self::serialize) for a resume at `from`; `None` at or
    /// before the warm-up, or for a state of the wrong length or a forged cursor.
    fn deserialize(state: &[f64], from: usize) -> Option<Self> {
        if state.len() != Self::STATE_LEN {
            return None;
        }
        let core = ht_core_resume_setup(state, from, PHASE_START)?;
        let mut off = CORE_STATE_LEN;
        let dc = DcPhase::from_state(state, &mut off)?;
        let itrend = ITrend::from_state(state, &mut off)?;
        let &[days, prev_phase, sine, lead] = &state[off..] else { return None };
        let mode = TrendMode { days_in_trend: days as i32, prev_phase, sine, lead };
        Some(TrendModeRun { core, dc, itrend, mode })
    }
}

/// Final state for `HT_TRENDMODE` ([`TrendModeRun`]'s). `None` under the warm-up.
pub fn ht_trendmode_state(price: &[f64]) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= PHASE_START {
        return None;
    }
    let mut run = TrendModeRun::seed(price);
    for today in PHASE_START..n {
        let _ = run.step(price, today);
    }
    Some(run.serialize())
}

/// Resume `HT_TRENDMODE` over `[from, n)` from the carried [`TrendModeRun`], emitting
/// `0.0` / `1.0`. Guarded by `from >= SMOOTH_PRICE_SIZE` (trendline window).
pub fn ht_trendmode_resume(
    price: &[f64],
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if from < SMOOTH_PRICE_SIZE {
        return None;
    }
    let mut run = TrendModeRun::deserialize(state, from)?;
    let n = price.len();
    let out = (from..n).map(|today| masked(today, LB_PHASE, run.step(price, today))).collect();
    Some((out, run.serialize()))
}

/// Final state for `MAMA`/`FAMA`: core ‖ `[mama, fama, prev_phase]`. The adaptive
/// `alpha` is a pure function of the (carried) core's `i1`/`q1`, so only the two
/// running averages and the previous phase are carried. `None` under the warm-up.
fn mama_final_state(price: &[f64], fast_limit: f64, slow_limit: f64) -> Option<Vec<f64>> {
    let n = price.len();
    if n <= PERIOD_START {
        return None;
    }
    let mut core = HtCoreState::seed(price, PERIOD_START);
    let (mut mama, mut fama, mut prev_phase) = (0.0f64, 0.0f64, 0.0f64);
    for today in PERIOD_START..n {
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
    *mama = alpha.mul_add(price_today, (1.0 - alpha) * *mama);
    let half = 0.5 * alpha;
    *fama = half.mul_add(*mama, (1.0 - half) * *fama);
}

/// Resume `MAMA`/`FAMA` over `[from, n)`, emitting `mama` (`want_fama == false`) or
/// `fama` (`want_fama == true`), masked under `LB_PERIOD` like the full compute.
pub fn mama_resume(
    price: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    want_fama: bool,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut core = ht_core_resume_setup(state, from, PERIOD_START)?;
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
        out.push(masked(today, LB_PERIOD, if want_fama { fama } else { mama }));
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
