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

use std::f64::consts::PI;

/// Hilbert weighting coefficients (`ta_utility.h`'s `DO_HILBERT_TRANSFORM`).
const A: f64 = 0.0962;
const B: f64 = 0.5769;
/// `rad2Deg = 45/atan(1) = 180/π`; `deg2Rad = 1/rad2Deg`; `constDeg2RadBy360 = atan(1)·8 = 2π`.
const RAD2DEG: f64 = 180.0 / PI;
const DEG2RAD: f64 = PI / 180.0;
const TWO_PI: f64 = 2.0 * PI;

/// TA-Lib's fixed WMA warm-up consumes the first 12 bars (3 seed + 9 unrolled)
/// before the Hilbert recurrence's first iteration (`today`).
const CORE_START: usize = 12;
/// Output lookback for `HT_DCPERIOD` / `HT_PHASOR` / `MAMA`.
const LB_PERIOD: usize = 32;
/// Output lookback for `HT_DCPHASE` / `HT_SINE` / `HT_TRENDLINE` / `HT_TRENDMODE`
/// (an extra 31 bars let the dominant-cycle phase / trend state stabilise).
const LB_PHASE: usize = 63;
/// Size of the rolling smoothed-price buffer used by the DC-phase computation.
const SMOOTH_PRICE_SIZE: usize = 50;

/// One Hilbert-transform channel: separate 3-slot circular buffers for odd and
/// even bars plus the carried previous output / input (TA-Lib's
/// `INIT_HILBERT_VARIABLES` + `DO_HILBERT_TRANSFORM` macros).
#[derive(Default)]
struct HilbertVar {
    odd: [f64; 3],
    even: [f64; 3],
    prev_odd: f64,
    prev_even: f64,
    prev_in_odd: f64,
    prev_in_even: f64,
}

impl HilbertVar {
    /// Transform `input` for the current bar, updating this channel's state in
    /// place and returning its output (scaled by `adj_period`).
    #[inline]
    fn transform(&mut self, input: f64, idx: usize, even: bool, adj_period: f64) -> f64 {
        let temp = A * input;
        let (buf, prev, prev_in) = if even {
            (&mut self.even, &mut self.prev_even, &mut self.prev_in_even)
        } else {
            (&mut self.odd, &mut self.prev_odd, &mut self.prev_in_odd)
        };
        let mut out = -buf[idx];
        buf[idx] = temp;
        out += temp;
        out -= *prev;
        *prev = B * *prev_in;
        out += *prev;
        *prev_in = input;
        out * adj_period
    }

    /// Append this channel's 10 state words (`odd[3] even[3] prev_odd prev_even
    /// prev_in_odd prev_in_even`) to `v`, for state-carry serialisation.
    #[inline]
    fn push_state(&self, v: &mut Vec<f64>) {
        v.extend_from_slice(&self.odd);
        v.extend_from_slice(&self.even);
        v.push(self.prev_odd);
        v.push(self.prev_even);
        v.push(self.prev_in_odd);
        v.push(self.prev_in_even);
    }

    /// Reconstruct a channel from 10 consecutive state words at `s[*off..]`,
    /// advancing `off`. Inverse of [`HilbertVar::push_state`].
    #[inline]
    fn from_state(s: &[f64], off: &mut usize) -> Self {
        let o = *off;
        let v = HilbertVar {
            odd: [s[o], s[o + 1], s[o + 2]],
            even: [s[o + 3], s[o + 4], s[o + 5]],
            prev_odd: s[o + 6],
            prev_even: s[o + 7],
            prev_in_odd: s[o + 8],
            prev_in_even: s[o + 9],
        };
        *off += 10;
        v
    }
}

/// The 4-bar weighted moving average that smooths price before the Hilbert
/// transform (TA-Lib's `DO_PRICE_WMA`; weights 4/3/2/1, normalised by 0.1).
///
/// `trailing` is the price word subtracted on the *next* push (TA-Lib loads it one
/// push ahead). It is loaded as `price[today - 3]` at bar `today` — the only price
/// lookback the WMA needs — so on a resume the index is reconstructed from `today`
/// (never carried as an absolute offset, which a head-dropping slice would shift).
struct PriceWma {
    sub: f64,
    sum: f64,
    trailing: f64,
}

impl PriceWma {
    /// Push bar `today` (`new_price == price[today]`), returning its 4/3/2/1-weighted
    /// smoothed value. Loads the next trailing word `price[today - 3]`; the seed
    /// guarantees the first main-loop push (`today == CORE_START`) reads `price[9]`.
    #[inline]
    fn push(&mut self, price: &[f64], today: usize, new_price: f64) -> f64 {
        self.sub += new_price;
        self.sub -= self.trailing;
        self.sum += new_price * 4.0;
        self.trailing = price[today - 3];
        let smoothed = self.sum * 0.1;
        self.sum -= self.sub;
        smoothed
    }
}

/// Per-bar state shared by all Hilbert functions. Meaningful only for
/// `i >= CORE_START`; earlier entries are warm-up and never read.
#[derive(Clone, Copy, Default)]
struct HtBar {
    /// Smoothed dominant-cycle period (`HT_DCPERIOD` output).
    smooth_period: f64,
    /// The 4-bar WMA-smoothed price for this bar.
    smoothed: f64,
    /// In-phase component `I1` (detrender delayed 3 bars).
    i1: f64,
    /// Quadrature component `Q1`.
    q1: f64,
}

/// The full mutable state of the shared Hilbert core *after* processing some bar
/// `t` (`>= CORE_START - 1`), i.e. everything the `while today < n` loop carries
/// between iterations: the WMA price-smoother, the four Hilbert channels + their
/// circular-buffer index, and the homodyne-discriminator scalars. Serialising it
/// lets a later append/slice **resume** the recurrence over only the new bars,
/// bit-identically to a full recompute (the loop body is identical — see
/// [`HtCoreState::step`]).
///
/// Layout for [`HtCoreState::serialize`] / [`HtCoreState::deserialize`]
/// (`CORE_STATE_LEN` words): `sub sum trailing | hilbert_idx | detrender(10)
/// q1v(10) ji(10) jq(10) | period smooth_period prev_i2 prev_q2 re im |
/// i1_odd_prev3 i1_even_prev3 i1_odd_prev2 i1_even_prev2`. The WMA's price index is
/// reconstructed from `today` (`price[today-3]`), never stored, so it stays correct
/// after a head-dropping slice rebases the price array.
struct HtCoreState {
    wma: PriceWma,
    hilbert_idx: usize,
    detrender: HilbertVar,
    q1v: HilbertVar,
    ji: HilbertVar,
    jq: HilbertVar,
    period: f64,
    smooth_period: f64,
    prev_i2: f64,
    prev_q2: f64,
    re: f64,
    im: f64,
    i1_odd_prev3: f64,
    i1_even_prev3: f64,
    i1_odd_prev2: f64,
    i1_even_prev2: f64,
}

/// Number of `f64` words in a serialised [`HtCoreState`] (3 WMA + 1 idx + 4·10
/// channels + 6 discriminator + 4 I1-delay).
const CORE_STATE_LEN: usize = 3 + 1 + 40 + 6 + 4;

impl HtCoreState {
    /// Seed the core from the first `CORE_START` bars (the WMA unrolled warm-up),
    /// leaving the state positioned to [`step`](Self::step) bar `CORE_START`.
    /// Requires `price.len() > CORE_START`.
    fn seed(price: &[f64]) -> Self {
        // WMA price-smoother seed (mirrors TA_WMA's unrolled initialisation).
        let mut sub = price[0];
        let mut sum = price[0];
        sub += price[1];
        sum += price[1] * 2.0;
        sub += price[2];
        sum += price[2] * 3.0;
        let mut wma = PriceWma { sub, sum, trailing: 0.0 };
        // Nine discarded warm-up pushes (bars 3..=11); each loads `price[bar-3]`.
        for bar in 3..CORE_START {
            let tv = price[bar];
            let _ = wma.push(price, bar, tv);
        }
        HtCoreState {
            wma,
            hilbert_idx: 0,
            detrender: HilbertVar::default(),
            q1v: HilbertVar::default(),
            ji: HilbertVar::default(),
            jq: HilbertVar::default(),
            period: 0.0,
            smooth_period: 0.0,
            prev_i2: 0.0,
            prev_q2: 0.0,
            re: 0.0,
            im: 0.0,
            i1_odd_prev3: 0.0,
            i1_even_prev3: 0.0,
            i1_odd_prev2: 0.0,
            i1_even_prev2: 0.0,
        }
    }

    /// Advance the core by one bar `today` (`>= CORE_START`), returning its
    /// [`HtBar`]. This is the verbatim body of [`ht_core`]'s main loop — the single
    /// source of truth shared by the full compute and every resume, guaranteeing
    /// bit-identical continuation.
    #[inline]
    fn step(&mut self, price: &[f64], today: usize) -> HtBar {
        // The homodyne smoothings below are `a·x + (1-a)·y` with `a+b=1`; fuse each as
        // `(x-y)·a + y` (one rounding, one fewer multiply). They are contractive
        // (b ∈ {0.8, 0.67}), so the ~1e-16 reassociation decays — within parity tolerance.
        let adj = self.period.mul_add(0.075, 0.54);
        let today_value = price[today];
        let smoothed = self.wma.push(price, today, today_value);
        let even = today % 2 == 0;

        let (i1, q1, q2, i2);
        if even {
            let det = self.detrender.transform(smoothed, self.hilbert_idx, true, adj);
            let q1c = self.q1v.transform(det, self.hilbert_idx, true, adj);
            let jiv = self.ji.transform(self.i1_even_prev3, self.hilbert_idx, true, adj);
            let jqv = self.jq.transform(q1c, self.hilbert_idx, true, adj);
            self.hilbert_idx += 1;
            if self.hilbert_idx == 3 {
                self.hilbert_idx = 0;
            }
            q2 = (q1c + jiv - self.prev_q2).mul_add(0.2, self.prev_q2);
            i2 = (self.i1_even_prev3 - jqv - self.prev_i2).mul_add(0.2, self.prev_i2);
            i1 = self.i1_even_prev3;
            q1 = q1c;
            self.i1_odd_prev3 = self.i1_odd_prev2;
            self.i1_odd_prev2 = det;
        } else {
            let det = self.detrender.transform(smoothed, self.hilbert_idx, false, adj);
            let q1c = self.q1v.transform(det, self.hilbert_idx, false, adj);
            let jiv = self.ji.transform(self.i1_odd_prev3, self.hilbert_idx, false, adj);
            let jqv = self.jq.transform(q1c, self.hilbert_idx, false, adj);
            q2 = (q1c + jiv - self.prev_q2).mul_add(0.2, self.prev_q2);
            i2 = (self.i1_odd_prev3 - jqv - self.prev_i2).mul_add(0.2, self.prev_i2);
            i1 = self.i1_odd_prev3;
            q1 = q1c;
            self.i1_even_prev3 = self.i1_even_prev2;
            self.i1_even_prev2 = det;
        }

        // Homodyne discriminator -> dominant cycle period.
        self.re = (i2 * self.prev_i2 + q2 * self.prev_q2 - self.re).mul_add(0.2, self.re);
        self.im = (i2 * self.prev_q2 - q2 * self.prev_i2 - self.im).mul_add(0.2, self.im);
        self.prev_q2 = q2;
        self.prev_i2 = i2;
        let prev_period = self.period;
        if self.im != 0.0 && self.re != 0.0 {
            self.period = 360.0 / ((self.im / self.re).atan() * RAD2DEG);
        }
        let hi = 1.5 * prev_period;
        if self.period > hi {
            self.period = hi;
        }
        let lo = 0.67 * prev_period;
        if self.period < lo {
            self.period = lo;
        }
        if self.period < 6.0 {
            self.period = 6.0;
        } else if self.period > 50.0 {
            self.period = 50.0;
        }
        self.period = (self.period - prev_period).mul_add(0.2, prev_period);
        self.smooth_period = (self.period - self.smooth_period).mul_add(0.33, self.smooth_period);

        HtBar { smooth_period: self.smooth_period, smoothed, i1, q1 }
    }

    /// Serialise to `CORE_STATE_LEN` `f64` words (see the type doc for the layout).
    fn serialize(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(CORE_STATE_LEN);
        v.push(self.wma.sub);
        v.push(self.wma.sum);
        v.push(self.wma.trailing);
        v.push(self.hilbert_idx as f64);
        self.detrender.push_state(&mut v);
        self.q1v.push_state(&mut v);
        self.ji.push_state(&mut v);
        self.jq.push_state(&mut v);
        v.push(self.period);
        v.push(self.smooth_period);
        v.push(self.prev_i2);
        v.push(self.prev_q2);
        v.push(self.re);
        v.push(self.im);
        v.push(self.i1_odd_prev3);
        v.push(self.i1_even_prev3);
        v.push(self.i1_odd_prev2);
        v.push(self.i1_even_prev2);
        debug_assert_eq!(v.len(), CORE_STATE_LEN);
        v
    }

    /// Reconstruct from the first `CORE_STATE_LEN` words of `s` (inverse of
    /// [`serialize`](Self::serialize)); `None` if `s` is too short.
    fn deserialize(s: &[f64]) -> Option<Self> {
        if s.len() < CORE_STATE_LEN {
            return None;
        }
        let wma = PriceWma { sub: s[0], sum: s[1], trailing: s[2] };
        let hilbert_idx = s[3] as usize;
        let mut off = 4usize;
        let detrender = HilbertVar::from_state(s, &mut off);
        let q1v = HilbertVar::from_state(s, &mut off);
        let ji = HilbertVar::from_state(s, &mut off);
        let jq = HilbertVar::from_state(s, &mut off);
        Some(HtCoreState {
            wma,
            hilbert_idx,
            detrender,
            q1v,
            ji,
            jq,
            period: s[off],
            smooth_period: s[off + 1],
            prev_i2: s[off + 2],
            prev_q2: s[off + 3],
            re: s[off + 4],
            im: s[off + 5],
            i1_odd_prev3: s[off + 6],
            i1_even_prev3: s[off + 7],
            i1_odd_prev2: s[off + 8],
            i1_even_prev2: s[off + 9],
        })
    }
}

/// Run TA-Lib's shared Hilbert core over the whole series. `out[i]` is meaningful
/// for `i >= CORE_START` only.
fn ht_core(price: &[f64]) -> Vec<HtBar> {
    let n = price.len();
    let mut bars = vec![HtBar::default(); n];
    if n <= CORE_START {
        return bars;
    }
    let mut core = HtCoreState::seed(price);
    for today in CORE_START..n {
        bars[today] = core.step(price, today);
    }
    bars
}

/// Rolling Dominant-Cycle-Phase accumulator (shared by `HT_DCPHASE`, `HT_SINE`
/// and `HT_TRENDMODE`). `push` consumes one bar and returns its DC phase in
/// degrees; the phase carries across bars (the `imagPart == 0` branch nudges the
/// previous value by ±90°).
struct DcPhase {
    smooth_price: [f64; SMOOTH_PRICE_SIZE],
    idx: usize,
    phase: f64,
}

impl DcPhase {
    fn new() -> Self {
        Self { smooth_price: [0.0; SMOOTH_PRICE_SIZE], idx: 0, phase: 0.0 }
    }

    fn push(&mut self, smoothed: f64, smooth_period: f64) -> f64 {
        self.smooth_price[self.idx] = smoothed;
        let dc_period_int = (smooth_period + 0.5) as usize;
        let mut real_part = 0.0f64;
        let mut imag_part = 0.0f64;
        let mut k = self.idx;
        for i in 0..dc_period_int {
            let t = (i as f64) * TWO_PI / (dc_period_int as f64);
            let v = self.smooth_price[k];
            real_part += t.sin() * v;
            imag_part += t.cos() * v;
            k = if k == 0 { SMOOTH_PRICE_SIZE - 1 } else { k - 1 };
        }
        let abs_imag = imag_part.abs();
        if abs_imag > 0.0 {
            self.phase = (real_part / imag_part).atan() * RAD2DEG;
        } else if abs_imag <= 0.01 && real_part != 0.0 {
            // imagPart == 0 (TA-Lib's defensive carry): nudge the previous phase ±90°
            // by the sign of realPart. (real == 0 leaves the phase unchanged, as in
            // TA-Lib; folded into the guard so there is no empty fall-through branch.)
            self.phase += if real_part < 0.0 { -90.0 } else { 90.0 };
        }
        self.phase += 90.0;
        // Compensate for the one-bar lag of the weighted moving average.
        self.phase += 360.0 / smooth_period;
        if imag_part < 0.0 {
            self.phase += 180.0;
        }
        if self.phase > 315.0 {
            self.phase -= 360.0;
        }
        self.idx = if self.idx + 1 == SMOOTH_PRICE_SIZE { 0 } else { self.idx + 1 };
        self.phase
    }

    /// Append this accumulator's state (`phase`, `idx`, then the 50-slot
    /// `smooth_price` ring) to `v`, for state-carry serialisation.
    fn push_state(&self, v: &mut Vec<f64>) {
        v.push(self.phase);
        v.push(self.idx as f64);
        v.extend_from_slice(&self.smooth_price);
    }

    /// Reconstruct from `DCPHASE_STATE_LEN` words at `s[off..]`, advancing `off`.
    /// Inverse of [`DcPhase::push_state`].
    fn from_state(s: &[f64], off: &mut usize) -> Self {
        let o = *off;
        let mut smooth_price = [0.0f64; SMOOTH_PRICE_SIZE];
        smooth_price.copy_from_slice(&s[o + 2..o + 2 + SMOOTH_PRICE_SIZE]);
        *off += DCPHASE_STATE_LEN;
        DcPhase { smooth_price, idx: s[o + 1] as usize, phase: s[o] }
    }
}

/// Words in a serialised [`DcPhase`] (`phase` + `idx` + the 50-slot ring).
const DCPHASE_STATE_LEN: usize = 2 + SMOOTH_PRICE_SIZE;

/// DC phase per bar (degrees); index `< CORE_START` holds the pre-loop state `0.0`
/// so `phase[i-1]` is valid at `i == CORE_START`.
fn dcphase_all(core: &[HtBar]) -> Vec<f64> {
    let n = core.len();
    let mut out = vec![0.0f64; n];
    let mut dc = DcPhase::new();
    for i in CORE_START..n {
        out[i] = dc.push(core[i].smoothed, core[i].smooth_period);
    }
    out
}

/// Trendline per bar: a `DCPeriod`-window average of *raw* price smoothed by the
/// 4/3/2/1-weighted `iTrend` recurrence. Index `< CORE_START` holds `0.0`.
fn trendline_all(core: &[HtBar], price: &[f64]) -> Vec<f64> {
    let n = core.len();
    let mut out = vec![0.0f64; n];
    let mut it1 = 0.0f64;
    let mut it2 = 0.0f64;
    let mut it3 = 0.0f64;
    for today in CORE_START..n {
        let avg = trendline_avg(price, today, core[today].smooth_period);
        out[today] = (4.0 * avg + 3.0 * it1 + 2.0 * it2 + it3) / 10.0;
        it3 = it2;
        it2 = it1;
        it1 = avg;
    }
    out
}

/// `HT_DCPERIOD`: the smoothed dominant-cycle period.
pub fn ht_dcperiod(price: &[f64]) -> Vec<f64> {
    let core = ht_core(price);
    let mut out = vec![f64::NAN; price.len()];
    for i in LB_PERIOD..price.len() {
        out[i] = core[i].smooth_period;
    }
    out
}

/// `HT_PHASOR`: the in-phase and quadrature components `(inPhase, quadrature)`.
pub fn ht_phasor(price: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let core = ht_core(price);
    let n = price.len();
    let mut inphase = vec![f64::NAN; n];
    let mut quad = vec![f64::NAN; n];
    for i in LB_PERIOD..n {
        inphase[i] = core[i].i1;
        quad[i] = core[i].q1;
    }
    (inphase, quad)
}

/// `HT_DCPHASE`: the dominant-cycle phase in degrees.
pub fn ht_dcphase(price: &[f64]) -> Vec<f64> {
    let core = ht_core(price);
    let phase = dcphase_all(&core);
    let mut out = vec![f64::NAN; price.len()];
    for i in LB_PHASE..price.len() {
        out[i] = phase[i];
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
    let core = ht_core(price);
    let trend = trendline_all(&core, price);
    let mut out = vec![f64::NAN; price.len()];
    for i in LB_PHASE..price.len() {
        out[i] = trend[i];
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
    let core = ht_core(price);
    let n = price.len();
    let mut mama_out = vec![f64::NAN; n];
    let mut fama_out = vec![f64::NAN; n];
    let mut mama = 0.0f64;
    let mut fama = 0.0f64;
    let mut prev_phase = 0.0f64;
    for i in CORE_START..n {
        let b = core[i];
        mama_step(&b, price[i], fast_limit, slow_limit, &mut mama, &mut fama, &mut prev_phase);
        if i >= LB_PERIOD {
            mama_out[i] = mama;
            fama_out[i] = fama;
        }
    }
    (mama_out, fama_out)
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
pub fn ht_dcperiod_resume(price: &[f64], from: usize, state: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
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
pub fn ht_dcphase_resume(price: &[f64], from: usize, state: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
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
pub fn ht_trendline_resume(price: &[f64], from: usize, state: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    if from < SMOOTH_PRICE_SIZE {
        return None;
    }
    let mut core = ht_core_resume_setup(state, from)?;
    if state.len() < CORE_STATE_LEN + 3 {
        return None;
    }
    let (mut it1, mut it2, mut it3) =
        (state[CORE_STATE_LEN], state[CORE_STATE_LEN + 1], state[CORE_STATE_LEN + 2]);
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
            phase, prev_phase, b.smooth_period, b.smoothed, trend_line, &mut days_in_trend,
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

/// Resume `HT_TRENDMODE` over `[from, n)`. Steps the core, reconstructs the DC phase
/// + trendline, runs the shared decision with the carried `days_in_trend`, and
/// emits `0.0`/`1.0`. Guarded by `from >= SMOOTH_PRICE_SIZE` (trendline window).
pub fn ht_trendmode_resume(price: &[f64], from: usize, state: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
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
            phase, prev_phase, b.smooth_period, b.smoothed, trend_line, &mut days_in_trend,
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
        mama_step(&b, price[today], fast_limit, slow_limit, &mut mama, &mut fama, &mut prev_phase);
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
    let phase = if b.i1 != 0.0 { (b.q1 / b.i1).atan() * RAD2DEG } else { 0.0 };
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
    let (mut mama, mut fama, mut prev_phase) =
        (state[CORE_STATE_LEN], state[CORE_STATE_LEN + 1], state[CORE_STATE_LEN + 2]);
    let n = price.len();
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for today in from..n {
        let b = core.step(price, today);
        mama_step(&b, price[today], fast_limit, slow_limit, &mut mama, &mut fama, &mut prev_phase);
        out.push(if want_fama { fama } else { mama });
    }
    let mut v = core.serialize();
    v.push(mama);
    v.push(fama);
    v.push(prev_phase);
    Some((out, v))
}

#[cfg(test)]
mod tests {
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
        assert_eq!((TWO_PI / 2.0).cos(), -1.0, "cos(π) must be exactly -1 for the cancellation");

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
            assert_bits(&v, &ht_trendline(&price)[from..], "trendline");
            let st = ht_trendmode_state(head).unwrap();
            let (v, _) = ht_trendmode_resume(&price, from, &st).unwrap();
            assert_bits(&v, &ht_trendmode(&price)[from..], "trendmode");

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
}
