//! The shared Hilbert-transform engine: the 4-bar WMA price smoother, the
//! quadrature Hilbert channels, the homodyne-discriminator core state, the
//! DC-phase accumulator, and the per-bar core (`ht_core`) every HT indicator
//! derives from. Internal to the `hilbert` module (items are `pub(super)`).

use std::f64::consts::PI;
use std::sync::OnceLock;

/// Hilbert weighting coefficients (`ta_utility.h`'s `DO_HILBERT_TRANSFORM`).
pub(super) const A: f64 = 0.0962;
const B: f64 = 0.5769;
/// `rad2Deg = 45/atan(1) = 180/π`; `deg2Rad = 1/rad2Deg`; `constDeg2RadBy360 = atan(1)·8 = 2π`.
pub(super) const RAD2DEG: f64 = 180.0 / PI;
pub(super) const DEG2RAD: f64 = PI / 180.0;
pub(super) const TWO_PI: f64 = 2.0 * PI;

/// TA-Lib's fixed WMA warm-up consumes the first 12 bars (3 seed + 9 unrolled)
/// before the Hilbert recurrence's first iteration (`today`).
pub(super) const CORE_START: usize = 12;
/// Output lookback for `HT_DCPERIOD` / `HT_PHASOR` / `MAMA`.
pub(super) const LB_PERIOD: usize = 32;
/// Output lookback for `HT_DCPHASE` / `HT_SINE` / `HT_TRENDLINE` / `HT_TRENDMODE`
/// (an extra 31 bars let the dominant-cycle phase / trend state stabilise).
pub(super) const LB_PHASE: usize = 63;
/// Size of the rolling smoothed-price buffer used by the DC-phase computation.
pub(super) const SMOOTH_PRICE_SIZE: usize = 50;

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
    fn transform_even(&mut self, input: f64, idx: usize, adj_period: f64) -> f64 {
        debug_assert!(idx < 3);
        let temp = A * input;
        let buf = &mut self.even;
        // `hilbert_idx` is a 0..3 ring cursor maintained by `HtCoreState::step`;
        // avoid four bounds checks per bar across the Hilbert channels.
        let mut out = -unsafe { *buf.get_unchecked(idx) };
        unsafe {
            *buf.get_unchecked_mut(idx) = temp;
        }
        out += temp;
        out -= self.prev_even;
        self.prev_even = B * self.prev_in_even;
        out += self.prev_even;
        self.prev_in_even = input;
        out * adj_period
    }

    /// Odd-bar variant of [`transform_even`](Self::transform_even). Keeping the
    /// buffers explicit mirrors TA-Lib's macro split and avoids a hot bool branch.
    #[inline]
    fn transform_odd(&mut self, input: f64, idx: usize, adj_period: f64) -> f64 {
        debug_assert!(idx < 3);
        let temp = A * input;
        let buf = &mut self.odd;
        // Same invariant as the even channel: `idx` is the 3-slot Hilbert ring.
        let mut out = -unsafe { *buf.get_unchecked(idx) };
        unsafe {
            *buf.get_unchecked_mut(idx) = temp;
        }
        out += temp;
        out -= self.prev_odd;
        self.prev_odd = B * self.prev_in_odd;
        out += self.prev_odd;
        self.prev_in_odd = input;
        out * adj_period
    }

    /// Append this channel's 10 state words (`odd[3] even[3] prev_odd prev_even
    /// prev_in_odd prev_in_even`) to `v`, for state-carry serialisation.
    #[inline]
    pub(super) fn push_state(&self, v: &mut Vec<f64>) {
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
    pub(super) fn from_state(s: &[f64], off: &mut usize) -> Self {
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
    pub(super) fn push(&mut self, price: &[f64], today: usize, new_price: f64) -> f64 {
        self.sub += new_price;
        self.sub -= self.trailing;
        self.sum += new_price * 4.0;
        // The WMA warm-up and Hilbert main loop only call this with `today >= 3`
        // and `today < price.len()`, so the trailing word is always in-bounds.
        self.trailing = unsafe { *price.get_unchecked(today - 3) };
        let smoothed = self.sum * 0.1;
        self.sum -= self.sub;
        smoothed
    }
}

/// Per-bar state shared by all Hilbert functions. Meaningful only for
/// `i >= CORE_START`; earlier entries are warm-up and never read.
#[derive(Clone, Copy, Default)]
pub(super) struct HtBar {
    /// Smoothed dominant-cycle period (`HT_DCPERIOD` output).
    pub(super) smooth_period: f64,
    /// The 4-bar WMA-smoothed price for this bar.
    pub(super) smoothed: f64,
    /// In-phase component `I1` (detrender delayed 3 bars).
    pub(super) i1: f64,
    /// Quadrature component `Q1`.
    pub(super) q1: f64,
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
pub(super) struct HtCoreState {
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
pub(super) const CORE_STATE_LEN: usize = 3 + 1 + 40 + 6 + 4;

impl HtCoreState {
    /// Seed the core from the first `CORE_START` bars (the WMA unrolled warm-up),
    /// leaving the state positioned to [`step`](Self::step) bar `CORE_START`.
    /// Requires `price.len() > CORE_START`.
    pub(super) fn seed(price: &[f64]) -> Self {
        // WMA price-smoother seed (mirrors TA_WMA's unrolled initialisation).
        let mut sub = price[0];
        let mut sum = price[0];
        sub += price[1];
        sum += price[1] * 2.0;
        sub += price[2];
        sum += price[2] * 3.0;
        let mut wma = PriceWma {
            sub,
            sum,
            trailing: 0.0,
        };
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
    #[inline(always)]
    pub(super) fn step(&mut self, price: &[f64], today: usize) -> HtBar {
        // Keep TA-Lib's explicit `a*x + b*y` smoothing form. On the release target
        // this beats the algebraic `mul_add` rewrite and preserves the C recurrence order.
        let adj = self.period.mul_add(0.075, 0.54);
        // All callers keep `today` inside the input slice; this is the innermost
        // loop for every Hilbert-derived indicator.
        let today_value = unsafe { *price.get_unchecked(today) };
        let smoothed = self.wma.push(price, today, today_value);
        let even = today % 2 == 0;

        let (i1, q1, q2, i2);
        if even {
            let det = self
                .detrender
                .transform_even(smoothed, self.hilbert_idx, adj);
            let q1c = self.q1v.transform_even(det, self.hilbert_idx, adj);
            let jiv = self
                .ji
                .transform_even(self.i1_even_prev3, self.hilbert_idx, adj);
            let jqv = self.jq.transform_even(q1c, self.hilbert_idx, adj);
            self.hilbert_idx += 1;
            if self.hilbert_idx == 3 {
                self.hilbert_idx = 0;
            }
            q2 = 0.2 * (q1c + jiv) + 0.8 * self.prev_q2;
            i2 = 0.2 * (self.i1_even_prev3 - jqv) + 0.8 * self.prev_i2;
            i1 = self.i1_even_prev3;
            q1 = q1c;
            self.i1_odd_prev3 = self.i1_odd_prev2;
            self.i1_odd_prev2 = det;
        } else {
            let det = self
                .detrender
                .transform_odd(smoothed, self.hilbert_idx, adj);
            let q1c = self.q1v.transform_odd(det, self.hilbert_idx, adj);
            let jiv = self
                .ji
                .transform_odd(self.i1_odd_prev3, self.hilbert_idx, adj);
            let jqv = self.jq.transform_odd(q1c, self.hilbert_idx, adj);
            q2 = 0.2 * (q1c + jiv) + 0.8 * self.prev_q2;
            i2 = 0.2 * (self.i1_odd_prev3 - jqv) + 0.8 * self.prev_i2;
            i1 = self.i1_odd_prev3;
            q1 = q1c;
            self.i1_even_prev3 = self.i1_even_prev2;
            self.i1_even_prev2 = det;
        }

        // Homodyne discriminator -> dominant cycle period.
        self.re = 0.2 * (i2 * self.prev_i2 + q2 * self.prev_q2) + 0.8 * self.re;
        self.im = 0.2 * (i2 * self.prev_q2 - q2 * self.prev_i2) + 0.8 * self.im;
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
        self.period = 0.2 * self.period + 0.8 * prev_period;
        self.smooth_period = 0.33 * self.period + 0.67 * self.smooth_period;

        HtBar {
            smooth_period: self.smooth_period,
            smoothed,
            i1,
            q1,
        }
    }

    /// Serialise to `CORE_STATE_LEN` `f64` words (see the type doc for the layout).
    pub(super) fn serialize(&self) -> Vec<f64> {
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
    pub(super) fn deserialize(s: &[f64]) -> Option<Self> {
        if s.len() < CORE_STATE_LEN {
            return None;
        }
        let wma = PriceWma {
            sub: s[0],
            sum: s[1],
            trailing: s[2],
        };
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
pub(super) fn ht_core(price: &[f64]) -> Vec<HtBar> {
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
/// Twiddle factors `(sin(i·2π/p), cos(i·2π/p))` for every DC period `p ≤ SMOOTH_PRICE_SIZE`,
/// indexed `[p][i]`. Built once on first use so the DC-phase DFT reads its sin/cos from a table
/// instead of recomputing them per bar. Row `p = 0` (a degenerate empty window) is never summed.
type DftTwiddle = [[(f64, f64); SMOOTH_PRICE_SIZE]; SMOOTH_PRICE_SIZE + 1];
static DFT_TWIDDLE: OnceLock<Box<DftTwiddle>> = OnceLock::new();

fn build_dft_twiddle() -> Box<DftTwiddle> {
    let mut t: Box<DftTwiddle> = Box::new([[(0.0, 0.0); SMOOTH_PRICE_SIZE]; SMOOTH_PRICE_SIZE + 1]);
    for (p, row) in t.iter_mut().enumerate().skip(1) {
        let delta = TWO_PI / p as f64;
        for (i, slot) in row.iter_mut().enumerate().take(p) {
            *slot = (i as f64 * delta).sin_cos();
        }
    }
    t
}

pub(super) struct DcPhase {
    smooth_price: [f64; SMOOTH_PRICE_SIZE],
    idx: usize,
    phase: f64,
}

impl DcPhase {
    pub(super) fn new() -> Self {
        Self {
            smooth_price: [0.0; SMOOTH_PRICE_SIZE],
            idx: 0,
            phase: 0.0,
        }
    }

    pub(super) fn push(&mut self, smoothed: f64, smooth_period: f64) -> f64 {
        self.smooth_price[self.idx] = smoothed;
        let dc_period_int = (smooth_period + 0.5) as usize;
        let mut real_part = 0.0f64;
        let mut imag_part = 0.0f64;
        let mut k = self.idx;
        // The DFT bins t_i = i·(2π/dc_period_int) have the same sin/cos for every bar of the
        // same period, so read them from a twiddle table built once (lazily) for all periods
        // rather than recomputing per bar — no per-bar transcendentals, no serial rotation
        // chain. The table holds the direct sin/cos (well inside the HT convergence tolerance),
        // and the full compute and the resume share it so they stay bit-identical to each other.
        let twiddle = &DFT_TWIDDLE.get_or_init(build_dft_twiddle)[dc_period_int];
        for &(s, c) in &twiddle[..dc_period_int] {
            let v = self.smooth_price[k];
            real_part += s * v;
            imag_part += c * v;
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
        self.idx = if self.idx + 1 == SMOOTH_PRICE_SIZE {
            0
        } else {
            self.idx + 1
        };
        self.phase
    }

    /// Append this accumulator's state (`phase`, `idx`, then the 50-slot
    /// `smooth_price` ring) to `v`, for state-carry serialisation.
    pub(super) fn push_state(&self, v: &mut Vec<f64>) {
        v.push(self.phase);
        v.push(self.idx as f64);
        v.extend_from_slice(&self.smooth_price);
    }

    /// Reconstruct from `DCPHASE_STATE_LEN` words at `s[off..]`, advancing `off`.
    /// Inverse of [`DcPhase::push_state`].
    pub(super) fn from_state(s: &[f64], off: &mut usize) -> Self {
        let o = *off;
        let mut smooth_price = [0.0f64; SMOOTH_PRICE_SIZE];
        smooth_price.copy_from_slice(&s[o + 2..o + 2 + SMOOTH_PRICE_SIZE]);
        *off += DCPHASE_STATE_LEN;
        DcPhase {
            smooth_price,
            idx: s[o + 1] as usize,
            phase: s[o],
        }
    }
}

/// Words in a serialised [`DcPhase`] (`phase` + `idx` + the 50-slot ring).
pub(super) const DCPHASE_STATE_LEN: usize = 2 + SMOOTH_PRICE_SIZE;

/// DC phase per bar (degrees); index `< CORE_START` holds the pre-loop state `0.0`
/// so `phase[i-1]` is valid at `i == CORE_START`.
pub(super) fn dcphase_all(core: &[HtBar]) -> Vec<f64> {
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
pub(super) fn trendline_all(core: &[HtBar], price: &[f64]) -> Vec<f64> {
    let n = core.len();
    let mut out = vec![0.0f64; n];
    // Prefix sums of price (`prefix[k] = Σ price[0..k]`) turn each bar's
    // variable-width DC-period window mean into an O(1) difference instead of the
    // O(period) rescan of [`trendline_avg`] — the whole pass becomes O(n), not
    // O(n·period) (the dominant cost of HT_TRENDLINE). The reassociated summation
    // drifts ~1e-12 relative: far inside ht_trendline's 1e-7 TA-Lib parity and the
    // 1e-9 resume-vs-batch tolerance (the per-bar resume keeps the exact rescan).
    let mut prefix = vec![0.0f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + price[i];
    }
    let (mut it1, mut it2, mut it3) = (0.0f64, 0.0f64, 0.0f64);
    for today in CORE_START..n {
        // `smooth_period` is the homodyne period (clamped to [6, 50]) blended 0.33
        // up from 0, so `dc >= 2` from the very first bar — never zero, no guard.
        let dc = (core[today].smooth_period + 0.5) as usize;
        let lo = (today + 1).saturating_sub(dc);
        let avg = (prefix[today + 1] - prefix[lo]) / dc as f64;
        out[today] = (4.0 * avg + 3.0 * it1 + 2.0 * it2 + it3) / 10.0;
        it3 = it2;
        it2 = it1;
        it1 = avg;
    }
    out
}
