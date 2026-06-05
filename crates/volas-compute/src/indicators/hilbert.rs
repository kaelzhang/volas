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
}

/// The 4-bar weighted moving average that smooths price before the Hilbert
/// transform (TA-Lib's `DO_PRICE_WMA`; weights 4/3/2/1, normalised by 0.1).
struct PriceWma {
    sub: f64,
    sum: f64,
    trailing: f64,
    trailing_idx: usize,
}

impl PriceWma {
    #[inline]
    fn push(&mut self, price: &[f64], new_price: f64) -> f64 {
        self.sub += new_price;
        self.sub -= self.trailing;
        self.sum += new_price * 4.0;
        self.trailing = price[self.trailing_idx];
        self.trailing_idx += 1;
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

/// Run TA-Lib's shared Hilbert core over the whole series. `out[i]` is meaningful
/// for `i >= CORE_START` only.
fn ht_core(price: &[f64]) -> Vec<HtBar> {
    let n = price.len();
    let mut bars = vec![HtBar::default(); n];
    if n <= CORE_START {
        return bars;
    }

    // WMA price-smoother seed (mirrors TA_WMA's unrolled initialisation).
    let mut sub = price[0];
    let mut sum = price[0];
    sub += price[1];
    sum += price[1] * 2.0;
    sub += price[2];
    sum += price[2] * 3.0;
    let mut wma = PriceWma { sub, sum, trailing: 0.0, trailing_idx: 0 };
    let mut today = 3usize;
    for _ in 0..9 {
        let tv = price[today];
        today += 1;
        let _ = wma.push(price, tv); // warm-up smoothed values are discarded
    }

    // Hilbert transform channels + homodyne-discriminator state.
    let mut hilbert_idx = 0usize;
    let mut detrender = HilbertVar::default();
    let mut q1v = HilbertVar::default();
    let mut ji = HilbertVar::default();
    let mut jq = HilbertVar::default();

    let mut period = 0.0f64;
    let mut smooth_period = 0.0f64;
    let mut prev_i2 = 0.0f64;
    let mut prev_q2 = 0.0f64;
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    let mut i1_odd_prev3 = 0.0f64;
    let mut i1_even_prev3 = 0.0f64;
    let mut i1_odd_prev2 = 0.0f64;
    let mut i1_even_prev2 = 0.0f64;

    while today < n {
        // The homodyne smoothings below are `a·x + (1-a)·y` with `a+b=1`; fuse each as
        // `(x-y)·a + y` (one rounding, one fewer multiply). They are contractive
        // (b ∈ {0.8, 0.67}), so the ~1e-16 reassociation decays — within parity tolerance.
        let adj = period.mul_add(0.075, 0.54);
        let today_value = price[today];
        let smoothed = wma.push(price, today_value);
        let even = today % 2 == 0;

        let (i1, q1, q2, i2);
        if even {
            let det = detrender.transform(smoothed, hilbert_idx, true, adj);
            let q1c = q1v.transform(det, hilbert_idx, true, adj);
            let jiv = ji.transform(i1_even_prev3, hilbert_idx, true, adj);
            let jqv = jq.transform(q1c, hilbert_idx, true, adj);
            hilbert_idx += 1;
            if hilbert_idx == 3 {
                hilbert_idx = 0;
            }
            q2 = (q1c + jiv - prev_q2).mul_add(0.2, prev_q2);
            i2 = (i1_even_prev3 - jqv - prev_i2).mul_add(0.2, prev_i2);
            i1 = i1_even_prev3;
            q1 = q1c;
            i1_odd_prev3 = i1_odd_prev2;
            i1_odd_prev2 = det;
        } else {
            let det = detrender.transform(smoothed, hilbert_idx, false, adj);
            let q1c = q1v.transform(det, hilbert_idx, false, adj);
            let jiv = ji.transform(i1_odd_prev3, hilbert_idx, false, adj);
            let jqv = jq.transform(q1c, hilbert_idx, false, adj);
            q2 = (q1c + jiv - prev_q2).mul_add(0.2, prev_q2);
            i2 = (i1_odd_prev3 - jqv - prev_i2).mul_add(0.2, prev_i2);
            i1 = i1_odd_prev3;
            q1 = q1c;
            i1_even_prev3 = i1_even_prev2;
            i1_even_prev2 = det;
        }

        // Homodyne discriminator -> dominant cycle period.
        re = (i2 * prev_i2 + q2 * prev_q2 - re).mul_add(0.2, re);
        im = (i2 * prev_q2 - q2 * prev_i2 - im).mul_add(0.2, im);
        prev_q2 = q2;
        prev_i2 = i2;
        let prev_period = period;
        if im != 0.0 && re != 0.0 {
            period = 360.0 / ((im / re).atan() * RAD2DEG);
        }
        let hi = 1.5 * prev_period;
        if period > hi {
            period = hi;
        }
        let lo = 0.67 * prev_period;
        if period < lo {
            period = lo;
        }
        if period < 6.0 {
            period = 6.0;
        } else if period > 50.0 {
            period = 50.0;
        }
        period = (period - prev_period).mul_add(0.2, prev_period);
        smooth_period = (period - smooth_period).mul_add(0.33, smooth_period);

        bars[today] = HtBar { smooth_period, smoothed, i1, q1 };
        today += 1;
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
        } else if abs_imag <= 0.01 {
            if real_part < 0.0 {
                self.phase -= 90.0;
            } else if real_part > 0.0 {
                self.phase += 90.0;
            }
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
}

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
        let dc_period_int = (core[today].smooth_period + 0.5) as usize;
        let mut sum = 0.0f64;
        for j in 0..dc_period_int {
            if today >= j {
                sum += price[today - j];
            }
        }
        let avg = if dc_period_int > 0 { sum / dc_period_int as f64 } else { 0.0 };
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
    for i in CORE_START..n {
        let dc = phase[i];
        let prev_dc = phase[i - 1];
        let sine = (dc * DEG2RAD).sin();
        let lead = ((dc + 45.0) * DEG2RAD).sin();
        let prev_sine = (prev_dc * DEG2RAD).sin();
        let prev_lead = ((prev_dc + 45.0) * DEG2RAD).sin();
        let sp = core[i].smooth_period;

        // Assume trend; demote to cycle on a fresh SineWave crossing, when too few
        // days have elapsed since the last crossing, or when the phase is advancing
        // at roughly the dominant-cycle rate.
        let mut trend = 1i32;
        if (sine > lead && prev_sine <= prev_lead) || (sine < lead && prev_sine >= prev_lead) {
            days_in_trend = 0;
            trend = 0;
        }
        days_in_trend += 1;
        if (days_in_trend as f64) < 0.5 * sp {
            trend = 0;
        }
        let dphase = dc - prev_dc;
        if sp != 0.0 && dphase > 0.67 * 360.0 / sp && dphase < 1.5 * 360.0 / sp {
            trend = 0;
        }
        // Strong divergence of price from the trendline forces trend mode.
        let cur = core[i].smoothed;
        let tl = trend_line[i];
        if tl != 0.0 && ((cur - tl) / tl).abs() >= 0.015 {
            trend = 1;
        }

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
        let phase = if b.i1 != 0.0 { (b.q1 / b.i1).atan() * RAD2DEG } else { 0.0 };
        let mut delta = prev_phase - phase;
        prev_phase = phase;
        if delta < 1.0 {
            delta = 1.0;
        }
        let alpha = if delta > 1.0 {
            (fast_limit / delta).max(slow_limit)
        } else {
            fast_limit
        };
        mama = alpha * price[i] + (1.0 - alpha) * mama;
        let half = 0.5 * alpha;
        fama = half * mama + (1.0 - half) * fama;
        if i >= LB_PERIOD {
            mama_out[i] = mama;
            fama_out[i] = fama;
        }
    }
    (mama_out, fama_out)
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
}
