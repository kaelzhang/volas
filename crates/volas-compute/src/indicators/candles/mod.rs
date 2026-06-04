// ---------------------------------------------------------------------------
// Candlestick patterns (TA-Lib CDL*)
// ---------------------------------------------------------------------------
//
// A cohesive sub-domain of `indicators`: candlestick-pattern recognition. This module
// owns the shared candle-settings framework (TA-Lib's `TA_CANDLE*`); the patterns
// themselves live in bar-count submodules (`one_bar`, `two_bar`, …) and are re-exported.
//
// Each pattern tests a candle's geometry against adaptive thresholds derived from a
// rolling average of a chosen "range" (real body / high-low / shadows) over the prior
// `avg_period` bars, scaled by a `factor`. Output is f64: `+100` bullish, `-100`
// bearish, `0` no pattern; the warm-up is NaN (volas's uniform convention — TA-Lib
// fills its integer outputs with 0, which is indistinguishable from "no pattern").
// Parity holds on the valid region.

mod one_bar;
pub use one_bar::*;

/// The range a candle-setting measures. The full set mirrors TA-Lib's settings table;
/// a variant is only *constructed* once a pattern needing it lands, so allow the lint.
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum RangeType {
    RealBody,
    HighLow,
    Shadows,
}

/// One candle-setting: which range, the averaging period (0 = the bar itself), and the
/// scaling factor. Values are TA-Lib's defaults (ta_global.c). Visible to the pattern
/// submodules (descendants) without `pub`.
#[derive(Clone, Copy)]
struct Setting {
    range: RangeType,
    avg_period: usize,
    factor: f64,
}

use RangeType::{HighLow, RealBody, Shadows};
const BODY_LONG: Setting = Setting { range: RealBody, avg_period: 10, factor: 1.0 };
const BODY_VERY_LONG: Setting = Setting { range: RealBody, avg_period: 10, factor: 3.0 };
const BODY_SHORT: Setting = Setting { range: RealBody, avg_period: 10, factor: 1.0 };
const BODY_DOJI: Setting = Setting { range: HighLow, avg_period: 10, factor: 0.1 };
const SHADOW_LONG: Setting = Setting { range: RealBody, avg_period: 0, factor: 1.0 };
const SHADOW_VERY_LONG: Setting = Setting { range: RealBody, avg_period: 0, factor: 2.0 };
const SHADOW_SHORT: Setting = Setting { range: Shadows, avg_period: 10, factor: 1.0 };
const SHADOW_VERY_SHORT: Setting = Setting { range: HighLow, avg_period: 10, factor: 0.1 };
const NEAR: Setting = Setting { range: HighLow, avg_period: 5, factor: 0.2 };
const FAR: Setting = Setting { range: HighLow, avg_period: 5, factor: 0.6 };
const EQUAL: Setting = Setting { range: HighLow, avg_period: 5, factor: 0.05 };

// Settings not yet consumed by a landed pattern (shrinks to nothing as patterns arrive).
#[allow(dead_code)]
const _UNUSED_SETTINGS: [Setting; 8] = [
    BODY_VERY_LONG, BODY_SHORT, SHADOW_LONG, SHADOW_VERY_LONG, SHADOW_SHORT, NEAR, FAR, EQUAL,
];

#[inline]
fn realbody(o: &[f64], c: &[f64], i: usize) -> f64 {
    (c[i] - o[i]).abs()
}
#[inline]
fn uppershadow(o: &[f64], h: &[f64], c: &[f64], i: usize) -> f64 {
    h[i] - o[i].max(c[i])
}
#[inline]
fn lowershadow(o: &[f64], l: &[f64], c: &[f64], i: usize) -> f64 {
    o[i].min(c[i]) - l[i]
}
/// `+1.0` white (close ≥ open), `-1.0` black.
#[inline]
fn color(o: &[f64], c: &[f64], i: usize) -> f64 {
    if c[i] >= o[i] {
        1.0
    } else {
        -1.0
    }
}

/// The setting's range at bar `i`.
#[inline]
fn range(s: Setting, o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize) -> f64 {
    match s.range {
        RangeType::RealBody => realbody(o, c, i),
        RangeType::HighLow => h[i] - l[i],
        RangeType::Shadows => uppershadow(o, h, c, i) + lowershadow(o, l, c, i),
    }
}

/// TA-Lib's `TA_CANDLEAVERAGE`: `factor · (avg of range over the prior avg_period bars,
/// or the bar's own range when avg_period == 0) / (2 if Shadows else 1)`. The average
/// window is `[i-avg_period, i-1]` — the bars *before* `i` (so callers start at `i >=
/// avg_period`).
fn candle_average(s: Setting, o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize) -> f64 {
    let base = if s.avg_period != 0 {
        let mut sum = 0.0;
        for j in (i - s.avg_period)..i {
            sum += range(s, o, h, l, c, j);
        }
        sum / s.avg_period as f64
    } else {
        range(s, o, h, l, c, i)
    };
    let div = if matches!(s.range, RangeType::Shadows) {
        2.0
    } else {
        1.0
    };
    s.factor * base / div
}
