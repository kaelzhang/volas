// ---------------------------------------------------------------------------
// Candlestick patterns (TA-Lib CDL*) — shared candle-settings framework
// ---------------------------------------------------------------------------
//
// Each pattern tests a candle's geometry against adaptive thresholds derived from a
// rolling average of a chosen "range" (real body / high-low / shadows) over the prior
// `avg_period` bars, scaled by a `factor` — TA-Lib's candle-settings system. Output is
// f64: `+100` bullish, `-100` bearish, `0` no pattern; the warm-up is NaN (volas's
// uniform convention — TA-Lib fills its integer outputs with 0, which would be
// indistinguishable from "no pattern"). Parity holds on the valid region.

/// The range a candle-setting measures. The full set is kept (it mirrors TA-Lib's
/// settings table); a variant is only *constructed* once a pattern that needs it lands,
/// so allow the lint until then.
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum RangeType {
    RealBody,
    HighLow,
    Shadows,
}

/// One candle-setting: which range, the averaging period (0 = the bar itself), and the
/// scaling factor. Values are TA-Lib's defaults (ta_global.c).
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

// Silence dead-code until every pattern that uses each setting lands.
#[allow(dead_code)]
const _UNUSED_SETTINGS: [Setting; 7] = [
    BODY_VERY_LONG, BODY_SHORT, SHADOW_LONG, SHADOW_VERY_LONG, NEAR, FAR, EQUAL,
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

/// Doji (TA-Lib CDLDOJI): a real body no larger than the recent doji-body threshold —
/// open ≈ close. Always non-negative (uncertainty, not direction): `100` or `0`.
/// Lookback `avg_period(BodyDoji)` = 10.
pub fn cdl_doji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    let lb = BODY_DOJI.avg_period;
    for i in lb..n {
        out[i] = if realbody(open, close, i) <= candle_average(BODY_DOJI, open, high, low, close, i)
        {
            100.0
        } else {
            0.0
        };
    }
    out
}

/// Marubozu (TA-Lib CDLMARUBOZU): a long real body with negligible shadows on both
/// ends. `color·100` (bullish white / bearish black) or `0`. Lookback
/// `max(avg_period(BodyLong), avg_period(ShadowVeryShort))` = 10.
pub fn cdl_marubozu(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    let lb = BODY_LONG.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    for i in lb..n {
        let very_short = candle_average(SHADOW_VERY_SHORT, open, high, low, close, i);
        out[i] = if realbody(open, close, i) > candle_average(BODY_LONG, open, high, low, close, i)
            && uppershadow(open, high, close, i) < very_short
            && lowershadow(open, low, close, i) < very_short
        {
            color(open, close, i) * 100.0
        } else {
            0.0
        };
    }
    out
}
