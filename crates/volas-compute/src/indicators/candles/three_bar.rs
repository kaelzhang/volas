//! Three-bar candlestick patterns.

use super::{
    candle_average, color, each_bar, lowershadow, realbody, realbody_gap_down, realbody_gap_up,
    uppershadow, BODY_LONG, BODY_SHORT, FAR, NEAR, SHADOW_VERY_SHORT,
};

/// Morning Star (TA-Lib CDLMORNINGSTAR): a long black body, a short star gapping down,
/// then a white body closing well into the 1st — bullish reversal `100`. Lookback 12.
pub fn cdl_morningstar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        if realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && color(o, c, i - 2) < 0.0
            && realbody(o, c, i - 1) <= candle_average(BODY_SHORT, o, h, l, c, i - 1)
            && realbody_gap_down(o, c, i - 1, i - 2)
            && realbody(o, c, i) > candle_average(BODY_SHORT, o, h, l, c, i)
            && color(o, c, i) > 0.0
            && c[i] > c[i - 2] + realbody(o, c, i - 2) * penetration
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Evening Star (TA-Lib CDLEVENINGSTAR): the bearish mirror of the morning star — long
/// white, short star gapping up, then black closing well into the 1st — `-100`. Lookback 12.
pub fn cdl_eveningstar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        if realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && color(o, c, i - 2) > 0.0
            && realbody(o, c, i - 1) <= candle_average(BODY_SHORT, o, h, l, c, i - 1)
            && realbody_gap_up(o, c, i - 1, i - 2)
            && realbody(o, c, i) > candle_average(BODY_SHORT, o, h, l, c, i)
            && color(o, c, i) < 0.0
            && c[i] < c[i - 2] - realbody(o, c, i - 2) * penetration
        {
            -100.0
        } else {
            0.0
        }
    })
}

/// Three Inside Up/Down (TA-Lib CDL3INSIDE): a harami (long body, engulfed short body)
/// confirmed by a 3rd candle closing out of the 1st. `-color(1st)·100`. Lookback 12.
pub fn cdl_3inside(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        let engulfed = o[i - 1].max(c[i - 1]) < o[i - 2].max(c[i - 2])
            && o[i - 1].min(c[i - 1]) > o[i - 2].min(c[i - 2]);
        let confirmed = (color(o, c, i - 2) > 0.0 && color(o, c, i) < 0.0 && c[i] < o[i - 2])
            || (color(o, c, i - 2) < 0.0 && color(o, c, i) > 0.0 && c[i] > o[i - 2]);
        if realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && realbody(o, c, i - 1) <= candle_average(BODY_SHORT, o, h, l, c, i - 1)
            && engulfed
            && confirmed
        {
            -color(o, c, i - 2) * 100.0
        } else {
            0.0
        }
    })
}

/// Three Outside Up/Down (TA-Lib CDL3OUTSIDE): an engulfing confirmed by a 3rd candle
/// continuing the move. `color(2nd)·100`. Lookback 3.
pub fn cdl_3outside(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let _ = (h, l);
    each_bar(c.len(), 3, |i| {
        let bull = color(o, c, i - 1) > 0.0
            && color(o, c, i - 2) < 0.0
            && c[i - 1] > o[i - 2]
            && o[i - 1] < c[i - 2]
            && c[i] > c[i - 1];
        let bear = color(o, c, i - 1) < 0.0
            && color(o, c, i - 2) > 0.0
            && o[i - 1] > c[i - 2]
            && c[i - 1] < o[i - 2]
            && c[i] < c[i - 1];
        if bull || bear {
            color(o, c, i - 1) * 100.0
        } else {
            0.0
        }
    })
}

/// Three White Soldiers (TA-Lib CDL3WHITESOLDIERS): three rising white candles, each
/// opening within the prior body, with tiny upper shadows and bodies not collapsing —
/// bullish `100`. Lookback 12.
pub fn cdl_3whitesoldiers(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period.max(BODY_SHORT.avg_period).max(NEAR.avg_period).max(FAR.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        let short_upper = |k: usize| uppershadow(o, h, c, k) < candle_average(SHADOW_VERY_SHORT, o, h, l, c, k);
        if color(o, c, i - 2) > 0.0
            && short_upper(i - 2)
            && color(o, c, i - 1) > 0.0
            && short_upper(i - 1)
            && color(o, c, i) > 0.0
            && short_upper(i)
            && c[i] > c[i - 1]
            && c[i - 1] > c[i - 2]
            && o[i - 1] > o[i - 2]
            && o[i - 1] <= c[i - 2] + candle_average(NEAR, o, h, l, c, i - 2)
            && o[i] > o[i - 1]
            && o[i] <= c[i - 1] + candle_average(NEAR, o, h, l, c, i - 1)
            && realbody(o, c, i - 1) > realbody(o, c, i - 2) - candle_average(FAR, o, h, l, c, i - 2)
            && realbody(o, c, i) > realbody(o, c, i - 1) - candle_average(FAR, o, h, l, c, i - 1)
            && realbody(o, c, i) > candle_average(BODY_SHORT, o, h, l, c, i)
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Three Black Crows (TA-Lib CDL3BLACKCROWS): after a white candle, three falling black
/// candles each opening within the prior body with tiny lower shadows — bearish `-100`.
/// Lookback 13.
pub fn cdl_3blackcrows(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period + 3;
    each_bar(c.len(), lb, |i| {
        let short_lower = |k: usize| lowershadow(o, l, c, k) < candle_average(SHADOW_VERY_SHORT, o, h, l, c, k);
        if color(o, c, i - 3) > 0.0
            && color(o, c, i - 2) < 0.0
            && short_lower(i - 2)
            && color(o, c, i - 1) < 0.0
            && short_lower(i - 1)
            && color(o, c, i) < 0.0
            && short_lower(i)
            && o[i - 1] < o[i - 2]
            && o[i - 1] > c[i - 2]
            && o[i] < o[i - 1]
            && o[i] > c[i - 1]
            && h[i - 3] > c[i - 2]
            && c[i - 2] > c[i - 1]
            && c[i - 1] > c[i]
        {
            -100.0
        } else {
            0.0
        }
    })
}
