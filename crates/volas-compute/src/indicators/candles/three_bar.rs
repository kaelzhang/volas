//! Three-bar candlestick patterns.

use super::{
    candle_average, candle_gap_down, candle_gap_up, color, each_bar, lowershadow, realbody,
    realbody_gap_down, realbody_gap_up, uppershadow, BODY_DOJI, BODY_LONG, BODY_SHORT, FAR, NEAR,
    SHADOW_LONG, SHADOW_SHORT, SHADOW_VERY_SHORT,
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

/// Morning Doji Star (TA-Lib CDLMORNINGDOJISTAR): a morning star whose middle candle is a
/// doji — bullish `100`. Lookback 12.
pub fn cdl_morningdojistar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(BODY_LONG.avg_period).max(BODY_SHORT.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        if realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && color(o, c, i - 2) < 0.0
            && realbody(o, c, i - 1) <= candle_average(BODY_DOJI, o, h, l, c, i - 1)
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

/// Evening Doji Star (TA-Lib CDLEVENINGDOJISTAR): the bearish mirror — `-100`. Lookback 12.
pub fn cdl_eveningdojistar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(BODY_LONG.avg_period).max(BODY_SHORT.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        if realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && color(o, c, i - 2) > 0.0
            && realbody(o, c, i - 1) <= candle_average(BODY_DOJI, o, h, l, c, i - 1)
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

/// Abandoned Baby (TA-Lib CDLABANDONEDBABY): a long body, an isolated doji (gapped on
/// both sides), then a body closing well into the 1st — reversal `color(i)·100`. Lookback 12.
pub fn cdl_abandonedbaby(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(BODY_LONG.avg_period).max(BODY_SHORT.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        let long_doji_body = realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && realbody(o, c, i - 1) <= candle_average(BODY_DOJI, o, h, l, c, i - 1)
            && realbody(o, c, i) > candle_average(BODY_SHORT, o, h, l, c, i);
        let bottom = color(o, c, i - 2) > 0.0
            && color(o, c, i) < 0.0
            && c[i] < c[i - 2] - realbody(o, c, i - 2) * penetration
            && candle_gap_up(h, l, i - 1, i - 2)
            && candle_gap_down(h, l, i, i - 1);
        let top = color(o, c, i - 2) < 0.0
            && color(o, c, i) > 0.0
            && c[i] > c[i - 2] + realbody(o, c, i - 2) * penetration
            && candle_gap_down(h, l, i - 1, i - 2)
            && candle_gap_up(h, l, i, i - 1);
        if long_doji_body && (bottom || top) {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Two Crows (TA-Lib CDL2CROWS): a long white, a black gapping up, then a black opening
/// inside the 2nd and closing inside the 1st — bearish `-100`. Lookback 12.
pub fn cdl_2crows(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period + 2;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 2) > 0.0
            && realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && color(o, c, i - 1) < 0.0
            && realbody_gap_up(o, c, i - 1, i - 2)
            && color(o, c, i) < 0.0
            && o[i] < o[i - 1]
            && o[i] > c[i - 1]
            && c[i] > o[i - 2]
            && c[i] < c[i - 2]
        {
            -100.0
        } else {
            0.0
        }
    })
}

/// Upside Gap Two Crows (TA-Lib CDLUPSIDEGAP2CROWS): a long white, a short black gapping
/// up, then a black engulfing the 2nd but still closing above the 1st — bearish `-100`.
/// Lookback 12.
pub fn cdl_upsidegap2crows(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 2) > 0.0
            && realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && color(o, c, i - 1) < 0.0
            && realbody(o, c, i - 1) <= candle_average(BODY_SHORT, o, h, l, c, i - 1)
            && realbody_gap_up(o, c, i - 1, i - 2)
            && color(o, c, i) < 0.0
            && o[i] > o[i - 1]
            && c[i] < c[i - 1]
            && c[i] > c[i - 2]
        {
            -100.0
        } else {
            0.0
        }
    })
}

/// Advance Block (TA-Lib CDLADVANCEBLOCK): three rising whites whose advance weakens
/// (shrinking bodies / lengthening upper shadows) — bearish `-100`. Lookback 12.
pub fn cdl_advanceblock(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_LONG.avg_period.max(SHADOW_SHORT.avg_period).max(BODY_LONG.avg_period)
        .max(NEAR.avg_period).max(FAR.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        let (rb, rb1, rb2) = (realbody(o, c, i), realbody(o, c, i - 1), realbody(o, c, i - 2));
        let weakening = (rb1 < rb2 - candle_average(FAR, o, h, l, c, i - 2)
            && rb < rb1 + candle_average(NEAR, o, h, l, c, i - 1))
            || (rb < rb1 - candle_average(FAR, o, h, l, c, i - 1))
            || (rb < rb1
                && rb1 < rb2
                && (uppershadow(o, h, c, i) > candle_average(SHADOW_SHORT, o, h, l, c, i)
                    || uppershadow(o, h, c, i - 1) > candle_average(SHADOW_SHORT, o, h, l, c, i - 1)))
            || (rb < rb1 && uppershadow(o, h, c, i) > candle_average(SHADOW_LONG, o, h, l, c, i));
        if color(o, c, i - 2) > 0.0
            && color(o, c, i - 1) > 0.0
            && color(o, c, i) > 0.0
            && c[i] > c[i - 1]
            && c[i - 1] > c[i - 2]
            && o[i - 1] > o[i - 2]
            && o[i - 1] <= c[i - 2] + candle_average(NEAR, o, h, l, c, i - 2)
            && o[i] > o[i - 1]
            && o[i] <= c[i - 1] + candle_average(NEAR, o, h, l, c, i - 1)
            && rb2 > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && uppershadow(o, h, c, i - 2) < candle_average(SHADOW_SHORT, o, h, l, c, i - 2)
            && weakening
        {
            -100.0
        } else {
            0.0
        }
    })
}

/// Stalled Pattern (TA-Lib CDLSTALLEDPATTERN): two long rising whites then a small white
/// riding on the 2nd's shoulder — bearish `-100`. Lookback 12.
pub fn cdl_stalledpattern(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period.max(BODY_SHORT.avg_period).max(SHADOW_VERY_SHORT.avg_period)
        .max(NEAR.avg_period) + 2;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 2) > 0.0
            && color(o, c, i - 1) > 0.0
            && color(o, c, i) > 0.0
            && c[i] > c[i - 1]
            && c[i - 1] > c[i - 2]
            && realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && uppershadow(o, h, c, i - 1) < candle_average(SHADOW_VERY_SHORT, o, h, l, c, i - 1)
            && o[i - 1] > o[i - 2]
            && o[i - 1] <= c[i - 2] + candle_average(NEAR, o, h, l, c, i - 2)
            && realbody(o, c, i) < candle_average(BODY_SHORT, o, h, l, c, i)
            && o[i] >= c[i - 1] - realbody(o, c, i) - candle_average(NEAR, o, h, l, c, i - 1)
        {
            -100.0
        } else {
            0.0
        }
    })
}
