//! Three-bar candlestick patterns.

use super::{
    candle_average, candle_gap_down, candle_gap_up, candle_output, color, each_bar, each_bar_avg_n,
    lowershadow, range, realbody, realbody_gap_down, realbody_gap_up, uppershadow, BODY_DOJI,
    BODY_LONG, BODY_SHORT, EQUAL, FAR, NEAR, SHADOW_LONG, SHADOW_SHORT, SHADOW_VERY_SHORT,
};

/// Morning Star (TA-Lib CDLMORNINGSTAR): a long black body, a short star gapping down,
/// then a white body closing well into the 1st — bullish reversal `100`. Lookback 12.
pub fn cdl_morningstar(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    each_bar_avg_n::<2, 3>([BODY_LONG, BODY_SHORT], lb, o, h, l, c, |i, hist| {
        if realbody(o, c, i - 2) > hist[2][0] // BODY_LONG at i-2
            && color(o, c, i - 2) < 0.0
            && realbody(o, c, i - 1) <= hist[1][1] // BODY_SHORT at i-1
            && realbody_gap_down(o, c, i - 1, i - 2)
            && realbody(o, c, i) > hist[0][1] // BODY_SHORT at i
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
    // Mirror Morning Star's rolling-total path: the three averages are fixed lags
    // (long at i-2, short at i-1 and i), so avoid per-bar candle-average rescans.
    each_bar_avg_n::<2, 3>([BODY_LONG, BODY_SHORT], lb, o, h, l, c, |i, hist| {
        if realbody(o, c, i - 2) > hist[2][0] // BODY_LONG at i-2
            && color(o, c, i - 2) > 0.0
            && realbody(o, c, i - 1) <= hist[1][1] // BODY_SHORT at i-1
            && realbody_gap_up(o, c, i - 1, i - 2)
            && realbody(o, c, i) > hist[0][1] // BODY_SHORT at i
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
    each_bar_avg_n::<2, 3>([BODY_LONG, BODY_SHORT], lb, o, h, l, c, |i, hist| {
        let engulfed = o[i - 1].max(c[i - 1]) < o[i - 2].max(c[i - 2])
            && o[i - 1].min(c[i - 1]) > o[i - 2].min(c[i - 2]);
        let confirmed = (color(o, c, i - 2) > 0.0 && color(o, c, i) < 0.0 && c[i] < o[i - 2])
            || (color(o, c, i - 2) < 0.0 && color(o, c, i) > 0.0 && c[i] > o[i - 2]);
        if realbody(o, c, i - 2) > hist[2][0] // BODY_LONG at i-2
            && realbody(o, c, i - 1) <= hist[1][1] // BODY_SHORT at i-1
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
    let lb = SHADOW_VERY_SHORT
        .avg_period
        .max(BODY_SHORT.avg_period)
        .max(NEAR.avg_period)
        .max(FAR.avg_period)
        + 2;
    each_bar_avg_n::<4, 3>(
        [SHADOW_VERY_SHORT, NEAR, FAR, BODY_SHORT],
        lb,
        o,
        h,
        l,
        c,
        |i, hist| {
            let short_upper = |bar: usize, lag: usize| uppershadow(o, h, c, bar) < hist[lag][0];
            if color(o, c, i - 2) > 0.0
            && short_upper(i - 2, 2)
            && color(o, c, i - 1) > 0.0
            && short_upper(i - 1, 1)
            && color(o, c, i) > 0.0
            && short_upper(i, 0)
            && c[i] > c[i - 1]
            && c[i - 1] > c[i - 2]
            && o[i - 1] > o[i - 2]
            && o[i - 1] <= c[i - 2] + hist[2][1] // NEAR at i-2
            && o[i] > o[i - 1]
            && o[i] <= c[i - 1] + hist[1][1] // NEAR at i-1
            && realbody(o, c, i - 1) > realbody(o, c, i - 2) - hist[2][2] // FAR at i-2
            && realbody(o, c, i) > realbody(o, c, i - 1) - hist[1][2] // FAR at i-1
            && realbody(o, c, i) > hist[0][3]
            // BODY_SHORT at i
            {
                100.0
            } else {
                0.0
            }
        },
    )
}

/// Three Black Crows (TA-Lib CDL3BLACKCROWS): after a white candle, three falling black
/// candles each opening within the prior body with tiny lower shadows — bearish `-100`.
/// Lookback 13.
pub fn cdl_3blackcrows(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period + 3;
    each_bar(c.len(), lb, |i| {
        let short_lower =
            |k: usize| lowershadow(o, l, c, k) < candle_average(SHADOW_VERY_SHORT, o, h, l, c, k);
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
pub fn cdl_morningdojistar(
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    penetration: f64,
) -> Vec<f64> {
    let lb = BODY_DOJI
        .avg_period
        .max(BODY_LONG.avg_period)
        .max(BODY_SHORT.avg_period)
        + 2;
    each_bar_avg_n::<3, 3>(
        [BODY_LONG, BODY_DOJI, BODY_SHORT],
        lb,
        o,
        h,
        l,
        c,
        |i, hist| {
            if realbody(o, c, i - 2) > hist[2][0] // BODY_LONG at i-2
            && color(o, c, i - 2) < 0.0
            && realbody(o, c, i - 1) <= hist[1][1] // BODY_DOJI at i-1
            && realbody_gap_down(o, c, i - 1, i - 2)
            && realbody(o, c, i) > hist[0][2] // BODY_SHORT at i
            && color(o, c, i) > 0.0
            && c[i] > c[i - 2] + realbody(o, c, i - 2) * penetration
            {
                100.0
            } else {
                0.0
            }
        },
    )
}

/// Evening Doji Star (TA-Lib CDLEVENINGDOJISTAR): the bearish mirror — `-100`. Lookback 12.
pub fn cdl_eveningdojistar(
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    penetration: f64,
) -> Vec<f64> {
    let lb = BODY_DOJI
        .avg_period
        .max(BODY_LONG.avg_period)
        .max(BODY_SHORT.avg_period)
        + 2;
    each_bar_avg_n::<3, 3>(
        [BODY_LONG, BODY_DOJI, BODY_SHORT],
        lb,
        o,
        h,
        l,
        c,
        |i, hist| {
            if realbody(o, c, i - 2) > hist[2][0] // BODY_LONG at i-2
            && color(o, c, i - 2) > 0.0
            && realbody(o, c, i - 1) <= hist[1][1] // BODY_DOJI at i-1
            && realbody_gap_up(o, c, i - 1, i - 2)
            && realbody(o, c, i) > hist[0][2] // BODY_SHORT at i
            && color(o, c, i) < 0.0
            && c[i] < c[i - 2] - realbody(o, c, i - 2) * penetration
            {
                -100.0
            } else {
                0.0
            }
        },
    )
}

/// Abandoned Baby (TA-Lib CDLABANDONEDBABY): a long body, an isolated doji (gapped on
/// both sides), then a body closing well into the 1st — reversal `color(i)·100`. Lookback 12.
pub fn cdl_abandonedbaby(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let lb = BODY_DOJI
        .avg_period
        .max(BODY_LONG.avg_period)
        .max(BODY_SHORT.avg_period)
        + 2;
    each_bar_avg_n::<3, 3>(
        [BODY_LONG, BODY_DOJI, BODY_SHORT],
        lb,
        o,
        h,
        l,
        c,
        |i, hist| {
            let long_doji_body = realbody(o, c, i - 2) > hist[2][0] // BODY_LONG at i-2
            && realbody(o, c, i - 1) <= hist[1][1] // BODY_DOJI at i-1
            && realbody(o, c, i) > hist[0][2]; // BODY_SHORT at i
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
        },
    )
}

/// Two Crows (TA-Lib CDL2CROWS): a long white, a black gapping up, then a black opening
/// inside the 2nd and closing inside the 1st — bearish `-100`. Lookback 12.
pub fn cdl_2crows(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period + 2;
    let n = c.len();
    let Some(mut out) = candle_output(n, lb) else {
        return vec![f64::NAN; n];
    };
    // TA-Lib's implementation carries only one BodyLong total at `i-2`; using the
    // generic average-history helper adds unnecessary per-row history movement here.
    let mut total = 0.0;
    let mut trailing = lb - 2 - BODY_LONG.avg_period;
    for j in trailing..(lb - 2) {
        total += range(BODY_LONG, o, h, l, c, j);
    }
    let body_long_scale = BODY_LONG.factor / BODY_LONG.avg_period as f64;
    for i in lb..n {
        let first_white = c[i - 2] >= o[i - 2];
        let second_black = c[i - 1] < o[i - 1];
        let third_black = c[i] < o[i];
        out.set(i, if first_white
            && second_black
            && third_black
            && realbody(o, c, i - 2) > total * body_long_scale
            && realbody_gap_up(o, c, i - 1, i - 2)
            && o[i] < o[i - 1]
            && o[i] > c[i - 1]
            && c[i] > o[i - 2]
            && c[i] < c[i - 2]
        {
            -100.0
        } else {
            0.0
        });
        total += range(BODY_LONG, o, h, l, c, i - 2) - range(BODY_LONG, o, h, l, c, trailing);
        trailing += 1;
    }
    out.finish()
}

/// Upside Gap Two Crows (TA-Lib CDLUPSIDEGAP2CROWS): a long white, a short black gapping
/// up, then a black engulfing the 2nd but still closing above the 1st — bearish `-100`.
/// Lookback 12.
pub fn cdl_upsidegap2crows(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    each_bar_avg_n::<2, 3>([BODY_LONG, BODY_SHORT], lb, o, h, l, c, |i, hist| {
        if color(o, c, i - 2) > 0.0
            && realbody(o, c, i - 2) > hist[2][0] // BODY_LONG at i-2
            && color(o, c, i - 1) < 0.0
            && realbody(o, c, i - 1) <= hist[1][1] // BODY_SHORT at i-1
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
    let lb = SHADOW_LONG
        .avg_period
        .max(SHADOW_SHORT.avg_period)
        .max(BODY_LONG.avg_period)
        .max(NEAR.avg_period)
        .max(FAR.avg_period)
        + 2;
    each_bar_avg_n::<5, 3>(
        [FAR, NEAR, SHADOW_SHORT, SHADOW_LONG, BODY_LONG],
        lb,
        o,
        h,
        l,
        c,
        |i, hist| {
            let (rb, rb1, rb2) = (
                realbody(o, c, i),
                realbody(o, c, i - 1),
                realbody(o, c, i - 2),
            );
            let weakening = (rb1 < rb2 - hist[2][0] // FAR at i-2
            && rb < rb1 + hist[1][1]) // NEAR at i-1
            || (rb < rb1 - hist[1][0]) // FAR at i-1
            || (rb < rb1
                && rb1 < rb2
                && (uppershadow(o, h, c, i) > hist[0][2] // SHADOW_SHORT at i
                    || uppershadow(o, h, c, i - 1) > hist[1][2])) // SHADOW_SHORT at i-1
            || (rb < rb1 && uppershadow(o, h, c, i) > hist[0][3]); // SHADOW_LONG at i
            if color(o, c, i - 2) > 0.0
            && color(o, c, i - 1) > 0.0
            && color(o, c, i) > 0.0
            && c[i] > c[i - 1]
            && c[i - 1] > c[i - 2]
            && o[i - 1] > o[i - 2]
            && o[i - 1] <= c[i - 2] + hist[2][1] // NEAR at i-2
            && o[i] > o[i - 1]
            && o[i] <= c[i - 1] + hist[1][1] // NEAR at i-1
            && rb2 > hist[2][4] // BODY_LONG at i-2
            && uppershadow(o, h, c, i - 2) < hist[2][2] // SHADOW_SHORT at i-2
            && weakening
            {
                -100.0
            } else {
                0.0
            }
        },
    )
}

/// Stalled Pattern (TA-Lib CDLSTALLEDPATTERN): two long rising whites then a small white
/// riding on the 2nd's shoulder — bearish `-100`. Lookback 12.
pub fn cdl_stalledpattern(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG
        .avg_period
        .max(BODY_SHORT.avg_period)
        .max(SHADOW_VERY_SHORT.avg_period)
        .max(NEAR.avg_period)
        + 2;
    let Some(mut out) = candle_output(c.len(), lb) else {
        return vec![f64::NAN; c.len()];
    };
    // This branch-heavy pattern regressed when routed through the generic
    // per-bar closure helper; keep the hot loop local and benchmark before
    // moving it back to shared candle infrastructure.
    let mut i = lb;
    while i < c.len() {
        out.set(i, if color(o, c, i - 2) > 0.0
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
        });
        i += 1;
    }
    out.finish()
}

/// Identical Three Crows (TA-Lib CDLIDENTICAL3CROWS): three declining black candles with
/// tiny lower shadows, each opening ≈ the prior close — bearish `-100`. Lookback 12.
pub fn cdl_identical3crows(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period.max(EQUAL.avg_period) + 2;
    each_bar_avg_n::<2, 3>([SHADOW_VERY_SHORT, EQUAL], lb, o, h, l, c, |i, hist| {
        // settings[0] = SHADOW_VERY_SHORT (per lag), settings[1] = EQUAL.
        let short_lower = |bar: usize, lag: usize| lowershadow(o, l, c, bar) < hist[lag][0];
        let eq2 = hist[2][1]; // EQUAL at i-2
        let eq1 = hist[1][1]; // EQUAL at i-1
        if color(o, c, i - 2) < 0.0
            && short_lower(i - 2, 2)
            && color(o, c, i - 1) < 0.0
            && short_lower(i - 1, 1)
            && color(o, c, i) < 0.0
            && short_lower(i, 0)
            && c[i - 2] > c[i - 1]
            && c[i - 1] > c[i]
            && o[i - 1] <= c[i - 2] + eq2
            && o[i - 1] >= c[i - 2] - eq2
            && o[i] <= c[i - 1] + eq1
            && o[i] >= c[i - 1] - eq1
        {
            -100.0
        } else {
            0.0
        }
    })
}

/// Stick Sandwich (TA-Lib CDLSTICKSANDWICH): black, white (its low above the 1st close),
/// black with the same close as the 1st — bullish `100`. Lookback 7.
pub fn cdl_sticksandwich(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = EQUAL.avg_period + 2;
    each_bar_avg_n::<1, 3>([EQUAL], lb, o, h, l, c, |i, hist| {
        let eq = hist[2][0]; // EQUAL at i-2
        if color(o, c, i - 2) < 0.0
            && color(o, c, i - 1) > 0.0
            && color(o, c, i) < 0.0
            && l[i - 1] > c[i - 2]
            && c[i] <= c[i - 2] + eq
            && c[i] >= c[i - 2] - eq
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Tristar (TA-Lib CDLTRISTAR): three doji, the middle one gapping; `-100` if it gaps up
/// (3rd not higher), `+100` if it gaps down (3rd not lower), else 0. Lookback 12. (All
/// three doji bodies are tested against the threshold computed at `i-2`.)
pub fn cdl_tristar(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period + 2;
    let n = c.len();
    let Some(mut out) = candle_output(n, lb) else {
        return vec![f64::NAN; n];
    };
    // TA-Lib uses the BodyDoji average at bar i-2 for all three candles. A single
    // rolling total avoids the generic three-lag history used by other patterns.
    let mut total = 0.0;
    let mut trailing = lb - 2 - BODY_DOJI.avg_period;
    for j in trailing..(lb - 2) {
        total += h[j] - l[j];
    }
    let body_doji_scale = BODY_DOJI.factor / BODY_DOJI.avg_period as f64;
    for i in lb..n {
        let doji = total * body_doji_scale;
        let prev_lo = o[i - 2].min(c[i - 2]);
        let prev_hi = o[i - 2].max(c[i - 2]);
        let mid_lo = o[i - 1].min(c[i - 1]);
        let mid_hi = o[i - 1].max(c[i - 1]);
        let cur_lo = o[i].min(c[i]);
        let cur_hi = o[i].max(c[i]);
        out.set(i, if prev_hi - prev_lo <= doji && mid_hi - mid_lo <= doji && cur_hi - cur_lo <= doji
        {
            if mid_lo > prev_hi && cur_hi < mid_hi {
                -100.0
            } else if mid_hi < prev_lo && cur_lo > mid_lo {
                100.0
            } else {
                0.0
            }
        } else {
            0.0
        });
        total += (h[i - 2] - l[i - 2]) - (h[trailing] - l[trailing]);
        trailing += 1;
    }
    out.finish()
}

/// Unique Three River Bottom (TA-Lib CDLUNIQUE3RIVER): a long black, a black harami with
/// a new low, then a small white opening above the 2nd low — bullish `100`. Lookback 12.
pub fn cdl_unique3river(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 2;
    // The pattern reads only BODY_LONG at i-2 and BODY_SHORT at i. Carry both
    // rolling totals once instead of rescanning the 10-bar settings on every row.
    each_bar_avg_n::<2, 3>([BODY_LONG, BODY_SHORT], lb, o, h, l, c, |i, hist| {
        if realbody(o, c, i - 2) > hist[2][0] // BODY_LONG at i-2
            && color(o, c, i - 2) < 0.0
            && color(o, c, i - 1) < 0.0
            && c[i - 1] > c[i - 2]
            && o[i - 1] <= o[i - 2]
            && l[i - 1] < l[i - 2]
            && realbody(o, c, i) < hist[0][1] // BODY_SHORT at i
            && color(o, c, i) > 0.0
            && o[i] > l[i - 1]
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Gapping Side-by-Side White Lines (TA-Lib CDLGAPSIDESIDEWHITE): two same-size white
/// candles with the same open, gapping the same way off the 1st. `+100` for an up-gap,
/// `-100` for a down-gap. Lookback 7.
pub fn cdl_gapsidesidewhite(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = NEAR.avg_period.max(EQUAL.avg_period) + 2;
    each_bar_avg_n::<2, 3>([NEAR, EQUAL], lb, o, h, l, c, |i, hist| {
        let up = realbody_gap_up(o, c, i - 1, i - 2) && realbody_gap_up(o, c, i, i - 2);
        let down = realbody_gap_down(o, c, i - 1, i - 2) && realbody_gap_down(o, c, i, i - 2);
        let near = hist[1][0]; // NEAR at i-1
        let eq = hist[1][1]; // EQUAL at i-1
        if (up || down)
            && color(o, c, i - 1) > 0.0
            && color(o, c, i) > 0.0
            && realbody(o, c, i) >= realbody(o, c, i - 1) - near
            && realbody(o, c, i) <= realbody(o, c, i - 1) + near
            && o[i] >= o[i - 1] - eq
            && o[i] <= o[i - 1] + eq
        {
            if up {
                100.0
            } else {
                -100.0
            }
        } else {
            0.0
        }
    })
}

/// Tasuki Gap (TA-Lib CDLTASUKIGAP): a gap, a candle, then an opposite candle opening
/// within and closing back into the gap with a similar body. `color(1st of pair)·100`.
/// Lookback 7.
pub fn cdl_tasukigap(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = NEAR.avg_period + 2;
    let n = c.len();
    let Some(mut out) = candle_output(n, lb) else {
        return vec![f64::NAN; n];
    };
    // Tasuki Gap only reads NEAR at i-1; a single scalar rolling total beats the
    // generic history helper and lets the costly body-similarity check run only after
    // the required gap/colour geometry passes.
    let mut near_total = 0.0;
    for j in (lb - 1 - NEAR.avg_period)..(lb - 1) {
        near_total += range(NEAR, o, h, l, c, j);
    }
    let near_scale = NEAR.factor / NEAR.avg_period as f64;
    let mut trailing = lb - 1 - NEAR.avg_period;
    for i in lb..n {
        let c1 = if c[i - 1] >= o[i - 1] { 1.0 } else { -1.0 };
        let c0 = if c[i] >= o[i] { 1.0 } else { -1.0 };
        let up = c1 > 0.0
            && c0 < 0.0
            && realbody_gap_up(o, c, i - 1, i - 2)
            && o[i] < c[i - 1]
            && o[i] > o[i - 1]
            && c[i] < o[i - 1]
            && c[i] > c[i - 2].max(o[i - 2]);
        let down = c1 < 0.0
            && c0 > 0.0
            && realbody_gap_down(o, c, i - 1, i - 2)
            && o[i] < o[i - 1]
            && o[i] > c[i - 1]
            && c[i] > o[i - 1]
            && c[i] < c[i - 2].min(o[i - 2]);
        out.set(i, if up || down {
            let near = near_total * near_scale;
            if (realbody(o, c, i - 1) - realbody(o, c, i)).abs() < near {
                c1 * 100.0
            } else {
                0.0
            }
        } else {
            0.0
        });
        near_total += range(NEAR, o, h, l, c, i - 1) - range(NEAR, o, h, l, c, trailing);
        trailing += 1;
    }
    out.finish()
}

/// Three Stars in the South (TA-Lib CDL3STARSINSOUTH): a long black with a long lower
/// shadow, a smaller black holding above the prior low, then a small black marubozu
/// inside the 2nd's range — bullish reversal `100`. Lookback 12.
pub fn cdl_3starsinsouth(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT
        .avg_period
        .max(SHADOW_LONG.avg_period)
        .max(BODY_LONG.avg_period)
        .max(BODY_SHORT.avg_period)
        + 2;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 2) < 0.0
            && color(o, c, i - 1) < 0.0
            && color(o, c, i) < 0.0
            && realbody(o, c, i - 2) > candle_average(BODY_LONG, o, h, l, c, i - 2)
            && lowershadow(o, l, c, i - 2) > candle_average(SHADOW_LONG, o, h, l, c, i - 2)
            && realbody(o, c, i - 1) < realbody(o, c, i - 2)
            && o[i - 1] > c[i - 2]
            && o[i - 1] <= h[i - 2]
            && l[i - 1] < c[i - 2]
            && l[i - 1] >= l[i - 2]
            && lowershadow(o, l, c, i - 1) > candle_average(SHADOW_VERY_SHORT, o, h, l, c, i - 1)
            && realbody(o, c, i) < candle_average(BODY_SHORT, o, h, l, c, i)
            && lowershadow(o, l, c, i) < candle_average(SHADOW_VERY_SHORT, o, h, l, c, i)
            && uppershadow(o, h, c, i) < candle_average(SHADOW_VERY_SHORT, o, h, l, c, i)
            && l[i] > l[i - 1]
            && h[i] < h[i - 1]
        {
            100.0
        } else {
            0.0
        }
    })
}
