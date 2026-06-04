//! Single-bar candlestick patterns.

use super::{
    candle_average, color, each_bar, lowershadow, realbody, realbody_gap_down, realbody_gap_up,
    uppershadow, BODY_DOJI, BODY_LONG, BODY_SHORT, NEAR, SHADOW_LONG, SHADOW_SHORT,
    SHADOW_VERY_LONG, SHADOW_VERY_SHORT,
};

/// Doji (TA-Lib CDLDOJI): real body no larger than the recent doji-body threshold —
/// open ≈ close. Non-directional (uncertainty): `100` or `0`. Lookback 10.
pub fn cdl_doji(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    each_bar(c.len(), BODY_DOJI.avg_period, |i| {
        if realbody(o, c, i) <= candle_average(BODY_DOJI, o, h, l, c, i) {
            100.0
        } else {
            0.0
        }
    })
}

/// Marubozu (TA-Lib CDLMARUBOZU): long body, negligible shadows both ends. Lookback 10.
pub fn cdl_marubozu(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    each_bar(c.len(), lb, |i| {
        let vs = candle_average(SHADOW_VERY_SHORT, o, h, l, c, i);
        if realbody(o, c, i) > candle_average(BODY_LONG, o, h, l, c, i)
            && uppershadow(o, h, c, i) < vs
            && lowershadow(o, l, c, i) < vs
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Closing Marubozu (TA-Lib CDLCLOSINGMARUBOZU): long body whose *closing* end has no
/// shadow (white → no upper shadow, black → no lower shadow). Lookback 10.
pub fn cdl_closingmarubozu(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    each_bar(c.len(), lb, |i| {
        let vs = candle_average(SHADOW_VERY_SHORT, o, h, l, c, i);
        let white = color(o, c, i) > 0.0;
        let long_body = realbody(o, c, i) > candle_average(BODY_LONG, o, h, l, c, i);
        let closed_marubozu = (white && uppershadow(o, h, c, i) < vs)
            || (!white && lowershadow(o, l, c, i) < vs);
        if long_body && closed_marubozu {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Long Line (TA-Lib CDLLONGLINE): long body, short shadows both ends. Lookback 10.
pub fn cdl_longline(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period.max(SHADOW_SHORT.avg_period);
    each_bar(c.len(), lb, |i| {
        let short = candle_average(SHADOW_SHORT, o, h, l, c, i);
        if realbody(o, c, i) > candle_average(BODY_LONG, o, h, l, c, i)
            && uppershadow(o, h, c, i) < short
            && lowershadow(o, l, c, i) < short
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Short Line (TA-Lib CDLSHORTLINE): short body, short shadows both ends. Lookback 10.
pub fn cdl_shortline(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(SHADOW_SHORT.avg_period);
    each_bar(c.len(), lb, |i| {
        let short = candle_average(SHADOW_SHORT, o, h, l, c, i);
        if realbody(o, c, i) < candle_average(BODY_SHORT, o, h, l, c, i)
            && uppershadow(o, h, c, i) < short
            && lowershadow(o, l, c, i) < short
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// High-Wave (TA-Lib CDLHIGHWAVE): short body, very long shadows both ends. Lookback 10.
pub fn cdl_highwave(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(SHADOW_VERY_LONG.avg_period);
    each_bar(c.len(), lb, |i| {
        let very_long = candle_average(SHADOW_VERY_LONG, o, h, l, c, i);
        if realbody(o, c, i) < candle_average(BODY_SHORT, o, h, l, c, i)
            && uppershadow(o, h, c, i) > very_long
            && lowershadow(o, l, c, i) > very_long
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Spinning Top (TA-Lib CDLSPINNINGTOP): short body smaller than both shadows. Lookback 10.
pub fn cdl_spinningtop(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    each_bar(c.len(), BODY_SHORT.avg_period, |i| {
        let body = realbody(o, c, i);
        if body < candle_average(BODY_SHORT, o, h, l, c, i)
            && uppershadow(o, h, c, i) > body
            && lowershadow(o, l, c, i) > body
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Dragonfly Doji (TA-Lib CDLDRAGONFLYDOJI): doji body, very short upper shadow, long
/// lower shadow. Non-directional: `100` or `0`. Lookback 10.
pub fn cdl_dragonflydoji(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    each_bar(c.len(), lb, |i| {
        let vs = candle_average(SHADOW_VERY_SHORT, o, h, l, c, i);
        if realbody(o, c, i) <= candle_average(BODY_DOJI, o, h, l, c, i)
            && uppershadow(o, h, c, i) < vs
            && lowershadow(o, l, c, i) > vs
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Gravestone Doji (TA-Lib CDLGRAVESTONEDOJI): doji body, very short lower shadow, long
/// upper shadow. Non-directional: `100` or `0`. Lookback 10.
pub fn cdl_gravestonedoji(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    each_bar(c.len(), lb, |i| {
        let vs = candle_average(SHADOW_VERY_SHORT, o, h, l, c, i);
        if realbody(o, c, i) <= candle_average(BODY_DOJI, o, h, l, c, i)
            && lowershadow(o, l, c, i) < vs
            && uppershadow(o, h, c, i) > vs
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Long-Legged Doji (TA-Lib CDLLONGLEGGEDDOJI): doji body with at least one long shadow.
/// Non-directional: `100` or `0`. Lookback 10. (ShadowLong has avg_period 0, so its
/// threshold is the bar's own real body.)
pub fn cdl_longleggeddoji(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(SHADOW_LONG.avg_period);
    each_bar(c.len(), lb, |i| {
        let long = candle_average(SHADOW_LONG, o, h, l, c, i);
        if realbody(o, c, i) <= candle_average(BODY_DOJI, o, h, l, c, i)
            && (lowershadow(o, l, c, i) > long || uppershadow(o, h, c, i) > long)
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Rickshaw Man (TA-Lib CDLRICKSHAWMAN): a long-legged doji whose body sits near the
/// midpoint of the high-low range. Non-directional: `100` or `0`. Lookback 10.
pub fn cdl_rickshawman(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(SHADOW_LONG.avg_period).max(NEAR.avg_period);
    each_bar(c.len(), lb, |i| {
        let long = candle_average(SHADOW_LONG, o, h, l, c, i);
        let near = candle_average(NEAR, o, h, l, c, i);
        let mid = l[i] + (h[i] - l[i]) / 2.0;
        let body_lo = o[i].min(c[i]);
        let body_hi = o[i].max(c[i]);
        if realbody(o, c, i) <= candle_average(BODY_DOJI, o, h, l, c, i)
            && lowershadow(o, l, c, i) > long
            && uppershadow(o, h, c, i) > long
            && body_lo <= mid + near
            && body_hi >= mid - near
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Belt-Hold (TA-Lib CDLBELTHOLD): a long body opening at its extreme — white with no
/// lower shadow (bullish) or black with no upper shadow (bearish). Lookback 10.
pub fn cdl_belthold(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period.max(SHADOW_VERY_SHORT.avg_period);
    each_bar(c.len(), lb, |i| {
        let vs = candle_average(SHADOW_VERY_SHORT, o, h, l, c, i);
        let white = color(o, c, i) > 0.0;
        let opens_at_extreme = (white && lowershadow(o, l, c, i) < vs)
            || (!white && uppershadow(o, h, c, i) < vs);
        if realbody(o, c, i) > candle_average(BODY_LONG, o, h, l, c, i) && opens_at_extreme {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Shared body/shadow test for hammer-family candles: a small body with one long shadow
/// and a very short shadow on the other end. `long_below` true ⇒ long lower / short upper
/// (hammer, hanging man); false ⇒ long upper / short lower (inverted hammer, shooting star).
#[inline]
fn hammer_shape(o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize, long_below: bool) -> bool {
    let long = candle_average(SHADOW_LONG, o, h, l, c, i);
    let very_short = candle_average(SHADOW_VERY_SHORT, o, h, l, c, i);
    let small_body = realbody(o, c, i) < candle_average(BODY_SHORT, o, h, l, c, i);
    let (upper, lower) = (uppershadow(o, h, c, i), lowershadow(o, l, c, i));
    if long_below {
        small_body && lower > long && upper < very_short
    } else {
        small_body && upper > long && lower < very_short
    }
}

/// Hammer (TA-Lib CDLHAMMER): a hammer-shape candle whose body sits near the prior bar's
/// low — bullish reversal `100`. Lookback 11 (prior-bar reference).
pub fn cdl_hammer(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(SHADOW_VERY_SHORT.avg_period).max(NEAR.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if hammer_shape(o, h, l, c, i, true)
            && o[i].min(c[i]) <= l[i - 1] + candle_average(NEAR, o, h, l, c, i - 1)
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Hanging Man (TA-Lib CDLHANGINGMAN): a hammer-shape candle whose body sits near the
/// prior bar's high — bearish reversal `-100`. Lookback 11.
pub fn cdl_hangingman(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(SHADOW_VERY_SHORT.avg_period).max(NEAR.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if hammer_shape(o, h, l, c, i, true)
            && o[i].min(c[i]) >= h[i - 1] - candle_average(NEAR, o, h, l, c, i - 1)
        {
            -100.0
        } else {
            0.0
        }
    })
}

/// Inverted Hammer (TA-Lib CDLINVERTEDHAMMER): inverted hammer-shape gapping down from
/// the prior body — bullish `100`. Lookback 11.
pub fn cdl_invertedhammer(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(SHADOW_VERY_SHORT.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if hammer_shape(o, h, l, c, i, false) && realbody_gap_down(o, c, i, i - 1) {
            100.0
        } else {
            0.0
        }
    })
}

/// Shooting Star (TA-Lib CDLSHOOTINGSTAR): inverted hammer-shape gapping up from the
/// prior body — bearish `-100`. Lookback 11.
pub fn cdl_shootingstar(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(SHADOW_VERY_SHORT.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if hammer_shape(o, h, l, c, i, false) && realbody_gap_up(o, c, i, i - 1) {
            -100.0
        } else {
            0.0
        }
    })
}

/// Takuri (TA-Lib CDLTAKURI): a dragonfly doji with an exceptionally long lower shadow.
/// Non-directional `100`. Lookback 10.
pub fn cdl_takuri(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(SHADOW_VERY_SHORT.avg_period).max(SHADOW_VERY_LONG.avg_period);
    each_bar(c.len(), lb, |i| {
        if realbody(o, c, i) <= candle_average(BODY_DOJI, o, h, l, c, i)
            && uppershadow(o, h, c, i) < candle_average(SHADOW_VERY_SHORT, o, h, l, c, i)
            && lowershadow(o, l, c, i) > candle_average(SHADOW_VERY_LONG, o, h, l, c, i)
        {
            100.0
        } else {
            0.0
        }
    })
}
