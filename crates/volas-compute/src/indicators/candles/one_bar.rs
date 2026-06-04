//! Single-bar candlestick patterns.

use super::{
    candle_average, color, lowershadow, realbody, uppershadow, BODY_DOJI, BODY_LONG, BODY_SHORT,
    SHADOW_SHORT, SHADOW_VERY_LONG, SHADOW_VERY_SHORT,
};

/// Build a per-bar pattern: NaN before `lookback`, then `f(i)` (0 / ±100) per bar.
#[inline]
fn each_bar(n: usize, lookback: usize, f: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut out = vec![f64::NAN; n];
    for i in lookback..n {
        out[i] = f(i);
    }
    out
}

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
