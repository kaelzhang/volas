//! Two-bar candlestick patterns.

use super::{candle_average, color, each_bar, realbody, BODY_DOJI, BODY_LONG, BODY_SHORT};

/// Engulfing (TA-Lib CDLENGULFING): the 2nd real body engulfs the 1st of the opposite
/// colour. `color(i)·100` for a strict engulf, `color(i)·80` when a boundary just
/// touches, else 0. Lookback 2.
pub fn cdl_engulfing(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let _ = (h, l);
    each_bar(c.len(), 2, |i| {
        let (ci, cp) = (color(o, c, i), color(o, c, i - 1));
        let white_over_black = ci > 0.0
            && cp < 0.0
            && ((c[i] >= o[i - 1] && o[i] < c[i - 1]) || (c[i] > o[i - 1] && o[i] <= c[i - 1]));
        let black_over_white = ci < 0.0
            && cp > 0.0
            && ((o[i] >= c[i - 1] && c[i] < o[i - 1]) || (o[i] > c[i - 1] && c[i] <= o[i - 1]));
        if white_over_black || black_over_white {
            if o[i] != c[i - 1] && c[i] != o[i - 1] {
                ci * 100.0
            } else {
                ci * 80.0
            }
        } else {
            0.0
        }
    })
}

/// Harami strength of the 2nd body inside the 1st: `100` when strictly contained, `80`
/// when a boundary just touches (`<=` / `>=`), else `0`.
#[inline]
fn harami_strength(o: &[f64], c: &[f64], i: usize) -> f64 {
    let (hi, lo) = (o[i].max(c[i]), o[i].min(c[i]));
    let (prev_hi, prev_lo) = (o[i - 1].max(c[i - 1]), o[i - 1].min(c[i - 1]));
    if hi < prev_hi && lo > prev_lo {
        100.0
    } else if hi <= prev_hi && lo >= prev_lo {
        80.0
    } else {
        0.0
    }
}

/// Harami (TA-Lib CDLHARAMI): a long body followed by a short body contained within it.
/// `-color(prev)·{100 strict | 80 touching}` (reversal against the 1st bar's colour) or
/// 0. Lookback 11.
pub fn cdl_harami(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && realbody(o, c, i) <= candle_average(BODY_SHORT, o, h, l, c, i)
        {
            -color(o, c, i - 1) * harami_strength(o, c, i)
        } else {
            0.0
        }
    })
}

/// Harami Cross (TA-Lib CDLHARAMICROSS): a harami whose 2nd body is a doji. Lookback 11.
pub fn cdl_haramicross(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && realbody(o, c, i) <= candle_average(BODY_DOJI, o, h, l, c, i)
        {
            -color(o, c, i - 1) * harami_strength(o, c, i)
        } else {
            0.0
        }
    })
}

/// Piercing (TA-Lib CDLPIERCING): a long black candle then a long white candle that
/// opens below the prior low and closes back above the prior body's midpoint — bullish
/// `100`. Lookback 11.
pub fn cdl_piercing(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period + 1;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 1) < 0.0
            && realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && color(o, c, i) > 0.0
            && realbody(o, c, i) > candle_average(BODY_LONG, o, h, l, c, i)
            && o[i] < l[i - 1]
            && c[i] < o[i - 1]
            && c[i] > c[i - 1] + realbody(o, c, i - 1) * 0.5
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Dark Cloud Cover (TA-Lib CDLDARKCLOUDCOVER): a long white candle then a black candle
/// that opens above the prior high and closes into the prior body by at least
/// `penetration` (default 0.5) — bearish `-100`. Lookback 11.
pub fn cdl_darkcloudcover(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let _ = l;
    let lb = BODY_LONG.avg_period + 1;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 1) > 0.0
            && realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && color(o, c, i) < 0.0
            && o[i] > h[i - 1]
            && c[i] > o[i - 1]
            && c[i] < c[i - 1] - realbody(o, c, i - 1) * penetration
        {
            -100.0
        } else {
            0.0
        }
    })
}
