//! Four- and five-bar candlestick patterns.

use super::{
    candle_average, color, each_bar, lowershadow, realbody, realbody_gap_down, realbody_gap_up,
    uppershadow, BODY_LONG, NEAR, SHADOW_VERY_SHORT,
};

/// Three-Line Strike (TA-Lib CDL3LINESTRIKE): three same-colour candles in a row, each
/// opening near the prior body, then a 4th opposite candle that engulfs the move.
/// `color(3rd)·100`. Lookback 8 (4-bar window + Near).
pub fn cdl_3linestrike(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = NEAR.avg_period + 3;
    // bar `at` opens within/near bar `body`'s real body.
    let near_open = |o: &[f64], c: &[f64], at: usize, body: usize, near: f64| {
        o[at] >= o[body].min(c[body]) - near && o[at] <= o[body].max(c[body]) + near
    };
    each_bar(c.len(), lb, |i| {
        let same3 =
            color(o, c, i - 3) == color(o, c, i - 2) && color(o, c, i - 2) == color(o, c, i - 1);
        let opens_ok = near_open(o, c, i - 2, i - 3, candle_average(NEAR, o, h, l, c, i - 3))
            && near_open(o, c, i - 1, i - 2, candle_average(NEAR, o, h, l, c, i - 2));
        let three_white = color(o, c, i - 1) > 0.0
            && c[i - 1] > c[i - 2]
            && c[i - 2] > c[i - 3]
            && o[i] > c[i - 1]
            && c[i] < o[i - 3];
        let three_black = color(o, c, i - 1) < 0.0
            && c[i - 1] < c[i - 2]
            && c[i - 2] < c[i - 3]
            && o[i] < c[i - 1]
            && c[i] > o[i - 3];
        if same3
            && color(o, c, i) == -color(o, c, i - 1)
            && opens_ok
            && (three_white || three_black)
        {
            color(o, c, i - 1) * 100.0
        } else {
            0.0
        }
    })
}

/// Breakaway (TA-Lib CDLBREAKAWAY): a long body, a gap, two more in the same direction,
/// then a 5th opposite candle closing back into the gap. `color(i)·100`. Lookback 14.
pub fn cdl_breakaway(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period + 4;
    each_bar(c.len(), lb, |i| {
        let black = color(o, c, i - 4) < 0.0
            && realbody_gap_down(o, c, i - 3, i - 4)
            && h[i - 2] < h[i - 3]
            && l[i - 2] < l[i - 3]
            && h[i - 1] < h[i - 2]
            && l[i - 1] < l[i - 2]
            && c[i] > o[i - 3]
            && c[i] < c[i - 4];
        let white = color(o, c, i - 4) > 0.0
            && realbody_gap_up(o, c, i - 3, i - 4)
            && h[i - 2] > h[i - 3]
            && l[i - 2] > l[i - 3]
            && h[i - 1] > h[i - 2]
            && l[i - 1] > l[i - 2]
            && c[i] < o[i - 3]
            && c[i] > c[i - 4];
        if realbody(o, c, i - 4) > candle_average(BODY_LONG, o, h, l, c, i - 4)
            && color(o, c, i - 4) == color(o, c, i - 3)
            && color(o, c, i - 3) == color(o, c, i - 1)
            && color(o, c, i - 1) == -color(o, c, i)
            && (black || white)
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Ladder Bottom (TA-Lib CDLLADDERBOTTOM): three falling black candles, a 4th black with
/// an upper shadow, then a white candle closing above the 4th's high — bullish `100`.
/// Lookback 14.
pub fn cdl_ladderbottom(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period + 4;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 4) < 0.0
            && color(o, c, i - 3) < 0.0
            && color(o, c, i - 2) < 0.0
            && o[i - 4] > o[i - 3]
            && o[i - 3] > o[i - 2]
            && c[i - 4] > c[i - 3]
            && c[i - 3] > c[i - 2]
            && color(o, c, i - 1) < 0.0
            && uppershadow(o, h, c, i - 1) > candle_average(SHADOW_VERY_SHORT, o, h, l, c, i - 1)
            && color(o, c, i) > 0.0
            && o[i] > o[i - 1]
            && c[i] > h[i - 1]
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Concealing Baby Swallow (TA-Lib CDLCONCEALBABYSWALL): two black marubozu, a 3rd black
/// gapping down with an upper shadow into the 2nd body, then a 4th black engulfing the
/// 3rd including its shadows — bullish `100`. Lookback 13.
pub fn cdl_concealbabyswall(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period + 3;
    each_bar(c.len(), lb, |i| {
        let vss = |k: usize| candle_average(SHADOW_VERY_SHORT, o, h, l, c, k);
        let marubozu = |k: usize| uppershadow(o, h, c, k) < vss(k) && lowershadow(o, l, c, k) < vss(k);
        if color(o, c, i - 3) < 0.0
            && color(o, c, i - 2) < 0.0
            && color(o, c, i - 1) < 0.0
            && color(o, c, i) < 0.0
            && marubozu(i - 3)
            && marubozu(i - 2)
            && realbody_gap_down(o, c, i - 1, i - 2)
            && uppershadow(o, h, c, i - 1) > vss(i - 1)
            && h[i - 1] > c[i - 2]
            && h[i] > h[i - 1]
            && l[i] < l[i - 1]
        {
            100.0
        } else {
            0.0
        }
    })
}
