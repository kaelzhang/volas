//! Two-bar candlestick patterns.

use super::{
    candle_average, candle_gap_down, candle_gap_up, color, each_bar, lowershadow, realbody,
    realbody_gap_down, realbody_gap_up, uppershadow, BODY_DOJI, BODY_LONG, BODY_SHORT, EQUAL,
    SHADOW_VERY_SHORT,
};

/// A marubozu: a long real body with negligible shadows both ends.
#[inline]
fn is_marubozu(o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize) -> bool {
    let vs = candle_average(SHADOW_VERY_SHORT, o, h, l, c, i);
    realbody(o, c, i) > candle_average(BODY_LONG, o, h, l, c, i)
        && uppershadow(o, h, c, i) < vs
        && lowershadow(o, l, c, i) < vs
}

/// Whether bar `i` gaps away from bar `i-1` in the direction implied by the *prior*
/// candle's colour (a black prior gaps up, a white prior gaps down) — the kicking gap.
#[inline]
fn kicking_gap(o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize) -> bool {
    (color(o, c, i - 1) < 0.0 && candle_gap_up(h, l, i, i - 1))
        || (color(o, c, i - 1) > 0.0 && candle_gap_down(h, l, i, i - 1))
}

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

/// Doji Star (TA-Lib CDLDOJISTAR): a long body then a doji gapping in the body's
/// direction. `-color(prev)·100` (potential reversal) or 0. Lookback 11.
pub fn cdl_dojistar(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        let star = (color(o, c, i - 1) > 0.0 && realbody_gap_up(o, c, i, i - 1))
            || (color(o, c, i - 1) < 0.0 && realbody_gap_down(o, c, i, i - 1));
        if realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && realbody(o, c, i) <= candle_average(BODY_DOJI, o, h, l, c, i)
            && star
        {
            -color(o, c, i - 1) * 100.0
        } else {
            0.0
        }
    })
}

/// Homing Pigeon (TA-Lib CDLHOMINGPIGEON): two black candles, the 2nd short body inside
/// the 1st long body — bullish `100`. Lookback 11.
pub fn cdl_homingpigeon(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 1) < 0.0
            && color(o, c, i) < 0.0
            && realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && realbody(o, c, i) <= candle_average(BODY_SHORT, o, h, l, c, i)
            && o[i] < o[i - 1]
            && c[i] > c[i - 1]
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Matching Low (TA-Lib CDLMATCHINGLOW): two black candles with equal closes — bullish
/// `100`. Lookback 6 (Equal average).
pub fn cdl_matchinglow(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = EQUAL.avg_period + 1;
    each_bar(c.len(), lb, |i| {
        let eq = candle_average(EQUAL, o, h, l, c, i - 1);
        if color(o, c, i - 1) < 0.0
            && color(o, c, i) < 0.0
            && c[i] <= c[i - 1] + eq
            && c[i] >= c[i - 1] - eq
        {
            100.0
        } else {
            0.0
        }
    })
}

/// In-Neck (TA-Lib CDLINNECK): a long black candle then a white candle closing just into
/// the prior body (≈ prior close) — bearish continuation `-100`. Lookback 11.
pub fn cdl_inneck(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 1) < 0.0
            && realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && color(o, c, i) > 0.0
            && o[i] < l[i - 1]
            && c[i] <= c[i - 1] + candle_average(EQUAL, o, h, l, c, i - 1)
            && c[i] >= c[i - 1]
        {
            -100.0
        } else {
            0.0
        }
    })
}

/// On-Neck (TA-Lib CDLONNECK): a long black candle then a white candle closing ≈ the
/// prior low — bearish continuation `-100`. Lookback 11.
pub fn cdl_onneck(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        let eq = candle_average(EQUAL, o, h, l, c, i - 1);
        if color(o, c, i - 1) < 0.0
            && realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && color(o, c, i) > 0.0
            && o[i] < l[i - 1]
            && c[i] <= l[i - 1] + eq
            && c[i] >= l[i - 1] - eq
        {
            -100.0
        } else {
            0.0
        }
    })
}

/// Thrusting (TA-Lib CDLTHRUSTING): a long black candle then a white candle closing into
/// the prior body but below its midpoint — bearish continuation `-100`. Lookback 11.
pub fn cdl_thrusting(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 1) < 0.0
            && realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && color(o, c, i) > 0.0
            && o[i] < l[i - 1]
            && c[i] > c[i - 1] + candle_average(EQUAL, o, h, l, c, i - 1)
            && c[i] <= c[i - 1] + realbody(o, c, i - 1) * 0.5
        {
            -100.0
        } else {
            0.0
        }
    })
}

/// Kicking (TA-Lib CDLKICKING): two opposite-colour marubozu separated by a gap in the
/// 2nd's direction. `color(i)·100`. Lookback 11.
pub fn cdl_kicking(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 1) == -color(o, c, i)
            && is_marubozu(o, h, l, c, i - 1)
            && is_marubozu(o, h, l, c, i)
            && kicking_gap(o, h, l, c, i)
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Kicking-by-Length (TA-Lib CDLKICKINGBYLENGTH): a kicking whose signal takes the colour
/// of the *longer* marubozu. `color(longer)·100`. Lookback 11.
pub fn cdl_kickingbylength(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        if color(o, c, i - 1) == -color(o, c, i)
            && is_marubozu(o, h, l, c, i - 1)
            && is_marubozu(o, h, l, c, i)
            && kicking_gap(o, h, l, c, i)
        {
            let longer = if realbody(o, c, i) > realbody(o, c, i - 1) { i } else { i - 1 };
            color(o, c, longer) * 100.0
        } else {
            0.0
        }
    })
}

/// Separating Lines (TA-Lib CDLSEPARATINGLINES): opposite-colour candles sharing an open,
/// the 2nd a belt-hold long body — continuation `color(i)·100`. Lookback 11.
pub fn cdl_separatinglines(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period.max(BODY_LONG.avg_period).max(EQUAL.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        let eq = candle_average(EQUAL, o, h, l, c, i - 1);
        let vs = candle_average(SHADOW_VERY_SHORT, o, h, l, c, i);
        let belt_hold = (color(o, c, i) > 0.0 && lowershadow(o, l, c, i) < vs)
            || (color(o, c, i) < 0.0 && uppershadow(o, h, c, i) < vs);
        if color(o, c, i - 1) == -color(o, c, i)
            && o[i] <= o[i - 1] + eq
            && o[i] >= o[i - 1] - eq
            && realbody(o, c, i) > candle_average(BODY_LONG, o, h, l, c, i)
            && belt_hold
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}

/// Counterattack (TA-Lib CDLCOUNTERATTACK): two long opposite-colour candles with equal
/// closes. `color(i)·100`. Lookback 11.
pub fn cdl_counterattack(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar(c.len(), lb, |i| {
        let eq = candle_average(EQUAL, o, h, l, c, i - 1);
        if color(o, c, i - 1) == -color(o, c, i)
            && realbody(o, c, i - 1) > candle_average(BODY_LONG, o, h, l, c, i - 1)
            && realbody(o, c, i) > candle_average(BODY_LONG, o, h, l, c, i)
            && c[i] <= c[i - 1] + eq
            && c[i] >= c[i - 1] - eq
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}
