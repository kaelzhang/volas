//! Two-bar candlestick patterns.

use super::{
    candle_average_from_total, candle_gap_down, candle_gap_up, candle_output, color, each_bar_avg2,
    lowershadow, range, realbody, realbody_gap_down, realbody_gap_up, uppershadow, BODY_DOJI,
    BODY_LONG, BODY_SHORT, EQUAL, SHADOW_VERY_SHORT,
};

// Two-bar patterns reference candle averages at the current and prior bar; `each_bar_avg2`
// carries both as scalar running sums (`cur[k]` = average at `i`, `prev[k]` = at `i-1`).

/// A marubozu at bar `i`: a long real body with negligible shadows both ends, against the
/// `body_long` / `shadow_very_short` averages already resolved for that bar.
#[inline]
fn is_marubozu(
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    i: usize,
    body_long: f64,
    vs: f64,
) -> bool {
    realbody(o, c, i) > body_long && uppershadow(o, h, c, i) < vs && lowershadow(o, l, c, i) < vs
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
    let n = c.len();
    let Some(mut out) = candle_output(n, 2) else {
        return vec![f64::NAN; n];
    };
    // Engulfing is a pure two-bar geometry check, so a dedicated loop avoids the
    // generic candle builder and keeps TA-Lib's white/black branches explicit.
    for i in 2..n {
        let white = c[i] >= o[i];
        let prev_white = c[i - 1] >= o[i - 1];
        out[i] = if white
            && !prev_white
            && ((c[i] >= o[i - 1] && o[i] < c[i - 1]) || (c[i] > o[i - 1] && o[i] <= c[i - 1]))
        {
            if o[i] != c[i - 1] && c[i] != o[i - 1] {
                100.0
            } else {
                80.0
            }
        } else if !white
            && prev_white
            && ((o[i] >= c[i - 1] && c[i] < o[i - 1]) || (o[i] > c[i - 1] && c[i] <= o[i - 1]))
        {
            if o[i] != c[i - 1] && c[i] != o[i - 1] {
                -100.0
            } else {
                -80.0
            }
        } else {
            0.0
        };
    }
    out
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
    each_bar_avg2([BODY_LONG, BODY_SHORT], lb, o, h, l, c, |i, cur, prev| {
        if realbody(o, c, i - 1) > prev[0] && realbody(o, c, i) <= cur[1] {
            -color(o, c, i - 1) * harami_strength(o, c, i)
        } else {
            0.0
        }
    })
}

/// Harami Cross (TA-Lib CDLHARAMICROSS): a harami whose 2nd body is a doji. Lookback 11.
pub fn cdl_haramicross(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar_avg2([BODY_LONG, BODY_DOJI], lb, o, h, l, c, |i, cur, prev| {
        if realbody(o, c, i - 1) > prev[0] && realbody(o, c, i) <= cur[1] {
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
    let _ = h;
    let lb = BODY_LONG.avg_period + 1;
    let n = c.len();
    let Some(mut out) = candle_output(n, lb) else {
        return vec![f64::NAN; n];
    };
    // TA-Lib carries BodyLong totals for both i-1 and i; specialize that pair instead
    // of routing through `each_bar_avg2`'s small-array loop.
    let mut prev_total = 0.0;
    let mut cur_total = 0.0;
    let mut trailing = lb - BODY_LONG.avg_period;
    for j in trailing..lb {
        prev_total += range(BODY_LONG, o, h, l, c, j - 1);
        cur_total += range(BODY_LONG, o, h, l, c, j);
    }
    for i in lb..n {
        let prev_body = realbody(o, c, i - 1);
        out[i] = if c[i - 1] < o[i - 1]
            && prev_body > candle_average_from_total(BODY_LONG, prev_total)
            && c[i] >= o[i]
            && realbody(o, c, i) > candle_average_from_total(BODY_LONG, cur_total)
            && o[i] < l[i - 1]
            && c[i] < o[i - 1]
            && c[i] > c[i - 1] + prev_body * 0.5
        {
            100.0
        } else {
            0.0
        };
        prev_total +=
            range(BODY_LONG, o, h, l, c, i - 1) - range(BODY_LONG, o, h, l, c, trailing - 1);
        cur_total += range(BODY_LONG, o, h, l, c, i) - range(BODY_LONG, o, h, l, c, trailing);
        trailing += 1;
    }
    out
}

/// Dark Cloud Cover (TA-Lib CDLDARKCLOUDCOVER): a long white candle then a black candle
/// that opens above the prior high and closes into the prior body by at least
/// `penetration` (default 0.5) — bearish `-100`. Lookback 11.
pub fn cdl_darkcloudcover(
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    penetration: f64,
) -> Vec<f64> {
    let _ = l;
    let lb = BODY_LONG.avg_period + 1;
    let n = c.len();
    let Some(mut out) = candle_output(n, lb) else {
        return vec![f64::NAN; n];
    };
    // Dark Cloud Cover only reads BodyLong at i-1, so this mirrors TA-Lib's one-total
    // loop and avoids computing an unused current average.
    let mut total = 0.0;
    let mut trailing = lb - BODY_LONG.avg_period;
    for j in trailing..lb {
        total += range(BODY_LONG, o, h, l, c, j - 1);
    }
    for i in lb..n {
        let prev_body = realbody(o, c, i - 1);
        out[i] = if c[i - 1] >= o[i - 1]
            && prev_body > candle_average_from_total(BODY_LONG, total)
            && c[i] < o[i]
            && o[i] > h[i - 1]
            && c[i] > o[i - 1]
            && c[i] < c[i - 1] - prev_body * penetration
        {
            -100.0
        } else {
            0.0
        };
        total += range(BODY_LONG, o, h, l, c, i - 1) - range(BODY_LONG, o, h, l, c, trailing - 1);
        trailing += 1;
    }
    out
}

/// Doji Star (TA-Lib CDLDOJISTAR): a long body then a doji gapping in the body's
/// direction. `-color(prev)·100` (potential reversal) or 0. Lookback 11.
pub fn cdl_dojistar(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_DOJI.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar_avg2([BODY_LONG, BODY_DOJI], lb, o, h, l, c, |i, cur, prev| {
        let star = (color(o, c, i - 1) > 0.0 && realbody_gap_up(o, c, i, i - 1))
            || (color(o, c, i - 1) < 0.0 && realbody_gap_down(o, c, i, i - 1));
        if realbody(o, c, i - 1) > prev[0] && realbody(o, c, i) <= cur[1] && star {
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
    each_bar_avg2([BODY_LONG, BODY_SHORT], lb, o, h, l, c, |i, cur, prev| {
        if color(o, c, i - 1) < 0.0
            && color(o, c, i) < 0.0
            && realbody(o, c, i - 1) > prev[0]
            && realbody(o, c, i) <= cur[1]
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
    each_bar_avg2([EQUAL], lb, o, h, l, c, |i, _cur, prev| {
        let eq = prev[0];
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
    let _ = h;
    let lb = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    let n = c.len();
    let Some(mut out) = candle_output(n, lb) else {
        return vec![f64::NAN; n];
    };
    // In-Neck reads both BodyLong and Equal at i-1. Two scalar totals match TA-Lib and
    // skip `each_bar_avg2`'s current averages, which this pattern never consumes.
    let mut body_total = 0.0;
    let mut equal_total = 0.0;
    let mut body_trailing = lb - BODY_LONG.avg_period;
    let mut equal_trailing = lb - EQUAL.avg_period;
    for j in body_trailing..lb {
        body_total += range(BODY_LONG, o, h, l, c, j - 1);
    }
    for j in equal_trailing..lb {
        equal_total += range(EQUAL, o, h, l, c, j - 1);
    }
    let body_long_scale = BODY_LONG.factor / BODY_LONG.avg_period as f64;
    let equal_scale = EQUAL.factor / EQUAL.avg_period as f64;
    for i in lb..n {
        out[i] = if c[i - 1] < o[i - 1]
            && realbody(o, c, i - 1) > body_total * body_long_scale
            && c[i] >= o[i]
            && o[i] < l[i - 1]
            && c[i] <= c[i - 1] + equal_total * equal_scale
            && c[i] >= c[i - 1]
        {
            -100.0
        } else {
            0.0
        };
        body_total +=
            range(BODY_LONG, o, h, l, c, i - 1) - range(BODY_LONG, o, h, l, c, body_trailing - 1);
        equal_total +=
            range(EQUAL, o, h, l, c, i - 1) - range(EQUAL, o, h, l, c, equal_trailing - 1);
        body_trailing += 1;
        equal_trailing += 1;
    }
    out
}

/// On-Neck (TA-Lib CDLONNECK): a long black candle then a white candle closing ≈ the
/// prior low — bearish continuation `-100`. Lookback 11.
pub fn cdl_onneck(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar_avg2([BODY_LONG, EQUAL], lb, o, h, l, c, |i, _cur, prev| {
        let eq = prev[1];
        if color(o, c, i - 1) < 0.0
            && realbody(o, c, i - 1) > prev[0]
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
    each_bar_avg2([BODY_LONG, EQUAL], lb, o, h, l, c, |i, _cur, prev| {
        if color(o, c, i - 1) < 0.0
            && realbody(o, c, i - 1) > prev[0]
            && color(o, c, i) > 0.0
            && o[i] < l[i - 1]
            && c[i] > c[i - 1] + prev[1]
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
    each_bar_avg2(
        [BODY_LONG, SHADOW_VERY_SHORT],
        lb,
        o,
        h,
        l,
        c,
        |i, cur, prev| {
            if color(o, c, i - 1) == -color(o, c, i)
                && is_marubozu(o, h, l, c, i - 1, prev[0], prev[1])
                && is_marubozu(o, h, l, c, i, cur[0], cur[1])
                && kicking_gap(o, h, l, c, i)
            {
                color(o, c, i) * 100.0
            } else {
                0.0
            }
        },
    )
}

/// Kicking-by-Length (TA-Lib CDLKICKINGBYLENGTH): a kicking whose signal takes the colour
/// of the *longer* marubozu. `color(longer)·100`. Lookback 11.
pub fn cdl_kickingbylength(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar_avg2(
        [BODY_LONG, SHADOW_VERY_SHORT],
        lb,
        o,
        h,
        l,
        c,
        |i, cur, prev| {
            if color(o, c, i - 1) == -color(o, c, i)
                && is_marubozu(o, h, l, c, i - 1, prev[0], prev[1])
                && is_marubozu(o, h, l, c, i, cur[0], cur[1])
                && kicking_gap(o, h, l, c, i)
            {
                let longer = if realbody(o, c, i) > realbody(o, c, i - 1) {
                    i
                } else {
                    i - 1
                };
                color(o, c, longer) * 100.0
            } else {
                0.0
            }
        },
    )
}

/// Separating Lines (TA-Lib CDLSEPARATINGLINES): opposite-colour candles sharing an open,
/// the 2nd a belt-hold long body — continuation `color(i)·100`. Lookback 11.
pub fn cdl_separatinglines(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = SHADOW_VERY_SHORT
        .avg_period
        .max(BODY_LONG.avg_period)
        .max(EQUAL.avg_period)
        + 1;
    each_bar_avg2(
        [BODY_LONG, SHADOW_VERY_SHORT, EQUAL],
        lb,
        o,
        h,
        l,
        c,
        |i, cur, prev| {
            let eq = prev[2];
            let vs = cur[1];
            let belt_hold = (color(o, c, i) > 0.0 && lowershadow(o, l, c, i) < vs)
                || (color(o, c, i) < 0.0 && uppershadow(o, h, c, i) < vs);
            if color(o, c, i - 1) == -color(o, c, i)
                && o[i] <= o[i - 1] + eq
                && o[i] >= o[i - 1] - eq
                && realbody(o, c, i) > cur[0]
                && belt_hold
            {
                color(o, c, i) * 100.0
            } else {
                0.0
            }
        },
    )
}

/// Counterattack (TA-Lib CDLCOUNTERATTACK): two long opposite-colour candles with equal
/// closes. `color(i)·100`. Lookback 11.
pub fn cdl_counterattack(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = EQUAL.avg_period.max(BODY_LONG.avg_period) + 1;
    each_bar_avg2([BODY_LONG, EQUAL], lb, o, h, l, c, |i, cur, prev| {
        let eq = prev[1];
        if color(o, c, i - 1) == -color(o, c, i)
            && realbody(o, c, i - 1) > prev[0]
            && realbody(o, c, i) > cur[0]
            && c[i] <= c[i - 1] + eq
            && c[i] >= c[i - 1] - eq
        {
            color(o, c, i) * 100.0
        } else {
            0.0
        }
    })
}
