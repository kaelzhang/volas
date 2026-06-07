//! Four- and five-bar candlestick patterns.

use super::{
    candle_average, candle_average_from_total, candle_output, color, each_bar, each_bar_avg_n,
    lowershadow, range, realbody, realbody_gap_down, realbody_gap_up, uppershadow, BODY_LONG,
    BODY_SHORT, NEAR, SHADOW_VERY_SHORT,
};

/// Three-Line Strike (TA-Lib CDL3LINESTRIKE): three same-colour candles in a row, each
/// opening near the prior body, then a 4th opposite candle that engulfs the move.
/// `color(3rd)·100`. Lookback 8 (4-bar window + Near).
pub fn cdl_3linestrike(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = NEAR.avg_period + 3;
    let n = c.len();
    let Some(mut out) = candle_output(n, lb) else {
        return vec![f64::NAN; n];
    };
    // TA-Lib carries two Near totals for the two prior opens; avoiding `each_bar_avg_n`
    // removes a per-bar history shift and keeps this four-bar pattern on one tight loop.
    let mut near3 = 0.0;
    let mut near2 = 0.0;
    let mut trailing = lb - NEAR.avg_period;
    for j in trailing..lb {
        near3 += range(NEAR, o, h, l, c, j - 3);
        near2 += range(NEAR, o, h, l, c, j - 2);
    }
    let near_scale = NEAR.factor / NEAR.avg_period as f64;
    for i in lb..n {
        let c3_white = c[i - 3] >= o[i - 3];
        let c2_white = c[i - 2] >= o[i - 2];
        let c1_white = c[i - 1] >= o[i - 1];
        let c0_white = c[i] >= o[i];
        let same_run = c3_white == c2_white && c2_white == c1_white && c0_white != c1_white;
        out[i] = if same_run
            && ((c1_white
                && c[i - 1] > c[i - 2]
                && c[i - 2] > c[i - 3]
                && o[i] > c[i - 1]
                && c[i] < o[i - 3])
                || (!c1_white
                    && c[i - 1] < c[i - 2]
                    && c[i - 2] < c[i - 3]
                    && o[i] < c[i - 1]
                    && c[i] > o[i - 3]))
        {
            // The NEAR averages are only needed after the sparse color/trend gates
            // pass; delaying them keeps the common non-pattern row on cheap tests.
            let a3 = near3 * near_scale;
            let a2 = near2 * near_scale;
            let body3_lo = o[i - 3].min(c[i - 3]);
            let body3_hi = o[i - 3].max(c[i - 3]);
            let body2_lo = o[i - 2].min(c[i - 2]);
            let body2_hi = o[i - 2].max(c[i - 2]);
            if o[i - 2] >= body3_lo - a3
                && o[i - 2] <= body3_hi + a3
                && o[i - 1] >= body2_lo - a2
                && o[i - 1] <= body2_hi + a2
            {
                if c1_white {
                    100.0
                } else {
                    -100.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        near3 += range(NEAR, o, h, l, c, i - 3) - range(NEAR, o, h, l, c, trailing - 3);
        near2 += range(NEAR, o, h, l, c, i - 2) - range(NEAR, o, h, l, c, trailing - 2);
        trailing += 1;
    }
    out
}

/// Breakaway (TA-Lib CDLBREAKAWAY): a long body, a gap, two more in the same direction,
/// then a 5th opposite candle closing back into the gap. `color(i)·100`. Lookback 14.
pub fn cdl_breakaway(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_LONG.avg_period + 4;
    let n = c.len();
    let Some(mut out) = candle_output(n, lb) else {
        return vec![f64::NAN; n];
    };
    // TA-Lib only needs BodyLong at i-4; a single rolling total is cheaper than the
    // generic five-slot average history this rare pattern does not otherwise use.
    let mut total = 0.0;
    let mut trailing = lb - BODY_LONG.avg_period;
    for j in trailing..lb {
        total += range(BODY_LONG, o, h, l, c, j - 4);
    }
    let body_long_scale = BODY_LONG.factor / BODY_LONG.avg_period as f64;
    for i in lb..n {
        let first_white = c[i - 4] >= o[i - 4];
        let same_direction = (c[i - 3] >= o[i - 3]) == first_white
            && (c[i - 1] >= o[i - 1]) == first_white
            && (c[i] >= o[i]) != first_white;
        out[i] = if same_direction && realbody(o, c, i - 4) > total * body_long_scale {
            if !first_white
                && realbody_gap_down(o, c, i - 3, i - 4)
                && h[i - 2] < h[i - 3]
                && l[i - 2] < l[i - 3]
                && h[i - 1] < h[i - 2]
                && l[i - 1] < l[i - 2]
                && c[i] > o[i - 3]
                && c[i] < c[i - 4]
            {
                100.0
            } else if first_white
                && realbody_gap_up(o, c, i - 3, i - 4)
                && h[i - 2] > h[i - 3]
                && l[i - 2] > l[i - 3]
                && h[i - 1] > h[i - 2]
                && l[i - 1] > l[i - 2]
                && c[i] < o[i - 3]
                && c[i] > c[i - 4]
            {
                -100.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        total += range(BODY_LONG, o, h, l, c, i - 4) - range(BODY_LONG, o, h, l, c, trailing - 4);
        trailing += 1;
    }
    out
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
    each_bar_avg_n::<1, 4>([SHADOW_VERY_SHORT], lb, o, h, l, c, |i, hist| {
        // SHADOW_VERY_SHORT average at bar `i-lag`.
        let marubozu = |bar: usize, lag: usize| {
            uppershadow(o, h, c, bar) < hist[lag][0] && lowershadow(o, l, c, bar) < hist[lag][0]
        };
        if color(o, c, i - 3) < 0.0
            && color(o, c, i - 2) < 0.0
            && color(o, c, i - 1) < 0.0
            && color(o, c, i) < 0.0
            && marubozu(i - 3, 3)
            && marubozu(i - 2, 2)
            && realbody_gap_down(o, c, i - 1, i - 2)
            && uppershadow(o, h, c, i - 1) > hist[1][0] // SHADOW_VERY_SHORT at i-1
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

/// Mat Hold (TA-Lib CDLMATHOLD): a long white, three small holding candles (penetrating
/// the 1st body by less than `penetration`, default 0.5), then a white breakout closing
/// above the reaction highs — bullish continuation `100`. Lookback 14.
pub fn cdl_mathold(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 4;
    each_bar_avg_n::<2, 5>([BODY_LONG, BODY_SHORT], lb, o, h, l, c, |i, hist| {
        let floor = c[i - 4] - realbody(o, c, i - 4) * penetration;
        if realbody(o, c, i - 4) > hist[4][0] // BODY_LONG at i-4
            && realbody(o, c, i - 3) < hist[3][1] // BODY_SHORT at i-3
            && realbody(o, c, i - 2) < hist[2][1] // BODY_SHORT at i-2
            && realbody(o, c, i - 1) < hist[1][1] // BODY_SHORT at i-1
            && color(o, c, i - 4) > 0.0
            && color(o, c, i - 3) < 0.0
            && color(o, c, i) > 0.0
            && realbody_gap_up(o, c, i - 3, i - 4)
            && o[i - 2].min(c[i - 2]) < c[i - 4]
            && o[i - 1].min(c[i - 1]) < c[i - 4]
            && o[i - 2].min(c[i - 2]) > floor
            && o[i - 1].min(c[i - 1]) > floor
            && o[i - 2].max(c[i - 2]) < o[i - 3]
            && o[i - 1].max(c[i - 1]) < o[i - 2].max(c[i - 2])
            && o[i] > c[i - 1]
            && c[i] > h[i - 3].max(h[i - 2]).max(h[i - 1])
        {
            100.0
        } else {
            0.0
        }
    })
}

/// Rising/Falling Three Methods (TA-Lib CDLRISEFALL3METHODS): a long body, three small
/// counter-trend bodies holding within its range, then a long body resuming the trend.
/// `color(1st)·100`. Lookback 14. (The `*color` multiplier handles both directions.)
pub fn cdl_risefall3methods(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 4;
    each_bar_avg_n::<2, 5>([BODY_LONG, BODY_SHORT], lb, o, h, l, c, |i, hist| {
        let dir = color(o, c, i - 4);
        let within = |k: usize| o[k].min(c[k]) < h[i - 4] && o[k].max(c[k]) > l[i - 4];
        if realbody(o, c, i - 4) > hist[4][0] // BODY_LONG at i-4
            && realbody(o, c, i - 3) < hist[3][1] // BODY_SHORT at i-3
            && realbody(o, c, i - 2) < hist[2][1] // BODY_SHORT at i-2
            && realbody(o, c, i - 1) < hist[1][1] // BODY_SHORT at i-1
            && realbody(o, c, i) > hist[0][0] // BODY_LONG at i
            && dir == -color(o, c, i - 3)
            && color(o, c, i - 3) == color(o, c, i - 2)
            && color(o, c, i - 2) == color(o, c, i - 1)
            && color(o, c, i - 1) == -color(o, c, i)
            && within(i - 3)
            && within(i - 2)
            && within(i - 1)
            && c[i - 2] * dir < c[i - 3] * dir
            && c[i - 1] * dir < c[i - 2] * dir
            && o[i] * dir > c[i - 1] * dir
            && c[i] * dir > c[i - 4] * dir
        {
            100.0 * dir
        } else {
            0.0
        }
    })
}

/// Upside/Downside Gap Three Methods (TA-Lib CDLXSIDEGAP3METHODS): two same-colour bodies
/// with a gap, then an opposite candle filling the gap (opening in the 2nd body, closing
/// in the 1st). `color(1st)·100`. Lookback 2.
pub fn cdl_xsidegap3methods(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let _ = (h, l);
    each_bar(c.len(), 2, |i| {
        let up = color(o, c, i - 2) > 0.0 && realbody_gap_up(o, c, i - 1, i - 2);
        let down = color(o, c, i - 2) < 0.0 && realbody_gap_down(o, c, i - 1, i - 2);
        if color(o, c, i - 2) == color(o, c, i - 1)
            && color(o, c, i - 1) == -color(o, c, i)
            && o[i] < o[i - 1].max(c[i - 1])
            && o[i] > o[i - 1].min(c[i - 1])
            && c[i] < o[i - 2].max(c[i - 2])
            && c[i] > o[i - 2].min(c[i - 2])
            && (up || down)
        {
            color(o, c, i - 2) * 100.0
        } else {
            0.0
        }
    })
}

/// Hikkake (TA-Lib CDLHIKKAKE): an inside bar then a breakout (`±100`), optionally
/// confirmed within the next 3 bars by a close beyond the inside bar's extreme (`±200`).
/// Stateful (carries the pending setup across bars). Lookback 5.
pub fn cdl_hikkake(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let _ = o;
    let n = c.len();
    let Some(mut out) = candle_output(n, 5) else {
        return vec![f64::NAN; n];
    };
    // Match TA-Lib's minimal state machine: pending setup index + signed result.
    // Confirmation rereads the inside bar's extreme only on the rare confirmation path.
    let (mut pattern_idx, mut pattern_result) = (0usize, 0_i32);
    for i in 2..5 {
        if h[i - 1] < h[i - 2] && l[i - 1] > l[i - 2] {
            let lower_break = h[i] < h[i - 1] && l[i] < l[i - 1];
            let higher_break = h[i] > h[i - 1] && l[i] > l[i - 1];
            if lower_break || higher_break {
                pattern_result = if lower_break { 100 } else { -100 };
                pattern_idx = i;
                continue;
            }
        }
        if i <= pattern_idx + 3
            && ((pattern_result > 0 && c[i] > h[pattern_idx - 1])
                || (pattern_result < 0 && c[i] < l[pattern_idx - 1]))
        {
            pattern_idx = 0;
        }
    }
    for i in 5..n {
        let value = if h[i - 1] < h[i - 2] && l[i - 1] > l[i - 2] {
            let lower_break = h[i] < h[i - 1] && l[i] < l[i - 1];
            let higher_break = h[i] > h[i - 1] && l[i] > l[i - 1];
            if lower_break || higher_break {
                let r = if lower_break { 100 } else { -100 };
                pattern_result = r;
                pattern_idx = i;
                r as f64
            } else if i <= pattern_idx + 3
                && ((pattern_result > 0 && c[i] > h[pattern_idx - 1])
                    || (pattern_result < 0 && c[i] < l[pattern_idx - 1]))
            {
                let confirmed = pattern_result + if pattern_result > 0 { 100 } else { -100 };
                pattern_idx = 0;
                confirmed as f64
            } else {
                0.0
            }
        } else if i <= pattern_idx + 3
            && ((pattern_result > 0 && c[i] > h[pattern_idx - 1])
                || (pattern_result < 0 && c[i] < l[pattern_idx - 1]))
        {
            let confirmed = pattern_result + if pattern_result > 0 { 100 } else { -100 };
            pattern_idx = 0;
            confirmed as f64
        } else {
            0.0
        };
        out[i] = value;
    }
    out
}

/// Modified Hikkake (TA-Lib CDLHIKKAKEMOD): a stricter hikkake — two nested inside bars
/// with the middle bar closing near its extreme — then the breakout (`±100`) and optional
/// 3-bar confirmation (`±200`). Stateful. Lookback 10.
pub fn cdl_hikkakemod(o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let n = c.len();
    let mut out = vec![f64::NAN; n];
    let start = NEAR.avg_period.max(1) + 5;
    if n <= start {
        return out;
    }
    // Keep TA-Lib's single rolling NEAR total instead of materialising a full
    // average series. The pattern is stateful, so the setup test uses the current
    // total first, then the loop slides it for the next bar.
    let mut near_total = 0.0;
    let mut near_trailing = start - 3 - NEAR.avg_period;
    let mut j = near_trailing;
    while j < start - 3 {
        near_total += range(NEAR, o, h, l, c, j - 2);
        j += 1;
    }
    let setup = |i: usize, near_total: f64| -> Option<f64> {
        let near = candle_average_from_total(NEAR, near_total);
        if h[i - 2] < h[i - 3]
            && l[i - 2] > l[i - 3]
            && h[i - 1] < h[i - 2]
            && l[i - 1] > l[i - 2]
            && ((h[i] < h[i - 1] && l[i] < l[i - 1] && c[i - 2] <= l[i - 2] + near)
                || (h[i] > h[i - 1] && l[i] > l[i - 1] && c[i - 2] >= h[i - 2] - near))
        {
            Some(if h[i] < h[i - 1] { 100.0 } else { -100.0 })
        } else {
            None
        }
    };
    let confirm = |i: usize, idx: usize, res: f64| {
        idx != 0
            && i <= idx + 3
            && ((res > 0.0 && c[i] > h[idx - 1]) || (res < 0.0 && c[i] < l[idx - 1]))
    };
    let (mut idx, mut res) = (0usize, 0.0_f64);
    for i in (start - 3)..n {
        let emit = i >= start;
        if let Some(r) = setup(i, near_total) {
            res = r;
            idx = i;
            if emit {
                out[i] = r;
            }
        } else if confirm(i, idx, res) {
            if emit {
                out[i] = res + res.signum() * 100.0;
            }
            idx = 0;
        } else if emit {
            out[i] = 0.0;
        }
        near_total += range(NEAR, o, h, l, c, i - 2) - range(NEAR, o, h, l, c, near_trailing - 2);
        near_trailing += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hikkake initial scan (`for i in 2..6`) only reaches its `i >= 5` output
    /// branches for a setup or confirmation landing exactly on bar 5 — which the long
    /// fuzz series (whose hikkakes occur later, in the continuation loop) never does.
    /// Both six-bar cases are derived directly from the setup/confirm geometry; full
    /// correctness vs TA-Lib is covered by the Python parity suite.
    #[test]
    fn hikkake_initial_loop_index_5_branches() {
        let o = [10.0; 6];
        let c = [10.0; 6];
        // Setup detected at bar 5: inside bar at 3-4, then a bullish breakout at 5.
        let h_a = [20.0, 20.0, 20.0, 18.0, 16.0, 15.0];
        let l_a = [5.0, 5.0, 5.0, 8.0, 10.0, 7.0];
        assert_eq!(cdl_hikkake(&o, &h_a, &l_a, &c)[5], 100.0);
        // Setup at bar 3, then confirmed at bar 5 -> ±200 and the index resets.
        let c_b = [10.0, 10.0, 10.0, 10.0, 10.0, 20.0];
        let h_b = [20.0, 20.0, 18.0, 16.0, 17.0, 21.0];
        let l_b = [5.0, 5.0, 8.0, 6.0, 9.0, 19.0];
        assert_eq!(cdl_hikkake(&o, &h_b, &l_b, &c_b)[5], 200.0);
    }
}
