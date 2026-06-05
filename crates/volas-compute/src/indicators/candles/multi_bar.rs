//! Four- and five-bar candlestick patterns.

use super::{
    candle_average, candle_average_series, color, each_bar, each_bar_avg_n, lowershadow, realbody,
    realbody_gap_down, realbody_gap_up, uppershadow, BODY_LONG, BODY_SHORT, NEAR, SHADOW_VERY_SHORT,
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
    each_bar_avg_n::<1, 4>([NEAR], lb, o, h, l, c, |i, hist| {
        let same3 =
            color(o, c, i - 3) == color(o, c, i - 2) && color(o, c, i - 2) == color(o, c, i - 1);
        let opens_ok = near_open(o, c, i - 2, i - 3, hist[3][0]) // NEAR at i-3
            && near_open(o, c, i - 1, i - 2, hist[2][0]); // NEAR at i-2
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
    each_bar_avg_n::<1, 5>([BODY_LONG], lb, o, h, l, c, |i, hist| {
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
        if realbody(o, c, i - 4) > hist[4][0] // BODY_LONG average at i-4
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

/// Mat Hold (TA-Lib CDLMATHOLD): a long white, three small holding candles (penetrating
/// the 1st body by less than `penetration`, default 0.5), then a white breakout closing
/// above the reaction highs — bullish continuation `100`. Lookback 14.
pub fn cdl_mathold(o: &[f64], h: &[f64], l: &[f64], c: &[f64], penetration: f64) -> Vec<f64> {
    let lb = BODY_SHORT.avg_period.max(BODY_LONG.avg_period) + 4;
    each_bar(c.len(), lb, |i| {
        let floor = c[i - 4] - realbody(o, c, i - 4) * penetration;
        if realbody(o, c, i - 4) > candle_average(BODY_LONG, o, h, l, c, i - 4)
            && realbody(o, c, i - 3) < candle_average(BODY_SHORT, o, h, l, c, i - 3)
            && realbody(o, c, i - 2) < candle_average(BODY_SHORT, o, h, l, c, i - 2)
            && realbody(o, c, i - 1) < candle_average(BODY_SHORT, o, h, l, c, i - 1)
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
    each_bar(c.len(), lb, |i| {
        let dir = color(o, c, i - 4);
        let within = |k: usize| o[k].min(c[k]) < h[i - 4] && o[k].max(c[k]) > l[i - 4];
        if realbody(o, c, i - 4) > candle_average(BODY_LONG, o, h, l, c, i - 4)
            && realbody(o, c, i - 3) < candle_average(BODY_SHORT, o, h, l, c, i - 3)
            && realbody(o, c, i - 2) < candle_average(BODY_SHORT, o, h, l, c, i - 2)
            && realbody(o, c, i - 1) < candle_average(BODY_SHORT, o, h, l, c, i - 1)
            && realbody(o, c, i) > candle_average(BODY_LONG, o, h, l, c, i)
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
    let mut out = vec![f64::NAN; n];
    if n < 6 {
        return out;
    }
    // Setup: bars i-2/i-1 form an inside bar, then bar i breaks out (bull = lower, bear
    // = higher). Returns the signed result, or None.
    let setup = |i: usize| -> Option<f64> {
        if h[i - 1] < h[i - 2]
            && l[i - 1] > l[i - 2]
            && ((h[i] < h[i - 1] && l[i] < l[i - 1]) || (h[i] > h[i - 1] && l[i] > l[i - 1]))
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
    for i in 2..6 {
        if let Some(r) = setup(i) {
            res = r;
            idx = i;
            if i >= 5 {
                out[i] = r;
            }
        } else if confirm(i, idx, res) {
            if i >= 5 {
                out[i] = res + res.signum() * 100.0;
            }
            idx = 0;
        } else if i >= 5 {
            out[i] = 0.0;
        }
    }
    for i in 6..n {
        if let Some(r) = setup(i) {
            res = r;
            idx = i;
            out[i] = r;
        } else if confirm(i, idx, res) {
            out[i] = res + res.signum() * 100.0;
            idx = 0;
        } else {
            out[i] = 0.0;
        }
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
    // Stateful loop, so precompute the NEAR average as an O(n) running-sum series and
    // read it at i-2 (instead of rescanning the window every setup test).
    let near_series = candle_average_series(NEAR, o, h, l, c);
    let setup = |i: usize| -> Option<f64> {
        let near = near_series[i - 2];
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
        if let Some(r) = setup(i) {
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
