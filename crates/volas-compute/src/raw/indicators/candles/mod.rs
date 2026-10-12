// ---------------------------------------------------------------------------
// Candlestick patterns (TA-Lib CDL*)
// ---------------------------------------------------------------------------
//
// A cohesive sub-domain of `indicators`: candlestick-pattern recognition. This module
// owns the shared candle-settings framework (TA-Lib's `TA_CANDLE*`); the patterns
// themselves live in bar-count submodules (`one_bar`, `two_bar`, …) and are re-exported.
//
// Each pattern tests a candle's geometry against adaptive thresholds derived from a
// rolling average of a chosen "range" (real body / high-low / shadows) over the prior
// `avg_period` bars, scaled by a `factor`. Output is f64: `+100` bullish, `-100`
// bearish, `0` no pattern; the warm-up is NaN (volas's uniform convention — TA-Lib
// fills its integer outputs with 0, which is indistinguishable from "no pattern").
// Parity holds on the valid region. A bar whose open / high / low / close is missing or
// infinite is a missing candle: every pattern is NaN on the bars whose lookback holds it
// and exactly its clean value elsewhere (`each_bar`), and each average is its own
// window's sum (`candle_average`), never a running total that would carry a bad bar on.

// Each pattern's warm-up is `max()` over the avg_periods of the settings it uses,
// expressing "the longest required lookback". That one constant currently dominates
// is incidental — keep the `max` so a future settings change stays correct.
#![allow(clippy::unnecessary_min_or_max)]

mod multi_bar;
mod one_bar;
mod three_bar;
mod two_bar;
pub use multi_bar::*;
pub use one_bar::*;
pub use three_bar::*;
pub use two_bar::*;

use crate::missing;
use crate::raw::kernels;
use ndarray::ArrayView1;

/// An OHLC candle recogniser: `(open, high, low, close) -> signal`.
type PlainFn = fn(&[f64], &[f64], &[f64], &[f64], usize) -> Vec<f64>;
/// An OHLC recogniser that also takes TA-Lib's `penetration` ratio.
type PenetrationFn = fn(&[f64], &[f64], &[f64], &[f64], f64, usize) -> Vec<f64>;

/// A candlestick-pattern recogniser. Most take only `(open, high, low, close)`; a few
/// also take TA-Lib's `penetration` ratio (with a pattern-specific default).
pub enum CandlePattern {
    /// `(open, high, low, close)`.
    Plain(PlainFn),
    /// `(open, high, low, close, penetration)`, plus the TA-Lib default penetration.
    Penetration {
        f: PenetrationFn,
        default: f64,
    },
}

/// Resolve a pattern by its (post-`CDL`, lower-case) name to its recogniser and
/// lookback. The single source of truth for which `style.<pattern>` / `cdl.<pattern>`
/// names exist — the directive layer (spec/exec/lookback) queries this, so adding a
/// pattern is one entry here plus its function.
pub fn candle_pattern(name: &str) -> Option<(CandlePattern, usize)> {
    use CandlePattern::{Penetration, Plain};
    // Penetration patterns: darkcloudcover defaults 0.5, the star family 0.3.
    let pen = |f, default| Penetration { f, default };
    let entry = match name {
        // one-bar
        "doji" => (Plain(cdl_doji as _), 10),
        "marubozu" => (Plain(cdl_marubozu as _), 10),
        "closingmarubozu" => (Plain(cdl_closingmarubozu as _), 10),
        "longline" => (Plain(cdl_longline as _), 10),
        "shortline" => (Plain(cdl_shortline as _), 10),
        "highwave" => (Plain(cdl_highwave as _), 10),
        "spinningtop" => (Plain(cdl_spinningtop as _), 10),
        "dragonflydoji" => (Plain(cdl_dragonflydoji as _), 10),
        "gravestonedoji" => (Plain(cdl_gravestonedoji as _), 10),
        "longleggeddoji" => (Plain(cdl_longleggeddoji as _), 10),
        "rickshawman" => (Plain(cdl_rickshawman as _), 10),
        "belthold" => (Plain(cdl_belthold as _), 10),
        "hammer" => (Plain(cdl_hammer as _), 11),
        "hangingman" => (Plain(cdl_hangingman as _), 11),
        "invertedhammer" => (Plain(cdl_invertedhammer as _), 11),
        "shootingstar" => (Plain(cdl_shootingstar as _), 11),
        "takuri" => (Plain(cdl_takuri as _), 10),
        // two-bar
        "engulfing" => (Plain(cdl_engulfing as _), 2),
        "harami" => (Plain(cdl_harami as _), 11),
        "haramicross" => (Plain(cdl_haramicross as _), 11),
        "piercing" => (Plain(cdl_piercing as _), 11),
        "darkcloudcover" => (pen(cdl_darkcloudcover, 0.5), 11),
        "dojistar" => (Plain(cdl_dojistar as _), 11),
        "homingpigeon" => (Plain(cdl_homingpigeon as _), 11),
        "matchinglow" => (Plain(cdl_matchinglow as _), 6),
        "inneck" => (Plain(cdl_inneck as _), 11),
        "onneck" => (Plain(cdl_onneck as _), 11),
        "thrusting" => (Plain(cdl_thrusting as _), 11),
        "kicking" => (Plain(cdl_kicking as _), 11),
        "kickingbylength" => (Plain(cdl_kickingbylength as _), 11),
        "separatinglines" => (Plain(cdl_separatinglines as _), 11),
        "counterattack" => (Plain(cdl_counterattack as _), 11),
        // three-bar
        "morningstar" => (pen(cdl_morningstar, 0.3), 12),
        "eveningstar" => (pen(cdl_eveningstar, 0.3), 12),
        "3inside" => (Plain(cdl_3inside as _), 12),
        "3outside" => (Plain(cdl_3outside as _), 3),
        "3whitesoldiers" => (Plain(cdl_3whitesoldiers as _), 12),
        "3blackcrows" => (Plain(cdl_3blackcrows as _), 13),
        "morningdojistar" => (pen(cdl_morningdojistar, 0.3), 12),
        "eveningdojistar" => (pen(cdl_eveningdojistar, 0.3), 12),
        "abandonedbaby" => (pen(cdl_abandonedbaby, 0.3), 12),
        "2crows" => (Plain(cdl_2crows as _), 12),
        "upsidegap2crows" => (Plain(cdl_upsidegap2crows as _), 12),
        "advanceblock" => (Plain(cdl_advanceblock as _), 12),
        "stalledpattern" => (Plain(cdl_stalledpattern as _), 12),
        "identical3crows" => (Plain(cdl_identical3crows as _), 12),
        "sticksandwich" => (Plain(cdl_sticksandwich as _), 7),
        "tristar" => (Plain(cdl_tristar as _), 12),
        "unique3river" => (Plain(cdl_unique3river as _), 12),
        "gapsidesidewhite" => (Plain(cdl_gapsidesidewhite as _), 7),
        "tasukigap" => (Plain(cdl_tasukigap as _), 7),
        "3starsinsouth" => (Plain(cdl_3starsinsouth as _), 12),
        // four / five-bar
        "3linestrike" => (Plain(cdl_3linestrike as _), 8),
        "breakaway" => (Plain(cdl_breakaway as _), 14),
        "ladderbottom" => (Plain(cdl_ladderbottom as _), 14),
        "concealbabyswall" => (Plain(cdl_concealbabyswall as _), 13),
        "mathold" => (pen(cdl_mathold, 0.5), 14),
        "risefall3methods" => (Plain(cdl_risefall3methods as _), 14),
        "xsidegap3methods" => (Plain(cdl_xsidegap3methods as _), 2),
        "hikkake" => (Plain(cdl_hikkake as _), 5),
        "hikkakemod" => (Plain(cdl_hikkakemod as _), 10),
        _ => return None,
    };
    Some(entry)
}

/// Every candlestick-pattern name (the `style.<pattern>` / `cdl.<pattern>`
/// vocabulary), as an enumerable list — the single source the indicator-count
/// script derives the pattern total from. A test keeps it in lock-step with
/// `candle_pattern`, so a pattern can never be in one and not the other.
pub const CANDLE_PATTERNS: &[&str] = &[
    "doji", "marubozu", "closingmarubozu", "longline", "shortline", "highwave",
    "spinningtop", "dragonflydoji", "gravestonedoji", "longleggeddoji", "rickshawman",
    "belthold", "hammer", "hangingman", "invertedhammer", "shootingstar", "takuri",
    "engulfing", "harami", "haramicross", "piercing", "darkcloudcover", "dojistar",
    "homingpigeon", "matchinglow", "inneck", "onneck", "thrusting", "kicking",
    "kickingbylength", "separatinglines", "counterattack", "morningstar", "eveningstar",
    "3inside", "3outside", "3whitesoldiers", "3blackcrows", "morningdojistar",
    "eveningdojistar", "abandonedbaby", "2crows", "upsidegap2crows", "advanceblock",
    "stalledpattern", "identical3crows", "sticksandwich", "tristar", "unique3river",
    "gapsidesidewhite", "tasukigap", "3starsinsouth", "3linestrike", "breakaway",
    "ladderbottom", "concealbabyswall", "mathold", "risefall3methods", "xsidegap3methods",
    "hikkake", "hikkakemod",
];

/// The range a candle-setting measures. The full set mirrors TA-Lib's settings table;
/// a variant is only *constructed* once a pattern needing it lands, so allow the lint.
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum RangeType {
    RealBody,
    HighLow,
    Shadows,
}

/// One candle-setting: which range, the averaging period (0 = the bar itself), and the
/// scaling factor. Values are TA-Lib's defaults (ta_global.c). Visible to the pattern
/// submodules (descendants) without `pub`.
#[derive(Clone, Copy)]
struct Setting {
    range: RangeType,
    avg_period: usize,
    factor: f64,
}

use RangeType::{HighLow, RealBody, Shadows};
const BODY_LONG: Setting = Setting {
    range: RealBody,
    avg_period: 10,
    factor: 1.0,
};
const BODY_VERY_LONG: Setting = Setting {
    range: RealBody,
    avg_period: 10,
    factor: 3.0,
};
const BODY_SHORT: Setting = Setting {
    range: RealBody,
    avg_period: 10,
    factor: 1.0,
};
const BODY_DOJI: Setting = Setting {
    range: HighLow,
    avg_period: 10,
    factor: 0.1,
};
const SHADOW_LONG: Setting = Setting {
    range: RealBody,
    avg_period: 0,
    factor: 1.0,
};
const SHADOW_VERY_LONG: Setting = Setting {
    range: RealBody,
    avg_period: 0,
    factor: 2.0,
};
const SHADOW_SHORT: Setting = Setting {
    range: Shadows,
    avg_period: 10,
    factor: 1.0,
};
const SHADOW_VERY_SHORT: Setting = Setting {
    range: HighLow,
    avg_period: 10,
    factor: 0.1,
};
const NEAR: Setting = Setting {
    range: HighLow,
    avg_period: 5,
    factor: 0.2,
};
const FAR: Setting = Setting {
    range: HighLow,
    avg_period: 5,
    factor: 0.6,
};
const EQUAL: Setting = Setting {
    range: HighLow,
    avg_period: 5,
    factor: 0.05,
};

// Settings not yet consumed by a landed pattern (shrinks to nothing as patterns arrive).
#[allow(dead_code)]
const _UNUSED_SETTINGS: [Setting; 1] = [BODY_VERY_LONG];

#[inline]
fn realbody(o: &[f64], c: &[f64], i: usize) -> f64 {
    (c[i] - o[i]).abs()
}
#[inline]
fn uppershadow(o: &[f64], h: &[f64], c: &[f64], i: usize) -> f64 {
    h[i] - o[i].max(c[i])
}
#[inline]
fn lowershadow(o: &[f64], l: &[f64], c: &[f64], i: usize) -> f64 {
    o[i].min(c[i]) - l[i]
}
/// `+1.0` white (close ≥ open), `-1.0` black.
#[inline]
fn color(o: &[f64], c: &[f64], i: usize) -> f64 {
    if c[i] >= o[i] {
        1.0
    } else {
        -1.0
    }
}

/// Real-body gap up between bar `i` and an earlier bar `j` (TA-Lib `TA_REALBODYGAPUP`):
/// the whole real body of `i` sits above that of `j`.
#[inline]
fn realbody_gap_up(o: &[f64], c: &[f64], i: usize, j: usize) -> bool {
    o[i].min(c[i]) > o[j].max(c[j])
}
/// Real-body gap down (TA-Lib `TA_REALBODYGAPDOWN`).
#[inline]
fn realbody_gap_down(o: &[f64], c: &[f64], i: usize, j: usize) -> bool {
    o[i].max(c[i]) < o[j].min(c[j])
}
/// Candle (high-low) gap up between bar `i` and an earlier bar `j` (TA-Lib
/// `TA_CANDLEGAPUP`): bar `i`'s low is above bar `j`'s high.
#[inline]
fn candle_gap_up(h: &[f64], l: &[f64], i: usize, j: usize) -> bool {
    l[i] > h[j]
}
/// Candle gap down (TA-Lib `TA_CANDLEGAPDOWN`).
#[inline]
fn candle_gap_down(h: &[f64], l: &[f64], i: usize, j: usize) -> bool {
    h[i] < l[j]
}

/// The divisor of a setting's average: `2` for the Shadows range (two shadows per bar),
/// else `1` (TA-Lib's `TA_CANDLERANGETYPE == Shadows ? 2.0 : 1.0`).
#[inline(always)]
fn divisor(s: Setting) -> f64 {
    if matches!(s.range, RangeType::Shadows) { 2.0 } else { 1.0 }
}

/// The setting's range at every bar (TA-Lib `TA_CANDLERANGE`): the real body, the
/// high − low span, or both shadows, with the range type resolved once.
fn ranges(s: Setting, o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let n = c.len();
    match s.range {
        RangeType::RealBody => (0..n).map(|i| (c[i] - o[i]).abs()).collect(),
        RangeType::HighLow => (0..n).map(|i| h[i] - l[i]).collect(),
        RangeType::Shadows => (0..n).map(|i| uppershadow(o, h, c, i) + lowershadow(o, l, c, i)).collect(),
    }
}

/// TA-Lib's `TA_CANDLEAVERAGE` of setting `s` at every bar (NaN before the first full
/// window): `factor · (Σ range over the prior `avg_period` bars [i − avg_period, i − 1])
/// / avg_period / div`, or `factor · range(i) / div` when `avg_period == 0`. Row 0 is at
/// absolute position `origin` (the frame's). The one definition of a candle average: the
/// sum slides like TA-Lib's running total (O(1) per bar), but it is the sliding mean every
/// moving average uses ([`kernels::sma`]: an exact sum re-anchored at absolute positions,
/// window-local — a non-finite range reaches only the windows holding it) — so a bad
/// candle never outlives the windows holding it, and a recomputed window (the live
/// refresh) gives the full computation's bits. On near-tie bars this follows the exact
/// arithmetic where TA-Lib's plain running total has drifted.
fn candle_average_series(s: Setting, o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let n = c.len();
    // `/ div` (1 or 2, a power of two) is exactly `* (1 / div)`.
    let inv_div = 1.0 / divisor(s);
    let r = ranges(s, o, h, l, c);
    let p = s.avg_period;
    if p == 0 {
        return r.into_iter().map(|x| s.factor * x * inv_div).collect();
    }
    // The average at bar `i` is the mean range of the window ending at bar `i − 1`: the
    // sliding mean every moving average uses (`kernels::sma`), one row behind.
    let mean = kernels::sma(ArrayView1::from(&r), p, origin);
    let mut out = Vec::with_capacity(n);
    out.push(f64::NAN);
    out.extend(mean.iter().take(n.saturating_sub(1)).map(|m| s.factor * m * inv_div));
    out
}

/// Build a pattern column: NaN before `lookback`, then `f(i)` (0 / ±100 / ±80 / ±200)
/// per bar — and NaN while the bars a pattern reads, `[i − lookback, i]`, hold a missing
/// candle (any open / high / low / close that is not finite): a pattern has no value
/// over a bar it cannot see, and is exactly what clean data gives once that bar has
/// left its reach. The one builder every pattern goes through.
#[inline]
fn each_bar(
    lookback: usize,
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    mut f: impl FnMut(usize) -> f64,
) -> Vec<f64> {
    let n = c.len();
    let warm = lookback.min(n);
    // Single-write (D2): NaN warm-up then one write per valid bar, no prefill memset
    // of the valid region. `build_f64` keeps it safe (debug poison + write-all assert).
    let mut out = crate::buf::build_f64(n, |out| {
        for slot in &mut out[..warm] {
            slot.write(f64::NAN);
        }
        #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
        for i in warm..n {
            out[i].write(f(i));
        }
    });
    for x in [o, h, l, c] {
        missing::blank_windows_where(&mut out, x, lookback + 1, |v| !v.is_finite());
    }
    out
}

/// [`each_bar`] for a pattern that reads the averages of `settings` at the bar itself:
/// `f(i, avgs)`, `avgs[k]` = the average of `settings[k]` at `i` ([`candle_average_series`]).
#[inline]
#[allow(clippy::too_many_arguments)] // the four prices, their origin, the settings, the reach, the test
fn each_bar_avg<const K: usize>(
    settings: [Setting; K],
    lookback: usize,
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    origin: usize,
    f: impl Fn(usize, &[f64; K]) -> f64,
) -> Vec<f64> {
    let avg = settings.map(|s| candle_average_series(s, o, h, l, c, origin));
    each_bar(lookback, o, h, l, c, |i| f(i, &std::array::from_fn(|k| avg[k][i])))
}

/// [`each_bar_avg`] for a two-bar pattern: `f(i, cur, prev)` with the averages at `i`
/// and at `i − 1`.
#[inline]
#[allow(clippy::too_many_arguments)] // the four prices, their origin, the settings, the reach, the test
fn each_bar_avg2<const K: usize>(
    settings: [Setting; K],
    lookback: usize,
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    origin: usize,
    f: impl Fn(usize, &[f64; K], &[f64; K]) -> f64,
) -> Vec<f64> {
    let avg = settings.map(|s| candle_average_series(s, o, h, l, c, origin));
    each_bar(lookback, o, h, l, c, |i| {
        f(i, &std::array::from_fn(|k| avg[k][i]), &std::array::from_fn(|k| avg[k][i - 1]))
    })
}

/// [`each_bar_avg`] for a pattern that reads averages up to `L − 1` bars back:
/// `f(i, hist)`, `hist[lag][k]` = the average of `settings[k]` at `i − lag`.
#[inline]
#[allow(clippy::too_many_arguments)] // the four prices, their origin, the settings, the reach, the test
fn each_bar_avg_n<const K: usize, const L: usize>(
    settings: [Setting; K],
    lookback: usize,
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    origin: usize,
    f: impl Fn(usize, &[[f64; K]; L]) -> f64,
) -> Vec<f64> {
    let avg = settings.map(|s| candle_average_series(s, o, h, l, c, origin));
    each_bar(lookback, o, h, l, c, |i| {
        f(i, &std::array::from_fn(|lag| std::array::from_fn(|k| avg[k][i - lag])))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CANDLE_PATTERNS` must stay in lock-step with `candle_pattern`: every listed
    /// name resolves (so the indicator count never lists a phantom pattern), and the
    /// list is duplicate-free. The reverse (every `candle_pattern` arm is listed) is
    /// guarded by the count being pinned in the test suite.
    #[test]
    fn candle_patterns_list_matches_registry() {
        for name in CANDLE_PATTERNS {
            assert!(candle_pattern(name).is_some(), "CANDLE_PATTERNS has a phantom {name:?}");
        }
        let mut sorted = CANDLE_PATTERNS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), CANDLE_PATTERNS.len(), "CANDLE_PATTERNS has duplicates");
    }

    fn ohlc(n: usize) -> [Vec<f64>; 4] {
        let c: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.37).sin() * 3.0 + 0.1 * i as f64).collect();
        let o: Vec<f64> = c.iter().enumerate().map(|(i, x)| x + (i as f64 * 0.71).cos()).collect();
        let h = o.iter().zip(&c).enumerate().map(|(i, (a, b))| a.max(*b) + (i as f64 * 0.3).sin().abs()).collect();
        let l = o.iter().zip(&c).enumerate().map(|(i, (a, b))| a.min(*b) - (i as f64 * 0.2).cos().abs()).collect();
        [o, h, l, c]
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| if x.is_nan() { u64::MAX } else { x.to_bits() }).collect()
    }

    /// Every setting's average is TA-Lib's: the mean range over the prior `avg_period`
    /// bars (to the exact sum's last bit), ÷2 for Shadows, the bar's own range for
    /// `avg_period == 0`; NaN before the first full window, nothing on a short series.
    /// And a recomputed tail (the live refresh: a slice keeps its rows' absolute
    /// positions) gives the full series' bits once its first anchor has passed.
    #[test]
    fn the_average_series_is_the_window_mean_and_a_tail_recompute_matches_it() {
        for n in [3, 10, 11, 14, 60] {
            let [o, h, l, c] = ohlc(n);
            for s in [BODY_LONG, BODY_DOJI, SHADOW_SHORT, SHADOW_LONG, NEAR, BODY_VERY_LONG] {
                let r = ranges(s, &o, &h, &l, &c);
                let series = candle_average_series(s, &o, &h, &l, &c, 0);
                for (i, v) in series.iter().enumerate() {
                    if i < s.avg_period {
                        assert!(v.is_nan());
                        continue;
                    }
                    let base = if s.avg_period == 0 {
                        r[i]
                    } else {
                        r[i - s.avg_period..i].iter().sum::<f64>() / s.avg_period as f64
                    };
                    let want = s.factor * base / divisor(s);
                    assert!((v - want).abs() <= 1e-12 * want.abs(), "n {n} bar {i}: {v} vs {want}");
                }
                if n == 60 {
                    for start in [7, 13, 20] {
                        let tail = |x: &[f64]| x[start..].to_vec();
                        let part = candle_average_series(s, &tail(&o), &tail(&h), &tail(&l), &tail(&c), start);
                        let settled = start + 2 * s.avg_period.max(1) + 1;
                        assert_eq!(bits(&part[settled - start..]), bits(&series[settled..]), "start {start}");
                    }
                }
            }
        }
    }

    /// A missing or infinite candle makes a pattern NaN on exactly the bars whose reach
    /// holds it, the bars before are untouched, and once it has left (and the sums'
    /// next anchor has passed) every bar is what clean data gives — an average never
    /// carries it further. Every builder shape.
    #[test]
    fn a_missing_candle_reaches_only_the_bars_that_read_it() {
        let [o, h, l, c] = ohlc(80);
        let run = |o: &[f64], h: &[f64], l: &[f64], c: &[f64]| {
            [
                each_bar_avg([BODY_DOJI], 10, o, h, l, c, 0, |i, a| realbody(o, c, i) - a[0]),
                each_bar_avg2([BODY_LONG, SHADOW_SHORT], 11, o, h, l, c, 0, |_, a, b| a[0] + b[1]),
                each_bar_avg_n::<2, 3>([NEAR, SHADOW_LONG], 12, o, h, l, c, 0, |_, x| x[2][0] + x[1][1]),
                each_bar(5, o, h, l, c, |i| c[i] - o[i - 5]),
            ]
        };
        let clean = run(&o, &h, &l, &c);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for col in 0..4 {
                let mut x = [o.clone(), h.clone(), l.clone(), c.clone()];
                x[col][30] = bad;
                let got = run(&x[0], &x[1], &x[2], &x[3]);
                for ((g, cl), lb) in got.iter().zip(&clean).zip([10, 11, 12, 5]) {
                    for i in 0..80 {
                        if (30..=30 + lb).contains(&i) {
                            assert!(g[i].is_nan(), "bar {i} reads the bad candle");
                        } else if i < 30 || i > 30 + lb + 20 {
                            assert_eq!(bits(&g[i..=i]), bits(&cl[i..=i]), "bar {i}");
                        }
                    }
                }
            }
        }
    }
}
