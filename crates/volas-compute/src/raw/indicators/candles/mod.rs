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

mod entry;
pub use entry::*;
mod multi_bar;
mod one_bar;
mod three_bar;
mod two_bar;
use multi_bar::*;
use one_bar::*;
use three_bar::*;
use two_bar::*;

use crate::accum::Csum;
use crate::missing;
use crate::raw::kernels;

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
    candle_pattern_with::<false>(name)
}

/// Resolve a recognizer for OHLC inputs already admitted as finite.
pub fn candle_pattern_complete(name: &str) -> Option<(CandlePattern, usize)> {
    candle_pattern_with::<true>(name)
}

fn candle_pattern_with<const CLEAN: bool>(name: &str) -> Option<(CandlePattern, usize)> {
    use CandlePattern::{Penetration, Plain};
    // Penetration patterns: darkcloudcover defaults 0.5, the star family 0.3.
    let pen = |f, default| Penetration { f, default };
    let entry = match name {
        // one-bar
        "doji" => (Plain(if CLEAN { cdl_doji_raw as _ } else { cdl_doji as _ }), 10),
        "marubozu" => (Plain(if CLEAN { cdl_marubozu_raw as _ } else { cdl_marubozu as _ }), 10),
        "closingmarubozu" => (Plain(if CLEAN { cdl_closingmarubozu_raw as _ } else { cdl_closingmarubozu as _ }), 10),
        "longline" => (Plain(if CLEAN { cdl_longline_raw as _ } else { cdl_longline as _ }), 10),
        "shortline" => (Plain(if CLEAN { cdl_shortline_raw as _ } else { cdl_shortline as _ }), 10),
        "highwave" => (Plain(if CLEAN { cdl_highwave_raw as _ } else { cdl_highwave as _ }), 10),
        "spinningtop" => (Plain(if CLEAN { cdl_spinningtop_raw as _ } else { cdl_spinningtop as _ }), 10),
        "dragonflydoji" => (Plain(if CLEAN { cdl_dragonflydoji_raw as _ } else { cdl_dragonflydoji as _ }), 10),
        "gravestonedoji" => (Plain(if CLEAN { cdl_gravestonedoji_raw as _ } else { cdl_gravestonedoji as _ }), 10),
        "longleggeddoji" => (Plain(if CLEAN { cdl_longleggeddoji_raw as _ } else { cdl_longleggeddoji as _ }), 10),
        "rickshawman" => (Plain(if CLEAN { cdl_rickshawman_raw as _ } else { cdl_rickshawman as _ }), 10),
        "belthold" => (Plain(if CLEAN { cdl_belthold_raw as _ } else { cdl_belthold as _ }), 10),
        "hammer" => (Plain(if CLEAN { cdl_hammer_raw as _ } else { cdl_hammer as _ }), 11),
        "hangingman" => (Plain(if CLEAN { cdl_hangingman_raw as _ } else { cdl_hangingman as _ }), 11),
        "invertedhammer" => (Plain(if CLEAN { cdl_invertedhammer_raw as _ } else { cdl_invertedhammer as _ }), 11),
        "shootingstar" => (Plain(if CLEAN { cdl_shootingstar_raw as _ } else { cdl_shootingstar as _ }), 11),
        "takuri" => (Plain(if CLEAN { cdl_takuri_raw as _ } else { cdl_takuri as _ }), 10),
        // two-bar
        "engulfing" => (Plain(if CLEAN { cdl_engulfing_raw as _ } else { cdl_engulfing as _ }), 2),
        "harami" => (Plain(if CLEAN { cdl_harami_raw as _ } else { cdl_harami as _ }), 11),
        "haramicross" => (Plain(if CLEAN { cdl_haramicross_raw as _ } else { cdl_haramicross as _ }), 11),
        "piercing" => (Plain(if CLEAN { cdl_piercing_raw as _ } else { cdl_piercing as _ }), 11),
        "darkcloudcover" => (pen(if CLEAN { cdl_darkcloudcover_raw as _ } else { cdl_darkcloudcover as _ }, 0.5), 11),
        "dojistar" => (Plain(if CLEAN { cdl_dojistar_raw as _ } else { cdl_dojistar as _ }), 11),
        "homingpigeon" => (Plain(if CLEAN { cdl_homingpigeon_raw as _ } else { cdl_homingpigeon as _ }), 11),
        "matchinglow" => (Plain(if CLEAN { cdl_matchinglow_raw as _ } else { cdl_matchinglow as _ }), 6),
        "inneck" => (Plain(if CLEAN { cdl_inneck_raw as _ } else { cdl_inneck as _ }), 11),
        "onneck" => (Plain(if CLEAN { cdl_onneck_raw as _ } else { cdl_onneck as _ }), 11),
        "thrusting" => (Plain(if CLEAN { cdl_thrusting_raw as _ } else { cdl_thrusting as _ }), 11),
        "kicking" => (Plain(if CLEAN { cdl_kicking_raw as _ } else { cdl_kicking as _ }), 11),
        "kickingbylength" => (Plain(if CLEAN { cdl_kickingbylength_raw as _ } else { cdl_kickingbylength as _ }), 11),
        "separatinglines" => (Plain(if CLEAN { cdl_separatinglines_raw as _ } else { cdl_separatinglines as _ }), 11),
        "counterattack" => (Plain(if CLEAN { cdl_counterattack_raw as _ } else { cdl_counterattack as _ }), 11),
        // three-bar
        "morningstar" => (pen(if CLEAN { cdl_morningstar_raw as _ } else { cdl_morningstar as _ }, 0.3), 12),
        "eveningstar" => (pen(if CLEAN { cdl_eveningstar_raw as _ } else { cdl_eveningstar as _ }, 0.3), 12),
        "3inside" => (Plain(if CLEAN { cdl_3inside_raw as _ } else { cdl_3inside as _ }), 12),
        "3outside" => (Plain(if CLEAN { cdl_3outside_raw as _ } else { cdl_3outside as _ }), 3),
        "3whitesoldiers" => (Plain(if CLEAN { cdl_3whitesoldiers_raw as _ } else { cdl_3whitesoldiers as _ }), 12),
        "3blackcrows" => (Plain(if CLEAN { cdl_3blackcrows_raw as _ } else { cdl_3blackcrows as _ }), 13),
        "morningdojistar" => (pen(if CLEAN { cdl_morningdojistar_raw as _ } else { cdl_morningdojistar as _ }, 0.3), 12),
        "eveningdojistar" => (pen(if CLEAN { cdl_eveningdojistar_raw as _ } else { cdl_eveningdojistar as _ }, 0.3), 12),
        "abandonedbaby" => (pen(if CLEAN { cdl_abandonedbaby_raw as _ } else { cdl_abandonedbaby as _ }, 0.3), 12),
        "2crows" => (Plain(if CLEAN { cdl_2crows_raw as _ } else { cdl_2crows as _ }), 12),
        "upsidegap2crows" => (Plain(if CLEAN { cdl_upsidegap2crows_raw as _ } else { cdl_upsidegap2crows as _ }), 12),
        "advanceblock" => (Plain(if CLEAN { cdl_advanceblock_raw as _ } else { cdl_advanceblock as _ }), 12),
        "stalledpattern" => (Plain(if CLEAN { cdl_stalledpattern_raw as _ } else { cdl_stalledpattern as _ }), 12),
        "identical3crows" => (Plain(if CLEAN { cdl_identical3crows_raw as _ } else { cdl_identical3crows as _ }), 12),
        "sticksandwich" => (Plain(if CLEAN { cdl_sticksandwich_raw as _ } else { cdl_sticksandwich as _ }), 7),
        "tristar" => (Plain(if CLEAN { cdl_tristar_raw as _ } else { cdl_tristar as _ }), 12),
        "unique3river" => (Plain(if CLEAN { cdl_unique3river_raw as _ } else { cdl_unique3river as _ }), 12),
        "gapsidesidewhite" => (Plain(if CLEAN { cdl_gapsidesidewhite_raw as _ } else { cdl_gapsidesidewhite as _ }), 7),
        "tasukigap" => (Plain(if CLEAN { cdl_tasukigap_raw as _ } else { cdl_tasukigap as _ }), 7),
        "3starsinsouth" => (Plain(if CLEAN { cdl_3starsinsouth_raw as _ } else { cdl_3starsinsouth as _ }), 12),
        // four / five-bar
        "3linestrike" => (Plain(if CLEAN { cdl_3linestrike_raw as _ } else { cdl_3linestrike as _ }), 8),
        "breakaway" => (Plain(if CLEAN { cdl_breakaway_raw as _ } else { cdl_breakaway as _ }), 14),
        "ladderbottom" => (Plain(if CLEAN { cdl_ladderbottom_raw as _ } else { cdl_ladderbottom as _ }), 14),
        "concealbabyswall" => (Plain(if CLEAN { cdl_concealbabyswall_raw as _ } else { cdl_concealbabyswall as _ }), 13),
        "mathold" => (pen(if CLEAN { cdl_mathold_raw as _ } else { cdl_mathold as _ }, 0.5), 14),
        "risefall3methods" => (Plain(if CLEAN { cdl_risefall3methods_raw as _ } else { cdl_risefall3methods as _ }), 14),
        "xsidegap3methods" => (Plain(if CLEAN { cdl_xsidegap3methods_raw as _ } else { cdl_xsidegap3methods as _ }), 2),
        "hikkake" => (Plain(if CLEAN { cdl_hikkake_raw as _ } else { cdl_hikkake as _ }), 5),
        "hikkakemod" => (Plain(if CLEAN { cdl_hikkakemod_raw as _ } else { cdl_hikkakemod as _ }), 10),
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

/// The range a candle-setting measures (TA-Lib's `TA_RangeType`).
#[derive(Clone, Copy, PartialEq)]
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

/// The range of type `range` at bar `i` (TA-Lib `TA_CANDLERANGE`): the real body, the
/// high − low span, or both shadows.
#[inline(always)]
fn range_at(range: RangeType, o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize) -> f64 {
    match range {
        RangeType::RealBody => (c[i] - o[i]).abs(),
        RangeType::HighLow => h[i] - l[i],
        RangeType::Shadows => uppershadow(o, h, c, i) + lowershadow(o, l, c, i),
    }
}

/// The mean of the `range` over the `p`-bar window ending at every bar — exactly
/// [`kernels::sma`] of the ranges (NaN before the first full window), in one fused pass:
/// each range is formed as it enters and leaves, and the exact sum ([`Csum`]) is re-summed
/// only where it could not hold its sum, so no ranges array is materialized. A
/// non-finite range takes `sma`'s window-local careful path. Row 0 is at absolute
/// position `origin`. With `p == 0`, the range of the bar itself.
fn range_mean(range: RangeType, p: usize, o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let n = c.len();
    let r = |i: usize| range_at(range, o, h, l, c, i);
    if p == 0 {
        return (0..n).map(r).collect();
    }
    if p > n {
        return vec![f64::NAN; n];
    }
    let pf = p as f64;
    let mut out = Vec::with_capacity(n);
    out.resize(p - 1, f64::NAN);
    let mut sum = Csum::default();
    let mut clean = true;
    // Call the forced-inline range owner directly: the iterator closure can be
    // outlined by LTO, which otherwise adds two calls per sliding-window row.
    for i in 0..n {
        sum.add(range_at(range, o, h, l, c, i));
        if i >= p {
            sum.sub(range_at(range, o, h, l, c, i - p));
        }
        if !sum.is_exact() && i + 1 >= p {
            clean = sum.value().is_finite();
            if !clean {
                break;
            }
            sum = Csum::of((i + 1 - p..=i).map(r));
        }
        if i + 1 >= p {
            out.push(sum.mean(pf));
        }
    }
    if clean && sum.value().is_finite() {
        return out;
    }
    let ranges: Vec<f64> = (0..n).map(r).collect();
    kernels::sma_internal(&ranges, p, origin)
}

/// A setting's average at bar `i` from its range's mean series `mean` ([`range_mean`]):
/// `factor · mean[i − 1] · inv_div` — the window of the `avg_period` bars before `i` —
/// or `factor · range(i) · inv_div` for a per-bar setting; NaN before the first window.
/// `inv_div` is `1 / divisor(s)` (`/ div`, 1 or 2, a power of two, is exactly that
/// product).
#[inline(always)]
fn setting_average(s: Setting, inv_div: f64, mean: &[f64], i: usize) -> f64 {
    let at = if s.avg_period == 0 { Some(i) } else { i.checked_sub(1) };
    at.map_or(f64::NAN, |j| s.factor * mean[j] * inv_div)
}

/// TA-Lib's `TA_CANDLEAVERAGE` of setting `s` at every bar (NaN before the first full
/// window): `factor · (Σ range over the prior `avg_period` bars [i − avg_period, i − 1])
/// / avg_period / div`, or `factor · range(i) / div` when `avg_period == 0`. Row 0 is at
/// absolute position `origin` (the frame's). The one definition of a candle average: the
/// sum slides like TA-Lib's running total (O(1) per bar), but it is the sliding mean every
/// moving average uses ([`kernels::sma`]: an exact sum, window-local — a non-finite range
/// reaches only the windows holding it) — so a bad candle never outlives the windows
/// holding it, and a recomputed window (the live refresh) gives the full computation's
/// bits. On near-tie bars this follows the exact arithmetic where TA-Lib's plain running
/// total has drifted.
fn candle_average_series(s: Setting, o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Vec<f64> {
    let mut mean = range_mean(s.range, s.avg_period, o, h, l, c, origin);
    let inv_div = 1.0 / divisor(s);
    // In place, last bar first: bar `i` reads the mean at `i − 1` (or `i` itself).
    for i in (0..mean.len()).rev() {
        mean[i] = setting_average(s, inv_div, &mean, i);
    }
    mean
}

/// The averages of `settings` at every bar, as [`candle_average_series`] gives them, from
/// one mean series per distinct (range, period) — settings that differ only in their
/// factor share it — read at each bar without materializing a series per setting.
struct Averages<const K: usize> {
    means: Vec<Vec<f64>>,
    which: [usize; K],
}

impl<const K: usize> Averages<K> {
    fn new(settings: [Setting; K], o: &[f64], h: &[f64], l: &[f64], c: &[f64], origin: usize) -> Self {
        let mut keys: Vec<(RangeType, usize)> = Vec::with_capacity(K);
        let mut means = Vec::with_capacity(K);
        let which = settings.map(|s| {
            let key = (s.range, s.avg_period);
            keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                keys.push(key);
                means.push(range_mean(s.range, s.avg_period, o, h, l, c, origin));
                keys.len() - 1
            })
        });
        Averages { means, which }
    }

    /// Prepare the fixed settings and mean slices once, outside the bar loop.
    #[inline(always)]
    fn views(&self, settings: [Setting; K]) -> AverageView<'_, K> {
        AverageView {
            settings,
            inv_div: settings.map(|s| 1.0 / divisor(s)),
            means: std::array::from_fn(|k| self.means[self.which[k]].as_slice()),
        }
    }
}

/// Borrowed means with their fixed scales and lags. Each bar borrows this record;
/// the array builder never copies settings or slice arrays for each lag.
struct AverageView<'a, const K: usize> {
    settings: [Setting; K],
    inv_div: [f64; K],
    means: [&'a [f64]; K],
}

impl<const K: usize> AverageView<'_, K> {
    /// The averages at bar `i`, in the original multiplication order.
    #[inline(always)]
    fn at(&self, i: usize) -> [f64; K] {
        std::array::from_fn(|k| setting_average(self.settings[k], self.inv_div[k], self.means[k], i))
    }
}

/// Build a pattern column: NaN before `lookback`, then `f(i)` (0 / ±100 / ±80 / ±200)
/// per bar — and NaN while the bars a pattern reads, `[i − lookback, i]`, hold a missing
/// candle (any open / high / low / close that is not finite): a pattern has no value
/// over a bar it cannot see, and is exactly what clean data gives once that bar has
/// left its reach. The one builder every pattern goes through.
#[inline]
fn each_bar(
    lookback: usize,
    _o: &[f64],
    _h: &[f64],
    _l: &[f64],
    c: &[f64],
    mut f: impl FnMut(usize) -> f64,
) -> Vec<f64> {
    let n = c.len();
    let warm = lookback.min(n);
    // Single-write (D2): NaN warm-up then one write per valid bar, no prefill memset
    // of the valid region. `build_f64` keeps it safe (debug poison + write-all assert).
    let out = crate::buf::build_f64(n, |out| {
        for slot in &mut out[..warm] {
            slot.write(f64::NAN);
        }
        #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
        for i in warm..n {
            out[i].write(f(i));
        }
    });
    out
}

/// Apply the input rule around a numerical candle kernel. Clean frame inputs
/// bypass this wrapper; unclassified/raw inputs use the one missing owner.
#[cold]
#[inline(never)]
fn admit_candles(mut out: Vec<f64>, o: &[f64], h: &[f64], l: &[f64], c: &[f64], lookback: usize) -> Vec<f64> {
    for x in [o, h, l, c] { missing::blank_windows(&mut out, x, lookback + 1); }
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
    let avg = Averages::new(settings, o, h, l, c, origin);
    let view = avg.views(settings);
    each_bar(lookback, o, h, l, c, |i| f(i, &view.at(i)))
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
    let avg = Averages::new(settings, o, h, l, c, origin);
    let view = avg.views(settings);
    each_bar(lookback, o, h, l, c, |i| f(i, &view.at(i), &view.at(i - 1)))
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
    let avg = Averages::new(settings, o, h, l, c, origin);
    let view = avg.views(settings);
    each_bar(lookback, o, h, l, c, |i| f(i, &std::array::from_fn(|lag| view.at(i - lag))))
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
    /// positions) gives the full series' bits.
    #[test]
    fn the_average_series_is_the_window_mean_and_a_tail_recompute_matches_it() {
        for n in [3, 10, 11, 14, 60] {
            let [o, h, l, c] = ohlc(n);
            for s in [BODY_LONG, BODY_DOJI, SHADOW_SHORT, SHADOW_LONG, NEAR] {
                let r: Vec<f64> = (0..n).map(|i| range_at(s.range, &o, &h, &l, &c, i)).collect();
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
    /// holds it, the bars before are untouched, and once it has left every bar is what
    /// clean data gives — an average never
    /// carries it further. Every builder shape.
    #[test]
    fn a_missing_candle_reaches_only_the_bars_that_read_it() {
        let [o, h, l, c] = ohlc(80);
        let run = |o: &[f64], h: &[f64], l: &[f64], c: &[f64]| {
            [
                admit_candles(each_bar_avg([BODY_DOJI], 10, o, h, l, c, 0, |i, a| realbody(o, c, i) - a[0]), o, h, l, c, 10),
                admit_candles(each_bar_avg2([BODY_LONG, SHADOW_SHORT], 11, o, h, l, c, 0, |_, a, b| a[0] + b[1]), o, h, l, c, 11),
                admit_candles(each_bar_avg_n::<2, 3>([NEAR, SHADOW_LONG], 12, o, h, l, c, 0, |_, x| x[2][0] + x[1][1]), o, h, l, c, 12),
                admit_candles(each_bar(5, o, h, l, c, |i| c[i] - o[i - 5]), o, h, l, c, 5),
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
