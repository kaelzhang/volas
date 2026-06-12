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
// Parity holds on the valid region.

mod multi_bar;
mod one_bar;
mod three_bar;
mod two_bar;
pub use multi_bar::*;
pub use one_bar::*;
pub use three_bar::*;
pub use two_bar::*;

/// A candlestick-pattern recogniser. Most take only `(open, high, low, close)`; a few
/// also take TA-Lib's `penetration` ratio (with a pattern-specific default).
pub enum CandlePattern {
    /// `(open, high, low, close)`.
    Plain(fn(&[f64], &[f64], &[f64], &[f64]) -> Vec<f64>),
    /// `(open, high, low, close, penetration)`, plus the TA-Lib default penetration.
    Penetration {
        f: fn(&[f64], &[f64], &[f64], &[f64], f64) -> Vec<f64>,
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

/// The setting's range at bar `i`.
#[inline]
fn range(s: Setting, o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize) -> f64 {
    match s.range {
        RangeType::RealBody => realbody(o, c, i),
        RangeType::HighLow => h[i] - l[i],
        RangeType::Shadows => uppershadow(o, h, c, i) + lowershadow(o, l, c, i),
    }
}

/// TA-Lib's `TA_CANDLEAVERAGE`: `factor · (avg of range over the prior avg_period bars,
/// or the bar's own range when avg_period == 0) / (2 if Shadows else 1)`. The average
/// window is `[i-avg_period, i-1]` — the bars *before* `i` (so callers start at `i >=
/// avg_period`).
fn candle_average(s: Setting, o: &[f64], h: &[f64], l: &[f64], c: &[f64], i: usize) -> f64 {
    let base = if s.avg_period != 0 {
        let mut sum = 0.0;
        for j in (i - s.avg_period)..i {
            sum += range(s, o, h, l, c, j);
        }
        sum / s.avg_period as f64
    } else {
        range(s, o, h, l, c, i)
    };
    let div = if matches!(s.range, RangeType::Shadows) {
        2.0
    } else {
        1.0
    };
    s.factor * base / div
}

#[inline]
fn candle_average_from_total(s: Setting, total: f64) -> f64 {
    let div = if matches!(s.range, RangeType::Shadows) {
        2.0
    } else {
        1.0
    };
    s.factor * (total / s.avg_period as f64) / div
}

/// TA-Lib's `TA_CANDLEAVERAGE` precomputed for **every** bar in O(n): a running window
/// sum of the setting's range over `[i-avg_period, i-1]` replaces [`candle_average`]'s
/// per-bar O(avg_period) rescan (the dominant cost of most CDL patterns — TA-Lib itself
/// slides this total). `out[i]` is valid for `i >= avg_period`; callers read only
/// `i >= lookback >= avg_period`. For `avg_period == 0` it is the bar's own range. The
/// final scaling keeps TA-Lib's exact `factor·(total/period)/div` order.
fn candle_average_series(s: Setting, o: &[f64], h: &[f64], l: &[f64], c: &[f64]) -> Vec<f64> {
    let n = c.len();
    let mut out = vec![0.0; n];
    let div = if matches!(s.range, RangeType::Shadows) {
        2.0
    } else {
        1.0
    };
    if s.avg_period == 0 {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = s.factor * range(s, o, h, l, c, i) / div;
        }
        return out;
    }
    let p = s.avg_period;
    if p > n {
        return out;
    }
    let pf = p as f64;
    // Seed the window [0, p-1] (so the first valid average lands at i = p), then slide
    // add-new / subtract-trailing — one range evaluation per bar instead of `p`.
    let mut total = 0.0;
    for j in 0..p {
        total += range(s, o, h, l, c, j);
    }
    for i in p..n {
        out[i] = s.factor * (total / pf) / div;
        total += range(s, o, h, l, c, i) - range(s, o, h, l, c, i - p);
    }
    out
}

/// Per-bar pattern builder that carries a **scalar** running window-sum for each of the
/// `K` settings (TA-Lib's own approach), so a pattern reads its candle averages in O(1)
/// per bar with no per-bar rescan and no materialised average arrays. Each total is
/// seeded over `[lookback - avg_period, lookback - 1]` (TA-Lib's trailing-index start)
/// and slid add-new / subtract-trailing; `avgs[k]` handed to `f` is `settings[k]`'s
/// average at bar `i` (window `[i-avg_period, i-1]`; the bar's own range when
/// `avg_period == 0`). NaN before `lookback`.
#[inline]
fn each_bar_avg<const K: usize>(
    settings: [Setting; K],
    lookback: usize,
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    f: impl Fn(usize, &[f64; K]) -> f64,
) -> Vec<f64> {
    let n = c.len();
    let mut out = vec![f64::NAN; n];
    if lookback >= n {
        return out;
    }
    let mut total = [0.0f64; K];
    for k in 0..K {
        let s = settings[k];
        if s.avg_period != 0 {
            for j in (lookback - s.avg_period)..lookback {
                total[k] += range(s, o, h, l, c, j);
            }
        }
    }
    let mut avgs = [0.0f64; K];
    for i in lookback..n {
        for k in 0..K {
            let s = settings[k];
            let div = if matches!(s.range, RangeType::Shadows) {
                2.0
            } else {
                1.0
            };
            avgs[k] = if s.avg_period != 0 {
                s.factor * (total[k] / s.avg_period as f64) / div
            } else {
                s.factor * range(s, o, h, l, c, i) / div
            };
        }
        out[i] = f(i, &avgs);
        for k in 0..K {
            let s = settings[k];
            if s.avg_period != 0 {
                total[k] += range(s, o, h, l, c, i) - range(s, o, h, l, c, i - s.avg_period);
            }
        }
    }
    out
}

/// Like [`each_bar_avg`] but for two-bar patterns: the closure also receives the prior
/// bar's averages (`prev`), so `candle_average(s, i-1)` reads `prev[k]` and
/// `candle_average(s, i)` reads `cur[k]` — both from the same scalar running totals (no
/// arrays). Seeds at bar `lookback-1`, carries `prev` forward each step.
#[inline]
fn each_bar_avg2<const K: usize>(
    settings: [Setting; K],
    lookback: usize,
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    f: impl Fn(usize, &[f64; K], &[f64; K]) -> f64,
) -> Vec<f64> {
    let n = c.len();
    let mut out = vec![f64::NAN; n];
    if lookback == 0 || lookback >= n {
        return out;
    }
    // Seed each total over bar (lookback-1)'s window `[(lookback-1)-avg_period, lookback-2]`.
    let mut total = [0.0f64; K];
    for k in 0..K {
        let s = settings[k];
        if s.avg_period != 0 {
            for j in ((lookback - 1) - s.avg_period)..(lookback - 1) {
                total[k] += range(s, o, h, l, c, j);
            }
        }
    }
    // Averages at bar (lookback-1) — the first `prev`. (Everything inlined, no inner
    // closures: the closure form here failed to inline and ran ~4x slower for K >= 2.)
    let mut prev = [0.0f64; K];
    for k in 0..K {
        let s = settings[k];
        let div = if matches!(s.range, RangeType::Shadows) {
            2.0
        } else {
            1.0
        };
        prev[k] = if s.avg_period != 0 {
            s.factor * (total[k] / s.avg_period as f64) / div
        } else {
            s.factor * range(s, o, h, l, c, lookback - 1) / div
        };
        if s.avg_period != 0 {
            total[k] += range(s, o, h, l, c, lookback - 1)
                - range(s, o, h, l, c, (lookback - 1) - s.avg_period);
        }
    }
    let mut cur = [0.0f64; K];
    for i in lookback..n {
        for k in 0..K {
            let s = settings[k];
            let div = if matches!(s.range, RangeType::Shadows) {
                2.0
            } else {
                1.0
            };
            cur[k] = if s.avg_period != 0 {
                s.factor * (total[k] / s.avg_period as f64) / div
            } else {
                s.factor * range(s, o, h, l, c, i) / div
            };
        }
        out[i] = f(i, &cur, &prev);
        prev = cur;
        for k in 0..K {
            let s = settings[k];
            if s.avg_period != 0 {
                total[k] += range(s, o, h, l, c, i) - range(s, o, h, l, c, i - s.avg_period);
            }
        }
    }
    out
}

/// Multi-bar generalisation of [`each_bar_avg`]: carries the last `L` bars' averages as
/// scalar running sums, handing the closure `hist` where `hist[lag][k]` is `settings[k]`'s
/// average at bar `i-lag` (lag `0` = current). For patterns that read a candle average a
/// fixed number of bars back (e.g. tristar at `i-2`, breakaway at `i-4`). No arrays — a
/// short `[[f64; K]; L]` history is shifted each bar.
///
/// Maintains all `K` running sums *unconditionally* on every bar, so it wins when a
/// pattern reads its averages on most bars (weak early gating) or has several settings
/// whose per-bar rescans add up. For *strongly-gated* patterns — those that reject most
/// bars on cheap geometry first (three same-colour candles, a required gap) — plain
/// [`each_bar`] + lazy [`candle_average`] is faster, since the average is only computed on
/// the few bars that pass the gate; several three-/multi-bar patterns deliberately keep
/// `each_bar` for that reason (measured against this helper — do not blindly migrate them).
#[inline]
fn each_bar_avg_n<const K: usize, const L: usize>(
    settings: [Setting; K],
    lookback: usize,
    o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    f: impl Fn(usize, &[[f64; K]; L]) -> f64,
) -> Vec<f64> {
    let n = c.len();
    let mut out = vec![f64::NAN; n];
    if lookback + 1 < L || lookback >= n {
        return out;
    }
    let first = lookback + 1 - L; // first bar whose averages enter the history
    let mut total = [0.0f64; K];
    for k in 0..K {
        let s = settings[k];
        if s.avg_period != 0 {
            for j in (first - s.avg_period)..first {
                total[k] += range(s, o, h, l, c, j);
            }
        }
    }
    let mut hist = [[0.0f64; K]; L];
    for bar in first..n {
        let mut a = [0.0f64; K];
        for k in 0..K {
            let s = settings[k];
            let div = if matches!(s.range, RangeType::Shadows) {
                2.0
            } else {
                1.0
            };
            a[k] = if s.avg_period != 0 {
                s.factor * (total[k] / s.avg_period as f64) / div
            } else {
                s.factor * range(s, o, h, l, c, bar) / div
            };
        }
        for j in (1..L).rev() {
            hist[j] = hist[j - 1];
        }
        hist[0] = a;
        if bar >= lookback {
            out[bar] = f(bar, &hist);
        }
        for k in 0..K {
            let s = settings[k];
            if s.avg_period != 0 {
                total[k] += range(s, o, h, l, c, bar) - range(s, o, h, l, c, bar - s.avg_period);
            }
        }
    }
    out
}

/// Build a per-bar pattern column: NaN before `lookback`, then `f(i)` (0 / ±100 / ±80)
/// per bar. Shared by all pattern submodules.
#[inline]
fn each_bar(n: usize, lookback: usize, f: impl Fn(usize) -> f64) -> Vec<f64> {
    let warm = lookback.min(n);
    // The old `extend(map(...))` measured slower for simple two-bar patterns.
    // NaN-prefilled buffer: the warm-up stays NaN and the loop overwrites the
    // valid region (D2 2026-06-12 — replaces the with_capacity + set_len pattern;
    // the prefill is a vectorized splat, measured at parity by make perf-ab).
    let mut out = vec![f64::NAN; n];
    let mut i = warm;
    while i < n {
        out[i] = f(i);
        i += 1;
    }
    out
}

/// Output buffer for specialised CDL loops: initialize only the warm-up NaN prefix and
/// let the caller write every valid row exactly once. `f64` has no destructor, so the
/// temporary uninitialised valid region is safe as long as callers fill `[lookback, n)`.
#[inline]
fn candle_output(n: usize, lookback: usize) -> Option<Vec<f64>> {
    if lookback >= n {
        return None;
    }
    // NaN-prefilled buffer: the warm-up stays NaN and the loop overwrites the
    // valid region (D2 2026-06-12 — replaces the with_capacity + set_len pattern;
    // the prefill is a vectorized splat, measured at parity by make perf-ab).
    Some(vec![f64::NAN; n])
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

    /// The candle-settings averagers have branches (Shadows range → ÷2, `avg_period == 0`
    /// → the bar's own range, and the short-series guards) that the 61 landed patterns
    /// never select for these specific helpers. Exercise them directly so the framework
    /// is fully covered regardless of which settings the current patterns happen to use.
    #[test]
    fn candle_average_helper_branches() {
        let o = vec![1.0; 20];
        let h = vec![3.0; 20];
        let l = vec![0.0; 20];
        let c = vec![2.0; 20];
        let s3 = vec![1.0; 3];

        // candle_average_series: Shadows range (÷2), avg_period 0, and period > n.
        candle_average_series(SHADOW_SHORT, &o, &h, &l, &c);
        candle_average_series(SHADOW_LONG, &o, &h, &l, &c);
        candle_average_series(NEAR, &s3, &s3, &s3, &s3);

        // candle_average (per-bar): Shadows range and avg_period 0.
        let _ = candle_average(SHADOW_SHORT, &o, &h, &l, &c, 12);
        let _ = candle_average(SHADOW_LONG, &o, &h, &l, &c, 12);
        let _ = candle_average_from_total(SHADOW_SHORT, 6.0);

        // each_bar_avg2: a zero lookback trips its guard; a valid call with an
        // avg_period-0 (SHADOW_LONG) and a Shadows (SHADOW_SHORT) setting covers the
        // body's ÷2 and own-range branches in both the seed and per-bar loops.
        each_bar_avg2([BODY_LONG, NEAR], 0, &o, &h, &l, &c, |_, a, _| a[0]);
        each_bar_avg2(
            [SHADOW_LONG, SHADOW_SHORT],
            12,
            &o,
            &h,
            &l,
            &c,
            |_, a, b| a[0] + b[0],
        );

        // each_bar_avg_n: the lookback guard, a Shadows setting (÷2), and an
        // avg_period-0 setting (the bar's own range).
        each_bar_avg_n::<1, 2>([NEAR], 0, &o, &h, &l, &c, |_, hist| hist[0][0]);
        each_bar_avg_n::<1, 2>([SHADOW_SHORT], 12, &o, &h, &l, &c, |_, hist| hist[0][0]);
        each_bar_avg_n::<1, 2>([SHADOW_LONG], 6, &o, &h, &l, &c, |_, hist| hist[0][0]);
    }
}
