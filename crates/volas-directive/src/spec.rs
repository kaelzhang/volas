//! Per-command metadata: canonical sub-command names, positional-argument
//! defaults **and validation bounds**, and default series. This is the single
//! source of command argument knowledge, shared by `exec` (runtime defaulting +
//! boundary validation), `lookback`, and `stringify`, so the values are never
//! duplicated.

use std::borrow::Cow;

use crate::types::ArgValue;

/// A positional-argument default. `Required` means the argument has no default
/// (the caller must supply it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgDefault {
    /// No default — required.
    Required,
    /// Integer default (periods).
    Int(usize),
    /// Float default (band multiples, KDJ seed).
    Float(f64),
    /// i64 default (e.g. trading days).
    I64(i64),
    /// String default (e.g. a time-frame for `hv`).
    Str(&'static str),
}

impl ArgDefault {
    /// The default as a bound [`ArgValue`], used by `bind` to fill an omitted
    /// argument slot. `Required` has no default (the caller must supply it).
    pub fn value(&self) -> Option<ArgValue> {
        match self {
            ArgDefault::Required => None,
            ArgDefault::Int(n) => Some(ArgValue::Usize(*n)),
            ArgDefault::Float(f) => Some(ArgValue::F64(*f)),
            ArgDefault::I64(n) => Some(ArgValue::I64(*n)),
            ArgDefault::Str(s) => Some(ArgValue::Str((*s).to_string())),
        }
    }
}

/// A positional argument's legal domain — what values are a usable
/// *configuration* for this parameter. Checked once per supplied argument at
/// the exec boundary (defaults are valid by construction). The principle (V17):
/// a value whose output is degenerate **by construction** — a no-signal column
/// (`ma:0` all-NaN, `cci:1` identically 0), a division-by-zero (`asi:0`), or an
/// arithmetic panic (`change:0`) — is an invalid configuration and errors,
/// while a merely *unusual* value (a period longer than the data: pure warm-up
/// NaN) stays legal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgBound {
    /// Integer >= `min`. `IntMin(1)` is the plain window / period / count
    /// (`0` would be a valid-shaped all-NaN column); `IntMin(2)` is the
    /// second-moment / regression family, whose formula is undefined or
    /// identically constant on a single sample.
    IntMin(usize),
    /// Integer within `[lo, hi]` — e.g. the TA-Lib matype selector (0..=8) or
    /// `mavp`'s period limits (whose cache allocates `max_period` slots, so it
    /// adopts TA-Lib's documented 100000 period ceiling).
    IntRange(usize, usize),
    /// i64 exactly one of the listed values (e.g. `increase`'s direction: +1
    /// rising / -1 falling; any other value can never match a run).
    OneOfI64(&'static [i64]),
    /// i64 >= `min` (e.g. `hv`'s annualization trading-days: 0 / negative
    /// would put 0 or NaN under the square root).
    I64Min(i64),
    /// Any finite float — band/deviation multipliers, sign-free like TA-Lib's
    /// `nbdev` (a negative multiplier flips the band; NaN/inf would poison the
    /// whole column silently).
    Finite,
    /// Finite float >= `lo` (e.g. SAR acceleration factors: TA-Lib's domain is
    /// non-negative; a negative factor walks the stop *away* from price).
    FloatMin(f64),
    /// Finite float > `lo` exclusive (e.g. `asi`'s limit-move divisor: the
    /// formula divides by it, so 0 yields ±inf).
    FloatGt(f64),
    /// Finite float within `[lo, hi]` (T3 vfactor 0..=1, KDJ seed 0..=100,
    /// MAMA limits 0.01..=0.99 — all TA-Lib / definitional domains).
    FloatRange(f64, f64),
    /// A free string argument, validated downstream (e.g. `hv`'s time-frame
    /// token, parsed by `tf_to_minutes`).
    AnyStr,
}

impl ArgBound {
    /// Parse a supplied argument token **once** against this bound, returning the
    /// typed [`ArgValue`] (validation and type conversion are the same single
    /// parse — there is no separate "validate then re-read" pass). `Err` carries
    /// the human-readable requirement (the caller adds command / position context).
    pub fn bind(&self, s: &str) -> std::result::Result<ArgValue, String> {
        match self {
            ArgBound::IntMin(min) => match s.parse::<usize>() {
                Ok(v) if v >= *min => Ok(ArgValue::Usize(v)),
                _ => Err(format!("must be an integer >= {min}")),
            },
            ArgBound::IntRange(lo, hi) => match s.parse::<usize>() {
                Ok(v) if v >= *lo && v <= *hi => Ok(ArgValue::Usize(v)),
                _ => Err(format!("must be an integer in [{lo}, {hi}]")),
            },
            ArgBound::OneOfI64(allowed) => match s.parse::<i64>() {
                Ok(v) if allowed.contains(&v) => Ok(ArgValue::I64(v)),
                _ => Err(format!(
                    "must be one of {}",
                    allowed
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(" / ")
                )),
            },
            ArgBound::I64Min(min) => match s.parse::<i64>() {
                Ok(v) if v >= *min => Ok(ArgValue::I64(v)),
                _ => Err(format!("must be an integer >= {min}")),
            },
            ArgBound::Finite => match s.parse::<f64>() {
                Ok(v) if v.is_finite() => Ok(ArgValue::F64(v)),
                _ => Err("must be a finite number".to_string()),
            },
            ArgBound::FloatMin(lo) => match s.parse::<f64>() {
                Ok(v) if v.is_finite() && v >= *lo => Ok(ArgValue::F64(v)),
                _ => Err(format!("must be a finite number >= {lo}")),
            },
            ArgBound::FloatGt(lo) => match s.parse::<f64>() {
                Ok(v) if v.is_finite() && v > *lo => Ok(ArgValue::F64(v)),
                _ => Err(format!("must be a finite number > {lo}")),
            },
            ArgBound::FloatRange(lo, hi) => match s.parse::<f64>() {
                Ok(v) if v.is_finite() && v >= *lo && v <= *hi => Ok(ArgValue::F64(v)),
                _ => Err(format!("must be a number in [{lo}, {hi}]")),
            },
            ArgBound::AnyStr => Ok(ArgValue::Str(s.to_string())),
        }
    }
}

/// One positional argument: its default plus its validation bound, declared
/// together so the table below documents each parameter's meaning and domain
/// in a single place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arg {
    pub default: ArgDefault,
    pub bound: ArgBound,
}

// --- argument constructors (the table's vocabulary) --------------------------
// Each names a parameter *kind*, so a spec arm reads as documentation:
// `p(20)` an optional window defaulting to 20; `p_req()` a required window;
// `p2(14)` a window needing >= 2 samples; `matype()` a TA-Lib MA selector;
// `mult(2.0)` a sign-free band multiplier; `frange/fmin/fgt` constrained floats.

/// Window / period / count argument with a default; >= 1.
const fn p(n: usize) -> Arg {
    Arg { default: ArgDefault::Int(n), bound: ArgBound::IntMin(1) }
}

/// Required window / period argument; >= 1.
const fn p_req() -> Arg {
    Arg { default: ArgDefault::Required, bound: ArgBound::IntMin(1) }
}

/// Second-moment / regression window with a default; >= 2 (variance, slope,
/// correlation and friends are undefined or identically constant on 1 sample).
const fn p2(n: usize) -> Arg {
    Arg { default: ArgDefault::Int(n), bound: ArgBound::IntMin(2) }
}

/// Required second-moment window; >= 2.
const fn p2_req() -> Arg {
    Arg { default: ArgDefault::Required, bound: ArgBound::IntMin(2) }
}

/// TA-Lib MA-type selector: 0 SMA, 1 EMA, 2 WMA, 3 DEMA, 4 TEMA, 5 TRIMA,
/// 6 KAMA, 7 MAMA, 8 T3.
const fn matype() -> Arg {
    Arg { default: ArgDefault::Int(0), bound: ArgBound::IntRange(0, 8) }
}

/// Required `mavp` period limit: >= 1 and <= TA-Lib's documented 100000 period
/// ceiling (the kernel's per-period cache allocates `max_period` slots, so an
/// unbounded value is a memory hazard). No dominant default — the caller supplies it.
const fn prange_req() -> Arg {
    Arg { default: ArgDefault::Required, bound: ArgBound::IntRange(1, 100_000) }
}

/// Sign-free finite multiplier (band / deviation multiples).
const fn mult(x: f64) -> Arg {
    Arg { default: ArgDefault::Float(x), bound: ArgBound::Finite }
}

/// Finite float bounded below (inclusive).
const fn fmin(x: f64, lo: f64) -> Arg {
    Arg { default: ArgDefault::Float(x), bound: ArgBound::FloatMin(lo) }
}

/// Finite float bounded below (exclusive).
const fn fgt(x: f64, lo: f64) -> Arg {
    Arg { default: ArgDefault::Float(x), bound: ArgBound::FloatGt(lo) }
}

/// Finite float within `[lo, hi]`.
const fn frange(x: f64, lo: f64, hi: f64) -> Arg {
    Arg { default: ArgDefault::Float(x), bound: ArgBound::FloatRange(lo, hi) }
}

/// i64 bounded below.
const fn i64min(x: i64, lo: i64) -> Arg {
    Arg { default: ArgDefault::I64(x), bound: ArgBound::I64Min(lo) }
}

/// i64 restricted to an explicit value set.
const fn one_of(x: i64, allowed: &'static [i64]) -> Arg {
    Arg { default: ArgDefault::I64(x), bound: ArgBound::OneOfI64(allowed) }
}

/// Free string argument (validated downstream).
const fn tf_str(s: &'static str) -> Arg {
    Arg { default: ArgDefault::Str(s), bound: ArgBound::AnyStr }
}

/// The metadata for one command (after sub-command canonicalization).
pub struct CommandSpec {
    /// Per-position argument defaults + validation bounds.
    pub args: Cow<'static, [Arg]>,
    /// Default series column for each `@` slot.
    pub series: &'static [&'static str],
}

/// Canonicalize a sub-command alias (e.g. `macd.dif` -> main, `boll.u` ->
/// `upper`). Returns the canonical sub, or `None` for the main command.
pub fn canon_sub<'a>(name: &str, sub: Option<&'a str>) -> Option<Cow<'a, str>> {
    match (name, sub) {
        ("macd" | "macdext" | "macdfix", None | Some("dif")) => None,
        ("macd" | "macdext" | "macdfix", Some("s" | "signal" | "dea")) => Some("signal".into()),
        ("macd" | "macdext" | "macdfix", Some("h" | "histogram" | "macd")) => {
            Some("histogram".into())
        }
        ("aroon", Some("u" | "up")) => Some("up".into()),
        ("aroon", Some("d" | "down")) => Some("down".into()),
        ("stoch", Some("k" | "slowk")) => Some("k".into()),
        ("stoch", Some("d" | "slowd")) => Some("d".into()),
        ("stochf", Some("k" | "fastk")) => Some("k".into()),
        ("stochf", Some("d" | "fastd")) => Some("d".into()),
        ("stochrsi", Some("k" | "fastk")) => Some("k".into()),
        ("stochrsi", Some("d" | "fastd")) => Some("d".into()),
        ("boll" | "donchian", None | Some("middle" | "m")) => None,
        ("boll" | "donchian", Some("u" | "upper")) => Some("upper".into()),
        ("boll" | "donchian", Some("l" | "lower")) => Some("lower".into()),
        ("accbands", None | Some("middle" | "m")) => None,
        ("accbands", Some("u" | "upper")) => Some("upper".into()),
        ("accbands", Some("l" | "lower")) => Some("lower".into()),
        // Hilbert multi-output lines: primary line is the main command (P3).
        ("ht_phasor", None | Some("i" | "inphase")) => None,
        ("ht_phasor", Some("q" | "quad" | "quadrature")) => Some("quadrature".into()),
        ("ht_sine", None | Some("sine")) => None,
        ("ht_sine", Some("lead" | "leadsine")) => Some("leadsine".into()),
        ("mama", None | Some("mama")) => None,
        ("mama", Some("fama")) => Some("fama".into()),
        // DMA (China 平行线差): the DDD difference line is the main command; AMA is its signal.
        ("dma", None | Some("ddd")) => None,
        ("dma", Some("ama")) => Some("ama".into()),
        // Vortex has no primary line — both outputs are required sub-commands.
        ("vortex", Some("p" | "plus")) => Some("plus".into()),
        ("vortex", Some("m" | "minus")) => Some("minus".into()),
        // Keltner: middle line (main) plus upper / lower bands.
        ("keltner", None | Some("middle" | "m")) => None,
        ("keltner", Some("u" | "upper")) => Some("upper".into()),
        ("keltner", Some("l" | "lower")) => Some("lower".into()),
        // Pivot Points: the pivot itself is the main line; the R/S levels are sub-commands.
        ("pivot_points", None | Some("p" | "pp")) => None,
        // Ichimoku: five required lines, no primary — accept the English aliases too.
        ("ichimoku", Some("tenkan" | "conversion")) => Some("tenkan".into()),
        ("ichimoku", Some("kijun" | "base")) => Some("kijun".into()),
        ("ichimoku", Some("senkou_a" | "span_a")) => Some("senkou_a".into()),
        ("ichimoku", Some("senkou_b" | "span_b")) => Some("senkou_b".into()),
        ("ichimoku", Some("chikou" | "lagging")) => Some("chikou".into()),
        // Supertrend: the trailing line is the main output; `.direction` gives the +1/−1 trend.
        ("supertrend", None | Some("line")) => None,
        ("supertrend", Some("direction" | "trend" | "d")) => Some("direction".into()),
        (_, Some(s)) => Some(Cow::Borrowed(s)),
        (_, None) => None,
    }
}

/// Every command name (the indicator vocabulary), as an enumerable list — the
/// single source `is_command` checks against and `scripts/count_indicators.py`
/// derives the documented indicator count from, so README / INDICATORS.md never
/// drift from the actual command set.
pub const COMMANDS: &[&str] = &[
    "ma", "ema", "smma", "apo", "ppo", "macdext", "macdfix", "wma", "dema", "tema",
    "trima", "t3", "kama", "mavp", "sar", "sarext", "accbands", "macd", "boll", "bbw",
    "rsv", "kdj", "rsi", "bbi", "psy", "pvt", "nvi", "pvi", "dpo", "cmf",
    "chop", "kst", "emv", "mass_index", "efi", "tsi", "crsi", "bias", "dma", "vortex",
    "brar", "vr", "coppock", "relative_vigor", "dkx", "wvad", "cdp", "mike", "keltner",
    "stoch_momentum", "ttm_squeeze", "pivot_points", "ichimoku", "wad", "asi", "supertrend",
    "tr", "atr", "llv", "hhv", "donchian", "hv", "increase", "style", "repeat", "change",
    "mom", "roc", "rocp", "rocr", "rocr100", "willr", "cmo", "cci", "trix", "imi", "mfi",
    "ultosc", "stoch", "stochf", "stochrsi", "plus_dm", "minus_dm", "plus_di", "minus_di",
    "dx", "adx", "adxr", "aroon", "aroonosc", "sum", "maxindex", "minindex", "minmax",
    "minmaxindex", "natr", "bop", "midpoint", "midprice", "linearreg", "linearreg_slope",
    "linearreg_intercept", "linearreg_angle", "tsf", "var", "stddev", "correl", "beta",
    "median", "quantile", "rank", "skew", "kurt", "sem",
    "vwma", "alma", "hma", "swma", "cog", "dev", "rci", "iii", "kcw", "mode",
    "pivothigh", "pivotlow",
    "obv", "ad", "adosc", "avgprice", "medprice", "typprice", "wclprice", "ht_dcperiod",
    "ht_dcphase", "ht_phasor", "ht_sine", "ht_trendline", "ht_trendmode", "mama",
];

/// Whether `name` is a known command (regardless of sub-command).
pub fn is_command(name: &str) -> bool {
    COMMANDS.contains(&name)
}

/// Canonicalize a command name: case-insensitive (P6), with `cdl` aliased to the
/// `style` candlestick namespace. Borrows for the common already-lowercase case
/// (every built-in name), so binding an ordinary command costs no allocation.
pub fn normalize(name: &str) -> Cow<'_, str> {
    if name.eq_ignore_ascii_case("cdl") {
        Cow::Borrowed("style")
    } else if name.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(name.to_ascii_lowercase())
    } else {
        Cow::Borrowed(name)
    }
}

/// The spec for a command given its (already-canonicalized) sub. `None` for an
/// unknown command **or an invalid sub-command** (the match is sub-strict, so it
/// also drives sub validation).
///
/// Each arm names its parameters in order; the [`Arg`] constructor used for a
/// slot documents that parameter's domain (see the constructor docs above).
pub fn command_spec(name: &str, sub: Option<&str>) -> Option<CommandSpec> {
    let (args, series): (&'static [Arg], &'static [&'static str]) = match (name, sub) {
        // ma: period, matype.
        ("ma", None) => (const { &[p_req(), matype()] }, &["close"]),
        ("ema" | "smma", None) => (const { &[p_req()] }, &["close"]),
        // apo / ppo: fast period, slow period, matype.
        ("apo" | "ppo", None) => (const { &[p(12), p(26), matype()] }, &["close"]),
        // macdext: fast, fastmatype, slow, slowmatype[, signal, signalmatype]. Matypes
        // default to SMA (0), unlike macd's fixed EMA.
        ("macdext", None) => (const { &[p(12), matype(), p(26), matype()] }, &["close"]),
        ("macdext", Some("signal" | "histogram")) => (
            const { &[p(12), matype(), p(26), matype(), p(9), matype()] },
            &["close"],
        ),
        ("wma" | "dema" | "tema" | "trima", None) => (const { &[p_req()] }, &["close"]),
        // TradingView moving averages.
        ("vwma", None) => (const { &[p_req()] }, &["close", "volume"]),
        // alma: period, offset (Gaussian peak position, [0,1]), sigma (>0, width).
        ("alma", None) => (const { &[p_req(), frange(0.85, 0.0, 1.0), fgt(6.0, 0.0)] }, &["close"]),
        ("hma", None) => (const { &[p_req()] }, &["close"]),
        ("swma", None) => (&[], &["close"]),
        // t3: period, vfactor (volume factor, TA-Lib domain [0, 1]).
        ("t3", None) => (const { &[p_req(), frange(0.7, 0.0, 1.0)] }, &["close"]),
        ("kama", None) => (const { &[p_req()] }, &["close"]),
        // mavp: min_period, max_period (each period value is clamped into this
        // range), matype; real defaults to close, periods is the required second
        // series. min <= max is cross-checked in the exec arm.
        ("mavp", None) => (const { &[prange_req(), prange_req(), matype()] }, &["close"]),
        // sar: acceleration factor, maximum factor — both non-negative (a
        // negative factor walks the stop away from price, producing garbage).
        ("sar", None) => (const { &[fmin(0.02, 0.0), fmin(0.2, 0.0)] }, &["high", "low"]),
        // sarext: start (signed — negative means start short), offset_on_reverse
        // (>= 0), then long (init/step/max) + short (init/step/max), all >= 0.
        ("sarext", None) => (
            const { &[
                mult(0.0),
                fmin(0.0, 0.0),
                fmin(0.02, 0.0),
                fmin(0.02, 0.0),
                fmin(0.2, 0.0),
                fmin(0.02, 0.0),
                fmin(0.02, 0.0),
                fmin(0.2, 0.0),
            ] },
            &["high", "low"],
        ),
        // macd: fast period, slow period[, signal period].
        ("macd", None) => (const { &[p(12), p(26)] }, &["close"]),
        ("macd", Some("signal" | "histogram")) => (const { &[p(12), p(26), p(9)] }, &["close"]),
        // macdfix: fast/slow fixed at 12/26; the line takes no args, the signal/histogram
        // take only the signal period.
        ("macdfix", None) => (&[], &["close"]),
        ("macdfix", Some("signal" | "histogram")) => (const { &[p(9)] }, &["close"]),
        // boll: period[, deviation multiplier (sign-free, TA-Lib nbdev)].
        ("boll", None) => (const { &[p(20)] }, &["close"]),
        ("boll", Some("upper" | "lower")) => (const { &[p(20), mult(2.0)] }, &["close"]),
        ("bbw", None) => (const { &[p(20)] }, &["close"]),
        ("accbands", None) => (const { &[p(20)] }, &["close"]),
        ("accbands", Some("upper" | "lower")) => (const { &[p(20)] }, &["high", "low"]),
        ("rsv", None) => (const { &[p_req()] }, &["high", "low", "close"]),
        // kdj: rsv period, K smoothing[, D smoothing], seed (a %K/%D percentage,
        // so its domain is [0, 100]).
        ("kdj", Some("k")) => (
            const { &[p(9), p(3), frange(50.0, 0.0, 100.0)] },
            &["high", "low", "close"],
        ),
        ("kdj", Some("d" | "j")) => (
            const { &[p(9), p(3), p(3), frange(50.0, 0.0, 100.0)] },
            &["high", "low", "close"],
        ),
        // rsi: required period — 14 is Wilder's original, but 2 / 9 / 21 / 25 are all
        // common, so there is no single dominant default.
        ("rsi", None) => (const { &[p_req()] }, &["close"]),
        ("bbi", None) => (const { &[p(3), p(6), p(12), p(24)] }, &["close"]),
        ("tr", None) => (&[], &["high", "low", "close"]),
        ("atr", None) => (const { &[p_req()] }, &["high", "low", "close"]),
        ("llv", None) => (const { &[p_req()] }, &["low"]),
        ("hhv", None) => (const { &[p_req()] }, &["high"]),
        // donchian: required period — 20 (Turtle) is common, but so are 10 / 55, so no
        // single dominant default.
        ("donchian", None) => (const { &[p_req()] }, &["high", "low"]),
        ("donchian", Some("upper")) => (const { &[p_req()] }, &["high"]),
        ("donchian", Some("lower")) => (const { &[p_req()] }, &["low"]),
        // hv: window (stddev of log returns — needs >= 2), bar time-frame,
        // annualization trading-days (>= 1: 0/negative puts 0/NaN under sqrt).
        ("hv", None) => (const { &[p2_req(), tf_str("1d"), i64min(252, 1)] }, &["close"]),
        // increase: run length (>= 1), direction (+1 rising / -1 falling).
        ("increase", None) => (const { &[p(1), one_of(1, &[-1, 1])] }, &["close"]),
        // Group A non-TA-Lib indicators (gap report 2026-06-07).
        ("psy", None) => (const { &[p(12)] }, &["close"]),
        ("pvt" | "nvi" | "pvi", None) => (&[], &["close", "volume"]),
        ("dpo", None) => (const { &[p_req()] }, &["close"]),
        ("cmf", None) => (const { &[p(20)] }, &["high", "low", "close", "volume"]),
        ("chop", None) => (const { &[p_req()] }, &["high", "low", "close"]),
        ("kst", None) => (&[], &["close"]),
        ("emv", None) => (const { &[p_req()] }, &["high", "low", "volume"]),
        ("mass_index", None) => (const { &[p(25)] }, &["high", "low"]),
        ("efi", None) => (const { &[p(13)] }, &["close", "volume"]),
        ("tsi", None) => (const { &[p(25), p(13)] }, &["close"]),
        ("crsi", None) => (const { &[p(3), p(2), p(100)] }, &["close"]),
        // Group E formula-equivalent wrappers (China-market names): bias ≡ ppo:1,N,0;
        // dma's DDD line ≡ apo:fast,slow,0, with AMA = the M-period SMA of that line.
        ("bias", None) => (const { &[p(6)] }, &["close"]),
        ("dma", None) => (const { &[p(10), p(50)] }, &["close"]),
        ("dma", Some("ama")) => (const { &[p(10), p(50), p(10)] }, &["close"]),
        // Group B convention-sensitive indicators (gap report §9).
        ("vortex", Some("plus" | "minus")) => (const { &[p(14)] }, &["high", "low", "close"]),
        ("brar", Some("ar")) => (const { &[p(26)] }, &["open", "high", "low"]),
        ("brar", Some("br")) => (const { &[p(26)] }, &["high", "low", "close"]),
        ("vr", None) => (const { &[p(26)] }, &["close", "volume"]),
        // coppock: wma_period, roc_long, roc_short.
        ("coppock", None) => (const { &[p(10), p(14), p(11)] }, &["close"]),
        ("relative_vigor", None | Some("signal")) => {
            (const { &[p(10)] }, &["open", "high", "low", "close"])
        }
        // dkx's DKX line is a fixed 20-period weighted MA (no args); the MADKX signal adds m.
        ("dkx", None) => (&[], &["open", "high", "low", "close"]),
        ("dkx", Some("ma")) => (const { &[p(10)] }, &["open", "high", "low", "close"]),
        ("wvad", None) => (const { &[p(24)] }, &["open", "high", "low", "close", "volume"]),
        // cdp: five intraday levels from the prior bar (no window parameter).
        ("cdp", None | Some("ah" | "nh" | "nl" | "al")) => (&[], &["high", "low", "close"]),
        // mike: six support/resistance lines — all required sub-commands, no primary line.
        ("mike", Some("weakr" | "midr" | "strongr" | "weaks" | "mids" | "strongs")) => {
            (const { &[p_req()] }, &["high", "low", "close"])
        }
        // keltner: middle = EMA(close) (ema_period only); bands add atr_period + multiplier.
        ("keltner", None) => (const { &[p(20)] }, &["close"]),
        ("keltner", Some("upper" | "lower")) => {
            (const { &[p(20), p(10), mult(2.0)] }, &["high", "low", "close"])
        }
        // stoch_momentum: k (HH/LL), d (double-EMA smoothing), signal (EMA of SMI).
        ("stoch_momentum", None | Some("signal")) => {
            (const { &[p(10), p(3), p(3)] }, &["high", "low", "close"])
        }
        // ttm_squeeze: period, Bollinger σ-multiplier, Keltner range-multiplier.
        ("ttm_squeeze", None | Some("on")) => {
            (const { &[p(20), mult(2.0), mult(1.5)] }, &["high", "low", "close"])
        }
        // pivot_points: PP plus the R1/R2/R3 / S1/S2/S3 levels, all from the prior bar.
        ("pivot_points", None | Some("r1" | "r2" | "r3" | "s1" | "s2" | "s3")) => {
            (&[], &["high", "low", "close"])
        }
        // ichimoku: tenkan / kijun / senkou_b periods (the displacement is the kijun period).
        ("ichimoku", Some("tenkan" | "kijun" | "senkou_a" | "senkou_b" | "chikou")) => {
            (const { &[p(9), p(26), p(52)] }, &["high", "low", "close"])
        }
        // wad: cumulative, no parameters. asi: Wilder's limit-move scaling `t` —
        // the formula divides by it, so it must be strictly positive.
        ("wad", None) => (&[], &["high", "low", "close"]),
        ("asi", None) => (const { &[fgt(3.0, 0.0)] }, &["open", "high", "low", "close"]),
        // supertrend: ATR period + band multiplier (shared by the line and direction).
        ("supertrend", None | Some("direction")) => {
            (const { &[p(10), mult(3.0)] }, &["high", "low", "close"])
        }
        ("style", Some("bullish" | "bearish")) => (&[], &["open", "close"]),
        // Candlestick patterns (style.<pattern> / cdl.<pattern>) — validated against the
        // compute layer's pattern registry, so new patterns need no change here. Patterns
        // taking a `penetration` ratio accept one optional Float arg (default 0.5,
        // non-negative — TA-Lib's domain).
        ("style", Some(pat)) => {
            // One registry lookup. This was a `.is_some()` guard plus a `.unwrap()`
            // body — two 61-arm pattern matches on every candle validate; `?` makes an
            // unknown pattern a `None` spec (a "no such sub-command" error), exactly as
            // the guard did. `bullish`/`bearish` are handled by the arm above.
            let (pattern, _) = volas_compute::raw::indicators::candle_pattern(pat)?;
            let args = match pattern {
                volas_compute::raw::indicators::CandlePattern::Penetration { default, .. } => {
                    vec![fmin(default, 0.0)]
                }
                volas_compute::raw::indicators::CandlePattern::Plain(_) => vec![],
            };
            return Some(CommandSpec {
                args: Cow::Owned(args),
                series: &["open", "high", "low", "close"],
            });
        }
        // repeat: consecutive-True run length (>= 1; 0 underflowed the kernel).
        ("repeat", None) => (const { &[p(1)] }, &[]),
        // change: % change over a window spanning `period` bars — `period` 2
        // compares with the previous bar; 1 compares a bar with itself
        // (identically 0), 0 underflowed the kernel, so the domain is >= 2.
        ("change", None) => (const { &[p2(2)] }, &["close"]),
        ("mom" | "roc" | "rocp" | "rocr" | "rocr100", None) => (const { &[p_req()] }, &["close"]),
        ("midpoint", None) => (const { &[p_req()] }, &["close"]),
        ("midprice", None) => (const { &[p_req()] }, &["high", "low"]),
        ("willr", None) => (const { &[p_req()] }, &["high", "low", "close"]),
        ("cmo", None) => (const { &[p_req()] }, &["close"]),
        // cci: window >= 2 — at 1 the numerator (TP - SMA(TP,1)) is identically
        // zero, a constant no-signal column.
        ("cci", None) => (const { &[p2_req()] }, &["high", "low", "close"]),
        ("imi", None) => (const { &[p_req()] }, &["open", "close"]),
        ("trix", None) => (const { &[p_req()] }, &["close"]),
        ("mfi", None) => (const { &[p_req()] }, &["high", "low", "close", "volume"]),
        ("ultosc", None) => (const { &[p(7), p(14), p(28)] }, &["high", "low", "close"]),
        // stoch: fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype.
        ("stoch", Some("k" | "d")) => (
            const { &[p(5), p(3), matype(), p(3), matype()] },
            &["high", "low", "close"],
        ),
        // stochf: fastk_period, fastd_period, fastd_matype.
        ("stochf", Some("k" | "d")) => (const { &[p(5), p(3), matype()] }, &["high", "low", "close"]),
        // stochrsi: rsi_period, fastk_period, fastd_period, fastd_matype.
        ("stochrsi", Some("k" | "d")) => (const { &[p(14), p(5), p(3), matype()] }, &["close"]),
        ("plus_dm" | "minus_dm", None) => (const { &[p_req()] }, &["high", "low"]),
        ("plus_di" | "minus_di" | "dx" | "adx" | "adxr", None) => {
            (const { &[p_req()] }, &["high", "low", "close"])
        }
        // aroon / aroonosc: window >= 2 — at 1 the lines are identically 100
        // (the extreme is always "this bar"), a constant no-signal column.
        ("aroon", Some("up" | "down")) => (const { &[p2_req()] }, &["high", "low"]),
        ("aroonosc", None) => (const { &[p2_req()] }, &["high", "low"]),
        ("sum" | "maxindex" | "minindex", None) => (const { &[p_req()] }, &["close"]),
        ("minmax" | "minmaxindex", Some("min" | "max")) => (const { &[p_req()] }, &["close"]),
        ("natr", None) => (const { &[p_req()] }, &["high", "low", "close"]),
        ("bop", None) => (&[], &["open", "high", "low", "close"]),
        // Linear-regression family: window >= 2 (the slope denominator
        // Σ(x - x̄)² is zero on a single sample — all-NaN by construction).
        (
            "linearreg" | "linearreg_slope" | "linearreg_intercept" | "linearreg_angle" | "tsf",
            None,
        ) => (const { &[p2_req()] }, &["close"]),
        // correl / beta: correlation / regression over >= 2 samples (the
        // single-sample variance is zero, so the statistic is 0/0).
        // First series defaults to close; the second is required (no spec default).
        ("correl", None) => (const { &[p2_req()] }, &["close"]),
        ("beta", None) => (const { &[p2_req()] }, &["close"]),
        // var / stddev: window >= 2 — the (population) variance of one sample is
        // identically zero, a constant no-signal column. (TA-Lib allows var:1;
        // volas rejects it on the V17 no-signal principle.) stddev's second arg
        // is the sign-free deviation multiplier.
        ("var", None) => (const { &[p2_req()] }, &["close"]),
        // pandas-window statistics promoted to directives (single kernel source
        // with the rolling API; full-window semantics — an NA in the window
        // yields NA, like the TA family's warm-up discipline).
        ("median", None) => (const { &[p2_req()] }, &["close"]),
        // quantile: window, then the quantile level in [0, 1].
        ("quantile", None) => (const { &[p2_req(), frange(0.5, 0.0, 1.0)] }, &["close"]),
        ("rank", None) => (const { &[p2_req()] }, &["close"]),
        // skew / kurt are undefined below 3 / 4 samples (V17: no all-NA columns).
        ("skew", None) => (
            const { &[Arg { default: ArgDefault::Required, bound: ArgBound::IntMin(3) }] },
            &["close"],
        ),
        ("kurt", None) => (
            const { &[Arg { default: ArgDefault::Required, bound: ArgBound::IntMin(4) }] },
            &["close"],
        ),
        ("sem", None) => (const { &[p2_req()] }, &["close"]),
        // TradingView oscillators / dispersion. cog / rci / dev / mode follow the
        // windowed-stat family (period required, >= 2). iii is pointwise (no arg);
        // kcw mirrors the keltner band parameters.
        ("cog", None) => (const { &[p2_req()] }, &["close"]),
        ("rci", None) => (const { &[p2_req()] }, &["close"]),
        ("dev", None) => (const { &[p2_req()] }, &["close"]),
        ("mode", None) => (const { &[p2_req()] }, &["close"]),
        ("iii", None) => (&[], &["high", "low", "close", "volume"]),
        ("kcw", None) => (const { &[p(20), p(10), mult(2.0)] }, &["high", "low", "close"]),
        // fractal pivots: leftbars, rightbars (both required, >= 1).
        ("pivothigh", None) => (const { &[p_req(), p_req()] }, &["high"]),
        ("pivotlow", None) => (const { &[p_req(), p_req()] }, &["low"]),
        ("stddev", None) => (const { &[p2_req(), mult(1.0)] }, &["close"]),
        ("obv", None) => (&[], &["close", "volume"]),
        ("ad", None) => (&[], &["high", "low", "close", "volume"]),
        // adosc: fast EMA period, slow EMA period.
        ("adosc", None) => (
            const { &[p(3), p(10)] },
            &["high", "low", "close", "volume"],
        ),
        ("avgprice", None) => (&[], &["open", "high", "low", "close"]),
        ("medprice", None) => (&[], &["high", "low"]),
        ("typprice", None) => (&[], &["high", "low", "close"]),
        ("wclprice", None) => (&[], &["high", "low", "close"]),
        // Hilbert-transform cycle suite: a single price series, no numeric args
        // (MAMA takes fast/slow limits). Multi-output lines share the base spec.
        ("ht_dcperiod" | "ht_dcphase" | "ht_trendline" | "ht_trendmode", None) => {
            (&[], &["close"])
        }
        ("ht_phasor", None | Some("quadrature")) => (&[], &["close"]),
        ("ht_sine", None | Some("leadsine")) => (&[], &["close"]),
        // mama: fast limit, slow limit — TA-Lib's domain [0.01, 0.99]; outside
        // it the adaptive alpha degenerates and the output is all-NaN.
        ("mama", None | Some("fama")) => (
            const { &[frange(0.5, 0.01, 0.99), frange(0.05, 0.01, 0.99)] },
            &["close"],
        ),
        _ => return None,
    };
    Some(CommandSpec { args: Cow::Borrowed(args), series })
}

/// Commands whose value at a row depends on every row before it — EMA / Wilder
/// smoothing, cumulative lines, the Hilbert transform, SAR, … — rather than on a fixed
/// trailing window. (A command with a TA-Lib MA-type argument is recursive when that
/// argument selects a recursive average; see [`is_recursive`].) A `match`, not a list
/// scan: every live resume asks.
fn is_recursive_name(name: &str) -> bool {
    matches!(
        name,
        "ema" | "smma" | "dema" | "tema" | "t3" | "kama" | "macd" | "macdfix" | "kdj"
            | "rsi" | "crsi" | "stochrsi" | "cmo" | "trix" | "pvt" | "nvi" | "pvi" | "obv"
            | "ad" | "adosc" | "mass_index" | "efi" | "tsi" | "keltner" | "kcw"
            | "stoch_momentum" | "ttm_squeeze" | "wad" | "asi" | "supertrend" | "atr"
            | "natr" | "plus_dm" | "minus_dm" | "plus_di" | "minus_di" | "dx" | "adx"
            | "adxr" | "sar" | "sarext" | "ht_dcperiod" | "ht_dcphase" | "ht_phasor"
            | "ht_sine" | "ht_trendline" | "ht_trendmode" | "mama"
    )
}

/// Commands whose output is an absolute row position, so it changes with where the
/// series starts.
const ABSOLUTE_POSITION: &[&str] = &["maxindex", "minindex", "minmaxindex"];

/// Whether a TA-Lib MA-type code selects a recursive moving average (EMA, DEMA, TEMA,
/// KAMA, MAMA, T3 — not SMA, WMA or TRIMA).
pub fn is_recursive_matype(matype: usize) -> bool {
    matches!(matype, 1 | 3 | 4 | 6 | 7 | 8)
}

/// Whether a bound command is recursive: its value at a row depends on every row
/// before it. A missing row of a recursive command's inputs does not exist for it —
/// it runs on the complete rows only (see `volas_compute::missing`).
pub fn is_recursive(name: &str, args: &[ArgValue]) -> bool {
    let matypes: &[usize] = match name {
        "ma" => &[1],
        "apo" | "ppo" | "mavp" | "stochf" => &[2],
        "macdext" => &[1, 3, 5],
        "stoch" => &[2, 4],
        _ => return is_recursive_name(name),
    };
    matypes
        .iter()
        .filter_map(|&i| args.get(i))
        .any(|a| is_recursive_matype(a.as_usize()))
}

/// Whether a command's output is an absolute row position.
pub fn emits_position(name: &str) -> bool {
    ABSOLUTE_POSITION.contains(&name)
}

/// Whether a finite-window command's full computation carries a running accumulator
/// across the series (a sliding sum, sum of squares, weighted sum, …): its rounding
/// depends on every row before, so a refresh resumes it from the kernel's state rather
/// than recomputing its last window. `sub` is the canonical sub-command (`stochf.k` is a
/// pure window; `stochf.d` averages it).
pub fn accumulates(name: &str, sub: Option<&str>) -> bool {
    match name {
        "stochf" => sub == Some("d"),
        _ => matches!(
            name,
            "accbands" | "apo" | "bbi" | "bbw" | "beta" | "bias" | "boll" | "cci" | "cmf"
                | "cog" | "coppock" | "correl" | "dkx" | "dma" | "dpo" | "emv" | "hma" | "hv"
                | "kst" | "linearreg" | "linearreg_angle" | "linearreg_intercept"
                | "linearreg_slope" | "ma" | "macdext" | "mavp" | "mfi" | "ppo"
                | "relative_vigor" | "stddev" | "stoch" | "sum" | "trima" | "tsf" | "var"
                | "vr" | "vwma" | "wma" | "wvad"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recursive command never emits a row position: the missing-row rule maps its
    /// outputs back by row, which a position-valued output would not survive.
    #[test]
    fn no_recursive_command_emits_a_position() {
        let recursive = COMMANDS.iter().filter(|c| is_recursive_name(c));
        assert_eq!(recursive.clone().count(), 48);
        assert!(recursive.into_iter().all(|c| !emits_position(c)));
        for name in ["ma", "apo", "ppo", "mavp", "stochf", "macdext", "stoch"] {
            assert!(!emits_position(name));
        }
    }

    /// Each argument constructor pairs its default with its kind's bound. (The table
    /// evaluates them at compile time; this checks the pairing itself.)
    #[test]
    fn argument_constructors_pair_default_and_bound() {
        use ArgBound as B;
        use ArgDefault as D;
        let arg = |default, bound| Arg { default, bound };
        assert_eq!(p(3), arg(D::Int(3), B::IntMin(1)));
        assert_eq!(p_req(), arg(D::Required, B::IntMin(1)));
        assert_eq!(p2(4), arg(D::Int(4), B::IntMin(2)));
        assert_eq!(p2_req(), arg(D::Required, B::IntMin(2)));
        assert_eq!(matype(), arg(D::Int(0), B::IntRange(0, 8)));
        assert_eq!(prange_req(), arg(D::Required, B::IntRange(1, 100_000)));
        assert_eq!(mult(2.0), arg(D::Float(2.0), B::Finite));
        assert_eq!(fmin(0.5, 0.0), arg(D::Float(0.5), B::FloatMin(0.0)));
        assert_eq!(fgt(3.0, 0.0), arg(D::Float(3.0), B::FloatGt(0.0)));
        assert_eq!(frange(0.7, 0.0, 1.0), arg(D::Float(0.7), B::FloatRange(0.0, 1.0)));
        assert_eq!(i64min(252, 1), arg(D::I64(252), B::I64Min(1)));
        assert_eq!(one_of(1, &[-1, 1]), arg(D::I64(1), B::OneOfI64(&[-1, 1])));
        assert_eq!(tf_str("1d"), arg(D::Str("1d"), B::AnyStr));
    }

    #[test]
    fn arg_default_value_per_kind() {
        assert_eq!(ArgDefault::Required.value(), None);
        assert_eq!(ArgDefault::Int(12).value(), Some(ArgValue::Usize(12)));
        assert_eq!(ArgDefault::Float(2.5).value(), Some(ArgValue::F64(2.5)));
        assert_eq!(ArgDefault::I64(-1).value(), Some(ArgValue::I64(-1)));
        assert_eq!(
            ArgDefault::Str("close").value(),
            Some(ArgValue::Str("close".into()))
        );
    }

    #[test]
    fn arg_bound_bind_per_kind() {
        // Each kind parses once into its typed value, rejecting out-of-domain tokens.
        // IntMin: the period rule (>= 1) and the second-moment rule (>= 2).
        assert_eq!(ArgBound::IntMin(1).bind("1"), Ok(ArgValue::Usize(1)));
        assert!(ArgBound::IntMin(1).bind("0").is_err());
        assert!(ArgBound::IntMin(1).bind("-3").is_err()); // not a usize
        assert!(ArgBound::IntMin(1).bind("abc").is_err());
        assert!(ArgBound::IntMin(2).bind("2").is_ok());
        assert!(ArgBound::IntMin(2).bind("1").is_err());
        // IntRange: matype selector / mavp period ceiling.
        assert_eq!(ArgBound::IntRange(0, 8).bind("0"), Ok(ArgValue::Usize(0)));
        assert!(ArgBound::IntRange(0, 8).bind("8").is_ok());
        assert!(ArgBound::IntRange(0, 8).bind("9").is_err());
        // OneOfI64: increase's ±1 direction.
        assert_eq!(ArgBound::OneOfI64(&[-1, 1]).bind("-1"), Ok(ArgValue::I64(-1)));
        assert!(ArgBound::OneOfI64(&[-1, 1]).bind("0").is_err());
        assert!(ArgBound::OneOfI64(&[-1, 1]).bind("x").is_err());
        // I64Min: hv trading days.
        assert_eq!(ArgBound::I64Min(1).bind("252"), Ok(ArgValue::I64(252)));
        assert!(ArgBound::I64Min(1).bind("0").is_err());
        // Finite: multipliers take any sign but never NaN / inf.
        assert_eq!(ArgBound::Finite.bind("-2.5"), Ok(ArgValue::F64(-2.5)));
        assert!(ArgBound::Finite.bind("nan").is_err());
        assert!(ArgBound::Finite.bind("inf").is_err());
        // FloatMin / FloatGt: SAR acceleration (>= 0) vs ASI divisor (> 0).
        assert_eq!(ArgBound::FloatMin(0.0).bind("0"), Ok(ArgValue::F64(0.0)));
        assert!(ArgBound::FloatMin(0.0).bind("-0.1").is_err());
        assert!(ArgBound::FloatGt(0.0).bind("0.5").is_ok());
        assert!(ArgBound::FloatGt(0.0).bind("0").is_err());
        // FloatRange: T3 vfactor / KDJ seed / MAMA limits.
        assert_eq!(ArgBound::FloatRange(0.0, 1.0).bind("0.7"), Ok(ArgValue::F64(0.7)));
        assert!(ArgBound::FloatRange(0.0, 1.0).bind("1.5").is_err());
        assert!(ArgBound::FloatRange(0.0, 100.0).bind("150").is_err());
        // AnyStr: free-form (validated downstream).
        assert_eq!(ArgBound::AnyStr.bind("1d"), Ok(ArgValue::Str("1d".into())));
    }
}
