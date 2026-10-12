//! Per-command metadata — the one owner of what a command line is: canonical
//! sub-command names, positional-argument defaults **and validation bounds**, operand
//! slots, and each line's structural facts ([`CommandSpec`]: history class, lookback,
//! row-position output, resume reach and coverage, bare usability). `bind` validates
//! against it once and stores the spec in the bound command, so `exec`, the resume
//! dispatch, `lookback` and `stringify` read the same facts and never re-derive them.

use std::borrow::Cow;

use volas_time::TimeFrame;

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
    /// Time-frame default (`hv`'s bar time frame).
    TimeFrame(TimeFrame),
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
            ArgDefault::TimeFrame(tf) => Some(ArgValue::TimeFrame(*tf)),
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
    /// A bar time frame: one of the [`TimeFrame`] labels (`1m`, `15m`, `1h`, `1d`,
    /// `1w`, `1M`, …), parsed by the time-frame owner [`TimeFrame::from_label`].
    TimeFrame,
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
            ArgBound::TimeFrame => TimeFrame::from_label(s)
                .map(ArgValue::TimeFrame)
                .map_err(|_| "must be a time-frame label such as 1m / 15m / 1h / 1d / 1w / 1M".to_string()),
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

/// Bar time-frame argument.
const fn time_frame(tf: TimeFrame) -> Arg {
    Arg { default: ArgDefault::TimeFrame(tf), bound: ArgBound::TimeFrame }
}

/// `hv`'s default trading days per year (the US-equity convention).
pub(crate) const HV_TRADING_DAYS: i64 = 252;

/// `hv`'s default minutes of trading per day (a 24 h market).
pub(crate) const HV_SESSION_MINUTES: usize = 1440;

/// Minutes of regular trading per trading day: 1..=1440 (a day has 1440 minutes).
const fn session_minutes(n: usize) -> Arg {
    Arg { default: ArgDefault::Int(n), bound: ArgBound::IntRange(1, 1440) }
}

/// How a command's value at a row depends on the rows before it — the history class,
/// one structural fact per bound output line, read by every consumer: the missing-row
/// rule, the refresh's window recompute, resume dispatch and slice continuation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum History {
    /// A function of a fixed trailing window of its inputs whose computation carries
    /// nothing across windows: recomputing the window is exact.
    Window,
    /// A finite window whose full computation carries a running accumulator across the
    /// series (a sliding sum, sum of squares, weighted sum, …): a refresh resumes it from
    /// the kernel's state rather than recomputing its last window.
    Accumulating,
    /// Each value depends on every row before it (EMA / Wilder smoothing, cumulative
    /// lines, the Hilbert transform, SAR, …): it runs on the complete rows of its inputs
    /// (see `volas_compute::missing`).
    Recursive,
}

/// How a class-3 line's history is kept by a frame that does not hold all of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    /// By the state its resume continues from: a checkpoint of that state replays it.
    State,
    /// By its window: its values are a function of a trailing window of its rows.
    Window,
    /// Not at all: it carries no state (it recomputes), so only the whole history gives it.
    Nothing,
}

/// A line's R13 history class (how its value depends on the rows before it): class 1
/// depends on a trailing window, class 2 on a running accumulator over a trailing window,
/// class 3 on every row before — a recursion, a cumulative line or an unbounded count,
/// with how it is kept. A bounded window computes a class-1 directive on request and the
/// others only when declared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoryClass {
    /// Class 1.
    Window,
    /// Class 2.
    Accumulator,
    /// Class 3.
    Recursion(Keep),
}

/// What continuing a cached column after new rows runs, beyond a recompute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resume {
    /// No carried state: a refresh recomputes (a trailing window, or the column).
    Recompute,
    /// A carried state continued over the new rows at once (no single-row step).
    Rows,
    /// A carried state continued one row at a time, with no allocation (the live path),
    /// or over several rows at once.
    Step,
}

/// A command's history rule: fixed, or decided by the generic moving averages it is
/// built of.
#[derive(Clone, Copy, Debug)]
enum HistoryRule {
    Fixed(History),
    /// `(period, matype)` argument positions of the generic moving averages this line
    /// runs: recursive when any selects a recursive average, else the fixed class.
    Averages(&'static [(usize, usize)], History),
}

/// A command line's lookback (warm-up rows) from its bound arguments.
pub type Lookback = fn(&[ArgValue]) -> usize;

/// The command a line runs as under its bound arguments, with that command's own
/// arguments (`None` when it runs as itself).
pub type RunsAs = fn(&[ArgValue]) -> Option<(&'static str, Vec<ArgValue>)>;

/// Where a line's lookback comes from: its arguments, or a fixed count (a candlestick
/// pattern's average period, known when its spec is built).
#[derive(Clone, Copy, Debug)]
enum LookbackRule {
    Of(Lookback),
    Fixed(usize),
}

/// The metadata of one output line — a command and its canonical sub-command: its
/// argument domains and defaults, its operand slots, and the structural facts every
/// consumer reads from here alone — the history class, the lookback, whether its output
/// is a row position, how far a resume reads, which resume it has, and whether it can be
/// written bare. Built by [`command_spec`]; a bound command carries its own.
#[derive(Clone, Debug)]
pub struct CommandSpec {
    /// Per-position argument defaults + validation bounds.
    pub args: Cow<'static, [Arg]>,
    /// The `@` operand slots: each one's default column, or `""` for a required operand
    /// (`correl` / `beta` / `mavp`'s second series, `repeat`'s condition). A command
    /// takes at most this many operands.
    pub series: &'static [&'static str],
    history: HistoryRule,
    lookback: LookbackRule,
    emits_position: bool,
    reads_past: bool,
    /// A count over its window (`increase`): class 3 by R13, kept by its window.
    counted: bool,
    resume: Resume,
    /// Arguments under which the declared resume is not available (falls back to
    /// [`Resume::Recompute`]).
    resume_unless: Option<fn(&[ArgValue]) -> bool>,
    /// The recursive command a line runs as under the given arguments, with its
    /// arguments — `ma` at a recursive MA type is exactly that average's command.
    runs_as: Option<RunsAs>,
}

impl CommandSpec {
    /// A window line over `args` / `series` with no warm-up and no carried state; the
    /// table refines it.
    pub(crate) fn line(args: &'static [Arg], series: &'static [&'static str]) -> CommandSpec {
        CommandSpec::owned(Cow::Borrowed(args), series)
    }

    pub(crate) fn owned(args: Cow<'static, [Arg]>, series: &'static [&'static str]) -> CommandSpec {
        CommandSpec {
            args,
            series,
            history: HistoryRule::Fixed(History::Window),
            lookback: LookbackRule::Fixed(0),
            emits_position: false,
            reads_past: false,
            counted: false,
            resume: Resume::Recompute,
            resume_unless: None,
            runs_as: None,
        }
    }

    /// Its lookback.
    pub(crate) fn lookback(mut self, f: Lookback) -> Self {
        self.lookback = LookbackRule::Of(f);
        self
    }

    /// A fixed lookback.
    pub(crate) fn fixed_lookback(mut self, rows: usize) -> Self {
        self.lookback = LookbackRule::Fixed(rows);
        self
    }

    /// A finite window carried by a running accumulator.
    pub(crate) fn accumulating(mut self) -> Self {
        self.history = HistoryRule::Fixed(History::Accumulating);
        self
    }

    /// Recursive whatever its arguments.
    pub(crate) fn recursive(mut self) -> Self {
        self.history = HistoryRule::Fixed(History::Recursive);
        self
    }

    /// Built of generic moving averages at `slots`: recursive when one selects a
    /// recursive average, else accumulating.
    pub(crate) fn averages(mut self, slots: &'static [(usize, usize)]) -> Self {
        self.history = HistoryRule::Averages(slots, History::Accumulating);
        self
    }

    /// Its output is an absolute row position.
    pub(crate) fn position(mut self) -> Self {
        self.emits_position = true;
        self
    }

    /// Continuing it reads one row more than its lookback (KAMA's volatility sum, PSY's
    /// count of up days, the extremum-position window, SAR's bar before the previous
    /// one) — an accumulating line always does (the row its window drops).
    pub(crate) fn reads_past_lookback(mut self) -> Self {
        self.reads_past = true;
        self
    }

    /// A count over its window: class 3, kept by its window.
    pub(crate) fn counted(mut self) -> Self {
        self.counted = true;
        self
    }

    /// It has a single-row step.
    pub(crate) fn step(mut self) -> Self {
        self.resume = Resume::Step;
        self
    }

    /// It continues over new rows at once only.
    pub(crate) fn rows(mut self) -> Self {
        self.resume = Resume::Rows;
        self
    }

    /// Its resume is not available under the arguments `unless` accepts.
    pub(crate) fn resume_unless(mut self, unless: fn(&[ArgValue]) -> bool) -> Self {
        self.resume_unless = Some(unless);
        self
    }

    /// Under some arguments it runs as another (recursive) command.
    pub(crate) fn runs_as(mut self, f: RunsAs) -> Self {
        self.runs_as = Some(f);
        self
    }

    /// The history class under the bound arguments `args`.
    pub fn history(&self, args: &[ArgValue]) -> History {
        match self.history {
            HistoryRule::Fixed(h) => h,
            HistoryRule::Averages(slots, otherwise) => {
                let recursive = slots.iter().any(|&(p, m)| {
                    volas_compute::raw::indicators::is_recursive_generic_ma(args[p].as_usize(), args[m].as_usize())
                });
                if recursive { History::Recursive } else { otherwise }
            }
        }
    }

    /// The R13 history class under the bound arguments `args` ([`HistoryClass`]): a
    /// count over its window is class 3 kept by its window; a recursion is class 3, kept
    /// by its state when it has a resume, else not at all; a running accumulator class 2;
    /// a pure window class 1.
    pub fn history_class(&self, args: &[ArgValue]) -> HistoryClass {
        if self.counted {
            return HistoryClass::Recursion(Keep::Window);
        }
        match self.history(args) {
            History::Recursive if self.resume(args) == Resume::Recompute => HistoryClass::Recursion(Keep::Nothing),
            History::Recursive => HistoryClass::Recursion(Keep::State),
            History::Accumulating => HistoryClass::Accumulator,
            History::Window => HistoryClass::Window,
        }
    }

    /// Whether the line is, by kind, a running accumulator — whatever its arguments: an
    /// accumulating line, or one of generic averages (even at a recursive type).
    pub fn accumulates(&self) -> bool {
        matches!(
            self.history,
            HistoryRule::Fixed(History::Accumulating) | HistoryRule::Averages(_, History::Accumulating)
        )
    }

    /// The lookback under the bound arguments `args`: the rows before its first value.
    pub fn lookback_of(&self, args: &[ArgValue]) -> usize {
        match self.lookback {
            LookbackRule::Of(f) => f(args),
            LookbackRule::Fixed(rows) => rows,
        }
    }

    /// Whether its output is an absolute row position (`maxindex` / `minindex` /
    /// `minmaxindex`), so it changes with where the series starts.
    pub fn emits_position(&self) -> bool {
        self.emits_position
    }

    /// Whether continuing it reads one row more than its lookback.
    pub fn reads_past(&self) -> bool {
        self.reads_past || self.accumulates()
    }

    /// The resume it has under the bound arguments `args`.
    pub fn resume(&self, args: &[ArgValue]) -> Resume {
        if let Some((name, args)) = self.runs_as_command(args) {
            return command_spec(name, None).map_or(Resume::Recompute, |spec| spec.resume(&args));
        }
        let averaged_recursion =
            matches!(self.history, HistoryRule::Averages(..)) && self.history(args) == History::Recursive;
        if averaged_recursion || self.resume_unless.is_some_and(|f| f(args)) {
            return Resume::Recompute;
        }
        self.resume
    }

    /// The recursive command this line runs as under `args`, with its arguments.
    pub fn runs_as_command(&self, args: &[ArgValue]) -> Option<(&'static str, Vec<ArgValue>)> {
        self.runs_as.and_then(|f| f(args))
    }

    /// Whether the line can be written as a bare name (no colon, no arguments): every
    /// argument has a default, and every input a default column.
    pub fn usable_bare(&self) -> bool {
        self.args.iter().all(|a| a.default != ArgDefault::Required)
            && !self.series.is_empty()
            && self.series.iter().all(|s| !s.is_empty())
    }
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

mod table;

pub use table::command_spec;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every output line: each command with no sub-command and with every sub-command
    /// token the table names.
    fn every_line() -> Vec<(String, Option<String>, CommandSpec)> {
        let src = include_str!("table.rs");
        let mut tokens: Vec<String> = Vec::new();
        for part in src.split('"').skip(1).step_by(2) {
            if !part.is_empty() && part.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_') {
                tokens.push(part.to_string());
            }
        }
        let mut out = Vec::new();
        for &c in COMMANDS {
            for sub in std::iter::once(None).chain(tokens.iter().map(|t| Some(t.clone()))) {
                if let Some(spec) = command_spec(c, sub.as_deref()) {
                    if !out.iter().any(|(n, s, _): &(String, Option<String>, CommandSpec)| n == c && *s == sub) {
                        out.push((c.to_string(), sub, spec));
                    }
                }
            }
        }
        out
    }

    /// The bound default arguments of a line (a required argument at its bound's
    /// minimum).
    fn defaults(spec: &CommandSpec) -> Vec<ArgValue> {
        spec.args
            .iter()
            .map(|a| {
                a.default.value().unwrap_or_else(|| match a.bound {
                    ArgBound::IntMin(m) | ArgBound::IntRange(m, _) => ArgValue::Usize(m.max(2)),
                    _ => unreachable!("every required argument is an integer"), // LCOV_EXCL_LINE
                })
            })
            .collect()
    }

    /// A recursive line never emits a row position: the missing-row rule maps its
    /// outputs back by row, which a position-valued output would not survive. And every
    /// line's facts hold together: a position output, a resume that reads past the
    /// lookback, a resume of the class it declares.
    #[test]
    fn the_line_facts_hold_together() {
        let lines = every_line();
        assert!(lines.len() > 150, "{}", lines.len());
        let mut recursive = 0;
        for (name, sub, spec) in &lines {
            let args = defaults(spec);
            let history = spec.history(&args);
            recursive += usize::from(history == History::Recursive);
            assert!(!(history == History::Recursive && spec.emits_position()), "{name}.{sub:?}");
            if spec.accumulates() || spec.emits_position() {
                assert!(spec.reads_past(), "{name}.{sub:?}");
            }
            // a line that steps is recursive or carries a running state, or a formula of
            // its own row (the stateless price transforms and the momentum family)
            let _ = spec.lookback_of(&args);
            if spec.resume(&args) == Resume::Rows {
                assert!(history == History::Recursive || spec.emits_position(), "{name}.{sub:?}");
            }
        }
        assert!(recursive >= 47, "{recursive}");
    }

    /// The history class is the bound sub-line's: `stochf.k` is a pure window whatever
    /// its unused D-line type, `stoch.k` reads only its %K smoothing, and a generic
    /// average decides by its own (period, type).
    #[test]
    fn the_history_class_is_the_lines() {
        let u = ArgValue::Usize;
        let k = command_spec("stochf", Some("k")).unwrap();
        assert_eq!(k.history(&[u(5), u(3), u(1)]), History::Window);
        let d = command_spec("stochf", Some("d")).unwrap();
        assert_eq!(d.history(&[u(5), u(3), u(1)]), History::Recursive);
        assert_eq!(d.history(&[u(5), u(3), u(0)]), History::Accumulating);
        let sk = command_spec("stoch", Some("k")).unwrap();
        assert_eq!(sk.history(&[u(5), u(3), u(0), u(3), u(1)]), History::Accumulating);
        let sd = command_spec("stoch", Some("d")).unwrap();
        assert_eq!(sd.history(&[u(5), u(3), u(0), u(3), u(1)]), History::Recursive);
        let ma = command_spec("ma", None).unwrap();
        assert_eq!(ma.history(&[u(5), u(1)]), History::Recursive);
        assert_eq!(ma.history(&[u(1), u(1)]), History::Accumulating); // period 1: the identity
        assert_eq!(ma.resume(&[u(5), u(7)]), Resume::Rows); // MAMA continues over rows
        assert_eq!(ma.resume(&[u(5), u(1)]), Resume::Step); // EMA steps
        assert_eq!(ma.runs_as_command(&[u(5), u(0)]), None);
        let apo = command_spec("apo", None).unwrap();
        assert_eq!(apo.resume(&[u(2), u(3), u(1)]), Resume::Recompute);
        let srd = command_spec("stochrsi", Some("d")).unwrap();
        assert_eq!(srd.resume(&[u(14), u(5), u(3), u(0)]), Resume::Rows);
        assert_eq!(srd.resume(&[u(14), u(5), u(3), u(2)]), Resume::Recompute);
    }

    /// A bare name is usable when every argument and every input has a default.
    #[test]
    fn bare_usability() {
        assert!(command_spec("obv", None).unwrap().usable_bare());
        assert!(command_spec("boll", None).unwrap().usable_bare());
        assert!(!command_spec("ema", None).unwrap().usable_bare()); // required period
        assert!(!command_spec("repeat", None).unwrap().usable_bare()); // required operand
        assert!(!command_spec("correl", None).unwrap().usable_bare());
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
        assert_eq!(time_frame(TimeFrame::Day1), arg(D::TimeFrame(TimeFrame::Day1), B::TimeFrame));
        assert_eq!(session_minutes(1440), arg(D::Int(1440), B::IntRange(1, 1440)));
    }

    #[test]
    fn arg_default_value_per_kind() {
        assert_eq!(ArgDefault::Required.value(), None);
        assert_eq!(ArgDefault::Int(12).value(), Some(ArgValue::Usize(12)));
        assert_eq!(ArgDefault::Float(2.5).value(), Some(ArgValue::F64(2.5)));
        assert_eq!(ArgDefault::I64(-1).value(), Some(ArgValue::I64(-1)));
        assert_eq!(
            ArgDefault::TimeFrame(TimeFrame::Min15).value(),
            Some(ArgValue::TimeFrame(TimeFrame::Min15))
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
        // TimeFrame: the TimeFrame labels only (no rounding of seconds, no panic on a
        // multibyte character).
        assert_eq!(ArgBound::TimeFrame.bind("1d"), Ok(ArgValue::TimeFrame(TimeFrame::Day1)));
        assert_eq!(ArgBound::TimeFrame.bind("1W"), Ok(ArgValue::TimeFrame(TimeFrame::Week1)));
        for bad in ["30s", "10m", "0m", "5x", "xm", "1\u{e9}", ""] {
            assert!(ArgBound::TimeFrame.bind(bad).is_err(), "{bad:?}");
        }
    }
}
