//! Per-command metadata: canonical sub-command names, positional-argument
//! defaults, and default series. This is the single source of command defaults,
//! shared by `exec` (runtime defaulting), `lookback`, and `stringify`, so the
//! values are never duplicated.

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
    /// The default as an integer, if it is one (used by `exec` / `lookback`).
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            ArgDefault::Int(n) => Some(*n),
            _ => None,
        }
    }

    /// The default rendered as a canonical string (used by `stringify` to decide
    /// whether a supplied argument equals its default).
    pub fn to_token(&self) -> Option<String> {
        match self {
            ArgDefault::Required => None,
            ArgDefault::Int(n) => Some(n.to_string()),
            ArgDefault::Float(f) => Some(format_float(*f)),
            ArgDefault::I64(n) => Some(n.to_string()),
            ArgDefault::Str(s) => Some(s.to_string()),
        }
    }
}

/// Render a float argument canonically (`2.0` -> "2", `2.5` -> "2.5") so it
/// compares equal to an integer-looking supplied argument.
fn format_float(f: f64) -> String {
    if f.fract() == 0.0 {
        format!("{}", f as i64)
    } else {
        format!("{f}")
    }
}

/// The metadata for one command (after sub-command canonicalization).
pub struct CommandSpec {
    /// Per-position argument defaults.
    pub args: Vec<ArgDefault>,
    /// Default series column for each `@` slot.
    pub series: Vec<&'static str>,
}

/// Canonicalize a sub-command alias (e.g. `macd.dif` -> main, `boll.u` ->
/// `upper`). Returns the canonical sub, or `None` for the main command.
pub fn canon_sub(name: &str, sub: Option<&str>) -> Option<String> {
    match (name, sub) {
        ("macd", None | Some("dif")) => None,
        ("macd", Some("s" | "signal" | "dea")) => Some("signal".into()),
        ("macd", Some("h" | "histogram" | "macd")) => Some("histogram".into()),
        ("boll" | "donchian", None | Some("middle")) => None,
        ("boll" | "donchian", Some("u" | "upper")) => Some("upper".into()),
        ("boll" | "donchian", Some("l" | "lower")) => Some("lower".into()),
        (_, Some(s)) => Some(s.to_string()),
        (_, None) => None,
    }
}

/// The spec for a command given its (already-canonicalized) sub. `None` for an
/// unknown command.
pub fn command_spec(name: &str, sub: Option<&str>) -> Option<CommandSpec> {
    use ArgDefault::*;
    let (args, series): (Vec<ArgDefault>, Vec<&'static str>) = match (name, sub) {
        ("ma" | "ema" | "smma", _) => (vec![Required], vec!["close"]),
        ("macd", None) => (vec![Int(12), Int(26)], vec!["close"]),
        ("macd", Some("signal" | "histogram")) => {
            (vec![Int(12), Int(26), Int(9)], vec!["close"])
        }
        ("boll", None) => (vec![Int(20)], vec!["close"]),
        ("boll", Some("upper" | "lower")) => (vec![Int(20), Float(2.0)], vec!["close"]),
        ("bbw", _) => (vec![Int(20)], vec!["close"]),
        ("rsv", _) => (vec![Required], vec!["high", "low", "close"]),
        ("kdj", Some("k")) => (vec![Int(9), Int(3), Float(50.0)], vec!["high", "low", "close"]),
        ("kdj", Some("d" | "j")) => (
            vec![Int(9), Int(3), Int(3), Float(50.0)],
            vec!["high", "low", "close"],
        ),
        ("rsi", _) => (vec![Required], vec!["close"]),
        ("bbi", _) => (vec![Int(3), Int(6), Int(12), Int(24)], vec!["close"]),
        ("tr", _) => (vec![], vec!["high", "low", "close"]),
        ("atr", _) => (vec![Int(14)], vec!["high", "low", "close"]),
        ("llv", _) => (vec![Required], vec!["low"]),
        ("hhv", _) => (vec![Required], vec!["high"]),
        ("donchian", None) => (vec![Required], vec!["high", "low"]),
        ("donchian", Some("upper")) => (vec![Required], vec!["high"]),
        ("donchian", Some("lower")) => (vec![Required], vec!["low"]),
        ("hv", _) => (vec![Required, Str("1d"), I64(252)], vec!["close"]),
        ("increase", _) => (vec![Int(1), I64(1)], vec!["close"]),
        ("style", _) => (vec![Required], vec!["open", "close"]),
        ("repeat", _) => (vec![Int(1)], vec![]),
        ("change", _) => (vec![Int(2)], vec!["close"]),
        _ => return None,
    };
    Some(CommandSpec { args, series })
}

/// The default integer for argument `i` of a command (its `Int` default, else
/// `fallback`). A convenience for `exec` / `lookback`.
pub fn arg_int_default(name: &str, sub: Option<&str>, i: usize, fallback: usize) -> usize {
    command_spec(name, sub)
        .and_then(|s| s.args.get(i).and_then(ArgDefault::as_usize))
        .unwrap_or(fallback)
}
