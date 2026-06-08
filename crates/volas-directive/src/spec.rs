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
        (_, Some(s)) => Some(s.to_string()),
        (_, None) => None,
    }
}

/// Whether `name` is a known command (regardless of sub-command).
pub fn is_command(name: &str) -> bool {
    matches!(
        name,
        "ma" | "ema"
            | "smma"
            | "apo"
            | "ppo"
            | "macdext"
            | "macdfix"
            | "wma"
            | "dema"
            | "tema"
            | "trima"
            | "t3"
            | "kama"
            | "mavp"
            | "sar"
            | "sarext"
            | "accbands"
            | "macd"
            | "boll"
            | "bbw"
            | "rsv"
            | "kdj"
            | "rsi"
            | "bbi"
            | "psy"
            | "pvt"
            | "nvi"
            | "pvi"
            | "dpo"
            | "cmf"
            | "chop"
            | "kst"
            | "emv"
            | "mass_index"
            | "efi"
            | "tsi"
            | "crsi"
            | "tr"
            | "atr"
            | "llv"
            | "hhv"
            | "donchian"
            | "hv"
            | "increase"
            | "style"
            | "repeat"
            | "change"
            | "mom"
            | "roc"
            | "rocp"
            | "rocr"
            | "rocr100"
            | "willr"
            | "cmo"
            | "cci"
            | "trix"
            | "imi"
            | "mfi"
            | "ultosc"
            | "stoch"
            | "stochf"
            | "stochrsi"
            | "plus_dm"
            | "minus_dm"
            | "plus_di"
            | "minus_di"
            | "dx"
            | "adx"
            | "adxr"
            | "aroon"
            | "aroonosc"
            | "sum"
            | "maxindex"
            | "minindex"
            | "minmax"
            | "minmaxindex"
            | "natr"
            | "bop"
            | "midpoint"
            | "midprice"
            | "linearreg"
            | "linearreg_slope"
            | "linearreg_intercept"
            | "linearreg_angle"
            | "tsf"
            | "var"
            | "stddev"
            | "correl"
            | "beta"
            | "obv"
            | "ad"
            | "adosc"
            | "avgprice"
            | "medprice"
            | "typprice"
            | "wclprice"
            | "ht_dcperiod"
            | "ht_dcphase"
            | "ht_phasor"
            | "ht_sine"
            | "ht_trendline"
            | "ht_trendmode"
            | "mama"
    )
}

/// The spec for a command given its (already-canonicalized) sub. `None` for an
/// unknown command **or an invalid sub-command** (the match is sub-strict, so it
/// also drives sub validation).
pub fn command_spec(name: &str, sub: Option<&str>) -> Option<CommandSpec> {
    use ArgDefault::*;
    let (args, series): (Vec<ArgDefault>, Vec<&'static str>) = match (name, sub) {
        ("ma", None) => (vec![Required, Int(0)], vec!["close"]),
        ("ema" | "smma", None) => (vec![Required], vec!["close"]),
        ("apo" | "ppo", None) => (vec![Int(12), Int(26), Int(0)], vec!["close"]),
        // macdext: fast, fastmatype, slow, slowmatype[, signal, signalmatype]. Matypes
        // default to SMA (0), unlike macd's fixed EMA.
        ("macdext", None) => (vec![Int(12), Int(0), Int(26), Int(0)], vec!["close"]),
        ("macdext", Some("signal" | "histogram")) => (
            vec![Int(12), Int(0), Int(26), Int(0), Int(9), Int(0)],
            vec!["close"],
        ),
        ("wma" | "dema" | "tema" | "trima", None) => (vec![Int(30)], vec!["close"]),
        ("t3", None) => (vec![Int(5), Float(0.7)], vec!["close"]),
        ("kama", None) => (vec![Int(30)], vec!["close"]),
        // mavp: min_period, max_period, matype; real defaults to close, periods is the
        // required second series.
        ("mavp", None) => (vec![Int(2), Int(30), Int(0)], vec!["close"]),
        ("sar", None) => (vec![Float(0.02), Float(0.2)], vec!["high", "low"]),
        // sarext: start, offset_on_reverse, then long (init/step/max) + short (init/step/max).
        ("sarext", None) => (
            vec![
                Float(0.0),
                Float(0.0),
                Float(0.02),
                Float(0.02),
                Float(0.2),
                Float(0.02),
                Float(0.02),
                Float(0.2),
            ],
            vec!["high", "low"],
        ),
        ("macd", None) => (vec![Int(12), Int(26)], vec!["close"]),
        ("macd", Some("signal" | "histogram")) => (vec![Int(12), Int(26), Int(9)], vec!["close"]),
        // macdfix: fast/slow fixed at 12/26; the line takes no args, the signal/histogram
        // take only the signal period.
        ("macdfix", None) => (vec![], vec!["close"]),
        ("macdfix", Some("signal" | "histogram")) => (vec![Int(9)], vec!["close"]),
        ("boll", None) => (vec![Int(20)], vec!["close"]),
        ("boll", Some("upper" | "lower")) => (vec![Int(20), Float(2.0)], vec!["close"]),
        ("bbw", None) => (vec![Int(20)], vec!["close"]),
        ("accbands", None) => (vec![Int(20)], vec!["close"]),
        ("accbands", Some("upper" | "lower")) => (vec![Int(20)], vec!["high", "low"]),
        ("rsv", None) => (vec![Required], vec!["high", "low", "close"]),
        ("kdj", Some("k")) => (
            vec![Int(9), Int(3), Float(50.0)],
            vec!["high", "low", "close"],
        ),
        ("kdj", Some("d" | "j")) => (
            vec![Int(9), Int(3), Int(3), Float(50.0)],
            vec!["high", "low", "close"],
        ),
        ("rsi", None) => (vec![Required], vec!["close"]),
        ("bbi", None) => (vec![Int(3), Int(6), Int(12), Int(24)], vec!["close"]),
        ("tr", None) => (vec![], vec!["high", "low", "close"]),
        ("atr", None) => (vec![Int(14)], vec!["high", "low", "close"]),
        ("llv", None) => (vec![Required], vec!["low"]),
        ("hhv", None) => (vec![Required], vec!["high"]),
        ("donchian", None) => (vec![Required], vec!["high", "low"]),
        ("donchian", Some("upper")) => (vec![Required], vec!["high"]),
        ("donchian", Some("lower")) => (vec![Required], vec!["low"]),
        ("hv", None) => (vec![Required, Str("1d"), I64(252)], vec!["close"]),
        ("increase", None) => (vec![Int(1), I64(1)], vec!["close"]),
        // Group A non-TA-Lib indicators (gap report 2026-06-07).
        ("psy", None) => (vec![Int(12)], vec!["close"]),
        ("pvt" | "nvi" | "pvi", None) => (vec![], vec!["close", "volume"]),
        ("dpo", None) => (vec![Int(20)], vec!["close"]),
        ("cmf", None) => (vec![Int(20)], vec!["high", "low", "close", "volume"]),
        ("chop", None) => (vec![Int(14)], vec!["high", "low", "close"]),
        ("kst", None) => (vec![], vec!["close"]),
        ("emv", None) => (vec![Int(14)], vec!["high", "low", "volume"]),
        ("mass_index", None) => (vec![Int(25)], vec!["high", "low"]),
        ("efi", None) => (vec![Int(13)], vec!["close", "volume"]),
        ("tsi", None) => (vec![Int(25), Int(13)], vec!["close"]),
        ("crsi", None) => (vec![Int(3), Int(2), Int(100)], vec!["close"]),
        ("style", Some("bullish" | "bearish")) => (vec![], vec!["open", "close"]),
        // Candlestick patterns (style.<pattern> / cdl.<pattern>) — validated against the
        // compute layer's pattern registry, so new patterns need no change here. Patterns
        // taking a `penetration` ratio accept one optional Float arg (default 0.5).
        ("style", Some(p)) if volas_compute::indicators::candle_pattern(p).is_some() => {
            let args = match volas_compute::indicators::candle_pattern(p).unwrap().0 {
                volas_compute::indicators::CandlePattern::Penetration { default, .. } => {
                    vec![Float(default)]
                }
                volas_compute::indicators::CandlePattern::Plain(_) => vec![],
            };
            (args, vec!["open", "high", "low", "close"])
        }
        ("repeat", None) => (vec![Int(1)], vec![]),
        ("change", None) => (vec![Int(2)], vec!["close"]),
        ("mom" | "roc" | "rocp" | "rocr" | "rocr100", None) => (vec![Int(10)], vec!["close"]),
        ("midpoint", None) => (vec![Int(14)], vec!["close"]),
        ("midprice", None) => (vec![Int(14)], vec!["high", "low"]),
        ("willr", None) => (vec![Int(14)], vec!["high", "low", "close"]),
        ("cmo", None) => (vec![Int(14)], vec!["close"]),
        ("cci", None) => (vec![Int(14)], vec!["high", "low", "close"]),
        ("imi", None) => (vec![Int(14)], vec!["open", "close"]),
        ("trix", None) => (vec![Int(30)], vec!["close"]),
        ("mfi", None) => (vec![Int(14)], vec!["high", "low", "close", "volume"]),
        ("ultosc", None) => (vec![Int(7), Int(14), Int(28)], vec!["high", "low", "close"]),
        // stoch: fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype.
        ("stoch", Some("k" | "d")) => (
            vec![Int(5), Int(3), Int(0), Int(3), Int(0)],
            vec!["high", "low", "close"],
        ),
        // stochf: fastk_period, fastd_period, fastd_matype.
        ("stochf", Some("k" | "d")) => (vec![Int(5), Int(3), Int(0)], vec!["high", "low", "close"]),
        // stochrsi: rsi_period, fastk_period, fastd_period, fastd_matype.
        ("stochrsi", Some("k" | "d")) => (vec![Int(14), Int(5), Int(3), Int(0)], vec!["close"]),
        ("plus_dm" | "minus_dm", None) => (vec![Int(14)], vec!["high", "low"]),
        ("plus_di" | "minus_di" | "dx" | "adx" | "adxr", None) => {
            (vec![Int(14)], vec!["high", "low", "close"])
        }
        ("aroon", Some("up" | "down")) => (vec![Int(14)], vec!["high", "low"]),
        ("aroonosc", None) => (vec![Int(14)], vec!["high", "low"]),
        ("sum" | "maxindex" | "minindex", None) => (vec![Int(30)], vec!["close"]),
        ("minmax" | "minmaxindex", Some("min" | "max")) => (vec![Int(30)], vec!["close"]),
        ("natr", None) => (vec![Int(14)], vec!["high", "low", "close"]),
        ("bop", None) => (vec![], vec!["open", "high", "low", "close"]),
        (
            "linearreg" | "linearreg_slope" | "linearreg_intercept" | "linearreg_angle" | "tsf",
            None,
        ) => (vec![Int(14)], vec!["close"]),
        // First series defaults to close; the second is required (no spec default).
        ("correl", None) => (vec![Int(30)], vec!["close"]),
        ("beta", None) => (vec![Int(5)], vec!["close"]),
        ("var", None) => (vec![Int(5)], vec!["close"]),
        ("stddev", None) => (vec![Int(5), Float(1.0)], vec!["close"]),
        ("obv", None) => (vec![], vec!["close", "volume"]),
        ("ad", None) => (vec![], vec!["high", "low", "close", "volume"]),
        ("adosc", None) => (
            vec![Int(3), Int(10)],
            vec!["high", "low", "close", "volume"],
        ),
        ("avgprice", None) => (vec![], vec!["open", "high", "low", "close"]),
        ("medprice", None) => (vec![], vec!["high", "low"]),
        ("typprice", None) => (vec![], vec!["high", "low", "close"]),
        ("wclprice", None) => (vec![], vec!["high", "low", "close"]),
        // Hilbert-transform cycle suite: a single price series, no numeric args
        // (MAMA takes fast/slow limits). Multi-output lines share the base spec.
        ("ht_dcperiod" | "ht_dcphase" | "ht_trendline" | "ht_trendmode", None) => {
            (vec![], vec!["close"])
        }
        ("ht_phasor", None | Some("quadrature")) => (vec![], vec!["close"]),
        ("ht_sine", None | Some("leadsine")) => (vec![], vec!["close"]),
        ("mama", None | Some("fama")) => (vec![Float(0.5), Float(0.05)], vec!["close"]),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arg_default_as_usize_and_to_token() {
        assert_eq!(ArgDefault::Int(5).as_usize(), Some(5));
        assert_eq!(ArgDefault::Float(2.0).as_usize(), None);
        assert_eq!(ArgDefault::Required.to_token(), None);
        assert_eq!(ArgDefault::Int(12).to_token().as_deref(), Some("12"));
        assert_eq!(ArgDefault::Float(2.0).to_token().as_deref(), Some("2")); // integral float
        assert_eq!(ArgDefault::Float(2.5).to_token().as_deref(), Some("2.5")); // fractional
        assert_eq!(ArgDefault::I64(-1).to_token().as_deref(), Some("-1"));
        assert_eq!(
            ArgDefault::Str("close").to_token().as_deref(),
            Some("close")
        );
    }

    #[test]
    fn arg_int_default_uses_spec_then_fallback() {
        assert_eq!(arg_int_default("macd", None, 0, 99), 12); // macd's Int default
        assert_eq!(arg_int_default("ma", None, 0, 7), 7); // a Required arg -> fallback
        assert_eq!(arg_int_default("nope", None, 0, 3), 3); // unknown command -> fallback
    }
}
