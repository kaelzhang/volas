//! Execute a directive AST against a [`DataFrame`], producing a [`Column`].

use std::borrow::Cow;

use crate::spec::canon_sub;
use crate::types::{Node, Op, UnaryOp};
use volas_compute::indicators as ind;
use volas_core::Column;
use volas_core::DataFrame;
use volas_core::{Result, VolasError};

/// Execute a directive node against `df`.
pub fn execute(df: &DataFrame, node: &Node) -> Result<Column> {
    match node {
        Node::Scalar(v) => Ok(Column::f64(vec![*v; df.height()])),
        Node::Name(name) => {
            if name.is_empty() {
                return Err(VolasError::Value("empty column / command name".into()));
            }
            if df.has_column(name) {
                Ok(df.column(name)?.clone())
            } else {
                exec_command(df, name, None, &[], &[])
            }
        }
        Node::Command(cmd) => {
            exec_command(df, &cmd.name, cmd.sub.as_deref(), &cmd.args, &cmd.series)
        }
        Node::Unary { op, operand } => {
            let c = execute(df, operand)?;
            require_numeric(&c)?; // a str/datetime operand can't go through the f64 funnel (C4)
            Ok(match op {
                UnaryOp::Not => Column::bool(c.to_f64_vec().iter().map(|&x| x == 0.0).collect()),
                UnaryOp::Neg => Column::f64(c.to_f64_vec().iter().map(|&x| -x).collect()),
            })
        }
        Node::Binary { left, op, right } => {
            let l = execute(df, left)?;
            let r = execute(df, right)?;
            require_numeric(&l)?; // both operands must be numeric (or bool) — C4, no
            require_numeric(&r)?; // silent str -> NaN arithmetic / comparison
            Ok(apply_binary(*op, &l, &r))
        }
    }
}

fn as_bool(col: &Column) -> Vec<bool> {
    match col {
        Column::Bool(v, _) => v.to_vec(),
        other => other.to_f64_vec().iter().map(|&x| x != 0.0).collect(),
    }
}

fn apply_binary(op: Op, l: &Column, r: &Column) -> Column {
    match op {
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            let (lf, rf) = (l.to_f64_vec(), r.to_f64_vec());
            let n = lf.len().min(rf.len());
            let mut out = vec![f64::NAN; lf.len()];
            for i in 0..n {
                out[i] = match op {
                    Op::Add => lf[i] + rf[i],
                    Op::Sub => lf[i] - rf[i],
                    Op::Mul => lf[i] * rf[i],
                    Op::Div => lf[i] / rf[i],
                    _ => unreachable!(), // LCOV_EXCL_LINE
                };
            }
            Column::f64(out)
        }
        Op::And | Op::Or | Op::Xor => {
            let (lb, rb) = (as_bool(l), as_bool(r));
            let n = lb.len().min(rb.len());
            let mut out = vec![false; lb.len()];
            for i in 0..n {
                out[i] = match op {
                    Op::And => lb[i] && rb[i],
                    Op::Or => lb[i] || rb[i],
                    Op::Xor => lb[i] ^ rb[i],
                    _ => unreachable!(), // LCOV_EXCL_LINE
                };
            }
            Column::bool(out)
        }
        _ => Column::bool(apply_cmp(op, &l.to_f64_vec(), &r.to_f64_vec())),
    }
}

fn apply_cmp(op: Op, l: &[f64], r: &[f64]) -> Vec<bool> {
    let n = l.len().min(r.len());
    let mut out = vec![false; l.len()];
    match op {
        Op::Lt => (0..n).for_each(|i| out[i] = l[i] < r[i]),
        Op::Le => (0..n).for_each(|i| out[i] = l[i] <= r[i]),
        Op::Eq => (0..n).for_each(|i| out[i] = l[i] == r[i]),
        Op::Ne => (0..n).for_each(|i| out[i] = l[i] != r[i]),
        Op::Ge => (0..n).for_each(|i| out[i] = l[i] >= r[i]),
        Op::Gt => (0..n).for_each(|i| out[i] = l[i] > r[i]),
        Op::CrossUp => (1..n).for_each(|i| out[i] = l[i - 1] <= r[i - 1] && l[i] > r[i]),
        Op::CrossDown => (1..n).for_each(|i| out[i] = l[i - 1] >= r[i - 1] && l[i] < r[i]),
        Op::Cross => (1..n).for_each(|i| {
            out[i] = (l[i - 1] <= r[i - 1] && l[i] > r[i]) || (l[i - 1] >= r[i - 1] && l[i] < r[i])
        }),
        _ => unreachable!("non-comparison op in apply_cmp"), // LCOV_EXCL_LINE
    }
    out
}

// --- argument helpers -------------------------------------------------------

fn arg_at<'a>(args: &'a [Option<String>], i: usize) -> Option<&'a str> {
    args.get(i).and_then(|o| o.as_deref())
}

pub(crate) fn arg_usize(
    args: &[Option<String>],
    i: usize,
    default: Option<usize>,
) -> Result<usize> {
    match arg_at(args, i) {
        Some(s) => s
            .parse()
            .map_err(|_| VolasError::Value(format!("expected an integer, got '{s}'"))),
        None => default.ok_or_else(|| VolasError::Value(format!("missing required argument #{i}"))),
    }
}

pub(crate) fn arg_f64(args: &[Option<String>], i: usize, default: f64) -> Result<f64> {
    match arg_at(args, i) {
        Some(s) => s
            .parse()
            .map_err(|_| VolasError::Value(format!("expected a number, got '{s}'"))),
        None => Ok(default),
    }
}

fn arg_i64(args: &[Option<String>], i: usize, default: i64) -> Result<i64> {
    match arg_at(args, i) {
        Some(s) => s
            .parse()
            .map_err(|_| VolasError::Value(format!("expected an integer, got '{s}'"))),
        None => Ok(default),
    }
}

// --- series resolution ------------------------------------------------------

/// Resolve a series operand to `&[f64]`. A plain `F64` column is **borrowed**
/// (`Cow::Borrowed`, no copy — the common case, e.g. `close`/`high`/`low`); a
/// computed sub-expression or a non-`F64` column is materialised (`Cow::Owned`).
/// Callers pass `&resolved` to the kernels, which deref-coerces to `&[f64]`.
pub(crate) fn series_f64<'a>(
    df: &'a DataFrame,
    series: &[Node],
    i: usize,
    default_col: &str,
) -> Result<Cow<'a, [f64]>> {
    match series.get(i) {
        Some(Node::Name(s)) if s.is_empty() => col_f64(df, default_col),
        Some(node) => {
            let c = execute(df, node)?;
            require_numeric(&c)?;
            Ok(Cow::Owned(c.to_f64_vec()))
        }
        None => col_f64(df, default_col),
    }
}

/// A directive numeric operand must be a numeric column (float / int / bool —
/// bool acts as 0/1). A `Str` / `Datetime` column funneled through `to_f64_vec`
/// would silently become all-`NaN` (or, for datetime, f64-quantized epoch), so
/// reject it (API contract C4 — a lossy implicit conversion errors, never a silent
/// all-NaN feature column).
fn require_numeric(col: &Column) -> Result<()> {
    match col {
        Column::Str(..) | Column::Datetime(..) => Err(VolasError::Value(format!(
            "directive operand must be a numeric column, got {}",
            col.dtype()
        ))),
        _ => Ok(()),
    }
}

/// Borrow a frame column as `&[f64]` without copying when it is already `F64`;
/// otherwise convert (e.g. an `I64` volume column).
fn col_f64<'a>(df: &'a DataFrame, name: &str) -> Result<Cow<'a, [f64]>> {
    let col = df.column(name)?;
    require_numeric(col)?;
    Ok(match col.as_f64() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(col.to_f64_vec()),
    })
}

/// Resolve a **required** numeric series operand at slot `i`: unlike [`series_f64`],
/// an absent or empty operand is an error rather than a column default. Used where a
/// command genuinely needs a second series the caller must name (e.g. `beta`/`correl`).
fn series_f64_required<'a>(df: &'a DataFrame, series: &[Node], i: usize) -> Result<Cow<'a, [f64]>> {
    match series.get(i) {
        Some(Node::Name(s)) if s.is_empty() => Err(VolasError::Value(format!(
            "series argument #{i} is required"
        ))),
        Some(node) => {
            let c = execute(df, node)?;
            require_numeric(&c)?;
            Ok(Cow::Owned(c.to_f64_vec()))
        }
        None => Err(VolasError::Value(format!(
            "series argument #{i} is required"
        ))),
    }
}

fn series_bool(df: &DataFrame, series: &[Node], i: usize) -> Result<Vec<bool>> {
    let node = series
        .get(i)
        .ok_or_else(|| VolasError::Value("a boolean series argument is required".into()))?;
    match execute(df, node)? {
        Column::Bool(v, _) => Ok(v.to_vec()),
        other => {
            require_numeric(&other)?;
            Ok(other.to_f64_vec().iter().map(|&x| x != 0.0).collect())
        }
    }
}

/// Dispatch a TA-Lib MA-type code to the matching moving average over `data`.
/// Codes follow TA-Lib's `TA_MAType`: 0 SMA, 1 EMA, 2 WMA, 3 DEMA, 4 TEMA, 5 TRIMA,
/// 6 KAMA, 7 MAMA, 8 T3 (with vfactor 0.7, as TA-Lib's MA dispatch fixes it). MAMA
/// ignores `period` and returns its primary (`mama`) line with the default
/// 0.5/0.05 limits, exactly as `TA_MA`. Shared by `ma`, `apo`, `ppo`, and `mavp`.
fn ma_typed(data: &[f64], period: usize, matype: usize) -> Result<Vec<f64>> {
    Ok(match matype {
        0 => ind::ma(data, period),
        1 => ind::ema(data, period),
        2 => ind::wma(data, period),
        3 => ind::dema(data, period),
        4 => ind::tema(data, period),
        5 => ind::trima(data, period),
        6 => ind::kama(data, period),
        7 => ind::mama(data, 0.5, 0.05).0,
        8 => ind::t3(data, period, 0.7),
        other => return Err(VolasError::Value(format!("unknown ma type {other}"))),
    })
}

// --- command dispatch -------------------------------------------------------


mod cycle;
mod momentum;
mod overlap;

fn exec_command(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    args: &[Option<String>],
    series: &[Node],
) -> Result<Column> {
    // Command names are case-insensitive (P6). The parser already lower-cases names it
    // knows are commands; this also covers a bare no-arg command reached as a Node::Name
    // (e.g. `TR`). Columns are resolved before this function, so their case is preserved.
    let name_lc;
    let name = if name.as_bytes().iter().any(u8::is_ascii_uppercase) {
        name_lc = name.to_ascii_lowercase();
        name_lc.as_str()
    } else {
        name
    };
    // `cdl` is an alias for `style` (the candlestick namespace): cdl.<x> == style.<x>.
    let name = if name == "cdl" { "style" } else { name };
    let sub = canon_sub(name, sub);
    let sub = sub.as_deref();

    // Validate the command, sub-command, and argument count against the spec.
    match crate::spec::command_spec(name, sub) {
        Some(spec) if args.len() > spec.args.len() => {
            return Err(VolasError::Value(format!(
                "command \"{name}\" accepts at most {} argument(s), got {}",
                spec.args.len(),
                args.len()
            )));
        }
        Some(_) => {}
        None if crate::spec::is_command(name) => {
            return Err(VolasError::Value(match sub {
                Some(s) => format!("command \"{name}\" has no sub-command \"{s}\""),
                None => format!("command \"{name}\" requires a sub-command"),
            }));
        }
        None => return Err(VolasError::Value(format!("unknown command \"{name}\""))),
    }

    overlap::dispatch(df, name, sub, args, series)
}

pub use crate::exec_resume::{
    execute_resume, execute_resume_default_series, execute_resume_default_series_one, initial_state,
};

/// Map a time-frame string like `"15m"` / `"1h"` / `"1d"` to minutes.
fn tf_to_minutes(s: &str) -> Result<i64> {
    let s = s.trim();
    let (num, unit) = s.split_at(s.len().saturating_sub(1));
    let n: i64 = num
        .parse()
        .map_err(|_| VolasError::Value(format!("invalid time frame '{s}'")))?;
    let mult = match unit {
        "m" => 1,
        "h" => 60,
        "d" => 1440,
        "W" => 10080,
        "M" => 43200,
        "Y" => 525600,
        // seconds: minutes = n/60, clamped to >= 1 to avoid division by zero
        "s" => return Ok((n / 60).max(1)),
        _ => {
            return Err(VolasError::Value(format!(
                "invalid time frame unit in '{s}'"
            )))
        }
    };
    Ok(n * mult)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn stock() -> DataFrame {
        DataFrame::new(
            vec!["open".into(), "high".into(), "low".into(), "close".into()],
            vec![
                Column::f64(vec![5.0, 6.0, 7.0, 8.0, 9.0]),
                Column::f64(vec![6.0, 7.0, 8.0, 9.0, 10.0]),
                Column::f64(vec![4.0, 5.0, 6.0, 7.0, 8.0]),
                Column::f64(vec![5.0, 6.0, 7.0, 8.0, 9.0]),
            ],
            None,
        )
        .unwrap()
    }

    fn run(df: &DataFrame, d: &str) -> Vec<f64> {
        execute(df, &parse(d).unwrap()).unwrap().to_f64_vec()
    }

    #[test]
    fn ma_directive() {
        let r = run(&stock(), "ma:2");
        assert!(r[0].is_nan());
        assert!((r[1] - 5.5).abs() < 1e-9);
        assert!((r[4] - 8.5).abs() < 1e-9);
    }

    #[test]
    fn ma_on_open() {
        let r = run(&stock(), "ma:2@open");
        assert!((r[1] - 5.5).abs() < 1e-9);
    }

    #[test]
    fn operator_bool() {
        let c = execute(&stock(), &parse("close > 7").unwrap()).unwrap();
        assert_eq!(c.as_bool().unwrap(), &[false, false, false, true, true]);
    }

    #[test]
    fn column_passthrough() {
        let r = run(&stock(), "close");
        assert_eq!(r, vec![5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    /// An `n`-row OHLCV frame (close 1..=n; high/low/open offset; volume scaled).
    fn ohlcv_n(n: usize) -> DataFrame {
        let close: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        let volume: Vec<f64> = close.iter().map(|c| c * 100.0).collect();
        DataFrame::new(
            vec![
                "open".into(),
                "high".into(),
                "low".into(),
                "close".into(),
                "volume".into(),
            ],
            vec![
                Column::f64(open),
                Column::f64(high),
                Column::f64(low),
                Column::f64(close),
                Column::f64(volume),
            ],
            None,
        )
        .unwrap()
    }

    /// A 30-row OHLCV frame so every indicator produces a full-length result.
    fn ohlcv() -> DataFrame {
        ohlcv_n(30)
    }

    /// Degenerate periods (0, or larger than the frame) must trip every indicator's
    /// warm-up guard and return an all-length result rather than panic — these guard
    /// branches are otherwise unreached by valid-period parity tests.
    #[test]
    fn degenerate_periods_hit_compute_guards() {
        let df = ohlcv();
        let n = 30;
        let zero = [
            "ma:0",
            "ema:0",
            "smma:0",
            "wma:0",
            "dema:0",
            "tema:0",
            "trima:0",
            "t3:0",
            "kama:0",
            "mom:0",
            "roc:0",
            "rocp:0",
            "rocr:0",
            "rocr100:0",
            "willr:0",
            "cci:0",
            "cmo:0",
            "mfi:0",
            "trix:0",
            "midpoint:0",
            "midprice:0",
            "atr:0",
            "natr:0",
            "rsi:0",
            "plus_dm:0",
            "minus_dm:0",
            "plus_di:0",
            "minus_di:0",
            "dx:0",
            "adx:0",
            "adxr:0",
            "aroon.up:0",
            "aroonosc:0",
            "sum:0",
            "maxindex:0",
            "minindex:0",
            "minmax.min:0",
            "minmaxindex.min:0",
            "linearreg:0",
            "linearreg_slope:0",
            "linearreg_intercept:0",
            "linearreg_angle:0",
            "tsf:0",
            "var:0",
            "stddev:0",
            "llv:0",
            "hhv:0",
            "boll:0",
            "bbw:0",
            "accbands:0",
            "correl:0@close,close",
            "beta:0@close,close",
            // period larger than the frame trips the `period > n` arm.
            "ma:99",
            "atr:99",
            "rsi:99",
            "sum:99",
            "linearreg:99",
        ];
        for d in zero {
            let col = execute(&df, &parse(d).unwrap())
                .unwrap_or_else(|e| panic!("directive {d:?} failed: {e:?}"));
            assert_eq!(col.len(), n, "directive {d:?} returned wrong length");
        }
    }

    /// An empty frame must trip the `n == 0` guards (volume / SAR / price transforms).
    #[test]
    fn empty_frame_hits_compute_guards() {
        let df = ohlcv_n(0);
        for d in [
            "obv",
            "ad",
            "adosc:3,10",
            "sar",
            "sarext",
            "ma:5",
            "tr",
            "avgprice",
            "ht_dcperiod",
            "mama",
        ] {
            let col = execute(&df, &parse(d).unwrap()).unwrap();
            assert_eq!(col.len(), 0, "directive {d:?}");
        }
    }

    /// A frame shorter than the candlestick lookbacks must trip their short-series
    /// guards (multi-bar `n < 6`, the candle-settings `avg_period > n`, etc.).
    #[test]
    fn short_frame_hits_candle_guards() {
        let df = ohlcv_n(4);
        for d in [
            "style.doji",
            "style.engulfing",
            "style.morningstar",
            "style.hikkake",
            "style.hikkakemod",
            "style.concealbabyswall",
            "style.3whitesoldiers",
            "style.risefall3methods",
            "style.breakaway",
            "style.mathold",
        ] {
            let col = execute(&df, &parse(d).unwrap()).unwrap();
            assert_eq!(col.len(), 4, "directive {d:?}");
        }
    }

    #[test]
    fn required_series_and_unknown_matype_errors() {
        let df = ohlcv();
        // correl needs a second series operand; absent or empty -> error
        // (series_f64_required's None and empty-Name arms).
        assert!(execute(&df, &parse("correl:30").unwrap()).is_err());
        assert!(execute(&df, &parse("correl:30@close,").unwrap()).is_err());
        // unknown MA type (9) -> ma_typed error propagates through the `ma` arm's `?`.
        assert!(execute(&df, &parse("ma:5,9").unwrap()).is_err());
    }

    #[test]
    fn every_command_arm_executes() {
        let df = ohlcv();
        for d in [
            "ma:5",
            "ema:5",
            "smma:5",
            "macd",
            "macd:12,26",
            "macd.signal",
            "macd.signal:12,26,9",
            "macd.histogram",
            "macdfix",
            "macdfix.signal:9",
            "macdfix.histogram",
            "boll",
            "boll:20",
            "boll.upper",
            "boll.upper:20,2",
            "boll.lower",
            "boll.lower:20,2",
            "bbw:20",
            "rsv:9",
            "kdj.k:9,3",
            "kdj.d:9,3,3",
            "kdj.j:9,3,3",
            "rsi:14",
            "bbi:3,6,12,24",
            "tr",
            "atr:14",
            "llv:5",
            "hhv:5",
            "donchian:20",
            "donchian.upper:20",
            "donchian.lower:20",
            "increase:1",
            "increase:3,-1@close",
            "style.bullish",
            "style.bearish",
            "repeat:2@(style.bullish)",
            "repeat:1@(close>10)",
            "repeat:2@close", // non-bool series -> coerced
            "change:2",
            // Hilbert-transform suite (lookback exceeds this 30-row frame, so the
            // results are all-NaN — this only exercises the dispatch + wiring).
            "ht_dcperiod",
            "ht_dcphase",
            "ht_phasor",
            "ht_phasor.quadrature",
            "ht_sine",
            "ht_sine.leadsine",
            "ht_trendline",
            "ht_trendmode",
            "mama",
            "mama:0.5,0.05",
            "mama.fama",
            "ma:10,7", // matype 7 = MAMA line
        ] {
            let col = execute(&df, &parse(d).unwrap())
                .unwrap_or_else(|e| panic!("directive {d:?} failed: {e:?}"));
            assert_eq!(col.len(), 30, "directive {d:?} returned wrong length");
        }
    }

    #[test]
    fn hv_covers_every_time_frame_unit() {
        let df = ohlcv();
        for d in [
            "hv:10",
            "hv:10,1d,252",
            "hv:10,15m",
            "hv:10,1h",
            "hv:10,1W",
            "hv:10,1M",
            "hv:10,1Y",
            "hv:10,30s", // seconds -> minutes clamps to >= 1
        ] {
            assert_eq!(execute(&df, &parse(d).unwrap()).unwrap().len(), 30);
        }
    }

    #[test]
    fn command_and_argument_validation_errors() {
        let df = ohlcv();
        let is_err = |d: &str| execute(&df, &parse(d).unwrap()).is_err();
        assert!(is_err("ema:5,6"), "too many args"); // ema takes one period (ma now takes matype too)
        assert!(is_err("frobnicate:5"), "unknown command");
        assert!(
            is_err("MACD.bogus"),
            "case-insensitive name, still bad sub (P6)"
        );
        assert!(is_err("kdj:9"), "missing required sub-command");
        assert!(is_err("macd.bogus"), "unknown sub-command");
        assert!(is_err("ma:abc"), "non-integer arg");
        assert!(is_err("style.cartoon"), "invalid style sub-command");
        assert!(is_err("hv:10,5x"), "invalid time-frame unit");
        assert!(is_err("hv:10,xm"), "invalid time-frame number");
    }

    #[test]
    fn operators_crosses_and_unary() {
        let df = ohlcv();
        let len = |d: &str| execute(&df, &parse(d).unwrap()).unwrap().len();
        // comparisons
        for d in [
            "close > 10",
            "close < 10",
            "close >= 10",
            "close <= 10",
            "close == 10",
            "close != 10",
        ] {
            assert_eq!(len(d), 30, "{d}");
        }
        // crosses (// up, \\ down, >< either)
        for d in ["ma:5 // ma:10", "ma:5 \\\\ ma:10", "ma:5 >< ma:10"] {
            assert_eq!(len(d), 30, "{d}");
        }
        // arithmetic
        for d in ["close + 1", "close - 1", "close * 2", "close / 2"] {
            assert_eq!(len(d), 30, "{d}");
        }
        // logical (& | ^), with a non-bool right operand to coerce
        for d in [
            "(close>10) & (close<20)",
            "(close>10) | (close<5)",
            "(close>10) ^ (close<20)",
            "(close>10) & close",
        ] {
            assert_eq!(len(d), 30, "{d}");
        }
        // unary not / negate
        assert_eq!(len("~(close>10)"), 30);
        assert_eq!(len("-close"), 30);
    }

    #[test]
    fn empty_name_is_an_error() {
        assert!(execute(&stock(), &Node::Name(String::new())).is_err());
    }

    /// `initial_state` / `execute_resume` decline (return `None`, so the engine keeps the
    /// full-recompute fallback) on the off-the-happy-path arms: a non-SMA stochrsi `.d`
    /// matype, a `*_resume` that returns `None` (from below its warm-up), and a directive
    /// with no resume kernel at all (the `_ => None` catch-all).
    #[test]
    fn state_carry_none_propagation() {
        let df = ohlcv_n(80);
        let col = Column::f64(vec![0.0; 80]); // `initial_state`'s computed arg is unused
        let p = |d: &str| parse(d).unwrap();

        // initial_state: stochrsi `.d` with a non-SMA matype (arg 3 != 0) -> None (exec:1021).
        assert!(initial_state(&df, &p("stochrsi.d:14,14,3,1"), &col).is_none());

        // execute_resume: SAR / SAREXT at from_row < 2 -> the inner `*_resume` is None, the
        // `?` propagates (exec:1095 / 1114). State contents are unread on the None path.
        let sar_state = vec![1.0, 0.02, 8.0, 6.0, 9.0, 7.0];
        assert!(execute_resume(&df, &p("sar"), &sar_state, 1, 0).is_none());
        let sarext_state = vec![1.0, 0.02, 0.02, 8.0, 6.0, 9.0, 7.0];
        assert!(execute_resume(&df, &p("sarext"), &sarext_state, 1, 0).is_none());

        // execute_resume: ±DI at from_row == 0 -> inner resume None, `?` propagates
        // (exec:1245 / 1254).
        let di_state = vec![0.0, 0.0];
        assert!(execute_resume(&df, &p("plus_di:14"), &di_state, 0, 0).is_none());
        assert!(execute_resume(&df, &p("minus_di:14"), &di_state, 0, 0).is_none());

        // execute_resume: MAMA with a too-short carried state -> mama_resume None (exec:1320).
        // `from_row` is past the HT core warm-up so the length check is what declines.
        let short_mama = vec![0.0; 10];
        assert!(execute_resume(&df, &p("mama"), &short_mama, 20, 0).is_none());

        // execute_resume: index family below its `period - 1` warm-up -> resume None
        // (exec:1335 / 1345). period 30, from_row 5 < 29.
        let idx_state = vec![3.0, 100.0];
        assert!(execute_resume(&df, &p("maxindex:30"), &idx_state, 5, 0).is_none());
        assert!(execute_resume(&df, &p("minindex:30"), &idx_state, 5, 0).is_none());

        // execute_resume: KDJ below its RSV window warm-up (from_row + 1 < period_rsv = 9) ->
        // kdj_resume None, the `?` propagates (exec:1402).
        let kdj_state = vec![50.0, 50.0];
        assert!(execute_resume(&df, &p("kdj.j"), &kdj_state, 2, 0).is_none());

        // execute_resume: stochrsi `.d` non-SMA matype -> `return None` (exec:1355).
        let sr_state = vec![0.0; 20];
        assert!(execute_resume(&df, &p("stochrsi.d:14,14,3,1"), &sr_state, 40, 0).is_none());

        // execute_resume: stochrsi `.k` with a malformed (wrong-length) state -> the inner
        // stochrsi_resume is None, the `?` propagates (exec:1365).
        let bad_sr = vec![0.0; 3]; // not `ctx_depth + 2` for these periods
        assert!(execute_resume(&df, &p("stochrsi.k:14,14,3"), &bad_sr, 40, 0).is_none());

        // execute_resume: a directive with no resume kernel hits the `_ => None` arm
        // (exec:1369). `llv` (lowest-low) has no incremental resume.
        assert!(execute_resume(&df, &p("llv:5"), &[0.0], 3, 0).is_none());
    }
}
