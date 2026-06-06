//! Execute a directive AST against a [`DataFrame`], producing a [`Column`].

use std::borrow::Cow;

use crate::spec::canon_sub;
use crate::types::{Node, Op, UnaryOp};
use volas_core::Column;
use volas_core::DataFrame;
use volas_core::{Result, VolasError};
use volas_compute::indicators as ind;

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
            Ok(match op {
                UnaryOp::Not => Column::bool(c.to_f64_vec().iter().map(|&x| x == 0.0).collect()),
                UnaryOp::Neg => Column::f64(c.to_f64_vec().iter().map(|&x| -x).collect()),
            })
        }
        Node::Binary { left, op, right } => {
            let l = execute(df, left)?;
            let r = execute(df, right)?;
            Ok(apply_binary(*op, &l, &r))
        }
    }
}

fn as_bool(col: &Column) -> Vec<bool> {
    match col {
        Column::Bool(v) => v.to_vec(),
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
            out[i] = (l[i - 1] <= r[i - 1] && l[i] > r[i])
                || (l[i - 1] >= r[i - 1] && l[i] < r[i])
        }),
        _ => unreachable!("non-comparison op in apply_cmp"), // LCOV_EXCL_LINE
    }
    out
}

// --- argument helpers -------------------------------------------------------

fn arg_at<'a>(args: &'a [Option<String>], i: usize) -> Option<&'a str> {
    args.get(i).and_then(|o| o.as_deref())
}

fn arg_usize(args: &[Option<String>], i: usize, default: Option<usize>) -> Result<usize> {
    match arg_at(args, i) {
        Some(s) => s
            .parse()
            .map_err(|_| VolasError::Value(format!("expected an integer, got '{s}'"))),
        None => default.ok_or_else(|| VolasError::Value(format!("missing required argument #{i}"))),
    }
}

fn arg_f64(args: &[Option<String>], i: usize, default: f64) -> Result<f64> {
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
fn series_f64<'a>(
    df: &'a DataFrame,
    series: &[Node],
    i: usize,
    default_col: &str,
) -> Result<Cow<'a, [f64]>> {
    match series.get(i) {
        Some(Node::Name(s)) if s.is_empty() => col_f64(df, default_col),
        Some(node) => Ok(Cow::Owned(execute(df, node)?.to_f64_vec())),
        None => col_f64(df, default_col),
    }
}

/// Borrow a frame column as `&[f64]` without copying when it is already `F64`;
/// otherwise convert (e.g. an `I64` volume column).
fn col_f64<'a>(df: &'a DataFrame, name: &str) -> Result<Cow<'a, [f64]>> {
    let col = df.column(name)?;
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
        Some(node) => Ok(Cow::Owned(execute(df, node)?.to_f64_vec())),
        None => Err(VolasError::Value(format!("series argument #{i} is required"))),
    }
}

fn series_bool(df: &DataFrame, series: &[Node], i: usize) -> Result<Vec<bool>> {
    let node = series
        .get(i)
        .ok_or_else(|| VolasError::Value("a boolean series argument is required".into()))?;
    match execute(df, node)? {
        Column::Bool(v) => Ok(v.to_vec()),
        other => Ok(other.to_f64_vec().iter().map(|&x| x != 0.0).collect()),
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
    let name_lc = name.to_ascii_lowercase();
    let name = name_lc.as_str();
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

    let close = |i| series_f64(df, series, i, "close");
    let f64col = |v: Vec<f64>| Ok(Column::f64(v));
    let boolcol = |v: Vec<bool>| Ok(Column::bool(v));

    match (name, sub) {
        ("ma", _) => f64col(ma_typed(
            &close(0)?,
            arg_usize(args, 0, None)?,
            arg_usize(args, 1, Some(0))?,
        )?),
        ("ema", _) => f64col(ind::ema(&close(0)?, arg_usize(args, 0, None)?)),
        ("smma", _) => f64col(ind::smma(&close(0)?, arg_usize(args, 0, None)?)),
        ("wma", _) => f64col(ind::wma(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("dema", _) => f64col(ind::dema(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("tema", _) => f64col(ind::tema(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("trima", _) => f64col(ind::trima(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("t3", _) => f64col(ind::t3(
            &close(0)?,
            arg_usize(args, 0, Some(5))?,
            arg_f64(args, 1, 0.7)?,
        )),
        ("kama", _) => f64col(ind::kama(&close(0)?, arg_usize(args, 0, Some(30))?)),
        // MA with a per-row variable period: the period for row i is the (truncated,
        // clamped) value of the required second `periods` series. Each distinct period's
        // MA is computed once and cached.
        ("mavp", _) => {
            let real = close(0)?;
            let periods = series_f64_required(df, series, 1)?;
            let min_p = arg_usize(args, 0, Some(2))?;
            let max_p = arg_usize(args, 1, Some(30))?;
            let matype = arg_usize(args, 2, Some(0))?;
            let lb = crate::lookback::ma_lookback(max_p, matype);
            let n = real.len();
            let mut out = vec![f64::NAN; n];
            // Each distinct period's MA is computed once, over the sub-slice that makes its
            // first valid value land exactly at `lb` — i.e. TA-Lib re-seeds every period's
            // MA at the output start (matters for recursive MAs like EMA; windowed MAs are
            // position-independent). `cache[p] = (start, sub_ma)`; MAVP[i] = sub_ma[i-start].
            // `cache[p] = (start, sub_ma)` indexed by the clamped period directly: `p` is a
            // small bounded usize (`<= max_p`), so a flat Vec beats a HashMap whose per-bar
            // hash + probe dominated this loop. Values are unchanged — only the lookup is faster.
            let mut cache: Vec<Option<(usize, Vec<f64>)>> = vec![None; max_p + 1];
            for i in lb..n {
                let p = (periods[i] as i64).clamp(min_p as i64, max_p as i64) as usize;
                if cache[p].is_none() {
                    let start = lb - crate::lookback::ma_lookback(p, matype);
                    let sub = ma_typed(&real[start..], p, matype)?;
                    cache[p] = Some((start, sub));
                }
                let (start, sub) = cache[p].as_ref().unwrap();
                out[i] = sub[i - start];
            }
            f64col(out)
        }
        ("sar", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::sar(&high, &low, arg_f64(args, 0, 0.02)?, arg_f64(args, 1, 0.2)?))
        }
        ("sarext", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::sarext(
                &high,
                &low,
                arg_f64(args, 0, 0.0)?,
                arg_f64(args, 1, 0.0)?,
                arg_f64(args, 2, 0.02)?,
                arg_f64(args, 3, 0.02)?,
                arg_f64(args, 4, 0.2)?,
                arg_f64(args, 5, 0.02)?,
                arg_f64(args, 6, 0.02)?,
                arg_f64(args, 7, 0.2)?,
            ))
        }
        // Price oscillators: fast MA - slow MA, of a chosen MA type (default SMA).
        ("apo", _) => {
            let data = close(0)?;
            let mt = arg_usize(args, 2, Some(0))?;
            let f = ma_typed(&data, arg_usize(args, 0, Some(12))?, mt)?;
            let s = ma_typed(&data, arg_usize(args, 1, Some(26))?, mt)?;
            f64col((0..data.len()).map(|i| f[i] - s[i]).collect())
        }
        // MACDEXT: a MACD whose fast/slow/signal MAs each take a configurable type
        // (matypes default to SMA, not macd's fixed EMA). The line is emitted at its
        // natural start (best practice, like macd); signal/histogram follow.
        ("macdext", sub) => {
            let data = close(0)?;
            let f = ma_typed(&data, arg_usize(args, 0, Some(12))?, arg_usize(args, 1, Some(0))?)?;
            let s = ma_typed(&data, arg_usize(args, 2, Some(26))?, arg_usize(args, 3, Some(0))?)?;
            let line: Vec<f64> = (0..data.len()).map(|i| f[i] - s[i]).collect();
            match sub {
                None => f64col(line),
                _ => {
                    let signal =
                        ma_typed(&line, arg_usize(args, 4, Some(9))?, arg_usize(args, 5, Some(0))?)?;
                    if sub == Some("signal") {
                        f64col(signal)
                    } else {
                        f64col((0..line.len()).map(|i| line[i] - signal[i]).collect())
                    }
                }
            }
        }
        ("ppo", _) => {
            let data = close(0)?;
            let mt = arg_usize(args, 2, Some(0))?;
            let f = ma_typed(&data, arg_usize(args, 0, Some(12))?, mt)?;
            let s = ma_typed(&data, arg_usize(args, 1, Some(26))?, mt)?;
            f64col((0..data.len()).map(|i| (f[i] - s[i]) / s[i] * 100.0).collect())
        }

        ("macd", None) => {
            f64col(ind::macd(&close(0)?, arg_usize(args, 0, Some(12))?, arg_usize(args, 1, Some(26))?))
        }
        ("macd", Some("signal")) => f64col(ind::macd_signal(
            &close(0)?,
            arg_usize(args, 0, Some(12))?,
            arg_usize(args, 1, Some(26))?,
            arg_usize(args, 2, Some(9))?,
        )),
        ("macd", Some("histogram")) => f64col(ind::macd_histogram(
            &close(0)?,
            arg_usize(args, 0, Some(12))?,
            arg_usize(args, 1, Some(26))?,
            arg_usize(args, 2, Some(9))?,
        )),

        // MACDFIX: MACD with fast/slow fixed at 12/26; only the signal period is
        // configurable. Reuses the (verified) macd line / signal / histogram.
        ("macdfix", None) => f64col(ind::macd(&close(0)?, 12, 26)),
        ("macdfix", Some("signal")) => {
            f64col(ind::macd_signal(&close(0)?, 12, 26, arg_usize(args, 0, Some(9))?))
        }
        ("macdfix", Some("histogram")) => {
            f64col(ind::macd_histogram(&close(0)?, 12, 26, arg_usize(args, 0, Some(9))?))
        }

        ("boll", None) => f64col(ind::boll(&close(0)?, arg_usize(args, 0, Some(20))?)),
        ("boll", Some("upper")) => f64col(ind::boll_upper(
            &close(0)?,
            arg_usize(args, 0, Some(20))?,
            arg_f64(args, 1, 2.0)?,
        )),
        ("boll", Some("lower")) => f64col(ind::boll_lower(
            &close(0)?,
            arg_usize(args, 0, Some(20))?,
            arg_f64(args, 1, 2.0)?,
        )),
        ("bbw", _) => f64col(ind::bbw(&close(0)?, arg_usize(args, 0, Some(20))?)),

        ("accbands", None) => {
            f64col(ind::accbands_middle(&close(0)?, arg_usize(args, 0, Some(20))?))
        }
        ("accbands", Some("upper")) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::accbands_upper(&high, &low, arg_usize(args, 0, Some(20))?))
        }
        ("accbands", Some("lower")) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::accbands_lower(&high, &low, arg_usize(args, 0, Some(20))?))
        }

        ("rsv", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::rsv(&high, &low, &close, arg_usize(args, 0, None)?))
        }
        ("kdj", Some(line @ ("k" | "d" | "j"))) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let period_rsv = arg_usize(args, 0, Some(9))?;
            let period_k = arg_usize(args, 1, Some(3))?;
            match line {
                "k" => {
                    let init = arg_f64(args, 2, 50.0)?;
                    f64col(ind::kdj_k(&high, &low, &close, period_rsv, period_k, init))
                }
                _ => {
                    let period_d = arg_usize(args, 2, Some(3))?;
                    let init = arg_f64(args, 3, 50.0)?;
                    let v = if line == "d" {
                        ind::kdj_d(&high, &low, &close, period_rsv, period_k, period_d, init)
                    } else {
                        ind::kdj_j(&high, &low, &close, period_rsv, period_k, period_d, init)
                    };
                    f64col(v)
                }
            }
        }

        ("rsi", _) => f64col(ind::rsi(&close(0)?, arg_usize(args, 0, None)?)),
        ("bbi", _) => f64col(ind::bbi(
            &close(0)?,
            arg_usize(args, 0, Some(3))?,
            arg_usize(args, 1, Some(6))?,
            arg_usize(args, 2, Some(12))?,
            arg_usize(args, 3, Some(24))?,
        )),

        ("tr", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::tr(&high, &low, &close))
        }
        ("atr", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::atr(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }

        ("llv", _) => f64col(ind::llv(&series_f64(df, series, 0, "low")?, arg_usize(args, 0, None)?)),
        ("hhv", _) => f64col(ind::hhv(&series_f64(df, series, 0, "high")?, arg_usize(args, 0, None)?)),

        ("donchian", None) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::donchian(&high, &low, arg_usize(args, 0, None)?))
        }
        ("donchian", Some("upper")) => {
            f64col(ind::hhv(&series_f64(df, series, 0, "high")?, arg_usize(args, 0, None)?))
        }
        ("donchian", Some("lower")) => {
            f64col(ind::llv(&series_f64(df, series, 0, "low")?, arg_usize(args, 0, None)?))
        }

        ("midpoint", _) => f64col(ind::midpoint(&close(0)?, arg_usize(args, 0, Some(14))?)),
        ("midprice", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::midprice(&high, &low, arg_usize(args, 0, Some(14))?))
        }

        ("hv", _) => {
            let close = series_f64(df, series, 0, "close")?;
            let period = arg_usize(args, 0, None)?;
            let minutes = arg_at(args, 1).map_or(Ok(1440), tf_to_minutes)?;
            let trading_days = arg_i64(args, 2, 252)?;
            f64col(ind::hv(&close, period, minutes, trading_days))
        }

        ("increase", _) => boolcol(ind::increase(
            &close(0)?,
            arg_usize(args, 0, Some(1))?,
            arg_i64(args, 1, 1)? as i32,
        )),
        // Candlestick patterns: style.<pattern> / cdl.<pattern>. Output f64 -100/0/100
        // (engulfing may also emit ±80). A few take an optional `penetration` ratio.
        ("style", Some(pat)) if ind::candle_pattern(pat).is_some() => {
            let (pattern, _) = ind::candle_pattern(pat).unwrap();
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            let close = series_f64(df, series, 3, "close")?;
            let v = match pattern {
                ind::CandlePattern::Plain(f) => f(&open, &high, &low, &close),
                ind::CandlePattern::Penetration { f, default } => {
                    f(&open, &high, &low, &close, arg_f64(args, 0, default)?)
                }
            };
            f64col(v)
        }

        ("style", sub) => {
            // Direction from the sub-command (`style.bullish`, also `cdl.bullish`) or,
            // for back-compat, the first argument (`style:bullish`).
            let style = match sub.or_else(|| arg_at(args, 0)) {
                Some("bullish") => ind::Style::Bullish,
                Some("bearish") => ind::Style::Bearish,
                other => {
                    return Err(VolasError::Value(format!(
                        "style should be 'bullish' or 'bearish', got '{}'",
                        other.unwrap_or("")
                    )))
                }
            };
            let open = series_f64(df, series, 0, "open")?;
            let close = series_f64(df, series, 1, "close")?;
            boolcol(ind::style(style, &open, &close))
        }
        ("repeat", _) => boolcol(ind::repeat(&series_bool(df, series, 0)?, arg_usize(args, 0, Some(1))?)),
        ("change", _) => f64col(ind::change(&close(0)?, arg_usize(args, 0, Some(2))?)),

        ("mom", _) => f64col(ind::mom(&close(0)?, arg_usize(args, 0, Some(10))?)),
        ("roc", _) => f64col(ind::roc(&close(0)?, arg_usize(args, 0, Some(10))?)),
        ("rocp", _) => f64col(ind::rocp(&close(0)?, arg_usize(args, 0, Some(10))?)),
        ("rocr", _) => f64col(ind::rocr(&close(0)?, arg_usize(args, 0, Some(10))?)),
        ("rocr100", _) => f64col(ind::rocr100(&close(0)?, arg_usize(args, 0, Some(10))?)),

        ("willr", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::willr(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("cmo", _) => f64col(ind::cmo(&close(0)?, arg_usize(args, 0, Some(14))?)),
        ("mfi", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let volume = series_f64(df, series, 3, "volume")?;
            f64col(ind::mfi(&high, &low, &close, &volume, arg_usize(args, 0, Some(14))?))
        }
        ("ultosc", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::ultosc(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(7))?,
                arg_usize(args, 1, Some(14))?,
                arg_usize(args, 2, Some(28))?,
            ))
        }
        // Stochastic family: raw %K (NaN warm-up) then matype-MA smoothing stages.
        ("stoch", Some(line @ ("k" | "d"))) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let fastk = ind::stoch_fastk(&high, &low, &close, arg_usize(args, 0, Some(5))?);
            let slowk = ma_typed(&fastk, arg_usize(args, 1, Some(3))?, arg_usize(args, 2, Some(0))?)?;
            if line == "k" {
                f64col(slowk)
            } else {
                f64col(ma_typed(&slowk, arg_usize(args, 3, Some(3))?, arg_usize(args, 4, Some(0))?)?)
            }
        }
        ("stochf", Some(line @ ("k" | "d"))) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let fastk = ind::stoch_fastk(&high, &low, &close, arg_usize(args, 0, Some(5))?);
            if line == "k" {
                f64col(fastk)
            } else {
                f64col(ma_typed(&fastk, arg_usize(args, 1, Some(3))?, arg_usize(args, 2, Some(0))?)?)
            }
        }
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let close = close(0)?;
            let fastk = ind::stochrsi_fastk(
                &close,
                arg_usize(args, 0, Some(14))?,
                arg_usize(args, 1, Some(5))?,
            );
            if line == "k" {
                f64col(fastk)
            } else {
                f64col(ma_typed(&fastk, arg_usize(args, 2, Some(3))?, arg_usize(args, 3, Some(0))?)?)
            }
        }

        // Directional movement family (+DM/-DM need only high/low; the rest add close).
        ("plus_dm", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::plus_dm(&high, &low, arg_usize(args, 0, Some(14))?))
        }
        ("minus_dm", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::minus_dm(&high, &low, arg_usize(args, 0, Some(14))?))
        }
        ("plus_di", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::plus_di(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("minus_di", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::minus_di(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("dx", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::dx(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("adx", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::adx(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("adxr", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::adxr(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("cci", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::cci(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("imi", _) => {
            let open = series_f64(df, series, 0, "open")?;
            let close = series_f64(df, series, 1, "close")?;
            f64col(ind::imi(&open, &close, arg_usize(args, 0, Some(14))?))
        }
        ("trix", _) => f64col(ind::trix(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("aroon", Some(dir @ ("up" | "down"))) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let period = arg_usize(args, 0, Some(14))?;
            let v = if dir == "up" {
                ind::aroon_up(&high, &low, period)
            } else {
                ind::aroon_down(&high, &low, period)
            };
            f64col(v)
        }
        ("aroonosc", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::aroonosc(&high, &low, arg_usize(args, 0, Some(14))?))
        }

        ("sum", _) => f64col(ind::sum(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("maxindex", _) => f64col(ind::maxindex(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("minindex", _) => f64col(ind::minindex(&close(0)?, arg_usize(args, 0, Some(30))?)),
        // minmax / minmaxindex are the (min, max) value / index pair over the window;
        // their outputs are exactly llv/hhv (values) and minindex/maxindex (indices).
        ("minmax", Some("min")) => f64col(ind::llv(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("minmax", Some("max")) => f64col(ind::hhv(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("minmaxindex", Some("min")) => {
            f64col(ind::minindex(&close(0)?, arg_usize(args, 0, Some(30))?))
        }
        ("minmaxindex", Some("max")) => {
            f64col(ind::maxindex(&close(0)?, arg_usize(args, 0, Some(30))?))
        }
        ("natr", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::natr(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("bop", _) => {
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            let close = series_f64(df, series, 3, "close")?;
            f64col(ind::bop(&open, &high, &low, &close))
        }

        ("linearreg", _) => f64col(ind::linearreg(&close(0)?, arg_usize(args, 0, Some(14))?)),
        ("linearreg_slope", _) => {
            f64col(ind::linearreg_slope(&close(0)?, arg_usize(args, 0, Some(14))?))
        }
        ("linearreg_intercept", _) => {
            f64col(ind::linearreg_intercept(&close(0)?, arg_usize(args, 0, Some(14))?))
        }
        ("linearreg_angle", _) => {
            f64col(ind::linearreg_angle(&close(0)?, arg_usize(args, 0, Some(14))?))
        }
        ("tsf", _) => f64col(ind::tsf(&close(0)?, arg_usize(args, 0, Some(14))?)),
        // beta/correl relate two series: the first defaults to close, the second is required.
        ("correl", _) => {
            let x = close(0)?;
            let y = series_f64_required(df, series, 1)?;
            f64col(ind::correl(&x, &y, arg_usize(args, 0, Some(30))?))
        }
        ("beta", _) => {
            let x = close(0)?;
            let y = series_f64_required(df, series, 1)?;
            f64col(ind::beta(&x, &y, arg_usize(args, 0, Some(5))?))
        }

        ("var", _) => f64col(ind::var(&close(0)?, arg_usize(args, 0, Some(5))?)),
        ("stddev", _) => f64col(ind::stddev(
            &close(0)?,
            arg_usize(args, 0, Some(5))?,
            arg_f64(args, 1, 1.0)?,
        )),

        ("obv", _) => {
            let real = series_f64(df, series, 0, "close")?;
            let volume = series_f64(df, series, 1, "volume")?;
            f64col(ind::obv(&real, &volume))
        }
        ("ad", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let volume = series_f64(df, series, 3, "volume")?;
            f64col(ind::ad(&high, &low, &close, &volume))
        }
        ("adosc", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let volume = series_f64(df, series, 3, "volume")?;
            f64col(ind::adosc(
                &high,
                &low,
                &close,
                &volume,
                arg_usize(args, 0, Some(3))?,
                arg_usize(args, 1, Some(10))?,
            ))
        }

        ("avgprice", _) => {
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            let close = series_f64(df, series, 3, "close")?;
            f64col(ind::avgprice(&open, &high, &low, &close))
        }
        ("medprice", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::medprice(&high, &low))
        }
        ("typprice", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::typprice(&high, &low, &close))
        }
        ("wclprice", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::wclprice(&high, &low, &close))
        }

        // Hilbert-transform cycle suite. Each takes a single price series (default
        // close); multi-output ones expose the secondary line via a sub-command (P3).
        ("ht_dcperiod", _) => f64col(ind::ht_dcperiod(&close(0)?)),
        ("ht_dcphase", _) => f64col(ind::ht_dcphase(&close(0)?)),
        ("ht_trendline", _) => f64col(ind::ht_trendline(&close(0)?)),
        ("ht_trendmode", _) => f64col(ind::ht_trendmode(&close(0)?)),
        ("ht_phasor", None) => f64col(ind::ht_phasor(&close(0)?).0),
        ("ht_phasor", Some("quadrature")) => f64col(ind::ht_phasor(&close(0)?).1),
        ("ht_sine", None) => f64col(ind::ht_sine(&close(0)?).0),
        ("ht_sine", Some("leadsine")) => f64col(ind::ht_sine(&close(0)?).1),
        ("mama", None) => f64col(
            ind::mama(&close(0)?, arg_f64(args, 0, 0.5)?, arg_f64(args, 1, 0.05)?).0,
        ),
        ("mama", Some("fama")) => f64col(
            ind::mama(&close(0)?, arg_f64(args, 0, 0.5)?, arg_f64(args, 1, 0.05)?).1,
        ),

        (other, _) => Err(VolasError::Value(format!("unknown command '{other}'"))), // LCOV_EXCL_LINE
    }
}

// --- state-carry resume (additive; fallback path stays correct) -------------
//
// A recursive indicator's whole history compresses into a small fixed-size state
// (a `Vec<f64>`). `initial_state` captures that state after a full compute;
// `execute_resume` continues the recursion over only the new tail rows, producing
// values bit-identical to a fresh full recompute. Both return `None` for any
// directive without a resume kernel, so the caller transparently falls back to the
// correct full-recompute path. Only the canonical no-operand forms (the directives
// volas auto-caches) are handled; an unusual `@`-operand override returns `None`
// and stays on the fallback.

/// Resolve a command node to `(name_lc, sub, args, series)` when it is a plain
/// `Node::Command` (or a bare `Node::Name` no-arg command, e.g. `obv`/`ad`); `None`
/// otherwise. The name is lower-cased and `cdl`→`style` aliased, matching
/// [`exec_command`]. A `Node::Name` carries no sub / args / series — the same way
/// [`execute`] dispatches it via `exec_command(df, name, None, &[], &[])`.
fn as_command(node: &Node) -> Option<(String, Option<String>, &[Option<String>], &[Node])> {
    let lc = |name: &str| {
        let name = name.to_ascii_lowercase();
        if name == "cdl" {
            "style".to_string()
        } else {
            name
        }
    };
    match node {
        Node::Command(cmd) => Some((lc(&cmd.name), cmd.sub.clone(), &cmd.args, &cmd.series)),
        Node::Name(name) if !name.is_empty() => Some((lc(name), None, &[], &[])),
        _ => None,
    }
}

/// The final recursive state after a full compute of `node` against `df`, matching
/// the just-computed `computed` column. `None` when `node` has no resume kernel
/// (the caller then keeps the full-recompute fallback). `computed` is accepted for
/// kernels that can read their state off the output column directly; the cumulative
/// family recomputes its (tiny) state from the raw inputs to stay bit-exact with
/// the canonical kernel.
pub fn initial_state(df: &DataFrame, node: &Node, _computed: &Column) -> Option<Vec<f64>> {
    let (name, sub, args, series) = as_command(node)?;
    let sub = sub.as_deref();
    match (name.as_str(), sub) {
        ("obv", _) => {
            let real = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::obv_final_state(&real, &volume)
        }
        ("ad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            ind::ad_final_state(&high, &low, &close, &volume)
        }
        ("adosc", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            let fast = arg_usize(args, 0, Some(3)).ok()?;
            let slow = arg_usize(args, 1, Some(10)).ok()?;
            ind::adosc_final_state(&high, &low, &close, &volume, fast, slow)
        }

        // SAR family — carry the recurrence's loop state (trend, accel factor(s),
        // extreme point, current SAR, and the prior bar's high/low).
        ("sar", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::sar_final_state(&high, &low, arg_f64(args, 0, 0.02).ok()?, arg_f64(args, 1, 0.2).ok()?)
        }
        ("sarext", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::sarext_final_state(
                &high,
                &low,
                arg_f64(args, 0, 0.0).ok()?,
                arg_f64(args, 1, 0.0).ok()?,
                arg_f64(args, 2, 0.02).ok()?,
                arg_f64(args, 3, 0.02).ok()?,
                arg_f64(args, 4, 0.2).ok()?,
                arg_f64(args, 5, 0.02).ok()?,
                arg_f64(args, 6, 0.02).ok()?,
                arg_f64(args, 7, 0.2).ok()?,
            )
        }

        // EMA-recursion family — carry the sub-EMA stage states (see exec's resume block).
        ("ema", _) => ind::ema_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, None).ok()?),
        ("smma", _) => ind::smma_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, None).ok()?),
        ("dema", _) => ind::dema_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, Some(30)).ok()?),
        ("tema", _) => ind::tema_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, Some(30)).ok()?),
        // T3's carried state is just the six EMA stages (vfactor only scales the combine,
        // not the cascade), so `t3_final_state` needs no vfactor.
        ("t3", _) => ind::t3_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(5)).ok()?,
        ),
        ("trix", _) => ind::trix_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, Some(30)).ok()?),
        ("kama", _) => ind::kama_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, Some(30)).ok()?),

        ("macd", None) => ind::macd_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(12)).ok()?,
            arg_usize(args, 1, Some(26)).ok()?,
        ),
        ("macd", Some("signal" | "histogram")) => ind::macd_signal_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(12)).ok()?,
            arg_usize(args, 1, Some(26)).ok()?,
            arg_usize(args, 2, Some(9)).ok()?,
        ),
        ("macdfix", None) => ind::macd_final_state(&series_f64(df, series, 0, "close").ok()?, 12, 26),
        ("macdfix", Some("signal" | "histogram")) => ind::macd_signal_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            12,
            26,
            arg_usize(args, 0, Some(9)).ok()?,
        ),

        // Wilder-smoothing family — carry the running average(s). RSI/CMO carry
        // [avg_gain, avg_loss]; ATR/NATR carry the running ATR; the directional ratios
        // carry the +DM/−DM/TR Wilder sums (and ADX/ADXR additionally the running ADX /
        // its trailing window).
        ("rsi", _) => ind::rsi_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, None).ok()?),
        ("cmo", _) => ind::cmo_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, Some(14)).ok()?),
        ("atr", _) => ind::atr_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("natr", _) => ind::atr_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("plus_dm", _) => ind::plus_dm_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("minus_dm", _) => ind::minus_dm_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("plus_di", _) => ind::plus_di_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("minus_di", _) => ind::minus_di_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("dx", _) => ind::dx_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("adx", _) => ind::adx_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("adxr", _) => ind::adxr_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),

        // Hilbert-transform family — carry the shared core state (WMA smoother + 4
        // Hilbert channels + homodyne discriminator) plus each output's small tail
        // (DC-phase ring, trendline iTrend triple, MAMA/FAMA accumulators). All read
        // the single `close` series (default), matching their dispatch in `execute`.
        ("ht_dcperiod", _) | ("ht_phasor", _) => {
            ind::ht_core_state(&series_f64(df, series, 0, "close").ok()?)
        }
        ("ht_dcphase", _) => ind::ht_dcphase_state(&series_f64(df, series, 0, "close").ok()?),
        ("ht_sine", _) => ind::ht_sine_state(&series_f64(df, series, 0, "close").ok()?),
        ("ht_trendline", _) => ind::ht_trendline_state(&series_f64(df, series, 0, "close").ok()?),
        ("ht_trendmode", _) => ind::ht_trendmode_state(&series_f64(df, series, 0, "close").ok()?),
        ("mama", _) => ind::mama_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_f64(args, 0, 0.5).ok()?,
            arg_f64(args, 1, 0.05).ok()?,
        ),

        // Index family — carry the incremental tracker's final running extreme
        // `[idx_abs, value]`. Captured at first compute, where `origin == 0`, so the
        // stored index is original-absolute (stable across a later slice).
        ("maxindex", _) | ("minmaxindex", Some("max")) => {
            ind::maxindex_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, Some(30)).ok()?)
        }
        ("minindex", _) | ("minmaxindex", Some("min")) => {
            ind::minindex_final_state(&series_f64(df, series, 0, "close").ok()?, arg_usize(args, 0, Some(30)).ok()?)
        }

        // StochRSI — carry the RSI Wilder pair + the recent RSI tail feeding the windows.
        // A recursive-MA `.d` (matype != 0) keeps the fallback (no resume).
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let is_d = line == "d";
            if is_d && arg_usize(args, 3, Some(0)).ok()? != 0 {
                return None;
            }
            ind::stochrsi_final_state(
                &series_f64(df, series, 0, "close").ok()?,
                arg_usize(args, 0, Some(14)).ok()?,
                arg_usize(args, 1, Some(5)).ok()?,
                is_d,
                arg_usize(args, 2, Some(3)).ok()?,
            )
        }

        // KDJ — carry the recursive %K (and %D for `.d`/`.j`). RSV is finite-memory and is
        // recomputed on resume, not carried; the `init` seed only affects the warm-up.
        ("kdj", Some(line @ ("k" | "d" | "j"))) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let period_rsv = arg_usize(args, 0, Some(9)).ok()?;
            let period_k = arg_usize(args, 1, Some(3)).ok()?;
            let want_d = line != "k";
            let (period_d, init) = if want_d {
                (arg_usize(args, 2, Some(3)).ok()?, arg_f64(args, 3, 50.0).ok()?)
            } else {
                (3, arg_f64(args, 2, 50.0).ok()?)
            };
            ind::kdj_final_state(&high, &low, &close, period_rsv, period_k, period_d, init, want_d)
        }

        _ => None,
    }
}

/// Resume `node` from `prev_state` over rows `[from_row, height)`, returning the
/// new-row [`Column`] and the updated state. `None` when `node` has no resume
/// kernel (caller falls back to a full recompute). The values are bit-identical to
/// a fresh full recompute, so writing them into the stale tail keeps the cached
/// column exact.
///
/// `origin` is the original-frame row this (possibly sliced) frame's row 0 maps to
/// (`ComputedMeta::origin`). Recursive *value* indicators ignore it; the
/// absolute-position index family adds it back so emitted positions stay
/// original-absolute across a head-dropping slice.
pub fn execute_resume(
    df: &DataFrame,
    node: &Node,
    prev_state: &[f64],
    from_row: usize,
    origin: usize,
) -> Option<(Column, Vec<f64>)> {
    let (name, sub, args, series) = as_command(node)?;
    let sub = sub.as_deref();
    let close = || series_f64(df, series, 0, "close");
    match (name.as_str(), sub) {
        ("obv", _) => {
            let real = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) = ind::obv_resume(&real, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("ad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            let (vals, st) = ind::ad_resume(&high, &low, &close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("adosc", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            let fast = arg_usize(args, 0, Some(3)).ok()?;
            let slow = arg_usize(args, 1, Some(10)).ok()?;
            let (vals, st) =
                ind::adosc_resume(&high, &low, &close, &volume, fast, slow, from_row, prev_state);
            Some((Column::f64(vals), st))
        }

        // SAR family — resume the state machine from the carried tuple. A resume at
        // `from_row < 2` (the SAR bootstrap needs bars 0 and 1) returns `None` and falls back.
        ("sar", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) = ind::sar_resume(
                &high,
                &low,
                arg_f64(args, 0, 0.02).ok()?,
                arg_f64(args, 1, 0.2).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("sarext", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            // `start_value` (arg 0) only steers the bar-1 bootstrap, never re-run on resume.
            let (vals, st) = ind::sarext_resume(
                &high,
                &low,
                arg_f64(args, 1, 0.0).ok()?,
                arg_f64(args, 2, 0.02).ok()?,
                arg_f64(args, 3, 0.02).ok()?,
                arg_f64(args, 4, 0.2).ok()?,
                arg_f64(args, 5, 0.02).ok()?,
                arg_f64(args, 6, 0.02).ok()?,
                arg_f64(args, 7, 0.2).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // EMA-recursion family — resume each carried sub-EMA from its last value (skipping
        // the SMA seed), bit-identical to the full kernel's steady-state recurrence.
        ("ema", _) => {
            let (vals, st) = ind::ema_resume(&close().ok()?, arg_usize(args, 0, None).ok()?, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("smma", _) => {
            let (vals, st) = ind::smma_resume(&close().ok()?, arg_usize(args, 0, None).ok()?, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("dema", _) => {
            let (vals, st) = ind::dema_resume(&close().ok()?, arg_usize(args, 0, Some(30)).ok()?, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("tema", _) => {
            let (vals, st) = ind::tema_resume(&close().ok()?, arg_usize(args, 0, Some(30)).ok()?, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("t3", _) => {
            let (vals, st) = ind::t3_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(5)).ok()?,
                arg_f64(args, 1, 0.7).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("trix", _) => {
            let (vals, st) = ind::trix_resume(&close().ok()?, arg_usize(args, 0, Some(30)).ok()?, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        // KAMA's sliding-sum resume can decline (short retained head) → None falls back.
        ("kama", _) => {
            let (vals, st) =
                ind::kama_resume(&close().ok()?, arg_usize(args, 0, Some(30)).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }

        ("macd", None) => {
            let (vals, st) = ind::macd_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(12)).ok()?,
                arg_usize(args, 1, Some(26)).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("macd", Some(line @ ("signal" | "histogram"))) => {
            let (vals, st) = ind::macd_signal_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(12)).ok()?,
                arg_usize(args, 1, Some(26)).ok()?,
                arg_usize(args, 2, Some(9)).ok()?,
                line == "histogram",
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("macdfix", None) => {
            let (vals, st) = ind::macd_resume(&close().ok()?, 12, 26, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("macdfix", Some(line @ ("signal" | "histogram"))) => {
            let (vals, st) = ind::macd_signal_resume(
                &close().ok()?,
                12,
                26,
                arg_usize(args, 0, Some(9)).ok()?,
                line == "histogram",
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }

        // Wilder-smoothing family — resume the running average(s) over the new rows.
        // Each reads only `…[from_row-1..]` (the per-bar term needs the prior bar), so a
        // resume at `from_row == 0` returns `None` (falls back). DM/DI/DX/ADX/ADXR pull
        // high/low(/close); RSI/CMO/ATR/NATR their named series.
        ("rsi", _) => {
            let (vals, st) = ind::rsi_resume(&close().ok()?, arg_usize(args, 0, None).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("cmo", _) => {
            let (vals, st) =
                ind::cmo_resume(&close().ok()?, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("atr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) =
                ind::atr_resume(&high, &low, &close, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("natr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) =
                ind::natr_resume(&high, &low, &close, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("plus_dm", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) =
                ind::plus_dm_resume(&high, &low, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("minus_dm", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) =
                ind::minus_dm_resume(&high, &low, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("plus_di", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::plus_di_resume(
                &high, &low, &close, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minus_di", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::minus_di_resume(
                &high, &low, &close, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("dx", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) =
                ind::dx_resume(&high, &low, &close, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("adx", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) =
                ind::adx_resume(&high, &low, &close, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("adxr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) =
                ind::adxr_resume(&high, &low, &close, arg_usize(args, 0, Some(14)).ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }

        // Hilbert-transform family — reconstruct the shared core + per-output tail and
        // step the recurrence over the new rows. A resume at/under the core warm-up
        // (or, for the price-windowed trendline/trendmode, before a full dominant-cycle
        // window is visible) returns `None` and falls back to the full recompute.
        ("ht_dcperiod", _) => {
            let (vals, st) = ind::ht_dcperiod_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_phasor", sub) => {
            let (vals, st) =
                ind::ht_phasor_resume(&close().ok()?, sub == Some("quadrature"), from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_dcphase", _) => {
            let (vals, st) = ind::ht_dcphase_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_sine", sub) => {
            let (vals, st) =
                ind::ht_sine_resume(&close().ok()?, sub == Some("leadsine"), from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_trendline", _) => {
            let (vals, st) = ind::ht_trendline_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_trendmode", _) => {
            let (vals, st) = ind::ht_trendmode_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("mama", sub) => {
            let (vals, st) = ind::mama_resume(
                &close().ok()?,
                arg_f64(args, 0, 0.5).ok()?,
                arg_f64(args, 1, 0.05).ok()?,
                sub == Some("fama"),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // Index family — windowed arg-extreme emitting ABSOLUTE positions. The carried
        // state is the incremental tracker's running extreme `[idx_abs, value]`; `origin`
        // rebases sub-frame positions back to original-absolute. minmaxindex.max / .min
        // are exactly maxindex / minindex (see `execute`).
        ("maxindex", _) | ("minmaxindex", Some("max")) => {
            let (vals, st) = ind::maxindex_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(30)).ok()?,
                from_row,
                origin,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minindex", _) | ("minmaxindex", Some("min")) => {
            let (vals, st) = ind::minindex_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(30)).ok()?,
                from_row,
                origin,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // StochRSI — a windowed %K (and SMA `.d`) of the Wilder-recursive RSI; resume by
        // carrying the RSI Wilder pair + the recent RSI values feeding the windows. Only
        // the canonical SMA `.d` (matype 0) resumes; a recursive-MA `.d` declines.
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let is_d = line == "d";
            if is_d && arg_usize(args, 3, Some(0)).ok()? != 0 {
                return None; // non-SMA `.d` smoothing is recursive — fall back.
            }
            let (vals, st) = ind::stochrsi_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(14)).ok()?,
                arg_usize(args, 1, Some(5)).ok()?,
                is_d,
                arg_usize(args, 2, Some(3)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // KDJ — resume the recursive %K (+ %D for `.d`/`.j`) from the carried state; RSV is
        // finite-memory, recomputed over the windowed high/low/close tail.
        ("kdj", Some(line @ ("k" | "d" | "j"))) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let period_rsv = arg_usize(args, 0, Some(9)).ok()?;
            let period_k = arg_usize(args, 1, Some(3)).ok()?;
            let kline = match line {
                "k" => ind::KdjLine::K,
                "d" => ind::KdjLine::D,
                _ => ind::KdjLine::J,
            };
            let period_d = if line == "k" { 3 } else { arg_usize(args, 2, Some(3)).ok()? };
            let (vals, st) = ind::kdj_resume(
                &high, &low, &close, period_rsv, period_k, period_d, kline, from_row, prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        _ => None,
    }
}

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
        _ => return Err(VolasError::Value(format!("invalid time frame unit in '{s}'"))),
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
            "ma:0", "ema:0", "smma:0", "wma:0", "dema:0", "tema:0", "trima:0", "t3:0",
            "kama:0", "mom:0", "roc:0", "rocp:0", "rocr:0", "rocr100:0", "willr:0", "cci:0",
            "cmo:0", "mfi:0", "trix:0", "midpoint:0", "midprice:0", "atr:0", "natr:0", "rsi:0",
            "plus_dm:0", "minus_dm:0", "plus_di:0", "minus_di:0", "dx:0", "adx:0", "adxr:0",
            "aroon.up:0", "aroonosc:0", "sum:0", "maxindex:0", "minindex:0", "minmax.min:0",
            "minmaxindex.min:0", "linearreg:0", "linearreg_slope:0", "linearreg_intercept:0",
            "linearreg_angle:0", "tsf:0", "var:0", "stddev:0", "llv:0", "hhv:0", "boll:0",
            "bbw:0", "accbands:0", "correl:0@close,close", "beta:0@close,close",
            // period larger than the frame trips the `period > n` arm.
            "ma:99", "atr:99", "rsi:99", "sum:99", "linearreg:99",
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
            "obv", "ad", "adosc:3,10", "sar", "sarext", "ma:5", "tr", "avgprice",
            "ht_dcperiod", "mama",
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
            "style.doji", "style.engulfing", "style.morningstar", "style.hikkake",
            "style.hikkakemod", "style.concealbabyswall", "style.3whitesoldiers",
            "style.risefall3methods", "style.breakaway", "style.mathold",
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
            "style:bullish",
            "style:bearish",
            "repeat:2@(style:bullish)",
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
        assert!(is_err("MACD.bogus"), "case-insensitive name, still bad sub (P6)");
        assert!(is_err("kdj:9"), "missing required sub-command");
        assert!(is_err("macd.bogus"), "unknown sub-command");
        assert!(is_err("ma:abc"), "non-integer arg");
        assert!(is_err("style:cartoon"), "invalid style descriptor");
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
