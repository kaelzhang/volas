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
                    _ => unreachable!(),
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
                    _ => unreachable!(),
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
        _ => unreachable!("non-comparison op in apply_cmp"),
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

fn series_bool(df: &DataFrame, series: &[Node], i: usize) -> Result<Vec<bool>> {
    let node = series
        .get(i)
        .ok_or_else(|| VolasError::Value("a boolean series argument is required".into()))?;
    match execute(df, node)? {
        Column::Bool(v) => Ok(v.to_vec()),
        other => Ok(other.to_f64_vec().iter().map(|&x| x != 0.0).collect()),
    }
}

fn arg_str<'a>(args: &'a [Option<String>], i: usize) -> Result<&'a str> {
    arg_at(args, i).ok_or_else(|| VolasError::Value(format!("missing required argument #{i}")))
}

// --- command dispatch -------------------------------------------------------

fn exec_command(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    args: &[Option<String>],
    series: &[Node],
) -> Result<Column> {
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
        ("ma", _) => f64col(ind::ma(&close(0)?, arg_usize(args, 0, None)?)),
        ("ema", _) => f64col(ind::ema(&close(0)?, arg_usize(args, 0, None)?)),
        ("smma", _) => f64col(ind::smma(&close(0)?, arg_usize(args, 0, None)?)),
        ("wma", _) => f64col(ind::wma(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("dema", _) => f64col(ind::dema(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("tema", _) => f64col(ind::tema(&close(0)?, arg_usize(args, 0, Some(30))?)),

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
        ("style", _) => {
            let style = match arg_str(args, 0)? {
                "bullish" => ind::Style::Bullish,
                "bearish" => ind::Style::Bearish,
                other => {
                    return Err(VolasError::Value(format!(
                        "style should be 'bullish' or 'bearish', got '{other}'"
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
        ("cci", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::cci(&high, &low, &close, arg_usize(args, 0, Some(14))?))
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

        (other, _) => Err(VolasError::Value(format!("unknown command '{other}'"))),
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

    /// A 30-row OHLCV frame so every indicator produces a full-length result.
    fn ohlcv() -> DataFrame {
        let close: Vec<f64> = (1..=30).map(|i| i as f64).collect();
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
        assert!(is_err("ma:5,6"), "too many args");
        assert!(is_err("frobnicate:5"), "unknown command");
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
}
