//! Execute a directive AST against a [`DataFrame`], producing a [`Column`].

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

fn series_f64(df: &DataFrame, series: &[Node], i: usize, default_col: &str) -> Result<Vec<f64>> {
    match series.get(i) {
        Some(Node::Name(s)) if s.is_empty() => Ok(df.column(default_col)?.to_f64_vec()),
        Some(node) => Ok(execute(df, node)?.to_f64_vec()),
        None => Ok(df.column(default_col)?.to_f64_vec()),
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

fn arg_str<'a>(args: &'a [Option<String>], i: usize) -> Result<&'a str> {
    arg_at(args, i).ok_or_else(|| VolasError::Value(format!("missing required argument #{i}")))
}

// --- command dispatch -------------------------------------------------------

/// Normalise a sub-command alias to its canonical form (or `None` for the main).
fn canon_sub(name: &str, sub: Option<&str>) -> Option<String> {
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

fn exec_command(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    args: &[Option<String>],
    series: &[Node],
) -> Result<Column> {
    let sub = canon_sub(name, sub);
    let sub = sub.as_deref();
    let close = |i| series_f64(df, series, i, "close");
    let f64col = |v: Vec<f64>| Ok(Column::f64(v));
    let boolcol = |v: Vec<bool>| Ok(Column::bool(v));

    match (name, sub) {
        ("ma", _) => f64col(ind::ma(&close(0)?, arg_usize(args, 0, None)?)),
        ("ema", _) => f64col(ind::ema(&close(0)?, arg_usize(args, 0, None)?)),
        ("smma", _) => f64col(ind::smma(&close(0)?, arg_usize(args, 0, None)?)),

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
            let open = series_f64(df, series, 0, "open")?;
            let close = series_f64(df, series, 1, "close")?;
            boolcol(ind::style(arg_str(args, 0)?, &open, &close)?)
        }
        ("repeat", _) => boolcol(ind::repeat(&series_bool(df, series, 0)?, arg_usize(args, 0, Some(1))?)),
        ("change", _) => f64col(ind::change(&close(0)?, arg_usize(args, 0, Some(2))?)),

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
}
