//! Lookback (warm-up) computation for a directive — the minimum number of prior
//! rows needed before an indicator produces a valid value.

use crate::types::{Command, Node};

fn arg(args: &[Option<String>], i: usize, default: usize) -> usize {
    args.get(i)
        .and_then(|o| o.as_deref())
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Lookback of a TA-Lib MA-type over `period` (mirrors `exec::ma_typed`): DEMA is
/// `2·(period-1)`, TEMA `3·(period-1)`, KAMA `period`, T3 `6·(period-1)`, and the
/// rest (SMA/EMA/WMA/TRIMA) `period-1`.
fn ma_lookback(period: usize, matype: usize) -> usize {
    match matype {
        3 => 2 * period.saturating_sub(1),
        4 => 3 * period.saturating_sub(1),
        6 => period,
        8 => 6 * period.saturating_sub(1),
        _ => period.saturating_sub(1),
    }
}

/// The command's own lookback (ignoring its series operands). `None` for a
/// plain column name.
fn own_lookback(name: &str, sub: Option<&str>, args: &[Option<String>]) -> Option<usize> {
    let lb = match name {
        "ema" | "smma" => arg(args, 0, 1).saturating_sub(1),
        "ma" => ma_lookback(arg(args, 0, 1), arg(args, 1, 0)),
        // apo/ppo warm up with the slower MA of the chosen type.
        "apo" | "ppo" => ma_lookback(arg(args, 1, 26), arg(args, 2, 0)),
        "wma" | "trima" => arg(args, 0, 30).saturating_sub(1),
        "dema" => 2 * arg(args, 0, 30).saturating_sub(1),
        "tema" => 3 * arg(args, 0, 30).saturating_sub(1),
        "t3" => 6 * arg(args, 0, 5).saturating_sub(1),
        "kama" => arg(args, 0, 30),
        "boll" | "bbw" | "accbands" => arg(args, 0, 20).saturating_sub(1),
        "macd" => match sub {
            None => arg(args, 0, 12).max(arg(args, 1, 26)).saturating_sub(1),
            Some(_) => arg(args, 0, 12).max(arg(args, 1, 26)) + arg(args, 2, 9) - 2,
        },
        "bbi" => arg(args, 0, 3)
            .max(arg(args, 1, 6))
            .max(arg(args, 2, 12))
            .max(arg(args, 3, 24)),
        "tr" => 1,
        "atr" => arg(args, 0, 14),
        "llv" | "hhv" | "donchian" | "rsv" => arg(args, 0, 1).saturating_sub(1),
        "kdj" => arg(args, 0, 9) * 3,
        "rsi" => arg(args, 0, 14),
        "hv" => arg(args, 0, 1),
        "increase" | "repeat" => arg(args, 0, 1).saturating_sub(1),
        "style" => 0,
        "change" => arg(args, 0, 2).saturating_sub(1),
        "mom" | "roc" | "rocp" | "rocr" | "rocr100" => arg(args, 0, 10),
        "willr" | "midpoint" | "midprice" => arg(args, 0, 14).saturating_sub(1),
        "cmo" | "natr" => arg(args, 0, 14),
        "cci" => arg(args, 0, 14).saturating_sub(1),
        "mfi" => arg(args, 0, 14),
        "trix" => 3 * arg(args, 0, 30).saturating_sub(1) + 1,
        "aroon" | "aroonosc" => arg(args, 0, 14),
        "sum" | "maxindex" | "minindex" | "minmax" | "minmaxindex" => {
            arg(args, 0, 30).saturating_sub(1)
        }
        "bop" => 0,
        "linearreg" | "linearreg_slope" | "linearreg_intercept" | "linearreg_angle" | "tsf" => {
            arg(args, 0, 14).saturating_sub(1)
        }
        "var" | "stddev" => arg(args, 0, 5).saturating_sub(1),
        "obv" | "ad" => 0,
        "adosc" => arg(args, 0, 3).max(arg(args, 1, 10)).saturating_sub(1),
        "avgprice" | "medprice" | "typprice" | "wclprice" => 0,
        _ => return None,
    };
    Some(lb)
}

/// Lookback for a parsed directive: a command's own lookback plus the largest
/// lookback among its series operands; operators take the max of their operands.
pub fn lookback(node: &Node) -> usize {
    match node {
        Node::Scalar(_) => 0,
        Node::Name(name) => own_lookback(name, None, &[]).unwrap_or(0),
        Node::Command(Command {
            name,
            sub,
            args,
            series,
        }) => {
            let own = own_lookback(name, sub.as_deref(), args).unwrap_or(0);
            let series_max = series.iter().map(lookback).max().unwrap_or(0);
            own + series_max
        }
        Node::Unary { operand, .. } => lookback(operand),
        Node::Binary { left, right, .. } => lookback(left).max(lookback(right)),
    }
}
