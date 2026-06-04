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
/// `2·(period-1)`, TEMA `3·(period-1)`, KAMA `period`, T3 `6·(period-1)`, MAMA a
/// fixed `32` (period-independent), and the rest (SMA/EMA/WMA/TRIMA) `period-1`.
pub(crate) fn ma_lookback(period: usize, matype: usize) -> usize {
    match matype {
        3 => 2 * period.saturating_sub(1),
        4 => 3 * period.saturating_sub(1),
        6 => period,
        7 => 32,
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
        // mavp warms up over the maximum period's MA (args: min, max, matype).
        "mavp" => ma_lookback(arg(args, 1, 30), arg(args, 2, 0)),
        "macdext" => {
            let line = ma_lookback(arg(args, 0, 12), arg(args, 1, 0))
                .max(ma_lookback(arg(args, 2, 26), arg(args, 3, 0)));
            match sub {
                None => line,
                _ => line + ma_lookback(arg(args, 4, 9), arg(args, 5, 0)),
            }
        }
        "wma" | "trima" => arg(args, 0, 30).saturating_sub(1),
        "dema" => 2 * arg(args, 0, 30).saturating_sub(1),
        "tema" => 3 * arg(args, 0, 30).saturating_sub(1),
        "t3" => 6 * arg(args, 0, 5).saturating_sub(1),
        "kama" => arg(args, 0, 30),
        "sar" | "sarext" => 1,
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
        // style.<x>: candlestick patterns warm up over their candle-settings avg period
        // (from the compute registry); bullish/bearish need none. (`cdl` resolves here too.)
        "style" | "cdl" => sub
            .and_then(|p| volas_compute::indicators::candle_pattern(p).map(|(_, lb)| lb))
            .unwrap_or(0),
        "change" => arg(args, 0, 2).saturating_sub(1),
        "mom" | "roc" | "rocp" | "rocr" | "rocr100" => arg(args, 0, 10),
        "willr" | "midpoint" | "midprice" => arg(args, 0, 14).saturating_sub(1),
        "cmo" | "natr" => arg(args, 0, 14),
        "cci" | "imi" => arg(args, 0, 14).saturating_sub(1),
        "mfi" => arg(args, 0, 14),
        "ultosc" => arg(args, 0, 7).max(arg(args, 1, 14)).max(arg(args, 2, 28)),
        // %K (lookback fastk-1) then matype-MA smoothing stage(s).
        "stoch" => {
            let base = arg(args, 0, 5).saturating_sub(1) + ma_lookback(arg(args, 1, 3), arg(args, 2, 0));
            match sub {
                Some("d") => base + ma_lookback(arg(args, 3, 3), arg(args, 4, 0)),
                _ => base,
            }
        }
        "stochf" => {
            let base = arg(args, 0, 5).saturating_sub(1);
            match sub {
                Some("d") => base + ma_lookback(arg(args, 1, 3), arg(args, 2, 0)),
                _ => base,
            }
        }
        "stochrsi" => {
            // RSI lookback (period) + the %K window (fastk_period-1), then the d-line MA.
            let base = arg(args, 0, 14) + arg(args, 1, 5).saturating_sub(1);
            match sub {
                Some("d") => base + ma_lookback(arg(args, 2, 3), arg(args, 3, 0)),
                _ => base,
            }
        }
        "plus_dm" | "minus_dm" => arg(args, 0, 14).saturating_sub(1),
        "plus_di" | "minus_di" | "dx" => arg(args, 0, 14),
        "adx" => 2 * arg(args, 0, 14) - 1,
        "adxr" => 3 * arg(args, 0, 14) - 2,
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
        "correl" => arg(args, 0, 30).saturating_sub(1),
        "beta" => arg(args, 0, 5),
        "obv" | "ad" => 0,
        "adosc" => arg(args, 0, 3).max(arg(args, 1, 10)).saturating_sub(1),
        "avgprice" | "medprice" | "typprice" | "wclprice" => 0,
        // Hilbert-transform suite: fixed warm-up (DCPERIOD/PHASOR/MAMA = 32; the
        // phase-dependent DCPHASE/SINE/TRENDLINE/TRENDMODE need 63).
        "ht_dcperiod" | "ht_phasor" | "mama" => 32,
        "ht_dcphase" | "ht_sine" | "ht_trendline" | "ht_trendmode" => 63,
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
