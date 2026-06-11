//! Command dispatch — MACD, bands, stochastic & candlestick patterns, and the
//! momentum oscillators (group 2 of the exec_command family chain).

use super::*;
use volas_core::Result;
use crate::types::Node;
use volas_compute::indicators as ind;
use volas_core::{Column, DataFrame};

pub(super) fn dispatch(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    args: &[Option<String>],
    series: &[Node],
) -> Result<Column> {
    let close = |i| series_f64(df, series, i, "close");
    let f64col = |v: Vec<f64>| Ok(Column::f64(v));
    let boolcol = |v: Vec<bool>| Ok(Column::bool(v));
    match (name, sub) {
        ("macd", None) => f64col(ind::macd(
            &close(0)?,
            arg_usize(args, 0, Some(12))?,
            arg_usize(args, 1, Some(26))?,
        )),
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
        ("macdfix", Some("signal")) => f64col(ind::macd_signal(
            &close(0)?,
            12,
            26,
            arg_usize(args, 0, Some(9))?,
        )),
        ("macdfix", Some("histogram")) => f64col(ind::macd_histogram(
            &close(0)?,
            12,
            26,
            arg_usize(args, 0, Some(9))?,
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

        ("accbands", None) => f64col(ind::accbands_middle(
            &close(0)?,
            arg_usize(args, 0, Some(20))?,
        )),
        ("accbands", Some("upper")) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::accbands_upper(
                &high,
                &low,
                arg_usize(args, 0, Some(20))?,
            ))
        }
        ("accbands", Some("lower")) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::accbands_lower(
                &high,
                &low,
                arg_usize(args, 0, Some(20))?,
            ))
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

        ("llv", _) => f64col(ind::llv(
            &series_f64(df, series, 0, "low")?,
            arg_usize(args, 0, None)?,
        )),
        ("hhv", _) => f64col(ind::hhv(
            &series_f64(df, series, 0, "high")?,
            arg_usize(args, 0, None)?,
        )),

        ("donchian", None) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::donchian(&high, &low, arg_usize(args, 0, None)?))
        }
        ("donchian", Some("upper")) => f64col(ind::hhv(
            &series_f64(df, series, 0, "high")?,
            arg_usize(args, 0, None)?,
        )),
        ("donchian", Some("lower")) => f64col(ind::llv(
            &series_f64(df, series, 0, "low")?,
            arg_usize(args, 0, None)?,
        )),

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

        ("style", Some(sub @ ("bullish" | "bearish"))) => {
            let style = match sub {
                "bullish" => ind::Style::Bullish,
                "bearish" => ind::Style::Bearish,
                _ => unreachable!("style color sub-command is validated by command_spec"), // LCOV_EXCL_LINE
            };
            let open = series_f64(df, series, 0, "open")?;
            let close = series_f64(df, series, 1, "close")?;
            boolcol(ind::style(style, &open, &close))
        }
        ("repeat", _) => boolcol(ind::repeat(
            &series_bool(df, series, 0)?,
            arg_usize(args, 0, Some(1))?,
        )),
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
            f64col(ind::willr(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14))?,
            ))
        }
        ("cmo", _) => f64col(ind::cmo(&close(0)?, arg_usize(args, 0, Some(14))?)),
        ("mfi", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let volume = series_f64(df, series, 3, "volume")?;
            f64col(ind::mfi(
                &high,
                &low,
                &close,
                &volume,
                arg_usize(args, 0, Some(14))?,
            ))
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
            let fastk_period = arg_usize(args, 0, Some(5))?;
            let slowk_period = arg_usize(args, 1, Some(3))?;
            let slowk_matype = arg_usize(args, 2, Some(0))?;
            let slowd_period = arg_usize(args, 3, Some(3))?;
            let slowd_matype = arg_usize(args, 4, Some(0))?;
            if line == "d"
                && fastk_period == 5
                && slowk_period == 3
                && slowk_matype == 0
                && slowd_period == 3
                && slowd_matype == 0
            {
                if let Some(out) = ind::stoch_d_default_sma(&high, &low, &close) {
                    return f64col(out);
                }
            }
            let fastk = ind::stoch_fastk(&high, &low, &close, fastk_period);
            let slowk = ma_typed(&fastk, slowk_period, slowk_matype);
            if line == "k" {
                f64col(slowk)
            } else {
                f64col(ma_typed(&slowk, slowd_period, slowd_matype))
            }
        }
        ("stochf", Some(line @ ("k" | "d"))) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let fastk_period = arg_usize(args, 0, Some(5))?;
            let fastd_period = arg_usize(args, 1, Some(3))?;
            let fastd_matype = arg_usize(args, 2, Some(0))?;
            if line == "d" && fastk_period == 5 && fastd_period == 3 && fastd_matype == 0 {
                if let Some(out) = ind::stochf_d_default_sma(&high, &low, &close) {
                    return f64col(out);
                }
            }
            let fastk = ind::stoch_fastk(&high, &low, &close, fastk_period);
            if line == "k" {
                f64col(fastk)
            } else {
                f64col(ma_typed(&fastk, fastd_period, fastd_matype))
            }
        }
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let close = close(0)?;
            let rsi_period = arg_usize(args, 0, Some(14))?;
            let fastk_period = arg_usize(args, 1, Some(5))?;
            let fastd_period = arg_usize(args, 2, Some(3))?;
            let fastd_matype = arg_usize(args, 3, Some(0))?;
            if line == "d"
                && rsi_period == 14
                && fastk_period == 5
                && fastd_period == 3
                && fastd_matype == 0
            {
                if let Some(out) = ind::stochrsi_d_default_sma(&close) {
                    return f64col(out);
                }
            }
            let fastk = ind::stochrsi_fastk(&close, rsi_period, fastk_period);
            if line == "k" {
                f64col(fastk)
            } else {
                f64col(ma_typed(&fastk, fastd_period, fastd_matype))
            }
        }

        _ => super::cycle::dispatch(df, name, sub, args, series),
    }
}
