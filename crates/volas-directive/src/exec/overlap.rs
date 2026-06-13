//! Command dispatch — moving averages, SAR, and oscillator / overlap studies
//! (group 1 of the exec_command family chain).

use super::*;
use volas_core::Result;
use crate::types::{ArgValue, Ast};
use volas_compute::indicators as ind;
use volas_core::{Column, DataFrame};

pub(super) fn dispatch(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    args: &[ArgValue],
    series: &[Ast],
) -> Result<Column> {
    let close = |i| series_f64(df, series, i, "close");
    let f64col = |v: Vec<f64>| Ok(Column::f64(v));
    match (name, sub) {
        ("ma", _) => f64col(ma_typed(
            &close(0)?,
            arg_usize(args, 0),
            arg_usize(args, 1),
        )),
        ("ema", _) => f64col(ind::ema(&close(0)?, arg_usize(args, 0))),
        ("smma", _) => f64col(ind::smma(&close(0)?, arg_usize(args, 0))),
        ("wma", _) => f64col(ind::wma(&close(0)?, arg_usize(args, 0))),
        ("dema", _) => f64col(ind::dema(&close(0)?, arg_usize(args, 0))),
        ("tema", _) => f64col(ind::tema(&close(0)?, arg_usize(args, 0))),
        ("trima", _) => f64col(ind::trima(&close(0)?, arg_usize(args, 0))),
        ("t3", _) => f64col(ind::t3(
            &close(0)?,
            arg_usize(args, 0),
            arg_f64(args, 1),
        )),
        ("kama", _) => f64col(ind::kama(&close(0)?, arg_usize(args, 0))),
        // MA with a per-row variable period: the period for row i is the (truncated,
        // clamped) value of the required second `periods` series. Each distinct period's
        // MA is computed once and cached.
        ("mavp", _) => {
            let real = close(0)?;
            let periods = series_f64_required(df, series, 1)?;
            // min_p <= max_p is guaranteed by validate_cross_args (which runs
            // before dispatch), so the i64::clamp below cannot panic.
            let min_p = arg_usize(args, 0);
            let max_p = arg_usize(args, 1);
            let matype = arg_usize(args, 2);
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
                    let sub = ma_typed(&real[start..], p, matype);
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
            f64col(ind::sar(
                &high,
                &low,
                arg_f64(args, 0),
                arg_f64(args, 1),
            ))
        }
        ("sarext", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::sarext(
                &high,
                &low,
                arg_f64(args, 0),
                arg_f64(args, 1),
                arg_f64(args, 2),
                arg_f64(args, 3),
                arg_f64(args, 4),
                arg_f64(args, 5),
                arg_f64(args, 6),
                arg_f64(args, 7),
            ))
        }
        // Price oscillators: fast MA - slow MA, of a chosen MA type (default SMA).
        ("apo", _) => {
            let data = close(0)?;
            let mt = arg_usize(args, 2);
            let f = ma_typed(&data, arg_usize(args, 0), mt);
            let s = ma_typed(&data, arg_usize(args, 1), mt);
            f64col((0..data.len()).map(|i| f[i] - s[i]).collect())
        }
        // MACDEXT: a MACD whose fast/slow/signal MAs each take a configurable type
        // (matypes default to SMA, not macd's fixed EMA). The line is emitted at its
        // natural start (best practice, like macd); signal/histogram follow.
        ("macdext", sub) => {
            let data = close(0)?;
            let f = ma_typed(
                &data,
                arg_usize(args, 0),
                arg_usize(args, 1),
            );
            let s = ma_typed(
                &data,
                arg_usize(args, 2),
                arg_usize(args, 3),
            );
            let line: Vec<f64> = (0..data.len()).map(|i| f[i] - s[i]).collect();
            match sub {
                None => f64col(line),
                _ => {
                    let signal = ma_typed(
                        &line,
                        arg_usize(args, 4),
                        arg_usize(args, 5),
                    );
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
            let mt = arg_usize(args, 2);
            let f = ma_typed(&data, arg_usize(args, 0), mt);
            let s = ma_typed(&data, arg_usize(args, 1), mt);
            f64col(
                (0..data.len())
                    .map(|i| (f[i] - s[i]) / s[i] * 100.0)
                    .collect(),
            )
        }

        // Group E China-market wrappers (formula-equivalent to apo/ppo, fixed to SMA).
        // bias:N ≡ ppo:1,N,0 — the percentage deviation of close from its N-period SMA.
        ("bias", _) => {
            let data = close(0)?;
            let f = ma_typed(&data, 1, 0);
            let s = ma_typed(&data, arg_usize(args, 0), 0);
            f64col((0..data.len()).map(|i| (f[i] - s[i]) / s[i] * 100.0).collect())
        }
        // dma's DDD line ≡ apo:fast,slow,0; dma.ama is the M-period SMA of that line.
        ("dma", sub) => {
            let data = close(0)?;
            let f = ma_typed(&data, arg_usize(args, 0), 0);
            let s = ma_typed(&data, arg_usize(args, 1), 0);
            let line: Vec<f64> = (0..data.len()).map(|i| f[i] - s[i]).collect();
            match sub {
                None => f64col(line),
                _ => f64col(ma_typed(&line, arg_usize(args, 2), 0)),
            }
        }

        // Group B convention-sensitive indicators (gap report §9).
        ("vortex", sub) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::vortex(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                sub == Some("plus"),
            ))
        }
        ("brar", Some("br")) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::brar_br(&high, &low, &close, arg_usize(args, 0)))
        }
        ("brar", _) => {
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            f64col(ind::brar_ar(&open, &high, &low, arg_usize(args, 0)))
        }
        ("vr", _) => {
            let close = series_f64(df, series, 0, "close")?;
            let volume = series_f64(df, series, 1, "volume")?;
            f64col(ind::vr(&close, &volume, arg_usize(args, 0)))
        }
        ("coppock", _) => f64col(ind::coppock(
            &close(0)?,
            arg_usize(args, 0),
            arg_usize(args, 1),
            arg_usize(args, 2),
        )),
        ("relative_vigor", sub) => {
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            let close = series_f64(df, series, 3, "close")?;
            let n = arg_usize(args, 0);
            if sub == Some("signal") {
                f64col(ind::relative_vigor_signal(&open, &high, &low, &close, n))
            } else {
                f64col(ind::relative_vigor(&open, &high, &low, &close, n))
            }
        }
        ("dkx", sub) => {
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            let close = series_f64(df, series, 3, "close")?;
            if sub == Some("ma") {
                f64col(ind::dkx_ma(&open, &high, &low, &close, arg_usize(args, 0)))
            } else {
                f64col(ind::dkx(&open, &high, &low, &close))
            }
        }
        ("wvad", _) => {
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            let close = series_f64(df, series, 3, "close")?;
            let volume = series_f64(df, series, 4, "volume")?;
            f64col(ind::wvad(&open, &high, &low, &close, &volume, arg_usize(args, 0)))
        }
        ("cdp", sub) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let line = match sub {
                Some("ah") => ind::CdpLine::Ah,
                Some("nh") => ind::CdpLine::Nh,
                Some("nl") => ind::CdpLine::Nl,
                Some("al") => ind::CdpLine::Al,
                _ => ind::CdpLine::Cdp,
            };
            f64col(ind::cdp(&high, &low, &close, line))
        }
        ("mike", sub) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let line = match sub {
                Some("midr") => ind::MikeLine::MidR,
                Some("strongr") => ind::MikeLine::StrongR,
                Some("weaks") => ind::MikeLine::WeakS,
                Some("mids") => ind::MikeLine::MidS,
                Some("strongs") => ind::MikeLine::StrongS,
                _ => ind::MikeLine::WeakR,
            };
            f64col(ind::mike(&high, &low, &close, arg_usize(args, 0), line))
        }
        ("keltner", sub) => {
            let ema_period = arg_usize(args, 0);
            match sub {
                None => f64col(ind::ema(&series_f64(df, series, 0, "close")?, ema_period)),
                _ => {
                    let high = series_f64(df, series, 0, "high")?;
                    let low = series_f64(df, series, 1, "low")?;
                    let close = series_f64(df, series, 2, "close")?;
                    f64col(ind::keltner_band(
                        &close,
                        &high,
                        &low,
                        ema_period,
                        arg_usize(args, 1),
                        arg_f64(args, 2),
                        sub == Some("upper"),
                    ))
                }
            }
        }
        ("stoch_momentum", sub) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let k = arg_usize(args, 0);
            let d = arg_usize(args, 1);
            if sub == Some("signal") {
                f64col(ind::stoch_momentum_signal(
                    &high,
                    &low,
                    &close,
                    k,
                    d,
                    arg_usize(args, 2),
                ))
            } else {
                f64col(ind::stoch_momentum(&high, &low, &close, k, d))
            }
        }
        ("ttm_squeeze", sub) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let n = arg_usize(args, 0);
            if sub == Some("on") {
                f64col(ind::ttm_squeeze_on(
                    &high,
                    &low,
                    &close,
                    n,
                    arg_f64(args, 1),
                    arg_f64(args, 2),
                ))
            } else {
                f64col(ind::ttm_squeeze_momentum(&high, &low, &close, n))
            }
        }
        ("pivot_points", sub) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let line = match sub {
                Some("r1") => ind::PivotLine::R1,
                Some("s1") => ind::PivotLine::S1,
                Some("r2") => ind::PivotLine::R2,
                Some("s2") => ind::PivotLine::S2,
                Some("r3") => ind::PivotLine::R3,
                Some("s3") => ind::PivotLine::S3,
                _ => ind::PivotLine::Pp,
            };
            f64col(ind::pivot_points(&high, &low, &close, line))
        }
        ("ichimoku", sub) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let line = match sub {
                Some("kijun") => ind::IchimokuLine::Kijun,
                Some("senkou_a") => ind::IchimokuLine::SenkouA,
                Some("senkou_b") => ind::IchimokuLine::SenkouB,
                Some("chikou") => ind::IchimokuLine::Chikou,
                _ => ind::IchimokuLine::Tenkan,
            };
            f64col(ind::ichimoku(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                arg_usize(args, 1),
                arg_usize(args, 2),
                line,
            ))
        }
        ("wad", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::wad(&high, &low, &close))
        }
        ("asi", _) => {
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            let close = series_f64(df, series, 3, "close")?;
            f64col(ind::asi(&open, &high, &low, &close, arg_f64(args, 0)))
        }
        ("supertrend", sub) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::supertrend(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                arg_f64(args, 1),
                sub == Some("direction"),
            ))
        }

        _ => super::momentum::dispatch(df, name, sub, args, series),
    }
}
