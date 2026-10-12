//! Command dispatch — moving averages, SAR, and oscillator / overlap studies
//! (group 1 of the exec_command family chain).

use super::*;
use volas_core::Result;
use crate::types::ArgValue;
use volas_compute::raw::indicators as ind;
use volas_core::Column;

pub(super) fn dispatch(
    ops: &Operands,
    name: &str,
    sub: Option<&str>,
    args: &[ArgValue],
) -> Result<Column> {
    let close = |i| ops.f64(i, "close");
    let f64col = |v: Vec<f64>| Ok(Column::f64(v));
    match (name, sub) {
        ("ma", _) => f64col(ma_typed(
            &close(0)?,
            arg_usize(args, 0),
            arg_usize(args, 1),
            true,
            ops.origin(),
        )),
        ("ema", _) => f64col(ind::ema(&close(0)?, arg_usize(args, 0))),
        ("smma", _) => f64col(ind::smma(&close(0)?, arg_usize(args, 0))),
        ("wma", _) => f64col(ind::wma(&close(0)?, arg_usize(args, 0), ops.origin())),
        ("vwma", _) => f64col(ind::vwma(
            &close(0)?,
            &ops.f64(1, "volume")?,
            arg_usize(args, 0),
            ops.origin(),
        )),
        ("alma", _) => f64col(ind::alma(
            &close(0)?,
            arg_usize(args, 0),
            arg_f64(args, 1),
            arg_f64(args, 2),
        )),
        ("hma", _) => f64col(ind::hma(&close(0)?, arg_usize(args, 0), ops.origin())),
        ("swma", _) => f64col(ind::swma(&close(0)?)),
        ("dema", _) => f64col(ind::dema(&close(0)?, arg_usize(args, 0))),
        ("tema", _) => f64col(ind::tema(&close(0)?, arg_usize(args, 0))),
        ("trima", _) => f64col(ind::trima(&close(0)?, arg_usize(args, 0), ops.origin())),
        ("t3", _) => f64col(ind::t3(
            &close(0)?,
            arg_usize(args, 0),
            arg_f64(args, 1),
        )),
        ("kama", _) => f64col(ind::kama(&close(0)?, arg_usize(args, 0), ops.origin())),
        // MA with a per-row variable period: the period for row i is the (truncated,
        // clamped) value of the required second `periods` series. Each distinct period's
        // MA is computed once and cached.
        ("mavp", _) => {
            let real = close(0)?;
            let periods = ops.f64_required(1)?;
            // min_p <= max_p is guaranteed by validate_cross_args (which runs
            // before dispatch), so the period clamp cannot panic.
            let min_p = arg_usize(args, 0);
            let max_p = arg_usize(args, 1);
            let matype = arg_usize(args, 2);
            let lb = ind::generic_ma_lookback(max_p, matype);
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
                let Some(p) = ind::mavp_period(periods[i], min_p, max_p) else {
                    continue; // a missing period: the row is missing
                };
                if cache[p].is_none() {
                    let start = lb - ind::generic_ma_lookback(p, matype);
                    let sub = ma_typed(&real[start..], p, matype, true, ops.origin() + start);
                    cache[p] = Some((start, sub));
                }
                let (start, sub) = cache[p].as_ref().unwrap();
                out[i] = sub[i - start];
            }
            f64col(out)
        }
        ("sar", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            f64col(ind::sar(
                &high,
                &low,
                arg_f64(args, 0),
                arg_f64(args, 1),
            ))
        }
        ("sarext", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
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
            let (f, s) = oscillator_mas(&data, (arg_usize(args, 0), mt), (arg_usize(args, 1), mt), ops.origin());
            f64col((0..data.len()).map(|i| f[i] - s[i]).collect())
        }
        // MACDEXT: a MACD whose fast/slow/signal MAs each take a configurable type
        // (matypes default to SMA, not macd's fixed EMA). The line is emitted at its
        // natural start (best practice, like macd); signal/histogram follow.
        ("macdext", sub) => {
            let data = close(0)?;
            let fast = (arg_usize(args, 0), arg_usize(args, 1));
            let (f, s) = oscillator_mas(&data, fast, (arg_usize(args, 2), arg_usize(args, 3)), ops.origin());
            let line: Vec<f64> = (0..data.len()).map(|i| f[i] - s[i]).collect();
            match sub {
                None => f64col(line),
                _ => {
                    // the MACD line warms up with NaN: not known complete
                    let signal = ma_typed(&line, arg_usize(args, 4), arg_usize(args, 5), false, ops.origin());
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
            let (f, s) = oscillator_mas(&data, (arg_usize(args, 0), mt), (arg_usize(args, 1), mt), ops.origin());
            f64col((0..data.len()).map(|i| percent_oscillator(f[i], s[i])).collect())
        }

        // Group E China-market wrappers (formula-equivalent to apo/ppo, fixed to SMA).
        // bias:N ≡ ppo:1,N,0 — the percentage deviation of close from its N-period SMA.
        ("bias", _) => {
            let data = close(0)?;
            let f = ma_typed(&data, 1, 0, true, ops.origin());
            let s = ma_typed(&data, arg_usize(args, 0), 0, true, ops.origin());
            f64col((0..data.len()).map(|i| percent_oscillator(f[i], s[i])).collect())
        }
        // dma's DDD line = SMA(fast) − SMA(slow), positional as 通达信 (≡ apo:fast,slow,0 for
        // fast ≤ slow; apo orders its averages as TA-Lib); dma.ama is the M-period SMA of that line.
        ("dma", sub) => {
            let data = close(0)?;
            let f = ma_typed(&data, arg_usize(args, 0), 0, true, ops.origin());
            let s = ma_typed(&data, arg_usize(args, 1), 0, true, ops.origin());
            let line: Vec<f64> = (0..data.len()).map(|i| f[i] - s[i]).collect();
            match sub {
                None => f64col(line),
                _ => f64col(ma_typed(&line, arg_usize(args, 2), 0, false, ops.origin())),
            }
        }

        // Group B convention-sensitive indicators (gap report §9).
        ("vortex", sub) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::vortex(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                sub == Some("plus"),
                ops.origin(),
            ))
        }
        ("brar", Some("br")) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::brar_br(&high, &low, &close, arg_usize(args, 0), ops.origin()))
        }
        ("brar", _) => {
            let open = ops.f64(0, "open")?;
            let high = ops.f64(1, "high")?;
            let low = ops.f64(2, "low")?;
            f64col(ind::brar_ar(&open, &high, &low, arg_usize(args, 0), ops.origin()))
        }
        ("vr", _) => {
            let close = ops.f64(0, "close")?;
            let volume = ops.f64(1, "volume")?;
            f64col(ind::vr(&close, &volume, arg_usize(args, 0), ops.origin()))
        }
        ("coppock", _) => f64col(ind::coppock(
            &close(0)?,
            arg_usize(args, 0),
            arg_usize(args, 1),
            arg_usize(args, 2),
            ops.origin(),
        )),
        ("relative_vigor", sub) => {
            let open = ops.f64(0, "open")?;
            let high = ops.f64(1, "high")?;
            let low = ops.f64(2, "low")?;
            let close = ops.f64(3, "close")?;
            let n = arg_usize(args, 0);
            if sub == Some("signal") {
                f64col(ind::relative_vigor_signal(&open, &high, &low, &close, n, ops.origin()))
            } else {
                f64col(ind::relative_vigor(&open, &high, &low, &close, n, ops.origin()))
            }
        }
        ("dkx", sub) => {
            let open = ops.f64(0, "open")?;
            let high = ops.f64(1, "high")?;
            let low = ops.f64(2, "low")?;
            let close = ops.f64(3, "close")?;
            if sub == Some("ma") {
                f64col(ind::dkx_ma(&open, &high, &low, &close, arg_usize(args, 0), ops.origin()))
            } else {
                f64col(ind::dkx(&open, &high, &low, &close, ops.origin()))
            }
        }
        ("wvad", _) => {
            let open = ops.f64(0, "open")?;
            let high = ops.f64(1, "high")?;
            let low = ops.f64(2, "low")?;
            let close = ops.f64(3, "close")?;
            let volume = ops.f64(4, "volume")?;
            f64col(ind::wvad(&open, &high, &low, &close, &volume, arg_usize(args, 0), ops.origin()))
        }
        ("cdp", sub) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
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
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
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
                None => f64col(ind::ema(&ops.f64(0, "close")?, ema_period)),
                _ => {
                    let high = ops.f64(0, "high")?;
                    let low = ops.f64(1, "low")?;
                    let close = ops.f64(2, "close")?;
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
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
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
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            let n = arg_usize(args, 0);
            if sub == Some("on") {
                f64col(ind::ttm_squeeze_on(
                    &high,
                    &low,
                    &close,
                    n,
                    arg_f64(args, 1),
                    arg_f64(args, 2),
                    ops.origin(),
                ))
            } else {
                f64col(ind::ttm_squeeze_momentum(&high, &low, &close, n, ops.origin()))
            }
        }
        ("pivot_points", sub) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
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
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
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
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::wad(&high, &low, &close))
        }
        ("asi", _) => {
            let open = ops.f64(0, "open")?;
            let high = ops.f64(1, "high")?;
            let low = ops.f64(2, "low")?;
            let close = ops.f64(3, "close")?;
            f64col(ind::asi(&open, &high, &low, &close, arg_f64(args, 0)))
        }
        ("supertrend", sub) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::supertrend(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
                arg_f64(args, 1),
                sub == Some("direction"),
            ))
        }

        _ => super::momentum::dispatch(ops, name, sub, args),
    }
}

/// The fast and slow `(period, matype)` generic averages of a TA-Lib price oscillator
/// (`apo`, `ppo`, `macdext`) over `data`, in TA-Lib's order
/// ([`ind::oscillator_order`]: the longer period is the slow average).
fn oscillator_mas(data: &[f64], fast: (usize, usize), slow: (usize, usize), origin: usize) -> (Vec<f64>, Vec<f64>) {
    let ((fp, fm), (sp, sm)) = ind::oscillator_order(fast, slow);
    (ma_typed(data, fp, fm, true, origin), ma_typed(data, sp, sm, true, origin))
}

/// TA-Lib `TA_INT_PO` (PPO, and `bias` ≡ `ppo:1,N,0`): `(fast - slow) / slow * 100`,
/// and 0 where the slow MA is exactly zero (TA-Lib's `TA_IS_ZERO` is an absolute
/// epsilon; volas tests exact zero). A fast MA still warming up stays NaN.
pub(crate) fn percent_oscillator(fast: f64, slow: f64) -> f64 {
    if slow == 0.0 && !fast.is_nan() {
        0.0
    } else {
        (fast - slow) / slow * 100.0
    }
}

#[cfg(test)]
mod tests {
    use crate::{exec::execute, parse};
    use volas_core::{Column, DataFrame};

    fn run(close: Vec<f64>, d: &str) -> Vec<f64> {
        let df = DataFrame::new(vec!["close".into()], vec![Column::f64(close)], None).unwrap();
        execute(&df, &parse(d).unwrap()).unwrap().to_f64_vec()
    }

    /// TA-Lib `TA_INT_PO`: a slow MA of exactly zero gives 0, not ±inf.
    #[test]
    fn ppo_zero_slow_ma_gives_zero() {
        let r = run(vec![1.0, 2.0, -3.0, 4.0], "ppo:2,3,0");
        assert!(r[0].is_nan() && r[1].is_nan());
        assert_eq!(r[2], 0.0); // SMA3 = 0, SMA2 = -0.5
        assert!((r[3] - -50.0).abs() < 1e-12); // (0.5 - 1) / 1 * 100
    }

    /// The zero guard does not invent a value where the fast MA is still warming up.
    #[test]
    fn ppo_zero_slow_ma_keeps_fast_warm_up_nan() {
        assert!(super::percent_oscillator(f64::NAN, 0.0).is_nan());
        assert_eq!(super::percent_oscillator(1.0, 0.0), 0.0);
    }

    /// TA-Lib's oscillator order: the longer period is the slow average whichever
    /// position it is given in, so `ppo:3,2` is `ppo:2,3` (and `apo` / `macdext` alike).
    #[test]
    fn a_slow_period_given_first_is_still_the_slow_average() {
        let close = vec![1.0, -1.0, 5.0, 2.0, 7.0, 3.0];
        for (a, b) in [("ppo:3,2,0", "ppo:2,3,0"), ("apo:5,2,1", "apo:2,5,1"), ("macdext:4,1,2,0", "macdext:2,0,4,1")] {
            let (x, y) = (run(close.clone(), a), run(close.clone(), b));
            assert!(x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()), "{a} vs {b}");
        }
        let r = run(vec![1.0, -1.0, 5.0], "ppo:3,2,0");
        assert!(r[1].is_nan()); // SMA3 not yet defined
        assert!((r[2] - (2.0 - 5.0 / 3.0) / (5.0 / 3.0) * 100.0).abs() < 1e-12);
    }

    /// `bias:N` ≡ `ppo:1,N,0`, so a zero SMA base gives 0 the same way.
    #[test]
    fn bias_zero_base_gives_zero() {
        let r = run(vec![1.0, -1.0, 3.0], "bias:2");
        assert!(r[0].is_nan());
        assert_eq!(r[1], 0.0); // SMA2 = 0
        assert!((r[2] - 200.0).abs() < 1e-12); // (3 - 1) / 1 * 100
    }
}
