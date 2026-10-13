//! The command table: one [`CommandSpec`] per output line — a command and its canonical
//! sub-command — naming its parameters in order (the [`Arg`] constructor used for a slot
//! documents that parameter's domain), its operand slots, and its structural facts: the
//! history class, the lookback (the rows before the kernel publishes its first value),
//! a row-position output, a resume reading past the lookback, and the resume it has.

use std::borrow::Cow;

use volas_compute::raw::indicators::{self as ind, generic_ma_lookback as gml};
use volas_time::TimeFrame;

use super::*;
use crate::types::ArgValue;

/// Bound argument `i` as a period / count / matype.
fn n(a: &[ArgValue], i: usize) -> usize {
    a[i].as_usize()
}

/// `x - 1`, floored at 0 (the window-minus-one warm-up).
fn m1(x: usize) -> usize {
    x.saturating_sub(1)
}

/// `ma` at a recursive MA type runs as that average's own command (`TA_MA`'s dispatch:
/// EMA, DEMA, TEMA, KAMA, MAMA at its fixed 0.5 / 0.05 limits, T3 at its fixed 0.7
/// volume factor), so it resumes from that command's state.
fn ma_runs_as(a: &[ArgValue]) -> Option<(&'static str, Vec<ArgValue>)> {
    let p = ArgValue::Usize(n(a, 0));
    Some(match ind::generic_ma_type(n(a, 0), n(a, 1)) {
        1 => ("ema", vec![p]),
        3 => ("dema", vec![p]),
        4 => ("tema", vec![p]),
        6 => ("kama", vec![p]),
        7 => ("mama", vec![ArgValue::F64(0.5), ArgValue::F64(0.05)]),
        8 => ("t3", vec![p, ArgValue::F64(0.7)]),
        _ => return None,
    })
}

/// The spec for a command given its (already-canonicalized) sub. `None` for an
/// unknown command **or an invalid sub-command** (the match is sub-strict, so it
/// also drives sub validation).
pub fn command_spec(name: &str, sub: Option<&str>) -> Option<CommandSpec> {
    let line = CommandSpec::line;
    Some(match (name, sub) {
        // ma: period, matype.
        ("ma", None) => line(const { &[p_req(), matype()] }, &["close"])
            .averages(&[(0, 1)])
            .step()
            .runs_as(ma_runs_as)
            .lookback(|a| gml(n(a, 0), n(a, 1))),
        ("ema" | "smma", None) => line(const { &[p_req()] }, &["close"]).recursive().resume_rows(0).step().lookback(|a| m1(n(a, 0))),
        // apo / ppo: fast period, slow period, matype — the later of the two averages.
        ("apo" | "ppo", None) => line(const { &[p(12), p(26), matype()] }, &["close"])
            .averages(&[(0, 2), (1, 2)])
            .rows()
            .lookback(|a| gml(n(a, 0), n(a, 2)).max(gml(n(a, 1), n(a, 2)))),
        // macdext: fast, fastmatype, slow, slowmatype[, signal, signalmatype]. Matypes
        // default to SMA (0), unlike macd's fixed EMA.
        ("macdext", None) => line(const { &[p(12), matype(), p(26), matype()] }, &["close"])
            .averages(&[(0, 1), (2, 3)])
            .rows()
            .lookback(|a| gml(n(a, 0), n(a, 1)).max(gml(n(a, 2), n(a, 3)))),
        ("macdext", Some("signal" | "histogram")) => {
            line(const { &[p(12), matype(), p(26), matype(), p(9), matype()] }, &["close"])
                .averages(&[(0, 1), (2, 3), (4, 5)])
                .rows()
                .lookback(|a| gml(n(a, 0), n(a, 1)).max(gml(n(a, 2), n(a, 3))) + gml(n(a, 4), n(a, 5)))
        }
        ("wma" | "trima", None) => line(const { &[p_req()] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("dema", None) => line(const { &[p_req()] }, &["close"]).recursive().resume_rows(0).step().lookback(|a| 2 * m1(n(a, 0))),
        ("tema", None) => line(const { &[p_req()] }, &["close"]).recursive().resume_rows(0).step().lookback(|a| 3 * m1(n(a, 0))),
        // TradingView moving averages.
        ("vwma", None) => line(const { &[p_req()] }, &["close", "volume"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        // alma: period, offset (Gaussian peak position, [0,1]), sigma (>0, width).
        ("alma", None) => line(const { &[p_req(), frange(0.85, 0.0, 1.0), fgt(6.0, 0.0)] }, &["close"])
            .lookback(|a| m1(n(a, 0))),
        // hma = wma(…, round(√n)) over a series valid from n-1: n + round(√n) - 2.
        ("hma", None) => line(const { &[p_req()] }, &["close"]).accumulating().step().lookback(|a| {
            let p = n(a, 0);
            m1(p) + m1((p as f64).sqrt().round() as usize)
        }),
        ("swma", None) => line(&[], &["close"]).lookback(|_| 3),
        // t3: period, vfactor (volume factor, TA-Lib domain [0, 1]).
        ("t3", None) => line(const { &[p_req(), frange(0.7, 0.0, 1.0)] }, &["close"])
            .recursive().resume_rows(0)
            .step()
            .lookback(|a| 6 * m1(n(a, 0))),
        ("kama", None) => line(const { &[p_req()] }, &["close"]).recursive().step().reads_past_lookback().lookback(|a| n(a, 0)),
        // mavp: min_period, max_period (each period value is clamped into this range),
        // matype; real defaults to close, periods is the required second series. min <=
        // max is cross-checked when binding. It warms up over the maximum period's MA.
        ("mavp", None) => line(const { &[prange_req(), prange_req(), matype()] }, &["close", ""])
            .averages(&[(1, 2)])
            .rows()
            .lookback(|a| gml(n(a, 1), n(a, 2))),
        // sar: acceleration factor, maximum factor — both non-negative (a negative
        // factor walks the stop away from price, producing garbage).
        ("sar", None) => line(const { &[fmin(0.02, 0.0), fmin(0.2, 0.0)] }, &["high", "low"])
            .recursive()
            .step()
            .reads_past_lookback()
            .lookback(|_| 1),
        // sarext: start (signed — negative means start short), offset_on_reverse
        // (>= 0), then long (init/step/max) + short (init/step/max), all >= 0.
        ("sarext", None) => line(
            const { &[
                mult(0.0),
                fmin(0.0, 0.0),
                fmin(0.02, 0.0),
                fmin(0.02, 0.0),
                fmin(0.2, 0.0),
                fmin(0.02, 0.0),
                fmin(0.02, 0.0),
                fmin(0.2, 0.0),
            ] },
            &["high", "low"],
        )
        .recursive()
        .step()
        .reads_past_lookback()
        .lookback(|_| 1),
        // macd: fast period, slow period[, signal period].
        ("macd", None) => line(const { &[p(12), p(26)] }, &["close"]).recursive().resume_rows(0).step().lookback(|a| m1(n(a, 0).max(n(a, 1)))),
        ("macd", Some("signal" | "histogram")) => line(const { &[p(12), p(26), p(9)] }, &["close"])
            .recursive().resume_rows(0)
            .step()
            .lookback(|a| n(a, 0).max(n(a, 1)) + n(a, 2) - 2),
        // macdfix: fast/slow fixed at 12/26; the line takes no args, the signal/histogram
        // take only the signal period.
        ("macdfix", None) => line(&[], &["close"]).recursive().resume_rows(0).step().lookback(|_| 25),
        ("macdfix", Some("signal" | "histogram")) => {
            line(const { &[p(9)] }, &["close"]).recursive().resume_rows(0).step().lookback(|a| 24 + n(a, 0))
        }
        // boll: period[, deviation multiplier (sign-free, TA-Lib nbdev)].
        ("boll", None) => line(const { &[p(20)] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("boll", Some("upper" | "lower")) => {
            line(const { &[p(20), mult(2.0)] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0)))
        }
        ("bbw", None) => line(const { &[p(20)] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("accbands", None) => line(const { &[p(20)] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("accbands", Some("upper" | "lower")) => {
            line(const { &[p(20)] }, &["high", "low"]).accumulating().step().lookback(|a| m1(n(a, 0)))
        }
        ("rsv", None) => line(const { &[p_req()] }, &["high", "low", "close"]).lookback(|a| m1(n(a, 0))),
        // kdj: rsv period, K smoothing[, D smoothing], seed (a %K/%D percentage, so its
        // domain is [0, 100]).
        ("kdj", Some("k")) => {
            line(const { &[p(9), p(3), frange(50.0, 0.0, 100.0)] }, &["high", "low", "close"])
                .recursive()
                .resume_inputs(|a| m1(n(a, 0)).max(1))
                .rows()
                .lookback(|a| n(a, 0) * 3)
        }
        ("kdj", Some("d" | "j")) => {
            line(const { &[p(9), p(3), p(3), frange(50.0, 0.0, 100.0)] }, &["high", "low", "close"])
                .recursive()
                .resume_inputs(|a| m1(n(a, 0)).max(1))
                .rows()
                .lookback(|a| n(a, 0) * 3)
        }
        // rsi: required period — 14 is Wilder's original, but 2 / 9 / 21 / 25 are all
        // common, so there is no single dominant default.
        ("rsi", None) => line(const { &[p_req()] }, &["close"]).recursive().step().lookback(|a| n(a, 0)),
        // bbi = the mean of four SMAs: first value where the longest SMA's is.
        ("bbi", None) => line(const { &[p(3), p(6), p(12), p(24)] }, &["close"])
            .accumulating()
            .step()
            .lookback(|a| m1(n(a, 0).max(n(a, 1)).max(n(a, 2)).max(n(a, 3)))),
        ("tr", None) => line(&[], &["high", "low", "close"]).step().lookback(|_| 1),
        ("atr", None) => line(const { &[p_req()] }, &["high", "low", "close"]).recursive().step().lookback(|a| n(a, 0)),
        ("llv", None) => line(const { &[p_req()] }, &["low"]).lookback(|a| m1(n(a, 0))),
        ("hhv", None) => line(const { &[p_req()] }, &["high"]).lookback(|a| m1(n(a, 0))),
        // donchian: required period — 20 (Turtle) is common, but so are 10 / 55, so no
        // single dominant default.
        ("donchian", None) => line(const { &[p_req()] }, &["high", "low"]).lookback(|a| m1(n(a, 0))),
        ("donchian", Some("upper")) => line(const { &[p_req()] }, &["high"]).lookback(|a| m1(n(a, 0))),
        ("donchian", Some("lower")) => line(const { &[p_req()] }, &["low"]).lookback(|a| m1(n(a, 0))),
        // hv: window (stddev of log returns — needs >= 2), bar time frame, trading days
        // per year (>= 1: 0/negative puts 0/NaN under sqrt), and minutes of trading per
        // day (1440: a 24 h market). The annualization is `exec::hv_annualization`.
        ("hv", None) => line(
            const {
                &[
                    p2_req(),
                    time_frame(TimeFrame::Day1),
                    i64min(HV_TRADING_DAYS, 1),
                    session_minutes(HV_SESSION_MINUTES),
                ]
            },
            &["close"],
        )
        .accumulating()
        .step()
        .lookback(|a| n(a, 0)),
        // increase: run length (>= 1), direction (+1 rising / -1 falling); k steps, each
        // reading a value and its predecessor.
        ("increase", None) => line(const { &[p(1), one_of(1, &[-1, 1])] }, &["close"]).counted().lookback(|a| n(a, 0)),
        // Group A non-TA-Lib indicators (gap report 2026-06-07).
        // psy averages `n` rising flags, each reading a close and the one before it
        ("psy", None) => line(const { &[p(12)] }, &["close"]).lookback(|a| n(a, 0)),
        // pvt / nvi / pvi are cumulative (lookback 0), carrying a running line.
        ("pvt" | "nvi" | "pvi", None) => line(&[], &["close", "volume"]).recursive().step(),
        // dpo reads the SMA (n−1) and the close n/2+1 bars back: the later of the two.
        ("dpo", None) => line(const { &[p_req()] }, &["close"])
            .accumulating()
            .step()
            .lookback(|a| m1(n(a, 0)).max(n(a, 0) / 2 + 1)),
        ("cmf", None) => line(const { &[p(20)] }, &["high", "low", "close", "volume"])
            .accumulating()
            .step()
            .lookback(|a| m1(n(a, 0))),
        // chop's TR sum needs one prior close, so it warms up one bar past the window.
        ("chop", None) => line(const { &[p_req()] }, &["high", "low", "close"]).lookback(|a| n(a, 0)),
        // kst's slowest term is SMA15 over ROC30: 30 + 15 - 1.
        ("kst", None) => line(&[], &["close"]).accumulating().step().lookback(|_| 44),
        ("emv", None) => line(const { &[p_req()] }, &["high", "low", "volume"]).accumulating().step().lookback(|a| n(a, 0)),
        // mass_index: EMA9 of EMA9 warms up at 2*(9-1)=16, then the n-sum adds n-1.
        ("mass_index", None) => line(const { &[p(25)] }, &["high", "low"]).recursive().resume_inputs(|a| n(a, 0)).step().lookback(|a| 15 + n(a, 0)),
        ("efi", None) => line(const { &[p(13)] }, &["close", "volume"]).recursive().step().lookback(|a| n(a, 0)),
        ("tsi", None) => line(const { &[p(25), p(13)] }, &["close"]).recursive().step().lookback(|a| n(a, 0) + n(a, 1) - 1),
        // crsi averages RSI(close) (rsi_len), RSI(streak) (streak_len; the streak starts
        // at bar 0) and the percent rank of ROC1 over rank_len prior values (rank_len+1).
        ("crsi", None) => line(const { &[p(3), p(2), p(100)] }, &["close"])
            .recursive().step().reads_past_lookback()
            .lookback(|a| n(a, 0).max(n(a, 1)).max(n(a, 2) + 1)),
        // Group E formula-equivalent wrappers (China-market names): bias ≡ ppo:1,N,0 (the
        // slow SMA_N gates it); dma's DDD line = SMA(fast) − SMA(slow), positional as 通达信
        // (≡ apo:fast,slow,0 for fast ≤ slow), with AMA = the M-period SMA of that line.
        ("bias", None) => line(const { &[p(6)] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("dma", None) => line(const { &[p(10), p(50)] }, &["close"])
            .accumulating()
            .step()
            .lookback(|a| m1(n(a, 0).max(n(a, 1)))),
        ("dma", Some("ama")) => line(const { &[p(10), p(50), p(10)] }, &["close"])
            .accumulating()
            .step()
            .lookback(|a| m1(n(a, 0).max(n(a, 1))) + m1(n(a, 2))),
        // Group B convention-sensitive indicators (gap report §9). +VM/−VM/TR each need
        // one prior bar, so the n-sum warms up at bar n.
        ("vortex", Some("plus" | "minus")) => line(const { &[p(14)] }, &["high", "low", "close"]).lookback(|a| n(a, 0)),
        // brar AR (H−O, O−L) has no prior-bar term (n−1); BR (vs prior close) has one (n).
        ("brar", Some("ar")) => line(const { &[p(26)] }, &["open", "high", "low"]).lookback(|a| m1(n(a, 0))),
        ("brar", Some("br")) => line(const { &[p(26)] }, &["high", "low", "close"]).lookback(|a| n(a, 0)),
        // up/down/flat volume is classified vs the prior close, so the n-sum warms up at n.
        ("vr", None) => line(const { &[p(26)] }, &["close", "volume"]).accumulating().step().lookback(|a| n(a, 0)),
        // coppock: wma_period, roc_long, roc_short — the longer ROC plus the WMA warm-up.
        ("coppock", None) => line(const { &[p(10), p(14), p(11)] }, &["close"])
            .accumulating()
            .step()
            .lookback(|a| n(a, 1).max(n(a, 2)) + m1(n(a, 0))),
        // relative_vigor: swma4 (3) + SMAₙ (n−1) = n+2; the signal adds another swma4 (3).
        ("relative_vigor", None) => line(const { &[p(10)] }, &["open", "high", "low", "close"])
            .accumulating()
            .step()
            .lookback(|a| n(a, 0) + 2),
        ("relative_vigor", Some("signal")) => line(const { &[p(10)] }, &["open", "high", "low", "close"])
            .accumulating()
            .step()
            .lookback(|a| n(a, 0) + 5),
        // dkx's DKX line is a fixed 20-period weighted MA (no args, 19); the MADKX signal
        // adds SMA_m (m−1).
        ("dkx", None) => line(&[], &["open", "high", "low", "close"]).accumulating().step().lookback(|_| 19),
        ("dkx", Some("ma")) => line(const { &[p(10)] }, &["open", "high", "low", "close"])
            .accumulating()
            .step()
            .lookback(|a| 18 + n(a, 0)),
        ("wvad", None) => line(const { &[p(24)] }, &["open", "high", "low", "close", "volume"])
            .accumulating()
            .step()
            .lookback(|a| m1(n(a, 0))),
        // cdp: five intraday levels from the prior bar (no window parameter).
        ("cdp", None | Some("ah" | "nh" | "nl" | "al")) => line(&[], &["high", "low", "close"]).lookback(|_| 1),
        // mike: six support/resistance lines — all required sub-commands, no primary line;
        // TYP±range is gated by the n-day HH/LL.
        ("mike", Some("weakr" | "midr" | "strongr" | "weaks" | "mids" | "strongs")) => {
            line(const { &[p_req()] }, &["high", "low", "close"]).lookback(|a| m1(n(a, 0)))
        }
        // keltner: middle = EMA(close) (ema_period only); bands add atr_period + multiplier
        // and also wait for the ATR.
        ("keltner", None) => line(const { &[p(20)] }, &["close"]).recursive().step().lookback(|a| m1(n(a, 0))),
        ("keltner", Some("upper" | "lower")) => {
            line(const { &[p(20), p(10), mult(2.0)] }, &["high", "low", "close"])
                .recursive()
                .step()
                .lookback(|a| m1(n(a, 0)).max(n(a, 1)))
        }
        // stoch_momentum: k (HH/LL), d (double-EMA smoothing), signal (EMA of SMI): HH/LL
        // (k−1) + the double EMA_d (2·(d−1)); the signal adds EMA_signal.
        ("stoch_momentum", None) => line(const { &[p(10), p(3), p(3)] }, &["high", "low", "close"])
            .recursive().rows().reads_past_lookback()
            .lookback(|a| m1(n(a, 0)) + 2 * m1(n(a, 1))),
        ("stoch_momentum", Some("signal")) => line(const { &[p(10), p(3), p(3)] }, &["high", "low", "close"])
            .recursive().rows().reads_past_lookback()
            .lookback(|a| m1(n(a, 0)) + 2 * m1(n(a, 1)) + m1(n(a, 2))),
        // ttm_squeeze: period, Bollinger σ-multiplier, Keltner range-multiplier. The
        // momentum's linreg over the n-bar delta (itself gated at n−1) warms up at
        // 2·(n−1); the squeeze flag waits for SMAₙ(TR) (n).
        ("ttm_squeeze", None) => line(const { &[p(20), mult(2.0), mult(1.5)] }, &["high", "low", "close"])
            .accumulating()
            .lookback(|a| 2 * m1(n(a, 0))),
        ("ttm_squeeze", Some("on")) => line(const { &[p(20), mult(2.0), mult(1.5)] }, &["high", "low", "close"])
            .accumulating()
            .lookback(|a| n(a, 0)),
        // pivot_points: PP plus the R1/R2/R3 / S1/S2/S3 levels, all from the prior bar.
        ("pivot_points", None | Some("r1" | "r2" | "r3" | "s1" | "s2" | "s3")) => {
            line(&[], &["high", "low", "close"]).lookback(|_| 1)
        }
        // ichimoku: tenkan / kijun / senkou_b periods (the displacement is the kijun
        // period): midpoint lines warm up at period−1; the leading spans add the kijun
        // displacement; the lagging span (chikou = close) has none.
        ("ichimoku", Some("tenkan")) => line(ICHIMOKU, &["high", "low", "close"]).lookback(|a| m1(n(a, 0))),
        ("ichimoku", Some("kijun")) => line(ICHIMOKU, &["high", "low", "close"]).lookback(|a| m1(n(a, 1))),
        // senkou_a averages tenkan and kijun: the longer of the two, displaced.
        ("ichimoku", Some("senkou_a")) => {
            line(ICHIMOKU, &["high", "low", "close"]).lookback(|a| m1(n(a, 0).max(n(a, 1))) + n(a, 1))
        }
        ("ichimoku", Some("senkou_b")) => {
            line(ICHIMOKU, &["high", "low", "close"]).lookback(|a| m1(n(a, 2)) + n(a, 1))
        }
        ("ichimoku", Some("chikou")) => line(ICHIMOKU, &["high", "low", "close"]),
        // wad: cumulative, no parameters. asi: Wilder's limit-move scaling `t` — the
        // formula divides by it, so it must be strictly positive. Both cumulative
        // (lookback 0), carrying a running line.
        ("wad", None) => line(&[], &["high", "low", "close"]).recursive().step(),
        ("asi", None) => line(const { &[fgt(3.0, 0.0)] }, &["open", "high", "low", "close"]).recursive().rows(),
        // supertrend: ATR period + band multiplier (shared by the line and direction);
        // it seeds with its ATR.
        ("supertrend", None | Some("direction")) => line(const { &[p(10), mult(3.0)] }, &["high", "low", "close"])
            .recursive()
            .step()
            .lookback(|a| n(a, 0)),
        ("style", Some("bullish" | "bearish")) => line(&[], &["open", "close"]),
        // Candlestick patterns (style.<pattern> / cdl.<pattern>) — validated against the
        // compute layer's pattern registry, so new patterns need no change here; each
        // warms up over its candle-settings average period. Patterns taking a
        // `penetration` ratio accept one optional Float arg (default 0.5, non-negative —
        // TA-Lib's domain).
        ("style", Some(pat)) => {
            // One registry lookup: an unknown pattern is a `None` spec (a "no such
            // sub-command" error). `bullish`/`bearish` are handled by the arm above.
            let (pattern, lookback) = ind::candle_pattern(pat)?;
            let args = match pattern {
                ind::CandlePattern::Penetration { default, .. } => vec![fmin(default, 0.0)],
                ind::CandlePattern::Plain(_) => vec![],
            };
            CommandSpec::owned(Cow::Owned(args), &["open", "high", "low", "close"]).fixed_lookback(lookback)
        }
        // repeat: consecutive-True run length (>= 1; 0 underflowed the kernel) of its
        // required condition operand.
        ("repeat", None) => line(const { &[p(1)] }, &[""]).lookback(|a| m1(n(a, 0))),
        // change: % change over a window spanning `period` bars — `period` 2 compares with
        // the previous bar; 1 compares a bar with itself (identically 0), 0 underflowed
        // the kernel, so the domain is >= 2.
        ("change", None) => line(const { &[p2(2)] }, &["close"]).lookback(|a| m1(n(a, 0))),
        ("mom" | "roc" | "rocp" | "rocr" | "rocr100", None) => {
            line(const { &[p_req()] }, &["close"]).step().lookback(|a| n(a, 0))
        }
        ("midpoint", None) => line(const { &[p_req()] }, &["close"]).lookback(|a| m1(n(a, 0))),
        ("midprice", None) => line(const { &[p_req()] }, &["high", "low"]).lookback(|a| m1(n(a, 0))),
        ("willr", None) => line(const { &[p_req()] }, &["high", "low", "close"]).lookback(|a| m1(n(a, 0))),
        ("cmo", None) => line(const { &[p_req()] }, &["close"]).recursive().step().lookback(|a| n(a, 0)),
        // cci: window >= 2 — at 1 the numerator (TP - SMA(TP,1)) is identically zero, a
        // constant no-signal column.
        ("cci", None) => line(const { &[p2_req()] }, &["high", "low", "close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("imi", None) => line(const { &[p_req()] }, &["open", "close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("trix", None) => line(const { &[p_req()] }, &["close"]).recursive().resume_rows(0).step().lookback(|a| 3 * m1(n(a, 0)) + 1),
        ("mfi", None) => line(const { &[p_req()] }, &["high", "low", "close", "volume"])
            .accumulating()
            .step()
            .lookback(|a| n(a, 0)),
        ("ultosc", None) => line(const { &[p(7), p(14), p(28)] }, &["high", "low", "close"])
            .accumulating()
            .step()
            .lookback(|a| n(a, 0).max(n(a, 1)).max(n(a, 2))),
        // stoch: fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype — %K
        // (fastk-1) then the matype-MA smoothing stage(s); each line reads its own stages.
        ("stoch", Some("k")) => line(STOCH, &["high", "low", "close"])
            .averages(&[(1, 2)])
            .rows()
            .lookback(|a| m1(n(a, 0)) + gml(n(a, 1), n(a, 2))),
        ("stoch", Some("d")) => line(STOCH, &["high", "low", "close"])
            .averages(&[(1, 2), (3, 4)])
            .rows()
            .lookback(|a| m1(n(a, 0)) + gml(n(a, 1), n(a, 2)) + gml(n(a, 3), n(a, 4))),
        // stochf: fastk_period, fastd_period, fastd_matype. %K is a pure window (its D
        // smoothing is not its input); %D averages it.
        ("stochf", Some("k")) => line(STOCHF, &["high", "low", "close"]).lookback(|a| m1(n(a, 0))),
        ("stochf", Some("d")) => line(STOCHF, &["high", "low", "close"])
            .averages(&[(1, 2)])
            .rows()
            .lookback(|a| m1(n(a, 0)) + gml(n(a, 1), n(a, 2))),
        // stochrsi: rsi_period, fastk_period, fastd_period, fastd_matype — the RSI lookback
        // plus the %K window, then the d-line MA. Every MA type resumes.
        ("stochrsi", Some("k")) => line(STOCHRSI, &["close"])
            .recursive()
            .resume_inputs(|a| m1(n(a, 1)).max(1))
            .rows()
            .lookback(|a| n(a, 0) + m1(n(a, 1))),
        ("stochrsi", Some("d")) => line(STOCHRSI, &["close"])
            .recursive()
            .resume_inputs(|a| m1(n(a, 1)) + n(a, 2))
            .rows()
            .lookback(|a| n(a, 0) + m1(n(a, 1)) + gml(n(a, 2), n(a, 3))),
        // ±DM seed at bar `period - 1`, but never bar 0 (period 1 seeds at bar 1).
        ("plus_dm" | "minus_dm", None) => line(const { &[p_req()] }, &["high", "low"])
            .recursive()
            .step()
            .lookback(|a| ind::wilder_seed_bar(n(a, 0)).unwrap_or(0)),
        ("plus_di" | "minus_di" | "dx", None) => {
            line(const { &[p_req()] }, &["high", "low", "close"]).recursive().step().lookback(|a| n(a, 0))
        }
        ("adx", None) => line(const { &[p_req()] }, &["high", "low", "close"]).recursive().step().lookback(|a| 2 * n(a, 0) - 1),
        ("adxr", None) => line(const { &[p_req()] }, &["high", "low", "close"]).recursive().step().lookback(|a| 3 * n(a, 0) - 2),
        // aroon / aroonosc: window >= 2 — at 1 the lines are identically 100 (the extreme
        // is always "this bar"), a constant no-signal column.
        ("aroon", Some("up" | "down")) => line(const { &[p2_req()] }, &["high", "low"]).lookback(|a| n(a, 0)),
        ("aroonosc", None) => line(const { &[p2_req()] }, &["high", "low"]).lookback(|a| n(a, 0)),
        ("sum", None) => line(const { &[p_req()] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        // The index family emits absolute positions, from an incremental extreme tracker.
        ("maxindex" | "minindex", None) => line(const { &[p_req()] }, &["close"])
            .position()
            .reads_past_lookback()
            .step()
            .lookback(|a| m1(n(a, 0))),
        ("minmaxindex", Some("min" | "max")) => line(const { &[p_req()] }, &["close"])
            .position()
            .reads_past_lookback()
            .step()
            .lookback(|a| m1(n(a, 0))),
        ("minmax", Some("min" | "max")) => line(const { &[p_req()] }, &["close"]).lookback(|a| m1(n(a, 0))),
        ("natr", None) => line(const { &[p_req()] }, &["high", "low", "close"]).recursive().step().lookback(|a| n(a, 0)),
        ("bop", None) => line(&[], &["open", "high", "low", "close"]).step(),
        // Linear-regression family: window >= 2 (the slope denominator Σ(x - x̄)² is zero
        // on a single sample — all-NaN by construction).
        (
            "linearreg" | "linearreg_slope" | "linearreg_intercept" | "linearreg_angle" | "tsf",
            None,
        ) => line(const { &[p2_req()] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        // correl / beta: correlation / regression over >= 2 samples (the single-sample
        // variance is zero, so the statistic is 0/0). The first series defaults to close;
        // the second is required.
        ("correl", None) => line(const { &[p2_req()] }, &["close", ""]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("beta", None) => line(const { &[p2_req()] }, &["close", ""]).accumulating().step().lookback(|a| n(a, 0)),
        // var / stddev: window >= 2 — the (population) variance of one sample is
        // identically zero, a constant no-signal column. (TA-Lib allows var:1; volas
        // rejects it on the V17 no-signal principle.) stddev's second arg is the
        // sign-free deviation multiplier.
        ("var", None) => line(const { &[p2_req()] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("stddev", None) => line(const { &[p2_req(), mult(1.0)] }, &["close"])
            .accumulating()
            .step()
            .lookback(|a| m1(n(a, 0))),
        // pandas-window statistics promoted to directives (single kernel source with the
        // rolling API; full-window semantics — an NA in the window yields NA, like the TA
        // family's warm-up discipline).
        ("median" | "rank" | "sem", None) => line(const { &[p2_req()] }, &["close"]).lookback(|a| m1(n(a, 0))),
        // quantile: window, then the quantile level in [0, 1].
        ("quantile", None) => line(const { &[p2_req(), frange(0.5, 0.0, 1.0)] }, &["close"]).lookback(|a| m1(n(a, 0))),
        // skew / kurt are undefined below 3 / 4 samples (V17: no all-NA columns).
        ("skew", None) => line(
            const { &[Arg { default: ArgDefault::Required, bound: ArgBound::IntMin(3) }] },
            &["close"],
        )
        .lookback(|a| m1(n(a, 0))),
        ("kurt", None) => line(
            const { &[Arg { default: ArgDefault::Required, bound: ArgBound::IntMin(4) }] },
            &["close"],
        )
        .lookback(|a| m1(n(a, 0))),
        // TradingView oscillators / dispersion. cog / rci / dev / mode follow the
        // windowed-stat family (period required, >= 2). iii is pointwise (no arg).
        ("cog", None) => line(const { &[p2_req()] }, &["close"]).accumulating().step().lookback(|a| m1(n(a, 0))),
        ("rci" | "dev" | "mode", None) => line(const { &[p2_req()] }, &["close"]).lookback(|a| m1(n(a, 0))),
        ("iii", None) => line(&[], &["high", "low", "close", "volume"]),
        // kcw = 2·mult·ATR/EMA, mirroring the keltner band parameters: the EMA warms up at
        // ema_period−1, the ATR at atr_period; it continues their pair of states.
        ("kcw", None) => line(const { &[p(20), p(10), mult(2.0)] }, &["high", "low", "close"])
            .recursive()
            .step()
            .lookback(|a| m1(n(a, 0)).max(n(a, 1))),
        // fractal pivots: leftbars, rightbars (both required, >= 1) — a pivot is confirmed
        // `rightbars` after a point needing `leftbars` of history.
        ("pivothigh", None) => line(const { &[p_req(), p_req()] }, &["high"]).lookback(|a| n(a, 0) + n(a, 1)),
        ("pivotlow", None) => line(const { &[p_req(), p_req()] }, &["low"]).lookback(|a| n(a, 0) + n(a, 1)),
        ("obv", None) => line(&[], &["close", "volume"]).recursive().step(),
        ("ad", None) => line(&[], &["high", "low", "close", "volume"]).recursive().resume_rows(0).step(),
        // adosc: fast EMA period, slow EMA period.
        ("adosc", None) => line(const { &[p(3), p(10)] }, &["high", "low", "close", "volume"])
            .recursive()
            .resume_inputs(|a| m1(n(a, 0).max(n(a, 1))))
            .rows()
            .lookback(|a| m1(n(a, 0).max(n(a, 1)))),
        ("avgprice", None) => line(&[], &["open", "high", "low", "close"]).step(),
        ("medprice", None) => line(&[], &["high", "low"]).step(),
        ("typprice" | "wclprice", None) => line(&[], &["high", "low", "close"]).step(),
        // Hilbert-transform cycle suite: a single price series, no numeric args (MAMA
        // takes fast/slow limits); fixed warm-up — DCPERIOD / PHASOR / MAMA 32, the
        // phase-dependent DCPHASE / SINE / TRENDLINE / TRENDMODE 63.
        ("ht_dcperiod", None) | ("ht_phasor", None | Some("quadrature")) => {
            line(&[], &["close"]).recursive().resume_rows(32).rows().lookback(|_| 32)
        }
        ("ht_dcphase" | "ht_trendline" | "ht_trendmode", None) | ("ht_sine", None | Some("leadsine")) => {
            line(&[], &["close"]).recursive().resume_rows(63).rows().lookback(|_| 63)
        }
        // mama: fast limit, slow limit — TA-Lib's domain [0.01, 0.99]; outside it the
        // adaptive alpha degenerates and the output is all-NaN.
        ("mama", None | Some("fama")) => line(const { &[frange(0.5, 0.01, 0.99), frange(0.05, 0.01, 0.99)] }, &["close"])
            .recursive()
            .resume_rows(32)
            .rows()
            .lookback(|_| 32),
        _ => return None,
    })
}

/// ichimoku's tenkan / kijun / senkou_b periods.
const ICHIMOKU: &[Arg] = &[p(9), p(26), p(52)];
/// stoch: fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype.
const STOCH: &[Arg] = &[p(5), p(3), matype(), p(3), matype()];
/// stochf: fastk_period, fastd_period, fastd_matype.
const STOCHF: &[Arg] = &[p(5), p(3), matype()];
/// stochrsi: rsi_period, fastk_period, fastd_period, fastd_matype.
const STOCHRSI: &[Arg] = &[p(14), p(5), p(3), matype()];
