//! Lookback (warm-up) computation for a directive — the minimum number of prior
//! rows needed before an indicator produces a valid value.

use std::borrow::Cow;

use crate::spec::HistoryClass;
use crate::types::{ArgValue, Ast, Command, Node};
use volas_compute::raw::indicators::generic_ma_lookback;

/// A bound period / count / matype argument. Arguments are resolved + type-checked
/// by `bind`, so every position is present — there is no default to re-apply here
/// (that knowledge lives only in the spec).
fn arg(args: &[ArgValue], i: usize) -> usize {
    args[i].as_usize()
}

/// The command's own lookback (ignoring its series operands). Only ever called
/// with a bound command name (every caller binds first), so every command has an
/// arm and an unrecognized name is unreachable.
pub(crate) fn own_lookback(name: &str, sub: Option<&str>, args: &[ArgValue]) -> usize {
    match name {
        "ema" | "smma" => arg(args, 0).saturating_sub(1),
        "ma" => generic_ma_lookback(arg(args, 0), arg(args, 1)),
        // apo/ppo need both MAs: the later of the two warm-ups (either may be the longer).
        "apo" | "ppo" => generic_ma_lookback(arg(args, 0), arg(args, 2)).max(generic_ma_lookback(arg(args, 1), arg(args, 2))),
        // mavp warms up over the maximum period's MA (args: min, max, matype).
        "mavp" => generic_ma_lookback(arg(args, 1), arg(args, 2)),
        "macdext" => {
            let line = generic_ma_lookback(arg(args, 0), arg(args, 1))
                .max(generic_ma_lookback(arg(args, 2), arg(args, 3)));
            match sub {
                None => line,
                _ => line + generic_ma_lookback(arg(args, 4), arg(args, 5)),
            }
        }
        "wma" | "trima" => arg(args, 0).saturating_sub(1),
        "vwma" | "alma" => arg(args, 0).saturating_sub(1),
        // hma = wma(…, round(√n)) over a series valid from n-1, so it warms an
        // extra round(√n)-1 rows: total n + round(√n) - 2.
        "hma" => {
            let n = arg(args, 0);
            let sq = (n as f64).sqrt().round() as usize;
            n.saturating_sub(1) + sq.saturating_sub(1)
        }
        "swma" => 3,
        "dema" => 2 * arg(args, 0).saturating_sub(1),
        "tema" => 3 * arg(args, 0).saturating_sub(1),
        "t3" => 6 * arg(args, 0).saturating_sub(1),
        "kama" => arg(args, 0),
        "sar" | "sarext" => 1,
        "boll" | "bbw" | "accbands" => arg(args, 0).saturating_sub(1),
        "macd" => match sub {
            None => arg(args, 0).max(arg(args, 1)).saturating_sub(1),
            Some(_) => arg(args, 0).max(arg(args, 1)) + arg(args, 2) - 2,
        },
        // macdfix: fast/slow fixed at 12/26; signal period is arg 0 on the sub-lines.
        "macdfix" => match sub {
            None => 25,
            _ => 24 + arg(args, 0),
        },
        // bbi = the mean of four SMAs: first value where the longest SMA's is, max − 1.
        "bbi" => arg(args, 0)
            .max(arg(args, 1))
            .max(arg(args, 2))
            .max(arg(args, 3))
            .saturating_sub(1),
        // psy averages `n` rising flags, each reading a close and the one before it
        "psy" => arg(args, 0),
        // pvt / nvi / pvi are cumulative (lookback 0); like obv/ad they carry a running-line
        // state so append resumes in O(new rows) and continues past a slice (see exec_resume).
        "pvt" | "nvi" | "pvi" => 0,
        // dpo reads the SMA (n−1) and the close n/2+1 bars back: the later of the two.
        "dpo" => arg(args, 0).saturating_sub(1).max(arg(args, 0) / 2 + 1),
        "cmf" => arg(args, 0).saturating_sub(1),
        // chop's TR sum needs one prior close, so it warms up one bar past the window.
        "chop" => arg(args, 0),
        // kst's slowest term is SMA15 over ROC30: 30 + 15 - 1.
        "kst" => 44,
        "emv" => arg(args, 0),
        // mass_index: EMA9 of EMA9 warms up at 2*(9-1)=16, then the n-sum adds n-1.
        "mass_index" => 15 + arg(args, 0),
        "efi" => arg(args, 0),
        "tsi" => arg(args, 0) + arg(args, 1) - 1,
        // crsi averages RSI(close) (rsi_len), RSI(streak) (streak_len; the streak starts
        // at bar 0) and the percent rank of ROC1 over rank_len prior values (rank_len+1).
        "crsi" => arg(args, 0).max(arg(args, 1)).max(arg(args, 2) + 1),
        // bias ≡ ppo:1,N,0 — the slow SMA_N gates it (the fast SMA_1 has no warm-up).
        "bias" => arg(args, 0).saturating_sub(1),
        // dma's DDD line = SMA(fast) − SMA(slow) (the later of the two SMAs gates it); AMA adds
        // its own SMA_M warm-up.
        "dma" => {
            let line = arg(args, 0).max(arg(args, 1)).saturating_sub(1);
            match sub {
                None => line,
                _ => line + arg(args, 2).saturating_sub(1),
            }
        }
        // +VM/−VM/TR each need one prior bar, so the n-sum warms up at bar n.
        "vortex" => arg(args, 0),
        // up/down/flat volume is classified vs the prior close, so the n-sum warms up at n.
        "vr" => arg(args, 0),
        // brar AR (H−O, O−L) has no prior-bar term (n−1); BR (vs prior close) has one (n).
        "brar" => match sub {
            Some("br") => arg(args, 0),
            _ => arg(args, 0).saturating_sub(1),
        },
        // coppock: the longer ROC plus the WMA warm-up.
        "coppock" => arg(args, 1).max(arg(args, 2)) + arg(args, 0).saturating_sub(1),
        // relative_vigor: swma4 (3) + SMAₙ (n−1) = n+2; the signal adds another swma4 (3).
        "relative_vigor" => match sub {
            Some("signal") => arg(args, 0) + 5,
            _ => arg(args, 0) + 2,
        },
        // dkx: the DDX line is WMA(20) (19); the MADKX signal adds SMA_m (m−1).
        "dkx" => match sub {
            Some("ma") => 18 + arg(args, 0),
            _ => 19,
        },
        "wvad" => arg(args, 0).saturating_sub(1),
        // cdp reads only the prior bar; mike's TYP±range is gated by the n-day HH/LL.
        "cdp" | "pivot_points" => 1,
        // wad / asi are cumulative (lookback 0); like obv they carry state for O(new rows)
        // append and slice continuation (see exec_resume).
        "wad" | "asi" => 0,
        // supertrend seeds with its ATR.
        "supertrend" => arg(args, 0),
        "mike" => arg(args, 0).saturating_sub(1),
        // ichimoku: midpoint lines warm up at period−1; the leading spans add the kijun
        // displacement; the lagging span (chikou = close) has none.
        "ichimoku" => {
            let (t, k, sb) = (arg(args, 0), arg(args, 1), arg(args, 2));
            match sub {
                Some("tenkan") => t.saturating_sub(1),
                Some("kijun") => k.saturating_sub(1),
                // senkou_a averages tenkan and kijun: the longer of the two, displaced.
                Some("senkou_a") => t.max(k).saturating_sub(1) + k,
                Some("senkou_b") => sb.saturating_sub(1) + k,
                _ => 0,
            }
        }
        // keltner: middle = EMA (ema_period−1); bands also wait for the ATR (atr_period).
        "keltner" => {
            let ema = arg(args, 0).saturating_sub(1);
            match sub {
                None => ema,
                _ => ema.max(arg(args, 1)),
            }
        }
        // stoch_momentum: HH/LL (k−1) + the double EMA_d (2·(d−1)); the signal adds EMA_signal.
        "stoch_momentum" => {
            let base = arg(args, 0).saturating_sub(1) + 2 * arg(args, 1).saturating_sub(1);
            match sub {
                Some("signal") => base + arg(args, 2).saturating_sub(1),
                _ => base,
            }
        }
        // ttm_squeeze: the squeeze flag waits for SMAₙ(TR) (n); the momentum's linreg over the
        // n-bar delta (itself gated at n−1) warms up at 2·(n−1).
        "ttm_squeeze" => match sub {
            Some("on") => arg(args, 0),
            _ => 2 * arg(args, 0).saturating_sub(1),
        },
        "tr" => 1,
        "atr" => arg(args, 0),
        "llv" | "hhv" | "donchian" | "rsv" => arg(args, 0).saturating_sub(1),
        "kdj" => arg(args, 0) * 3,
        "rsi" => arg(args, 0),
        "hv" => arg(args, 0),
        // increase:k ANDs k steps, each reading a value and its predecessor: k + 1 values.
        "increase" => arg(args, 0),
        "repeat" => arg(args, 0).saturating_sub(1),
        // style.<x>: candlestick patterns warm up over their candle-settings avg period
        // (from the compute registry); bullish/bearish need none. (`cdl` resolves here too.)
        "style" => sub
            .and_then(|p| volas_compute::raw::indicators::candle_pattern(p).map(|(_, lb)| lb))
            .unwrap_or(0),
        "change" => arg(args, 0).saturating_sub(1),
        "mom" | "roc" | "rocp" | "rocr" | "rocr100" => arg(args, 0),
        "willr" | "midpoint" | "midprice" => arg(args, 0).saturating_sub(1),
        "cmo" | "natr" => arg(args, 0),
        "cci" | "imi" => arg(args, 0).saturating_sub(1),
        "mfi" => arg(args, 0),
        "ultosc" => arg(args, 0).max(arg(args, 1)).max(arg(args, 2)),
        // %K (lookback fastk-1) then matype-MA smoothing stage(s).
        "stoch" => {
            let base =
                arg(args, 0).saturating_sub(1) + generic_ma_lookback(arg(args, 1), arg(args, 2));
            match sub {
                Some("d") => base + generic_ma_lookback(arg(args, 3), arg(args, 4)),
                _ => base,
            }
        }
        "stochf" => {
            let base = arg(args, 0).saturating_sub(1);
            match sub {
                Some("d") => base + generic_ma_lookback(arg(args, 1), arg(args, 2)),
                _ => base,
            }
        }
        "stochrsi" => {
            // RSI lookback (period) + the %K window (fastk_period-1), then the d-line MA.
            let base = arg(args, 0) + arg(args, 1).saturating_sub(1);
            match sub {
                Some("d") => base + generic_ma_lookback(arg(args, 2), arg(args, 3)),
                _ => base,
            }
        }
        // ±DM seed at bar `period - 1`, but never bar 0 (period 1 seeds at bar 1).
        "plus_dm" | "minus_dm" => volas_compute::raw::indicators::wilder_seed_bar(arg(args, 0)).unwrap_or(0),
        "plus_di" | "minus_di" | "dx" => arg(args, 0),
        "adx" => 2 * arg(args, 0) - 1,
        "adxr" => 3 * arg(args, 0) - 2,
        "trix" => 3 * arg(args, 0).saturating_sub(1) + 1,
        "aroon" | "aroonosc" => arg(args, 0),
        "sum" | "maxindex" | "minindex" | "minmax" | "minmaxindex" => {
            arg(args, 0).saturating_sub(1)
        }
        "bop" => 0,
        "linearreg" | "linearreg_slope" | "linearreg_intercept" | "linearreg_angle" | "tsf" => {
            arg(args, 0).saturating_sub(1)
        }
        "var" | "stddev" => arg(args, 0).saturating_sub(1),
        "median" | "quantile" | "rank" | "skew" | "kurt" | "sem" => {
            arg(args, 0).saturating_sub(1)
        }
        "cog" | "rci" | "dev" | "mode" => arg(args, 0).saturating_sub(1),
        // iii is pointwise (no warm-up); kcw warms with its slower component
        // (EMA(ema_period) vs ATR(atr_period)).
        "iii" => 0,
        // kcw = 2·mult·ATR/EMA: the EMA warms up at ema_period−1, the ATR at atr_period.
        "kcw" => arg(args, 0).saturating_sub(1).max(arg(args, 1)),
        // a pivot is confirmed `rightbars` after a point needing `leftbars` of
        // history, so the first possible output is at bar left+right.
        "pivothigh" | "pivotlow" => arg(args, 0) + arg(args, 1),
        "correl" => arg(args, 0).saturating_sub(1),
        "beta" => arg(args, 0),
        "obv" | "ad" => 0,
        "adosc" => arg(args, 0).max(arg(args, 1)).saturating_sub(1),
        "avgprice" | "medprice" | "typprice" | "wclprice" => 0,
        // Hilbert-transform suite: fixed warm-up (DCPERIOD/PHASOR/MAMA = 32; the
        // phase-dependent DCPHASE/SINE/TRENDLINE/TRENDMODE need 63).
        "ht_dcperiod" | "ht_phasor" | "mama" => 32,
        "ht_dcphase" | "ht_sine" | "ht_trendline" | "ht_trendmode" => 63,
        _ => unreachable!("every bound command has a lookback arm"), // LCOV_EXCL_LINE
    }
}

/// Lookback for a parsed directive: a command's own lookback plus the largest
/// lookback among its series operands; operators take the max of their operands.
pub fn lookback(node: &Ast) -> usize {
    match node {
        Node::Scalar(_) => 0,
        // A bare name is a column (no warm-up) or a no-argument command: bind it to
        // resolve its defaults — the same single bind `exec` uses — and an unbindable
        // name (a column reference) contributes no lookback.
        Node::Name(name) => match crate::bind::bind_command(name, None, &[], &[]) {
            Ok(cmd) => own_lookback(&cmd.name, cmd.sub.as_deref(), &cmd.args),
            Err(_) => 0,
        },
        Node::Command(Command {
            name,
            sub,
            args,
            series,
        }) => {
            let own = own_lookback(name, sub.as_deref(), args);
            let series_max = series.iter().map(lookback).max().unwrap_or(0);
            own + series_max
        }
        Node::Unary { operand, .. } => lookback(operand),
        Node::Binary { left, right, .. } => lookback(left).max(lookback(right)),
    }
}

fn command_recomputes_on_a_window(name: &str, sub: Option<&str>, args: &[ArgValue]) -> bool {
    !crate::spec::is_recursive(name, args)
        && !crate::spec::emits_position(name)
        && !crate::spec::accumulates(name, sub)
}

/// Whether every value of a directive can be recomputed exactly from a trailing window
/// of its inputs: each command in it has finite memory, a position-free output, and no
/// running accumulator ([`crate::spec::accumulates`] — those resume from a state). A
/// refresh after an append then recomputes only a window (its lookback plus a margin)
/// instead of the whole column; anything else recomputes from row 0. A bare name is
/// taken as the command it names (conservative); an unbindable one is a column.
pub fn recomputes_on_a_window(node: &Ast) -> bool {
    match node {
        Node::Scalar(_) => true,
        Node::Name(name) => match crate::bind::bind_command(name, None, &[], &[]) {
            Ok(cmd) => command_recomputes_on_a_window(&cmd.name, cmd.sub.as_deref(), &cmd.args),
            Err(_) => true,
        },
        Node::Command(c) => {
            command_recomputes_on_a_window(&c.name, c.sub.as_deref(), &c.args)
                && c.series.iter().all(recomputes_on_a_window)
        }
        Node::Unary { operand, .. } => recomputes_on_a_window(operand),
        Node::Binary { left, right, .. } => {
            recomputes_on_a_window(left) && recomputes_on_a_window(right)
        }
    }
}

/// The command a directive's resumable state belongs to — its name and sub-command: a
/// top-level command, or a bare name binding to one (a composite carries no state).
fn top_command(node: &Ast) -> Option<(String, Option<String>)> {
    match node {
        Node::Name(name) => crate::bind::bind_command(name, None, &[], &[]).ok().map(|c| (c.name, c.sub)),
        Node::Command(c) => Some((c.name.clone(), c.sub.clone())),
        _ => None,
    }
}

/// The rows a cached directive column reads around its own ([`volas_core::Reach`]),
/// fixed per directive: computed once when the column is cached, never per refresh.
pub fn reach(node: &Ast) -> volas_core::Reach {
    volas_core::Reach {
        lookback: lookback(node),
        resume: resume_reach(node),
        continuation: continuation_reach(node),
        history: history(node),
    }
}

/// How a directive's values depend on the rows before them ([`volas_core::History`]),
/// derived from the history class of each command in it ([`crate::spec::history_class`]):
/// its highest class, and for class 3 how a frame keeps it — by the state of its top
/// command when that is the one recursion in it (its operands of a lower class or
/// counts), by its window when every class-3 part is a count, otherwise not at all.
pub fn history(node: &Ast) -> volas_core::History {
    use crate::spec::Keep;
    use volas_core::History;
    let (class, recursions) = classes(node);
    match class {
        0 | 1 => return History::Window,
        2 => return History::Accumulator,
        _ if recursions == 0 => return History::Counted,
        _ => {}
    }
    let carried = |c: &Command<Box<[ArgValue]>>| {
        matches!(crate::spec::history_class(&c.name, c.sub.as_deref(), &c.args), HistoryClass::Recursion(Keep::State))
            && c.series.iter().all(|s| classes(s).1 == 0)
    };
    let top = match node {
        Node::Command(c) => Some(Cow::Borrowed(c)),
        Node::Name(name) => crate::bind::bind_command(name, None, &[], &[]).ok().map(Cow::Owned),
        _ => None,
    };
    match top {
        Some(c) if carried(&c) => History::Carried { operands: c.series.iter().map(lookback).max().unwrap_or(0) },
        _ => History::Unkept,
    }
}

/// The highest history class among a directive's commands (0 with none), and how many of
/// them are recursions (class 3 not kept by its window). A bare name is the command it
/// binds to; an unbindable one is a column.
fn classes(node: &Ast) -> (u8, usize) {
    let of = |name: &str, sub: Option<&str>, args: &[ArgValue]| match crate::spec::history_class(name, sub, args) {
        HistoryClass::Window => (1, 0),
        HistoryClass::Accumulator => (2, 0),
        HistoryClass::Recursion(crate::spec::Keep::Window) => (3, 0),
        HistoryClass::Recursion(_) => (3, 1),
    };
    let join = |a: (u8, usize), b: (u8, usize)| (a.0.max(b.0), a.1 + b.1);
    match node {
        Node::Scalar(_) => (0, 0),
        Node::Name(name) => crate::bind::bind_command(name, None, &[], &[])
            .map_or((0, 0), |c| of(&c.name, c.sub.as_deref(), &c.args)),
        Node::Command(c) => c.series.iter().map(classes).fold(of(&c.name, c.sub.as_deref(), &c.args), join),
        Node::Unary { operand, .. } => classes(operand),
        Node::Binary { left, right, .. } => join(classes(left), classes(right)),
    }
}

/// The rows before a new row that resuming a directive's carried state reads: none
/// for a state that is all its command needs (a recursion), its lookback plus one for a
/// command that reads the row its window drops or SAR's earlier bar
/// ([`crate::spec::reads_past_lookback`]). A frame holding fewer valid rows cannot
/// resume it.
fn resume_reach(node: &Ast) -> usize {
    match top_command(node) {
        Some((n, s)) if crate::spec::reads_past_lookback(&n, s.as_deref()) => lookback(node) + 1,
        _ => 0,
    }
}

/// The rows a cached directive column needs to be continued exactly at all — what a
/// contiguous slice must keep for the column to stay cached: its lookback, from which a
/// sliding sum or an extremum-position window recomputes exactly over the slice's own
/// rows when its state cannot resume; any other command reading past its lookback (a
/// recursion — SAR, KAMA) continues exactly only with
/// its [`resume_reach`] rows.
fn continuation_reach(node: &Ast) -> usize {
    match top_command(node) {
        Some((n, s))
            if !crate::spec::accumulates(&n, s.as_deref()) && !crate::spec::emits_position(&n) =>
        {
            resume_reach(node).max(lookback(node))
        }
        _ => lookback(node),
    }
}
/// Like [`lookback`], but requires the directive to name an **indicator**. A
/// top-level bare name that binds to no command — a raw column reference, or a
/// typo'd directive such as `ma5` for `ma:5` — is rejected (the same
/// `unknown command "…"` error `df.exec` raises) rather than silently treated as a
/// 0-lookback column; a bare sub-requiring name (`kdj`) is rejected too. This is
/// the form a windowed frame's `max_lookback` list takes, where a bare column
/// reference is meaningless and would silently under-size the retention margin.
///
/// Dataframe-agnostic: it validates the directive's form only — it never consults
/// a column — so it stays a pure function of the directive, exactly like
/// [`lookback`].
pub fn indicator_lookback(node: &Ast) -> volas_core::Result<usize> {
    // Reject a bare sub-requiring name anywhere (e.g. `kdj`, or nested `ma:5@kdj`).
    crate::bind::check_bare(node)?;
    // A standalone bare name must bind to a real command; an unbindable one is a
    // typo / column reference, not an indicator. (Nested bare names are data-source
    // column references, validated at execution, and intentionally not rejected.)
    if let Node::Name(name) = node {
        crate::bind::bind_command(name, None, &[], &[])?;
    }
    Ok(lookback(node))
}

#[cfg(test)]
mod tests {
    use super::lookback;
    use crate::parser::parse;

    /// Every `own_lookback` arm (and every generic-MA type) must compute without
    /// panicking. `lookback` is only reached on column caching, so `df.exec`-based
    /// parity tests never touch it — this is its coverage.
    #[test]
    fn lookback_covers_every_arm() {
        let directives = [
            // generic_ma_lookback: every MA type (3 DEMA, 4 TEMA, 6 KAMA, 7 MAMA, 8 T3, _ rest)
            "ma:5",
            "ma:5,1",
            "ma:5,3",
            "ma:5,4",
            "ma:5,6",
            "ma:5,7",
            "ma:5,8",
            "ema:5",
            "smma:5",
            "apo:12,26,1",
            "ppo:12,26,1",
            "mavp:2,30,1@close,close",
            "macdext",
            "macdext.signal",
            "macdfix",
            "macdfix.signal",
            "wma:30",
            "trima:30",
            "dema:30",
            "tema:30",
            "t3:5",
            "kama:30",
            "sar",
            "sarext",
            "boll",
            "bbw",
            "accbands",
            "macd",
            "macd.signal",
            "bbi",
            "tr",
            "atr:14",
            "llv:5",
            "hhv:5",
            "donchian:20",
            "rsv:9",
            "kdj.k:9,3",
            "rsi:14",
            "hv:10",
            "increase:3",
            "repeat:2",
            "style.doji",
            "cdl.doji",
            "change:2",
            "mom:10",
            "roc:10",
            "rocp:10",
            "rocr:10",
            "rocr100:10",
            "willr:14",
            "midpoint:14",
            "midprice:14",
            "cmo:14",
            "natr:14",
            "cci:14",
            "imi:14",
            "mfi:14",
            "ultosc",
            "stoch.k",
            "stoch.d",
            "stochf.k",
            "stochf.d",
            "stochrsi.k",
            "stochrsi.d",
            "plus_dm:14",
            "minus_dm:14",
            "plus_di:14",
            "minus_di:14",
            "dx:14",
            "adx:14",
            "adxr:14",
            "trix:30",
            "aroon.up:14",
            "aroonosc:14",
            "sum:30",
            "maxindex:30",
            "minindex:30",
            "minmax.min:30",
            "minmaxindex.min:30",
            "bop",
            "linearreg:14",
            "linearreg_slope:14",
            "tsf:14",
            "var:5",
            "stddev:5",
            "correl:30@close,close",
            "beta:5@close,close",
            "obv",
            "ad",
            "adosc:3,10",
            "avgprice",
            "medprice",
            "typprice",
            "wclprice",
            "ht_dcperiod",
            "ht_phasor",
            "ht_phasor.quadrature",
            "mama",
            "mama.fama",
            "ht_dcphase",
            "ht_sine",
            "ht_sine.leadsine",
            "ht_trendline",
            "ht_trendmode",
            // node kinds: scalar, plain column name, unary, binary
            "5",
            "close",
            "~(close > 5)",
            "ma:5 + ma:10",
        ];
        for d in directives {
            let _ = lookback(&parse(d).unwrap());
        }
    }

    #[test]
    fn indicator_lookback_accepts_indicators_rejects_bare_non_commands() {
        use super::indicator_lookback;
        // command directives + a bare no-arg command derive their warm-up
        assert_eq!(indicator_lookback(&parse("ma:5").unwrap()).unwrap(), 4);
        assert_eq!(indicator_lookback(&parse("atr:14").unwrap()).unwrap(), 14);
        assert_eq!(indicator_lookback(&parse("ma:5 + ema:10").unwrap()).unwrap(), 9);
        indicator_lookback(&parse("obv").unwrap()).unwrap(); // bare command -> Ok
        // a command may reference columns via @col — those nested bare names are
        // data sources, not rejected (validated at execution).
        indicator_lookback(&parse("ma:5@close").unwrap()).unwrap();
        // a standalone bare name that binds to no command is a typo / column ref
        for bad in ["ma5", "atr14", "close", "kdj"] {
            assert!(
                indicator_lookback(&parse(bad).unwrap()).is_err(),
                "{bad:?} must be rejected as a non-indicator directive"
            );
        }
    }
}
