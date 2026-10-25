//! Execute a directive AST against a [`DataFrame`], producing a [`Column`].


use volas_core::ColumnView;
use crate::bind::bind_command_node;
use crate::operands::Operands;
use crate::spec::History;
use crate::types::{ArgValue, Ast, Bound, BoundNode, Command, Node, Number};
use volas_compute::raw::indicators as ind;
use volas_compute::missing::{on_complete_rows, Input};
use volas_core::Column;
use volas_core::DataFrame;
use volas_core::{Result, VolasError};
use volas_time::Span;

/// Execute a directive node against `df`.
///
/// Numbers keep their exactness: an integer literal is an int64 column, a
/// comparison orders integers exactly ([`Column::order_rows`]), and float64 math
/// (`+ - * /`, unary `-`, every indicator input) raises for an int64 past 2^53
/// (contract #26). Booleans carry presence (contract #27): a comparison with a
/// missing operand is missing, and `~ & | ^` use Kleene logic.
pub fn execute(df: &DataFrame, node: &Ast) -> Result<Column> {
    execute_node(df, node.node())
}

pub(crate) fn execute_node(df: &DataFrame, node: &BoundNode) -> Result<Column> {
    match node {
        Node::Scalar(Number::Int(v)) => Ok(Column::i64(vec![*v; df.height()])),
        Node::Scalar(Number::Float(v)) => Ok(Column::f64(vec![*v; df.height()])),
        Node::Data(name) => Ok(df.column(name)?.clone()),
        Node::Name(name) => {
            if name.is_empty() {
                return Err(VolasError::Value("empty column / command name".into()));
            }
            if df.has_column(name) {
                Ok(df.column(name)?.clone())
            } else {
                // not bound against a frame (`crate::bind::bind_frame`): a bare name that
                // is no column is the command it names
                dispatch(df, &bind_command_node(name, None, &[], &[])?)
            }
        }
        Node::Command(cmd) => dispatch(df, cmd),
        Node::Unary { op, operand } => {
            let c = execute_node(df, operand)?;
            require_numeric(&c)?; // a str/datetime operand can't go through the f64 funnel (C4)
            crate::element::unary_column(*op, &c)
        }
        Node::Binary { left, op, right } => {
            let l = execute_node(df, left)?;
            let r = execute_node(df, right)?;
            require_numeric(&l)?; // both operands must be numeric (or bool) — C4, no
            require_numeric(&r)?; // silent str -> NaN arithmetic / comparison
            crate::element::binary_column(*op, &l, &r)
        }
    }
}

// --- argument accessors -----------------------------------------------------
// Arguments are bound once (parsed, defaulted, type-checked against the spec) by
// `bind`, so every position is present and its variant is fixed — dispatch reads
// the typed value directly, with no re-parsing, defaulting, or fallibility.

pub(crate) fn arg_usize(args: &[ArgValue], i: usize) -> usize {
    args[i].as_usize()
}

pub(crate) fn arg_f64(args: &[ArgValue], i: usize) -> f64 {
    args[i].as_f64()
}

pub(crate) fn arg_i64(args: &[ArgValue], i: usize) -> i64 {
    args[i].as_i64()
}

// --- series resolution ------------------------------------------------------

/// The `@` operand filling series slot `i`, if any: an absent slot or an empty name
/// leaves it to the command's default series.
pub(crate) fn operand(series: &[BoundNode], i: usize) -> Option<&BoundNode> {
    series
        .get(i)
        .filter(|n| !matches!(n, Node::Name(s) if s.is_empty()))
}

/// A directive numeric operand must be a numeric column (float / int / bool —
/// bool acts as 0/1). A `Str` / `Datetime` column funneled through `to_f64_vec`
/// would silently become all-`NaN` (or, for datetime, f64-quantized epoch), so
/// reject it (API contract C4 — a lossy implicit conversion errors, never a silent
/// all-NaN feature column).
pub(crate) fn require_numeric(col: &Column) -> Result<()> {
    match col.view() {
        ColumnView::Str(..) | ColumnView::Datetime(..) => Err(VolasError::Value(format!(
            "directive operand must be a numeric column, got {}",
            col.dtype()
        ))),
        _ => Ok(()),
    }
}

/// An indicator input: a numeric column whose values are what float64 math reads —
/// an int64 past 2^53 raises (contract #26) instead of feeding a rounded value.
pub(crate) fn require_f64_input(col: &Column) -> Result<()> {
    require_numeric(col)?;
    col.check_f64_domain("an indicator input")
}

/// The generic moving average (TA-Lib `TA_MA`, [`ind::generic_ma`]) of `period` and
/// `matype` over `data`. A recursive average runs on the complete rows of its input — the
/// input may be a derived line that warms up with NaN (see `volas_compute::missing`). A
/// recursive command's own input is complete wherever its result is kept (`complete`), so
/// it is not scanned. Shared by `ma`, `apo`, `ppo`, `macdext`, `mavp` and the stochastics.
fn ma_typed(data: &[f64], period: usize, matype: usize, complete: bool, origin: usize) -> Vec<f64> {
    if ind::is_recursive_generic_ma(period, matype) {
        let input = if complete { Input::known(data, false) } else { Input::new(data) };
        return on_complete_rows(&[input], |x| ind::generic_ma(x[0], period, matype, origin));
    }
    ind::generic_ma(data, period, matype, origin)
}

// --- command dispatch -------------------------------------------------------


mod cycle;
mod momentum;
mod overlap;
pub(crate) use overlap::percent_oscillator;

/// The dispatch entry point: a bound command (canonical name / sub, typed args, bound
/// series, its line's spec) routed to its kernel. Validation / defaulting /
/// normalization already happened at `bind`, so this is pure dispatch — the `overlap`
/// -> `momentum` -> `cycle` chain is an internal split of the one command table. A
/// recursive line runs on the complete rows of its inputs (see `crate::complete`).
fn dispatch(df: &DataFrame, cmd: &Command<Bound>) -> Result<Column> {
    let (name, sub, args) = (cmd.name.as_str(), cmd.sub.as_deref(), &cmd.args.values[..]);
    if cmd.args.spec.history(args) == History::Recursive {
        return crate::complete::compute(df, &cmd.series, |ops| overlap::dispatch(ops, name, sub, args));
    }
    overlap::dispatch(&Operands::new(df, &cmd.series), name, sub, args)
}

/// Execute one bound command against inputs already resolved by the continuation owner.
pub(crate) fn execute_inputs(ops: &Operands, cmd: &crate::complete::BoundCommand) -> Result<Column> {
    overlap::dispatch(ops, cmd.name, cmd.sub, cmd.args)
}

pub use crate::exec_resume::{
    capture, capture_states, checkpoint, checkpoints, execute_resume, execute_resume_step, execute_with_states, execute_with_states_and_anchors, execute_for_cache, initial_state, replay,
    WithStates,
};

/// `hv`'s annualization factor √(bars per year), from its bound arguments
/// `[period, time_frame, trading_days, session_minutes]` — the one owner of the rule,
/// shared by the full and the resumed computation. Bars per year follow the time
/// frame's unit:
///
/// * a sub-day bar of `s` seconds: `trading_days × session_minutes × 60 / s` (bars per
///   trading session times sessions per year, exact for second bars);
/// * `k` days: `trading_days / k`;
/// * `k` weeks: `52 / k`; `k` months: `12 / k`; `k` years: `1 / k`.
///
/// `session_minutes` applies to sub-day bars only and `trading_days` to sub-day and
/// day bars only.
pub(crate) fn hv_annualization(args: &[ArgValue]) -> f64 {
    let trading_days = args[2].as_i64() as f64;
    let bars_per_year = match args[1].as_time_frame().span() {
        Span::Seconds(s) => trading_days * (args[3].as_usize() * 60) as f64 / s as f64,
        Span::Days(k) => trading_days / k as f64,
        Span::Weeks(k) => 52.0 / k as f64,
        Span::Months(k) => 12.0 / k as f64,
        Span::Years(k) => 1.0 / k as f64,
    };
    bars_per_year.sqrt()
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

    /// A period far larger than the frame is a LEGAL configuration (pure
    /// warm-up: a valid-length all-NaN column) and must trip every indicator's
    /// `period == 0 || period > n` warm-up guard line rather than panic. (A
    /// period of 0 is no longer reachable here — the exec boundary rejects it,
    /// V17 — so the over-length form is what exercises the shared guard line;
    /// the kernels' own `== 0` defense is covered by the direct calls below.)
    #[test]
    fn degenerate_periods_hit_compute_guards() {
        let df = ohlcv();
        let n = 30;
        let over = [
            "ma:99999",
            "ema:99999",
            "smma:99999",
            "wma:99999",
            "dema:99999",
            "tema:99999",
            "trima:99999",
            "t3:99999",
            "kama:99999",
            "mom:99999",
            "roc:99999",
            "rocp:99999",
            "rocr:99999",
            "rocr100:99999",
            "willr:99999",
            "cci:99999",
            "cmo:99999",
            "mfi:99999",
            "trix:99999",
            "midpoint:99999",
            "midprice:99999",
            "atr:99999",
            "natr:99999",
            "rsi:99999",
            "plus_dm:99999",
            "minus_dm:99999",
            "plus_di:99999",
            "minus_di:99999",
            "dx:99999",
            "adx:99999",
            "adxr:99999",
            "aroon.up:99999",
            "aroonosc:99999",
            "sum:99999",
            "maxindex:99999",
            "minindex:99999",
            "minmax.min:99999",
            "minmaxindex.min:99999",
            "linearreg:99999",
            "linearreg_slope:99999",
            "linearreg_intercept:99999",
            "linearreg_angle:99999",
            "tsf:99999",
            "var:99999",
            "stddev:99999",
            "llv:99999",
            "hhv:99999",
            "boll:99999",
            "bbw:99999",
            "accbands:99999",
            "correl:99999@close,close",
            "beta:99999@close,close",
            "change:99999",
            "repeat:99999@(close>10)",
        ];
        for d in over {
            let col = execute(&df, &parse(d).unwrap())
                .unwrap_or_else(|e| panic!("directive {d:?} failed: {e:?}"));
            assert_eq!(col.len(), n, "directive {d:?} returned wrong length");
        }
        // The kernels' own `period == 0` defense stays (a direct Rust caller can
        // still pass 0); the kernels whose zero-guard sits on its own line —
        // unreachable from the validated directive boundary — are exercised
        // directly: the underflow-prone tools, the ADX/ADXR pair, and the
        // EMA-cascade warm-up (via DEMA / TEMA / T3).
        assert!(ind::change(&[1.0, 2.0], 0).iter().all(|x| x.is_nan()));
        assert!(ind::repeat(&[Some(true), Some(true)], 0).iter().all(Option::is_none));
        assert!(ind::increase(&[1.0, 2.0], 0, 1).iter().all(Option::is_none)); // no window, no underflow
        let (h, l, c) = (vec![2.0; 5], vec![1.0; 5], vec![1.5; 5]);
        assert!(ind::adx(&h, &l, &c, 0).iter().all(|x| x.is_nan()));
        assert!(ind::adxr(&h, &l, &c, 0).iter().all(|x| x.is_nan()));
        assert!(ind::dema(&c, 0).iter().all(|x| x.is_nan()));
        assert!(ind::tema(&c, 0).iter().all(|x| x.is_nan()));
        assert!(ind::t3(&c, 0, 0.7).iter().all(|x| x.is_nan()));
    }

    /// Every command family rejects an out-of-domain argument at the exec
    /// boundary (V17): a zero window, a sub-2 second-moment window, an
    /// out-of-range matype / seed / limit, a negative SAR factor, a zero ASI
    /// divisor, a non-±1 direction, a NaN multiplier, inverted mavp limits.
    #[test]
    fn argument_bounds_reject_invalid_configurations() {
        let df = ohlcv();
        // `parse` now binds + validates, so an invalid directive is rejected there;
        // chain through execute to also catch the runtime-only failures (series, columns).
        let is_err = |d: &str| parse(d).and_then(|n| execute(&df, &n)).is_err();
        // zero periods: a valid-shaped all-NaN column is not a legal config.
        for d in [
            "ma:0", "ema:0", "atr:0", "rsi:0", "sum:0", "mom:0", "mfi:0",
            "boll:0", "kdj.k:0", "stoch.k:0,3,0,3,0", "ultosc:0,14,28",
            "repeat:0@(close>10)",
        ] {
            assert!(is_err(d), "{d} must be rejected");
        }
        // second-moment / regression windows need >= 2 samples.
        for d in [
            "linearreg:1", "linearreg_slope:1", "tsf:1", "var:1", "stddev:1",
            "correl:1@close,close", "beta:1@close,close", "cci:1",
            "aroon.up:1", "aroonosc:1", "change:1", "hv:1",
        ] {
            assert!(is_err(d), "{d} must be rejected");
        }
        // selector / domain arguments.
        assert!(is_err("ma:5,9"), "matype past 8");
        assert!(is_err("stoch.k:5,3,9,3,0"), "stoch matype past 8");
        assert!(is_err("kdj.k:9,3,150"), "KDJ seed outside [0, 100]");
        assert!(is_err("t3:5,1.5"), "T3 vfactor outside [0, 1]");
        assert!(is_err("mama:1.5,0.05"), "MAMA fast limit outside [0.01, 0.99]");
        assert!(is_err("mama:0,0"), "MAMA limits outside [0.01, 0.99]");
        assert!(is_err("sar:-0.1,0.2"), "negative SAR acceleration");
        assert!(is_err("sarext:0,-1"), "negative SAR reverse offset");
        assert!(is_err("asi:0"), "ASI divides by its limit-move argument");
        assert!(is_err("increase:3,0"), "direction must be +1 / -1");
        assert!(is_err("hv:10,1d,0"), "hv trading days must be >= 1");
        assert!(is_err("hv:10,0m"), "a zero time-frame span");
        assert!(is_err("hv:10,30s"), "not a TimeFrame label");
        assert!(is_err("hv:10,1\u{e9}"), "a multibyte last character");
        assert!(is_err("hv:10,15m,252,0"), "a zero-minute session");
        assert!(is_err("hv:10,15m,252,1441"), "a session longer than a day");
        assert!(is_err("boll.upper:20,nan"), "NaN multiplier poisons the band");
        assert!(is_err("mavp:30,2@close,close"), "inverted mavp period limits");
        assert!(is_err("mavp:0,30@close,close"), "mavp min period below 1");
        // the legal edges of the same domains still execute.
        for d in [
            "ma:1", "atr:1", "rsi:1", "t3:5,0", "t3:5,1", "kdj.k:9,3,0",
            "kdj.k:9,3,100", "mama:0.01,0.01", "mama:0.99,0.99", "sar:0,0",
            "increase:3,-1", "boll.upper:20,-2", "var:2", "linearreg:2",
            "change:2", "aroon.up:2",
        ] {
            let col = execute(&df, &parse(d).unwrap())
                .unwrap_or_else(|e| panic!("directive {d:?} failed: {e:?}"));
            assert_eq!(col.len(), 30, "directive {d:?}");
        }
    }

    /// An empty frame must trip the `n == 0` guards (volume / SAR / price transforms).
    #[test]
    fn empty_frame_hits_compute_guards() {
        let df = ohlcv_n(0);
        for d in [
            "obv",
            "ad",
            "adosc:3,10",
            "sar",
            "sarext",
            "ma:5",
            "tr",
            "avgprice",
            "ht_dcperiod",
            "mama",
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
            "style.doji",
            "style.engulfing",
            "style.morningstar",
            "style.hikkake",
            "style.hikkakemod",
            "style.concealbabyswall",
            "style.3whitesoldiers",
            "style.risefall3methods",
            "style.breakaway",
            "style.mathold",
        ] {
            let col = execute(&df, &parse(d).unwrap()).unwrap();
            assert_eq!(col.len(), 4, "directive {d:?}");
        }
    }

    #[test]
    fn required_series_and_unknown_matype_errors() {
        let df = ohlcv();
        let is_err = |d: &str| parse(d).and_then(|n| execute(&df, &n)).is_err();
        // correl needs a second series operand; absent or empty -> error at execute
        // (`Operands::f64_required`'s None and empty-Name arms).
        assert!(is_err("correl:30"));
        assert!(is_err("correl:30@close,"));
        // a matype past 8 is rejected when binding (IntRange bound); ma_typed itself is
        // infallible — every caller's matype is validated.
        assert!(is_err("ma:5,9"));
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
            "style.bullish",
            "style.bearish",
            "repeat:2@(style.bullish)",
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
            "hv:10,1s,252,390",
            "hv:10,3d",
            "hv:10,1W",
            "hv:10,1M",
            "hv:10,1Y",
        ] {
            assert_eq!(execute(&df, &parse(d).unwrap()).unwrap().len(), 30);
        }
    }

    /// Bars per year by the time frame's unit: sub-day bars scale by the session,
    /// day bars by the trading days, and weeks / months / years are fixed counts.
    #[test]
    fn hv_annualization_follows_the_time_frame_unit() {
        let bars_per_year = |d: &str| {
            let args: Vec<Option<String>> =
                d["hv:".len()..].split(',').map(|t| Some(t.to_string())).collect();
            hv_annualization(&bind_command_node("hv", None, &args, &[]).unwrap().args).powi(2)
        };
        let close = |d: &str, want: f64| {
            let got = bars_per_year(d);
            assert!((got - want).abs() <= want * 1e-14, "{d}: {got} vs {want}");
        };
        close("hv:10", 252.0);
        close("hv:10,1d,365", 365.0);
        close("hv:10,3d", 84.0);
        close("hv:10,1w", 52.0);
        close("hv:10,1M", 12.0);
        close("hv:10,1y", 1.0);
        close("hv:10,15m,252,390", 252.0 * 26.0);
        close("hv:10,4h,252,390", 252.0 * 1.625);
        close("hv:10,1s,252,240", 252.0 * 240.0 * 60.0);
        close("hv:10,15m,365", 365.0 * 96.0);
        close("hv:10,4h,252,240", 252.0); // a bar as long as the session: one per day
    }

    #[test]
    fn command_and_argument_validation_errors() {
        let df = ohlcv();
        // `parse` now binds + validates, so an invalid directive is rejected there;
        // chain through execute to also catch the runtime-only failures (series, columns).
        let is_err = |d: &str| parse(d).and_then(|n| execute(&df, &n)).is_err();
        assert!(is_err("ema:5,6"), "too many args"); // ema takes one period (ma now takes matype too)
        assert!(is_err("frobnicate:5"), "unknown command");
        assert!(
            is_err("MACD.bogus"),
            "case-insensitive name, still bad sub (P6)"
        );
        assert!(is_err("kdj:9"), "missing required sub-command");
        assert!(is_err("macd.bogus"), "unknown sub-command");
        assert!(is_err("ma:abc"), "non-integer arg");
        assert!(is_err("style.cartoon"), "invalid style sub-command");
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
        assert!(execute_node(&stock(), &Node::Name(String::new())).is_err());
    }

    /// `initial_state` / `execute_resume` decline (return `None`, so the engine keeps the
    /// full-recompute fallback) on malformed states, a `*_resume` that returns `None`
    /// (from below its warm-up), and a directive
    /// with no resume kernel at all (the `_ => None` catch-all).
    #[test]
    fn state_carry_none_propagation() {
        let df = ohlcv_n(80);
        let p = |d: &str| parse(d).unwrap();

        // Every StochRSI MA type now carries a state.
        assert!(initial_state(&df, &p("stochrsi.d:14,14,3,1"), usize::MAX).is_some());

        // execute_resume: SAR / SAREXT at from_row < 2 -> the inner `*_resume` is None, the
        // `?` propagates (exec:1095 / 1114). State contents are unread on the None path.
        let sar_state = vec![1.0, 0.02, 8.0, 6.0, 9.0, 7.0];
        assert!(execute_resume(&df, &p("sar"), &sar_state, 1, df.height(), 0).is_none());
        let sarext_state = vec![1.0, 0.02, 0.02, 8.0, 6.0, 9.0, 7.0];
        assert!(execute_resume(&df, &p("sarext"), &sarext_state, 1, df.height(), 0).is_none());

        // execute_resume: ±DI at from_row == 0 -> inner resume None, `?` propagates
        // (exec:1245 / 1254).
        let di_state = vec![0.0, 0.0];
        assert!(execute_resume(&df, &p("plus_di:14"), &di_state, 0, df.height(), 0).is_none());
        assert!(execute_resume(&df, &p("minus_di:14"), &di_state, 0, df.height(), 0).is_none());

        // execute_resume: MAMA with a too-short carried state -> mama_resume None (exec:1320).
        // `from_row` is past the HT core warm-up so the length check is what declines.
        let short_mama = vec![0.0; 10];
        assert!(execute_resume(&df, &p("mama"), &short_mama, 20, df.height(), 0).is_none());

        // execute_resume: index family below its `period - 1` warm-up -> resume None
        // (exec:1335 / 1345). period 30, from_row 5 < 29.
        let idx_state = vec![3.0, 100.0];
        assert!(execute_resume(&df, &p("maxindex:30"), &idx_state, 5, df.height(), 0).is_none());
        assert!(execute_resume(&df, &p("minindex:30"), &idx_state, 5, df.height(), 0).is_none());

        // execute_resume: KDJ below its RSV window warm-up (from_row + 1 < period_rsv = 9) ->
        // kdj_resume None, the `?` propagates (exec:1402).
        let kdj_state = vec![50.0, 50.0];
        assert!(execute_resume(&df, &p("kdj.j"), &kdj_state, 2, df.height(), 0).is_none());

        // execute_resume: a recursive-MA StochRSI state of the wrong shape declines.
        let sr_state = vec![0.0; 20];
        assert!(execute_resume(&df, &p("stochrsi.d:14,14,3,1"), &sr_state, 40, df.height(), 0).is_none());

        // execute_resume: stochrsi `.k` with a malformed (wrong-length) state -> the inner
        // stochrsi_resume is None, the `?` propagates (exec:1365).
        let bad_sr = vec![0.0; 3]; // not `ctx_depth + 2` for these periods
        assert!(execute_resume(&df, &p("stochrsi.k:14,14,3"), &bad_sr, 40, df.height(), 0).is_none());

        // execute_resume: a directive with no resume kernel hits the `_ => None` arm
        // (exec:1369). `llv` (lowest-low) has no incremental resume.
        assert!(execute_resume(&df, &p("llv:5"), &[0.0], 3, df.height(), 0).is_none());

        // execute_resume_step: a recursive directive without a single-row step
        // (`stochrsi`), a finite-memory one (`llv`), a `macd` sub-line that is no step's,
        // and a `next` of the wrong length all decline, so the refresh takes the Vec path.
        let mut next = [0.0; 2];
        assert!(execute_resume_step(&df, &p("stochrsi.k:14,14,3"), &[1.0], (5, 0), &mut next[..1])
            .is_none());
        assert!(execute_resume_step(&df, &p("llv:5"), &[1.0], (5, 0), &mut next[..1]).is_none());
        assert!(execute_resume_step(&df, &p("rsi:14"), &[1.0, 1.0], (5, 0), &mut next[..1]).is_none());
        // a composite average declines a state of the wrong length, and a row past the end
        assert!(execute_resume_step(&df, &p("apo"), &[1.0, 1.0], (5, 0), &mut next[..2]).is_none());
        let apo = initial_state(&df, &p("apo"), usize::MAX).unwrap();
        let mut n = apo.clone();
        assert!(execute_resume_step(&df, &p("apo"), &apo, (df.height(), 0), &mut n).is_none());
    }
}
