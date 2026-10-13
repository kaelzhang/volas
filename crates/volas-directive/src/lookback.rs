//! A directive's row-reach facts — its lookback (the rows before its first value), and
//! the [`volas_core::Reach`] a cached column carries — derived from the specs of the
//! commands in it ([`crate::spec::CommandSpec`]): each command line owns its lookback and
//! history class, and this module only walks the tree.

use std::borrow::Cow;

use crate::spec::{CommandSpec, History, HistoryClass, Keep};
use crate::types::{ArgValue, Ast, BoundNode, Node};

/// The spec and bound arguments of a command node, or of a bare name binding to a
/// command (`None` for anything else: a column, a scalar, an operation).
fn command_of(node: &BoundNode) -> Option<(Cow<'_, CommandSpec>, Cow<'_, [ArgValue]>)> {
    match node {
        Node::Command(c) => Some((Cow::Borrowed(&c.args.spec), Cow::Borrowed(&c.args.values[..]))),
        Node::Name(name) => crate::bind::bind_command_node(name, None, &[], &[])
            .ok()
            .map(|c| (Cow::Owned(c.args.spec), Cow::Owned(c.args.values.into_vec()))),
        _ => None,
    }
}

/// Lookback for a bound directive: a command's own lookback plus the largest lookback
/// among its series operands; an operator takes the max of its operands plus the earlier
/// rows it reads itself ([`crate::types::Op::lookback`]: a cross reads one).
pub fn lookback(node: &Ast) -> usize { lookback_node(node.node()) }

pub(crate) fn lookback_node(node: &BoundNode) -> usize {
    match node {
        Node::Scalar(_) => 0,
        // A bare name is a column (no warm-up) or a no-argument command.
        Node::Name(_) => command_of(node).map_or(0, |(spec, args)| spec.lookback_of(&args)),
        Node::Command(c) => {
            let own = c.args.spec.lookback_of(&c.args);
            own + c.series.iter().map(lookback_node).max().unwrap_or(0)
        }
        Node::Unary { operand, .. } => lookback_node(operand),
        Node::Binary { left, op, right } => op.lookback() + lookback_node(left).max(lookback_node(right)),
    }
}

/// Whether a directive evaluated over a trailing window of the frame's rows (keeping
/// their absolute positions) gives, on the window's rows past its lookback, exactly its
/// values over the whole frame: no recursion and no row-position output anywhere in it
/// — each command a pure window, or a running accumulator whose sums are exact and
/// whose anchors are absolute positions. A continuation evaluates such an `@` operand
/// over the window it reads instead of the frame, and a refresh recomputes such a
/// directive with no carried state over a trailing window (its lookback plus a margin)
/// instead of the whole column. A bare name is taken as the command it names; an
/// unbindable one is a column.
pub fn evaluates_on_a_window(node: &Ast) -> bool { evaluates_on_a_window_node(node.node()) }

pub(crate) fn evaluates_on_a_window_node(node: &BoundNode) -> bool {
    let own = |node: &BoundNode| {
        command_of(node).is_none_or(|(spec, args)| spec.history(&args) != History::Recursive && !spec.emits_position())
    };
    match node {
        Node::Scalar(_) | Node::Name(_) => own(node),
        Node::Command(c) => own(node) && c.series.iter().all(evaluates_on_a_window_node),
        Node::Unary { operand, .. } => evaluates_on_a_window_node(operand),
        Node::Binary { left, right, .. } => evaluates_on_a_window_node(left) && evaluates_on_a_window_node(right),
    }
}

/// The rows a cached directive column reads around its own ([`volas_core::Reach`]),
/// fixed per directive: computed once when the column is cached, never per refresh.
pub fn reach(node: &Ast) -> volas_core::Reach {
    volas_core::Reach {
        lookback: lookback_node(node.node()),
        resume: resume_reach(node.node()),
        continuation: continuation_reach(node.node()),
        history: history(node),
    }
}

/// The highest command history class, with recursive keeping derived solely from
/// the stored continuation plan. Own and Chain retain their state; a Recompute
/// recursion anywhere leaves the directive Unkept.
pub fn history(node: &Ast) -> volas_core::History {
    use crate::types::Continuation;
    let h = history_node(node.node());
    if !h.recursive() { return h; }
    match node.continuation() {
        Continuation::Own => volas_core::History::Carried { operands: series_of(node.node()).iter().map(lookback_node).max().unwrap_or(0) },
        Continuation::Chain(p) => volas_core::History::Carried { operands: p.frame_span },
        Continuation::None => volas_core::History::Unkept,
    }
}

fn history_node(node: &BoundNode) -> volas_core::History {
    use volas_core::History;
    let (class, recursions) = classes(node);
    match class {
        0 | 1 => return History::Window,
        2 => return History::Accumulator,
        _ if recursions == 0 => return History::Counted,
        _ => {}
    }

    History::Unkept
}

/// The `@` operands of a command node (none for anything else).
fn series_of(node: &BoundNode) -> &[BoundNode] {
    match node {
        Node::Command(c) => &c.series,
        _ => &[],
    }
}

/// The highest history class among a directive's commands (0 with none), and how many of
/// them are recursions (class 3 not kept by its window). A bare name is the command it
/// binds to; an unbindable one is a column.
fn classes(node: &BoundNode) -> (u8, usize) {
    let own = |node: &BoundNode| {
        command_of(node).map_or((0, 0), |(spec, args)| match spec.history_class(&args) {
            HistoryClass::Window => (1, 0),
            HistoryClass::Accumulator => (2, 0),
            HistoryClass::Recursion(Keep::Window) => (3, 0),
            HistoryClass::Recursion(_) => (3, 1),
        })
    };
    let join = |a: (u8, usize), b: (u8, usize)| (a.0.max(b.0), a.1 + b.1);
    match node {
        Node::Scalar(_) | Node::Name(_) => own(node),
        Node::Command(c) => c.series.iter().map(classes).fold(own(node), join),
        Node::Unary { operand, .. } => classes(operand),
        Node::Binary { left, right, .. } => join(classes(left), classes(right)),
    }
}

/// The kernel's published resume reach plus the window of its operands. Complete
/// input tails carry those kernel rows across missing runs and head compaction.
fn resume_reach(node: &BoundNode) -> usize {
    match command_of(node) {
        Some((spec, args)) => spec.resume_reach(&args) + series_of(node).iter().map(lookback_node).max().unwrap_or(0),
        _ => 0,
    }
}

/// The rows a cached directive column needs to be continued exactly at all — what a
/// contiguous slice must keep for the column to stay cached: its lookback, from which a
/// sliding sum or an extremum-position window recomputes exactly over the slice's own
/// rows when its state cannot resume; any other command reading past its lookback (a
/// recursion — SAR, KAMA) continues exactly only with
/// its [`resume_reach`] rows.
fn continuation_reach(node: &BoundNode) -> usize {
    match command_of(node) {
        Some((spec, _)) if !spec.accumulates() && !spec.emits_position() => resume_reach(node).max(lookback_node(node)),
        _ => lookback_node(node),
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
pub fn indicator_lookback(node: &Ast) -> volas_core::Result<usize> { indicator_lookback_node(node.node()) }

fn indicator_lookback_node(node: &BoundNode) -> volas_core::Result<usize> {
    // Reject a bare sub-requiring name anywhere (e.g. `kdj`, or nested `ma:5@kdj`).
    crate::bind::check_bare_node(node)?;
    // A standalone bare name must bind to a real command; an unbindable one is a
    // typo / column reference, not an indicator. (Nested bare names are data-source
    // column references, validated at execution, and intentionally not rejected.)
    if let Node::Name(name) = node {
        crate::bind::bind_command_node(name, None, &[], &[])?;
    }
    Ok(lookback_node(node))
}

#[cfg(test)]
mod tests {
    use super::lookback;
    use crate::parser::parse;

    /// Every line's lookback (and every generic-MA type) must compute without
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
            "repeat:2@(close > 1)",
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
            "ma:5 >< ma:10",
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

#[cfg(test)]
mod reach_tests {
    use super::{evaluates_on_a_window, lookback};
    use crate::parser::parse;

    /// A cross reads the row before, so it warms up one row after its operands — and a
    /// nested cross one more each level.
    #[test]
    fn a_cross_reads_one_row_more() {
        let lb = |d: &str| lookback(&parse(d).unwrap());
        assert_eq!(lb("close >< open"), 1);
        assert_eq!(lb("ma:5 // ma:3"), 5);
        assert_eq!(lb("((close >< open) >< (open > close)) >< (close > 1)"), 3);
        assert_eq!(lb("close > open"), 0);
    }

    #[test]
    fn a_window_evaluation_needs_no_recursion_and_no_position() {
        let e = |d: &str| evaluates_on_a_window(&parse(d).unwrap());
        assert!(e("llv:5 + 1") && e("-(roc:3)") && e("stochf.k:5,3,1") && e("close > ma:20"));
        assert!(!e("ema:5") && !e("maxindex:5") && !e("llv:5@(ema:3)"));
        assert!(e("ma:5") && e("sum:3@(close-open)") && e("-(var:4)") && e("close") && e("llv:5"));
        assert!(!e("ema:3") && !e("ma:5@(rsi:3)") && !e("minindex:4") && !e("obv") && !e("1+ma:3,1"));
    }
}
