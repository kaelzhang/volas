//! A recursive command runs on the complete rows of its inputs.
//!
//! The rule — what a missing row is, and which rows a full computation, its
//! carried-state capture and a resume run on — is owned by [`volas_compute::missing`].
//! This module applies it to a bound recursive command. The command's arm runs in
//! place first, reading its inputs through [`Operands`], which records exactly what it
//! read: when every column read is known complete (the column's own remembered fact,
//! `Column::has_nonfinite`) that result stands with no scan at all; otherwise the record says
//! which rows are complete, and the arm runs again over just those rows of the same
//! inputs, its outputs scattered back. The arms and kernels are unchanged. A read that
//! failed leaves the record without that input, so the arm's own result — the error —
//! stands and nothing reruns (see [`Operands::failed`]).

use std::borrow::Cow;

use volas_compute::missing::{CompleteRows, Input};
use volas_core::{Column, DataFrame, Result};

use crate::operands::Operands;
use crate::spec::CommandSpec;
use crate::types::{ArgValue, BoundNode};

/// A bound command: canonical name / sub, typed arguments, `@` series, its line's spec.
#[derive(Clone, Copy)]
pub(crate) struct BoundCommand<'a> {
    pub(crate) name: &'a str,
    pub(crate) sub: Option<&'a str>,
    pub(crate) args: &'a [ArgValue],
    pub(crate) series: &'a [BoundNode],
    pub(crate) spec: &'a CommandSpec,
}

impl BoundCommand<'_> {
    /// How many rows before its first new row a resume of the command reads, at most.
    pub(crate) fn reach(&self) -> usize {
        self.spec.resume_reach(self.args)
    }
}

/// One input an arm read, as the rule sees it.
struct Recorded<'v> {
    slot: usize,
    values: Cow<'v, [f64]>,
    /// Whether the values hold a NaN, if known.
    has_missing: Option<bool>,
}

impl Recorded<'_> {
    /// The rows `[..end)`, with the NaN fact (which a complete whole implies for them).
    fn input(&self, end: usize) -> Input<'_> {
        let v = &self.values[..end.min(self.values.len())];
        self.has_missing.map_or_else(|| Input::new(v), |k| Input::known(v, k))
    }
}

/// The inputs `ops` recorded, by slot (`remember`: a frame column's NaN fact is
/// computed and remembered; otherwise only what is already known).
fn recorded<'o>(ops: &'o Operands, remember: bool) -> Vec<Recorded<'o>> {
    ops.reads()
        .map(|(slot, read)| Recorded {
            slot,
            values: Cow::Borrowed(read.view()),
            has_missing: if remember { Some(read.has_nonfinite()) } else { read.known_has_nonfinite() },
        })
        .collect()
}

fn inputs<'v>(recorded: &'v [Recorded], end: usize) -> Vec<Input<'v>> {
    recorded.iter().map(|r| r.input(end)).collect()
}

/// Operands that read, for each slot `ops` recorded, `gather(values)`.
fn over<'v>(
    ops: &Operands<'v>,
    df: &'v DataFrame,
    series: &'v [BoundNode],
    recorded: &'v [Recorded<'v>],
    gather: impl Fn(&'v [f64]) -> Cow<'v, [f64]>,
) -> Operands<'v> {
    let slots = recorded.iter().map(|r| r.slot + 1).max().unwrap_or(0);
    let mut views: Vec<Option<Cow<[f64]>>> = vec![None; slots];
    for r in recorded {
        views[r.slot] = Some(gather(&r.values));
    }
    Operands::over(df, series, ops.start(), views)
}

/// A result column's values as `f64`, taken over without a copy when they are.
pub(crate) fn into_f64(column: Column) -> Vec<f64> {
    match column {
        Column::F64(values) => values.into_vec(),
        other => other.to_f64_vec(),
    }
}

/// Run a full computation (or a state capture) of a recursive command over the
/// complete rows of its inputs' rows `[0, end)`: `run` is its arm, `finish` places the
/// result of a run over the complete rows back on those rows.
fn on_complete_rows<T>(
    df: &DataFrame,
    series: &[BoundNode],
    end: usize,
    run: impl Fn(&Operands) -> T,
    finish: impl FnOnce(T, &CompleteRows, usize) -> T,
) -> T {
    let ops = Operands::stopping(df, series, end);
    let first = run(&ops);
    if ops.failed() || (!ops.missing() && ops.clean()) {
        return first;
    }
    if ops.missing() {
        // A column read is known to hold a NaN: the in-place result is thrown away
        // unseen — complete the record without computing, then run on the complete rows.
        ops.probe();
        let probed = run(&ops);
        if ops.failed() {
            return probed;
        }
    }
    let recorded = recorded(&ops, true);
    let rows = CompleteRows::of(&inputs(&recorded, end));
    if rows == CompleteRows::All {
        // The inputs turned out complete: the in-place run stands — unless it stopped at
        // a column whose remembered fact claimed a NaN the values no longer hold. Then
        // it runs in place again, never stopping: a stale fact is never the answer.
        return if ops.missing() { run(&Operands::rows(df, series, 0, end)) } else { first };
    }
    let out = run(&over(&ops, df, series, &recorded, |v| rows.gather(v)));
    finish(out, &rows, ops.len())
}

/// Compute a recursive command over the complete rows of its inputs: `run` is its
/// dispatch arm.
pub(crate) fn compute(
    df: &DataFrame,
    series: &[BoundNode],
    run: impl Fn(&Operands) -> Result<Column>,
) -> Result<Column> {
    on_complete_rows(df, series, df.height(), run, |out, rows, n| {
        Ok(Column::f64(rows.scatter(n, into_f64(out?))))
    })
}

/// The carried state of a recursive command after a full computation over the
/// complete rows of its inputs' rows `[0, end)` — the state as of the last complete row
/// among them. `run` is its state-capture arm.
pub(crate) fn state(
    df: &DataFrame,
    series: &[BoundNode],
    end: usize,
    run: impl Fn(&Operands) -> Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    on_complete_rows(df, series, end, run, |out, _, _| out)
}

#[cfg(test)]
mod tests {
    use crate::exec::{execute, execute_resume_step, initial_state};
    use crate::parse;
    use volas_core::{Column, DataFrame};

    fn frame(open: Vec<f64>) -> DataFrame {
        let close: Vec<f64> = (0..open.len())
            .map(|i| 50.0 + (i as f64 * 0.7).sin())
            .collect();
        DataFrame::new(
            vec!["open".into(), "close".into()],
            vec![Column::f64(open), Column::f64(close)],
            None,
        )
        .unwrap()
    }

    /// The kernel runs once per compute, on the rows it is given: in place over a
    /// frame whose columns' NaN facts are unknown, and — once a read column is known
    /// to hold a NaN — straight over the complete rows, with no in-place run first.
    #[test]
    fn a_column_known_to_miss_rows_skips_the_in_place_run() {
        use std::cell::RefCell;
        let mut open: Vec<f64> = (0..12).map(|i| 40.0 + i as f64).collect();
        open[0] = f64::NAN;
        open[5] = f64::NAN;
        let df = frame(open);
        let runs = RefCell::new(Vec::new()); // the length of each kernel run's input
        let run = |ops: &crate::operands::Operands| {
            let x = ops.f64(0, "open")?;
            if !x.is_empty() {
                runs.borrow_mut().push(x.len());
            }
            Ok(Column::f64(x.to_vec()))
        };
        let first = super::compute(&df, &[], run).unwrap().to_f64_vec();
        assert_eq!(*runs.borrow(), vec![12, 10]); // unknown: in place, then gathered
        runs.borrow_mut().clear();
        let again = super::compute(&df, &[], run).unwrap().to_f64_vec();
        assert_eq!(*runs.borrow(), vec![10]); // known: gathered only
        assert_eq!(format!("{first:?}"), format!("{again:?}"));
        // a clean frame column, once known clean, is read in place with no scan
        let clean = frame((0..12).map(|i| 40.0 + i as f64).collect());
        assert!(!clean.column("open").unwrap().has_nonfinite());
        runs.borrow_mut().clear();
        super::compute(&clean, &[], run).unwrap();
        assert_eq!(*runs.borrow(), vec![12]);
    }

    #[test]
    fn a_result_column_is_taken_over_as_f64() {
        assert_eq!(super::into_f64(Column::f64(vec![1.5])), vec![1.5]);
        assert_eq!(super::into_f64(Column::i64(vec![2])), vec![2.0]);
    }

    /// A single-row resume over an explicit operand evaluates it like any other read —
    /// an operand carries no remembered fact, so the rows it read are checked — and
    /// agrees with a full compute; a missing operand row is NaN.
    #[test]
    fn a_scalar_resume_over_an_explicit_operand_checks_its_rows() {
        let n = 30;
        let open: Vec<f64> = (0..n).map(|i| 40.0 + (i as f64 * 0.3).cos()).collect();
        let node = parse("rsi:3@open").unwrap();
        let head = frame(open[..n - 1].to_vec());
        let st = initial_state(&head, &node, usize::MAX).unwrap();
        let full = frame(open.clone());
        let mut next = vec![0.0; st.len()];
        let value = execute_resume_step(&full, &node, &st, (n - 1, 0), &mut next).unwrap();
        let want = execute(&full, &node).unwrap().to_f64_vec()[n - 1];
        assert_eq!(value.to_bits(), want.to_bits());
        let mut gap = open;
        gap[n - 1] = f64::NAN;
        let value = execute_resume_step(&frame(gap), &node, &st, (n - 1, 0), &mut next).unwrap();
        assert!(value.is_nan());
        assert_eq!(next, st);
    }

    /// A resume carries the complete rows its kernel needs, so a dropped head and a
    /// missing run no longer force either scalar or Vec resume to restart the history.
    #[test]
    fn a_resume_carries_dropped_complete_history() {
        use crate::exec::execute_resume;
        let n = 40;
        let mut open: Vec<f64> = (0..n).map(|i| 40.0 + (i as f64 * 0.3).cos()).collect();
        for x in &mut open[20..30] {
            *x = f64::NAN;
        }
        let node = parse("rsi:3@open").unwrap();
        let st = initial_state(&frame(open[..20].to_vec()), &node, usize::MAX).unwrap();
        // the frame keeps rows 18.. (origin 18): two complete rows before row 30
        let kept = frame(open.clone()).slice_data(18, n);
        let mut next = vec![0.0; st.len()];
        assert!(execute_resume(&kept, &node, &st, 12, kept.height(), 0).is_some());
        assert!(execute_resume_step(&kept, &node, &st, (12, 0), &mut next).is_some());
        // the history starts at row 18: nothing is missing, the resume runs
        assert!(execute_resume(&kept, &node, &st, 12, kept.height(), 18).is_some());
        assert!(execute_resume_step(&kept, &node, &st, (12, 18), &mut next).is_some());
        // a missing new row needs no history: NaN, the state untouched
        assert!(execute_resume_step(&kept, &node, &st, (5, 0), &mut next).unwrap().is_nan());
        // the whole frame holds every complete row: the resume runs, through a row before
        // the last as well (the inputs read up to it, borrowed)
        let whole = frame(open);
        let (tail, _) = execute_resume(&whole, &node, &st, 20, 35, 0).unwrap();
        let want = execute(&whole, &node).unwrap().to_f64_vec();
        assert_eq!(tail.len(), 15);
        assert!(tail.to_f64_vec().iter().zip(&want[20..35]).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// An input the frame cannot supply is the command's own error — also when another
    /// input misses a row, where the record of what it reads would otherwise be cut
    /// short and the rerun over the complete rows read an empty series.
    #[test]
    fn an_input_that_cannot_be_read_is_the_error_never_a_rerun() {
        for known in [false, true] {
            // the close column's NaN fact unknown (the first run fails at the absent
            // column) or known (the run stops at close; the probe fails)
            let close = Column::f64(vec![1.0, f64::NAN, 3.0, 4.0]);
            if known {
                assert!(close.has_nonfinite());
            }
            let df = DataFrame::new(vec!["close".into()], vec![close], None).unwrap();
            for d in ["nvi", "pvi", "pvt", "obv", "efi:2", "ad", "adosc"] {
                let err = execute(&df, &parse(d).unwrap()).unwrap_err();
                assert!(err.to_string().contains("not found"), "{d}: {err}");
                assert!(initial_state(&df, &parse(d).unwrap(), usize::MAX).is_none(), "{d}");
            }
        }
    }
}
