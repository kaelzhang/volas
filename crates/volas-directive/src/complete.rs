//! A recursive command runs on the complete rows of its inputs.
//!
//! The rule — what a missing row is, and which rows a full computation, its
//! carried-state capture and a resume run on — is owned by [`volas_compute::missing`].
//! This module applies it to a bound recursive command. The command's arm runs in
//! place first, reading its inputs through [`Operands`], which records exactly what it
//! read: when every column read is known complete (the column's own remembered fact,
//! `Column::has_nan`) that result stands with no scan at all; otherwise the record says
//! which rows are complete, and the arm runs again over just those rows of the same
//! inputs, its outputs scattered back. The arms and kernels are unchanged. A read that
//! failed leaves the record without that input, so the arm's own result — the error —
//! stands and nothing reruns (see [`Operands::failed`]).

use std::borrow::Cow;

use volas_compute::missing::{CompleteRows, Input, ResumeRows};
use volas_core::{Column, DataFrame, Result};

use crate::operands::Operands;
use crate::spec::CommandSpec;
use crate::types::{ArgValue, Ast};

/// A bound command: canonical name / sub, typed arguments, `@` series, its line's spec.
#[derive(Clone, Copy)]
pub(crate) struct BoundCommand<'a> {
    pub(crate) name: &'a str,
    pub(crate) sub: Option<&'a str>,
    pub(crate) args: &'a [ArgValue],
    pub(crate) series: &'a [Ast],
    pub(crate) spec: &'a CommandSpec,
}

impl BoundCommand<'_> {
    /// How many rows before its first new row a resume of the command reads, at most.
    pub(crate) fn reach(&self) -> usize {
        self.spec.lookback_of(self.args) + 1
    }
}

/// One input an arm read, as the rule sees it.
struct Recorded<'v> {
    slot: usize,
    values: Cow<'v, [f64]>,
    /// Whether the values hold a NaN, if known.
    has_nan: Option<bool>,
}

impl Recorded<'_> {
    /// The rows `[..end)`, with the NaN fact (which a complete whole implies for them).
    fn input(&self, end: usize) -> Input<'_> {
        let v = &self.values[..end.min(self.values.len())];
        self.has_nan.map_or_else(|| Input::new(v), |k| Input::known(v, k))
    }
}

/// The inputs `ops` recorded, by slot (`remember`: a frame column's NaN fact is
/// computed and remembered; otherwise only what is already known).
fn recorded<'o>(ops: &'o Operands, remember: bool) -> Vec<Recorded<'o>> {
    ops.reads()
        .map(|(slot, read)| Recorded {
            slot,
            values: Cow::Borrowed(read.view()),
            has_nan: if remember { Some(read.has_nan()) } else { read.known_has_nan() },
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
    series: &'v [Ast],
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
    series: &[Ast],
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
    let rows = CompleteRows::of(&inputs(&recorded, usize::MAX));
    if rows == CompleteRows::All {
        // The inputs turned out complete: the in-place run stands — unless it stopped at
        // a column whose remembered fact claimed a NaN the values no longer hold. Then
        // it runs in place again, never stopping: a stale fact is never the answer.
        return if ops.missing() { run(&Operands::new(df, series)) } else { first };
    }
    let out = run(&over(&ops, df, series, &recorded, |v| rows.gather(v)));
    finish(out, &rows, ops.len())
}

/// Compute a recursive command over the complete rows of its inputs: `run` is its
/// dispatch arm.
pub(crate) fn compute(
    df: &DataFrame,
    series: &[Ast],
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
    series: &[Ast],
    end: usize,
    run: impl Fn(&Operands) -> Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    on_complete_rows(df, series, end, run, |out, _, _| out)
}

/// Whether the resume `ops` just ran read every input's rows `[from - reach, end)`
/// complete (see [`ResumeRows::in_place`]) — free for a column known complete, and with
/// no allocation for a frame column (the live single-row refresh runs this per row).
fn read_in_place(ops: &Operands, cmd: &BoundCommand, from: usize, end: usize) -> bool {
    let mut reach = None; // only needed for an input that must be scanned
    ops.reads().all(|(slot, read)| {
        let r = Recorded { slot, values: Cow::Borrowed(read.view()), has_nan: read.known_has_nan() };
        r.has_nan == Some(false)
            || ResumeRows::in_place(r.input(end), from, *reach.get_or_insert_with(|| cmd.reach()))
    })
}

/// Whether a resume over gathered complete rows lacks history it reads: fewer complete
/// rows before its first new row than it may read, while the column's history (from the
/// absolute row `since`) starts before the frame's row 0 — the rows it needs were dropped
/// with the frame's head. It then declines; the caller replays from the column's
/// checkpoint instead. A frame holding the whole history lacks nothing: fewer rows are
/// all the series has.
fn lacks_history(df: &DataFrame, cmd: &BoundCommand, rows: &ResumeRows, since: usize) -> bool {
    rows.from < cmd.reach() && df.origin() > since
}

/// Resume a recursive command over the new rows `[from, end)` from `prev_state`, on the
/// complete rows only: a missing new row is NaN and leaves the state where the last
/// complete row put it. `run(operands, from)` is its resume arm, over the rows the
/// operands read: `[start, end)` (a trailing window, see [`Operands::rows`]) — or, when
/// that window holds a missing row, `[0, end)` (the complete rows the resume reads before
/// its new ones may lie before the window). The column's history starts at the absolute
/// row `since`.
///
/// The arm runs in place first: when every input it read is known complete, or the
/// rows it read turn out complete, that is the answer. Otherwise it runs again over the
/// complete rows of the same inputs — declining when the frame dropped complete rows of
/// the history it reads ([`lacks_history`]).
pub(crate) fn resume(
    df: &DataFrame,
    cmd: &BoundCommand,
    start: usize,
    prev_state: &[f64],
    (from, end, since): (usize, usize, usize),
    run: impl Fn(&Operands, usize) -> Option<(Column, Vec<f64>)>,
) -> Option<(Column, Vec<f64>)> {
    let ops = Operands::rows(df, cmd.series, start.min(from), end);
    let (start, n) = (ops.start(), ops.len());
    let local = from - start;
    let out = run(&ops, local);
    if ops.failed() || ops.clean() || read_in_place(&ops, cmd, local, n) {
        return out;
    }
    if start > 0 {
        return resume(df, cmd, 0, prev_state, (from, end, since), run);
    }
    let recorded = recorded(&ops, false);
    let refs: Vec<&[f64]> = recorded.iter().map(|r| &r.values[..n]).collect();
    let rows = ResumeRows::of(&refs, from, cmd.reach());
    if !rows.has_new() {
        return Some((Column::f64(vec![f64::NAN; n - from]), prev_state.to_vec()));
    }
    if lacks_history(df, cmd, &rows, since) {
        return None;
    }
    let gathered = over(&ops, df, cmd.series, &recorded, |v| {
        Cow::Owned(rows.rows.iter().map(|&i| v[i]).collect())
    });
    let (col, st) = run(&gathered, rows.from)?;
    Some((Column::f64(rows.scatter(n, from, into_f64(col))), st))
}

/// Resume a recursive command at the single row `row` (the scalar twin of [`resume`]):
/// `missing` when that row is missing, else `run(operands, row)` on the complete rows,
/// over the rows from `start` on (as there, the whole frame when that window holds a
/// missing row; declining like it when the frame dropped history it reads, `since` as
/// there). The arm runs in place first and its answer stands when the rows it read are
/// complete; an `@` operand is evaluated once.
pub(crate) fn resume_one<T>(
    df: &DataFrame,
    cmd: &BoundCommand,
    start: usize,
    (row, since): (usize, usize),
    missing: impl FnOnce() -> Option<T>,
    mut run: impl FnMut(&Operands, usize) -> Option<T>,
) -> Option<T> {
    let ops = Operands::window(df, cmd.series, start.min(row));
    let start = ops.start();
    let out = run(&ops, row - start);
    // The rows up to and including `row` (an out-of-range `row` resumes in place, where
    // the arm declines it).
    let end = (row + 1).min(df.height()) - start;
    if ops.failed() || ops.clean() || read_in_place(&ops, cmd, row - start, end) {
        return out;
    }
    if start > 0 {
        return resume_one(df, cmd, 0, (row, since), missing, run);
    }
    let recorded = recorded(&ops, false);
    let upto: Vec<&[f64]> = recorded.iter().map(|r| &r.values[..end]).collect();
    let rows = ResumeRows::of(&upto, row, cmd.reach());
    if !rows.has_new() {
        return missing();
    }
    if lacks_history(df, cmd, &rows, since) {
        return None;
    }
    let gathered = over(&ops, df, cmd.series, &recorded, |v| {
        Cow::Owned(rows.rows.iter().map(|&i| v[i]).collect())
    });
    run(&gathered, rows.from)
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
        assert!(!clean.column("open").unwrap().has_nan());
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

    /// A resume over gathered complete rows reads `reach` of them before its first new
    /// row: on a frame that dropped its history's head (the history starts before row 0)
    /// with fewer of them held — a run of missing rows reaching the frame's start — it
    /// declines, scalar and `Vec` alike; on a frame holding the whole history it runs.
    #[test]
    fn a_resume_lacking_dropped_history_declines() {
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
        assert!(execute_resume(&kept, &node, &st, 12, kept.height(), 0).is_none());
        assert!(execute_resume_step(&kept, &node, &st, (12, 0), &mut next).is_none());
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
                assert!(close.has_nan());
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
