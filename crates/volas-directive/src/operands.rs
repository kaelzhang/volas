//! What a command reads: its input series, resolved when its arm asks for them and
//! remembered.
//!
//! A command's arm reads each input through [`Operands`]. The first read of a slot
//! resolves it — a frame column (borrowed, with its remembered NaN fact) or an
//! evaluated `@` operand (kept) — and records it, so the record is exactly the columns
//! the command read: the missing-row rule (`crate::complete`) decides from it, and
//! reruns the arm over the complete rows of those same columns, without a second
//! lookup or a second evaluation.
//!
//! `Operands` is the one owner of the kernels' shared precondition — every series a
//! command's arm receives has the same length: the frame's height, or in a rerun the
//! number of complete rows. Each input is checked against that length as it is handed
//! out, so a mismatch is an error, never an out-of-bounds read in a kernel. A read that
//! fails (an absent column, an operand that does not evaluate, an input outside the
//! `f64` domain) is remembered: the record then lacks that input, so the rule never
//! reruns the arm without it — the arm's own result, that error, is the answer.

use std::borrow::Cow;
use std::cell::{Cell, OnceCell};

use volas_core::{Column, DataFrame, Result, VolasError};

use crate::exec::{execute, operand, require_f64_input, series_truths};
use crate::types::Ast;

/// How many input slots a command can read (every command reads at most five; an arm
/// reading past this is a bug, caught by every command's test).
const SLOTS: usize = 8;

/// One input a command read.
pub(crate) enum Read<'a> {
    /// A frame column, borrowed.
    Frame(&'a Column),
    /// An evaluated `@` operand.
    Operand(Column),
}

impl Read<'_> {
    /// The values as `f64` (borrowed for an `f64` column).
    pub(crate) fn view(&self) -> Cow<'_, [f64]> {
        let column = self.column();
        column
            .as_f64()
            .map_or_else(|| Cow::Owned(column.to_f64_vec()), Cow::Borrowed)
    }

    fn column(&self) -> &Column {
        match self {
            Read::Frame(c) => c,
            Read::Operand(c) => c,
        }
    }

    /// Whether the values hold a NaN — remembered for a frame column.
    pub(crate) fn has_nan(&self) -> bool {
        self.column().has_nan()
    }

    /// Whether the values hold a NaN, if known without a scan.
    pub(crate) fn known_has_nan(&self) -> Option<bool> {
        match self {
            Read::Frame(c) => c.known_has_nan(),
            Read::Operand(_) => None,
        }
    }
}

/// The operands a command's arm reads (see the module docs).
pub(crate) struct Operands<'a> {
    df: &'a DataFrame,
    series: &'a [Ast],
    reads: [OnceCell<Read<'a>>; SLOTS],
    /// The length of every input handed to the arm: the frame's height (or the rows
    /// before [`upto`](Self::upto)'s end), or the number of complete rows a rerun reads.
    len: usize,
    /// Rows to read instead of the frame: each slot's complete rows (the rerun).
    over: Option<Vec<Option<Cow<'a, [f64]>>>>,
    /// Stop at a frame column known to hold a NaN (the arm's result would be thrown
    /// away; the record is what matters).
    stop_on_missing: bool,
    /// Hand every input over empty: complete the record without computing.
    probing: Cell<bool>,
    /// Every input read so far is a frame column known to hold no NaN.
    clean: Cell<bool>,
    /// An input read was a frame column known to hold a NaN.
    missing: Cell<bool>,
    /// A read failed: the record lacks that input.
    failed: Cell<bool>,
}

impl<'a> Operands<'a> {
    /// Read the frame's columns and the `@` operands evaluated over it.
    pub(crate) fn new(df: &'a DataFrame, series: &'a [Ast]) -> Self {
        Operands {
            df,
            series,
            reads: Default::default(),
            len: df.height(),
            over: None,
            stop_on_missing: false,
            probing: Cell::new(false),
            clean: Cell::new(true),
            missing: Cell::new(false),
            failed: Cell::new(false),
        }
    }

    /// As [`new`](Self::new), handing the arm each input's rows `[0, end)` only (`end`
    /// at most the frame's height) — borrowed, so a resume of rows before the last reads
    /// no copy of the frame.
    pub(crate) fn upto(df: &'a DataFrame, series: &'a [Ast], end: usize) -> Self {
        Operands::new(df, series).upto_rows(end)
    }

    /// These operands reading the rows `[0, end)` only (at most the frame's height).
    pub(crate) fn upto_rows(self, end: usize) -> Self {
        Operands { len: end.min(self.df.height()), ..self }
    }

    /// The length of every input handed to the arm.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// As [`new`](Self::new), stopping the arm at the first frame column known to hold a
    /// NaN (see [`missing`](Self::missing)).
    pub(crate) fn stopping(df: &'a DataFrame, series: &'a [Ast]) -> Self {
        Operands {
            stop_on_missing: true,
            ..Operands::new(df, series)
        }
    }

    /// Read `views[slot]` for each slot instead (the complete rows of the inputs: each
    /// as long as the first).
    pub(crate) fn over(
        df: &'a DataFrame,
        series: &'a [Ast],
        views: Vec<Option<Cow<'a, [f64]>>>,
    ) -> Self {
        Operands {
            len: views.iter().flatten().next().map_or(0, |v| v.len()),
            over: Some(views),
            ..Operands::new(df, series)
        }
    }

    /// The absolute position of the frame's row 0 (`DataFrame::origin`): where a
    /// sliding-window kernel re-anchors its running sums.
    pub(crate) fn origin(&self) -> usize {
        self.df.origin()
    }

    /// `Ok` when an input of `len` rows fits the others, `want` rows (see the module docs).
    fn fit(&self, slot: usize, len: usize, want: usize) -> Result<()> {
        if len == want {
            return Ok(());
        }
        Err(VolasError::Shape(format!(
            "series argument #{slot} has {len} rows where the command's other inputs have {want}"
        )))
    }

    /// The input at `slot`, resolved on its first read (`default` names its column
    /// when no operand fills the slot); a failed read is remembered.
    fn read(&self, slot: usize, default: &str) -> Result<&Read<'a>> {
        let cell = &self.reads[slot];
        if let Some(read) = cell.get() {
            return Ok(read);
        }
        let read = self.resolve(slot, default).inspect_err(|_| self.failed.set(true))?;
        Ok(cell.get_or_init(|| read))
    }

    fn resolve(&self, slot: usize, default: &str) -> Result<Read<'a>> {
        let read = match operand(self.series, slot) {
            Some(node) => Read::Operand(execute(self.df, node)?),
            None => Read::Frame(self.df.column(default)?),
        };
        require_f64_input(read.column())?;
        // an input spans the frame; the arm reads its first `len` rows
        self.fit(slot, read.column().len(), self.df.height())?;
        Ok(read)
    }

    /// Input `slot` as `f64` (`default` names its column when no operand fills it).
    pub(crate) fn f64(&self, slot: usize, default: &str) -> Result<Cow<'_, [f64]>> {
        if let Some(views) = &self.over {
            // A rerun reads the inputs the first run recorded, and only those.
            let view = views.get(slot).and_then(|v| v.as_deref()).ok_or_else(|| {
                VolasError::Value(format!("series argument #{slot} was not read before the rerun"))
            })?;
            self.fit(slot, view.len(), self.len)?;
            return Ok(Cow::Borrowed(view));
        }
        let read = self.read(slot, default)?;
        if self.probing.get() {
            return Ok(Cow::Borrowed(&[]));
        }
        match read.known_has_nan() {
            Some(false) => {}
            Some(true) if self.stop_on_missing => {
                self.missing.set(true);
                return Err(VolasError::Value("an input holds a missing row".into()));
            }
            _ => self.clean.set(false),
        }
        Ok(match read.view() {
            Cow::Borrowed(v) => Cow::Borrowed(&v[..self.len]),
            Cow::Owned(mut v) => {
                v.truncate(self.len);
                Cow::Owned(v)
            }
        })
    }

    /// A required operand at `slot` as `f64`: an absent or empty operand is an error
    /// rather than a column default (e.g. `beta` / `correl`'s second series).
    pub(crate) fn f64_required(&self, slot: usize) -> Result<Cow<'_, [f64]>> {
        if operand(self.series, slot).is_none() {
            return Err(VolasError::Value(format!(
                "series argument #{slot} is required"
            )));
        }
        self.f64(slot, "")
    }

    /// A boolean operand at `slot`.
    pub(crate) fn bool(&self, slot: usize) -> Result<Vec<Option<bool>>> {
        self.clean.set(false);
        let mut truths = series_truths(self.df, self.series, slot)?;
        if self.over.is_none() && truths.len() == self.df.height() {
            truths.truncate(self.len);
        }
        self.fit(slot, truths.len(), self.len)?;
        Ok(truths)
    }

    /// Whether every input read is a frame column known to hold no NaN.
    pub(crate) fn clean(&self) -> bool {
        self.clean.get()
    }

    /// Whether the arm stopped at a frame column known to hold a NaN.
    pub(crate) fn missing(&self) -> bool {
        self.missing.get()
    }

    /// Whether a read failed, so the record lacks an input the arm reads: the arm's own
    /// result (that error) is the answer, never a rerun over the complete rows.
    pub(crate) fn failed(&self) -> bool {
        self.failed.get()
    }

    /// Hand every further read over empty: rerunning the arm then completes the
    /// record of what it reads, at no computing cost.
    pub(crate) fn probe(&self) {
        self.probing.set(true);
    }

    /// The inputs read, by slot.
    pub(crate) fn reads(&self) -> impl Iterator<Item = (usize, &Read<'a>)> {
        self.reads
            .iter()
            .enumerate()
            .filter_map(|(slot, cell)| cell.get().map(|r| (slot, r)))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use volas_core::{Column, DataFrame, VolasError};

    use super::Operands;

    fn frame() -> DataFrame {
        DataFrame::new(vec!["close".into()], vec![Column::f64(vec![1.0, 2.0, 3.0])], None).unwrap()
    }

    /// Every input a rerun hands out is as long as the others; a shorter one is a shape
    /// error, never a slice a kernel reads past.
    #[test]
    fn a_rerun_input_of_another_length_is_an_error() {
        let df = frame();
        let (a, b) = ([1.0, 2.0, 3.0], [1.0, 2.0]);
        let views = vec![Some(Cow::Borrowed(&a[..])), Some(Cow::Borrowed(&b[..]))];
        let ops = Operands::over(&df, &[], views);
        assert_eq!(ops.f64(0, "close").unwrap().len(), 3);
        let err = ops.f64(1, "close").unwrap_err();
        assert!(matches!(err, VolasError::Shape(ref m) if m.contains("#1 has 2 rows") && m.ends_with("have 3")));
    }

    /// A rerun reads only what the first run recorded: an unrecorded slot is an error,
    /// not an empty series.
    #[test]
    fn a_rerun_reads_only_recorded_inputs() {
        let df = frame();
        let ops = Operands::over(&df, &[], vec![None]);
        assert!(matches!(ops.f64(0, "close"), Err(VolasError::Value(m)) if m.contains("not read before")));
        assert!(Operands::over(&df, &[], vec![]).f64(3, "close").is_err());
    }

    /// Up to an end row, every input is handed over as its leading rows — a frame column
    /// borrowed, an operand and a boolean read cut to them — and an input of another
    /// length is still an error.
    #[test]
    fn up_to_an_end_row_the_inputs_are_their_leading_rows() {
        let df = frame();
        let series = [crate::parse("close + 1").unwrap().as_ref().clone(), crate::parse("close > 1").unwrap().as_ref().clone()];
        let ops = Operands::upto(&df, &series, 2);
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops.f64(2, "close").unwrap(), Cow::Borrowed(v) if v == [1.0, 2.0]));
        assert_eq!(ops.f64(0, "close").unwrap().as_ref(), &[2.0, 3.0]);
        assert_eq!(ops.bool(1).unwrap(), vec![Some(false), Some(true)]);
        assert_eq!(Operands::upto(&df, &[], 9).len(), 3);
        // an integer column is converted, then cut
        let ints = DataFrame::new(vec!["v".into()], vec![Column::i64(vec![4, 5, 6])], None).unwrap();
        let ops = Operands::upto(&ints, &[], 1);
        assert!(matches!(ops.f64(0, "v").unwrap(), Cow::Owned(v) if v == [4.0]));
    }

    /// A read that fails is remembered — the record lacks that input — and a later
    /// successful read does not clear it.
    #[test]
    fn a_failed_read_is_remembered() {
        let df = frame();
        let ops = Operands::new(&df, &[]);
        assert!(!ops.failed());
        assert!(matches!(ops.f64(0, "volume"), Err(VolasError::ColumnNotFound(_))));
        assert!(ops.failed());
        assert!(ops.f64(1, "close").is_ok());
        assert!(ops.failed());
    }
}
