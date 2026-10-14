//! What a command reads: its input series, resolved when its arm asks for them,
//! converted once and remembered.
//!
//! A command's arm reads each input through [`Operands`]. The first read of a slot
//! resolves it — a frame column (with its remembered all-finite fact) or an evaluated `@`
//! operand — converts it to `f64` once (a float64 frame column is borrowed, never
//! copied), and records it, so the record is exactly the columns the command read: the
//! missing-row rule (`crate::complete`) decides from it, and reruns the arm over the
//! complete rows of those same columns, without a second lookup, evaluation or
//! conversion. A boolean operand is read and recorded the same way (its truths as
//! `1` / `0`, a missing truth as NaN).
//!
//! `Operands` is the one owner of the kernels' shared precondition — every series a
//! command's arm receives has the same length: the frame's height, the rows of the
//! window it reads, or in a rerun the number of complete rows. Each input is checked
//! against that length as it is handed out, so a mismatch is an error, never an
//! out-of-bounds read in a kernel. A read that fails (an absent column, an operand that
//! does not evaluate, an input outside the `f64` domain) is remembered: the record then
//! lacks that input, so the rule never reruns the arm without it — the arm's own
//! result, that error, is the answer. While probing ([`Operands::probe`]) every input
//! is handed out empty — all of them, so they still share one length (0) — and a rerun
//! reading a slot the first run did not record is an internal error, never an input.
//!
//! It also owns what reading costs. A one-row continuation of a cached column reads a
//! trailing window of the frame instead of the whole of it ([`Operands::window`]): only
//! the window's rows of a non-float64 column are converted, and an `@` expression
//! operand is evaluated over the window alone — so a live refresh costs the window, not
//! the frame's height. The window's rows keep their absolute positions (its origin), so
//! a kernel continues over them exactly as over the frame.

use std::borrow::Cow;
use std::cell::{Cell, OnceCell};

use volas_core::{Column, DataFrame, Result, VolasError};

use crate::exec::{execute_node, operand, require_f64_input, require_numeric};
use crate::types::{BoundNode, Node};

/// How many input slots a command can read (every command reads at most five; an arm
/// reading past this is a bug, caught by every command's test).
const SLOTS: usize = 8;

/// One input a command read, converted to `f64` once.
pub(crate) struct Read<'a> {
    /// The values (borrowed from a float64 frame column).
    values: Cow<'a, [f64]>,
    /// The frame column read whole, which remembers its all-finite fact.
    column: Option<&'a Column>,
    /// Whether the values hold a missing input, when known without a scan: a window of a frame
    /// column known to hold none holds none.
    known_missing: Option<bool>,
}

impl Read<'_> {
    /// The values as `f64`.
    pub(crate) fn view(&self) -> &[f64] {
        &self.values
    }

    /// Whether the values hold a missing input — remembered for a frame column read whole.
    pub(crate) fn has_nonfinite(&self) -> bool {
        match self.column {
            Some(c) => c.has_nonfinite(),
            None => self.known_missing.unwrap_or_else(|| volas_compute::missing::any_missing(&self.values)),
        }
    }

    /// Whether the values hold a missing input, if known without a scan.
    pub(crate) fn known_has_nonfinite(&self) -> Option<bool> {
        self.column.map_or(self.known_missing, Column::known_has_nonfinite)
    }
}

/// The values of `col`'s rows `[start, end)` as `f64` — borrowed for a float64 column;
/// any other is converted over those rows only.
fn convert(col: &Column, start: usize, end: usize) -> Result<Cow<'_, [f64]>> {
    if let Some(v) = col.as_f64() {
        return Ok(Cow::Borrowed(&v[start..end]));
    }
    if (start, end) == (0, col.len()) {
        require_f64_input(col)?;
        return Ok(Cow::Owned(col.to_f64_vec()));
    }
    let rows = col.slice(start, end);
    require_f64_input(&rows)?;
    Ok(Cow::Owned(rows.to_f64_vec()))
}

/// The operands a command's arm reads (see the module docs).
pub(crate) struct Operands<'a> {
    df: &'a DataFrame,
    series: &'a [BoundNode],
    reads: [OnceCell<Read<'a>>; SLOTS],
    /// The first frame row read: `0`, or the start of a trailing window.
    start: usize,
    absolute_origin: Option<usize>,
    /// The row after the last frame row read: the frame's height, or an earlier end.
    end: usize,
    /// The window's rows as a frame, built for evaluating an `@` operand over them.
    window: OnceCell<DataFrame>,
    /// The length of every input handed to the arm: the rows read, or the number of
    /// complete rows a rerun reads.
    len: usize,
    /// Rows to read instead of the frame: each slot's complete rows (the rerun).
    over: bool,
    borrowed: Option<[Option<&'a [f64]>; SLOTS]>,
    /// Stop at a frame column known to hold a missing input (the arm's result would be thrown
    /// away; the record is what matters).
    stop_on_missing: bool,
    /// Hand every input over empty: complete the record without computing.
    probing: Cell<bool>,
    /// Every input read so far is a frame column known to hold no missing input.
    clean: Cell<bool>,
    /// An input read was a frame column known to hold a missing input.
    missing: Cell<bool>,
    /// A read failed: the record lacks that input.
    failed: Cell<bool>,
}

impl<'a> Operands<'a> {
    /// Place prepared carried rows at their original absolute positions.
    pub(crate) fn with_origin(mut self, origin: usize) -> Self { self.absolute_origin = Some(origin); self }

    /// Read the frame's columns and the `@` operands evaluated over it.
    pub(crate) fn new(df: &'a DataFrame, series: &'a [BoundNode]) -> Self {
        Operands::rows(df, series, 0, df.height())
    }

    /// Read only the frame's rows from `start` on (each keeping its absolute position).
    pub(crate) fn window(df: &'a DataFrame, series: &'a [BoundNode], start: usize) -> Self {
        Operands::rows(df, series, start, df.height())
    }

    /// Read only the frame's rows `[start, end)` (each keeping its absolute position):
    /// the columns' rows there, and the `@` operands evaluated over them. The arm sees
    /// row `start` as its row 0 (see the module docs).
    pub(crate) fn rows(df: &'a DataFrame, series: &'a [BoundNode], start: usize, end: usize) -> Self {
        let end = end.min(df.height());
        let start = start.min(end);
        Operands {
            df,
            series,
            reads: Default::default(),
            start,
            absolute_origin: None,
            end,
            window: OnceCell::new(),
            len: end - start,
            over: false,
            borrowed: None,
            stop_on_missing: false,
            probing: Cell::new(false),
            clean: Cell::new(true),
            missing: Cell::new(false),
            failed: Cell::new(false),
        }
    }

    /// As [`rows`](Self::rows) over the rows `[0, end)`, stopping the arm at the first
    /// frame column known to hold a missing input (see [`missing`](Self::missing)).
    pub(crate) fn stopping(df: &'a DataFrame, series: &'a [BoundNode], end: usize) -> Self {
        Operands {
            stop_on_missing: true,
            ..Operands::rows(df, series, 0, end)
        }
    }

    /// Read `views[slot]` for each slot instead (the complete rows of the inputs: each
    /// as long as the first) of the frame's rows from `start` on.
    pub(crate) fn over(
        df: &'a DataFrame,
        series: &'a [BoundNode],
        start: usize,
        views: Vec<Option<Cow<'a, [f64]>>>,
    ) -> Self {
        let mut ops = Operands { len: views.iter().flatten().next().map_or(0, |v| v.len()),
            over: true, ..Operands::window(df, series, start) };
        for (slot, values) in views.into_iter().enumerate() {
            if let Some(values) = values {
                ops.reads[slot] = OnceCell::from(Read { values, column: None, known_missing: None });
            }
        }
        ops
    }

    /// Prepared input slots for a scalar plan step, borrowed from reusable scratch.
    pub(crate) fn borrowed(df: &'a DataFrame, series: &'a [BoundNode], start: usize, views: [Option<&'a [f64]>; SLOTS]) -> Self {
        Self { len: views.iter().flatten().next().map_or(0, |v| v.len()), borrowed: Some(views), ..Self::window(df, series, start) }
    }

    /// Whether this reads every row of the frame.
    fn whole(&self) -> bool {
        (self.start, self.end) == (0, self.df.height())
    }

    /// The length of every input handed to the arm.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The first frame row this reads (`0` unless it reads a trailing window).
    pub(crate) fn start(&self) -> usize {
        self.start
    }

    /// The absolute position of the row the arm sees as row 0 (`DataFrame::origin`,
    /// plus a window's start): where a sliding-window kernel re-anchors its running sums.
    pub(crate) fn origin(&self) -> usize {
        self.absolute_origin.unwrap_or(self.df.origin() + self.start)
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

    /// The frame an `@` operand is evaluated over: the frame, or its window's rows.
    fn operand_frame(&self) -> &DataFrame {
        if self.whole() {
            return self.df;
        }
        self.window.get_or_init(|| self.df.slice_data(self.start, self.end))
    }

    /// The input at `slot`, resolved on its first read by `resolve`; a failed read is
    /// remembered.
    fn read(&self, slot: usize, resolve: impl FnOnce() -> Result<Read<'a>>) -> Result<&Read<'a>> {
        let cell = &self.reads[slot];
        if let Some(read) = cell.get() {
            self.fit(slot, read.values.len(), self.len)?;
            return Ok(read);
        }
        if self.over {
            return Err(VolasError::Value(format!("internal error: series argument #{slot} was not read before the rerun")));
        }
        let read = resolve().and_then(|r| self.fit(slot, r.values.len(), self.len).map(|()| r));
        let read = read.inspect_err(|_| self.failed.set(true))?;
        Ok(cell.get_or_init(|| read))
    }

    /// A numeric input: the operand filling `slot`, else the frame column `default`.
    fn resolve(&self, slot: usize, default: &str) -> Result<Read<'a>> {
        match operand(self.series, slot) {
            Some(Node::Data(name)) => self.column_read(slot, name),
            Some(node) => {
                let col = execute_node(self.operand_frame(), node)?;
                require_f64_input(&col)?;
                Ok(Read { values: Cow::Owned(col.to_f64_vec()), column: None, known_missing: None })
            }
            None => self.column_read(slot, default),
        }
    }

    /// A bound data reference shares the default-column read path: borrow float64
    /// rows and retain the column's missing-value fact without evaluating a subtree.
    fn column_read(&self, slot: usize, name: &str) -> Result<Read<'a>> {
        let col = self.df.column(name)?;
        self.fit(slot, col.len(), self.df.height())?;
        let values = convert(col, self.start, self.end)?;
        Ok(match self.whole() {
            true => Read { values, column: Some(col), known_missing: None },
            false => Read { values, column: None, known_missing: col.known_has_nonfinite().filter(|has| !has) },
        })
    }

    /// Hand `read` to the arm: empty while probing; note what is known of its NaNs.
    fn hand_out<'r>(&self, read: &'r Read<'a>) -> Result<Cow<'r, [f64]>> {
        if self.probing.get() {
            return Ok(Cow::Borrowed(&[]));
        }
        match read.known_has_nonfinite() {
            Some(false) => {}
            Some(true) if self.stop_on_missing => {
                self.missing.set(true);
                return Err(VolasError::Value("an input holds a missing row".into()));
            }
            _ => self.clean.set(false),
        }
        Ok(Cow::Borrowed(read.view()))
    }

    /// Input `slot` as `f64` (`default` names its column when no operand fills it).
    pub(crate) fn f64(&self, slot: usize, default: &str) -> Result<Cow<'_, [f64]>> {
        if let Some(views) = &self.borrowed {
            return views[slot].map(Cow::Borrowed).ok_or_else(|| VolasError::Value("unprepared input slot".into()));
        }
        let read = self.read(slot, || self.resolve(slot, default))?;
        self.hand_out(read)
    }

    /// A required operand at `slot` as `f64` (`bind` guarantees one is given:
    /// `beta` / `correl` / `mavp`'s second series).
    pub(crate) fn f64_required(&self, slot: usize) -> Result<Cow<'_, [f64]>> {
        self.f64(slot, "")
    }

    /// A boolean operand at `slot`, row by row: a bool reads its value, a number
    /// `x != 0`, and a missing cell is a missing truth (contract #27). Recorded like any
    /// other input.
    pub(crate) fn bool(&self, slot: usize) -> Result<Vec<Option<bool>>> {
        let values = {
                let read = self.read(slot, || {
                    let node = operand(self.series, slot)
                        .ok_or_else(|| VolasError::Value("a boolean series argument is required".into()))?;
                    let col = execute_node(self.operand_frame(), node)?;
                    require_numeric(&col)?;
                    let truths = (0..col.len()).map(|i| match col.truth_at(i) {
                        Some(t) => f64::from(u8::from(t)),
                        None => f64::NAN,
                    });
                    Ok(Read { values: Cow::Owned(truths.collect()), column: None, known_missing: None })
                })?;
                self.hand_out(read)?
        };
        Ok(values.iter().map(|&v| (!volas_compute::missing::is_missing(v)).then_some(v != 0.0)).collect())
    }

    /// Whether every input read is a frame column known to hold no missing input.
    pub(crate) fn clean(&self) -> bool {
        self.clean.get()
    }

    /// Whether the arm stopped at a frame column known to hold a missing input.
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
mod tests;
