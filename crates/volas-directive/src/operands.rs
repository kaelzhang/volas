//! What a command reads: its input series, resolved when its arm asks for them and
//! remembered.
//!
//! A command's arm reads each input through [`Operands`]. The first read of a slot
//! resolves it — a frame column (borrowed, with its remembered NaN fact) or an
//! evaluated `@` operand (kept) — and records it, so the record is exactly the columns
//! the command read: the missing-row rule (`crate::complete`) decides from it, and
//! reruns the arm over the complete rows of those same columns, without a second
//! lookup or a second evaluation.

use std::borrow::Cow;
use std::cell::{Cell, OnceCell};

use volas_core::{Column, DataFrame, Result, VolasError};

use crate::exec::{execute, operand, require_numeric, series_bool};
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
}

impl<'a> Operands<'a> {
    /// Read the frame's columns and the `@` operands evaluated over it.
    pub(crate) fn new(df: &'a DataFrame, series: &'a [Ast]) -> Self {
        Operands {
            df,
            series,
            reads: Default::default(),
            over: None,
            stop_on_missing: false,
            probing: Cell::new(false),
            clean: Cell::new(true),
            missing: Cell::new(false),
        }
    }

    /// As [`new`](Self::new), stopping the arm at the first frame column known to hold a
    /// NaN (see [`missing`](Self::missing)).
    pub(crate) fn stopping(df: &'a DataFrame, series: &'a [Ast]) -> Self {
        Operands {
            stop_on_missing: true,
            ..Operands::new(df, series)
        }
    }

    /// Read `views[slot]` for each slot instead (the complete rows of the inputs).
    pub(crate) fn over(
        df: &'a DataFrame,
        series: &'a [Ast],
        views: Vec<Option<Cow<'a, [f64]>>>,
    ) -> Self {
        Operands {
            over: Some(views),
            ..Operands::new(df, series)
        }
    }

    /// The absolute position of the frame's row 0 (`DataFrame::origin`): where a
    /// sliding-window kernel re-anchors its running sums.
    pub(crate) fn origin(&self) -> usize {
        self.df.origin()
    }

    /// The input at `slot`, resolved on its first read (`default` names its column
    /// when no operand fills the slot).
    fn read(&self, slot: usize, default: &str) -> Result<&Read<'a>> {
        let cell = &self.reads[slot];
        if let Some(read) = cell.get() {
            return Ok(read);
        }
        let read = match operand(self.series, slot) {
            Some(node) => Read::Operand(execute(self.df, node)?),
            None => Read::Frame(self.df.column(default)?),
        };
        require_numeric(read.column())?;
        Ok(cell.get_or_init(|| read))
    }

    /// Input `slot` as `f64` (`default` names its column when no operand fills it).
    pub(crate) fn f64(&self, slot: usize, default: &str) -> Result<Cow<'_, [f64]>> {
        if let Some(views) = &self.over {
            return Ok(Cow::Borrowed(views.get(slot).and_then(|v| v.as_deref()).unwrap_or(&[])));
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
        Ok(read.view())
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
    pub(crate) fn bool(&self, slot: usize) -> Result<Vec<bool>> {
        self.clean.set(false);
        series_bool(self.df, self.series, slot)
    }

    /// Whether every input read is a frame column known to hold no NaN.
    pub(crate) fn clean(&self) -> bool {
        self.clean.get()
    }

    /// Whether the arm stopped at a frame column known to hold a NaN.
    pub(crate) fn missing(&self) -> bool {
        self.missing.get()
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
