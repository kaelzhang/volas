//! A recursive command runs on the complete rows of its inputs.
//!
//! The rule — what a missing row is, and which rows a full computation, its
//! carried-state capture and a resume run on — is owned by [`volas_compute::missing`].
//! This module applies it to a bound recursive command: it resolves the command's input
//! series once against the frame, asks the rule for the rows, and runs the command's
//! ordinary dispatch / state / resume arm on a frame of just those rows, scattering the
//! outputs back. The arms and kernels are unchanged. When no row is missing the arm runs
//! on the caller's frame as it is. Whether a frame column holds a missing value is the
//! column's own remembered fact (`Column::has_nan`), so a frame known to be complete is
//! never scanned here.

use std::borrow::Cow;
use std::cell::Cell;

use volas_compute::missing::{CompleteRows, Input, ResumeRows};
use volas_core::{Column, DataFrame, Result};

use crate::exec::{execute, require_numeric, series_f64};
use crate::lookback::own_lookback;
use crate::spec::command_spec;
use crate::types::{ArgValue, Ast, Node};

/// A recursive command's input series, resolved once against a frame (a frame column
/// is borrowed, not copied).
struct Inputs<'a> {
    /// The column each input is read from in an inputs frame: the command's default
    /// series name, so its arm reads it with no operand, or `#<slot>` for a slot without
    /// a distinct default.
    names: Vec<Cow<'static, str>>,
    views: Vec<Cow<'a, [f64]>>,
    /// Each input column's NaN fact (`Column::has_nan`).
    has_nan: Vec<bool>,
}

/// The input slots of a command: each slot's default series (`None` for a slot only an
/// operand fills) and whether an explicit operand fills it. Stops at a required operand
/// left out — the arm reports it.
fn slots<'s>(
    name: &str,
    sub: Option<&str>,
    series: &'s [Ast],
) -> impl Iterator<Item = (usize, Option<&'static str>, bool)> + 's {
    let defaults = command_spec(name, sub).map_or(&[][..], |s| s.series);
    (0..defaults.len().max(series.len()))
        .map(move |slot| {
            let operand = series
                .get(slot)
                .is_some_and(|n| !matches!(n, Node::Name(s) if s.is_empty()));
            (slot, defaults.get(slot).copied(), operand)
        })
        .take_while(|&(_, default, operand)| operand || default.is_some())
}

/// One input slot read as `f64` (a frame column borrowed), with what `know` says about
/// its column's NaN fact.
fn input<'a>(
    df: &'a DataFrame,
    series: &[Ast],
    slot: usize,
    default: Option<&str>,
    know: impl Fn(&Column) -> Option<bool>,
) -> Result<(Cow<'a, [f64]>, Option<bool>)> {
    let operand = series
        .get(slot)
        .filter(|n| !matches!(n, Node::Name(s) if s.is_empty()));
    match operand {
        Some(node) => {
            let column = execute(df, node)?;
            require_numeric(&column)?;
            let known = know(&column);
            Ok((Cow::Owned(column.to_f64_vec()), known))
        }
        None => {
            let column = df.column(default.unwrap_or_default())?;
            require_numeric(column)?;
            let view = column
                .as_f64()
                .map_or_else(|| Cow::Owned(column.to_f64_vec()), Cow::Borrowed);
            Ok((view, know(column)))
        }
    }
}

/// The operands a resume arm reads: the frame and its `@` series. Each input is looked
/// up once, by the arm; while it does, the operands note whether every input it read is
/// a frame column already known to hold no NaN ([`clean`](Self::clean)) — then the
/// resume it just computed in place stands, with no second lookup and no scan.
pub(crate) struct Operands<'a> {
    df: &'a DataFrame,
    series: &'a [Ast],
    clean: Cell<bool>,
}

impl<'a> Operands<'a> {
    pub(crate) fn new(df: &'a DataFrame, series: &'a [Ast]) -> Self {
        Operands {
            df,
            series,
            clean: Cell::new(true),
        }
    }

    /// Input `slot` as `f64` (`default` names its column when no operand fills it).
    pub(crate) fn f64(&self, slot: usize, default: &str) -> Result<Cow<'a, [f64]>> {
        let operand = self
            .series
            .get(slot)
            .is_some_and(|n| !matches!(n, Node::Name(s) if s.is_empty()));
        if operand {
            self.clean.set(false); // an evaluated operand carries no remembered fact
            return series_f64(self.df, self.series, slot, default);
        }
        let column = self.df.column(default)?;
        require_numeric(column)?;
        if column.known_has_nan() != Some(false) {
            self.clean.set(false);
        }
        Ok(column
            .as_f64()
            .map_or_else(|| Cow::Owned(column.to_f64_vec()), Cow::Borrowed))
    }

    /// Whether every input read so far is a frame column known to hold no NaN.
    fn clean(&self) -> bool {
        self.clean.get()
    }
}

/// Whether a resume from `from` reads every input's rows `[.., end)` in place (see
/// [`ResumeRows::in_place`]) — free for a column that already knows it holds no NaN, a
/// scan of the reach and the new rows otherwise.
#[allow(clippy::too_many_arguments)] // the command's parts plus the resumed rows
fn in_place(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    args: &[ArgValue],
    series: &[Ast],
    from: usize,
    end: usize,
) -> Result<bool> {
    let mut reach = None; // only needed for an input that must be scanned
    for (slot, default, _) in slots(name, sub, series) {
        let (x, known) = input(df, series, slot, default, Column::known_has_nan)?;
        if known == Some(false) {
            continue;
        }
        let reach = *reach.get_or_insert_with(|| self::reach(name, sub, args));
        if !ResumeRows::in_place(Input::new(&x[..end]), from, reach) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether every input of a default-series command is a frame column that holds no
/// NaN — the column's remembered fact, so a known-complete frame costs no scan and no
/// allocation.
fn complete_in_place(df: &DataFrame, name: &str, sub: Option<&str>) -> Result<bool> {
    for (_, default, _) in slots(name, sub, &[]) {
        let column = df.column(default.unwrap_or_default())?;
        require_numeric(column)?;
        if column.has_nan() {
            return Ok(false);
        }
    }
    Ok(true)
}

impl<'a> Inputs<'a> {
    fn resolve(
        df: &'a DataFrame,
        name: &str,
        sub: Option<&str>,
        series: &[Ast],
    ) -> Result<Inputs<'a>> {
        let mut inputs = Inputs {
            names: Vec::new(),
            views: Vec::new(),
            has_nan: Vec::new(),
        };
        for (slot, default, _) in slots(name, sub, series) {
            let (view, has_nan) = input(df, series, slot, default, |c| Some(c.has_nan()))?;
            inputs.views.push(view);
            inputs.has_nan.push(has_nan.unwrap_or(true));
            inputs.names.push(match default {
                Some(d) if !inputs.names.iter().any(|n| n == d) => Cow::Borrowed(d),
                _ => Cow::Owned(format!("#{slot}")),
            });
        }
        Ok(inputs)
    }

    /// The inputs for the rule, each with its column's NaN fact.
    fn known(&self) -> Vec<Input<'_>> {
        self.views
            .iter()
            .zip(&self.has_nan)
            .map(|(v, &has_nan)| Input::known(v, has_nan))
            .collect()
    }

    fn refs(&self) -> Vec<&[f64]> {
        self.views.iter().map(|v| v.as_ref()).collect()
    }

    /// The operands that read every input from an inputs frame.
    fn series(&self) -> Vec<Ast> {
        self.names
            .iter()
            .map(|n| {
                Node::Name(if n.starts_with('#') {
                    n.to_string()
                } else {
                    String::new()
                })
            })
            .collect()
    }

    /// A frame of the inputs, each gathered by `gather`.
    fn frame(&self, gather: impl Fn(&[f64]) -> Vec<f64>) -> (DataFrame, Vec<Ast>) {
        let names = self.names.iter().map(|n| n.to_string()).collect();
        let columns = self.views.iter().map(|v| Column::f64(gather(v))).collect();
        let frame = DataFrame::new(names, columns, None).expect("inputs share one height");
        (frame, self.series())
    }

    /// A frame of the inputs at `rows` only.
    fn gathered(&self, rows: &[usize]) -> (DataFrame, Vec<Ast>) {
        self.frame(|v| rows.iter().map(|&i| v[i]).collect())
    }
}

/// The frame a full computation (or a state capture) runs on when an input is
/// explicit or a row is missing: the inputs as they are, or their complete rows.
fn over_complete_rows(inputs: &Inputs, rows: &CompleteRows) -> (DataFrame, Vec<Ast>) {
    match rows {
        CompleteRows::All => inputs.frame(|v| v.to_vec()),
        CompleteRows::After(start) => inputs.frame(|v| v[*start..].to_vec()),
        CompleteRows::Kept(kept) => inputs.gathered(kept),
    }
}

/// Compute a recursive command over the complete rows of its inputs: `run` is its
/// dispatch arm.
pub(crate) fn compute(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    series: &[Ast],
    run: impl FnOnce(&DataFrame, &[Ast]) -> Result<Column>,
) -> Result<Column> {
    if default_series(series) && complete_in_place(df, name, sub)? {
        return run(df, series);
    }
    let inputs = Inputs::resolve(df, name, sub, series)?;
    let rows = CompleteRows::of(&inputs.known());
    let (frame, operands) = over_complete_rows(&inputs, &rows);
    let out = run(&frame, &operands)?;
    if rows == CompleteRows::All {
        return Ok(out);
    }
    Ok(Column::f64(rows.scatter(df.height(), out.to_f64_vec())))
}

/// The carried state of a recursive command after a full computation over the
/// complete rows of its inputs — the state as of the last complete row. `run` is its
/// state-capture arm.
pub(crate) fn state(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    series: &[Ast],
    run: impl FnOnce(&DataFrame, &[Ast]) -> Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    if default_series(series) && complete_in_place(df, name, sub).ok()? {
        return run(df, series);
    }
    let inputs = Inputs::resolve(df, name, sub, series).ok()?;
    let rows = CompleteRows::of(&inputs.known());
    let (frame, operands) = over_complete_rows(&inputs, &rows);
    run(&frame, &operands)
}

/// Whether every operand is left to its default series (the arm then reads the
/// caller's frame directly, with nothing to evaluate twice).
fn default_series(series: &[Ast]) -> bool {
    series
        .iter()
        .all(|n| matches!(n, Node::Name(s) if s.is_empty()))
}

/// How many rows before its first new row a resume of a command reads, at most.
fn reach(name: &str, sub: Option<&str>, args: &[ArgValue]) -> usize {
    own_lookback(name, sub, args) + 1
}

/// Resume a recursive command over the new rows `[from, height)` from `prev_state`, on
/// the complete rows only: a missing new row is NaN and leaves the state where the last
/// complete row put it. `run(operands, from)` is its resume arm.
///
/// The arm runs in place first: when every input it read is known complete, or the
/// rows it read turn out complete, that is the answer. Otherwise it runs again on the
/// complete rows.
#[allow(clippy::too_many_arguments)] // the command's parts plus the resume position
pub(crate) fn resume(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    args: &[ArgValue],
    series: &[Ast],
    prev_state: &[f64],
    from: usize,
    run: impl Fn(&Operands, usize) -> Option<(Column, Vec<f64>)>,
) -> Option<(Column, Vec<f64>)> {
    let n = df.height();
    if default_series(series) {
        let ops = Operands::new(df, series);
        let out = run(&ops, from);
        if ops.clean() || in_place(df, name, sub, args, series, from, n).ok()? {
            return out;
        }
    }
    let inputs = Inputs::resolve(df, name, sub, series).ok()?;
    let reach = reach(name, sub, args);
    let refs = inputs.refs();
    if inputs
        .known()
        .into_iter()
        .all(|x| ResumeRows::in_place(x, from, reach))
    {
        // explicit operands, every row they read present: resume on them as they are
        let (frame, operands) = inputs.frame(|v| v.to_vec());
        return run(&Operands::new(&frame, &operands), from);
    }
    let rows = ResumeRows::of(&refs, from, reach);
    if !rows.has_new() {
        return Some((Column::f64(vec![f64::NAN; n - from]), prev_state.to_vec()));
    }
    let (frame, operands) = inputs.gathered(&rows.rows);
    let (col, st) = run(&Operands::new(&frame, &operands), rows.from)?;
    Some((Column::f64(rows.scatter(n, from, col.to_f64_vec())), st))
}

/// Resume a recursive command at the single row `row` (the scalar twin of [`resume`]):
/// `missing` when that row is missing, else `run(operands, row)` on the complete rows.
/// As there, the arm runs in place first and its answer stands when the rows it read
/// are complete.
#[allow(clippy::too_many_arguments)] // the command's parts plus the resume position
pub(crate) fn resume_one<T>(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    args: &[ArgValue],
    series: &[Ast],
    row: usize,
    missing: impl FnOnce() -> Option<T>,
    run: impl Fn(&Operands, usize) -> Option<T>,
) -> Option<T> {
    let ops = Operands::new(df, series);
    let out = run(&ops, row);
    // The rows up to and including `row` (an out-of-range `row` resumes in place, where
    // the arm declines it).
    let end = (row + 1).min(df.height());
    if ops.clean() || in_place(df, name, sub, args, series, row, end).ok()? {
        return out;
    }
    let inputs = Inputs::resolve(df, name, sub, series).ok()?;
    let upto: Vec<&[f64]> = inputs.views.iter().map(|v| &v[..end]).collect();
    let rows = ResumeRows::of(&upto, row, reach(name, sub, args));
    if !rows.has_new() {
        return missing();
    }
    let (frame, operands) = inputs.gathered(&rows.rows);
    run(&Operands::new(&frame, &operands), rows.from)
}

#[cfg(test)]
mod tests {
    use crate::exec::{execute, execute_resume_one_node, initial_state};
    use crate::parse;
    use volas_core::{Column, DataFrame};

    fn frame(open: Vec<f64>) -> DataFrame {
        let close: Vec<f64> = (0..open.len()).map(|i| 50.0 + (i as f64 * 0.7).sin()).collect();
        DataFrame::new(
            vec!["open".into(), "close".into()],
            vec![Column::f64(open), Column::f64(close)],
            None,
        )
        .unwrap()
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
        let st = initial_state(&head, &node, &Column::f64(vec![])).unwrap();
        let full = frame(open.clone());
        let step = execute_resume_one_node(&full, &node, &st, n - 1).unwrap();
        let want = execute(&full, &node).unwrap().to_f64_vec()[n - 1];
        assert_eq!(step.value.to_bits(), want.to_bits());
        let mut gap = open;
        gap[n - 1] = f64::NAN;
        let step = execute_resume_one_node(&frame(gap), &node, &st, n - 1).unwrap();
        assert!(step.value.is_nan());
        assert_eq!(step.state(), &st[..]);
    }
}
