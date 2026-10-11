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

use volas_compute::missing::{CompleteRows, Input, ResumeRows};
use volas_core::{Column, DataFrame, Result};

use crate::exec::{execute, require_numeric, Operands};
use crate::lookback::own_lookback;
use crate::spec::command_spec;
use crate::types::{ArgValue, Ast, Node};

/// A recursive command's input series, resolved once against a frame: each a column —
/// a frame column shared (an `Arc` clone, its NaN fact with it), an evaluated operand
/// owned — never a copy of the values.
struct Inputs {
    /// The column each input is read from in an inputs frame: the command's default
    /// series name, so its arm reads it with no operand, or `#<slot>` for a slot without
    /// a distinct default.
    names: Vec<Cow<'static, str>>,
    columns: Vec<Column>,
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

impl Inputs {
    fn resolve(df: &DataFrame, name: &str, sub: Option<&str>, series: &[Ast]) -> Result<Inputs> {
        let mut inputs = Inputs {
            names: Vec::new(),
            columns: Vec::new(),
            has_nan: Vec::new(),
        };
        for (slot, default, _) in slots(name, sub, series) {
            let column = match series
                .get(slot)
                .filter(|n| !matches!(n, Node::Name(s) if s.is_empty()))
            {
                Some(node) => execute(df, node)?,
                None => df.column(default.unwrap_or_default())?.clone(),
            };
            require_numeric(&column)?;
            inputs.has_nan.push(column.has_nan());
            inputs.columns.push(column);
            inputs.names.push(match default {
                Some(d) if !inputs.names.iter().any(|n| n == d) => Cow::Borrowed(d),
                _ => Cow::Owned(format!("#{slot}")),
            });
        }
        Ok(inputs)
    }

    /// Each input as `f64` (borrowed for an `f64` column).
    fn views(&self) -> Vec<Cow<'_, [f64]>> {
        self.columns
            .iter()
            .map(|c| {
                c.as_f64()
                    .map_or_else(|| Cow::Owned(c.to_f64_vec()), Cow::Borrowed)
            })
            .collect()
    }

    /// The inputs for the rule, each with its column's NaN fact.
    fn known<'v>(&self, views: &'v [Cow<[f64]>]) -> Vec<Input<'v>> {
        views
            .iter()
            .zip(&self.has_nan)
            .map(|(v, &has_nan)| Input::known(v, has_nan))
            .collect()
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

    fn frame_of(&self, columns: Vec<Column>) -> (DataFrame, Vec<Ast>) {
        let names = self.names.iter().map(|n| n.to_string()).collect();
        let frame = DataFrame::new(names, columns, None).expect("inputs share one height");
        (frame, self.series())
    }

    /// A frame of the inputs as they are (the columns shared, not copied).
    fn whole(&self) -> (DataFrame, Vec<Ast>) {
        self.frame_of(self.columns.clone())
    }

    /// A frame of the inputs at `rows` only.
    fn gathered(&self, views: &[Cow<[f64]>], rows: &[usize]) -> (DataFrame, Vec<Ast>) {
        let columns = views
            .iter()
            .map(|v| Column::f64(rows.iter().map(|&i| v[i]).collect()))
            .collect();
        self.frame_of(columns)
    }
}

/// The frame a full computation (or a state capture) runs on when an input is
/// explicit or a row is missing: the inputs as they are, or their complete rows.
fn over_complete_rows(
    inputs: &Inputs,
    views: &[Cow<[f64]>],
    rows: &CompleteRows,
) -> (DataFrame, Vec<Ast>) {
    match rows {
        CompleteRows::All => inputs.whole(),
        CompleteRows::After(start) => {
            let (frame, operands) = inputs.whole();
            (frame.slice_data(*start, frame.height()), operands)
        }
        CompleteRows::Kept(kept) => inputs.gathered(views, kept),
    }
}

/// Compute a recursive command over the complete rows of its inputs: `run` is its
/// dispatch arm.
pub(crate) fn compute(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    series: &[Ast],
    run: impl Fn(&Operands) -> Result<Column>,
) -> Result<Column> {
    if default_series(series) {
        // In place first: the arm's own lookups say whether every column it read is
        // known complete; only a column not yet known is scanned (and then remembered).
        let ops = Operands::new(df, series);
        let out = run(&ops)?;
        if ops.clean() || complete_in_place(df, name, sub)? {
            return Ok(out);
        }
    }
    let inputs = Inputs::resolve(df, name, sub, series)?;
    let views = inputs.views();
    let rows = CompleteRows::of(&inputs.known(&views));
    let (frame, operands) = over_complete_rows(&inputs, &views, &rows);
    let out = run(&Operands::new(&frame, &operands))?;
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
    let views = inputs.views();
    let rows = CompleteRows::of(&inputs.known(&views));
    let (frame, operands) = over_complete_rows(&inputs, &views, &rows);
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
    let views = inputs.views();
    let reach = reach(name, sub, args);
    if inputs
        .known(&views)
        .into_iter()
        .all(|x| ResumeRows::in_place(x, from, reach))
    {
        // explicit operands, every row they read present: resume on them as they are
        let (frame, operands) = inputs.whole();
        return run(&Operands::new(&frame, &operands), from);
    }
    let refs: Vec<&[f64]> = views.iter().map(|v| v.as_ref()).collect();
    let rows = ResumeRows::of(&refs, from, reach);
    if !rows.has_new() {
        return Some((Column::f64(vec![f64::NAN; n - from]), prev_state.to_vec()));
    }
    let (frame, operands) = inputs.gathered(&views, &rows.rows);
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
    let views = inputs.views();
    let upto: Vec<&[f64]> = views.iter().map(|v| &v[..end]).collect();
    let rows = ResumeRows::of(&upto, row, reach(name, sub, args));
    if !rows.has_new() {
        return missing();
    }
    let (frame, operands) = inputs.gathered(&views, &rows.rows);
    run(&Operands::new(&frame, &operands), rows.from)
}

#[cfg(test)]
mod tests {
    use crate::exec::{execute, execute_resume_one_node, initial_state};
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
