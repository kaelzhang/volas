//! Rows entering a [`DataFrame`]: the one owner of "validate an incoming row set
//! against this frame", and the mutations that take admitted rows — appending them,
//! replacing the last row with one, and folding one into a time-frame forming row.
//!
//! Every such mutation runs in two phases. [`DataFrame::admit`] checks the whole row
//! set first — no cached directive column, no unknown column, an index the frame's
//! index can take ([`crate::Index::check_append`]), every value converted losslessly
//! to the frame's dtype ([`Column::convert_lossless`]) — and [`DataFrame::append`] then
//! decides every row's label before the first write. Only then is the frame written,
//! by steps that cannot fail on admitted rows. So a rejected row set leaves the frame
//! exactly as it was: columns, index, validity and cache cursors.

use std::borrow::Cow;
use std::sync::Arc;

use crate::column::{Column, CombineOp};
use crate::error::{Result, VolasError};
use crate::index::{IndexKind, Label};

use super::DataFrame;

/// What [`DataFrame::append`] does with each admitted row.
enum Route {
    /// Every row is appended (no label repeats the last).
    Append,
    /// Per row: `true` replaces the current last row, `false` appends.
    Rows(Vec<bool>),
}

impl DataFrame {
    /// Reject cached directive columns among `names` (the columns of rows to append):
    /// this frame computes them itself, so appended values would silently become data.
    /// The message names them and the data columns to select instead.
    pub fn check_no_cached<'n>(&self, names: impl IntoIterator<Item = &'n str>) -> Result<()> {
        if self.computed.is_empty() {
            return Ok(());
        }
        let cached: Vec<&str> = names.into_iter().filter(|n| self.computed.contains_key(*n)).collect();
        if cached.is_empty() {
            return Ok(());
        }
        let data: Vec<&str> = self
            .names
            .iter()
            .map(String::as_str)
            .filter(|n| !self.computed.contains_key(*n))
            .collect();
        Err(VolasError::Value(format!(
            "cannot append columns {}: they are cached directives that this DataFrame computes \
             itself. Select the data columns first, e.g. df.append(other[{}]).",
            py_list(&cached),
            py_list(&data)
        )))
    }

    /// Admit `rows` for appending to (or replacing the last row of) this frame — the
    /// one validation of an incoming row set. `rows` must carry no cached directive
    /// column ([`check_no_cached`](Self::check_no_cached)) and no column this frame
    /// lacks (a column it lacks is padded on append); its index must be one this
    /// frame's index can take ([`crate::Index::check_append`]); and each column is
    /// converted to this frame's dtype without changing a value ([`Column::convert_lossless`],
    /// contract #20 — a lossy value raises). Returns the rows with this frame's dtypes,
    /// borrowed when nothing needed converting (the live single-bar path).
    pub fn admit<'a>(&self, rows: &'a DataFrame) -> Result<Cow<'a, DataFrame>> {
        self.check_no_cached(rows.names.iter().map(String::as_str))?;
        let same = self.names == rows.names;
        if !same {
            if let Some(name) = rows.names.iter().find(|n| !self.has_column(n)) {
                return Err(VolasError::Value(format!(
                    "append: column {name:?} is not in the target frame — appended rows must \
                     not introduce a new column (a missing column is NaN-padded; an extra one \
                     is rejected so data is never silently dropped)"
                )));
            }
        }
        self.index.check_append(&rows.index)?;
        let mut converted: Option<Vec<Column>> = None;
        for (k, (name, col)) in rows.names.iter().zip(&rows.columns).enumerate() {
            let target = if same { k } else { self.name_to_idx[name] };
            let dtype = self.columns[target].dtype();
            if col.dtype() != dtype {
                let col = col.convert_lossless(dtype).map_err(|e| in_column(name, e))?;
                converted.get_or_insert_with(|| rows.columns.clone())[k] = col;
            }
        }
        Ok(match converted {
            None => Cow::Borrowed(rows),
            Some(columns) => {
                Cow::Owned(rows.same_schema(columns, (*rows.index).clone(), rows.origin))
            }
        })
    }

    /// Append `other`'s rows in place, keyed by their labels: a row labelled like the
    /// current last row replaces it whole ([`replace_last_row`](Self::replace_last_row));
    /// any other row is appended, except that a datetime index only ever increases (a
    /// row labelled before the last is an error) and an integer / string index stays
    /// unique (a row labelled like an earlier row is an error). Rows without explicit
    /// labels (a `RangeIndex`) take the next labels. Columns `other` lacks are padded
    /// (a cached directive column with a stale placeholder its refresh overwrites, a
    /// data column with a missing value); cached columns are left stale from the first
    /// changed row. The rows are admitted ([`admit`](Self::admit)) and every label
    /// decided before the first write, so on an error the frame is unchanged. Deciding
    /// a datetime row is O(1); an integer / string row looks its label up.
    pub fn append(&mut self, other: &DataFrame) -> Result<()> {
        let rows = self.admit(other)?;
        let rows = rows.as_ref();
        match self.route(rows)? {
            Route::Append => self.push_rows(rows),
            Route::Rows(replaces) => {
                for (k, replace) in replaces.into_iter().enumerate() {
                    if replace {
                        self.replace_last_row(rows, k)?;
                    } else {
                        self.push_rows(&rows.slice_data(k, k + 1))?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Decide, for each admitted row, whether it replaces the last row or is appended
    /// (see [`append`](Self::append)), or reject the row set. Writes nothing.
    fn route(&self, rows: &DataFrame) -> Result<Route> {
        if matches!(rows.index.kind(), IndexKind::Range(_)) || rows.height == 0 {
            return Ok(Route::Append); // unlabelled rows: `check_append` vetted their labels
        }
        let datetime = matches!(rows.index.kind(), IndexKind::Datetime(..));
        if rows.height == 1 && self.height > 0 {
            // The live single-bar path: one comparison with the last label.
            let label = rows.index.label_at(0);
            let last = self.index.label_at(self.height - 1);
            if label == last {
                return Ok(Route::Rows(vec![true]));
            }
            let fresh = if datetime {
                !label_precedes(&label, &last)
            } else {
                self.index.position_of(&label).is_none()
            };
            if fresh {
                return Ok(Route::Append);
            }
        }
        let mut replaces = Vec::with_capacity(rows.height);
        for k in 0..rows.height {
            let label = rows.index.label_at(k);
            let prev = if k > 0 {
                Some(rows.index.label_at(k - 1))
            } else {
                self.height.checked_sub(1).map(|i| self.index.label_at(i))
            };
            let replace = prev.as_ref() == Some(&label);
            if !replace && datetime && prev.as_ref().is_some_and(|p| label_precedes(&label, p)) {
                return Err(VolasError::Value(format!(
                    "cannot append a row labelled {}: a datetime index only increases \
                     (only the last row can be replaced)",
                    label_text(&label)
                )));
            }
            if !replace && !datetime {
                let earlier = self.index.position_of(&label).is_some()
                    || (0..k).any(|j| rows.index.label_at(j) == label);
                if earlier {
                    return Err(VolasError::Value(format!(
                        "cannot append a row labelled {}: an earlier row has that label \
                         (only the last row can be replaced)",
                        label_text(&label)
                    )));
                }
            }
            replaces.push(replace);
        }
        Ok(if replaces.contains(&true) { Route::Rows(replaces) } else { Route::Append })
    }

    /// Append admitted rows (see [`admit`](Self::admit)) at the end — the commit step:
    /// the index is extended first (the one step that checks anything), then every
    /// column grows. Columns `rows` lacks are padded; computed metadata is retained,
    /// leaving the new rows stale.
    fn push_rows(&mut self, rows: &DataFrame) -> Result<()> {
        Arc::make_mut(&mut self.index).extend(&rows.index)?;
        let oh = rows.height;
        if self.names == rows.names {
            // Identical schema (the live-streaming case): append positionally.
            for (dst, src) in self.columns.iter_mut().zip(&rows.columns) {
                dst.append(src)?;
            }
        } else {
            for pos in 0..self.names.len() {
                let n = &self.names[pos];
                if let Some(j) = rows.column_pos(n) {
                    self.columns[pos].append(&rows.columns[j])?;
                } else if self.computed.contains_key(n) {
                    // A cached directive (F64 indicator / Bool mask): a cheap stale
                    // placeholder that the refresh recomputes and overwrites.
                    self.columns[pos].append_missing(oh)?;
                } else {
                    // A data column keeps its data semantics: a dtype-preserving NA.
                    self.columns[pos].append_na(oh);
                }
            }
        }
        self.height += oh;
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }

    /// Replace the last row whole with `src`'s row `src_row`, in place: each data column
    /// takes `src`'s value of the same name, or a missing value when `src` lacks it; a
    /// cached directive column is left to its refresh. The label is unchanged. `src`'s
    /// columns must already have this frame's dtypes ([`admit`](Self::admit) converts
    /// them); every column is checked before the first write, so on an error nothing
    /// changes. Only the last row changed, so each cached column keeps what it carries
    /// and is stale from that row on (see `invalidate_computed_forming_row`).
    pub fn replace_last_row(&mut self, src: &DataFrame, src_row: usize) -> Result<()> {
        let Some(last) = self.height.checked_sub(1) else {
            return Err(VolasError::Shape("no last row to replace".into()));
        };
        if src_row >= src.height {
            return Err(VolasError::Shape(format!(
                "row position {src_row} is out of range (height {})",
                src.height
            )));
        }
        let same = self.names == src.names;
        let source = |pos: usize| if same { Some(pos) } else { src.column_pos(&self.names[pos]) };
        for pos in 0..self.columns.len() {
            if let Some(j) = source(pos) {
                let (to, from) = (self.columns[pos].dtype(), src.columns[j].dtype());
                if to != from && !self.computed.contains_key(&self.names[pos]) {
                    return Err(VolasError::DType(format!(
                        "cannot replace a {to} cell with a {from} value"
                    )));
                }
            }
        }
        for pos in 0..self.columns.len() {
            if !self.computed.is_empty() && self.computed.contains_key(&self.names[pos]) {
                continue;
            }
            match source(pos) {
                Some(j) => self.columns[pos].set_last_from(&src.columns[j], src_row)?,
                None => self.columns[pos].set_last_na(),
            }
        }
        self.invalidate_computed_forming_row(last);
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }

    /// Fold `src[src_row]` into the forming aggregate at `row`, in place, per the
    /// `(dst_col, src_col, op)` plan — the allocation-free live tf-fold. Each cell is
    /// combined through [`Column::combine_at`], after every column's fold has been
    /// checked ([`Column::check_fold`]), so a fold that overflows changes nothing. Only
    /// the forming row changed: it goes stale while each cached column keeps its carried
    /// states. The caller guarantees every `op`'s column is fold-eligible (numeric /
    /// datetime) and that `src` has this frame's dtypes.
    pub fn fold_forming_row(
        &mut self,
        row: usize,
        src: &DataFrame,
        src_row: usize,
        ops: &[(usize, usize, CombineOp)],
    ) -> Result<()> {
        for &(dst_col, src_col, op) in ops {
            let dst = &self.columns[dst_col];
            if !dst.dtype().is_float() {
                dst.check_fold(row, op, false, [(&src.columns[src_col], src_row)])?;
            }
        }
        for &(dst_col, src_col, op) in ops {
            self.columns[dst_col].combine_at(row, op, &src.columns[src_col], src_row)?;
        }
        self.invalidate_computed_forming_row(row);
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }

    /// Check, writing nothing, that re-cumulating the forming row `row` from `src`'s rows
    /// with `last`'s row 0 in place of `src`'s last row (a re-sent bar: a column `last`
    /// lacks keeps `src`'s value) stays in range — so the caller can check the fold
    /// before it replaces that row in `src`. Float columns are not looked at.
    pub fn check_refold(
        &self,
        row: usize,
        src: &DataFrame,
        last: &DataFrame,
        ops: &[(usize, usize, CombineOp)],
    ) -> Result<()> {
        let Some(h) = src.height.checked_sub(1) else { return Ok(()) };
        for &(dst_col, src_col, op) in ops {
            let dst = &self.columns[dst_col];
            if dst.dtype().is_float() {
                continue;
            }
            let from = &src.columns[src_col];
            let tail = match last.column_pos(&src.names[src_col]) {
                Some(j) => (&last.columns[j], 0),
                None => (from, h),
            };
            dst.check_fold(row, op, true, (0..h).map(|r| (from, r)).chain([tail]))?;
        }
        Ok(())
    }

    /// Re-cumulate the forming aggregate at `row` from all of `src`'s rows, in place, per
    /// the fold plan `ops` (as [`fold_forming_row`](Self::fold_forming_row)): the first
    /// row initializes each cell ([`Column::init_at`]), every later one folds in
    /// ([`Column::combine_at`]) — exactly the batch reduce of the period, with no
    /// allocation. Used when a bar already folded in is re-sent with new values (a
    /// max / min cannot be un-combined). Every column's fold is checked first, so a
    /// fold that overflows changes nothing. Only that row changed, so each cached column
    /// keeps what it carries and goes stale from it.
    pub fn refold_forming_row(
        &mut self,
        row: usize,
        src: &DataFrame,
        ops: &[(usize, usize, CombineOp)],
    ) -> Result<()> {
        for &(dst_col, src_col, op) in ops {
            let from = &src.columns[src_col];
            self.columns[dst_col].check_fold(row, op, true, (0..src.height).map(|r| (from, r)))?;
        }
        for &(dst_col, src_col, op) in ops {
            let (dst, from) = (&mut self.columns[dst_col], &src.columns[src_col]);
            dst.init_at(row, op, from, 0)?;
            for r in 1..src.height {
                dst.combine_at(row, op, from, r)?;
            }
        }
        self.invalidate_computed_forming_row(row);
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }
}

/// A conversion error, naming the column it happened in.
fn in_column(name: &str, e: VolasError) -> VolasError {
    let at = |m: String| format!("column {name:?}: {m}");
    match e {
        VolasError::DType(m) => VolasError::DType(at(m)),
        VolasError::Overflow(m) => VolasError::Overflow(at(m)),
        other => other, // LCOV_EXCL_LINE (convert_lossless raises DType / Overflow only)
    }
}

/// Names written as a Python list literal (`['a', 'b']`), as the user types them.
fn py_list(names: &[&str]) -> String {
    let items: Vec<String> = names.iter().map(|n| py_str(n)).collect();
    format!("[{}]", items.join(", "))
}

/// A string written as Python's `repr` writes it: single quotes unless the text holds
/// a single quote and no double quote; backslashes, the quote and control characters
/// escaped.
fn py_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_control() => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Whether datetime label `a` orders before `b` (only a datetime index is ordered by
/// the append rule; a label of another kind never precedes).
fn label_precedes(a: &Label, b: &Label) -> bool {
    matches!((a, b), (Label::I64(x), Label::I64(y)) if x < y)
}

/// A label as written in a message.
fn label_text(label: &Label) -> String {
    match label {
        Label::I64(v) => v.to_string(),
        Label::Str(s) => format!("{s:?}"),
    }
}

#[cfg(test)]
#[path = "tests_admit.rs"]
mod tests;
