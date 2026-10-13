//! Rows entering a [`DataFrame`]: the one owner of "validate an incoming row set
//! against this frame", and the mutations that take admitted rows — appending them,
//! replacing the last row with one, and folding one into a time-frame forming row.
//!
//! Every such mutation runs in two phases. `admit_rows` checks the whole row
//! set first — no cached directive column, no unknown column, an index the frame's
//! index can take ([`crate::Index::check_append`]), every value converted losslessly
//! to the frame's dtype ([`Column::convert_lossless`]) — and [`DataFrame::append`] then
//! decides every row's label before the first write. Only then is the frame written,
//! by steps that cannot fail on admitted rows. So a rejected row set leaves the frame
//! exactly as it was: columns, index, validity and cache cursors.

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Arc;

use crate::column::Column;
use crate::column::CombineOp;
use crate::error::{Result, VolasError};
use crate::index::{IndexKind, Label};

use super::DataFrame;

/// The rows a frame keeps (see [`DataFrame::append_retaining`]): those from `from` on
/// are retained, those from `shown` on are visible.
#[derive(Clone, Copy, Debug)]
pub struct Retained {
    /// The first retained row (the rows before count as dropped).
    pub from: usize,
    /// The first visible row.
    pub shown: usize,
}

/// What [`DataFrame::append`] does with each admitted row.
enum Route {
    /// Every row is appended (no label repeats the last).
    Append,
    /// The one row replaces the current last row (the live re-sent bar).
    ReplaceLast,
    /// Per row: `true` replaces the current last row, `false` appends.
    Rows(Vec<bool>),
}

/// Rows admitted for a frame (`DataFrame::admit_rows`), with whether they carry its
/// leading columns in order — decided once, so the commit maps their columns
/// positionally without comparing the names again.
struct Admitted<'a> {
    rows: Cow<'a, DataFrame>,
    leads: bool,
}

impl DataFrame {
    /// The error for rows to append whose columns `names` include cached directive
    /// columns: this frame computes them itself, so appended values would silently
    /// become data. The message names them and the data columns to select instead.
    fn cached_columns_error<'n>(&self, names: impl IntoIterator<Item = &'n str>) -> VolasError {
        let cached: Vec<&str> = names
            .into_iter()
            .filter(|n| self.computed.contains_key(*n))
            .collect();
        let data: Vec<&str> = self
            .names
            .iter()
            .map(String::as_str)
            .filter(|n| !self.computed.contains_key(*n))
            .collect();
        VolasError::Value(format!(
            "cannot append columns {}: they are cached directives that this DataFrame computes \
             itself. Select the data columns first, e.g. df.append(other[{}]).",
            py_list(&cached),
            py_list(&data)
        ))
    }

    /// Admit `rows` (`admit_rows`) and give them exactly this frame's data columns —
    /// every column but the cached directive ones, in this frame's order and dtypes:
    /// a data column the rows lack becomes a missing cell of its dtype, the value an
    /// append pads it with. The one coercion of a time-frame frame's incoming bars, so
    /// its open period, its fold and its refold all read the same complete bar
    /// whatever the append form left out. Borrowed when the rows already are exactly
    /// the data columns in order (the live bar), so the per-bar path allocates nothing.
    pub fn admit_data_rows<'a>(&self, rows: &'a DataFrame) -> Result<Cow<'a, DataFrame>> {
        let admitted = self.admit_rows(rows)?;
        if admitted.leads && admitted.rows.width() == self.width() - self.computed.len() {
            return Ok(admitted.rows); // the leading columns, none cached: all the data columns
        }
        let rows = admitted.rows.as_ref();
        let (mut names, mut columns) = (Vec::new(), Vec::new());
        for (name, col) in self.names.iter().zip(&self.columns) {
            if self.computed.contains_key(name) {
                continue;
            }
            columns.push(match rows.column_pos(name) {
                Some(j) => rows.columns[j].clone(),
                None => Column::na_of(col.dtype(), rows.height),
            });
            names.push(name.clone());
        }
        DataFrame::new(names, columns, Some((*rows.index).clone())).map(Cow::Owned)
    }

    /// Admit `rows` for appending to (or replacing the last row of) this frame — the
    /// one validation of an incoming row set. `rows` must carry no cached directive
    /// column ([`cached_columns_error`](Self::cached_columns_error)) and no column this frame
    /// lacks (a column it lacks is padded on append); its index must be one this
    /// frame's index can take ([`crate::Index::check_append`]); and each column is
    /// converted to this frame's dtype without changing a value ([`Column::convert_lossless`],
    /// contract #20 — a lossy value raises). Returns the rows with this frame's dtypes,
    /// borrowed when nothing needed converting (the live single-bar path), and whether
    /// they lead with this frame's columns.
    fn admit_rows<'a>(&self, rows: &'a DataFrame) -> Result<Admitted<'a>> {
        // One pass over the rows' columns: each is looked up once — positionally when the
        // rows carry the frame's leading columns in order (the live bar of a frame whose
        // cached columns follow its data columns) — and its dtype compared. Errors are
        // reported in a fixed order afterwards.
        let prefix = self.leads_with(rows);
        let (mut extra, mut convert) = (None, false);
        for (k, (name, col)) in rows.names.iter().zip(&rows.columns).enumerate() {
            let target = if prefix {
                Some(k)
            } else {
                self.name_to_idx.get(name).copied()
            };
            let Some(target) = target else {
                extra.get_or_insert(name);
                continue;
            };
            convert |= col.dtype() != self.columns[target].dtype();
        }
        // Whether the rows carry a cached column, looked up from the smaller side (a live
        // frame caches a few directives over a few data columns).
        let has_cached = if prefix {
            self.first_computed.is_some_and(|p| p < rows.width())
        } else if self.computed.len() <= rows.width() {
            self.computed
                .keys()
                .any(|k| rows.name_to_idx.contains_key(k))
        } else {
            rows.names.iter().any(|n| self.computed.contains_key(n))
        };
        if has_cached {
            return Err(self.cached_columns_error(rows.names.iter().map(String::as_str)));
        }
        if let Some(name) = extra {
            return Err(VolasError::Value(format!(
                "append: column {name:?} is not in the target frame — appended rows must \
                 not introduce a new column (a missing column is NaN-padded; an extra one \
                 is rejected so data is never silently dropped)"
            )));
        }
        self.index.check_append(&rows.index)?;
        if !convert {
            return Ok(Admitted { rows: Cow::Borrowed(rows), leads: prefix });
        }
        let mut columns = rows.columns.clone();
        for (k, name) in rows.names.iter().enumerate() {
            let dtype = self.columns[self.name_to_idx[name]].dtype();
            if columns[k].dtype() != dtype {
                columns[k] = columns[k]
                    .convert_lossless(dtype)
                    .map_err(|e| in_column(name, e))?;
            }
        }
        Ok(Admitted {
            rows: Cow::Owned(rows.same_schema(columns, (*rows.index).clone(), rows.origin)),
            leads: prefix,
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
    /// changed row. The rows are admitted (`admit_rows`) and every label
    /// decided before the first write, so on an error the frame is unchanged. Deciding
    /// a datetime row is O(1); an integer / string row looks its label up.
    pub fn append(&mut self, other: &DataFrame) -> Result<()> {
        self.append_retaining(other, Retained { from: 0, shown: 0 })
    }

    /// [`append`](Self::append) to a frame that retains only its rows from
    /// `retained.from` on (a bounded window: the last `window + max_lookback` rows; the
    /// rows before count as dropped, whenever they are physically compacted) and shows
    /// them from `retained.shown` on: an integer / string label stays unique over the
    /// retained rows only, and a clash with a retained but hidden row says so.
    pub fn append_retaining(&mut self, other: &DataFrame, retained: Retained) -> Result<()> {
        let admitted = self.admit_rows(other)?;
        self.commit_rows(&admitted.rows, admitted.leads, retained)
    }

    /// [`append`](Self::append) of rows already admitted against a frame with this
    /// frame's columns and index (a time-frame frame's open period takes the bars the
    /// frame admitted): only the label routing and the commit. Columns the rows carry
    /// that this frame lacks are not kept.
    pub fn append_admitted(&mut self, rows: &DataFrame) -> Result<()> {
        let leads = self.leads_with(rows);
        self.commit_rows(rows, leads, Retained { from: 0, shown: 0 })
    }

    /// Route admitted rows (`leads`: they carry this frame's leading columns) by label
    /// and write them.
    fn commit_rows(&mut self, rows: &DataFrame, leads: bool, retained: Retained) -> Result<()> {
        match self.route(rows, retained)? {
            Route::Append => self.push_rows(rows, leads),
            Route::ReplaceLast => self.write_last_row(rows, 0, leads),
            Route::Rows(replaces) => {
                for (k, replace) in replaces.into_iter().enumerate() {
                    if replace {
                        self.write_last_row(rows, k, leads)?;
                    } else {
                        self.push_rows(&rows.slice_data(k, k + 1), leads)?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Append rows that each start a new row — a time-frame rollover's period row: every
    /// label must follow the one before it (the last row's first), so it can never fall
    /// into a same-label replacement and merge into a closed row. Admitted like any row
    /// set; on an error the frame is unchanged.
    pub fn append_new(&mut self, other: &DataFrame) -> Result<()> {
        let admitted = self.admit_rows(other)?;
        let rows = admitted.rows.as_ref();
        let mut prev = self.height.checked_sub(1).map(|i| self.index.label_at(i));
        for k in 0..rows.height {
            let label = rows.index.label_at(k);
            if let Some(p) = &prev {
                if !label_precedes(p, &label) {
                    return Err(VolasError::Value(format!(
                        "cannot start a new row labelled {}: it does not follow the last row \
                         (labelled {})",
                        label_text(&label),
                        label_text(p)
                    )));
                }
            }
            prev = Some(label);
        }
        self.push_rows(rows, admitted.leads)
    }

    /// Decide, for each admitted row, whether it replaces the last row or is appended
    /// (see [`append`](Self::append)), or reject the row set. Writes nothing.
    fn route(&self, rows: &DataFrame, retained: Retained) -> Result<Route> {
        let held = |label: &Label| self.index.position_in(label, retained.from..self.height);
        if rows.index.is_unlabelled() || rows.height == 0 {
            // unlabelled rows continue the last label: unique over the retained rows
            if let Some(dup) = self.index.continuation_clash(rows.height, retained.from..self.height) {
                return Err(VolasError::Value(format!(
                    "cannot append unlabelled rows: the next label {dup} is already in the index \
                     (label the rows explicitly)"
                )));
            }
            return Ok(Route::Append);
        }
        let datetime = matches!(rows.index.kind(), IndexKind::Datetime(..));
        if rows.height == 1 && self.height > 0 {
            // The live single-bar path: one comparison with the last label.
            let label = rows.index.label_at(0);
            let last = self.index.label_at(self.height - 1);
            if label == last {
                return Ok(Route::ReplaceLast);
            }
            let fresh = if datetime {
                !label_precedes(&label, &last)
            } else {
                held(&label).is_none()
            };
            if fresh {
                return Ok(Route::Append);
            }
        }
        let mut replaces = Vec::with_capacity(rows.height);
        // Ordered labels can repeat only next to each other (the replacement
        // case below). Otherwise remember the incoming labels once, rather than
        // looking through the whole preceding batch for every row.
        let mut seen = (!datetime && !rows.index.labels_are_ordered())
            .then(|| HashSet::with_capacity(rows.height));
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
            let seen_before = seen.as_mut().is_some_and(|seen| !seen.insert(label.clone()));
            if !replace && !datetime {
                let held = held(&label);
                if held.is_some_and(|p| p < retained.shown) {
                    return Err(VolasError::Value(format!(
                        "cannot append label {}: an earlier row with that label is still \
                         retained in the window's hidden margin (max_lookback)",
                        label_text(&label)
                    )));
                }
                let earlier = held.is_some() || seen_before;
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
        Ok(if replaces.contains(&true) {
            Route::Rows(replaces)
        } else {
            Route::Append
        })
    }

    /// Append admitted rows (see `admit_rows`) at the end — the commit step:
    /// the index is extended first (the one step that checks anything), then every
    /// column grows. Columns `rows` lacks are padded; computed metadata is retained,
    /// leaving the new rows stale.
    fn push_rows(&mut self, rows: &DataFrame, leads: bool) -> Result<()> {
        Arc::make_mut(&mut self.index).extend(&rows.index)?;
        let oh = rows.height;
        if leads {
            // The rows carry the leading columns in order (the live-streaming case):
            // append them positionally and pad the rest.
            let w = rows.width();
            for (dst, src) in self.columns.iter_mut().zip(&rows.columns) {
                dst.append(src)?;
            }
            for pos in w..self.names.len() {
                self.pad_rows(pos, oh)?;
            }
        } else {
            for pos in 0..self.names.len() {
                let n = &self.names[pos];
                match rows.column_pos(n) {
                    Some(j) => self.columns[pos].append(&rows.columns[j])?,
                    None => self.pad_rows(pos, oh)?,
                }
            }
        }
        self.height += oh;
        self.stale_by = super::STALE_BY_APPEND;
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }

    /// Whether `rows` carries this frame's leading columns, in order.
    fn leads_with(&self, rows: &DataFrame) -> bool {
        rows.width() <= self.width() && self.names[..rows.width()] == rows.names[..]
    }

    /// Grow the column at `pos` by `n` rows the appended rows did not carry: a cached
    /// directive (F64 indicator / Bool mask) by a cheap stale placeholder its refresh
    /// overwrites, a data column by a dtype-preserving missing value.
    fn pad_rows(&mut self, pos: usize, n: usize) -> Result<()> {
        self.columns[pos].append_na(n);
        Ok(())
    }

    /// Replace the last row whole with `src`'s row `src_row`, in place: each data column
    /// takes `src`'s value of the same name, or a missing value when `src` lacks it; a
    /// cached directive column is left to its refresh. The label is unchanged. `src`'s
    /// columns must already have this frame's dtypes (`admit_rows` converts
    /// them); every column is checked before the first write, so on an error nothing
    /// changes. Only the last row changed, so each cached column keeps what it carries
    /// and is stale from that row on (see `invalidate_computed_forming_row`).
    pub fn replace_last_row(&mut self, src: &DataFrame, src_row: usize) -> Result<()> {
        if self.height == 0 {
            return Err(VolasError::Shape("no last row to replace".into()));
        }
        if src_row >= src.height {
            return Err(VolasError::Shape(format!(
                "row position {src_row} is out of range (height {})",
                src.height
            )));
        }
        let leads = self.leads_with(src);
        for pos in 0..self.columns.len() {
            if let Some(j) = Self::source_of(&self.names[pos], pos, src, leads) {
                let to = self.columns[pos].dtype();
                if src.columns[j].dtype() != to && !self.computed.contains_key(&self.names[pos]) {
                    // a differently typed value converts losslessly (as an append does)
                    src.columns[j].slice(src_row, src_row + 1).convert_lossless(to)?;
                }
            }
        }
        self.write_last_row(src, src_row, leads)
    }

    /// `src`'s column for this frame's column `pos` (named `name`): positional when
    /// `src` carries this frame's leading columns (`leads`), else looked up by name.
    fn source_of(name: &str, pos: usize, src: &DataFrame, leads: bool) -> Option<usize> {
        match leads {
            true => (pos < src.width()).then_some(pos),
            false => src.column_pos(name),
        }
    }

    /// The write step of [`replace_last_row`](Self::replace_last_row) for a row of
    /// `src` whose columns already have this frame's dtypes (admitted rows): each data
    /// column takes its value, or a missing one; a cached column is left to its refresh.
    fn write_last_row(&mut self, src: &DataFrame, src_row: usize, leads: bool) -> Result<()> {
        let last = self.height - 1;
        for pos in 0..self.columns.len() {
            if !self.computed.is_empty() && self.computed.contains_key(&self.names[pos]) {
                continue;
            }
            match Self::source_of(&self.names[pos], pos, src, leads) {
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
    /// checked ([`Column::check_fold`]), so a fold that fails changes nothing. Only the
    /// forming row changed: it goes stale while each cached column keeps its carried
    /// states. The caller guarantees each `op` is supported for its column's dtype
    /// ([`CombineOp::supports`]); an unsupported pair is the fold's error.
    pub fn fold_forming_row(
        &mut self,
        row: usize,
        src: &DataFrame,
        src_row: usize,
        ops: &[(usize, usize, CombineOp)],
    ) -> Result<()> {
        for &(dst_col, src_col, op) in ops {
            let (dst, from) = (&self.columns[dst_col], &src.columns[src_col]);
            if !cannot_fail(dst, from) {
                dst.check_fold(row, op, false, [(from, src_row)])?;
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

    /// Continue a one-row reduction from its cached prefix, preserving the batch
    /// fold's missing-value and NaN-bit behavior. `missing` records which source
    /// columns had a missing input in that prefix (an arithmetic NaN alone does
    /// not stop a sum). `before` has the same schema as this aggregate: a new
    /// input checkpoints the current row there; a replacement restarts from it.
    /// Both snapshots change only after every fold passes. All rows have already
    /// been admitted, and `ops` matches their schemas, as for `fold_forming_row`.
    pub fn fold_with_checkpoint(&mut self, before: &mut DataFrame, src: &DataFrame, ops: &[(usize, usize, CombineOp)], missing: &[bool], replace: bool) -> Result<()> {
        for &(dst_col, src_col, op) in ops {
            let dst = if replace { &before.columns[dst_col] } else { &self.columns[dst_col] };
            let from = &src.columns[src_col];
            if !cannot_fail(dst, from) { dst.check_fold(0, op, false, [(from, 0)])?; }
        }
        for &(dst_col, src_col, op) in ops {
            if replace {
                self.columns[dst_col].set_last_from(&before.columns[dst_col], 0)?;
            } else {
                before.columns[dst_col].set_last_from(&self.columns[dst_col], 0)?;
            }
            self.columns[dst_col].combine_reduced_at(0, op, &src.columns[src_col], 0, missing[src_col])?;
        }
        self.invalidate_computed_forming_row(0);
        Ok(())
    }

    /// Publish an already reduced, admitted row at the forming row, using the
    /// fold plan's positions. The source has this frame's data dtypes and every
    /// aggregation check has succeeded, so neither conversion nor reduction is
    /// repeated. Missing integer / bool / string cells retain the destination's
    /// unused payload, as `refold_forming_row` does; floats keep the reduced NaN bits.
    pub fn write_folded_row(&mut self, src: &DataFrame, ops: &[(usize, usize, CombineOp)]) -> Result<()> {
        for &(dst_col, src_col, _) in ops {
            let from = &src.columns[src_col];
            if matches!(from, Column::F64(_) | Column::F32(_)) || from.is_valid(0) {
                self.columns[dst_col].set_last_from(from, 0)?;
            } else {
                self.columns[dst_col].set_last_na();
            }
        }
        self.invalidate_computed_forming_row(self.height - 1);
        Ok(())
    }

    /// Check, writing nothing, that re-cumulating the forming row `row` from `src`'s rows
    /// with `last`'s row 0 in place of `src`'s last row (a re-sent bar: a column `last`
    /// lacks keeps `src`'s value) succeeds — so the caller can check the fold before
    /// it replaces that row in `src`. `last` is admitted (`admit_rows`): its
    /// columns have this frame's dtypes.
    pub fn check_refold(
        &self,
        row: usize,
        src: &DataFrame,
        last: &DataFrame,
        ops: &[(usize, usize, CombineOp)],
    ) -> Result<()> {
        let Some(h) = src.height.checked_sub(1) else {
            return Ok(());
        };
        for &(dst_col, src_col, op) in ops {
            let (dst, from) = (&self.columns[dst_col], &src.columns[src_col]);
            if cannot_fail(dst, from) {
                continue; // `last`'s value has the column's dtype too (no name lookup)
            }
            let tail = match last.column_pos(&src.names[src_col]) {
                Some(j) => (&last.columns[j], 0),
                None => (from, h),
            };
            dst.check_fold(row, op, true, (0..h).map(|r| (from, r)).chain([tail]))?;
        }
        Ok(())
    }

    /// Re-cumulate the forming aggregate at `row` from all of `src`'s rows, in place, per
    /// the fold plan `ops` (as [`fold_forming_row`](Self::fold_forming_row)), one pass
    /// per column ([`Column::reduce_into`]) — exactly the batch reduce of the period,
    /// with no allocation. Used when a bar already folded in is re-sent with new values
    /// (a max / min cannot be un-combined). Every column's fold is checked first, so a
    /// fold that fails changes nothing. Only that row changed, so each cached column
    /// keeps what it carries and goes stale from it.
    pub fn refold_forming_row(
        &mut self,
        row: usize,
        src: &DataFrame,
        ops: &[(usize, usize, CombineOp)],
    ) -> Result<()> {
        for &(dst_col, src_col, op) in ops {
            let (dst, from) = (&self.columns[dst_col], &src.columns[src_col]);
            if !cannot_fail(dst, from) {
                dst.check_fold(row, op, true, (0..src.height).map(|r| (from, r)))?;
            }
        }
        for &(dst_col, src_col, op) in ops {
            self.columns[dst_col].reduce_into(row, op, &src.columns[src_col], 0..src.height)?;
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
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
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

/// Whether folding `src` into `dst` cannot fail, so its check is skipped (the live
/// float bar): a float column of the source's own dtype takes every aggregation.
#[inline]
fn cannot_fail(dst: &Column, src: &Column) -> bool {
    dst.dtype().is_float() && src.dtype() == dst.dtype()
}

#[cfg(test)]
#[path = "tests_admit.rs"]
mod tests;
