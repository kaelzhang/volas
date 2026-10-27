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

use crate::column::ColumnData;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::column::Column;
use crate::column::CombineOp;
use crate::error::{Result, VolasError};
use crate::index::{AppendMode, AppendPlan, AppendTail, IndexInput, RowPlan};

use super::{DataFrame, Replay};

/// The rows a frame keeps (see [`DataFrame::append_retaining`]): those from `from` on
/// are retained, those from `shown` on are visible.
#[derive(Clone, Copy, Debug)]
pub struct Retained {
    /// The first retained row (the rows before count as dropped).
    pub from: usize,
    /// The first visible row.
    pub shown: usize,
}

/// Rows admitted for a frame (`DataFrame::admit_rows`), with whether they carry its
/// leading columns in order — decided once, so the commit maps their columns
/// positionally without comparing the names again.
struct Admitted<'a, 'target> {
    rows: Cow<'a, DataFrame>,
    leads: bool,
    index: IndexInput<'target, 'a>,
}

/// An admitted reduced row, bound by exclusive borrow to its destination.
/// Only `DataFrame::check_folded_row` can construct it.
pub struct FoldedRow<'a> {
    frame: &'a mut DataFrame,
    src: &'a DataFrame,
    ops: &'a [(usize, usize, CombineOp)],
}
impl FoldedRow<'_> {
    /// Publish the checked row on the same destination.
    pub fn land(self) -> Result<()> { self.frame.write_folded_row(self.src, self.ops) }
}

/// A target-bound forming-row position plan. Landing still checks every column's
/// aggregation before the first write, without adding guards to the hot fold body.
pub struct FormingFold<'a> {
    frame: &'a mut DataFrame,
    row: usize,
    source: &'a DataFrame,
    source_row: usize,
    ops: &'a [(usize, usize, CombineOp)],
}
impl FormingFold<'_> {
    /// Check the arithmetic and publish the admitted fold atomically.
    #[inline]
    pub fn land(self) -> Result<()> { self.frame.fold_forming_row(self.row, self.source, self.source_row, self.ops, true) }
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
            // These are this target's data columns, even if the source computed
            // some of them. Its cache/history identity must not enter a raw bar.
            return Ok(if admitted.rows.computed.is_empty() { admitted.rows }
                else { Cow::Owned(admitted.rows.admitted_values()) });
        }
        let rows = admitted.rows.as_ref();
        let (mut names, mut columns) = (Vec::new(), Vec::new());
        for (name, col) in self.names.iter().zip(&self.columns) {
            if self.computed.contains_key(name.as_str()) {
                continue;
            }
            columns.push(match rows.column_pos(name) {
                Some(j) => rows.columns[j].clone(),
                None => {
                    let mut missing = Column::na_of(col.dtype(), 0);
                    missing.prepare_growth(rows.height, None, None, false)?;
                    missing.append_na(rows.height);
                    missing
                },
            });
            names.push(name.clone());
        }
        DataFrame::new(names, columns, Some((*rows.index).clone())).map(Cow::Owned)
    }

    /// The already admitted physical values as a new plain row set. Share the
    /// checked labels and buffers; no historical values or labels are copied.
    fn admitted_values(&self) -> DataFrame {
        DataFrame {
            names: self.names.clone(), name_to_idx: self.name_to_idx.clone(),
            columns: self.columns.clone(), index: self.index.clone(), height: self.height,
            computed: Default::default(), first_computed: None, origin: 0,
            stale_by: super::STALE_BY_APPEND, lead: None,
        }
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
    fn admit_rows<'a, 't>(&'t self, rows: &'a DataFrame) -> Result<Admitted<'a, 't>> {
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
                .any(|k| rows.name_to_idx.contains_key(k.as_ref()))
        } else {
            rows.names.iter().any(|n| self.computed.contains_key(n.as_str()))
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
        let index = self.index.admit_input(&rows.index)?;
        if !convert {
            if self.lead.is_some() {
                for (name, col) in rows.names.iter().zip(&rows.columns) {
                    self.check_lead_values(name, col, 0, col.len())?;
                }
            }
            return Ok(Admitted { rows: Cow::Borrowed(rows), leads: prefix, index });
        }
        let mut columns = rows.columns.clone();
        for (k, name) in rows.names.iter().enumerate() {
            let dtype = self.columns[self.name_to_idx[name]].dtype();
            if columns[k].dtype() != dtype {
                columns[k] = columns[k]
                    .convert_lossless(dtype)
                    .map_err(|e| in_column(name, e))?;
            }
            self.check_lead_values(name, &columns[k], 0, columns[k].len())?;
        }
        let mut converted = rows.same_schema(columns, (*rows.index).clone(), rows.origin);
        // `convert` already proves a dtype changed. Retire the schema here,
        // rather than re-reading every dtype on all schema-preserving derivations.
        converted.names = Arc::new((*rows.names).clone());
        Ok(Admitted { rows: Cow::Owned(converted), leads: prefix, index })
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
        let Admitted { rows, leads, index } = self.admit_rows(other)?;
        let plan = index.row_plan(0, 0)?;
        self.commit_rows(&rows, leads, plan)
    }

    /// [`append`](Self::append) to a frame that retains only its rows from
    /// `retained.from` on (a bounded window: the last `window + max_lookback` rows; the
    /// rows before count as dropped, whenever they are physically compacted) and shows
    /// them from `retained.shown` on: an integer / string label stays unique over the
    /// retained rows only, and a clash with a retained but hidden row says so.
    /// Reusing a dropped label physically compacts that prefix through `replay`
    /// before the new rows are committed, so even physical indexes stay unique.
    pub fn append_retaining(&mut self, other: &DataFrame, retained: Retained, replay: &dyn Replay) -> Result<()> {
        if retained.from > retained.shown || retained.shown > self.height {
            return Err(VolasError::Shape("retained rows must satisfy from <= shown <= frame height".into()));
        }
        let Admitted { rows, leads, index } = self.admit_rows(other)?;
        let plan = index.row_plan(retained.from, retained.shown)?;
        let from = plan.compact_from();
        if from == 0 {
            return self.commit_rows(&rows, leads, plan);
        }
        // The same plan proves that only dropped labels collide. Compact through
        // the derivation owner on a candidate, then consume it without re-routing.
        let mut candidate = self.clone();
        candidate.compact_tail(self.height - from, replay);
        candidate.commit_rows(&rows, leads, plan)?;
        *self = candidate;
        Ok(())
    }

    pub(super) fn commit_rows(&mut self, rows: &DataFrame, leads: bool, plan: RowPlan<'_>) -> Result<()> {
        self.prepare_columns(rows, leads, plan.appended(), plan.replaces(), None)?;
        let result = self.commit_prepared_rows(rows, leads, plan);
        if result.is_err() { for col in &mut self.columns { col.finish_growth(); } }
        result
    }

    /// Consume the Column owners' capacity preparation exactly once.
    pub(super) fn commit_prepared_rows(&mut self, rows: &DataFrame, leads: bool, plan: RowPlan<'_>) -> Result<()> {
        let added = plan.appended();
        match plan {
            RowPlan::Range(step) => {
                Arc::make_mut(&mut self.index).apply_range_step(step);
                self.push_columns::<0>(rows, leads, added)
            }
            RowPlan::Replacement(_) => {
                self.write_last_row(rows, 0, leads)?;
                for col in &mut self.columns { col.finish_growth(); }
                #[cfg(debug_assertions)]
                self.debug_check();
                Ok(())
            },
            RowPlan::Append(plan) => self.commit_plan(rows, leads, plan),
        }
    }

    fn commit_plan(&mut self, rows: &DataFrame, leads: bool, tail: AppendTail<'_>) -> Result<()> {
        let plan = tail.into_plan();
        let replaces = plan.replaces();
        if replaces {
            // All label checks precede the replacement. Resource preparation also
            // precedes it, so a representability/capacity error cannot publish it.
            Arc::make_mut(&mut self.index).prepare_replacement(&plan)?;
            self.write_last_row(rows, 0, leads)?;
            self.push_prepared_rows::<1>(rows, leads, plan)
        } else {
            self.push_prepared_rows::<0>(rows, leads, plan)
        }
    }

    /// Append rows that each start a new row — a time-frame rollover's period row: every
    /// label must follow the one before it (the last row's first), so it can never fall
    /// into a same-label replacement and merge into a closed row. Admitted like any row
    /// set; on an error the frame is unchanged.
    pub fn append_new(&mut self, other: &DataFrame) -> Result<()> {
        let admitted = self.admit_rows(other)?;
        let rows = admitted.rows.as_ref();
        let plan = admitted.index.plan(AppendMode::NewRows)?;
        self.push_rows(rows, admitted.leads, plan)
    }

    /// Append admitted rows (see `admit_rows`) at the end — the commit step:
    /// the validated index plan is consumed first, then every
    /// column grows. Columns `rows` lacks are padded; computed metadata is retained,
    /// leaving the new rows stale.
    pub(super) fn push_rows(&mut self, rows: &DataFrame, leads: bool, plan: AppendPlan<'_>) -> Result<()> {
        self.prepare_columns(rows, leads, plan.appended(), false, None)?;
        let result = self.push_prepared_rows::<0>(rows, leads, plan);
        if result.is_err() { for col in &mut self.columns { col.finish_growth(); } }
        result
    }

    fn push_prepared_rows<const START: usize>(&mut self, rows: &DataFrame, leads: bool, plan: AppendPlan<'_>) -> Result<()> {
        let added = plan.appended();
        Arc::make_mut(&mut self.index).apply_append(plan)?;
        self.push_columns::<START>(rows, leads, added)
    }

    /// Column owns storage bounds, COW, payload and bitmap reservation. Every
    /// fallible reservation precedes index, replacement values, height and cache writes.
    pub(super) fn prepare_columns(&mut self, rows: &DataFrame, leads: bool, added: usize, replaces: bool, cell: Option<usize>) -> Result<()> {
        if added == 0 && !replaces { return Ok(()); }
        for pos in 0..self.columns.len() {
            if added == 0 && self.computed.contains_key(self.names[pos].as_str()) { continue; }
            let source = Self::source_of(&self.names[pos], pos, rows, leads).map(|j| &rows.columns[j]);
            if let Err(error) = self.columns[pos].prepare_growth(added, source, cell, replaces) {
                for col in &mut self.columns { col.finish_growth(); }
                return Err(error);
            }
        }
        Ok(())
    }

    fn push_columns<const START: usize>(&mut self, rows: &DataFrame, leads: bool, oh: usize) -> Result<()> {
        if leads {
            // The rows carry the leading columns in order (the live-streaming case):
            // append them positionally and pad the rest.
            let w = rows.width();
            for (dst, src) in self.columns.iter_mut().zip(&rows.columns) {
                if START == 0 { dst.append_whole(src.whole())?; } else { dst.append_from::<START>(src)?; }
            }
            for pos in w..self.names.len() {
                self.pad_rows(pos, oh)?;
            }
        } else {
            for pos in 0..self.names.len() {
                let n = &self.names[pos];
                match rows.column_pos(n) {
                    Some(j) => if START == 0 { self.columns[pos].append_whole(rows.columns[j].whole())?; } else { self.columns[pos].append_from::<START>(&rows.columns[j])?; },
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
    /// values are admitted once, including lossless dtype conversion; every
    /// column is checked before the first write, so on an error nothing
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
        let mut converted: Option<HashMap<usize, Column>> = None;
        for pos in 0..self.columns.len() {
            if self.computed.contains_key(self.names[pos].as_str()) { continue; }
            if let Some(j) = Self::source_of(&self.names[pos], pos, src, leads) {
                let to = self.columns[pos].dtype();
                self.check_lead_values(&self.names[pos], &src.columns[j], src_row, src_row + 1)?;
                if src.columns[j].dtype() != to {
                    let cell = src.columns[j].slice(src_row, src_row + 1).convert_lossless(to)?;
                    converted.get_or_insert_with(HashMap::new).insert(j, cell);
                }
            }
        }
        self.prepare_columns(src, leads, 0, true, Some(src_row))?;
        let result = match converted {
            None => self.write_last_row(src, src_row, leads),
            Some(cells) => self.write_last_row_with(|pos, name| {
                Self::source_of(name, pos, src, leads).map(|j| match cells.get(&j) {
                    Some(cell) => (cell, 0),
                    None => (&src.columns[j], src_row),
                })
            }),
        };
        for col in &mut self.columns { col.finish_growth(); }
        #[cfg(debug_assertions)]
        self.debug_check();
        result
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
        self.write_last_row_with(|pos, name| Self::source_of(name, pos, src, leads).map(|j| (&src.columns[j], src_row)))
    }

    fn write_last_row_with<'a>(&mut self, mut source: impl FnMut(usize, &str) -> Option<(&'a Column, usize)>) -> Result<()> {
        let last = self.height - 1;
        for pos in 0..self.columns.len() {
            if !self.computed.is_empty() && self.computed.contains_key(self.names[pos].as_str()) {
                continue;
            }
            match source(pos, &self.names[pos]) {
                Some((column, row)) => self.columns[pos].set_last_from_admitted(last, column, row),
                None => self.columns[pos].set_last_na_admitted(last),
            }
        }
        self.invalidate_computed_forming_row(last);
        Ok(())
    }

    /// Admit the complete position plan against this exact destination before a
    /// forming-row fold. The returned plan cannot land on another frame.
    pub fn check_forming_fold<'a>(&'a mut self, row: usize, src: &'a DataFrame, src_row: usize, ops: &'a [(usize, usize, CombineOp)]) -> Result<FormingFold<'a>> {
        self.check_fold_positions(row, src, Some(src_row), ops)?;
        Ok(FormingFold { frame: self, row, source: src, source_row: src_row, ops })
    }

    /// Fold `src[src_row]` into the forming aggregate at `row`, in place, per the
    /// `(dst_col, src_col, op)` plan — the allocation-free live tf-fold. Each cell is
    /// combined through [`Column::combine_at`], after every column's fold has been
    /// checked ([`Column::check_fold`]), so a fold that fails changes nothing. Only the
    /// forming row changed: it goes stale while each cached column keeps its carried
    /// states. The caller guarantees each `op` is supported for its column's dtype
    /// ([`CombineOp::supports`]); an unsupported pair is the fold's error.
    // Keep the gated commit body present after sealing it behind the position plan.
    #[inline(never)]
    pub(crate) fn fold_forming_row(
        &mut self,
        row: usize,
        src: &DataFrame,
        src_row: usize,
        ops: &[(usize, usize, CombineOp)],
        arithmetic: bool,
    ) -> Result<()> {
        if arithmetic { self.check_forming_arithmetic(row, src, src_row, ops)?; }
        for &(dst_col, src_col, op) in ops {
            self.columns[dst_col].combine_at(row, op, &src.columns[src_col], src_row)?;
        }
        self.invalidate_computed_forming_row(row);
        #[cfg(debug_assertions)]
        self.debug_check();
        Ok(())
    }

    // Schema-admitted floating plans have already proved this loop vacuous.
    // Independent checked entries and fallible layouts retain the full preflight.
    #[inline(never)]
    fn check_forming_arithmetic(&self, row: usize, src: &DataFrame, src_row: usize, ops: &[(usize, usize, CombineOp)]) -> Result<()> {
        for &(dst_col, src_col, op) in ops {
            let (dst, from) = (&self.columns[dst_col], &src.columns[src_col]);
            if !cannot_fail(dst, from) {
                self.check_fold_with_lead(dst_col, dst, (row, op, false), [(from, src_row)])?;
            }
        }
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
        self.check_fold_positions(0, src, Some(0), ops)?;
        before.check_fold_positions(0, src, Some(0), ops)?;
        if self.height != 1 || before.height != 1 || self.names != before.names || self.columns.iter().zip(&before.columns).any(|(a, b)| a.dtype() != b.dtype()) || missing.len() < src.width() {
            return Err(VolasError::Shape("fold checkpoint schema or missing mask does not match".into()));
        }
        self.fold_checkpoint_admitted(before, src, ops, missing, replace, true)
    }

    pub(super) fn fold_checkpoint_admitted(&mut self, before: &mut DataFrame, src: &DataFrame, ops: &[(usize, usize, CombineOp)], missing: &[bool], replace: bool, arithmetic: bool) -> Result<()> {
        if arithmetic {
            for &(dst_col, src_col, op) in ops {
                let dst = if replace { &before.columns[dst_col] } else { &self.columns[dst_col] };
                let from = &src.columns[src_col];
                if !cannot_fail(dst, from) { self.check_fold_with_lead(dst_col, dst, (0, op, false), [(from, 0)])?; }
            }
        }
        for &(dst_col, src_col, op) in ops {
            if replace {
                self.columns[dst_col].set_last_from_admitted(0, &before.columns[dst_col], 0);
            } else {
                before.columns[dst_col].set_last_from_admitted(0, &self.columns[dst_col], 0);
            }
            self.columns[dst_col].combine_reduced_at(0, op, &src.columns[src_col], 0, missing[src_col])?;
        }
        self.invalidate_computed_forming_row(0);
        Ok(())
    }

    /// A bulk fold's validation frame: only the forming data row is copied, but
    /// the original historical dtype constraints still apply. Older visible rows
    /// have these same data dtypes; cached values and their computation are unused.
    pub fn forming_fold_probe(&self) -> DataFrame {
        let mut probe = self.data_rows(self.height.saturating_sub(1), self.height);
        probe.lead = self.lead.clone();
        probe
    }

    /// Validate a checkpoint prefix followed by one admitted bar against this
    /// destination, in column order, before either the checkpoints or frame change.
    /// `None` starts a one-bar reduction (a correction with no earlier prefix).
    /// Prefix and source use the source positions in `ops`, with matching dtypes.
    pub fn check_fold_from(&self, prefix: Option<&DataFrame>, src: &DataFrame, ops: &[(usize, usize, CombineOp)]) -> Result<()> {
        let row = self.height.checked_sub(1).ok_or_else(|| VolasError::Shape("no last row to fold".into()))?;
        self.check_fold_positions(row, src, Some(0), ops)?;
        if let Some(prefix) = prefix { self.check_fold_positions(row, prefix, Some(0), ops)?; }
        for &(dst_col, src_col, op) in ops {
            let (dst, from) = (&self.columns[dst_col], &src.columns[src_col]);
            if !cannot_fail(dst, from) {
                let cells = prefix.map(|p| (&p.columns[src_col], 0)).into_iter().chain([(from, 0)]);
                self.check_fold_with_lead(dst_col, dst, (row, op, true), cells)?;
            }
        }
        Ok(())
    }

    /// Admit a reduced row against this exact target before any value is written.
    pub fn check_folded_row<'a>(&'a mut self, src: &'a DataFrame, ops: &'a [(usize, usize, CombineOp)]) -> Result<FoldedRow<'a>> {
        let row = self.height.checked_sub(1).ok_or_else(|| VolasError::Shape("no last row to replace".into()))?;
        self.check_fold_positions(row, src, Some(0), ops)?;
        for &(dst_col, src_col, _) in ops {
            if self.columns[dst_col].dtype() != src.columns[src_col].dtype() {
                return Err(VolasError::DType("folded row must have the target column dtypes".into()));
            }
            self.check_lead_values(&self.names[dst_col], &src.columns[src_col], 0, 1)?;
        }
        Ok(FoldedRow { frame: self, src, ops })
    }

    /// Publish an already reduced, admitted row at the forming row, using the
    /// fold plan's positions. The source has this frame's data dtypes and every
    /// aggregation check has succeeded, so neither conversion nor reduction is
    /// repeated. Missing integer / bool / string cells retain the destination's
    /// unused payload, as `refold_forming_row` does; floats keep the reduced NaN bits.
    pub(crate) fn write_folded_row(&mut self, src: &DataFrame, ops: &[(usize, usize, CombineOp)]) -> Result<()> {
        let last = self.height - 1;
        for &(dst_col, src_col, _) in ops {
            let from = &src.columns[src_col];
            if matches!(from, Column { data: ColumnData::F64(_) } | Column { data: ColumnData::F32(_) }) || from.is_valid(0) {
                self.columns[dst_col].set_last_from_admitted(last, from, 0);
            } else {
                self.columns[dst_col].set_last_na_admitted(last);
            }
        }
        self.invalidate_computed_forming_row(last);
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
        self.check_fold_positions(row, src, None, ops)?;
        if last.height == 0 { return Err(VolasError::Shape("no replacement row to fold".into())); }
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
            self.check_fold_with_lead(dst_col, dst, (row, op, true), (0..h).map(|r| (from, r)).chain([tail]))?;
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
        self.check_fold_positions(row, src, None, ops)?;
        self.refold_admitted(row, src, ops, true)
    }

    pub(super) fn refold_admitted(&mut self, row: usize, src: &DataFrame, ops: &[(usize, usize, CombineOp)], arithmetic: bool) -> Result<()> {
        if arithmetic {
            for &(dst_col, src_col, op) in ops {
                let (dst, from) = (&self.columns[dst_col], &src.columns[src_col]);
                if !cannot_fail(dst, from) {
                    self.check_fold_with_lead(dst_col, dst, (row, op, true), (0..src.height).map(|r| (from, r)))?;
                }
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
    fn check_fold_positions(&self, row: usize, src: &DataFrame, src_row: Option<usize>, ops: &[(usize, usize, CombineOp)]) -> Result<()> {
        if row >= self.height || src_row.is_some_and(|r| r >= src.height) {
            return Err(VolasError::Shape("fold row position out of bounds".into()));
        }
        for (i, &(dst, from, _)) in ops.iter().enumerate() {
            if dst >= self.width() || from >= src.width() || ops[..i].iter().any(|&(previous, _, _)| previous == dst) {
                return Err(VolasError::Shape("fold columns must be in bounds with distinct destinations".into()));
            }
        }
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

/// Whether folding `src` into `dst` cannot fail, so its check is skipped (the live
/// float bar): a float column of the source's own dtype takes every aggregation.
#[inline]
pub(super) fn cannot_fail(dst: &Column, src: &Column) -> bool {
    dst.dtype().is_float() && dst.dtype() == src.dtype()
}

#[cfg(test)]
#[path = "tests_admit.rs"]
mod tests;
