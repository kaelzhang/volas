//! Reusable schema proofs and admitted bars for composed live aggregation.
//! A plan owns schema identities; changing names or dtypes retires an identity.
use super::*;
use crate::{CombineOp, DType};
use std::borrow::Cow;

/// A checked mapping from complete data bars to a destination schema.
/// Fields are private; projections check their destination once, never per bar.
#[derive(Clone, Debug)]
pub struct FoldLayout {
    target: Arc<Vec<String>>,
    source: Arc<Vec<(String, DType)>>,
    ops: Vec<(usize, usize, CombineOp)>,
    complete: bool,
    leads: bool,
    all_float: bool,
}

/// Complete rows admitted once against a reusable fold layout.
pub struct AdmittedBars<'a, 'p> {
    rows: Cow<'a, DataFrame>,
    layout: &'p FoldLayout,
}
impl AdmittedBars<'_, '_> {
    /// Borrow the admitted batch.
    pub fn rows(&self) -> &DataFrame { &self.rows }
    /// Take one immutable bar, inheriting the batch's schema proof.
    pub fn bar(&self, row: usize) -> Result<AdmittedBar<'_, '_>> {
        if row >= self.rows.height { return Err(schema_error()); }
        let rows = if self.rows.height == 1 { Cow::Borrowed(self.rows.as_ref()) }
            else { Cow::Owned(self.rows.slice_data(row, row + 1)) };
        Ok(AdmittedBar { rows, layout: self.layout })
    }
}

/// A complete single bar with the schema required by its fold layout.
/// It cannot be constructed by callers or changed after admission.
///
/// ```compile_fail
/// use std::borrow::Cow;
/// use volas_core::{AdmittedBar, Column, CombineOp, DataFrame, FoldLayout};
/// let frame = DataFrame::new(vec!["x".into()], vec![Column::f64(vec![1.])], None).unwrap();
/// let layout = FoldLayout::new(&frame, |_| CombineOp::Sum).unwrap();
/// let _ = AdmittedBar { rows: Cow::Borrowed(&frame), layout: &layout };
/// ```
pub struct AdmittedBar<'a, 'p> {
    rows: Cow<'a, DataFrame>,
    layout: &'p FoldLayout,
}

impl FoldLayout {
    /// Bind a column aggregation to a frame's data schema. Each data column
    /// appears exactly once, in data-column order; cached columns are excluded.
    pub fn new(frame: &DataFrame, operation: impl Fn(&str) -> CombineOp) -> Result<Self> {
        let mut source = Vec::new();
        let mut ops = Vec::new();
        let mut all_float = true;
        for (dst, (name, column)) in frame.names.iter().zip(&frame.columns).enumerate() {
            if frame.computed.contains_key(name.as_str()) { continue; }
            let op = operation(name);
            let dtype = column.dtype();
            if !op.supports(dtype) { return Err(op.unsupported(dtype)); }
            all_float &= dtype.is_float();
            ops.push((dst, source.len(), op));
            source.push((name.clone(), dtype));
        }
        Ok(Self { all_float, leads: ops.iter().all(|&(dst, from, _)| dst == from), complete: frame.width() == source.len(), target: frame.names.clone(), source: Arc::new(source), ops })
    }

    /// Whether the exact schema this plan checked is still present.
    pub fn matches(&self, frame: &DataFrame) -> bool { Arc::ptr_eq(&self.target, &frame.names) && self.ops.len() + frame.computed.len() == frame.width() }

    /// Check a second destination once (aggregate/checkpoint/open bars).
    /// Its complete column order and dtypes must agree with the data bars.
    pub fn project(&self, frame: &DataFrame) -> Result<Self> {
        let mut ops = Vec::with_capacity(self.ops.len());
        for &(_, from, op) in &self.ops {
            let (name, dtype) = &self.source[from];
            let dst = from;
            if frame.names.get(dst) != Some(name) { return Err(schema_error()); }
            if frame.columns[dst].dtype() != *dtype || frame.computed.contains_key(name.as_str()) { return Err(schema_error()); }
            ops.push((dst, from, op));
        }
        if frame.width() != self.source.len() { return Err(schema_error()); }
        Ok(Self { all_float: self.all_float, leads: true, complete: true, target: frame.names.clone(), source: self.source.clone(), ops })
    }

    /// Reuse or rebuild a cached layout, binding its target exactly once.
    /// The selected layout's fact flows straight into input admission.
    pub fn admit_cached<'a, 'p>(cache: &'p mut Option<Self>, frame: &DataFrame, rows: &'a DataFrame, operation: impl Fn(&str) -> CombineOp) -> Result<AdmittedBars<'a, 'p>> {
        if !cache.as_ref().is_some_and(|p| p.matches(frame)) { *cache = Some(Self::new(frame, operation)?); }
        Ok(AdmittedBars { rows: frame.admit_data_rows(rows)?, layout: cache.as_ref().unwrap() })
    }

    /// Admit schema and values once at the outer boundary. Index compatibility
    /// is checked by the frame owner, while time ordering belongs to the period.
    pub fn admit<'a, 'p>(&'p self, frame: &DataFrame, rows: &'a DataFrame) -> Result<AdmittedBars<'a, 'p>> {
        if !self.matches(frame) { return Err(schema_error()); }
        Ok(AdmittedBars { rows: frame.admit_data_rows(rows)?, layout: self })
    }
}

impl AdmittedBar<'_, '_> {
    /// Immutable admitted values and labels.
    pub fn rows(&self) -> &DataFrame { &self.rows }

    /// Reduce a first bar on an already checked one-row index. Column operations
    /// and source positions come from the layout, rather than a fresh admission.
    pub fn reduced(&self, index: crate::Index) -> Result<AdmittedBar<'static, '_>> {
        if index.len() != 1 { return Err(schema_error()); }
        let mut columns = self.rows.columns.clone();
        for &(_, from, op) in &self.layout.ops {
            columns[from].reduce_into(0, op, &self.rows.columns[from], 0..1)?;
        }
        Ok(AdmittedBar { rows: Cow::Owned(self.rows.same_schema(columns, index, 0)), layout: self.layout })
    }

    /// Take the admitted immutable row as a standalone frame.
    pub fn into_frame(self) -> DataFrame { self.rows.into_owned() }

    /// Append a reduced first bar to its original destination schema. Only the
    /// new label boundary remains; it cannot replace a closed row.
    pub fn append_new(&self, frame: &mut DataFrame) -> Result<()> {
        self.bind(frame, self.layout)?;
        if frame.lead.is_some() {
            for &(dst, from, _) in &self.layout.ops { frame.check_lead_values(&frame.names[dst], &self.rows.columns[from], 0, 1)?; }
        }
        let plan = frame.index.plan_append(&self.rows.index, crate::index::AppendMode::NewRows)?;
        frame.push_rows(&self.rows, self.layout.leads, plan)
    }

    /// The admitted operation for a source column (not a mutable plan).
    pub fn operation(&self, column: usize) -> Option<CombineOp> { self.layout.ops.get(column).map(|&(_, _, op)| op) }

    /// Project this bar's schema proof onto another complete data destination.
    pub fn project(&self, frame: &DataFrame) -> Result<FoldLayout> { self.layout.project(frame) }
    /// Whether a projected layout comes from this bar's admission owner.
    pub fn accepts(&self, layout: &FoldLayout) -> bool { Arc::ptr_eq(&layout.source, &self.layout.source) }

    fn bind(&self, frame: &DataFrame, layout: &FoldLayout) -> Result<()> {
        if !layout.matches(frame) || !Arc::ptr_eq(&layout.source, &self.layout.source) { return Err(schema_error()); }
        Ok(())
    }

    /// Continue the displayed forming row. Only arithmetic remains to check.
    pub fn fold(&self, frame: &mut DataFrame) -> Result<()> {
        self.bind(frame, self.layout)?;
        let row = last(frame)?;
        self.check_into(frame, frame, self.layout, row, false)?;
        frame.fold_forming_row(row, &self.rows, 0, &self.layout.ops, false)
    }

    /// Validate the complete proposed reduction before changing checkpoints.
    /// Prefixes are bound to a projected layout, so no positions are rechecked.
    pub fn check_from(&self, frame: &DataFrame, prefix: Option<(&DataFrame, &FoldLayout)>) -> Result<()> {
        self.bind(frame, self.layout)?;
        let row = last(frame)?;
        if let Some((prefix, layout)) = prefix { self.bind(prefix, layout)?; last(prefix)?; }
        self.check_prefix_bound(frame, row, prefix)
    }

    /// Copy an admitted reduction to the forming row. Historical dtype rules
    /// were checked by `check_from`; this still admits the actual output values.
    pub fn publish(&self, frame: &mut DataFrame, reduced: &DataFrame, layout: &FoldLayout) -> Result<()> {
        self.bind(frame, self.layout)?;
        self.bind(reduced, layout)?;
        last(frame)?;
        if !layout.complete || reduced.height != 1 { return Err(schema_error()); }
        self.publish_bound(frame, reduced, layout)
    }

    /// Fold a one-row aggregate and checkpoint using their checked layouts.
    /// The mask's length is checked before either snapshot is written.
    pub fn checkpoint(&self, current: &mut DataFrame, current_layout: &FoldLayout, before: &mut DataFrame, before_layout: &FoldLayout, missing: &[bool], replace: bool) -> Result<()> {
        self.bind(current, current_layout)?;
        self.bind(before, before_layout)?;
        if !current_layout.complete || !before_layout.complete || current.height != 1 || before.height != 1 || missing.len() < self.rows.width() { return Err(schema_error()); }
        if !replace && before.lead.is_some() {
            for &(dst, _, _) in &before_layout.ops { before.check_lead_values(&before.names[dst], &current.columns[dst], 0, 1)?; }
        }
        let destination = if replace { &*before } else { &*current };
        self.check_into(current, destination, current_layout, 0, false)?;
        current.fold_checkpoint_admitted(before, &self.rows, &current_layout.ops, missing, replace, false)
    }

    /// Reduce a first-bar correction into a one-row aggregate.
    pub fn reduce(&self, frame: &mut DataFrame, layout: &FoldLayout) -> Result<()> {
        self.bind(frame, layout)?;
        if !layout.complete || frame.height != 1 { return Err(schema_error()); }
        self.check_into(frame, frame, layout, 0, true)?;
        frame.refold_admitted(0, &self.rows, &layout.ops, false)
    }

    // Both sides have the layout's admitted dtypes. Choose the necessary value
    // preflight from that immutable fact, never by re-reading the input's tags.
    fn check_into(&self, frame: &DataFrame, destination: &DataFrame, layout: &FoldLayout, row: usize, restart: bool) -> Result<()> {
        if self.layout.all_float { return Ok(()); }
        for &(dst, from, op) in &layout.ops {
            if !self.layout.source[from].1.is_float() {
                frame.check_fold_with_lead(dst, &destination.columns[dst], (row, op, restart), [(&self.rows.columns[from], 0)])?;
            }
        }
        Ok(())
    }

    /// Append to a projected destination without admitting the schema again.
    /// The Index owner checks its single boundary/capacity step before writing.
    pub fn append(&self, frame: &mut DataFrame, layout: &FoldLayout) -> Result<()> {
        self.prepare_append(frame, layout)?.land()
    }

    /// Bind the checked index boundary and schema to the exact destination;
    /// callers can check other aggregates before this infallible-schema commit.
    pub fn prepare_append<'f, 'r>(&'r self, frame: &'f mut DataFrame, layout: &FoldLayout) -> Result<AdmittedAppend<'f, 'r>> {
        self.bind(frame, layout)?;
        self.prepare_append_bound(frame, layout)
    }

    fn prepare_append_bound<'f, 'r>(&'r self, frame: &'f mut DataFrame, layout: &FoldLayout) -> Result<AdmittedAppend<'f, 'r>> {
        if !layout.complete { return Err(schema_error()); }
        if frame.lead.is_some() {
            for (name, col) in frame.names.iter().zip(self.rows.columns()) { frame.check_lead_values(name, col, 0, 1)?; }
        }
        let plan = frame.index.admit_input(&self.rows.index)?.row_plan(0, 0)?;
        frame.prepare_columns(&self.rows, true, plan.appended(), plan.replaces(), None)?;
        Ok(AdmittedAppend { frame, rows: &self.rows, plan: Some(plan) })
    }
}

#[cold]
fn schema_error() -> VolasError { VolasError::Shape("fold admission schema no longer matches its target".into()) }
fn last(frame: &DataFrame) -> Result<usize> { frame.height.checked_sub(1).ok_or_else(schema_error) }

/// An append whose schema and labels have already been admitted. The mutable
/// destination cannot change while a downstream aggregate checks its arithmetic.
pub struct AdmittedAppend<'f, 'r> {
    frame: &'f mut DataFrame,
    rows: &'r DataFrame,
    plan: Option<crate::index::RowPlan<'r>>,
}
impl AdmittedAppend<'_, '_> {
    /// Whether this bar replaces the destination's latest row.
    pub fn replaces(&self) -> bool { self.plan.as_ref().unwrap().replaces() }
    /// Publish the single admitted index plan and its bar.
    pub fn land(mut self) -> Result<()> {
        let result = self.frame.commit_prepared_rows(self.rows, true, self.plan.take().unwrap());
        if result.is_err() { for col in &mut self.frame.columns { col.finish_growth(); } }
        result
    }
}

impl Drop for AdmittedAppend<'_, '_> {
    fn drop(&mut self) { if self.plan.is_some() { for col in &mut self.frame.columns { col.finish_growth(); } } }
}

mod bound;
pub use bound::CheckpointFold;
