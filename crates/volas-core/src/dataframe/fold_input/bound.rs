//! Borrow-bound aggregate plans. Schema selection and binding are one operation;
//! arithmetic, checkpoint and publication consume those same checked targets.
use super::*;

/// One admitted bar bound to both aggregate snapshots until the operation ends.
/// Dropping a plan publishes nothing; its mutable borrows prevent schema changes.
pub struct CheckpointFold<'f, 'r, 'a, 'p> {
    bar: &'r AdmittedBar<'a, 'p>,
    current: &'f mut DataFrame,
    before: &'f mut DataFrame,
    layouts: &'f (FoldLayout, FoldLayout),
    missing: &'r [bool],
    replace: bool,
}

impl<'a, 'p> AdmittedBar<'a, 'p> {
    /// Select a projected layout and bind it to this exact append destination.
    /// Cache validation is consumed here, rather than repeated by prepare_append.
    pub fn prepare_append_cached<'f, 'r>(&'r self, frame: &'f mut DataFrame, cache: &mut Option<FoldLayout>) -> Result<AdmittedAppend<'f, 'r>> {
        if !cache.as_ref().is_some_and(|p| self.accepts(p) && p.matches(frame)) {
            *cache = Some(self.project(frame)?);
        }
        self.prepare_append_bound(frame, cache.as_ref().unwrap())
    }

    /// Bind both snapshots once. Positions, shape and source identities then flow
    /// to the arithmetic and publication consumers in the returned plan. The
    /// operation owner admits a changed source once, before caching its binding.
    pub fn prepare_checkpoint<'f, 'r>(&'r self, current: &'f mut DataFrame, before: &'f mut DataFrame, cache: &'f mut Option<(FoldLayout, FoldLayout)>, missing: &'r [bool], replace: bool, check_operations: impl FnOnce() -> Result<()>) -> Result<CheckpointFold<'f, 'r, 'a, 'p>> {
        let (source_bound, targets_bound) = cache.as_ref().map_or((false, false), |(a, b)| {
            let source_bound = self.accepts(a) && self.accepts(b);
            (source_bound, source_bound && a.matches(current) && b.matches(before))
        });
        if !source_bound { check_operations()?; }
        if !targets_bound { *cache = Some((self.project(current)?, self.project(before)?)); }
        let layouts = cache.as_ref().unwrap();
        if !layouts.0.complete || !layouts.1.complete || current.height != 1 || before.height != 1 || missing.len() < self.rows.width() { return Err(schema_error()); }
        Ok(CheckpointFold { bar: self, current, before, layouts, missing, replace })
    }

    pub(super) fn check_prefix_bound(&self, frame: &DataFrame, row: usize, prefix: Option<(&DataFrame, &FoldLayout)>) -> Result<()> {
        if self.layout.all_float { return Ok(()); }
        for &(dst, from, op) in &self.layout.ops {
            if !self.layout.source[from].1.is_float() {
                let cells = prefix.map(|(p, l)| (&p.columns[l.ops[from].0], p.height - 1)).into_iter().chain([(&self.rows.columns[from], 0)]);
                frame.check_fold_with_lead(dst, &frame.columns[dst], (row, op, true), cells)?;
            }
        }
        Ok(())
    }

    pub(super) fn publish_bound(&self, frame: &mut DataFrame, reduced: &DataFrame, layout: &FoldLayout) -> Result<()> {
        for &(dst, from, _) in &self.layout.ops {
            frame.check_lead_values(&frame.names[dst], &reduced.columns[layout.ops[from].0], 0, 1)?;
        }
        frame.write_folded_row(reduced, &self.layout.ops)
    }
}

impl CheckpointFold<'_, '_, '_, '_> {
    /// Validate and publish this aggregate's single push or first-bar correction.
    pub fn land(mut self, has_before: bool) -> Result<()> { self.commit(has_before) }

    /// Bind the displayed row once, check its full proposed reduction, then
    /// publish the snapshots and displayed values without rebinding either one.
    pub fn refold(mut self, target: &mut DataFrame, has_before: bool) -> Result<()> {
        self.bar.bind(target, self.bar.layout)?;
        let row = last(target)?;
        let prefix = if !self.replace { Some((&*self.current, &self.layouts.0)) }
            else if has_before { Some((&*self.before, &self.layouts.1)) } else { None };
        self.bar.check_prefix_bound(target, row, prefix)?;
        self.commit(has_before)?;
        // The joint preflight already checked the reduced value against the
        // displayed target (including its historical lead).
        target.write_folded_row(self.current, &self.bar.layout.ops)
    }

    fn commit(&mut self, has_before: bool) -> Result<()> {
        let bar = self.bar;
        let (current_layout, before_layout) = self.layouts;
        if self.replace && !has_before {
            bar.check_into(self.current, self.current, current_layout, 0, true)?;
            return self.current.refold_admitted(0, &bar.rows, &current_layout.ops, false);
        }
        if !self.replace && self.before.lead.is_some() {
            for &(dst, _, _) in &before_layout.ops {
                self.before.check_lead_values(&self.before.names[dst], &self.current.columns[dst], 0, 1)?;
            }
        }
        let destination = if self.replace { &*self.before } else { &*self.current };
        bar.check_into(self.current, destination, current_layout, 0, false)?;
        self.current.fold_checkpoint_admitted(self.before, &bar.rows, &current_layout.ops, self.missing, self.replace, false)
    }
}
