//! Validation proofs owned by Index. No caller can fabricate labels or an append
//! plan; construction, contiguous derivation and append publish through this owner.
use super::*;

pub(super) struct ValidatedLabels {
    kind: IndexKind,
    order: LabelOrder,
}

impl ValidatedLabels {
    pub(super) fn check(kind: IndexKind) -> Result<Self> {
        let order = validate_labels(&kind)?;
        Ok(Self { kind, order })
    }

    #[inline]
    pub(super) fn range(start: i64, len: usize) -> Result<Self> {
        validate_range(start, len)?;
        Ok(Self { kind: IndexKind::Range { start, len }, order: LabelOrder::ascending(len) })
    }

    /// A checked Column owns valid Rust storage. Its length is bounded by a
    /// real slice/Vec allocation (at most isize::MAX bytes), so positional i64
    /// labels cannot overflow. No independent arbitrary length enters here.
    pub(super) fn column(column: &crate::Column) -> Self {
        let len = column.len();
        Self { kind: IndexKind::Range { start: 0, len }, order: LabelOrder::ascending(len) }
    }

    pub(super) fn into_parts(self) -> (IndexKind, LabelOrder) { (self.kind, self.order) }
}

/// A contiguous part of an opaque, already validated index. Borrowing its storage
/// preserves uniqueness and datetime order without revalidating either history.
pub(super) struct ValidatedSlice<'a> {
    labels: Labels<'a>,
    order: LabelOrder,
}

enum Labels<'a> {
    Range { start: i64, len: usize, unlabelled: bool },
    Int64(&'a [i64]),
    Datetime(&'a [i64], &'a Tz),
    Str(&'a [String]),
}

impl ValidatedSlice<'_> {
    fn len(&self) -> usize {
        match &self.labels {
            Labels::Range { len, .. } => *len,
            Labels::Int64(v) | Labels::Datetime(v, _) => v.len(),
            Labels::Str(v) => v.len(),
        }
    }

    pub(super) fn to_index(&self, name: Option<String>) -> Index {
        let kind = match &self.labels {
            Labels::Range { start, len, .. } => IndexKind::Range { start: *start, len: *len },
            Labels::Int64(v) => IndexKind::Int64(v.to_vec()),
            Labels::Datetime(v, tz) => IndexKind::Datetime(v.to_vec(), (*tz).clone()),
            Labels::Str(v) => IndexKind::Str(v.to_vec()),
        };
        Index::from_validated(name, ValidatedLabels { kind, order: self.order })
    }
}

#[derive(Clone, Copy)]
pub(crate) enum AppendMode {
    Extend,
    Rows { from: usize, shown: usize },
    NewRows,
}

/// Kind admission borrows its exact target until routing consumes it. Conversion
/// of incoming values does not change these source labels.
pub(crate) struct IndexInput<'target, 'rows> {
    target: &'target Index,
    rows: &'rows Index,
}
/// Checked positional continuation, consumed only by the frame's append owner.
pub(crate) struct RangeStep { len: usize }
pub(crate) struct Replacement { _private: () }

/// The minimal proof for a row operation: replacements have no appended labels
/// or capacity work. Only a genuine appended tail needs the general plan.
pub(crate) enum RowPlan<'a> {
    Range(RangeStep),
    Replacement(Replacement),
    Append(AppendTail<'a>),
}
/// A general row tail, sealed by Index's row router. A replacement carried by
/// this type always has appended rows; zero-growth replacements use their own
/// proof. Other modules cannot wrap an arbitrary generic Index plan as a tail.
#[repr(transparent)]
pub(crate) struct AppendTail<'a> { plan: AppendPlan<'a> }
impl<'a> AppendTail<'a> {
    #[inline]
    pub(crate) fn into_plan(self) -> AppendPlan<'a> { self.plan }
}

impl RowPlan<'_> {
    pub(crate) fn compact_from(&self) -> usize { match self { Self::Append(p) => p.plan.compact_from(), _ => 0 } }
    pub(crate) fn replaces(&self) -> bool { match self { Self::Replacement(_) => true, Self::Append(p) => p.plan.replaces(), _ => false } }
}
impl<'rows> IndexInput<'_, 'rows> {
    pub(crate) fn range_step(&self) -> Result<Option<RangeStep>> {
        let (IndexKind::Range { start, len }, IndexKind::Range { start: 0, len: added }) = (&self.target.kind, &self.rows.kind) else { return Ok(None); };
        let total = len.checked_add(*added).ok_or_else(|| VolasError::Overflow("range index length overflow".into()))?;
        validate_range(*start, total)?;
        Ok(Some(RangeStep { len: total }))
    }

    pub(crate) fn row_plan(self, from: usize, shown: usize) -> Result<RowPlan<'rows>> {
        if let Some(step) = self.range_step()? { return Ok(RowPlan::Range(step)); }
        if self.rows.len() == 1 {
            let replaces = !self.rows.is_unlabelled() && !self.target.is_empty()
                && self.target.same_label(self.target.len() - 1, self.rows, 0);
            if replaces { return Ok(RowPlan::Replacement(Replacement { _private: () })); }
            return self.target.plan_routed(self.rows, AppendMode::Rows { from, shown }, false).map(|plan| RowPlan::Append(AppendTail { plan }));
        }
        self.target.plan_compatible(self.rows, AppendMode::Rows { from, shown }).map(|plan| RowPlan::Append(AppendTail { plan }))
    }

    pub(crate) fn plan(self, mode: AppendMode) -> Result<AppendPlan<'rows>> {
        self.target.plan_compatible(self.rows, mode)
    }
}

/// Proof that the appended portion is disjoint from all retained labels. Only
/// this module constructs one. Frame admission holds it until its single commit;
/// it never re-routes or rechecks individual labels after writing values.
pub(crate) struct AppendPlan<'a> {
    labels: ValidatedSlice<'a>,
    replaces: bool,
    compact: usize,
    next: Option<i64>,
}

impl AppendPlan<'_> {
    pub(crate) fn replaces(&self) -> bool { self.replaces }
    pub(crate) fn compact_from(&self) -> usize { self.compact }
    pub(crate) fn appended(&self) -> usize { self.labels.len() }
}

impl Index {
    pub(crate) fn apply_range_step(&mut self, step: RangeStep) {
        let IndexKind::Range { len, .. } = &mut self.kind else { unreachable!("range admission binds the target kind") }; // LCOV_EXCL_LINE
        *len = step.len;
        self.order = LabelOrder::ascending(step.len);
    }

    pub(super) fn validated_slice(&self, start: usize, end: usize) -> ValidatedSlice<'_> {
        let start = start.min(self.len());
        let end = end.max(start).min(self.len());
        let subset_order = |order: LabelOrder| if end - start <= 1 { LabelOrder::EMPTY } else { order };
        let (labels, order) = match &self.kind {
            IndexKind::Range { start: first, .. } => (Labels::Range {
                start: (*first as i128 + start as i128).min(i64::MAX as i128) as i64,
                len: end - start, unlabelled: *first == 0 && start == 0,
            }, LabelOrder::ascending(end - start)),
            IndexKind::Int64(v) => {
                let labels = &v[start..end];
                let order = if self.order.increasing || self.order.decreasing || (start == 0 && end == self.len()) { self.order } else { LabelOrder::of(labels) };
                (Labels::Int64(labels), subset_order(order))
            }
            IndexKind::Datetime(v, tz) => (Labels::Datetime(&v[start..end], tz), LabelOrder::ascending(end - start)),
            IndexKind::Str(v) => {
                let labels = &v[start..end];
                let order = if self.order.increasing || self.order.decreasing || (start == 0 && end == self.len()) { self.order } else { LabelOrder::of(labels) };
                (Labels::Str(labels), subset_order(order))
            }
        };
        ValidatedSlice { labels, order }
    }

    pub(crate) fn plan_append<'a>(&self, other: &'a Index, mode: AppendMode) -> Result<AppendPlan<'a>> {
        self.admit_input(other)?.plan(mode)
    }

    pub(crate) fn admit_input<'t, 'r>(&'t self, other: &'r Index) -> Result<IndexInput<'t, 'r>> {
        self.check_append(other)?;
        Ok(IndexInput { target: self, rows: other })
    }

    fn plan_compatible<'a>(&self, other: &'a Index, mode: AppendMode) -> Result<AppendPlan<'a>> {
        let replaces = matches!(mode, AppendMode::Rows { .. }) && !other.is_unlabelled()
            && !self.is_empty() && !other.is_empty() && self.same_label(self.len() - 1, other, 0);
        self.plan_routed(other, mode, replaces)
    }

    fn plan_routed<'a>(&self, other: &'a Index, mode: AppendMode, replaces: bool) -> Result<AppendPlan<'a>> {
        let (from, shown) = match mode {
            AppendMode::Rows { from, shown } => (from, shown),
            _ => (0, 0),
        };
        // Checked datetime sources already prove their internal order. Every
        // append mode shares the one cross-boundary comparison below.
        if matches!(mode, AppendMode::NewRows) && !matches!(&other.kind, IndexKind::Datetime(..)) {
            self.check_new_rows(other)?;
        }
        let labels = other.validated_slice(usize::from(replaces), other.len());
        let mut plan = AppendPlan { labels, replaces, compact: 0, next: None };
        let added = plan.appended();
        let mut compact = 0;
        if matches!(self.kind, IndexKind::Range { .. } | IndexKind::Int64(_)) {
            let count = self.len().saturating_add(added);
            if !matches!(&plan.labels.labels, Labels::Range { unlabelled: true, .. })
                && count > isize::MAX as usize / std::mem::size_of::<i64>() {
                return Err(integer_capacity_error(count));
            }
        }
        match &plan.labels.labels {
            Labels::Range { len, unlabelled: true, .. } => {
                match &self.kind {
                    IndexKind::Range { start, len: held } => {
                        let total = held.checked_add(*len).ok_or_else(|| VolasError::Overflow("range index length overflow".into()))?;
                        // The same range invariant as explicit construction.
                        validate_range(*start, total)?;
                        if *len > 0 { plan.next = Some((*start as i128 + *held as i128) as i64); }
                    }
                    IndexKind::Int64(v) => {
                        plan.next = Some(continuation(v, *len)?);
                        if let Some(dup) = self.continuation_clash(*len, from..self.len()) {
                            return Err(if matches!(mode, AppendMode::Extend) { duplicate_label_error("int64") }
                                else { VolasError::Value(format!("cannot append unlabelled rows: the next label {dup} is already in the index (label the rows explicitly)")) });
                        }
                        if from > 0 && self.continuation_clash(*len, 0..from).is_some() { compact = from; }
                        let count = self.len().saturating_add(added);
                        if count > isize::MAX as usize / std::mem::size_of::<i64>() { return Err(integer_capacity_error(count)); }
                    }
                    _ => unreachable!("kind admitted"), // LCOV_EXCL_LINE
                }
            }
            Labels::Datetime(v, _) => {
                if let (IndexKind::Datetime(held, _), Some(first)) = (&self.kind, v.first()) {
                    if let Some(last) = held.last().filter(|&last| last >= first) {
                        return Err(match mode {
                            AppendMode::Extend => datetime_order_error(),
                            AppendMode::NewRows => VolasError::Value(format!("cannot start a new row labelled {first}: it does not follow the last row (labelled {last})")),
                            AppendMode::Rows { .. } => VolasError::Value(format!("cannot append a row labelled {first}: a datetime index only increases (only the last row can be replaced)")),
                        });
                    }
                }
            }
            Labels::Int64(v) => {
                for &label in *v { note_overlap(self.find_integer(label), &mut compact, from, shown, mode, "int64", || label.to_string())?; }
            }
            Labels::Range { start, len, .. } => {
                for k in 0..*len {
                    let label = *start + k as i64;
                    note_overlap(self.find_integer(label), &mut compact, from, shown, mode, "int64", || label.to_string())?;
                }
            }
            Labels::Str(v) => {
                let IndexKind::Str(held) = &self.kind else { unreachable!() }; // LCOV_EXCL_LINE
                for label in *v { note_overlap(self.order.find(held, label), &mut compact, from, shown, mode, "str", || format!("{label:?}"))?; }
            }
        }
        plan.compact = if replaces { compact.min(self.len() - 1) } else { compact };
        Ok(plan)
    }

    fn find_integer(&self, label: i64) -> Option<usize> {
        match &self.kind {
            IndexKind::Range { start, len } => usize::try_from(label as i128 - *start as i128).ok().filter(|i| i < len),
            IndexKind::Int64(v) => self.order.find(v, &label),
            _ => unreachable!("integer kind admission precedes label routing"), // LCOV_EXCL_LINE
        }
    }

    fn same_label(&self, at: usize, other: &Index, row: usize) -> bool {
        match (&self.kind, &other.kind) {
            (IndexKind::Str(a), IndexKind::Str(b)) => a[at] == b[row],
            _ => self.label_at(at) == other.label_at(row),
        }
    }

    fn check_new_rows(&self, other: &Index) -> Result<()> {
        let mut previous = self.len().checked_sub(1).map(|i| self.label_at(i));
        for k in 0..other.len() {
            let label = other.label_at(k);
            if let Some(p) = &previous {
                if !matches!((p, &label), (Label::I64(a), Label::I64(b)) if a < b) {
                    return Err(VolasError::Value(format!("cannot start a new row labelled {}: it does not follow the last row (labelled {})", label_text(&label), label_text(p))));
                }
            }
            previous = Some(label);
        }
        Ok(())
    }

    /// Prepare a replacement-plus-append before the frame writes values. Its
    /// first row repeats the last, so its checked tail has explicit labels.
    /// Changing Range storage here preserves exactly the same validated labels.
    pub(crate) fn prepare_replacement(&mut self, plan: &AppendPlan<'_>) -> Result<()> {
        if let IndexKind::Range { start, len } = self.kind {
            self.kind = IndexKind::Int64(materialize_range(start, len, plan.appended())?);
        }
        if let IndexKind::Int64(values) = &mut self.kind {
            reserve_integer_labels(values, plan.appended())?;
        }
        Ok(())
    }

    /// Consume the proof without a second per-label lookup. The frame owner may
    /// first compact the proved-dropped prefix; all remaining labels are disjoint.
    pub(crate) fn apply_append(&mut self, plan: AppendPlan<'_>) -> Result<()> {
        let added = plan.appended();
        let order = self.order;
        match (&mut self.kind, plan.labels.labels) {
            (IndexKind::Range { start, len }, Labels::Range { unlabelled: true, .. }) => {
                if *len == 0 && added > 0 { *start = plan.next.unwrap(); }
                *len += added;
                self.order = LabelOrder { increasing: true, decreasing: *len <= 1 };
            }
            (IndexKind::Datetime(a, _), Labels::Datetime(b, _)) => {
                a.extend_from_slice(b);
                self.order = LabelOrder { increasing: true, decreasing: a.len() <= 1 };
            }
            (IndexKind::Int64(a), Labels::Int64(b)) => {
                self.order = joined_order(order, a.last(), plan.labels.order, b.first(), a.is_empty(), b.is_empty());
                a.extend_from_slice(b);
            }
            (IndexKind::Str(a), Labels::Str(b)) => {
                self.order = joined_order(order, a.last(), plan.labels.order, b.first(), a.is_empty(), b.is_empty());
                a.extend(b.iter().cloned());
            }
            (IndexKind::Int64(a), Labels::Range { start, unlabelled, .. }) => {
                let next = if unlabelled { plan.next.unwrap() } else { start };
                reserve_integer_labels(a, added)?;
                let added_order = LabelOrder { increasing: true, decreasing: added <= 1 };
                self.order = joined_order(order, a.last(), added_order, (added > 0).then_some(&next), a.is_empty(), added == 0);
                a.extend((0..added).map(|k| next + k as i64));
            }
            (IndexKind::Range { start, len }, labels) => {
                let last = (*len > 0).then(|| *start + (*len - 1) as i64);
                let mut values = materialize_range(*start, *len, added)?;
                let first = match labels {
                    Labels::Int64(v) => { values.extend_from_slice(v); v.first().copied() }
                    Labels::Range { start, .. } => { values.extend((0..added).map(|k| start + k as i64)); (added > 0).then_some(start) }
                    _ => unreachable!("kind admitted"), // LCOV_EXCL_LINE
                };
                self.order = joined_order(order, last.as_ref(), plan.labels.order, first.as_ref(), *len == 0, added == 0);
                self.kind = IndexKind::Int64(values);
            }
            _ => unreachable!("kind admitted"), // LCOV_EXCL_LINE
        }
        Ok(())
    }
}

fn note_overlap(position: Option<usize>, compact: &mut usize, from: usize, shown: usize, mode: AppendMode, kind: &str, text: impl FnOnce() -> String) -> Result<()> {
    let Some(position) = position else { return Ok(()) };
    if position < from { *compact = from; return Ok(()); }
    if matches!(mode, AppendMode::Extend) { return Err(duplicate_label_error(kind)); }
    Err(VolasError::Value(if position < shown {
        format!("cannot append label {}: an earlier row with that label is still retained in the window's hidden margin (max_lookback)", text())
    } else {
        format!("cannot append a row labelled {}: an earlier row has that label (only the last row can be replaced)", text())
    }))
}

fn joined_order<T: Ord>(a: LabelOrder, last: Option<&T>, b: LabelOrder, first: Option<&T>, a_empty: bool, b_empty: bool) -> LabelOrder {
    if a_empty { return b; }
    if b_empty { return a; }
    LabelOrder {
        increasing: a.increasing && b.increasing && last <= first,
        decreasing: a.decreasing && b.decreasing && last >= first,
    }
}

fn label_text(label: &Label) -> String {
    match label { Label::I64(v) => v.to_string(), Label::Str(v) => format!("{v:?}") }
}
