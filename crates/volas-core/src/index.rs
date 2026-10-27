//! Index: the row labels shared by a frame and the series drawn from it.

use crate::column::{Column, ColumnView};
use crate::datetime::NAT;
use crate::error::{Result, VolasError};
use crate::tz::Tz;
mod admission;
pub(crate) use admission::{AppendMode, AppendPlan, AppendTail, IndexInput, RowPlan};
use admission::ValidatedLabels;

/// Row labels plus an optional name.
///
/// `name` is index-wide metadata (pandas `Index.name`): it is recorded by
/// `set_index` (from the source column), restored by `reset_index`, and shown in
/// a frame's / series' repr. It is `None` for an unnamed index (the default
/// `Range` index, a freshly built datetime index, …).
///
/// `kind` is the label storage. A `Datetime` kind carries its own [`Tz`]:
/// storage is always UTC epoch-ns, but the tz governs how those instants render
/// and how bare-string / day-bucket matching maps to wall-clock time (see
/// [`crate::tz`]). Both the name and a datetime kind's tz ride with the shared
/// `Arc<Index>`, so a frame and every series drawn from it agree on them for free.
///
/// The index also owns the **order** of its labels ([`LabelOrder`]): computed
/// once when the labels are built and kept current by [`extend`](Self::extend)
/// in O(appended labels), so a label slice never rescans the labels to learn it.
/// The labels are reachable only through [`kind`](Self::kind) (read-only), which
/// keeps the stored order true to them.
#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    /// Optional index name (pandas `Index.name`); `None` when unnamed.
    name: Option<String>,
    /// The label storage / kind.
    kind: IndexKind,
    /// Whether the labels are non-decreasing / non-increasing.
    order: LabelOrder,
}

/// The monotonicity of an index's labels (pandas `is_monotonic_increasing` /
/// `is_monotonic_decreasing`): numeric labels by value, string labels by code
/// point. An empty or single-label index is both. A label slice searches by this order — see
/// [`Index::label_slice`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LabelOrder {
    increasing: bool,
    decreasing: bool,
}

impl LabelOrder {
    /// The order of no labels: both directions hold.
    const EMPTY: LabelOrder = LabelOrder { increasing: true, decreasing: true };
    /// Neither direction holds.
    const NONE: LabelOrder = LabelOrder { increasing: false, decreasing: false };

    /// This order after `added` labels follow a sequence whose last label is
    /// `last`: one comparison per appended label, and none once neither direction
    /// holds any more.
    fn extended<T: Ord>(mut self, last: Option<&T>, added: &[T]) -> LabelOrder {
        let mut prev = last;
        for x in added {
            if self == LabelOrder::NONE {
                break;
            }
            if let Some(p) = prev {
                self.increasing &= p <= x;
                self.decreasing &= p >= x;
            }
            prev = Some(x);
        }
        self
    }

    /// The order of `labels` (one pass).
    fn of<T: Ord>(labels: &[T]) -> LabelOrder {
        LabelOrder::EMPTY.extended(None, labels)
    }

    /// A proven strictly increasing sequence (empty/singleton also decreases).
    fn ascending(len: usize) -> Self { Self { increasing: true, decreasing: len <= 1 } }

    /// The first position of `x` in `labels`, labels of this order: a binary search
    /// when they are monotonic — O(log n) — a scan otherwise.
    fn find<T: Ord>(self, labels: &[T], x: &T) -> Option<usize> {
        let at = if self.increasing {
            labels.partition_point(|l| l < x)
        } else if self.decreasing {
            labels.partition_point(|l| l > x)
        } else {
            return labels.iter().position(|l| l == x);
        };
        (labels.get(at) == Some(x)).then_some(at)
    }
}

/// The label storage backing an [`Index`].
///
/// Defaults to an implicit `0..n` range; a `Datetime` kind is the common OHLCV
/// case (i64 nanoseconds since the Unix epoch); a `Str` kind (pandas
/// object/string index) supports symbol-keyed lookup.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexKind {
    /// Implicit consecutive integer labels `start..start + len` — the positions of a
    /// series (`0..n`), kept implicit when a slice starts later (pandas `RangeIndex`).
    Range {
        /// The first label.
        start: i64,
        /// The number of labels.
        len: usize,
    },
    /// Explicit integer labels.
    Int64(Vec<i64>),
    /// Datetime labels as i64 nanoseconds since the Unix epoch, with a display /
    /// matching timezone (UTC by default).
    Datetime(Vec<i64>, Tz),
    /// String labels (pandas object/string index).
    Str(Vec<String>),
}

impl IndexKind {
    /// Materialize the numeric labels as `i64`. Numeric kinds only — a string
    /// kind has no i64 labels (its callers guard against it).
    fn to_i64_labels(&self) -> Vec<i64> {
        match self {
            IndexKind::Range { start, len } => (0..*len).map(|i| *start + i as i64).collect(),
            IndexKind::Int64(v) => v.clone(),
            IndexKind::Datetime(v, _) => v.clone(),
            IndexKind::Str(_) => unreachable!("string indexes have no i64 labels"), // LCOV_EXCL_LINE
        }
    }
}

/// A single row-index label: an integer / datetime (`I64`, ns) or a string
/// (`Str`). It decouples label *lookup* from the index's storage so the same
/// `.loc` / `.at` / `.loc[a:b]` / `drop` paths serve every index kind.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Label {
    /// An integer or datetime-ns label.
    I64(i64),
    /// A string label.
    Str(String),
}

impl Label {
    /// The i64 payload, if this is an `I64` label.
    pub fn as_i64(&self) -> Option<i64> {
        if let Label::I64(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// The string payload, if this is a `Str` label.
    pub fn as_str(&self) -> Option<&str> {
        if let Label::Str(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }
}

/// One bound of a label slice (`.loc[lo:hi]`): the label, and whether it was
/// written as **text** (a string parsed into the label). The text fact matters on
/// a datetime index only, where pandas reads a string bound as a partial-date
/// value rather than an exact label (see [`Index::label_slice`]).
#[derive(Clone, Debug, PartialEq)]
pub struct SliceBound {
    /// The bound's label.
    pub label: Label,
    /// Whether the label was parsed from a string.
    pub from_text: bool,
}

impl SliceBound {
    /// A bound given as a label value (an int, a timestamp, …).
    pub fn exact(label: Label) -> SliceBound {
        SliceBound { label, from_text: false }
    }

    /// A bound given as a string, parsed into `label`.
    pub fn text(label: Label) -> SliceBound {
        SliceBound { label, from_text: true }
    }
}

impl Index {
    /// Construct an index after validating every label invariant.
    /// Datetime labels are present and strictly increasing; integer and string
    /// labels are unique. All explicit-label construction uses this owner.
    #[inline]
    pub fn new(name: Option<String>, kind: IndexKind) -> Result<Index> {
        ValidatedLabels::check(kind).map(|labels| Self::from_validated(name, labels))
    }

    /// Construct a checked, unnamed positional index without moving a generic
    /// label descriptor across the admission boundary.
    #[inline]
    pub fn positional(len: usize) -> Result<Self> {
        ValidatedLabels::range(0, len).map(|labels| Self::from_validated(None, labels))
    }

    /// Inherit the representable length of admitted physical column storage.
    #[inline]
    pub(crate) fn for_column(column: &Column) -> Self {
        Self::for_extent(column.extent())
    }

    /// Consume an already admitted physical extent without reading storage again.
    #[inline]
    pub(crate) fn for_extent(extent: crate::column::ColumnExtent) -> Self {
        Self::from_validated(None, ValidatedLabels::extent(extent))
    }

    /// Whether this index carries the checked strictly increasing datetime fact.
    pub fn is_strict_datetime(&self) -> bool {
        matches!(self.kind, IndexKind::Datetime(..)) && self.order.increasing
    }

    /// Check the opaque construction fact in O(1), including internal admissions.
    #[inline]
    pub(crate) fn validate(&self) -> Result<()> {
        if matches!(self.kind, IndexKind::Datetime(..)) && !self.order.increasing {
            return Err(datetime_order_error());
        }
        Ok(())
    }

    /// The only publication point for owned labels and their validation proof.
    fn from_validated(name: Option<String>, labels: ValidatedLabels) -> Self {
        let (kind, order) = labels.into_parts();
        Self { name, kind, order }
    }

    /// The label storage / kind.
    pub fn kind(&self) -> &IndexKind {
        &self.kind
    }

    /// The index name, if any (pandas `Index.name`).
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Return this index with its name set (builder; consumes `self`).
    pub fn with_name(mut self, name: Option<String>) -> Index {
        self.name = name;
        self
    }

    /// Build an unnamed index from a column (for `set_index`): a `Datetime`
    /// column becomes a `DatetimeIndex`, an `I64` column an `Int64Index`, a `Str`
    /// column a string index. Float / bool columns are not valid labels.
    pub fn from_column(col: &Column) -> Result<Index> {
        Index::from_column_tz(col, Tz::Naive)
    }

    /// Build an unnamed index from a column, tagging a `Datetime` column with
    /// `tz` (otherwise the tz is ignored).
    ///
    /// An `I64` / `Str` column carrying `volas.NA` is rejected: an int/str index
    /// has no missing-label representation, so building one would silently turn
    /// each NA into its physical placeholder (`0` / `""`) — an ordinary label a
    /// `.loc` lookup can match (C2/C4). Datetime labels cannot contain `NaT`.
    pub fn from_column_tz(col: &Column, tz: Tz) -> Result<Index> {
        let kind = match col.view() {
            ColumnView::Datetime(v) => IndexKind::Datetime(v.to_vec(), tz),
            ColumnView::I64(v, _) => {
                require_no_missing_labels(col, "int64")?;
                IndexKind::Int64(v.to_vec())
            }
            ColumnView::Str(v, _) => {
                require_no_missing_labels(col, "str")?;
                IndexKind::Str(v.to_vec())
            }
            _ => {
                return Err(VolasError::DType(format!(
                    "cannot use a {} column as an index (only datetime / int64 / string)",
                    col.dtype()
                )))
            }
        };
        Index::new(None, kind)
    }

    /// The timezone of a `Datetime` index ([`Tz::Naive`] for every other kind).
    pub fn tz(&self) -> Tz {
        match &self.kind {
            IndexKind::Datetime(_, tz) => tz.clone(),
            _ => Tz::Naive,
        }
    }

    /// Return this index with its timezone set (no-op for a non-datetime index);
    /// the name is preserved.
    pub fn with_tz(mut self, tz: Tz) -> Index {
        if let IndexKind::Datetime(_, cur) = &mut self.kind {
            *cur = tz;
        }
        self
    }

    /// Number of labels.
    pub fn len(&self) -> usize {
        match &self.kind {
            IndexKind::Range { len, .. } => *len,
            IndexKind::Int64(v) => v.len(),
            IndexKind::Datetime(v, _) => v.len(),
            IndexKind::Str(v) => v.len(),
        }
    }

    /// The label at position `i` (for membership tests like `drop`).
    pub fn label_at(&self, i: usize) -> Label {
        #[cfg(test)]
        tests_cost::record_label_read();
        match &self.kind {
            IndexKind::Range { start, .. } => Label::I64(start + i as i64),
            IndexKind::Int64(v) => Label::I64(v[i]),
            IndexKind::Datetime(v, _) => Label::I64(v[i]),
            IndexKind::Str(v) => Label::Str(v[i].clone()),
        }
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Materialize the numeric labels as `i64`. Numeric indexes only — string
    /// indexes are handled by their own paths (`label_slice` / `append` guard).
    pub fn to_i64_labels(&self) -> Vec<i64> {
        self.kind.to_i64_labels()
    }

    /// A `[start, end)` slice, clamped to the available rows like frame slices.
    /// The name and a datetime timezone are preserved.
    pub fn slice(&self, start: usize, end: usize) -> Index {
        self.validated_slice(start, end).to_index(self.name.clone())
    }

    /// Gather the given positions (the name and a datetime tz are preserved).
    pub fn take(&self, idx: &[usize]) -> Result<Index> {
        if idx.iter().any(|&i| i >= self.len()) {
            return Err(VolasError::Shape("index selection position out of bounds".into()));
        }
        let kind = match &self.kind {
            IndexKind::Range { start, .. } => {
                IndexKind::Int64(idx.iter().map(|&i| start + i as i64).collect())
            }
            IndexKind::Int64(v) => IndexKind::Int64(idx.iter().map(|&i| v[i]).collect()),
            IndexKind::Datetime(v, tz) => {
                IndexKind::Datetime(idx.iter().map(|&i| v[i]).collect(), tz.clone())
            }
            IndexKind::Str(v) => IndexKind::Str(idx.iter().map(|&i| v[i].clone()).collect()),
        };
        Index::new(self.name.clone(), kind)
    }

    /// Value equality of the **labels** (pandas `.equals` index semantics): a
    /// `RangeIndex` equals the same integer labels materialized as `Int64`, but an
    /// integer index never equals a datetime or string index. The index *name* and
    /// a datetime *tz* (display metadata) are ignored.
    pub fn label_eq(&self, other: &Index) -> bool {
        use IndexKind::*;
        match (&self.kind, &other.kind) {
            (Range { .. } | Int64(_), Range { .. } | Int64(_)) => {
                self.to_i64_labels() == other.to_i64_labels()
            }
            (Datetime(a, _), Datetime(b, _)) => a == b,
            (Str(a), Str(b)) => a == b,
            _ => false, // different label kinds are never equal
        }
    }

    /// The alignment rule for a positional binary operation between two labelled
    /// operands (arithmetic, comparison, `where` / `mask`, pairwise windows): the
    /// labels must be equal ([`label_eq`](Self::label_eq): the same instants, the
    /// index name and display zone ignored), and a tz-naive datetime index never
    /// aligns with a tz-aware one (a wall clock is not an instant). volas pairs by
    /// position and never re-aligns by label, so anything else is an error.
    pub fn require_aligned(&self, other: &Index) -> Result<()> {
        if let (IndexKind::Datetime(_, a), IndexKind::Datetime(_, b)) = (&self.kind, &other.kind) {
            if a.is_aware() != b.is_aware() {
                return Err(VolasError::Value(
                    "cannot align a tz-naive index with a tz-aware index; tz_localize the \
                     naive one first — one built from an aware frame's `.index` (its UTC \
                     instants) takes `.tz_localize('UTC').tz_convert(zone)`"
                        .into(),
                ));
            }
        }
        if self.label_eq(other) {
            Ok(())
        } else {
            Err(VolasError::Value(
                "operands have different indexes; volas aligns by position, not by label — \
                 reindex or slice them to a common index first"
                    .into(),
            ))
        }
    }

    /// The positions that sort the index by label (`sort_index`). Numeric kinds
    /// sort numerically, a string index lexicographically.
    pub fn argsort(&self, ascending: bool) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.len()).collect();
        let cmp_dir = |o: std::cmp::Ordering| if ascending { o } else { o.reverse() };
        match &self.kind {
            IndexKind::Str(v) => idx.sort_by(|&a, &b| cmp_dir(v[a].cmp(&v[b]))),
            IndexKind::Int64(v) | IndexKind::Datetime(v, _) => idx.sort_by(|&a, &b| cmp_dir(v[a].cmp(&v[b]))),
            IndexKind::Range { .. } => {
                if !ascending {
                    idx.reverse();
                }
            }
        }
        idx
    }

    /// Materialize the labels as a [`Column`] (for `reset_index`).
    pub fn to_column(&self) -> Column {
        match &self.kind {
            IndexKind::Range { start, len } => Column::i64((0..*len).map(|i| *start + i as i64).collect()),
            IndexKind::Int64(v) => Column::i64(v.clone()),
            IndexKind::Datetime(v, _) => Column::datetime(v.clone()),
            IndexKind::Str(v) => Column::str(v.clone()),
        }
    }

    /// The append rule for an incoming row set's labels (contract #17) — the one
    /// predicate every append path checks before it writes anything:
    /// - a datetime index takes datetime rows only, of the same awareness (a tz-naive
    ///   wall clock is not an instant); aware rows of another zone are the same
    ///   instants, so they are kept in this index's zone. A `NaT` label is rejected:
    ///   a missing instant has no place in the order;
    /// - an integer index takes integer rows, or unlabelled (`RangeIndex`) rows, which
    ///   continue the labels from the last one (`last + 1`, ...) — checked, and never
    ///   onto a label the index already holds;
    /// - a `RangeIndex` takes unlabelled or integer rows; a string index string rows.
    ///
    /// The order of explicit labels (a datetime index only increases, an integer or
    /// string one stays unique) is checked by this owner's append admission plan.
    pub fn check_append(&self, other: &Index) -> Result<()> {
        use IndexKind::*;
        if let Range { start, len } = &other.kind {
            if *start != 0 && *len > isize::MAX as usize / std::mem::size_of::<i64>() { return Err(integer_capacity_error(*len)); }
        }
        match (&self.kind, &other.kind) {
            (Range { .. }, Range { .. } | Int64(_))
            | (Int64(_), Int64(_))
            | (Str(_), Str(_)) => Ok(()),
            (Datetime(_, a), Datetime(_, b)) if a.is_aware() == b.is_aware() => Ok(()),
            (Int64(a), Range { start: 0, len: k }) => continuation(a, *k).map(drop),
            (Int64(_), Range { .. }) => Ok(()),
            (mine, theirs) => Err(VolasError::Value(format!(
                "cannot append rows with a {} to a frame with a {}{}",
                if matches!(&other.kind, Range { start, .. } if *start != 0) { "integer index" } else { other.kind_text() },
                self.kind_text(),
                match (mine, theirs) {
                    (Datetime(..), Range { start: 0, .. }) => " — label the rows with their timestamps",
                    (Datetime(_, a), Datetime(..)) if a.is_aware() => {
                        " — tz_localize the rows to their zone first"
                    }
                    (Datetime(..), Datetime(..)) => " — tz_localize the frame first",
                    _ => "",
                }
            ))),
        }
    }

    /// Whether these are rows without labels of their own: a `RangeIndex` from 0 (a
    /// fresh frame's positions), which an append continues from the frame's last label.
    /// A `RangeIndex` starting later (a slice) carries its labels.
    pub fn is_unlabelled(&self) -> bool {
        matches!(self.kind, IndexKind::Range { start: 0, .. })
    }

    /// How the index kind reads in a message.
    fn kind_text(&self) -> &'static str {
        match &self.kind {
            IndexKind::Range { .. } => "RangeIndex",
            IndexKind::Int64(_) => "integer index",
            IndexKind::Datetime(_, tz) if tz.is_aware() => "tz-aware DatetimeIndex",
            IndexKind::Datetime(..) => "tz-naive DatetimeIndex",
            IndexKind::Str(_) => "string index",
        }
    }

    /// Extend in place by the labels of `other` — the amortized-O(1) live append; the
    /// growing index keeps its own name (so appending an unnamed bar to a named index
    /// does not drop the name) and a datetime index its zone. `other` must pass
    /// [`check_append`](Self::check_append), which this checks first, so on an error
    /// the index is unchanged.
    pub fn extend(&mut self, other: &Index) -> Result<()> {
        let plan = self.plan_append(other, AppendMode::Extend)?;
        self.apply_append(plan)
    }

    /// The first label among the positions `rows` (a window's retained rows) that `k`
    /// unlabelled rows appended to this integer index would take again — they continue
    /// from the last label, so with increasing labels none can (O(1)); otherwise the
    /// rows are scanned. `None` for any other index kind (a `RangeIndex` continues its
    /// own positions) or when the continuation overflows ([`Index::check_append`]
    /// reports that).
    pub fn continuation_clash(&self, k: usize, rows: std::ops::Range<usize>) -> Option<i64> {
        let IndexKind::Int64(labels) = &self.kind else { return None };
        if self.order.increasing || k == 0 {
            return None;
        }
        let next = continuation(labels, k).ok()?;
        let end = next + (k - 1) as i64;
        labels.get(rows)?.iter().find(|&&l| l >= next && l <= end).copied()
    }

    /// Position of the first label equal to `label` among the positions `rows` (a
    /// window's retained or visible rows). `None` when none there has it, or the kinds
    /// differ. Searched by the stored [`LabelOrder`]: O(log rows) over ordered labels
    /// (a time series), a scan over unordered ones — never a copy of the labels.
    pub fn position_in(&self, label: &Label, rows: std::ops::Range<usize>) -> Option<usize> {
        let first = rows.start;
        match (&self.kind, label) {
            (IndexKind::Range { start, len }, Label::I64(v)) => {
                let at = usize::try_from(v.checked_sub(*start)?).ok()?;
                return (at < *len).then_some(at).filter(|p| rows.contains(p));
            }
            (IndexKind::Int64(vs) | IndexKind::Datetime(vs, _), Label::I64(v)) => self.order.find(vs.get(rows)?, v),
            (IndexKind::Str(vs), Label::Str(s)) => self.order.find(vs.get(rows)?, s),
            _ => None,
        }
        .map(|p| p + first)
    }

    /// Position of the first label exactly equal to `label` ([`position_in`](Self::position_in)
    /// every row). Returns `None` if the label's kind does not match the index's kind.
    pub fn position_of(&self, label: &Label) -> Option<usize> {
        self.position_in(label, 0..self.len())
    }

    /// `[start, end)` positions of the inclusive label slice `[lo, hi]` (pandas
    /// `.loc[lo:hi]`); either bound may be `None` for open-ended. Numeric labels
    /// compare by value, string labels by code point.
    ///
    /// The search follows the index's [`LabelOrder`] (pandas `slice_locs`):
    /// - non-decreasing labels: the rows with `lo <= label <= hi` (a bound need not
    ///   be a label of the index);
    /// - otherwise non-increasing labels: the rows with `lo >= label >= hi`;
    /// - neither (int / str labels need only be unique): each bound is located by
    ///   the position of its one equal label, and the slice runs between the two.
    ///   A bound that is not a label, or that labels several rows, is a
    ///   [`VolasError::Key`] — an ordered comparison would select rows outside the
    ///   requested labels.
    ///
    /// A bound of the other label kind (a string against a numeric index, or the
    /// reverse) is a [`VolasError::DType`].
    pub fn label_slice(
        &self,
        lo: Option<&SliceBound>,
        hi: Option<&SliceBound>,
    ) -> Result<(usize, usize)> {
        match &self.kind {
            IndexKind::Range { start: first, len } => {
                let (lo, hi) = (i64_bound(lo, "a range")?, i64_bound(hi, "a range")?);
                // the labels are first..first + len, at positions 0..len
                let clamp = |x: i128| (x - *first as i128).clamp(0, *len as i128) as usize;
                let start = lo.map_or(0, |l| clamp(l as i128));
                let end = hi.map_or(*len, |h| clamp(h as i128 + 1));
                Ok((start, end.max(start)))
            }
            IndexKind::Int64(v) => {
                let (lo, hi) = (i64_bound(lo, "an int64")?, i64_bound(hi, "an int64")?);
                slice_bounds(v, self.order, lo.as_ref(), hi.as_ref(), |x| x.to_string())
            }
            IndexKind::Datetime(v, _) => {
                let (lo, hi) = (i64_bound(lo, "a datetime")?, i64_bound(hi, "a datetime")?);
                Ok(increasing_bounds(v, lo.as_ref(), hi.as_ref()))
            }
            IndexKind::Str(v) => {
                let (lo, hi) = (str_bound(lo)?, str_bound(hi)?);
                slice_bounds(v, self.order, lo, hi, |x| format!("'{x}'"))
            }
        }
    }
}

/// A numeric slice bound: an integer / datetime label, or `None` (open-ended).
/// A string label against the numeric index of `kind` is a `DType` error.
fn i64_bound(bound: Option<&SliceBound>, kind: &str) -> Result<Option<i64>> {
    match bound.map(|b| &b.label) {
        None => Ok(None),
        Some(Label::I64(x)) => Ok(Some(*x)),
        Some(Label::Str(s)) => Err(VolasError::DType(format!(
            "cannot do slice indexing on {kind} index with these indexers [{s}] of type str"
        ))),
    }
}

/// A string slice bound, or `None` (open-ended). An integer / datetime label
/// against a string index is a `DType` error.
fn str_bound(bound: Option<&SliceBound>) -> Result<Option<&String>> {
    match bound.map(|b| &b.label) {
        None => Ok(None),
        Some(Label::Str(s)) => Ok(Some(s)),
        Some(Label::I64(x)) => Err(VolasError::DType(format!(
            "cannot do slice indexing on a string index with these indexers [{x}] of type int"
        ))),
    }
}

/// Search present increasing labels; datetime construction already proves this
/// order, while the generic integer/string search selects it from its fact.
fn increasing_bounds<T: Ord>(labels: &[T], lo: Option<&T>, hi: Option<&T>) -> (usize, usize) {
    let start = lo.map_or(0, |lo| labels.partition_point(|x| x < lo));
    let end = hi.map_or(labels.len(), |hi| labels.partition_point(|x| x <= hi));
    (start, end.max(start))
}

/// The `[start, end)` positions of the inclusive slice `[lo, hi]` over `labels`
/// whose monotonicity is `order` (the rule of [`Index::label_slice`]). An ordered
/// index is binary-searched; an unordered one locates each bound exactly.
/// `render` spells a label in a `Key` error.
fn slice_bounds<T: Ord>(
    labels: &[T],
    order: LabelOrder,
    lo: Option<&T>,
    hi: Option<&T>,
    render: impl Fn(&T) -> String,
) -> Result<(usize, usize)> {
    let n = labels.len();
    let (start, end) = if order.increasing {
        return Ok(increasing_bounds(labels, lo, hi));
    } else if order.decreasing {
        (
            lo.map_or(0, |lo| labels.partition_point(|x| x > lo)),
            hi.map_or(n, |hi| labels.partition_point(|x| x >= hi)),
        )
    } else {
        let exact = |bound: &T, side: &str| -> Result<usize> {
            match labels.iter().position(|x| x == bound) {
                Some(p) => Ok(p),
                None => Err(VolasError::Key(format!(
                    "Cannot get {side} slice bound for non-monotonic index with a missing \
                     label {}. Either sort the index or specify an existing label.",
                    render(bound)
                ))),
            }
        };
        (
            lo.map_or(Ok(0), |lo| exact(lo, "left"))?,
            hi.map_or(Ok(n), |hi| exact(hi, "right").map(|p| p + 1))?,
        )
    };
    Ok((start, end.max(start)))
}

/// The single label-invariant owner, used by every explicit constructor.
fn validate_labels(kind: &IndexKind) -> Result<LabelOrder> {
    match kind {
        IndexKind::Datetime(labels, _) => {
            if labels.first() == Some(&NAT) || labels.windows(2).any(|w| w[1] == NAT || w[0] >= w[1]) {
                return invalid_datetime_labels(labels);
            }
            Ok(LabelOrder::ascending(labels.len()))
        }
        IndexKind::Int64(labels) => require_unique_labels(labels, "int64"),
        IndexKind::Str(labels) => require_unique_labels(labels, "str"),
        IndexKind::Range { start, len } => {
            validate_range(*start, *len)?;
            Ok(LabelOrder::ascending(*len))
        }
    }
}

/// The range invariant shared by explicit construction and append plans.
#[inline]
fn validate_range(start: i64, len: usize) -> Result<()> {
    if len > 0 && i64::try_from(len - 1).ok().and_then(|span| start.checked_add(span)).is_none() {
        return Err(range_overflow());
    }
    Ok(())
}

/// Keep allocation and error construction out of positional factories and steps.
#[cold]
#[inline(never)]
fn range_overflow() -> VolasError { VolasError::Overflow("range index labels overflow int64".into()) }

/// Detailed failures stay off the valid, single-pass datetime construction path.
#[cold]
#[inline(never)]
fn invalid_datetime_labels(labels: &[i64]) -> Result<LabelOrder> {
    if labels.contains(&NAT) {
        return Err(VolasError::Value(
            "time_frame requires a DatetimeIndex without NaT; drop or fill the missing index timestamps first".into()));
    }
    require_unique_labels(labels, "datetime")?;
    Err(datetime_order_error())
}

fn datetime_order_error() -> VolasError {
    VolasError::Value("time_frame requires a monotonic (time-sorted) DatetimeIndex; sort the bars by time first".into())
}

fn duplicate_label_error(kind: &str) -> VolasError {
    VolasError::Value(format!("cannot use a {kind} column with duplicate labels as an index (label access assumes unique labels)"))
}

/// Reject building an index from a column that carries `volas.NA`: with no
/// missing-label representation for `int64` / `str`, the NA's physical
/// placeholder (`0` / `""`) would become an ordinary, lookup-matchable label.
fn require_no_missing_labels(col: &Column, kind: &str) -> Result<()> {
    if col.null_count() > 0 {
        return Err(VolasError::Value(format!(
            "cannot use a {kind} column containing volas.NA as an index (a missing \
             label has no {kind} representation); drop or fill the NA rows first"
        )));
    }
    Ok(())
}

/// Label access assumes unique labels (F34, decision 1B): a duplicate-label
/// index makes `.loc[label]` ill-defined (one row or many?), so it is rejected
/// at creation — the same creation-time guard as the NA-label rule above.
fn require_unique_labels<T: std::hash::Hash + Ord>(labels: &[T], kind: &str) -> Result<LabelOrder> {
    if labels.windows(2).all(|w| w[0] < w[1]) {
        // The same pass proves uniqueness and order for an increasing index.
        return Ok(LabelOrder { increasing: true, decreasing: labels.len() <= 1 });
    }
    unique_unordered_labels(labels, kind)
}

/// Strictly decreasing labels also prove uniqueness without a set. Only
/// unordered labels need hashing; carry their order in that same pass.
#[cold]
#[inline(never)]
fn unique_unordered_labels<T: std::hash::Hash + Ord>(labels: &[T], kind: &str) -> Result<LabelOrder> {
    if labels.windows(2).all(|w| w[0] > w[1]) {
        return Ok(LabelOrder { increasing: labels.len() <= 1, decreasing: true });
    }
    let mut seen = std::collections::HashSet::with_capacity(labels.len());
    let mut order = LabelOrder::EMPTY;
    let mut previous = None;
    for label in labels {
        if !seen.insert(label) {
            return Err(duplicate_label_error(kind));
        }
        order = order.extended(previous, std::slice::from_ref(label));
        previous = Some(label);
    }
    Ok(order)
}

/// Convert a checked implicit Range to explicit labels, reserving both its
/// labels and any incoming tail before writing the local candidate.
fn materialize_range(start: i64, len: usize, tail: usize) -> Result<Vec<i64>> {
    let mut labels = Vec::new();
    let needed = len.saturating_add(tail);
    labels.try_reserve(crate::capacity::growth_capacity(len, needed))
        .map_err(|_| integer_capacity_error(needed))?;
    labels.extend((0..len).map(|i| start + i as i64));
    Ok(labels)
}

/// Reserve before writing labels or order. A valid nominal Range can be wider
/// than an explicit Vec can represent; capacity overflow is a normal error.
fn reserve_integer_labels(labels: &mut Vec<i64>, count: usize) -> Result<()> {
    labels.try_reserve(count).map_err(|_| integer_capacity_error(count))
}

#[cold]
fn integer_capacity_error(count: usize) -> VolasError {
    VolasError::Overflow(format!(
        "cannot extend the integer index by {count} labels: allocation capacity exceeded"
    ))
}

/// The first label `k` unlabelled rows take after the integer labels `labels`
/// (`last + 1`, or 0 after none), with checked arithmetic over the continued labels
/// `next ..= next + k - 1`. Whether one of them is already held is the frame's rule
/// over the rows it retains ([`Index::continuation_clash`]). No rows (`k == 0`) take
/// no label.
fn continuation(labels: &[i64], k: usize) -> Result<i64> {
    let overflow = || {
        VolasError::Overflow(format!(
            "cannot continue the integer index past {} with {k} unlabelled rows",
            labels.last().copied().unwrap_or(0)
        ))
    };
    let next = match labels.last() {
        Some(&last) if k > 0 => last.checked_add(1).ok_or_else(overflow)?,
        Some(&last) => return Ok(last), // no rows: nothing continues
        None => 0,
    };
    let Some(span) = k.checked_sub(1) else { return Ok(next) };
    i64::try_from(span).ok().and_then(|s| next.checked_add(s)).ok_or_else(overflow)?;
    Ok(next)
}

#[cfg(test)]
mod tests_order;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_invariants;

#[cfg(test)]
mod tests_cost;
