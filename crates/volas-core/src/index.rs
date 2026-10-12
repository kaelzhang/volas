//! Index: the row labels shared by a frame and the series drawn from it.

use crate::column::Column;
use crate::datetime::{format_ns_offset, NAT};
use crate::error::{Result, VolasError};
use crate::tz::Tz;

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
    pub name: Option<String>,
    /// The label storage / kind.
    kind: IndexKind,
    /// Whether the labels are non-decreasing / non-increasing.
    order: LabelOrder,
}

/// The monotonicity of an index's labels (pandas `is_monotonic_increasing` /
/// `is_monotonic_decreasing`): numeric labels by value, string labels by code
/// point. An empty or constant index is both; a datetime index holding `NaT` is
/// neither (pandas). A label slice searches by this order — see
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

    /// The order after datetime labels `added` follow `last`: a `NaT` makes the
    /// labels unordered (pandas).
    fn extended_datetime(self, last: Option<&i64>, added: &[i64]) -> LabelOrder {
        if added.contains(&NAT) {
            LabelOrder::NONE
        } else {
            self.extended(last, added)
        }
    }

    /// The order of the labels of `kind`.
    fn of_kind(kind: &IndexKind) -> LabelOrder {
        match kind {
            IndexKind::Range(n) => LabelOrder { increasing: true, decreasing: *n <= 1 },
            IndexKind::Int64(v) => LabelOrder::of(v),
            IndexKind::Datetime(v, _) => LabelOrder::EMPTY.extended_datetime(None, v),
            IndexKind::Str(v) => LabelOrder::of(v),
        }
    }
}

/// The label storage backing an [`Index`].
///
/// Defaults to an implicit `0..n` range; a `Datetime` kind is the common OHLCV
/// case (i64 nanoseconds since the Unix epoch); a `Str` kind (pandas
/// object/string index) supports symbol-keyed lookup.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexKind {
    /// Implicit `0..n` integer labels.
    Range(usize),
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
            IndexKind::Range(n) => (0..*n as i64).collect(),
            IndexKind::Int64(v) => v.clone(),
            IndexKind::Datetime(v, _) => v.clone(),
            IndexKind::Str(_) => unreachable!("string indexes have no i64 labels"), // LCOV_EXCL_LINE
        }
    }
}

/// A single row-index label: an integer / datetime (`I64`, ns) or a string
/// (`Str`). It decouples label *lookup* from the index's storage so the same
/// `.loc` / `.at` / `.loc[a:b]` / `drop` paths serve every index kind.
#[derive(Clone, Debug, PartialEq)]
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
    /// An index over `kind`'s labels with `name`, its label order computed once.
    fn new(name: Option<String>, kind: IndexKind) -> Index {
        let order = LabelOrder::of_kind(&kind);
        Index { name, kind, order }
    }

    /// An unnamed index over `kind`'s labels.
    fn unnamed(kind: IndexKind) -> Index {
        Index::new(None, kind)
    }

    /// An unnamed implicit `0..n` range index.
    pub fn range(n: usize) -> Index {
        Index::unnamed(IndexKind::Range(n))
    }

    /// An unnamed explicit integer index.
    pub fn int64(labels: Vec<i64>) -> Index {
        Index::unnamed(IndexKind::Int64(labels))
    }

    /// An unnamed datetime index (UTC-ns `labels`, tagged with `tz`).
    pub fn datetime(labels: Vec<i64>, tz: Tz) -> Index {
        Index::unnamed(IndexKind::Datetime(labels, tz))
    }

    /// An unnamed string index.
    pub fn str(labels: Vec<String>) -> Index {
        Index::unnamed(IndexKind::Str(labels))
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
    /// `.loc` lookup can match (C2/C4). Datetime is the one nullable index kind:
    /// `NaT` is a physical sentinel inside the label vector itself, rendered as
    /// `NaT` and sorted last.
    pub fn from_column_tz(col: &Column, tz: Tz) -> Result<Index> {
        let kind = match col {
            // Datetime is exempt from the unique-label rule (F34 refinement):
            // real market data legitimately carries duplicate timestamps (resent
            // forming bars, multiple ticks per ts, NaT batches) and cumulate /
            // sort own the dedup semantics. int64/str labels stay strictly unique.
            Column::Datetime(v) => IndexKind::Datetime(v.to_vec(), tz),
            Column::I64(v, _) => {
                require_no_missing_labels(col, "int64")?;
                require_unique_labels(v, "int64")?;
                IndexKind::Int64(v.to_vec())
            }
            Column::Str(v, _) => {
                require_no_missing_labels(col, "str")?;
                let labels = v.to_vec();
                require_unique_labels(&labels, "str")?;
                IndexKind::Str(labels)
            }
            other => {
                return Err(VolasError::DType(format!(
                    "cannot use a {} column as an index (only datetime / int64 / string)",
                    other.dtype()
                )))
            }
        };
        Ok(Index::unnamed(kind))
    }

    /// The timezone of a `Datetime` index ([`Tz::Naive`] for every other kind).
    pub fn tz(&self) -> Tz {
        match &self.kind {
            IndexKind::Datetime(_, tz) => *tz,
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
            IndexKind::Range(n) => *n,
            IndexKind::Int64(v) => v.len(),
            IndexKind::Datetime(v, _) => v.len(),
            IndexKind::Str(v) => v.len(),
        }
    }

    /// The label at position `i` (for membership tests like `drop`).
    pub fn label_at(&self, i: usize) -> Label {
        match &self.kind {
            IndexKind::Range(_) => Label::I64(i as i64),
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

    /// A `[start, end)` slice (the name and a datetime tz are preserved).
    pub fn slice(&self, start: usize, end: usize) -> Index {
        let kind = match &self.kind {
            // A range starting at 0 stays an (implicit) range; a non-zero start
            // cannot be expressed as `Range`, so it materializes the actual labels
            // (`start..end`) — e.g. `tail(2)` of `0..7` keeps labels `[5, 6]`, not
            // a reset `[0, 1]`, matching pandas.
            IndexKind::Range(_) if start == 0 => IndexKind::Range(end),
            IndexKind::Range(_) => IndexKind::Int64((start as i64..end as i64).collect()),
            IndexKind::Int64(v) => IndexKind::Int64(v[start..end].to_vec()),
            IndexKind::Datetime(v, tz) => IndexKind::Datetime(v[start..end].to_vec(), *tz),
            IndexKind::Str(v) => IndexKind::Str(v[start..end].to_vec()),
        };
        Index::new(self.name.clone(), kind)
    }

    /// Gather the given positions (the name and a datetime tz are preserved).
    pub fn take(&self, idx: &[usize]) -> Index {
        let kind = match &self.kind {
            IndexKind::Range(_) => IndexKind::Int64(idx.iter().map(|&i| i as i64).collect()),
            IndexKind::Int64(v) => IndexKind::Int64(idx.iter().map(|&i| v[i]).collect()),
            IndexKind::Datetime(v, tz) => {
                IndexKind::Datetime(idx.iter().map(|&i| v[i]).collect(), *tz)
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
            (Range(_) | Int64(_), Range(_) | Int64(_)) => {
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
                     naive one first"
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
            // a datetime index sinks a NaT label (i64::MIN) to the end in both
            // directions — matching sort_values / pandas na_position='last' (V9) —
            // rather than sorting it as the smallest i64, which put NaT first.
            IndexKind::Datetime(v, _) => idx.sort_by(|&a, &b| {
                use std::cmp::Ordering::*;
                match (v[a] == i64::MIN, v[b] == i64::MIN) {
                    (true, true) => Equal,
                    (true, false) => Greater,
                    (false, true) => Less,
                    (false, false) => cmp_dir(v[a].cmp(&v[b])),
                }
            }),
            _ => {
                let labels = self.to_i64_labels();
                idx.sort_by(|&a, &b| cmp_dir(labels[a].cmp(&labels[b])));
            }
        }
        idx
    }

    /// Materialize the labels as a [`Column`] (for `reset_index`).
    pub fn to_column(&self) -> Column {
        match &self.kind {
            IndexKind::Range(n) => Column::i64((0..*n as i64).collect()),
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
    /// string one stays unique) is the frame's append rule, decided per row there.
    pub fn check_append(&self, other: &Index) -> Result<()> {
        use IndexKind::*;
        if let Datetime(b, _) = &other.kind {
            if b.contains(&i64::MIN) {
                return Err(VolasError::Value(
                    "cannot append a row labelled NaT: a missing instant has no place in \
                     the index (drop it or supply a real timestamp)"
                        .into(),
                ));
            }
        }
        match (&self.kind, &other.kind) {
            (Range(_), Range(_) | Int64(_))
            | (Int64(_), Int64(_))
            | (Str(_), Str(_)) => Ok(()),
            (Datetime(_, a), Datetime(_, b)) if a.is_aware() == b.is_aware() => Ok(()),
            (Int64(a), Range(k)) => continuation(a, self.order.increasing, *k).map(drop),
            (mine, theirs) => Err(VolasError::Value(format!(
                "cannot append rows with a {} to a frame with a {}{}",
                other.kind_text(),
                self.kind_text(),
                match (mine, theirs) {
                    (Datetime(..), Range(_)) => " — label the rows with their timestamps",
                    (Datetime(_, a), Datetime(..)) if a.is_aware() => {
                        " — tz_localize the rows to their zone first"
                    }
                    (Datetime(..), Datetime(..)) => " — tz_localize the frame first",
                    _ => "",
                }
            ))),
        }
    }

    /// How the index kind reads in a message.
    fn kind_text(&self) -> &'static str {
        match &self.kind {
            IndexKind::Range(_) => "RangeIndex",
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
        use IndexKind::*;
        self.check_append(other)?;
        let order = self.order;
        match (&mut self.kind, &other.kind) {
            (Range(a), Range(b)) => {
                *a += b;
                self.order = LabelOrder { increasing: true, decreasing: *a <= 1 };
            }
            (Datetime(a, _), Datetime(b, _)) => {
                self.order = order.extended_datetime(a.last(), b);
                a.extend_from_slice(b);
            }
            (Int64(a), Int64(b)) => {
                self.order = order.extended(a.last(), b);
                a.extend_from_slice(b);
            }
            (Str(a), Str(b)) => {
                self.order = order.extended(a.last(), b);
                a.extend(b.iter().cloned());
            }
            // An implicit positional (Range) index appended to explicit Int64 labels
            // continues the sequence (last + 1, ...), mirroring Range+Range's auto-
            // increment — so a sliced / compacted frame (whose Range collapsed to
            // absolute Int64 labels) keeps assigning monotonic labels to appended bars
            // instead of restarting at the bar's own 0. The bare-RangeIndex bar is the
            // "no explicit label, take the next row position" signal; honoring it keeps
            // a windowed (rolling-compacted) RangeIndex frame's labels absolute and
            // gap-free, matching pandas' last-M view of a RangeIndex.
            (Int64(a), Range(bn)) => {
                let next = continuation(a, order.increasing, *bn)?;
                let start = a.len();
                a.extend((0..*bn as i64).map(|k| next + k));
                let (head, added) = a.split_at(start);
                self.order = order.extended(head.last(), added);
            }
            // explicit integer labels after implicit positions: the positions become labels
            (slot @ Range(_), Int64(_)) => {
                let mut labels = slot.to_i64_labels();
                labels.extend(other.to_i64_labels());
                *slot = Int64(labels);
                self.order = LabelOrder::of_kind(slot);
            }
            _ => unreachable!("rejected by check_append"), // LCOV_EXCL_LINE
        }
        Ok(())
    }

    /// Position of the first label exactly equal to `label`. Returns `None` if
    /// the label's kind does not match the index's kind.
    pub fn position_of(&self, label: &Label) -> Option<usize> {
        match (&self.kind, label) {
            (IndexKind::Range(n), Label::I64(v)) => {
                if *v >= 0 && (*v as usize) < *n {
                    Some(*v as usize)
                } else {
                    None
                }
            }
            (IndexKind::Int64(vs), Label::I64(v)) => vs.iter().position(|x| x == v),
            (IndexKind::Datetime(vs, _), Label::I64(v)) => vs.iter().position(|x| x == v),
            (IndexKind::Str(vs), Label::Str(s)) => vs.iter().position(|x| x == s),
            _ => None,
        }
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
    /// A datetime index that is not non-decreasing (decreasing, unordered, or
    /// holding `NaT`) rejects a bound written as text with a [`VolasError::Key`]:
    /// pandas reads such a bound as a partial-date value and selects every row it
    /// covers, which may be non-contiguous and which this exact-label search cannot
    /// reproduce, so it fails closed instead of selecting different rows.
    ///
    /// A bound of the other label kind (a string against a numeric index, or the
    /// reverse) is a [`VolasError::DType`].
    pub fn label_slice(
        &self,
        lo: Option<&SliceBound>,
        hi: Option<&SliceBound>,
    ) -> Result<(usize, usize)> {
        match &self.kind {
            IndexKind::Range(n) => {
                let (lo, hi) = (i64_bound(lo, "a range")?, i64_bound(hi, "a range")?);
                // the labels are the positions 0..n
                let clamp = |x: i64| x.clamp(0, *n as i64) as usize;
                let start = lo.map_or(0, clamp);
                let end = hi.map_or(*n, |h| clamp(h.saturating_add(1)));
                Ok((start, end.max(start)))
            }
            IndexKind::Int64(v) => {
                let (lo, hi) = (i64_bound(lo, "an int64")?, i64_bound(hi, "an int64")?);
                slice_bounds(v, self.order, lo.as_ref(), hi.as_ref(), |x| x.to_string())
            }
            IndexKind::Datetime(v, tz) => {
                let text = |b: Option<&SliceBound>| b.is_some_and(|b| b.from_text);
                if !self.order.increasing && (text(lo) || text(hi)) {
                    return Err(VolasError::Key(
                        "Value based partial slicing on non-monotonic DatetimeIndexes with \
                         non-existing keys is not allowed."
                            .into(),
                    ));
                }
                let (lo, hi) = (i64_bound(lo, "a datetime")?, i64_bound(hi, "a datetime")?);
                slice_bounds(v, self.order, lo.as_ref(), hi.as_ref(), |&x| {
                    format_ns_offset(x, *tz)
                })
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
        (
            lo.map_or(0, |lo| labels.partition_point(|x| x < lo)),
            hi.map_or(n, |hi| labels.partition_point(|x| x <= hi)),
        )
    } else if order.decreasing {
        (
            lo.map_or(0, |lo| labels.partition_point(|x| x > lo)),
            hi.map_or(n, |hi| labels.partition_point(|x| x >= hi)),
        )
    } else {
        let exact = |bound: &T, side: &str| -> Result<usize> {
            let mut hits = labels.iter().enumerate().filter(|(_, x)| *x == bound);
            match (hits.next(), hits.next()) {
                (Some((p, _)), None) => Ok(p),
                (None, _) => Err(VolasError::Key(format!(
                    "Cannot get {side} slice bound for non-monotonic index with a missing \
                     label {}. Either sort the index or specify an existing label.",
                    render(bound)
                ))),
                (Some(_), Some(_)) => Err(VolasError::Key(format!(
                    "Cannot get {side} slice bound for non-unique label: {}",
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
fn require_unique_labels<T: std::hash::Hash + Eq>(labels: &[T], kind: &str) -> Result<()> {
    let mut seen = std::collections::HashSet::with_capacity(labels.len());
    for l in labels {
        if !seen.insert(l) {
            return Err(VolasError::Value(format!(
                "cannot use a {kind} column with duplicate labels as an index \
                 (label access assumes unique labels)"
            )));
        }
    }
    Ok(())
}

/// The first label `k` unlabelled rows take after the integer labels `labels`
/// (`last + 1`, or 0 after none): checked arithmetic over the continued labels `next ..=
/// next + k - 1`, none of which may already be in `labels` (an integer index stays
/// unique). With `increasing` labels the last is the largest, so none can repeat (O(1));
/// otherwise they are looked up. No rows (`k == 0`) take no label.
fn continuation(labels: &[i64], increasing: bool, k: usize) -> Result<i64> {
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
    let end = i64::try_from(span).ok().and_then(|s| next.checked_add(s)).ok_or_else(overflow)?;
    if !increasing {
        if let Some(dup) = labels.iter().find(|&&l| l >= next && l <= end) {
            return Err(VolasError::Value(format!(
                "cannot append unlabelled rows: the next label {dup} is already in the index \
                 (label the rows explicitly)"
            )));
        }
    }
    Ok(next)
}

#[cfg(test)]
mod tests_order;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_datetime_and_int_columns() {
        assert_eq!(
            Index::from_column(&Column::datetime(vec![5, 6])).unwrap(),
            Index::datetime(vec![5, 6], Tz::Naive)
        );
        assert_eq!(
            Index::from_column(&Column::i64(vec![1, 2])).unwrap(),
            Index::int64(vec![1, 2])
        );
    }

    #[test]
    fn from_unsupported_column_errors() {
        // float / bool columns are not valid labels (string is — see below)
        assert!(Index::from_column(&Column::f64(vec![1.0])).is_err());
        assert!(Index::from_column(&Column::bool(vec![true])).is_err());
    }

    #[test]
    fn is_empty_labels_and_position_of() {
        assert!(Index::range(0).is_empty());
        assert!(!Index::range(3).is_empty());

        assert_eq!(Index::range(3).to_i64_labels(), vec![0, 1, 2]);
        assert_eq!(Index::int64(vec![5, 6]).to_i64_labels(), vec![5, 6]);
        assert_eq!(
            Index::datetime(vec![10, 20], Tz::Utc).to_i64_labels(),
            vec![10, 20]
        );

        let i64 = Label::I64;
        assert_eq!(Index::range(5).position_of(&i64(3)), Some(3));
        assert_eq!(Index::range(5).position_of(&i64(9)), None);
        assert_eq!(Index::range(5).position_of(&i64(-1)), None);
        assert_eq!(Index::int64(vec![10, 20, 30]).position_of(&i64(20)), Some(1));
        assert_eq!(Index::int64(vec![10, 20]).position_of(&i64(99)), None);
        assert_eq!(
            Index::datetime(vec![100, 200], Tz::Utc).position_of(&i64(200)),
            Some(1)
        );

        // take() on an Int64 index gathers the labels at those positions
        assert_eq!(
            Index::int64(vec![10, 20, 30]).take(&[2, 0]),
            Index::int64(vec![30, 10])
        );
    }

    fn str_index(labels: &[&str]) -> Index {
        Index::str(labels.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn string_index_construction_and_ops() {
        // a string column becomes a string index
        let ix = Index::from_column(&Column::str(vec!["a".into(), "b".into()])).unwrap();
        assert_eq!(ix, str_index(&["a", "b"]));

        let ix = str_index(&["a", "b", "c", "d"]);
        assert_eq!(ix.len(), 4);
        assert!(!ix.is_empty());
        assert_eq!(ix.label_at(2), Label::Str("c".into()));
        assert_eq!(ix.slice(1, 3), str_index(&["b", "c"]));
        assert_eq!(ix.take(&[3, 0]), str_index(&["d", "a"]));
    }

    #[test]
    fn string_index_lookup_and_slice() {
        let ix = str_index(&["aa", "bb", "cc", "dd"]);
        // exact lookup, kind-matched
        assert_eq!(ix.position_of(&Label::Str("cc".into())), Some(2));
        assert_eq!(ix.position_of(&Label::Str("zz".into())), None);
        // a numeric label against a string index never matches
        assert_eq!(ix.position_of(&Label::I64(1)), None);
        // lexicographic slice [bb, cc]
        let lo = Label::Str("bb".into());
        let hi = Label::Str("cc".into());
        let (lo, hi) = (SliceBound::text(lo), SliceBound::text(hi));
        assert_eq!(ix.label_slice(Some(&lo), Some(&hi)).unwrap(), (1, 3));
        // open-ended upper bound: [cc, end)
        assert_eq!(ix.label_slice(Some(&lo), None).unwrap(), (1, 4));
    }

    #[test]
    fn append_rule_per_kind() {
        let ny = Tz::parse("America/New_York").unwrap();
        let aware = Index::datetime(vec![1, 2], Tz::Utc);
        let naive = Index::datetime(vec![1, 2], Tz::Naive);
        let msg = |a: &Index, b: &Index| a.check_append(b).unwrap_err().to_string();
        // the same instants in another zone are fine; awareness must match
        assert!(aware.check_append(&Index::datetime(vec![3], ny)).is_ok());
        assert!(msg(&aware, &Index::datetime(vec![3], Tz::Naive)).contains("tz_localize the rows"));
        assert!(msg(&naive, &Index::datetime(vec![3], Tz::Utc)).contains("tz_localize the frame"));
        // NaT has no place in the order, on every kind of target
        assert!(msg(&aware, &Index::datetime(vec![i64::MIN], Tz::Utc)).contains("NaT"));
        assert!(msg(&Index::range(0), &Index::datetime(vec![i64::MIN], Tz::Utc)).contains("NaT"));
        // an empty RangeIndex is still a RangeIndex
        assert!(msg(&Index::range(0), &aware).contains("RangeIndex"));
        // a datetime index takes datetime rows only, and the reverse
        assert!(msg(&aware, &Index::range(1)).contains("label the rows with their timestamps"));
        assert!(msg(&aware, &Index::int64(vec![3])).contains("integer index"));
        assert!(msg(&Index::int64(vec![1]), &aware).contains("tz-aware DatetimeIndex"));
        assert!(msg(&Index::range(1), &naive).contains("tz-naive DatetimeIndex"));
        // a string index takes string rows only, and the reverse
        assert!(msg(&str_index(&["x"]), &Index::range(1)).contains("string index"));
        assert!(msg(&Index::range(1), &str_index(&["x"])).contains("RangeIndex"));
        assert!(str_index(&["x"]).check_append(&str_index(&["y"])).is_ok());
        assert!(Index::range(0).check_append(&Index::int64(vec![4])).is_ok());
    }

    #[test]
    fn unlabelled_rows_continue_unique_checked_integer_labels() {
        // continuing onto a label already present (the labels need not be sorted)
        let err = Index::int64(vec![1, 0]).check_append(&Index::range(1)).unwrap_err();
        assert!(err.to_string().contains("next label 1 is already in the index"), "{err}");
        // continuing past i64::MAX overflows instead of wrapping
        let mut ix = Index::int64(vec![i64::MAX]);
        let err = ix.extend(&Index::range(1)).unwrap_err();
        assert!(matches!(err, VolasError::Overflow(_)), "{err}");
        assert_eq!(ix, Index::int64(vec![i64::MAX]));
        let err = Index::int64(vec![i64::MAX - 1]).check_append(&Index::range(2)).unwrap_err();
        assert!(matches!(err, VolasError::Overflow(_)), "{err}");
        // the last continued label may be i64::MAX itself; no rows continue nothing
        let mut top = Index::int64(vec![i64::MAX - 1]);
        top.extend(&Index::range(1)).unwrap();
        assert_eq!(top, Index::int64(vec![i64::MAX - 1, i64::MAX]));
        top.extend(&Index::range(0)).unwrap();
        assert_eq!(top.len(), 2);
        // an empty Int64 index continues from 0
        assert!(Index::int64(vec![]).check_append(&Index::range(3)).is_ok());
        // the largest label follows every extension, so a repeat is still found after
        // the labels stop increasing
        let mut ix = Index::range(2);
        ix.extend(&Index::int64(vec![9, 5])).unwrap(); // [0, 1, 9, 5]: 9 is the largest
        assert!(ix.check_append(&Index::range(4)).unwrap_err().to_string().contains("9"));
        ix.extend(&Index::range(3)).unwrap(); // continues 6, 7, 8
        assert_eq!(ix, Index::int64(vec![0, 1, 9, 5, 6, 7, 8]));
        assert!(ix.check_append(&Index::range(1)).is_err()); // 9 again
        ix.extend(&Index::int64(vec![20])).unwrap();
        ix.extend(&Index::range(1)).unwrap(); // 21: past the largest, no scan needed
        assert_eq!(ix.to_i64_labels().last(), Some(&21));
        // derived indexes keep their own largest label
        assert!(ix.slice(0, 4).check_append(&Index::range(4)).unwrap_err().to_string().contains("9"));
        assert!(ix.take(&[0, 1]).check_append(&Index::range(1)).is_ok());
    }

    #[test]
    fn extend_grows_in_place_per_kind() {
        // same-kind grows the buffer in place (the live append hot path)
        let mut r = Index::range(3);
        r.extend(&Index::range(2)).unwrap();
        assert_eq!(r, Index::range(5));

        let mut d = Index::datetime(vec![1, 2], Tz::Utc);
        d.extend(&Index::datetime(vec![3], Tz::Utc)).unwrap();
        assert_eq!(d, Index::datetime(vec![1, 2, 3], Tz::Utc));

        let mut s = str_index(&["a", "b"]);
        s.extend(&str_index(&["c"])).unwrap();
        assert_eq!(s, str_index(&["a", "b", "c"]));

        // a rejected extension leaves the index unchanged
        let mut keep = Index::datetime(vec![1], Tz::Utc);
        assert!(keep.extend(&Index::range(1)).is_err());
        assert_eq!(keep, Index::datetime(vec![1], Tz::Utc));

        // explicit integer labels after implicit positions: the positions become labels
        let mut m = Index::range(2);
        m.extend(&Index::int64(vec![5, 6])).unwrap();
        assert_eq!(m, Index::int64(vec![0, 1, 5, 6]));

        // an implicit Range appended to explicit Int64 labels CONTINUES the sequence
        // (last + 1, ...) rather than restarting at the Range's own 0 — the
        // sliced/compacted-frame + bare-bar append path (windowed rolling buffers).
        let mut c = Index::int64(vec![44, 45, 46]);
        c.extend(&Index::range(2)).unwrap();
        assert_eq!(c, Index::int64(vec![44, 45, 46, 47, 48]));
        // continuing from an empty Int64 starts at 0.
        let mut e = Index::int64(vec![]);
        e.extend(&Index::range(2)).unwrap();
        assert_eq!(e, Index::int64(vec![0, 1]));

        // mixing string with numeric is an error, either way
        assert!(str_index(&["x"]).extend(&Index::range(1)).is_err());
        assert!(Index::range(1).extend(&str_index(&["x"])).is_err());
    }

    #[test]
    fn alignment_is_equal_labels_with_matching_awareness() {
        let ny = Tz::parse("America/New_York").unwrap();
        let named = Index::datetime(vec![1, 2], Tz::Utc).with_name(Some("t".into()));
        // same instants: the name and the display zone do not matter
        assert!(named.require_aligned(&Index::datetime(vec![1, 2], ny)).is_ok());
        assert!(Index::range(2).require_aligned(&Index::int64(vec![0, 1])).is_ok());
        // naive vs aware never aligns, even over the same stored values
        let err = Index::datetime(vec![1, 2], Tz::Naive)
            .require_aligned(&named)
            .unwrap_err();
        assert!(err.to_string().contains("tz-naive"), "{err}");
        let err = Index::str(vec!["a".into()]).require_aligned(&Index::str(vec!["b".into()]));
        assert!(err.unwrap_err().to_string().contains("different indexes"));
    }

    #[test]
    fn name_set_and_propagates_through_ops() {
        let ix = Index::datetime(vec![1, 2, 3], Tz::Utc).with_name(Some("date".into()));
        assert_eq!(ix.name(), Some("date"));
        // an unnamed index reports None
        assert_eq!(Index::range(3).name(), None);
        // the name rides through identity-preserving ops
        assert_eq!(ix.slice(0, 2).name(), Some("date"));
        assert_eq!(ix.take(&[2, 0]).name(), Some("date"));
        assert_eq!(ix.clone().with_tz(Tz::Offset(28800)).name(), Some("date"));
        // extend keeps the growing index's name
        let mut g = ix.clone();
        g.extend(&Index::datetime(vec![4], Tz::Utc)).unwrap();
        assert_eq!(g.name(), Some("date"));
        // with_name(None) clears it
        assert_eq!(ix.with_name(None).name(), None);
    }

    #[test]
    fn label_accessors_and_numeric_label_at() {
        // accessors return None on the other variant
        assert_eq!(Label::I64(5).as_i64(), Some(5));
        assert_eq!(Label::I64(5).as_str(), None);
        assert_eq!(Label::Str("x".into()).as_str(), Some("x"));
        assert_eq!(Label::Str("x".into()).as_i64(), None);
        // label_at over the numeric index kinds
        assert_eq!(Index::range(3).label_at(2), Label::I64(2));
        assert_eq!(Index::int64(vec![10, 20]).label_at(1), Label::I64(20));
        assert_eq!(
            Index::datetime(vec![100, 200], Tz::Utc).label_at(0),
            Label::I64(100)
        );
    }

    #[test]
    fn index_kind_branch_coverage() {
        // tz() / with_tz() are no-ops on a non-datetime index.
        assert_eq!(Index::range(3).tz(), Tz::Naive);
        assert!(matches!(
            Index::range(3).with_tz(Tz::Utc).kind,
            IndexKind::Range(3)
        ));
        // slice over the non-range kinds.
        assert_eq!(
            Index::int64(vec![1, 2, 3]).slice(0, 2),
            Index::int64(vec![1, 2])
        );
        assert!(matches!(
            Index::datetime(vec![1, 2], Tz::Utc).slice(0, 1).kind,
            IndexKind::Datetime(_, _)
        ));
        assert_eq!(
            Index::str(vec!["a".into(), "b".into()]).slice(1, 2),
            Index::str(vec!["b".into()])
        );
        // argsort lexicographically over a string index.
        assert_eq!(
            Index::str(vec!["b".into(), "a".into()]).argsort(true),
            vec![1, 0]
        );
        // to_column over every kind.
        assert_eq!(Index::range(2).to_column().len(), 2);
        assert_eq!(Index::datetime(vec![5], Tz::Utc).to_column().len(), 1);
        assert_eq!(Index::str(vec!["x".into()]).to_column().len(), 1);
    }

    #[test]
    fn label_eq_value_semantics() {
        // a RangeIndex equals the same integer labels materialized as Int64
        assert!(Index::range(3).label_eq(&Index::int64(vec![0, 1, 2])));
        assert!(!Index::range(3).label_eq(&Index::int64(vec![0, 1, 9])));
        // datetime and string indexes compare by value
        assert!(Index::datetime(vec![1, 2], Tz::Utc).label_eq(&Index::datetime(vec![1, 2], Tz::Utc)));
        assert!(!Index::datetime(vec![1, 2], Tz::Utc).label_eq(&Index::datetime(vec![1, 9], Tz::Utc)));
        assert!(Index::str(vec!["a".into()]).label_eq(&Index::str(vec!["a".into()])));
        assert!(!Index::str(vec!["a".into()]).label_eq(&Index::str(vec!["b".into()])));
        // different label kinds are never equal
        assert!(!Index::range(2).label_eq(&Index::datetime(vec![0, 1], Tz::Utc)));
        assert!(!Index::str(vec!["a".into()]).label_eq(&Index::int64(vec![0])));
    }
}
