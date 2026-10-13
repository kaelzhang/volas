//! Immutable, compiled UTC and local-wall interval indexes. No rule evaluation,
//! filesystem access, or allocation occurs in a scalar or cursor lookup.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use crate::datetime::Civil;
use crate::error::{Result, VolasError};
use super::{posix, tzif::{File, TimeType}};

const NS: i64 = 1_000_000_000;
const FIRST: i64 = i64::MIN.div_euclid(NS);
const LAST: i64 = i64::MAX.div_euclid(NS);

/// The exact TZif snapshot attached to a named zone. Cloning a `Tz` shares this
/// object; changing a data file or Python's TZPATH cannot mutate an old value.
pub struct ZoneRules {
    key: String,
    bytes: Vec<u8>,
    source: String,
    version: Option<String>,
    types: Vec<TimeType>,
    utc: Vec<UtcSpan>,
    local: Vec<LocalSpan>,
}

impl fmt::Debug for ZoneRules {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZoneRules").field("key", &self.key)
            .field("source", &self.source).field("version", &self.version).finish()
    }
}

impl PartialEq for ZoneRules {
    fn eq(&self, other: &Self) -> bool { self.key == other.key && self.bytes == other.bytes }
}
impl Eq for ZoneRules {}

#[derive(Clone, Copy)]
struct UtcSpan { start: i64, kind: usize }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Local {
    Unique(i32),
    /// The larger offset names the earliest instant in a fold.
    Fold(i32),
    /// The first real instant after a local gap, in UTC seconds.
    Gap(i64),
}

#[derive(Clone, Copy)]
struct LocalSpan { start: i64, kind: Local }

impl ZoneRules {
    pub(super) fn load(key: &str, bytes: Vec<u8>, source: String, version: Option<String>) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::compile(key.to_string(), bytes, source, version)?))
    }

    fn compile(key: String, bytes: Vec<u8>, source: String, version: Option<String>) -> Result<Self> {
        let file = File::read(&bytes)?;
        let mut types = file.types;
        // Include enough UTC and local time on either side for every legal
        // TZif offset and every POSIX transition time (up to 167 hours).
        let margin = types.iter().map(|t| i64::from(t.offset).abs()).max().unwrap_or(0) + 8 * 86_400;
        let (first, last) = (FIRST - margin, LAST + margin);
        let initial = if file.transitions.is_empty() && file.footer.is_empty() { types.len() - 1 } else { file.before };
        let mut utc = vec![UtcSpan { start: first, kind: initial }];
        for (&at, &kind) in file.transitions.iter().zip(&file.indices) {
            if at <= first { utc[0].kind = kind; }
            else if at <= last { utc.push(UtcSpan { start: at, kind }); }
        }
        if !file.footer.is_empty() {
            let rule = posix::Rule::parse(&file.footer)
                .ok_or_else(|| VolasError::Value("invalid TZif: invalid POSIX footer rule".into()))?;
            // CPython keeps the last explicit time type at the transition
            // second itself; the footer governs subsequent seconds.
            let from = file.transitions.last().map_or(first, |&at| at.saturating_add(1).max(first));
            if from <= last {
                let mut add = |at, reading: posix::Reading<'_>| {
                    let value = TimeType { offset: reading.kind.offset, dst: reading.dst, abbr: reading.kind.abbr.to_string() };
                    let kind = types.iter().position(|t| t == &value).unwrap_or_else(|| { types.push(value); types.len() - 1 });
                    utc.push(UtcSpan { start: at, kind });
                };
                add(from, rule.reading_at(Civil::from_epoch_seconds(from, 0)));
                let first_year = Civil::from_epoch_seconds(from, 0).year() - 1;
                let last_year = Civil::from_epoch_seconds(last, 0).year() + 1;
                for year in first_year..=last_year {
                    for (at, reading) in rule.transitions(year) {
                        if at > from && at <= last { add(at, reading); }
                    }
                }
            }
        }
        utc.sort_by_key(|s| s.start);
        let mut spans: Vec<UtcSpan> = Vec::with_capacity(utc.len());
        for span in utc {
            if let Some(prev) = spans.last_mut() {
                if prev.start == span.start { *prev = span; continue; }
                if types[prev.kind] == types[span.kind] { continue; }
            }
            spans.push(span);
        }
        let local = local_index(&spans, &types, last + 1);
        Ok(Self { key, bytes, source, version, types, utc: spans, local })
    }

    /// The name supplied with the snapshot (aliases keep their own names).
    pub fn key(&self) -> &str { &self.key }
    /// Exact source bytes, suitable for `ZoneInfo.from_file`.
    pub fn bytes(&self) -> &[u8] { &self.bytes }
    /// The selected directory or the `tzdata` package.
    pub fn source(&self) -> &str { &self.source }
    /// IANA release when the source reports one; never guessed from the key.
    pub fn version(&self) -> Option<&str> { self.version.as_deref() }

    /// Whether every reachable time type is canonical UTC over the complete
    /// nanosecond axis, including its partial first and last seconds. Unused
    /// types and intervals wholly outside that axis do not affect the result.
    pub fn is_canonical_utc(&self) -> bool {
        let first = self.utc.partition_point(|span| span.start <= FIRST).saturating_sub(1);
        self.utc[first..].iter().take_while(|span| span.start <= LAST).all(|span| {
            let kind = &self.types[span.kind];
            kind.offset == 0 && kind.dst == 0 && kind.abbr == "UTC"
        })
    }

    #[inline]
    pub(super) fn reading(&self, seconds: i64) -> &TimeType {
        let i = self.utc.partition_point(|s| s.start <= seconds).saturating_sub(1);
        &self.types[self.utc[i].kind]
    }

    #[inline]
    pub(super) fn local(&self, seconds: i64) -> Local {
        self.local[self.local.partition_point(|s| s.start <= seconds).saturating_sub(1)].kind
    }

    #[inline]
    pub(super) fn offset_cursor(&self, seconds: i64, at: &mut usize) -> i32 {
        advance(&self.utc, seconds, at, |s| s.start);
        self.types[self.utc[*at].kind].offset
    }

    #[inline]
    pub(super) fn local_cursor(&self, seconds: i64, at: &mut usize) -> Local {
        advance(&self.local, seconds, at, |s| s.start);
        self.local[*at].kind
    }

    pub(super) fn interval_start(&self, at: usize) -> i64 { self.utc[at].start }

    pub(super) fn previous_offset(&self, at: usize) -> Option<i32> {
        at.checked_sub(1).map(|i| self.types[self.utc[i].kind].offset)
    }
}

/// Ordered input advances once per interval; a backward jump uses binary
/// search, so the same cursor also correctly serves unordered input.
#[inline]
fn advance<T>(spans: &[T], seconds: i64, at: &mut usize, start: impl Fn(&T) -> i64) {
    if *at == usize::MAX || seconds < start(&spans[*at]) {
        *at = spans.partition_point(|s| start(s) <= seconds).saturating_sub(1);
    } else {
        while *at + 1 < spans.len() && seconds >= start(&spans[*at + 1]) { *at += 1; }
    }
}

/// Sweep the images of all UTC spans onto the wall axis. An empty active set
/// is a gap, multiple images are a fold. This handles even adjacent unusual
/// transitions without assuming they are a day (or a DST season) apart.
fn local_index(utc: &[UtcSpan], types: &[TimeType], end: i64) -> Vec<LocalSpan> {
    let mut events = Vec::with_capacity(2 * utc.len());
    for (i, span) in utc.iter().enumerate() {
        let off = types[span.kind].offset as i64;
        events.push((span.start + off, true, i));
        events.push((utc.get(i + 1).map_or(end, |s| s.start) + off, false, i));
    }
    events.sort_unstable();
    let mut active = BTreeSet::new();
    let mut local: Vec<LocalSpan> = Vec::with_capacity(events.len());
    let mut i = 0;
    while i < events.len() {
        let start = events[i].0;
        while i < events.len() && events[i].0 == start {
            let (_, enter, span) = events[i];
            if enter { active.insert(span); } else { active.remove(&span); }
            i += 1;
        }
        if i == events.len() { break; }
        let kind = if active.is_empty() {
            let next = events[i].0;
            let end = events[i..].iter().take_while(|e| e.0 == next)
                .filter(|e| e.1).map(|e| utc[e.2].start).min().expect("a bounded gap has an end");
            Local::Gap(end)
        } else {
            let offset = active.iter().map(|&i| types[utc[i].kind].offset).max().expect("nonempty interval");
            if active.len() == 1 { Local::Unique(offset) } else { Local::Fold(offset) }
        };
        if local.last().is_none_or(|s| s.kind != kind) { local.push(LocalSpan { start, kind }); }
    }
    local
}
