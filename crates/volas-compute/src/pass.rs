//! A kernel pass may publish its last two continuation states along with its values.
//! The value-only sink is zero-sized: its callbacks (including state serialization)
//! disappear from the batch loop. A capturing sink serializes terminal rows and requested anchor rows.

/// Transient kernel publication storage. Offsets delimit opaque kernel words; only
/// the directive continuation owner frames them into persisted continuation states.
#[derive(Default)]
pub struct KernelPoints {
    rows: Vec<(usize, std::ops::Range<usize>)>,
    words: Vec<f64>,
}

impl KernelPoints {
    /// Reserve one batch's snapshot metadata and its common small kernel words.
    pub fn with_capacity(count: usize) -> Self {
        Self { rows: Vec::with_capacity(count), words: Vec::with_capacity(4 * count) }
    }
    /// Publish directly into the arena, without a temporary kernel allocation.
    pub fn capture(&mut self, end: usize, write: impl FnOnce(&mut Vec<f64>)) {
        self.capture_if(end, |words| { write(words); true });
    }
    /// A composition publishes only when all required component states exist.
    pub fn capture_if(&mut self, end: usize, write: impl FnOnce(&mut Vec<f64>) -> bool) {
        let start = self.words.len();
        if !write(&mut self.words) { self.words.truncate(start); return; }
        let stop = self.words.len();
        if self.rows.is_empty() {
            self.words.reserve((stop - start).saturating_mul(self.rows.capacity()).saturating_sub(stop));
        }
        self.rows.push((end, start..stop));
    }
    /// Borrow published states in row order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[f64])> {
        self.rows.iter().map(|(end, range)| (*end, &self.words[range.clone()]))
    }
    /// Borrow one row's published kernel state.
    pub fn at(&self, end: usize) -> Option<&[f64]> {
        let at = self.rows.binary_search_by_key(&end, |(row, _)| *row).ok()?;
        Some(&self.words[self.rows[at].1.clone()])
    }
    /// Walk an ascending series of requested rows once, including repeated rows.
    pub fn cursor(&self) -> KernelCursor<'_> { KernelCursor { points: self, at: 0 } }
    /// A child pass's rows become the enclosing pass's rows.
    pub fn restore_offset(&mut self, start: usize) {
        for (end, _) in &mut self.rows { *end += start; }
    }
    /// Transform a composition's publication without allocating one Vec per row.
    pub fn transform(&mut self, mut write: impl FnMut(usize, &[f64], &mut Vec<f64>) -> bool) {
        let previous = std::mem::take(self);
        let mut next = Self::with_capacity(previous.rows.len());
        for (end, state) in previous.iter() { next.capture_if(end, |words| write(end, state, words)); }
        *self = next;
    }
}

/// A forward lookup over one publication arena. Requested row counts must ascend;
/// duplicate counts reuse the same borrowed state without advancing the cursor.
pub struct KernelCursor<'a> {
    points: &'a KernelPoints,
    at: usize,
}

impl<'a> KernelCursor<'a> {
    /// Borrow a requested state, skipping only earlier publications.
    pub fn seek(&mut self, end: usize) -> Option<&'a [f64]> {
        while self.points.rows.get(self.at).is_some_and(|(row, _)| *row < end) { self.at += 1; }
        let (row, range) = self.points.rows.get(self.at)?;
        (*row == end).then(|| &self.points.words[range.clone()])
    }
}

/// Where a batch kernel publishes a row's state.
pub trait StateSink: Default {
    /// A component pass with the same requested snapshot rows.
    fn child(&self) -> Self { Self::default() }
    /// Discard snapshots from a speculative pass before its careful replacement.
    fn restart(&mut self) { *self = self.child(); }
    /// A pass starting after a leading warm-up prefix.
    fn shifted(&self, _start: usize) -> Self { self.child() }
    /// Restore snapshots to the caller's row coordinates.
    fn restore_offset(&mut self, _start: usize) {}
    /// Snapshot row counts requested by the caller.
    fn marks(&self) -> &[usize] { &[] }
    /// End row count of the next publication span, including terminal rows.
    #[inline(always)]
    fn next_end(&self, _start: usize, len: usize) -> usize { len }
    /// Whether state publication is requested.
    const CAPTURE: bool;
    /// Publish terminal or requested snapshot states, lazily avoiding per-row copies.
    fn record(&mut self, row: usize, len: usize, state: impl FnOnce(&mut Vec<f64>));
    /// Publish a composition only when all its component stages have terminal states.
    fn compose(&mut self, len: usize, stages: &[&Self], tail: impl Fn(usize, &mut Vec<f64>));
}

impl StateSink for () {
    const CAPTURE: bool = false;
    #[inline(always)]
    fn compose(&mut self, _: usize, _: &[&Self], _: impl Fn(usize, &mut Vec<f64>)) {}
    #[inline(always)]
    fn record(&mut self, _: usize, _: usize, _: impl FnOnce(&mut Vec<f64>)) {}
}

/// The terminal states of the same kernel pass that produced the values.
#[derive(Default)]
pub struct States {
    /// State after the last row.
    pub after: Option<Vec<f64>>,
    /// State before the last row.
    pub before: Option<Vec<f64>>,
    /// Requested row counts, separate from the terminal publication.
    pub marks: Vec<usize>,
    /// Kernel states captured at requested row counts in the producing pass.
    pub points: KernelPoints,
    /// Cursor into requested marks for constant-time row publication.
    pub next_mark: usize,
    /// Next row count needing publication, including terminal rows.
    pub next_capture: usize,
}

impl StateSink for States {
    const CAPTURE: bool = true;
    fn marks(&self) -> &[usize] { &self.marks }
    #[inline(always)]
    fn next_end(&self, start: usize, len: usize) -> usize {
        let terminal = if start < len { len - 1 } else { len };
        if self.next_capture >= start { return self.next_capture.min(terminal); }
        let cursor = if self.marks.get(self.next_mark).is_some_and(|&end| end >= start) { self.next_mark }
            else { self.marks.partition_point(|&end| end < start) };
        self.marks.get(cursor).copied().unwrap_or(terminal).min(terminal)
    }
    fn child(&self) -> Self { Self { marks: self.marks.clone(), points: KernelPoints::with_capacity(self.marks.len()), ..Self::default() } }
    fn shifted(&self, start: usize) -> Self {
        Self { points: KernelPoints::with_capacity(self.marks.len()), marks: self.marks.iter().filter_map(|&end| end.checked_sub(start).filter(|&end| end > 0)).collect(), ..Self::default() }
    }
    fn restore_offset(&mut self, start: usize) { self.points.restore_offset(start); }

    fn compose(&mut self, len: usize, stages: &[&Self], tail: impl Fn(usize, &mut Vec<f64>)) {
        let mut cursors: Vec<_> = stages.iter().map(|stage| stage.points.cursor()).collect();
        let mut write = |end, out: &mut Vec<f64>| {
            for (stage, cursor) in stages.iter().zip(&mut cursors) {
                let part = if end == len { stage.after.as_deref() }
                    else if end + 1 == len { stage.before.as_deref() }
                    else { cursor.seek(end) };
                let Some(part) = part else { return false; };
                out.extend_from_slice(part);
            }
            tail(end, out);
            true
        };
        for &end in self.marks.iter().filter(|&&end| end + 1 < len) {
            self.points.capture_if(end, |words| write(end, words));
        }
        for end in len.saturating_sub(1).max(1)..=len {
            let mut out = Vec::new();
            let state = write(end, &mut out).then_some(out);
            if end == len { self.after = state; } else { self.before = state; }
        }
    }
    #[inline(always)]
    fn record(&mut self, row: usize, len: usize, state: impl FnOnce(&mut Vec<f64>)) {
        let end = row + 1;
        // The common row does one comparison. Neither terminal bookkeeping nor
        // snapshot-vector bounds checks belong in the accumulating kernel loop.
        if end < self.next_capture { return; }
        while self.marks.get(self.next_mark).is_some_and(|&mark| mark < end) { self.next_mark += 1; }
        if end + 1 == len { let mut words = Vec::new(); state(&mut words); self.before = Some(words); }
        else if end == len { let mut words = Vec::new(); state(&mut words); self.after = Some(words); }
        else if self.marks.get(self.next_mark) == Some(&end) { self.points.capture(end, state); }
        while self.marks.get(self.next_mark).is_some_and(|&mark| mark <= end) { self.next_mark += 1; }
        self.next_capture = if end + 1 < len {
            self.marks.get(self.next_mark).copied().unwrap_or(len - 1).min(len - 1)
        } else { len };
    }
}

/// Values and the terminal states produced by one kernel pass.
pub struct KernelPass {
    /// Output values.
    pub values: Vec<f64>,
    /// Kernel state after the last row.
    pub after: Vec<f64>,
    /// Kernel state before the last row.
    pub before: Vec<f64>,
    /// Snapshots from the same kernel pass.
    pub points: KernelPoints,
}

impl KernelPass {
    /// Run a batch kernel with state publication enabled. A kernel may decline before
    /// its warm-up; the continuation owner handles that short history normally.
    pub fn run(run: impl FnOnce(&mut States) -> Vec<f64>) -> Option<Self> {
        Self::run_marked(&[], run)
    }

    /// Run with selected snapshot row counts, without revisiting any kernel rows.
    pub fn run_marked(marks: &[usize], run: impl FnOnce(&mut States) -> Vec<f64>) -> Option<Self> {
        let mut states = States { marks: marks.to_vec(), points: KernelPoints::with_capacity(marks.len()), ..States::default() };
        let values = run(&mut states);
        Some(Self { values, after: states.after?, before: states.before?, points: states.points })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unseeded_smoother_does_not_publish_terminal_states() {
        use crate::raw::indicators::ema_pass;
        for (data, period) in [(vec![], 3), (vec![f64::NAN; 4], 2), (vec![1.0, 2.0], 3), (vec![1.0, 2.0], 0)] {
            let mut states = States::default();
            let values = ema_pass(&data, period, &mut states);
            assert_eq!(values.len(), data.len());
            assert!(values.iter().all(|v| v.is_nan()));
            assert!(states.after.is_none() && states.before.is_none());
        }
        // A seed on the final row has no preceding seeded terminal state; the
        // continuation owner must retain its partial seed through the warm path.
        assert!(KernelPass::run(|states| ema_pass(&[1.0, 2.0, 3.0], 3, states)).is_none());
    }

    #[test]
    fn a_composition_waits_for_every_component_state() {
        use crate::raw::indicators::{bbi, bbi_pass, bbi_state};
        for n in [10, 21] {
            let data: Vec<_> = (0..n).map(|i| i as f64 / 7.0).collect();
            let pass = KernelPass::run(|s| bbi_pass(&data, 2, 3, 4, 20, 13, s));
            if n < 20 {
                assert!(pass.is_none());
            } else {
                let pass = pass.unwrap();
                for (a, b) in pass.values.iter().zip(bbi(&data, 2, 3, 4, 20, 13)) {
                    assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan());
                }
                assert_eq!(pass.after, bbi_state(&data, 2, 3, 4, 20, 13).unwrap());
                assert_eq!(pass.before, bbi_state(&data[..n - 1], 2, 3, 4, 20, 13).unwrap());
            }
        }
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;

    #[test]
    fn captured_sma_matches_value_only_at_dynamic_window_edges() {
        use crate::raw::kernels::{sma, sma_pass};
        for n in [0, 1, 40, 1100] {
            for mode in ["clean", "leading", "interior", "infinite"] {
                let data: Vec<_> = (0..n).map(|i| {
                    if mode == "leading" && i < 23 || mode == "interior" && i % 19 == 0 { f64::NAN }
                    else if mode == "infinite" && i == 25 { f64::INFINITY }
                    else { 100.0 + (i as f64 * 0.7).sin() }
                }).collect();
                for origin in [0, 17] {
                    for period in [0, 1, 20, 1000, n + 1].map(std::hint::black_box) {
                        let marks: Vec<_> = (16..n).step_by(16).collect();
                        let mut states = States { points: KernelPoints::with_capacity(marks.len()), marks, ..Default::default() };
                        let actual = sma_pass((&*data).into(), period, origin, &mut states);
                        let expected = sma((&*data).into(), period, origin);
                        assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{mode}, n{n}, period{period}, origin{origin}");
                    }
                }
            }
        }
    }

    #[test]
    fn captured_seeded_smoothing_preserves_strided_rows_and_states() {
        use crate::raw::kernels;
        for mode in ["clean", "leading", "interior", "infinite"] {
            let data = ndarray::Array1::from_iter((0..2200).map(|i| {
                if mode == "leading" && i < 46 || mode == "interior" && i % 19 == 0 { f64::NAN }
                else if mode == "infinite" && i == 50 { f64::INFINITY }
                else { 100.0 + (i as f64 * 0.7).sin() }
            }));
            for view in [data.view(), data.slice(ndarray::s![..;2]), data.slice(ndarray::s![..;-2])] {
                for period in [0, 1, 20, 1000, 3000].map(std::hint::black_box) {
                    for wilder in [false, true] {
                        let step = |previous, input| if wilder { kernels::wilder_step(period)(previous, input) }
                            else { kernels::ema_step(period)(previous, input) };
                        let marks: Vec<_> = (16..view.len()).step_by(16).collect();
                        let mut states = States { points: KernelPoints::with_capacity(marks.len()), marks, ..Default::default() };
                        let actual = kernels::sma_seeded_pass(view, period, step, &mut states);
                        let expected = kernels::sma_seeded_pass(view, period, step, &mut ());
                        assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{mode}, period{period}, wilder{wilder}, stride{}", view.strides()[0]);
                        for (end, words) in states.points.iter() {
                            assert_eq!(words[0].to_bits(), expected[end - 1].to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn captured_crsi_matches_value_only_for_rank_and_seed_edges() {
        use crate::raw::indicators::{crsi, crsi_pass};
        for n in [0, 1, 1200] {
            for mode in ["clean", "leading", "interior", "infinite"] {
                let data: Vec<_> = (0..n).map(|i| {
                    if mode == "leading" && i < 23 || mode == "interior" && i % 19 == 0 { f64::NAN }
                    else if mode == "infinite" && i == 25 { f64::INFINITY }
                    else { 100.0 + (i as f64 * 0.7).sin() }
                }).collect();
                for (rsi, streak) in [(0, 0), (1, 1), (3, 2), (1000, 1000)] {
                    for rank in [0, 1, 100, 1000, n + 1].map(std::hint::black_box) {
                        let marks = vec![16, 512, 1024];
                        let mut states = States { points: KernelPoints::with_capacity(marks.len()), marks, ..Default::default() };
                        let actual = crsi_pass(&data, rsi, streak, rank, &mut states);
                        let expected = crsi(&data, rsi, streak, rank);
                        assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{mode}, n{n}, rsi{rsi}, streak{streak}, rank{rank}");
                    }
                }
            }
        }
    }

    #[test]
    fn publication_arena_keeps_opaque_words_and_discards_partial_compositions() {
        let mut points = KernelPoints::with_capacity(3);
        points.capture(512, |out| out.extend_from_slice(&[1.0, f64::NAN]));
        points.capture_if(1024, |out| { out.push(99.0); false });
        points.capture(1536, |out| out.extend_from_slice(&[2.0, 3.0, 4.0]));
        assert_eq!(points.iter().map(|(end, _)| end).collect::<Vec<_>>(), vec![512, 1536]);
        assert!(points.at(512).unwrap()[1].is_nan());
        assert_eq!(points.at(1024), None);
        assert_eq!(points.at(1536), Some(&[2.0, 3.0, 4.0][..]));
        let mut cursor = points.cursor();
        assert_eq!(cursor.seek(0), None);
        assert!(cursor.seek(512).unwrap()[1].is_nan());
        assert!(cursor.seek(512).unwrap()[1].is_nan());
        assert_eq!(cursor.seek(1024), None);
        assert_eq!(cursor.seek(1536), Some(&[2.0, 3.0, 4.0][..]));
        assert_eq!(cursor.seek(1537), None);
        assert_eq!(cursor.seek(2000), None);
        points.restore_offset(17);
        points.transform(|_, state, out| { out.extend_from_slice(state); out.push(5.0); true });
        assert_eq!(points.at(1553), Some(&[2.0, 3.0, 4.0, 5.0][..]));
    }
}

#[cfg(test)]
mod schedule_tests {
    use super::*;

    #[test]
    fn spans_publish_requested_rows_and_terminal_states_after_a_warmup_prefix() {
        let mut values_only = ();
        for len in [0, 1, 9] {
            assert_eq!(values_only.next_end(1, len), len);
            assert!(values_only.marks().is_empty());
            let mut published = false;
            values_only.record(0, len, |_| published = true);
            assert!(!published);
        }
        let mut states = States { marks: vec![3, 7], points: KernelPoints::with_capacity(2), ..Default::default() };
        let mut from = 6;
        while from < 9 {
            let end = states.next_end(from + 1, 9);
            assert!(end > from);
            states.record(end - 1, 9, |out| out.push(end as f64));
            from = end;
        }
        assert_eq!(states.points.at(3), None);
        assert_eq!(states.points.at(7), Some(&[7.0][..]));
        assert_eq!(states.before.as_deref(), Some(&[8.0][..]));
        assert_eq!(states.after.as_deref(), Some(&[9.0][..]));
    }
}
