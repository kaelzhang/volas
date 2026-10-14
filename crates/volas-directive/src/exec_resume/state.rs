//! The continuation's wire format. Core stores these words without interpreting them.
//!
//! A node is `[kernel_len, reach, slots_mask, complete_count, scratch_stride, seed_origin, kernel..., tails..., scratch...]`.
//! Each used input slot has `reach` words, oldest first, padded on the left until
//! warm-up supplies them. Recursive tails contain complete rows; finite-window tails
//! retain raw window rows, including NaNs. Scratch is zeroed before publishing state.
//! A zero-length recursive kernel carries the complete seed rows until it can seed.
//! Chains start with their retained ring span, then concatenate `[node_len, node..., output_ring...]`.
//! Only this module reads or writes either layout.

use std::borrow::Cow;
use volas_core::DataFrame;
use crate::{complete::BoundCommand, operands::Operands};

/// Framed continuation snapshots share one immutable word buffer. The byte layout
/// is exactly this module's ordinary node/chain wire format; core stores only slices.
pub(super) struct FramedPoints {
    rows: Vec<(usize, std::ops::Range<usize>)>,
    words: Vec<f64>,
}

impl FramedPoints {
    pub fn with_capacity(count: usize, width: usize) -> Self {
        Self { rows: Vec::with_capacity(count), words: Vec::with_capacity(count * width) }
    }
    pub fn capture(&mut self, encoder: &FrameEncoder, end: usize, kernel: &[f64]) {
        let start = self.words.len();
        if encoder.write(end, kernel, &mut self.words).is_some() {
            self.rows.push((end, start..self.words.len()));
        }
    }
    /// Append a chain already formed through this owner's chain framing helpers.
    pub fn push(&mut self, end: usize, state: &[f64]) {
        let start = self.words.len();
        self.words.extend_from_slice(state);
        self.rows.push((end, start..self.words.len()));
    }
    /// Compose a chain into one buffer while preserving this owner's wire layout.
    pub fn transform(&mut self, mut write: impl FnMut(usize, &[f64], &mut Vec<f64>) -> bool) {
        let previous = std::mem::replace(self, Self::with_capacity(0, 0));
        let mut next = Self::with_capacity(previous.rows.len(), 0);
        let count = previous.rows.len();
        for (end, range) in previous.rows {
            let start = next.words.len();
            if write(end, &previous.words[range], &mut next.words) {
                // The first successful frame reveals the common chain width.
                // Reserve once; a later wider warm frame grows normally.
                if next.rows.is_empty() {
                    let width = next.words.len() - start;
                    next.words.reserve(width.saturating_mul(count).saturating_sub(next.words.len()));
                }
                next.rows.push((end, start..next.words.len()));
            }
            else { next.words.truncate(start); }
        }
        *self = next;
    }
    pub fn finish(self) -> Vec<(usize, volas_core::StateWords)> {
        if self.rows.is_empty() { return Vec::new(); }
        let words = std::sync::Arc::new(self.words);
        self.rows.into_iter().map(|(end, range)| {
            (end, volas_core::StateWords::from_shared(std::sync::Arc::clone(&words), range).expect("owner-framed range"))
        }).collect()
    }
}

pub(super) struct Node<'a> {
    words: &'a [f64],
    pub kernel: &'a [f64],
    reach: usize,
    mask: usize,
    count: usize,
    stride: usize,
    origin: usize,
}

impl<'a> Node<'a> {
    pub fn read(words: &'a [f64]) -> Option<Self> {
        let len = *words.first()? as usize;
        let reach = *words.get(1)? as usize;
        let mask = *words.get(2)? as usize;
        let count = *words.get(3)? as usize;
        let stride = *words.get(4)? as usize;
        if mask >= 256 || count > reach || words.len() != 6 + len + (reach + stride) * mask.count_ones() as usize {
            return None;
        }
        Some(Self { words, kernel: words.get(6..6 + len)?, reach, mask, count, stride, origin: *words.get(5)? as usize })
    }
    fn offset(&self, slot: usize) -> Option<usize> {
        (self.mask & (1 << slot) != 0).then(|| 6 + self.kernel.len() + self.reach * (self.mask & ((1 << slot) - 1)).count_ones() as usize)
    }
    fn tail(&self, slot: usize) -> Option<&[f64]> {
        let end = self.offset(slot)? + self.reach;
        Some(&self.words[end - self.count..end])
    }
    pub fn kernel_mut<'b>(&self, next: &'b mut [f64]) -> &'b mut [f64] {
        &mut next[6..6 + self.kernel.len()]
    }
    fn advance(&self, ops: &Operands, from: usize, end: usize, recursive: bool, next: &mut [f64]) -> Option<()> {
        // Only the final `reach` accepted rows survive in the published tail.
        // Walk backwards to those rows instead of shifting every input through the
        // tail after the batch kernel has already computed the whole batch. A
        // zero-reach recursion therefore reads no rows here.
        let rows = (from..end).rev()
            .filter(|&row| !recursive || ops.reads().all(|(_, r)| !volas_compute::missing::is_missing(r.view()[row])))
            .take(self.reach);
        let mut kept = 0;
        for row in rows {
            for (slot, r) in ops.reads() {
                let at = self.offset(slot)?;
                next[at + self.reach - 1 - kept] = r.view()[row];
            }
            kept += 1;
        }
        // Short batches keep the un-evicted suffix of the previous tail, including
        // any warm-up padding. Read it from the previous state, not the new suffix.
        for (slot, _) in ops.reads() {
            let at = self.offset(slot)?;
            next[at..at + self.reach - kept]
                .copy_from_slice(&self.words[at + kept..at + self.reach]);
        }
        next[3] = (self.count + kept).min(self.reach) as f64;
        Some(())
    }
    fn record(&self, ops: &Operands, cmd: &BoundCommand) -> Option<()> {
        for slot in 0..8 {
            if self.mask & (1 << slot) != 0 { ops.f64(slot, cmd.spec.series.get(slot)?).ok()?; }
        }
        Some(())
    }
    fn in_place(&self, ops: &Operands, from: usize, end: usize, recursive: bool) -> bool {
        from >= self.count && (self.count == self.reach || from == self.count) && (!recursive || ops.reads().all(|(_, r)| {
            r.known_has_nonfinite() == Some(false) || !volas_compute::missing::any_missing(&r.view()[from.saturating_sub(self.reach)..end])
        }))
    }
}

pub(super) fn capture(df: &DataFrame, cmd: &BoundCommand, end: usize, kernel: Vec<f64>, recursive: bool) -> Option<Vec<f64>> {
    let ops = Operands::rows(df, cmd.series, 0, end);
    ops.probe();
    if recursive {
        let _ = super::initial_state::state_of(&ops, cmd.name, cmd.sub, cmd.args);
    } else {
        for (slot, default) in cmd.spec.series.iter().enumerate() { ops.f64(slot, default).ok()?; }
    }
    if ops.failed() { return None; }
    capture_read(&ops, cmd, end, kernel, recursive)
}

/// Frame the states published by a kernel using its already-resolved input reads.
pub(super) fn capture_read(ops: &Operands, cmd: &BoundCommand, end: usize, kernel: impl AsRef<[f64]>, recursive: bool) -> Option<Vec<f64>> {
    FrameEncoder::new(ops, cmd, recursive, None).with_kernel(kernel.as_ref()).capture(end, kernel.as_ref())
}

/// A producing pass's input layout and complete-row proof. Foreign columns keep
/// their unknown NaN metadata; this proof lives only as long as the current pass.
pub(super) struct FrameEncoder<'a> {
    reads: [&'a [f64]; 8],
    slots: usize,
    header: [f64; 6],
    size: usize,
    kernel_len: usize,
    reach: usize,
    warm_limit: usize,
    rows: Option<Cow<'a, volas_compute::missing::CompleteRows>>,
}

impl<'a> FrameEncoder<'a> {
    pub fn new(ops: &'a Operands, cmd: &BoundCommand, recursive: bool, rows: Option<&'a volas_compute::missing::CompleteRows>) -> Self {
        let mut reads = [&[][..]; 8];
        let mut mask = 0;
        let mut slots = 0;
        for (slot, read) in ops.reads() { reads[slots] = read.view(); slots += 1; mask |= 1 << slot; }
        let rows = recursive.then(|| rows.map_or_else(|| {
            let inputs: Vec<_> = ops.reads().map(|(_, r)| volas_compute::missing::Input::known(r.view(), r.has_nonfinite())).collect();
            Cow::Owned(volas_compute::missing::CompleteRows::of(&inputs))
        }, Cow::Borrowed));
        let reach = if recursive { cmd.reach() } else { cmd.reach().max(cmd.spec.lookback_of(cmd.args)) };
        Self { reads, slots, header: [0.0, reach as f64, mask as f64, reach as f64, (reach + 1) as f64, ops.origin() as f64],
            size: 6 + (2 * reach + 1) * slots, kernel_len: 0, reach,
            warm_limit: 2 * (cmd.spec.lookback_of(cmd.args) + 1), rows }
    }

    /// All seeded publications use the producing kernel's fixed wire width.
    pub fn with_kernel(mut self, kernel: &[f64]) -> Self {
        self.kernel_len = kernel.len();
        self.header[0] = kernel.len() as f64;
        self.size += kernel.len();
        self
    }

    pub fn count_at(&self, end: usize) -> usize {
        self.rows.as_deref().map_or(end, |rows| rows.before(end))
    }

    /// Only a recursive prefix before its first seed may omit kernel words. A
    /// missing publication after warm-up must decline the batch path, never turn
    /// into an invented partial seed. The bound lookback owns this seed boundary.
    pub fn unseeded_kernel(&self, end: usize) -> Option<&'static [f64]> {
        (self.rows.is_some() && self.count_at(end) < self.warm_limit / 2).then_some(&[])
    }

    pub fn capture(&self, end: usize, kernel: &[f64]) -> Option<Vec<f64>> {
        let mut words = Vec::new();
        self.write(end, kernel, &mut words)?;
        Some(words)
    }

    fn write(&self, end: usize, kernel: &[f64], words: &mut Vec<f64>) -> Option<()> {
        let recursive = self.rows.is_some();
        let available = self.count_at(end);
        let kernel = if recursive && available < self.reach { &[][..] } else { kernel };
        let warm = recursive && kernel.is_empty();
        let header_storage;
        let (reach, count, size, header) = if !warm {
            if available < self.reach { return None; }
            if kernel.len() == self.kernel_len {
                (self.reach, self.reach, self.size, &self.header)
            } else {
                let mut adjusted = self.header;
                adjusted[0] = kernel.len() as f64;
                header_storage = adjusted;
                (self.reach, self.reach, self.size - self.kernel_len + kernel.len(), &header_storage)
            }
        } else {
            let reach = (available + 1).min(self.warm_limit);
            let count = available.min(reach);
            if count == reach { return None; }
            let stride = reach + 1;
            header_storage = [0.0, reach as f64, self.header[2], count as f64, stride as f64, self.header[5]];
            (reach, count, 6 + (reach + stride) * self.slots, &header_storage)
        };
        let rows = match self.rows.as_deref() {
            Some(volas_compute::missing::CompleteRows::Kept(rows)) => Some(&rows[available - count..available]),
            _ => None,
        };
        let start = words.len();
        words.reserve(size);
        words.extend_from_slice(header);
        words.extend_from_slice(kernel);
        for read in &self.reads[..self.slots] {
            if count < reach { words.extend(std::iter::repeat_n(f64::NAN, reach - count)); }
            if let Some(rows) = rows { words.extend(rows.iter().map(|&row| read[row])); }
            else { words.extend_from_slice(&read[end - count..end]); }
        }
        // The fixed layout already includes every slot's zeroed scratch span.
        words.resize(start + size, 0.0);
        Some(())
    }

}

/// Gather only the carried complete tail and complete *new* rows. Never look behind
/// the operand window, even after an arbitrarily long run of missing rows.
fn gathered<'a>(df: &'a DataFrame, cmd: &'a BoundCommand, ops: &Operands, prev: &Node, from: usize) -> Option<(Operands<'a>, Vec<usize>)> {
    let recursive = cmd.spec.history(cmd.args) == crate::spec::History::Recursive;
    let rows: Vec<usize> = (from..ops.len()).filter(|&i| !recursive || ops.reads().all(|(_, r)| !volas_compute::missing::is_missing(r.view()[i]))).collect();
    let mut views = vec![None; 8];
    for (slot, r) in ops.reads() {
        let mut values = prev.tail(slot)?.to_vec();
        values.extend(rows.iter().map(|&i| r.view()[i]));
        views[slot] = Some(Cow::Owned(values));
    }
    Some((Operands::over(df, cmd.series, ops.start(), views).with_origin(if prev.kernel.is_empty() { prev.origin } else { df.origin() + ops.start() + from - prev.count }), rows))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resume(
    df: &DataFrame, cmd: &BoundCommand, start: usize, words: &[f64], from: usize, end: usize,
    out: &mut Vec<f64>, run: impl Fn(&Operands, &[f64], usize, &mut Vec<f64>) -> Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    let prev = Node::read(words)?;
    let recursive = cmd.spec.history(cmd.args) == crate::spec::History::Recursive;
    let ops = Operands::rows(df, cmd.series, start, end);
    let local = from - start;
    let at = out.len();
    // The state describes the carried tail, not an arbitrary raw frame window.
    // Resolve its recorded inputs before choosing the rows to slide: a speculative
    // run across a missing row can evict a value the kernel never accumulated.
    prev.record(&ops, cmd)?;
    let mut next = words.to_vec();
    if recursive && prev.kernel.is_empty() {
        let (gather, rows) = gathered(df, cmd, &ops, &prev, local)?;
        if rows.is_empty() {
            out.resize(at + end - from, f64::NAN);
            return Some(next);
        }
        let values = crate::complete::into_f64(crate::exec::execute_inputs(&gather, cmd).ok()?);
        let kernel = super::initial_state::state_of(&gather, cmd.name, cmd.sub, cmd.args).unwrap_or_default();
        out.resize(at + end - from, f64::NAN);
        for (row, value) in rows.into_iter().zip(values.iter().skip(prev.count)) { out[at + row - local] = *value; }
        return capture_read(&gather, cmd, gather.len(), kernel, true);
    }
    let kernel = if prev.in_place(&ops, local, ops.len(), recursive) {
        run(&ops, prev.kernel, local, out)?
    } else {
        out.truncate(at);
        let (gather, rows) = gathered(df, cmd, &ops, &prev, local)?;
        let mut values = Vec::new();
        let kernel = if rows.is_empty() { prev.kernel.to_vec() } else { run(&gather, prev.kernel, prev.count, &mut values)? };
        out.resize(at + end - from, f64::NAN);
        for (row, value) in rows.into_iter().zip(values) { out[at + row - local] = value; }
        kernel
    };
    if kernel.len() != prev.kernel.len() { return None; }
    prev.kernel_mut(&mut next).copy_from_slice(&kernel);
    prev.advance(&ops, local, ops.len(), recursive, &mut next)?;
    Some(next)
}

pub(super) fn push_chain(chain: &mut Vec<f64>, state: &[f64], ring: impl Iterator<Item=f64>) {
    chain.push(state.len() as f64);
    chain.extend_from_slice(state);
    chain.extend(ring);
}

pub(super) fn chain_node<'a>(chain: &'a [f64], at: &mut usize, span: usize) -> Option<(&'a [f64], &'a [f64])> {
    let len = *chain.get(*at)? as usize;
    let state = chain.get(*at + 1..*at + 1 + len)?;
    let ring = chain.get(*at + 1 + len..*at + 1 + len + span)?;
    *at += 1 + len + span;
    Some((state, ring))
}

/// Step a node directly through the plan's scalar operand reader. Scratch lives after
/// the tails in the serialized buffer and is zeroed before publishing the new state.
pub(super) fn planned_step(
    df: &DataFrame, cmd: &BoundCommand, words: &[f64], row: usize, next: &mut [f64],
    mut read: impl FnMut(usize, &str, usize) -> Option<f64>,
) -> Option<f64> {
    let prev = Node::read(words)?;
    if next.len() != words.len() { return None; }
    next.copy_from_slice(words);
    let recursive = cmd.spec.history(cmd.args) == crate::spec::History::Recursive;
    let count = prev.count;
    let origin = (df.origin() + row).checked_sub(count)?;
    let scratch_at = 6 + prev.kernel.len() + prev.reach * prev.mask.count_ones() as usize;
    let (stored, scratch) = next.split_at_mut(scratch_at);
    let mut missing = false;
    let mut k = 0;
    for slot in 0..8 {
        if prev.mask & (1 << slot) == 0 { continue; }
        let buf = &mut scratch[k * prev.stride..(k + 1) * prev.stride];
        let default = cmd.spec.series.get(slot)?;
        buf[..count].copy_from_slice(prev.tail(slot)?);
        buf[count] = read(slot, default, row)?;
        missing |= volas_compute::missing::is_missing(buf[count]);
        k += 1;
    }
    if recursive && missing { scratch.fill(0.0); return Some(f64::NAN); }
    if recursive && prev.kernel.is_empty() { scratch.fill(0.0); return None; }
    let mut views = [None; 8];
    let mut k = 0;
    for (slot, view) in views.iter_mut().enumerate() {
        if prev.mask & (1 << slot) != 0 {
            *view = Some(&scratch[k * prev.stride..k * prev.stride + count + 1]);
            k += 1;
        }
    }
    let ops = Operands::borrowed(df, cmd.series, 0, views).with_origin(origin);
    let kernel = &mut stored[6..6 + prev.kernel.len()];
    let value = if recursive {
        super::scalar::step_of(&ops, cmd.name, cmd.sub, cmd.args, prev.kernel, count, kernel)
    } else {
        super::window::step(&ops, cmd.name, cmd.sub, cmd.args, prev.kernel, count, kernel)
    }?;
    if prev.reach > 0 {
        for (slot, view) in views.iter().enumerate() {
            if let Some(view) = view {
                let at = prev.offset(slot)?;
                stored.copy_within(at + 1..at + prev.reach, at);
                stored[at + prev.reach - 1] = view[count];
            }
        }
        stored[3] = (stored[3] + 1.0).min(prev.reach as f64);
    }
    scratch.fill(0.0);
    Some(value)
}

/// Locate a node and its ring without exposing framing to the plan executor.
pub(super) fn chain_part_mut(words: &mut [f64], state_len: usize, span: usize) -> Option<(&mut [f64], &mut [f64])> {
    let (state, ring) = words.get_mut(1..1 + state_len + span)?.split_at_mut(state_len);
    Some((state, ring))
}

#[cfg(test)]
mod framing_tests {
    use super::*;

    #[test]
    fn prepared_layout_preserves_opaque_kernel_words_and_raw_finite_tails() {
        let df = DataFrame::new(vec!["close".into()], vec![volas_core::Column::f64(vec![1.0, f64::NAN, 3.0])], None).unwrap();
        let ast = crate::parse("ma:2").unwrap();
        let crate::types::Continuation::Own(cmd) = ast.continuation() else { panic!("finite command owner") };
        let cmd = super::super::bound(cmd);
        let ops = Operands::new(&df, cmd.series);
        ops.f64(0, "close").unwrap();
        let encoder = FrameEncoder::new(&ops, &cmd, false, None).with_kernel(&[10.0, 11.0]);
        assert!(encoder.unseeded_kernel(1).is_none());
        assert!(encoder.capture(1, &[10.0, 11.0]).is_none());
        // Framing owns lengths and tails, not the kernel's opaque word contents.
        for kernel in [&[10.0, 11.0][..], &[12.0][..]] {
            let words = encoder.capture(3, kernel).unwrap();
            let node = Node::read(&words).unwrap();
            assert_eq!(node.kernel, kernel);
            let tail = node.tail(0).unwrap();
            assert!(volas_compute::missing::is_missing(tail[0]));
            assert_eq!(tail[1], 3.0);
            assert!(words[words.len() - node.stride..].iter().all(|&v| v == 0.0));
        }
    }
}
