//! The continuation's wire format. Core stores these words without interpreting them.
//!
//! A node is `[kernel_len, reach, slots_mask, complete_count, scratch_stride, kernel..., tails..., scratch...]`.
//! Each used input slot has `reach` words, oldest first, padded on the left until
//! warm-up supplies them. Recursive tails contain complete rows; finite-window tails
//! retain raw window rows, including NaNs. Scratch is zeroed before publishing state.
//! Chains concatenate `[node_len, node..., output_ring...]`.
//! Only this module reads or writes either layout.

use std::borrow::Cow;
use volas_core::DataFrame;
use crate::{complete::BoundCommand, operands::Operands};

pub(super) struct Node<'a> {
    words: &'a [f64],
    pub kernel: &'a [f64],
    reach: usize,
    mask: usize,
    count: usize,
    stride: usize,
}

impl<'a> Node<'a> {
    pub fn read(words: &'a [f64]) -> Option<Self> {
        let len = *words.first()? as usize;
        let reach = *words.get(1)? as usize;
        let mask = *words.get(2)? as usize;
        let count = *words.get(3)? as usize;
        let stride = *words.get(4)? as usize;
        if mask >= 256 || count > reach || words.len() != 5 + len + (reach + stride) * mask.count_ones() as usize {
            return None;
        }
        Some(Self { words, kernel: words.get(5..5 + len)?, reach, mask, count, stride })
    }
    fn offset(&self, slot: usize) -> Option<usize> {
        (self.mask & (1 << slot) != 0).then(|| 5 + self.kernel.len() + self.reach * (self.mask & ((1 << slot) - 1)).count_ones() as usize)
    }
    fn tail(&self, slot: usize) -> Option<&[f64]> {
        let end = self.offset(slot)? + self.reach;
        Some(&self.words[end - self.count..end])
    }
    pub fn kernel_mut<'b>(&self, next: &'b mut [f64]) -> &'b mut [f64] {
        &mut next[5..5 + self.kernel.len()]
    }
    fn advance(&self, ops: &Operands, from: usize, end: usize, recursive: bool, next: &mut [f64]) -> Option<()> {
        // Only the final `reach` accepted rows survive in the published tail.
        // Walk backwards to those rows instead of shifting every input through the
        // tail after the batch kernel has already computed the whole batch. A
        // zero-reach recursion therefore reads no rows here.
        let rows = (from..end).rev()
            .filter(|&row| !recursive || ops.reads().all(|(_, r)| !r.view()[row].is_nan()))
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
            r.known_has_nan() == Some(false) || r.view()[from.saturating_sub(self.reach)..end].iter().all(|v| !v.is_nan())
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
    // Finite-window nodes read raw frame windows; only recursions gather complete rows.
    let reach = if recursive { cmd.reach() } else { cmd.reach().max(cmd.spec.lookback_of(cmd.args)) };
    let mask = ops.reads().fold(0usize, |m, (s, _)| m | (1 << s));
    let rows: Vec<usize> = (0..end).rev().filter(|&i| !recursive || ops.reads().all(|(_, r)| !r.view()[i].is_nan())).take(reach).collect();
    if rows.len() < reach { return None; }
    let stride = reach + 1;
    let mut words = vec![kernel.len() as f64, reach as f64, mask as f64, rows.len() as f64, stride as f64];
    words.extend(kernel);
    for (_, r) in ops.reads() {
        words.extend(std::iter::repeat_n(f64::NAN, reach - rows.len()));
        words.extend(rows.iter().rev().map(|&i| r.view()[i]));
    }
    words.resize(words.len() + stride * mask.count_ones() as usize, 0.0);
    Some(words)
}

/// Gather only the carried complete tail and complete *new* rows. Never look behind
/// the operand window, even after an arbitrarily long run of missing rows.
fn gathered<'a>(df: &'a DataFrame, cmd: &'a BoundCommand, ops: &Operands, prev: &Node, from: usize) -> Option<(Operands<'a>, Vec<usize>)> {
    let recursive = cmd.spec.history(cmd.args) == crate::spec::History::Recursive;
    let rows: Vec<usize> = (from..ops.len()).filter(|&i| !recursive || ops.reads().all(|(_, r)| !r.view()[i].is_nan())).collect();
    let mut views = vec![None; 8];
    for (slot, r) in ops.reads() {
        let mut values = prev.tail(slot)?.to_vec();
        values.extend(rows.iter().map(|&i| r.view()[i]));
        views[slot] = Some(Cow::Owned(values));
    }
    Some((Operands::over(df, cmd.series, ops.start(), views).with_origin(df.origin() + ops.start() + from - prev.count), rows))
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

pub(super) fn step(
    df: &DataFrame, cmd: &BoundCommand, start: usize, words: &[f64], row: usize, next: &mut [f64],
) -> Option<f64> {
    let prev = Node::read(words)?;
    if next.len() != words.len() { return None; }
    next.copy_from_slice(words);
    let ops = Operands::rows(df, cmd.series, start, row + 1);
    let local = row - start;
    let recursive = cmd.spec.history(cmd.args) == crate::spec::History::Recursive;
    let run = |ops: &Operands, at, next: &mut [f64]| {
        if recursive { super::scalar::step_of(ops, cmd.name, cmd.sub, cmd.args, prev.kernel, at, next) }
        else { super::window::step(ops, cmd.name, cmd.sub, cmd.args, prev.kernel, at, next) }
    };
    // Scalar continuation has the same state/input precondition as the row resume.
    prev.record(&ops, cmd)?;
    if prev.in_place(&ops, local, ops.len(), recursive) {
        let value = run(&ops, local, prev.kernel_mut(next))?;
        prev.advance(&ops, local, ops.len(), recursive, next)?;
        return Some(value);
    }
    let (gather, rows) = gathered(df, cmd, &ops, &prev, local)?;
    if rows.is_empty() { next.copy_from_slice(words); return Some(f64::NAN); }
    let value = run(&gather, prev.count, prev.kernel_mut(next))?;
    prev.advance(&ops, local, ops.len(), recursive, next)?;
    Some(value)
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
    let scratch_at = 5 + prev.kernel.len() + prev.reach * prev.mask.count_ones() as usize;
    let (stored, scratch) = next.split_at_mut(scratch_at);
    let mut missing = false;
    let mut k = 0;
    for slot in 0..8 {
        if prev.mask & (1 << slot) == 0 { continue; }
        let buf = &mut scratch[k * prev.stride..(k + 1) * prev.stride];
        let default = cmd.spec.series.get(slot)?;
        buf[..count].copy_from_slice(prev.tail(slot)?);
        buf[count] = read(slot, default, row)?;
        missing |= buf[count].is_nan();
        k += 1;
    }
    if recursive && missing { scratch.fill(0.0); return Some(f64::NAN); }
    let mut views = [None; 8];
    let mut k = 0;
    for (slot, view) in views.iter_mut().enumerate() {
        if prev.mask & (1 << slot) != 0 {
            *view = Some(&scratch[k * prev.stride..k * prev.stride + count + 1]);
            k += 1;
        }
    }
    let ops = Operands::borrowed(df, cmd.series, 0, views).with_origin(origin);
    let kernel = &mut stored[5..5 + prev.kernel.len()];
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
