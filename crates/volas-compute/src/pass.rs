//! A kernel pass may publish its last two continuation states along with its values.
//! The value-only sink is zero-sized: its callbacks (including state serialization)
//! disappear from the batch loop. A capturing sink serializes only the last two rows.

/// Where a batch kernel publishes a row's state.
pub trait StateSink: Default {
    /// Whether state publication is requested.
    const CAPTURE: bool;
    /// Publish only the last two states; the closure is lazy to avoid per-row copies.
    fn record(&mut self, row: usize, len: usize, state: impl FnOnce() -> Vec<f64>);
    /// Publish a composition only when all its component stages have terminal states.
    fn compose(&mut self, len: usize, stages: &[&Self], tail: impl Fn(usize, &mut Vec<f64>));
}

impl StateSink for () {
    const CAPTURE: bool = false;
    #[inline(always)]
    fn compose(&mut self, _: usize, _: &[&Self], _: impl Fn(usize, &mut Vec<f64>)) {}
    #[inline(always)]
    fn record(&mut self, _: usize, _: usize, _: impl FnOnce() -> Vec<f64>) {}
}

/// The terminal states of the same kernel pass that produced the values.
#[derive(Default)]
pub struct States {
    /// State after the last row.
    pub after: Option<Vec<f64>>,
    /// State before the last row.
    pub before: Option<Vec<f64>>,
}

impl StateSink for States {
    const CAPTURE: bool = true;
    fn compose(&mut self, len: usize, stages: &[&Self], tail: impl Fn(usize, &mut Vec<f64>)) {
        for end in len.saturating_sub(1).max(1)..=len {
            let parts: Option<Vec<_>> = stages.iter().map(|stage| {
                if end < len { stage.before.as_deref() } else { stage.after.as_deref() }
            }).collect();
            let state = parts.map(|parts| {
                let mut out = parts.concat();
                tail(end, &mut out);
                out
            });
            if end < len { self.before = state; } else { self.after = state; }
        }
    }
    #[inline]
    fn record(&mut self, row: usize, len: usize, state: impl FnOnce() -> Vec<f64>) {
        if row + 2 == len { self.before = Some(state()); }
        else if row + 1 == len { self.after = Some(state()); }
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
}

impl KernelPass {
    /// Run a batch kernel with state publication enabled. A kernel may decline before
    /// its warm-up; the continuation owner handles that short history normally.
    pub fn run(run: impl FnOnce(&mut States) -> Vec<f64>) -> Option<Self> {
        let mut states = States::default();
        let values = run(&mut states);
        Some(Self { values, after: states.after?, before: states.before? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
