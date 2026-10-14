//! The generic moving-average stage: finite sums and recursive kernels share one
//! compositional interface. Recursive warm-up retains only the seed window; after
//! seeding, the existing kernel resumes over a fixed complete tail.
use crate::guard::{ring_push, state_after, Slide};
use crate::raw::kernels;
use super::generic::{FiniteKind, MaKind, RecursiveKind};
use super::ma::{wma_initial, wma_row, WMA_STATE};

/// A moving average of any TA-Lib MA type (0 through 8) as a resumable stage over any series, for the commands composed of them
/// (`ma`, `apo`, `ppo`, `macdext`, `bias`, `dma`, `mavp`, the stochastics). Its step is
/// the kernel's own row ([`kernels::sma_row`], [`wma_row`], [`super::trima_row`]).
#[derive(Clone, Copy, Debug)]
pub struct MaStage {
    period: usize,
    kind: MaKind,
}

impl MaStage {
    /// The stage of `matype` over `period` — the type `TA_MA` runs there
    /// ([`super::generic_ma_type`]: any type at period 1 is the identity) — or `None` for
    /// an unknown type. Recursive stages carry their kernel and complete input tail.
    pub fn new(period: usize, matype: usize) -> Option<Self> {
        let kind = MaKind::new(period, matype)?;
        Some(Self { period, kind })
    }

    /// An EMA stage with the shared complete-input rule.
    pub fn ema(period: usize) -> Self { Self { period, kind: MaKind::Recursive(RecursiveKind::Ema) } }

    /// The batch kernel publishes the exact states used by `step`, including
    /// partially seeded recursive stages.
    pub fn pass<S: crate::pass::StateSink>(&self, data: &[f64], origin: usize, states: &mut S) -> Option<Vec<f64>> {
        Some(match self.kind {
            MaKind::Finite(FiniteKind::Sma) => kernels::sma_pass(data, self.period, origin, states),
            MaKind::Finite(FiniteKind::Wma) => super::wma_pass(data, self.period, origin, states),
            MaKind::Finite(FiniteKind::Trima) => super::trima_pass(data, self.period, origin, states),
            MaKind::Recursive(kind) => return kind.pass(data, self.period, states),
        })
    }

    /// The length of its state.
    pub fn state_len(&self) -> usize {
        match self.kind {
            MaKind::Finite(FiniteKind::Sma) => kernels::SMA_STATE,
            MaKind::Finite(FiniteKind::Wma) => WMA_STATE,
            MaKind::Finite(FiniteKind::Trima) => super::trima_state_len(self.period),
            MaKind::Recursive(kind) => kind.state_len(self.period),
        }
    }

    /// How many rows before the current one its step reads (a stage over another
    /// stage's output keeps that many of them).
    pub fn reach(&self) -> usize {
        match self.kind {
            MaKind::Finite(FiniteKind::Trima) => super::trima_reach(self.period),
            MaKind::Finite(FiniteKind::Sma | FiniteKind::Wma) => self.period,
            MaKind::Recursive(_) => 0,
        }
    }

    /// One row of the average of `term` at `row`, from `state` into `next`.
    pub fn step(
        &self,
        row: usize,
        term: impl Fn(usize) -> f64,
        state: &[f64],
        next: &mut [f64],
    ) -> Option<f64> {
        if next.len() != self.state_len() || state.len() != next.len() {
            return None;
        }
        match self.kind {
            MaKind::Finite(FiniteKind::Sma) => kernels::sma_step(self.period, row, term, state, next),
            MaKind::Finite(FiniteKind::Wma) => wma_row(self.period, row, &term, state, next),
            MaKind::Finite(FiniteKind::Trima) => super::trima_row(self.period, row, &term, state, next),
            MaKind::Recursive(kind) => kind.step(self.period, term(row), state, next),
        }
    }

    /// Its state before its first row, at absolute position `origin`.
    pub fn initial(&self, origin: usize) -> Vec<f64> {
        match self.kind {
            MaKind::Finite(FiniteKind::Sma) => Slide::<1>::initial(origin),
            MaKind::Finite(FiniteKind::Wma) => wma_initial(),
            MaKind::Finite(FiniteKind::Trima) => super::trima_initial(self.period, origin),
            MaKind::Recursive(kind) => vec![0.0; kind.state_len(self.period)],
        }
    }

    /// Its state as a stage over a series produced one row at a time, before its first
    /// row (at absolute position `origin`): [`initial`](Self::initial), then the
    /// series' (not yet seen) previous values.
    pub fn chained_initial(&self, origin: usize) -> Vec<f64> {
        let mut v = self.initial(origin);
        v.resize(self.chained_len(), 0.0);
        v
    }

    /// The length of its state as a stage over a series produced one row at a time:
    /// its own state, then the last [`reach`](Self::reach) values of that series.
    pub fn chained_len(&self) -> usize {
        self.state_len() + self.reach()
    }

    /// One row of the average of a series produced one row at a time — `newest` is its
    /// value at `row` — from `state` into `next` (both [`chained_len`](Self::chained_len)
    /// long: the stage's state, then the series' previous values, oldest first).
    pub fn step_chained(&self, row: usize, newest: f64, state: &[f64], next: &mut [f64]) -> Option<f64> {
        let (len, reach) = (self.state_len(), self.reach());
        if state.len() != len + reach || next.len() != state.len() {
            return None;
        }
        let (own, ring) = state.split_at(len);
        let (n_own, n_ring) = next.split_at_mut(len);
        // the series at row j: `ring` holds rows `row-reach .. row-1`, oldest first
        let at = |j: usize| if j == row { newest } else { ring[reach - (row - j)] };
        let value = self.step(row, at, own, n_own)?;
        ring_push(ring, newest, n_ring);
        Some(value)
    }

    /// The state after the rows `0..n` of `term`, row 0 at absolute position `origin`.
    pub fn state(&self, n: usize, origin: usize, term: impl Fn(usize) -> f64) -> Option<Vec<f64>> {
        state_after(self.initial(origin), n, |row, prev, next| self.step(row, &term, prev, next))
    }
}


impl RecursiveKind {
    fn kernel_len(self) -> usize {
        match self {
            Self::Ema => 1,
            Self::Dema => 2,
            Self::Tema | Self::Kama => 3,
            Self::Mama => crate::raw::indicators::MAMA_STATE_LEN,
            Self::T3 => 6,
        }
    }
    fn seed_len(self, period: usize) -> usize { self.lookback(period) + 1 }
    fn state_len(self, period: usize) -> usize { 1 + self.kernel_len() + self.seed_len(period) + 1 }

    // [complete count (saturated), kernel..., complete tail + newest scratch].
    // The seed uses exactly the same rows and kernels as generic_ma. NaN stage
    // inputs are skipped, matching ma_typed's complete-row evaluation.
    fn step(self, p: usize, x: f64, state: &[f64], next: &mut [f64]) -> Option<f64> {
        next.copy_from_slice(state);
        if crate::missing::is_missing(x) { return Some(f64::NAN); }
        let seed = self.seed_len(p);
        let seen = state[0] as usize;
        let k = self.kernel_len();
        let (head, tail) = next.split_at_mut(1 + k);
        tail.copy_within(1.., 0);
        tail[seed] = x;
        head[0] = (seen + 1).min(seed) as f64;
        if seen + 1 < seed { return Some(f64::NAN); }
        let kernel = &mut head[1..];
        if seen < seed {
            let data = &tail[1..];
            let captured = match self {
                Self::Ema => super::ema_final_state(data, p),
                Self::Dema => super::dema_final_state(data, p),
                Self::Tema => super::tema_final_state(data, p),
                Self::Kama => super::kama_final_state(data, p, 0),
                Self::Mama => crate::raw::indicators::mama_state(data, 0.5, 0.05),
                Self::T3 => super::t3_final_state(data, p),
            }?;
            kernel.copy_from_slice(&captured);
            return self.compute(data, p, 0).last().copied();
        }
        let prev = &state[1..1 + k];
        match self {
            Self::Ema => super::ema_resume_one(tail, p, seed, prev, kernel),
            Self::Dema => super::dema_resume_one(tail, p, seed, prev, kernel),
            Self::Tema => super::tema_resume_one(tail, p, seed, prev, kernel),
            Self::Kama => super::kama_resume_one(tail, p, seed, prev, kernel),
            Self::Mama => crate::raw::indicators::mama_resume_one(tail, 0.5, 0.05, false, seed, prev, kernel),
            Self::T3 => super::t3_resume_one(tail, p, 0.7, seed, prev, kernel),
        }
    }
}

impl RecursiveKind {
    fn pass<S: crate::pass::StateSink>(self, data: &[f64], p: usize, states: &mut S) -> Option<Vec<f64>> {
        use crate::pass::States;
        use crate::missing::{CompleteRows, Input};
        let rows = CompleteRows::of(&[Input::new(data)]);
        if !S::CAPTURE {
            // A value-only pass needs no terminal-state publications. Preserve the
            // raw kernel's NaN warm-up prefix without a second output allocation;
            // a prefix containing infinity still goes through the shared row map.
            if let CompleteRows::After(start) = &rows {
                if data[..*start].iter().all(|v| v.is_nan()) {
                    return Some(self.compute(data, p, 0));
                }
            }
            return Some(rows.scatter(data.len(), self.compute(&rows.gather(data), p, 0)));
        }
        let input = rows.gather(data);
        let input = &*input;
        let mut marks: Vec<_> = states.marks().iter().copied().map(|end| rows.before(end)).collect();
        marks.sort_unstable(); marks.dedup();
        let mut raw = States { marks, ..States::default() };
        let result = match self {
            Self::Ema => kernels::sma_seeded_pass(input, p, kernels::ema_step(p), &mut raw),
            Self::Dema => super::dema_pass(input, p, &mut raw),
            Self::Tema => super::tema_pass(input, p, &mut raw),
            Self::Kama => super::kama_pass(input, p, 0, &mut raw),
            Self::Mama => crate::raw::indicators::mama_pass(input, 0.5, 0.05, false, &mut raw),
            Self::T3 => super::t3_pass(input, p, 0.7, &mut raw),
        };
        let seed = self.seed_len(p);
        let mut kernels = raw.points.cursor();
        for end in states.marks().to_vec().into_iter().filter(|&end| end + 1 < data.len()).chain(data.len().saturating_sub(1).max(1)..=data.len()) {
            let count = rows.before(end);
            let kernel = if count == input.len() { raw.after.as_deref() } else if count + 1 == input.len() { raw.before.as_deref() } else { kernels.seek(count) };
            let state = if count >= seed {
                let kernel = kernel?;
                let mut state = vec![seed as f64];
                state.extend_from_slice(kernel);
                state.extend(std::iter::repeat_n(0.0, (seed + 1).saturating_sub(count)));
                state.extend_from_slice(&input[count.saturating_sub(seed + 1)..count]);
                state
            } else {
                let mut state = vec![0.0; self.state_len(p)];
                let mut next = state.clone();
                for &x in &input[..count] {
                    self.step(p, x, &state, &mut next)?;
                    std::mem::swap(&mut state, &mut next);
                }
                state
            };
            states.record(end - 1, data.len(), |out| out.extend_from_slice(&state));
        }
        Some(rows.scatter(data.len(), result))
    }
}

#[cfg(test)]
mod nonfinite_tests {
    use super::*;

    #[test]
    fn stage_inputs_treat_infinity_as_nan_in_batch_and_step() {
        for stage in [MaStage::new(3, 1).unwrap(), MaStage::ema(3)] {
            for bad in [f64::INFINITY, f64::NEG_INFINITY] {
                for lane in [0, 1, 6] {
                    let mut inf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
                    let mut nan = inf.clone();
                    inf[lane] = bad;
                    nan[lane] = f64::NAN;
                    let mut passes = Vec::new();
                    for data in [&inf, &nan] {
                        let batch = stage.pass(data, 0, &mut ()).unwrap();
                        let mut state = stage.initial(0);
                        let mut next = state.clone();
                        let mut values = Vec::new();
                        for row in 0..data.len() {
                            values.push(stage.step(row, |i| data[i], &state, &mut next).unwrap());
                            std::mem::swap(&mut state, &mut next);
                        }
                        passes.push((batch, values));
                    }
                    for (a, b) in passes[0].0.iter().zip(&passes[0].1)
                        .chain(passes[1].0.iter().zip(&passes[1].1))
                        .chain(passes[0].0.iter().zip(&passes[1].0))
                        .chain(passes[0].1.iter().zip(&passes[1].1)) {
                        assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
                            "lane={lane}, bad={bad}: {a} vs {b}");
                    }
                }
            }
        }
    }
}
