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
    complete: bool,
}

impl MaStage {
    /// The stage of `matype` over `period` — the type `TA_MA` runs there
    /// ([`super::generic_ma_type`]: any type at period 1 is the identity) — or `None` for
    /// an unknown type. Recursive stages carry their kernel and complete input tail.
    pub fn new(period: usize, matype: usize) -> Option<Self> {
        let kind = MaKind::new(period, matype)?;
        Some(Self { period, kind, complete: true })
    }

    /// An EMA stage with the raw kernel's leading-NaN skip and propagation thereafter.
    pub fn ema(period: usize) -> Self { Self { period, kind: MaKind::Recursive(RecursiveKind::Ema), complete: false } }

    /// The batch kernel publishes the exact states used by `step`, including
    /// partially seeded recursive stages.
    pub fn pass<S: crate::pass::StateSink>(&self, data: &[f64], origin: usize, states: &mut S) -> Option<Vec<f64>> {
        if !S::CAPTURE && !self.complete { return Some(self.kind.compute(data, self.period, origin)); }
        Some(match self.kind {
            MaKind::Finite(FiniteKind::Sma) => kernels::sma_pass(data.into(), self.period, origin, states).into_raw_vec_and_offset().0,
            MaKind::Finite(FiniteKind::Wma) => super::wma_pass(data, self.period, origin, states),
            MaKind::Finite(FiniteKind::Trima) => super::trima_pass(data, self.period, origin, states),
            MaKind::Recursive(kind) => return kind.pass(data, self.period, self.complete, states),
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
            MaKind::Recursive(kind) => kind.step(self.period, self.complete, term(row), state, next),
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
    fn step(self, p: usize, complete: bool, x: f64, state: &[f64], next: &mut [f64]) -> Option<f64> {
        next.copy_from_slice(state);
        if x.is_nan() && (complete || state[0] < self.seed_len(p) as f64) { return Some(f64::NAN); }
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
    fn pass(self, data: &[f64], p: usize, complete: bool, states: &mut impl crate::pass::StateSink) -> Option<Vec<f64>> {
        use crate::pass::States;
        use crate::missing::{CompleteRows, Input};
        let rows = if complete { CompleteRows::of(&[Input::new(data)]) } else {
            let start = data.iter().position(|v| !v.is_nan()).unwrap_or(data.len());
            if data[start..data.len().min(start + self.seed_len(p))].iter().all(|v| !v.is_nan()) {
                CompleteRows::After(start)
            } else {
                let mut seen = 0;
                CompleteRows::Kept((start..data.len()).filter(|&i| {
                    if seen < self.seed_len(p) && data[i].is_nan() { false }
                    else { seen += 1; true }
                }).collect())
            }
        };
        let input = rows.gather(data);
        let input = &*input;
        let mut raw = States::default();
        let result = match self {
            Self::Ema => kernels::sma_seeded_pass(input.into(), p, kernels::ema_step(p), &mut raw).into_raw_vec_and_offset().0,
            Self::Dema => super::dema_pass(input, p, &mut raw),
            Self::Tema => super::tema_pass(input, p, &mut raw),
            Self::Kama => super::kama_pass(input, p, 0, &mut raw),
            Self::Mama => crate::raw::indicators::mama_pass(input, 0.5, 0.05, false, &mut raw),
            Self::T3 => super::t3_pass(input, p, 0.7, &mut raw),
        };
        let seed = self.seed_len(p);
        for end in data.len().saturating_sub(1).max(1)..=data.len() {
            let count = match &rows {
                CompleteRows::All => end,
                CompleteRows::After(start) => end.saturating_sub(*start),
                CompleteRows::Kept(rows) => rows.partition_point(|&i| i < end),
            };
            let kernel = if count == input.len() { raw.after.as_ref() } else { raw.before.as_ref() };
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
                    self.step(p, complete, x, &state, &mut next)?;
                    std::mem::swap(&mut state, &mut next);
                }
                state
            };
            states.record(end - 1, data.len(), || state);
        }
        Some(rows.scatter(data.len(), result))
    }
}
