//! Stochastic momentum's two double-EMA cascades and optional signal EMA.
use crate::raw::indicators::MaStage;

fn initial(d: usize, signal: Option<usize>) -> Vec<f64> {
    let mut state = Vec::new();
    for p in std::iter::repeat_n(d, 4).chain(signal) { state.extend(MaStage::ema(p).initial(0)); }
    state
}

/// Capture by feeding the identical HH/LL-derived inputs from row `k-1`. Each EMA
/// seeds at its first non-NaN input, including the second cascade and signal stages.
pub fn stoch_momentum_final_state(high: &[f64], low: &[f64], close: &[f64], k: usize, d: usize, signal: Option<usize>) -> Option<Vec<f64>> {
    crate::guard::state_after(initial(d, signal), close.len(), |row, prev, next| {
        step(high, low, close, k, d, signal, row, prev, next)
    })
}

#[allow(clippy::too_many_arguments)]
fn step(high: &[f64], low: &[f64], close: &[f64], k: usize, d: usize, signal: Option<usize>, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    let ema = MaStage::ema(d);
    let len = ema.state_len();
    let expected = 4 * len + signal.map_or(0, |p| MaStage::ema(p).state_len());
    if state.len() != expected || next.len() != expected { return None; }
    let (rdiff, diff) = if row + 1 < k { (f64::NAN, f64::NAN) } else {
        let hh = high[row + 1 - k..=row].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ll = low[row + 1 - k..=row].iter().copied().fold(f64::INFINITY, f64::min);
        (close[row] - (hh + ll) / 2.0, hh - ll)
    };
    let mut values = [rdiff, diff];
    for (i, value) in values.iter_mut().enumerate() {
        for j in 0..2 {
            let start = (2 * i + j) * len;
            *value = ema.step(row, |_| *value, &state[start..start + len], &mut next[start..start + len])?;
        }
    }
    let smi = values[0] / (values[1] * 0.5) * 100.0;
    if let Some(p) = signal { MaStage::ema(p).step(row, |_| smi, &state[4 * len..], &mut next[4 * len..]) }
    else { Some(smi) }
}

/// Append only the new rows, carrying all EMA stages (including partially seeded ones).
#[allow(clippy::too_many_arguments)]
pub fn stoch_momentum_resume(high: &[f64], low: &[f64], close: &[f64], k: usize, d: usize, signal: Option<usize>, from: usize, state: &[f64], out: &mut Vec<f64>) -> Option<Vec<f64>> {
    let mut prev = state.to_vec();
    let mut next = prev.clone();
    for row in from..close.len() {
        out.push(step(high, low, close, k, d, signal, row, &prev, &mut next)?);
        std::mem::swap(&mut prev, &mut next);
    }
    Some(prev)
}

/// Batch SMI composition; each smoothing kernel publishes its own terminal state.
#[allow(clippy::too_many_arguments)]
pub fn stoch_momentum_pass<S: crate::pass::StateSink>(high: &[f64], low: &[f64], close: &[f64], k: usize, d: usize, signal: Option<usize>, states: &mut S) -> Vec<f64> {
    let n = close.len();
    let hh = crate::raw::indicators::hhv(high, k);
    let ll = crate::raw::indicators::llv(low, k);
    let rdiff: Vec<_> = (0..n).map(|i| close[i] - (hh[i] + ll[i]) / 2.0).collect();
    let diff: Vec<_> = (0..n).map(|i| hh[i] - ll[i]).collect();
    let ema = MaStage::ema(d);
    let (mut a, mut b, mut c, mut e) = (S::default(), S::default(), S::default(), S::default());
    let a_values = ema.pass(&rdiff, 0, &mut a).expect("EMA stage");
    let numerator = ema.pass(&a_values, 0, &mut b).expect("EMA stage");
    let c_values = ema.pass(&diff, 0, &mut c).expect("EMA stage");
    let denominator = ema.pass(&c_values, 0, &mut e).expect("EMA stage");
    let mut values: Vec<_> = numerator.iter().zip(denominator).map(|(a, b)| a / (b * 0.5) * 100.0).collect();
    if let Some(p) = signal {
        let mut s = S::default();
        values = MaStage::ema(p).pass(&values, 0, &mut s).expect("EMA stage");
        states.compose(n, &[&a, &b, &c, &e, &s], |_, _| {});
    } else { states.compose(n, &[&a, &b, &c, &e], |_, _| {}); }
    values
}
