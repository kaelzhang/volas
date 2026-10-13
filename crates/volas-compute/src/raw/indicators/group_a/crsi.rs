//! Connors RSI: two Wilder pairs, the signed streak and a finite ROC rank window.
use crate::raw::indicators::{rsi_final_state, rsi_resume_one};

fn streak_next(prior: f64, current: f64, streak: f64) -> f64 {
    if current > prior { if streak > 0.0 { streak + 1.0 } else { 1.0 } }
    else if current < prior { if streak < 0.0 { streak - 1.0 } else { -1.0 } }
    else { 0.0 }
}

/// CRSI Connors RSI: `mean( RSI(close, rsi_len), RSI(streak, streak_len),
/// PercentRank(1-period ROC%, rank_len) )`, where `streak` is the signed run length of
/// consecutive up / down closes and PercentRank is the share of the prior `rank_len` ROC
/// values strictly below the current one. Source: Connors Research / TradingView.
pub fn crsi(close: &[f64], rsi_len: usize, streak_len: usize, rank_len: usize) -> Vec<f64> {
    crsi_pass(close, rsi_len, streak_len, rank_len, &mut ())
}

/// CRSI batch composition publishes its two Wilder pairs and terminal streak.
pub fn crsi_pass<S: crate::pass::StateSink>(close: &[f64], rsi_len: usize, streak_len: usize, rank_len: usize, states: &mut S) -> Vec<f64> {
    let len = close.len();
    let mut streak = vec![0.0; len];
    for i in 1..len {
        streak[i] = streak_next(close[i - 1], close[i], streak[i - 1]);
    }
    let (mut a, mut b) = (S::default(), S::default());
    let rsi_close = crate::raw::indicators::rsi_pass(close, rsi_len, &mut a);
    let rsi_streak = crate::raw::indicators::rsi_pass(&streak, streak_len, &mut b);
    if len > rank_len + 2 { states.compose(len, &[&a, &b], |end, s| s.push(streak[end - 1])); }
    let roc1: Vec<f64> = (0..len)
        .map(|i| if i > 0 { (close[i] / close[i - 1] - 1.0) * 100.0 } else { f64::NAN })
        .collect();
    let mut prank = vec![f64::NAN; len];
    for i in (rank_len + 1)..len {
        let window = &roc1[i - rank_len..i];
        let below = window.iter().filter(|&&x| x < roc1[i]).count();
        prank[i] = below as f64 / rank_len as f64 * 100.0;
    }
    (0..len)
        .map(|i| (rsi_close[i] + rsi_streak[i] + prank[i]) / 3.0)
        .collect()
}


/// [close RSI gain/loss, streak RSI gain/loss, last streak]. Both RSI seeds
/// include row 0; the streak starts at zero, exactly as in the full computation.
pub fn crsi_final_state(close: &[f64], rp: usize, sp: usize, rank: usize) -> Option<Vec<f64>> {
    if close.len() <= rp.max(sp).max(rank + 1) { return None; }
    let mut streak = vec![0.0; close.len()];
    for i in 1..close.len() { streak[i] = streak_next(close[i - 1], close[i], streak[i - 1]); }
    let mut state = rsi_final_state(close, rp)?;
    state.extend(rsi_final_state(&streak, sp)?);
    state.push(*streak.last()?);
    Some(state)
}

/// One allocation-free row of CRSI. The rank reads the preceding `rank` one-bar
/// returns; their comparisons and arithmetic order are the full kernel's.
pub fn crsi_resume_one(close: &[f64], rp: usize, sp: usize, rank: usize, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    if state.len() != 5 || next.len() != 5 || row <= rank || row >= close.len() { return None; }
    let streak = streak_next(close[row - 1], close[row], state[4]);
    let price_rsi = rsi_resume_one(close, rp, row, &state[..2], &mut next[..2])?;
    let streak_rsi = rsi_resume_one(&[state[4], streak], sp, 1, &state[2..4], &mut next[2..4])?;
    next[4] = streak;
    let roc = |i: usize| (close[i] / close[i - 1] - 1.0) * 100.0;
    let value = roc(row);
    let below = (row - rank..row).filter(|&i| roc(i) < value).count();
    Some((price_rsi + streak_rsi + below as f64 / rank as f64 * 100.0) / 3.0)
}

/// Append CRSI's new rows and return the same five-word state.
pub fn crsi_resume(close: &[f64], rp: usize, sp: usize, rank: usize, from: usize, state: &[f64], out: &mut Vec<f64>) -> Option<Vec<f64>> {
    let mut prev = state.to_vec();
    let mut next = prev.clone();
    for row in from..close.len() {
        out.push(crsi_resume_one(close, rp, sp, rank, row, &prev, &mut next)?);
        std::mem::swap(&mut prev, &mut next);
    }
    Some(prev)
}
