//! Connors RSI: two Wilder pairs, the signed streak and a finite ROC rank window.
use crate::raw::indicators::{rsi_final_state, rsi_resume_one};

/// CRSI alone owns the signed-streak word after the two RSI continuations.
fn append_streak(streak: &[f64], end: usize, state: &mut Vec<f64>) { state.push(streak[end - 1]); }

fn streak_next(prior: f64, current: f64, streak: f64) -> f64 {
    if current > prior { if streak > 0.0 { streak + 1.0 } else { 1.0 } }
    else if current < prior { if streak < 0.0 { streak - 1.0 } else { -1.0 } }
    else { 0.0 }
}

#[inline(always)]
fn rsi_pairs<S: crate::pass::StateSink>(close: &[f64], streak: &[f64], rsi_len: usize, streak_len: usize, rank_len: usize, states: &mut S) -> (Vec<f64>, Vec<f64>) {
    let (mut a, mut b) = (states.child(), states.child());
    let rsi_close = crate::raw::indicators::rsi_pass(close, rsi_len, &mut a);
    let rsi_streak = crate::raw::indicators::rsi_pass(streak, streak_len, &mut b);
    if close.len() > rank_len + 2 { states.compose(close.len(), &[&a, &b], |end, s| append_streak(streak, end, s)); }
    (rsi_close, rsi_streak)
}

// Capturing child arenas are finished before the independent rank loop. Keeping
// their stack frame separate lets that numerical loop use the value-only layout.
#[inline(never)]
fn captured_rsi_pairs<S: crate::pass::StateSink>(close: &[f64], streak: &[f64], rsi_len: usize, streak_len: usize, rank_len: usize, states: &mut S) -> (Vec<f64>, Vec<f64>) {
    if close.len() > rank_len + 2 {
        if let Some(pair) = super::super::oscillators::rsi_pair_pass(close, streak, rsi_len, streak_len, states,
            |end, out| append_streak(streak, end, out)) { return pair; }
    }
    rsi_pairs(close, streak, rsi_len, streak_len, rank_len, states)
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
    let mut streak = if S::CAPTURE { Vec::with_capacity(len) } else { vec![0.0; len] };
    if S::CAPTURE {
        if len > 0 { streak.push(0.0); }
        let mut prior = close.first().copied().unwrap_or(0.0);
        let mut previous = 0.0;
        streak.extend(close.iter().skip(1).map(|&current| {
            previous = streak_next(prior, current, previous);
            prior = current;
            previous
        }));
    } else {
        for i in 1..len { streak[i] = streak_next(close[i - 1], close[i], streak[i - 1]); }
    }
    let (rsi_close, rsi_streak) = if S::CAPTURE {
        captured_rsi_pairs(close, &streak, rsi_len, streak_len, rank_len, states)
    } else { rsi_pairs(close, &streak, rsi_len, streak_len, rank_len, states) };
    let roc1: Vec<f64> = (0..len)
        .map(|i| if i > 0 { (close[i] / close[i - 1] - 1.0) * 100.0 } else { f64::NAN })
        .collect();
    let rank_window = |current: f64, window: &[f64]| {
        let below = window.iter().filter(|&&x| x < current).count();
        below as f64 / rank_len as f64 * 100.0
    };
    let mut prank = if S::CAPTURE { Vec::with_capacity(len) } else { vec![f64::NAN; len] };
    if S::CAPTURE {
        prank.resize((rank_len + 1).min(len), f64::NAN);
        prank.extend(roc1.windows(rank_len + 1).skip(1).map(|window| {
            let (&current, history) = window.split_last().expect("rank window includes current");
            rank_window(current, history)
        }));
    } else {
        // Preserve the value-only numerical loop's original indexed stores.
        #[allow(clippy::needless_range_loop)]
        for i in (rank_len + 1)..len { prank[i] = rank_window(roc1[i], &roc1[i - rank_len..i]); }
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
    append_streak(&streak, streak.len(), &mut state);
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

#[cfg(test)]
mod publication_tests {
    use super::*;
    use crate::pass::{KernelPoints, States};

    fn bits(values: &[f64]) -> Vec<u64> { values.iter().map(|v| v.to_bits()).collect() }
    type Publications = (Option<Vec<u64>>, Option<Vec<u64>>, Vec<(usize, Vec<u64>)>);
    fn words(states: &States) -> Publications {
        (states.after.as_deref().map(bits), states.before.as_deref().map(bits),
            states.points.iter().map(|(end, state)| (end, bits(state))).collect())
    }
    fn sink(marks: &[usize]) -> States {
        States { points: KernelPoints::with_capacity(marks.len()), marks: marks.to_vec(), ..Default::default() }
    }

    #[test]
    fn paired_publications_match_independent_rsi_passes_and_full_prefixes() {
        for n in [0, 1, 4, 1100, 4097] {
            for mode in 0..4 {
                let data: Vec<_> = (0..n).map(|i| {
                    if mode == 1 && i < 23 { f64::NAN }
                    else if mode == 2 && (i % 19 == 0 || (490..1050).contains(&i)) { f64::from_bits(0xfff8000000001234) }
                    else if mode == 3 && i == 25 { f64::INFINITY }
                    else { 100.0 + (i as f64 * 0.7).sin() }
                }).collect();
                let streak: Vec<_> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
                for origin in [0, 17] {
                    let marks: Vec<_> = (1..n).filter(|end| (origin + end) % 512 == 0).collect();
                    for (rp, sp, rank) in [(0, 0, 0), (1, 1, 1), (3, 2, 100), (2, 3, 100),
                        (1, 1000, 100), (1000, 1, 100), (1000, 1000, 100),
                        (3, 2, 1000), (3, 2, n + 1), (n + 1, 2, 100),
                        (usize::MAX, 2, 100), (2, usize::MAX, 100)].map(std::hint::black_box) {
                        let case = format!("n{n}, mode{mode}, rp{rp}, sp{sp}, rank{rank}, origin{origin}");
                        let mut former = sink(&marks);
                        let expected = rsi_pairs(&data, &streak, rp, sp, rank, &mut former);
                        let mut paired = sink(&marks);
                        let actual = captured_rsi_pairs(&data, &streak, rp, sp, rank, &mut paired);
                        assert_eq!(bits(&actual.0), bits(&expected.0), "left: {case}");
                        assert_eq!(bits(&actual.1), bits(&expected.1), "right: {case}");
                        assert_eq!(words(&paired), words(&former), "publications: {case}");
                        let mut states = sink(&marks);
                        let actual = crsi_pass(&data, rp, sp, rank, &mut states);
                        assert_eq!(bits(&actual), bits(&crsi(&data, rp, sp, rank)), "CRSI: {case}");
                        for (end, state) in states.points.iter() {
                            if let Some(prefix) = crsi_final_state(&data[..end], rp, sp, rank) {
                                assert_eq!(bits(state), bits(&prefix), "prefix{end}: {case}");
                            } else {
                                // A raw RSI pair can be seeded while the finite rank
                                // is still warming; the framing owner retains seed rows.
                                assert!(end <= rp.max(sp) || end <= rank + 1, "unseeded prefix{end}: {case}");
                            }
                        }
                    }
                }
            }
        }
    }
}
