//! Two RSI series share one publication schedule and the existing GainLoss owner.
use super::{GainLoss, smoothing};
use crate::pass::StateSink;

/// Compute two seeded RSI series together; the caller owns any composite tail words.
/// Short or invalid seeds retain the ordinary component-pass fallback.
pub(crate) fn rsi_pair_pass<S: StateSink>(close: &[f64], streak: &[f64], rp: usize, sp: usize, states: &mut S, tail: impl Fn(usize, &mut Vec<f64>)) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = close.len();
    let seed = rp.max(sp);
    if !S::CAPTURE || rp == 0 || sp == 0 || n.saturating_sub(seed) <= 1 { return None; }
    let mut left = GainLoss::seed(close, rp);
    let mut right = GainLoss::seed(streak, sp);
    let mut a = vec![f64::NAN; n];
    let mut b = vec![f64::NAN; n];
    a[rp] = left.rsi();
    b[sp] = right.rsi();
    advance(close, &mut a, rp, seed, &mut left);
    advance(streak, &mut b, sp, seed, &mut right);
    let avg_left = smoothing::Wilder::new(rp);
    let avg_right = smoothing::Wilder::new(sp);
    let mut prior_left = close[seed];
    let mut prior_right = streak[seed];
    states.record(seed, n, |out| { left.append_state(out); right.append_state(out); tail(seed + 1, out); });
    let mut from = seed + 1;
    while from < n {
        let end = states.next_end(from + 1, n);
        for (((&x, &y), ax), by) in close[from..end].iter().zip(&streak[from..end]).zip(&mut a[from..end]).zip(&mut b[from..end]) {
            left.step(avg_left, x - prior_left);
            right.step(avg_right, y - prior_right);
            prior_left = x;
            prior_right = y;
            *ax = left.rsi();
            *by = right.rsi();
        }
        states.record(end - 1, n, |out| { left.append_state(out); right.append_state(out); tail(end, out); });
        from = end;
    }
    Some((a, b))
}

fn advance(data: &[f64], out: &mut [f64], period: usize, end: usize, pair: &mut GainLoss) {
    let avg = smoothing::Wilder::new(period);
    let mut prior = data[period];
    for (&current, target) in data[period + 1..end + 1].iter().zip(&mut out[period + 1..end + 1]) {
        pair.step(avg, current - prior);
        prior = current;
        *target = pair.rsi();
    }
}
