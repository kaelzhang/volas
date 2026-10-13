//! Money flow: the batch loop, row terms, and continuation state.
use super::*;

/// Money Flow Index (TA-Lib MFI): `100·posMF/(posMF+negMF)` over `period`, where each
/// bar's money flow `tp·volume` (tp = (h+l+c)/3) is signed by `tp`'s direction vs the
/// prior bar. A window whose flows sum to exactly 0 yields 0 — TA-Lib's result there,
/// which TA-Lib reaches through an absolute `total < 1.0` threshold instead (a
/// deliberate divergence: that threshold makes TA-Lib's MFI depend on the volume's
/// unit; the exact sums make the zero test exact). Row 0 is at absolute
/// position `origin` (flow `j`, of bar `j + 1`, at `origin + j`). First value at index
/// `period` (lookback = period). The sliding sums recompute the leaving bar's flow,
/// which is bit-identical to TA-Lib's stored value.
pub fn mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize, origin: usize) -> Vec<f64> {
    mfi_pass(high, low, close, volume, period, origin, &mut ())
}

/// MFI batch pass with terminal-state publication.
pub fn mfi_pass(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize, origin: usize, states: &mut impl crate::pass::StateSink) -> Vec<f64> {
    // The flow loop reads its inputs unchecked: inputs of unequal length line up no row.
    if !super::super::same_len(&[high, low, close, volume]) {
        return Vec::new();
    }
    let n = close.len();
    if period == 0 || period + 1 > n {
        return vec![f64::NAN; n];
    }
    // Single-write (D2): NaN warm-up `[0, period)`, valid region written once via
    // `out.set` (no prefill memset); `out.finish()` proves every slot was written.
    let mut out = crate::buf::OutBuf::warmup(n, period);
    // Single pass over O(period) memory (TA-Lib's shape): a ring buffer holds the last
    // `period` per-bar (positive, negative) money flows so the window can drop the
    // departing bar without an n-sized side array. The typical price is carried across
    // bars (`prev_tp`), so `(high+low+close)/3` is divided exactly once per bar, and the
    // slide keeps the subtract-then-add order — bit-identical to the per-bar-flow form.
    let mut ring_pos = vec![0.0; period];
    let mut ring_neg = vec![0.0; period];
    // `period + 1 <= n` and both loops keep `i < n`, so the unchecked OHLCV loads
    // stay in bounds while removing checks from MFI's hot path.
    let mut prev_tp = (unsafe { *high.get_unchecked(0) }
        + unsafe { *low.get_unchecked(0) }
        + unsafe { *close.get_unchecked(0) })
        / 3.0;
    let mut pos_sum = Csum::default();
    let mut neg_sum = Csum::default();
    // Warm up the first window — bars `1..=period` fill ring slots `0..period`.
    let mut i = 1usize;
    while i <= period {
        let tp = (unsafe { *high.get_unchecked(i) }
            + unsafe { *low.get_unchecked(i) }
            + unsafe { *close.get_unchecked(i) })
            / 3.0;
        // Match TA-Lib's signed-flow branch shape; on real price series adjacent
        // typical-price direction is predictable enough to beat boolean-to-float masks.
        let [p, ng] = signed_flow(tp, prev_tp, unsafe { *volume.get_unchecked(i) });
        ring_pos[i - 1] = p;
        ring_neg[i - 1] = ng;
        pos_sum.add(p);
        neg_sum.add(ng);
        prev_tp = tp;
        i += 1;
    }
    // Re-anchor from the ring: the window's flows, oldest (slot `oldest`) first.
    let mut clean = true;
    let mut reanchor = |pos_sum: &mut Csum, neg_sum: &mut Csum, ring_pos: &[f64], ring_neg: &[f64], oldest: usize| {
        clean &= pos_sum.value().is_finite() && neg_sum.value().is_finite();
        if clean {
            let window = (0..period).map(|k| (oldest + k) % period);
            [*pos_sum, *neg_sum] = Csum::of_rows(window.map(|k| [ring_pos[k], ring_neg[k]]));
        }
    };
    let exact = |a: &Csum, b: &Csum| a.is_exact() && b.is_exact();
    if !exact(&pos_sum, &neg_sum) {
        reanchor(&mut pos_sum, &mut neg_sum, &ring_pos, &ring_neg, 0);
    }
    out.set(period, mfi_value([pos_sum.value(), neg_sum.value()]));
    states.record(period, n, || Slide::state([pos_sum, neg_sum], 0, origin + period));
    let mut slot = 0usize; // ring slot holding the oldest in-window bar (to evict next)
    i = period + 1;
    while i < n {
        pos_sum.sub(ring_pos[slot]); // bar leaving the window
        neg_sum.sub(ring_neg[slot]);
        let tp = (unsafe { *high.get_unchecked(i) }
            + unsafe { *low.get_unchecked(i) }
            + unsafe { *close.get_unchecked(i) })
            / 3.0;
        let [p, ng] = signed_flow(tp, prev_tp, unsafe { *volume.get_unchecked(i) });
        ring_pos[slot] = p; // bar entering
        ring_neg[slot] = ng;
        pos_sum.add(p);
        neg_sum.add(ng);
        if !exact(&pos_sum, &neg_sum) {
            reanchor(&mut pos_sum, &mut neg_sum, &ring_pos, &ring_neg, (slot + 1) % period);
        }
        out.set(i, mfi_value([pos_sum.value(), neg_sum.value()]));
        states.record(i, n, || Slide::state([pos_sum, neg_sum], 0, origin + i));
        prev_tp = tp;
        slot += 1;
        if slot == period {
            slot = 0;
        }
        i += 1;
    }
    if clean && pos_sum.value().is_finite() && neg_sum.value().is_finite() {
        return out.finish();
    }
    // A non-finite flow entered: window-local careful path (see `crate::guard`). Flows
    // exist for bars `1..n`; flow `j` is bar `j + 1`'s.
    let flow = |j: usize| mfi_flow(high, low, close, volume, j);
    let mut out = crate::buf::OutBuf::warmup(n, period);
    crate::guard::window_sums_pass(n - 1, period, Order::DropAdd, origin, flow, |j, s| out.set(j + 1, mfi_value(s)), states);
    out.finish()
}

/// MFI's signed money flow of bar `j + 1` (term `j`): `[positive, negative]`.
#[inline]
fn mfi_flow(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], j: usize) -> [f64; 2] {
    let (now, prev) = (typical(high, low, close, j + 1), typical(high, low, close, j));
    signed_flow(now, prev, volume[j + 1])
}

/// A bar's money flow `tp·volume`, signed by the typical price `tp`'s direction against
/// the prior bar's `prev_tp` (`[flow, 0]` up, `[0, flow]` down, `[0, 0]` flat) — the one
/// owner of MFI's term, for the fast loop and [`mfi_flow`] alike. It reads both typical
/// prices and the volume, and is missing when one of them is (the per-row term rule,
/// [`missing::terms`]): a comparison with a missing value would read as "flat".
#[inline(always)]
fn signed_flow(tp: f64, prev_tp: f64, volume: f64) -> [f64; 2] {
    let flow = tp * volume;
    let signed = if tp > prev_tp {
        [flow, 0.0]
    } else if tp < prev_tp {
        [0.0, flow]
    } else {
        [0.0, 0.0]
    };
    missing::terms(&[tp, prev_tp, volume], signed)
}

/// MFI from its window's flow sums: `100·pos/(pos+neg)`, 0 when no money flowed (both
/// sums are non-negative and exact, so their total is zero exactly when they are).
#[inline]
fn mfi_value([pos, neg]: [f64; 2]) -> f64 {
    let total = pos + neg;
    if total == 0.0 { 0.0 } else { 100.0 * pos / total }
}

/// [`mfi`]'s state after the rows: the [`Slide`] of the flow sums `[Σpos, Σneg]`.
pub fn mfi_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: usize,
    origin: usize,
) -> Option<Vec<f64>> {
    state_after(Slide::<2>::initial(origin), close.len(), |row, prev, next| {
        mfi_step(high, low, close, volume, period, row, prev, next)
    })
}

/// One row of [`mfi`] at `row` (its flow is term `row - 1`; drop, then add), from
/// `state` into `next`.
#[allow(clippy::too_many_arguments)]
pub fn mfi_step(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || row >= close.len() || next.len() != Slide::<2>::STATE {
        return None;
    }
    if row == 0 {
        next.copy_from_slice(state);
        return Some(f64::NAN);
    }
    let flow = |j: usize| mfi_flow(high, low, close, volume, j);
    let mut slide = Slide::<2>::from_state(state)?;
    let j = row - 1;
    let out = match slide.row(Order::DropAdd, period, j, |k| all_finite(flow(k))) {
        Window::Partial => f64::NAN,
        Window::Sums(s) => mfi_value(s.map(|c| c.value())),
        Window::Bad => mfi_value(own_sums(j + 1 - period..=j, flow)),
    };
    slide.write_state(next);
    Some(out)
}
