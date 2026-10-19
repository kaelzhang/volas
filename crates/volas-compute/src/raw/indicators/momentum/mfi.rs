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
pub fn mfi_pass<S: crate::pass::StateSink>(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize, origin: usize, states: &mut S) -> Vec<f64> {
    mfi_pass_with(high, low, close, volume, period, origin, false, states)
}

/// Batch pass using the caller's remembered all-finite input fact.
#[allow(clippy::too_many_arguments)]
pub fn mfi_pass_with<S: crate::pass::StateSink>(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize, origin: usize, complete: bool, states: &mut S) -> Vec<f64> {
    if complete {
        return mfi_pass_rows::<true, S>(high, low, close, volume, period, origin, states);
    }
    mfi_pass_rows::<false, S>(high, low, close, volume, period, origin, states)
}

/// MFI on input columns already admitted as finite.
pub fn mfi_complete(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize, origin: usize) -> Vec<f64> {
    mfi_pass_rows::<true, ()>(high, low, close, volume, period, origin, &mut ())
}

fn mfi_pass_rows<const COMPLETE: bool, S: crate::pass::StateSink>(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize, origin: usize, states: &mut S) -> Vec<f64> {
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
    // Unknown inputs are admitted in the existing read pass. A non-finite
    // typical-price/volume witness enters the window-local fallback, whose flow
    // owner checks the original operands and preserves finite-input overflow.
    let mut clean = COMPLETE || prev_tp.is_finite();
    // Warm up the first window — bars `1..=period` fill ring slots `0..period`.
    let mut i = 1usize;
    while i <= period {
        let tp = (unsafe { *high.get_unchecked(i) }
            + unsafe { *low.get_unchecked(i) }
            + unsafe { *close.get_unchecked(i) })
            / 3.0;
        // Match TA-Lib's signed-flow branch shape; on real price series adjacent
        // typical-price direction is predictable enough to beat boolean-to-float masks.
        let volume = unsafe { *volume.get_unchecked(i) };
        let [p, ng] = signed_flow_with::<COMPLETE>(tp, prev_tp, volume, &mut clean);
        ring_pos[i - 1] = p;
        ring_neg[i - 1] = ng;
        pos_sum.add(p);
        neg_sum.add(ng);
        prev_tp = tp;
        i += 1;
    }
    // Re-anchor from the ring: the window's flows, oldest (slot `oldest`) first.
    let exact = |a: &Csum, b: &Csum| a.is_exact() && b.is_exact();
    if !exact(&pos_sum, &neg_sum) {
        mfi_reanchor(period, &mut clean, &mut pos_sum, &mut neg_sum, &ring_pos, &ring_neg, 0);
    }
    out.set(period, mfi_value([pos_sum.value(), neg_sum.value()]));
    states.record(period, n, |out| Slide::append_components(out, [pos_sum, neg_sum], 0, origin + period));
    let mut running = MfiLoop { out, ring_pos, ring_neg, pos_sum, neg_sum, prev_tp, slot: 0, clean };
    let mut from = period + 1;
    while from < n {
        let end = states.next_end(from + 1, n).clamp(from + 1, n);
        running = mfi_span::<COMPLETE>(running, [high, low, close, volume], period, from..end);
        states.record(end - 1, n, |out| Slide::append_components(out, [running.pos_sum, running.neg_sum], 0, origin + end - 1));
        from = end;
    }
    if running.clean && running.pos_sum.value().is_finite() && running.neg_sum.value().is_finite() {
        return running.out.finish();
    }
    mfi_nonfinite(high, low, close, volume, period, origin, states)
}

#[cold]
#[inline(never)]
fn mfi_nonfinite<S: crate::pass::StateSink>(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize, origin: usize, states: &mut S) -> Vec<f64> {
    let n = close.len();
    states.restart();
    // A non-finite flow entered: window-local careful path (see `crate::guard`). Flows
    // exist for bars `1..n`; flow `j` is bar `j + 1`'s.
    let flow = |j: usize| mfi_flow(high, low, close, volume, j);
    let mut out = crate::buf::OutBuf::warmup(n, period);
    let mut child = states.shifted(1);
    crate::guard::window_sums_pass(n - 1, period, Order::DropAdd, origin, flow, |j, s| out.set(j + 1, mfi_value(s)), &mut child);
    child.restore_offset(1);
    *states = child;
    out.finish()
}

/// Re-anchor the window from its ring in the original subtract-then-add order.
#[inline]
fn mfi_reanchor(period: usize, clean: &mut bool, pos_sum: &mut Csum, neg_sum: &mut Csum, ring_pos: &[f64], ring_neg: &[f64], oldest: usize) {
    *clean &= pos_sum.value().is_finite() && neg_sum.value().is_finite();
    if *clean {
        let window = (0..period).map(|k| (oldest + k) % period);
        [*pos_sum, *neg_sum] = Csum::of_rows(window.map(|k| [ring_pos[k], ring_neg[k]]));
    }
}

/// Numerical continuation across publication spans; the output remains write-once.
struct MfiLoop {
    out: crate::buf::OutBuf,
    ring_pos: Vec<f64>,
    ring_neg: Vec<f64>,
    pos_sum: Csum,
    neg_sum: Csum,
    prev_tp: f64,
    slot: usize,
    clean: bool,
}

// The publication arena must not share the numerical loop's register frame.
// Both value-only and capturing calls run this same arithmetic owner.
#[inline(never)]
fn mfi_span<const COMPLETE: bool>(running: MfiLoop, [high, low, close, volume]: [&[f64]; 4], period: usize, rows: std::ops::Range<usize>) -> MfiLoop {
    let MfiLoop { mut out, mut ring_pos, mut ring_neg, mut pos_sum, mut neg_sum, mut prev_tp, mut slot, mut clean } = running;
    let exact = |a: &Csum, b: &Csum| a.is_exact() && b.is_exact();
    for i in rows {
        pos_sum.sub(ring_pos[slot]); // bar leaving the window
        neg_sum.sub(ring_neg[slot]);
        let tp = (unsafe { *high.get_unchecked(i) }
            + unsafe { *low.get_unchecked(i) }
            + unsafe { *close.get_unchecked(i) })
            / 3.0;
        let volume = unsafe { *volume.get_unchecked(i) };
        let [p, ng] = signed_flow_with::<COMPLETE>(tp, prev_tp, volume, &mut clean);
        ring_pos[slot] = p; // bar entering
        ring_neg[slot] = ng;
        pos_sum.add(p);
        neg_sum.add(ng);
        if !exact(&pos_sum, &neg_sum) {
            mfi_reanchor(period, &mut clean, &mut pos_sum, &mut neg_sum, &ring_pos, &ring_neg, (slot + 1) % period);
        }
        out.set(i, mfi_value([pos_sum.value(), neg_sum.value()]));
        prev_tp = tp;
        slot += 1;
        if slot == period {
            slot = 0;
        }
    }
    MfiLoop { out, ring_pos, ring_neg, pos_sum, neg_sum, prev_tp, slot, clean }
}

/// MFI's signed money flow of bar `j + 1` (term `j`): `[positive, negative]`.
#[inline]
fn mfi_flow(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], j: usize) -> [f64; 2] {
    let (now, prev) = (typical(high, low, close, j + 1), typical(high, low, close, j));
    missing::terms(&[high[j], low[j], close[j], high[j + 1], low[j + 1], close[j + 1], volume[j + 1]],
        signed_flow(now, prev, volume[j + 1]))
}

/// A bar's money flow `tp·volume`, signed by the typical price `tp`'s direction against
/// the prior bar's `prev_tp` (`[flow, 0]` up, `[0, flow]` down, `[0, 0]` flat) — the one
/// owner of MFI's term, for the fast loop and [`mfi_flow`] alike. It reads both typical
/// prices and the volume, and is missing when one of them is (the per-row term rule,
/// [`missing::terms`]): a comparison with a missing value would read as "flat".
#[inline(always)]
fn signed_flow(tp: f64, prev_tp: f64, volume: f64) -> [f64; 2] {
    let signed = signed_flow_with::<true>(tp, prev_tp, volume, &mut true);
    if tp.is_nan() | prev_tp.is_nan() | volume.is_nan() { [f64::NAN; 2] } else { signed }
}

// Finite original operands make the typical price finite or infinite, never NaN:
// its additions can overflow, but add no opposite infinite input. Keep that IEEE
// overflow in the same arithmetic; only unadmitted inputs need the NaN guard.
#[inline(always)]
fn signed_flow_with<const COMPLETE: bool>(tp: f64, prev_tp: f64, volume: f64, clean: &mut bool) -> [f64; 2] {
    let flow = tp * volume;
    if tp > prev_tp {
        [flow, 0.0]
    } else if tp < prev_tp {
        [0.0, flow]
    } else {
        // A moving non-finite flow is already caught by the running sum before
        // publication. Only this flat branch can hide a missing input behind zero.
        if !COMPLETE { *clean &= flow.is_finite(); }
        [0.0, 0.0]
    }
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
