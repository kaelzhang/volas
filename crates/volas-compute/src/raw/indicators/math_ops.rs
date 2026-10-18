use crate::accum::Csum;
use crate::guard::{NonFiniteRows, RowGuard, Unguarded};
use crate::missing;

// ---------------------------------------------------------------------------
// Math operators — rolling reductions over a fixed window
// ---------------------------------------------------------------------------

/// Rolling sum over `period` (TA-Lib SUM), O(n) via a sliding total (TA-Lib's
/// order: add current, emit, drop trailing), exact ([`Csum`], re-summed whenever it
/// could not hold its sum) so each window's sum is its own however long the slide —
/// which is also why it does not depend on `_origin`, the frame's position every kernel
/// takes. Lookback `period-1`. Window-local: a non-finite row affects only the windows
/// holding it (see [`crate::guard`]).
pub fn sum(data: &[f64], period: usize, _origin: usize) -> Vec<f64> {
    sum_pass(data, period, _origin, &mut ())
}

/// SUM batch pass with terminal-state publication.
pub fn sum_pass(data: &[f64], period: usize, origin: usize, states: &mut impl crate::pass::StateSink) -> Vec<f64> {
    sum_pass_with::<false>(data, period, origin, states)
}

/// SUM of internal arithmetic terms, preserving generated infinities.
pub(crate) fn sum_internal_pass(data: &[f64], period: usize, origin: usize, states: &mut impl crate::pass::StateSink) -> Vec<f64> {
    sum_pass_with::<true>(data, period, origin, states)
}

fn sum_pass_with<const INTERNAL: bool>(data: &[f64], period: usize, origin: usize, states: &mut impl crate::pass::StateSink) -> Vec<f64> {
    sum_pass_with_finiteness::<INTERNAL>(data, period, origin, states).0
}

/// Reuse the existing fast sum's finite result without another input scan.
/// True proves the values it read were finite; false can also mean finite-input
/// overflow, so callers still inspect original inputs on their cold repair path.
pub(crate) fn sum_source_pass_with_finiteness(data: &[f64], period: usize, origin: usize, states: &mut impl crate::pass::StateSink) -> (Vec<f64>, bool) {
    sum_pass_with_finiteness::<false>(data, period, origin, states)
}

fn sum_pass_with_finiteness<const INTERNAL: bool>(data: &[f64], period: usize, origin: usize, states: &mut impl crate::pass::StateSink) -> (Vec<f64>, bool) {
    let (out, finite) = sum_with::<Unguarded, INTERNAL>(data, period, origin, states);
    if finite {
        (out, true)
    } else {
        (sum_nonfinite::<INTERNAL>(data, period, origin, states), false)
    }
}

#[cold]
#[inline(never)]
fn sum_nonfinite<const INTERNAL: bool>(data: &[f64], period: usize, origin: usize, states: &mut impl crate::pass::StateSink) -> Vec<f64> {
    states.restart();
    sum_with::<NonFiniteRows, INTERNAL>(data, period, origin, states).0
}

/// The sliding sum under a row guard; also returns whether the total stayed finite.
fn sum_with<G: RowGuard, const INTERNAL: bool>(data: &[f64], period: usize, origin: usize, states: &mut impl crate::pass::StateSink) -> (Vec<f64>, bool) {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return (out, true);
    }
    let mut guard = G::default();
    let mut finite = true;
    let mut total = Csum::default();
    for &x in &data[..period - 1] {
        if guard.enter(!crate::missing::is_missing(x)) {
            total.add(x);
        }
    }
    let mut trailing = 0;
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for i in (period - 1)..n {
        let x = data[i];
        if guard.enter(!crate::missing::is_missing(x)) {
            total.add(x);
        }
        if !total.is_exact() {
            // the fast pass stops re-summing once a non-finite row has poisoned it (the
            // careful pass recomputes every row); the careful pass always re-sums
            finite &= total.value().is_finite();
            if finite || !guard.counts(false) {
                let window = data[i + 1 - period..=i].iter().copied();
                total = Csum::of(window.filter(|&x| guard.counts(!crate::missing::is_missing(x))));
            }
        }
        out[i] = if guard.clean() {
            total.value()
        } else if INTERNAL {
            data[i + 1 - period..=i].iter().sum()
        } else {
            f64::NAN
        };
        let old = data[trailing];
        if guard.leave(!crate::missing::is_missing(old)) {
            total.sub(old);
        }
        states.record(i, n, |out| crate::guard::Slide::append_components(out, [total], guard.bad(), origin + i + 1));
        trailing += 1;
    }
    (out, finite && total.value().is_finite())
}

/// A resumed position row: the tracker's absolute position `at`, or NaN while the
/// window `data[trailing..=today]` holds a missing value (the finite-window rule, as
/// [`missing::blank_windows`] applies it to the full computation).
#[inline]
fn position(data: &[f64], trailing: usize, today: usize, at: usize) -> f64 {
    if missing::holds_missing(&data[trailing..=today]) { f64::NAN } else { at as f64 }
}

/// Absolute index (as a float) of the highest value in each trailing `period`-bar
/// window (TA-Lib MAXINDEX). A tie reports the latest of the equal highs — when the
/// running max is rebuilt after the old one leaves as when a fresh bar matches it — so a
/// value is a function of its window alone (TA-Lib's tracker reports the earliest on a
/// rebuild, which makes the position depend on when the last rebuild happened; a
/// documented divergence). Row 0 is at
/// absolute position `origin` (the frame's), so the positions are the series' own —
/// the same on a slice or a bounded frame as in the full history. A window holding a
/// missing value is NaN. Lookback `period-1`.
pub fn maxindex(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    maxindex_with(data, period, origin, false)
}

/// `maxindex` with a trusted all-finite input fact.
pub fn maxindex_with(data: &[f64], period: usize, origin: usize, complete: bool) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut hi_idx = 0usize;
    let mut hi = f64::NEG_INFINITY;
    let mut first = true;
    let mut trailing = 0usize;
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for today in (period - 1)..n {
        let tmp = data[today];
        if first || hi_idx < trailing {
            // The tracked max left the window — rebuild over [trailing, today].
            // Scanning the window as a slice keeps this O(period) hot path free of
            // per-element bounds checks (TA-Lib's C rescan has none).
            let win = &data[trailing..=today];
            let mut h = win[0];
            let mut hidx = trailing;
            for (off, &val) in win.iter().enumerate().skip(1) {
                if val >= h {
                    h = val;
                    hidx = trailing + off;
                }
            }
            hi = h;
            hi_idx = hidx;
            first = false;
        } else if tmp >= hi {
            hi_idx = today;
            hi = tmp;
        }
        out[today] = (hi_idx + origin) as f64;
        trailing += 1;
    }
    if !complete { missing::blank_windows(&mut out, data, period); }
    out
}

/// Absolute index (as a float) of the lowest value in each trailing `period`-bar
/// window (TA-Lib MININDEX). Mirror of [`maxindex`] (a tie reports the latest of the
/// equal lows). Row 0 is at absolute position `origin`. A window holding a
/// missing value is NaN. Lookback `period-1`.
pub fn minindex(data: &[f64], period: usize, origin: usize) -> Vec<f64> {
    minindex_with(data, period, origin, false)
}

/// `minindex` with a trusted all-finite input fact.
pub fn minindex_with(data: &[f64], period: usize, origin: usize, complete: bool) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut lo_idx = 0usize;
    let mut lo = f64::INFINITY;
    let mut first = true;
    let mut trailing = 0usize;
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for today in (period - 1)..n {
        let tmp = data[today];
        if first || lo_idx < trailing {
            // Rebuild over [trailing, today] as a slice — the O(period) hot path then
            // carries no per-element bounds checks (mirror of `maxindex`).
            let win = &data[trailing..=today];
            let mut l = win[0];
            let mut lidx = trailing;
            for (off, &val) in win.iter().enumerate().skip(1) {
                if val <= l {
                    l = val;
                    lidx = trailing + off;
                }
            }
            lo = l;
            lo_idx = lidx;
            first = false;
        } else if tmp <= lo {
            lo_idx = today;
            lo = tmp;
        }
        out[today] = (lo_idx + origin) as f64;
        trailing += 1;
    }
    if !complete { missing::blank_windows(&mut out, data, period); }
    out
}

// --- index-family state-carry (additive; the full-recompute fallback stays correct) ---
//
// MAXINDEX / MININDEX are finite-memory (a windowed arg-extreme) but emit ABSOLUTE row
// positions, so the value-probe fast-path declines them (a head-dropping slice rebases
// the window but the cached indices stay original-absolute). Two pieces continue them
// bit-exactly in O(new rows):
//   * a carried `state = [extreme_idx_abs, extreme_value]` — the incremental tracker's
//     running extreme as of the last valid row (the window's latest arg-extreme), which
//     saves the resume a rescan of the window.
//   * the frame's `origin` (the absolute position of its row 0 — `DataFrame::origin`),
//     so emitted positions are ABSOLUTE on every path: a frame position `p` is series
//     position `p + origin`. The stored `*_idx_abs` is absolute too, hence stable across a
//     slice (a slice carries the state verbatim; its origin moves with its rows).
// `from < period - 1` (the tracker never reached a full window before `from`) returns
// `None` and falls back; a carried slice always keeps `>= lookback` rows so `from >=
// period - 1` holds and the resume is taken.

/// Final MAXINDEX tracker state `[hi_idx_abs, hi_value]` after a full [`maxindex`]
/// compute, or `None` if it never warms up (`period == 0 || period > n`). Reproduces the
/// incremental tracker exactly (the latest of equal highs); `hi_idx_abs` is the absolute position (`origin` + the position in `data`).
pub fn maxindex_final_state(data: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    let n = data.len();
    if period == 0 || period > n {
        return None;
    }
    let mut hi_idx = 0usize;
    let mut hi = f64::NEG_INFINITY;
    let mut first = true;
    let mut trailing = 0usize;
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for today in (period - 1)..n {
        let tmp = data[today];
        if first || hi_idx < trailing {
            let win = &data[trailing..=today];
            let mut h = win[0];
            let mut hidx = trailing;
            for (off, &val) in win.iter().enumerate().skip(1) {
                if val >= h {
                    h = val;
                    hidx = trailing + off;
                }
            }
            hi = h;
            hi_idx = hidx;
            first = false;
        } else if tmp >= hi {
            hi_idx = today;
            hi = tmp;
        }
        trailing += 1;
    }
    Some(vec![(hi_idx + origin) as f64, hi])
}

/// Resume [`maxindex`] from `state = [hi_idx_abs, hi_value]` over rows `[from, n)`,
/// emitting ABSOLUTE positions (`sub_pos + origin`) and returning the updated state.
/// `None` at `from < period - 1` (the tracker is not yet warm). Continues the SAME
/// incremental tracker — bit-identical to a fresh full [`maxindex`].
/// Appends rows `[from, n)` to `out` and returns the carried state.
pub fn maxindex_resume(
    data: &[f64],
    period: usize,
    from: usize,
    origin: usize,
    state: &[f64],
    out: &mut Vec<f64>,
) -> Option<Vec<f64>> {
    let n = data.len();
    if period == 0 || from < period - 1 || from > n {
        return None;
    }
    // The carried extreme index is original-absolute; map it to this frame's coordinates.
    // One before the frame's row 0 has left every window here: it is rebuilt at once.
    let carried = (state[0] as usize).checked_sub(origin);
    let mut expired = carried.is_none();
    let mut hi_idx = carried.unwrap_or(0);
    let mut hi = state[1];
    out.reserve(n - from);
    for today in from..n {
        let trailing = today - (period - 1);
        let tmp = data[today];
        if expired || hi_idx < trailing {
            expired = false;
            let win = &data[trailing..=today];
            let mut h = win[0];
            let mut hidx = trailing;
            for (off, &val) in win.iter().enumerate().skip(1) {
                if val >= h {
                    h = val;
                    hidx = trailing + off;
                }
            }
            hi = h;
            hi_idx = hidx;
        } else if tmp >= hi {
            hi_idx = today;
            hi = tmp;
        }
        out.push(position(data, trailing, today, hi_idx + origin));
    }
    Some(vec![(hi_idx + origin) as f64, hi])
}

/// Final MININDEX tracker state `[lo_idx_abs, lo_value]` after a full [`minindex`]
/// compute, or `None` if it never warms up. Mirror of [`maxindex_final_state`] (the latest
/// of equal lows).
pub fn minindex_final_state(data: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    let n = data.len();
    if period == 0 || period > n {
        return None;
    }
    let mut lo_idx = 0usize;
    let mut lo = f64::INFINITY;
    let mut first = true;
    let mut trailing = 0usize;
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for today in (period - 1)..n {
        let tmp = data[today];
        if first || lo_idx < trailing {
            let win = &data[trailing..=today];
            let mut l = win[0];
            let mut lidx = trailing;
            for (off, &val) in win.iter().enumerate().skip(1) {
                if val <= l {
                    l = val;
                    lidx = trailing + off;
                }
            }
            lo = l;
            lo_idx = lidx;
            first = false;
        } else if tmp <= lo {
            lo_idx = today;
            lo = tmp;
        }
        trailing += 1;
    }
    Some(vec![(lo_idx + origin) as f64, lo])
}

/// Resume [`minindex`] from `state = [lo_idx_abs, lo_value]` over rows `[from, n)`,
/// emitting ABSOLUTE positions. Mirror of [`maxindex_resume`].
/// Appends rows `[from, n)` to `out` and returns the carried state.
pub fn minindex_resume(
    data: &[f64],
    period: usize,
    from: usize,
    origin: usize,
    state: &[f64],
    out: &mut Vec<f64>,
) -> Option<Vec<f64>> {
    let n = data.len();
    if period == 0 || from < period - 1 || from > n {
        return None;
    }
    let carried = (state[0] as usize).checked_sub(origin);
    let mut expired = carried.is_none();
    let mut lo_idx = carried.unwrap_or(0);
    let mut lo = state[1];
    out.reserve(n - from);
    for today in from..n {
        let trailing = today - (period - 1);
        let tmp = data[today];
        if expired || lo_idx < trailing {
            expired = false;
            let win = &data[trailing..=today];
            let mut l = win[0];
            let mut lidx = trailing;
            for (off, &val) in win.iter().enumerate().skip(1) {
                if val <= l {
                    l = val;
                    lidx = trailing + off;
                }
            }
            lo = l;
            lo_idx = lidx;
        } else if tmp <= lo {
            lo_idx = today;
            lo = tmp;
        }
        out.push(position(data, trailing, today, lo_idx + origin));
    }
    Some(vec![(lo_idx + origin) as f64, lo])
}


/// The length of [`sum`]'s resumable state: the [`Slide`](crate::guard::Slide) of the
/// total.
pub(crate) const SUM_STATE: usize = crate::guard::Slide::<1>::STATE;

/// One row of [`sum`] over the series `term` at `row` — the careful [`sum_with`] loop's
/// row (add, read, drop) — from `state` into `next`.
pub(crate) fn sum_row(
    period: usize,
    row: usize,
    term: impl Fn(usize) -> f64,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    sum_row_with::<false>(period, row, term, state, next)
}

/// Step of the internal-term SUM, with its unchanged slide state.
pub(crate) fn sum_internal_row(period: usize, row: usize, term: impl Fn(usize) -> f64, state: &[f64], next: &mut [f64]) -> Option<f64> {
    sum_row_with::<true>(period, row, term, state, next)
}

fn sum_row_with<const INTERNAL: bool>(period: usize, row: usize, term: impl Fn(usize) -> f64, state: &[f64], next: &mut [f64]) -> Option<f64> {
    use crate::guard::{Order, Slide, Window};
    if period == 0 || next.len() != SUM_STATE {
        return None;
    }
    let mut slide = Slide::<1>::from_state(state)?;
    let value = match slide.row(Order::AddReadDrop, period, row, |j| {
        let x = term(j);
        ([x], !crate::missing::is_missing(x))
    }) {
        Window::Partial => f64::NAN,
        Window::Sums([total]) => total.value(),
        Window::Bad if INTERNAL => crate::guard::own_sums(row + 1 - period..=row, |j| [term(j)])[0],
        Window::Bad => f64::NAN,
    };
    slide.write_state(next);
    Some(value)
}

/// [`sum`]'s state after the rows of `data` (row 0 at absolute position `origin`).
pub fn sum_state(data: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    let init = crate::guard::Slide::<1>::initial(origin);
    crate::guard::state_after(init, data.len(), |row, prev, next| {
        sum_step(data, period, row, prev, next)
    })
}

/// One row of [`sum`] at `row`, from `state` into `next`.
pub fn sum_step(data: &[f64], period: usize, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    if row >= data.len() {
        return None;
    }
    sum_row(period, row, |j| data[j], state, next)
}

/// One extremum-position tracker step, preserving the full tracker's tie and missing
/// window rules. The state is the absolute index and value of the retained extreme.
pub fn index_resume_one(data: &[f64], period: usize, row: usize, origin: usize, max: bool, state: &[f64], next: &mut [f64]) -> Option<f64> {
    if period == 0 || row + 1 < period || row >= data.len() || state.len() != 2 || next.len() != 2 { return None; }
    let trailing = row + 1 - period;
    let mut index = state[0] as usize;
    let mut value = state[1];
    let better = |a: f64, b: f64| if max { a >= b } else { a <= b };
    if index < origin + trailing {
        value = data[trailing];
        index = origin + trailing;
        for (i, &v) in data.iter().enumerate().take(row + 1).skip(trailing + 1) {
            if better(v, value) { value = v; index = origin + i; }
        }
    } else if better(data[row], value) { value = data[row]; index = origin + row; }
    next.copy_from_slice(&[index as f64, value]);
    Some(position(data, trailing, row, index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::indicators::test_support::*;

    /// MAXINDEX/MININDEX resume (origin 0), fed the carried tracker state of a full
    /// compute over the head, reproduces the tail of a full compute over the whole
    /// input — bit-for-bit. Long resume spans over an oscillating series fire both the
    /// window-rebuild arm (the tracked extreme leaves the window) and the new-extreme
    /// `>=`/`<=` arm.
    #[test]
    fn index_resume_is_bit_identical_to_full() {
        let data = series(120);
        let period = 30usize;
        let max_full = maxindex(&data, period, 0);
        let min_full = minindex(&data, period, 0);
        // `from >= period` so the head `data[..from]` is long enough to seed a state
        // (a head of exactly `period - 1` rows never fills a window — see the guard test).
        for &from in &[period, period + 1, 40, 60, 119] {
            let st = maxindex_final_state(&data[..from], period, 0).unwrap();
            let mut tail = Vec::new();
            let _ = maxindex_resume(&data, period, from, 0, &st, &mut tail).unwrap();
            assert_bits(&tail, &max_full[from..], "maxindex");

            let st = minindex_final_state(&data[..from], period, 0).unwrap();
            let mut tail = Vec::new();
            let _ = minindex_resume(&data, period, from, 0, &st, &mut tail).unwrap();
            assert_bits(&tail, &min_full[from..], "minindex");
        }
    }

    /// With a non-zero `origin` (a head-dropped slice), the resume still emits
    /// ORIGINAL-frame absolute positions: continue a slice that drops the first `cut`
    /// rows and compare to the full-frame `maxindex` absolute indices.
    #[test]
    fn index_resume_emits_absolute_positions_with_origin() {
        let data = series(120);
        let period = 30usize;
        let cut = 20usize; // drop the head; this sub-frame's row 0 == original row `cut`
        let sub = &data[cut..];
        let max_full = maxindex(&data, period, 0); // original-absolute ground truth
        let min_full = minindex(&data, period, 0);

        // The carried state is original-absolute (computed over the original head), and
        // `origin == cut` rebases the sub-frame window while keeping emitted positions
        // original-absolute.
        let from_sub = 40usize; // a sub-frame row well past warm-up
        let head_orig = &data[..cut + from_sub];

        let st = maxindex_final_state(head_orig, period, 0).unwrap();
        let mut tail = Vec::new();
        let end = maxindex_resume(sub, period, from_sub, cut, &st, &mut tail).unwrap();
        assert_bits(&tail, &max_full[cut + from_sub..], "maxindex origin");
        assert_eq!(end[0], *max_full.last().unwrap(), "maxindex end pos");

        let st = minindex_final_state(head_orig, period, 0).unwrap();
        let mut tail = Vec::new();
        let end = minindex_resume(sub, period, from_sub, cut, &st, &mut tail).unwrap();
        assert_bits(&tail, &min_full[cut + from_sub..], "minindex origin");
        assert_eq!(end[0], *min_full.last().unwrap(), "minindex end pos");
    }

    /// Warm-up guards: `*_final_state` declines when the period never fills a window
    /// (`period > n`), and `*_resume` declines below the tracker's warm-up
    /// (`from < period - 1`).
    #[test]
    fn index_guards_decline() {
        let data = series(120);
        // period > n -> no full window ever forms.
        assert!(maxindex_final_state(&data, 200, 0).is_none()); // math_ops.rs:136
        assert!(minindex_final_state(&data, 200, 0).is_none()); // math_ops.rs:215
                                                             // from < period - 1 -> tracker not yet warm.
        let st = maxindex_final_state(&data, 10, 0).unwrap();
        assert!(maxindex_resume(&data, 10, 5, 0, &st, &mut Vec::new()).is_none()); // math_ops.rs:179
        let st = minindex_final_state(&data, 10, 0).unwrap();
        assert!(minindex_resume(&data, 10, 5, 0, &st, &mut Vec::new()).is_none()); // math_ops.rs:256
    }
}
