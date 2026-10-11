use crate::guard::{NonFiniteRows, RowGuard, Unguarded};

// ---------------------------------------------------------------------------
// Math operators — rolling reductions over a fixed window
// ---------------------------------------------------------------------------

/// Rolling sum over `period` (TA-Lib SUM), O(n) via a sliding total (TA-Lib's exact
/// order: add current, emit, drop trailing). Lookback `period-1`. Window-local: a
/// non-finite row affects only the windows holding it (see [`crate::guard`]).
pub fn sum(data: &[f64], period: usize) -> Vec<f64> {
    let (out, total) = sum_with::<Unguarded>(data, period);
    if total.is_finite() {
        out
    } else {
        sum_with::<NonFiniteRows>(data, period).0
    }
}

/// The sliding sum under a row guard; also returns the final running total.
fn sum_with<G: RowGuard>(data: &[f64], period: usize) -> (Vec<f64>, f64) {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return (out, 0.0);
    }
    let mut guard = G::default();
    let mut total = 0.0;
    for &x in &data[..period - 1] {
        if guard.enter(x.is_finite()) {
            total += x;
        }
    }
    let mut trailing = 0;
    #[allow(clippy::explicit_counter_loop)] // numeric kernel: explicit counter kept for hot-path codegen stability
    for i in (period - 1)..n {
        let x = data[i];
        if guard.enter(x.is_finite()) {
            total += x;
        }
        out[i] = if guard.clean() {
            total
        } else {
            data[i + 1 - period..=i].iter().sum()
        };
        let old = data[trailing];
        if guard.leave(old.is_finite()) {
            total -= old;
        }
        trailing += 1;
    }
    (out, total)
}

/// Absolute index (as a float) of the highest value in each trailing `period`-bar
/// window (TA-Lib MAXINDEX). Tie-breaking mirrors TA-Lib's incremental tracker: the
/// earliest index when the running max is rebuilt after the old one leaves (strict
/// `>`), the latest when a fresh bar matches the current max (`>=`). Lookback `period-1`.
pub fn maxindex(data: &[f64], period: usize) -> Vec<f64> {
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
                if val > h {
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
        out[today] = hi_idx as f64;
        trailing += 1;
    }
    out
}

/// Absolute index (as a float) of the lowest value in each trailing `period`-bar
/// window (TA-Lib MININDEX). Mirror of [`maxindex`] (strict `<` on rebuild, `<=` on
/// a fresh matching bar). Lookback `period-1`.
pub fn minindex(data: &[f64], period: usize) -> Vec<f64> {
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
                if val < l {
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
        out[today] = lo_idx as f64;
        trailing += 1;
    }
    out
}

// --- index-family state-carry (additive; the full-recompute fallback stays correct) ---
//
// MAXINDEX / MININDEX are finite-memory (a windowed arg-extreme) but emit ABSOLUTE row
// positions, so the value-probe fast-path declines them (a head-dropping slice rebases
// the window but the cached indices stay original-absolute). Two pieces continue them
// bit-exactly in O(new rows):
//   * a carried recursive `state = [extreme_idx_abs, extreme_value]` — the incremental
//     tracker's running extreme as of the last valid row. The tracker's `hi_idx` is NOT a
//     pure function of the current window (a `>=` match pushes it forward; a rebuild snaps
//     it to the earliest), so it genuinely must be carried, not re-derived.
//   * the column's `origin` offset (the original-frame row that this — possibly sliced —
//     frame's row 0 maps to), so emitted positions stay ABSOLUTE: a sub-frame position `p`
//     is original row `p + origin`, matching the verbatim-carried head and the
//     full-history ground truth. The stored `*_idx_abs` is original-absolute too, hence
//     stable across a slice (a slice carries the state verbatim and only bumps `origin`).
// `from < period - 1` (the tracker never reached a full window before `from`) returns
// `None` and falls back; a carried slice always keeps `>= lookback` rows so `from >=
// period - 1` holds and the resume is taken.

/// Final MAXINDEX tracker state `[hi_idx_abs, hi_value]` after a full [`maxindex`]
/// compute, or `None` if it never warms up (`period == 0 || period > n`). Reproduces the
/// incremental tracker exactly (earliest on rebuild via `>`, latest on a fresh tie via
/// `>=`); `hi_idx_abs` is a 0-based position in `data` (the freshly-computed frame, so
/// `origin == 0`).
pub fn maxindex_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
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
                if val > h {
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
    Some(vec![hi_idx as f64, hi])
}

/// Resume [`maxindex`] from `state = [hi_idx_abs, hi_value]` over rows `[from, n)`,
/// emitting ABSOLUTE positions (`sub_pos + origin`) and returning the updated state.
/// `None` at `from < period - 1` (the tracker is not yet warm). Continues the SAME
/// incremental tracker — bit-identical to a fresh full [`maxindex`].
pub fn maxindex_resume(
    data: &[f64],
    period: usize,
    from: usize,
    origin: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = data.len();
    if period == 0 || from < period - 1 || from > n {
        return None;
    }
    // The carried extreme index is original-absolute; map it to this frame's coordinates.
    let mut hi_idx = (state[0] as usize).saturating_sub(origin);
    let mut hi = state[1];
    let mut out = Vec::with_capacity(n - from);
    for today in from..n {
        let trailing = today - (period - 1);
        let tmp = data[today];
        if hi_idx < trailing {
            let win = &data[trailing..=today];
            let mut h = win[0];
            let mut hidx = trailing;
            for (off, &val) in win.iter().enumerate().skip(1) {
                if val > h {
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
        out.push((hi_idx + origin) as f64);
    }
    Some((out, vec![(hi_idx + origin) as f64, hi]))
}

/// Final MININDEX tracker state `[lo_idx_abs, lo_value]` after a full [`minindex`]
/// compute, or `None` if it never warms up. Mirror of [`maxindex_final_state`] (strict
/// `<` on rebuild, `<=` on a fresh tie).
pub fn minindex_final_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
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
                if val < l {
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
    Some(vec![lo_idx as f64, lo])
}

/// Resume [`minindex`] from `state = [lo_idx_abs, lo_value]` over rows `[from, n)`,
/// emitting ABSOLUTE positions. Mirror of [`maxindex_resume`].
pub fn minindex_resume(
    data: &[f64],
    period: usize,
    from: usize,
    origin: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = data.len();
    if period == 0 || from < period - 1 || from > n {
        return None;
    }
    let mut lo_idx = (state[0] as usize).saturating_sub(origin);
    let mut lo = state[1];
    let mut out = Vec::with_capacity(n - from);
    for today in from..n {
        let trailing = today - (period - 1);
        let tmp = data[today];
        if lo_idx < trailing {
            let win = &data[trailing..=today];
            let mut l = win[0];
            let mut lidx = trailing;
            for (off, &val) in win.iter().enumerate().skip(1) {
                if val < l {
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
        out.push((lo_idx + origin) as f64);
    }
    Some((out, vec![(lo_idx + origin) as f64, lo]))
}


/// [`sum`]'s resumable state after the rows of `term`: `[total, non-finite rows]`.
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
    use crate::guard::{Order, Slide, Window};
    if period == 0 || next.len() != SUM_STATE {
        return None;
    }
    let mut slide = Slide::<1>::from_state(state)?;
    let value = match slide.row(Order::AddReadDrop, period, row, |j| {
        let x = term(j);
        ([x], x.is_finite())
    }) {
        Window::Partial => f64::NAN,
        Window::Sums([total]) => total,
        Window::Bad => (row + 1 - period..=row).map(&term).sum(),
    };
    slide.write_state(next);
    Some(value)
}

/// [`sum`]'s state after the rows of `data`.
pub fn sum_state(data: &[f64], period: usize) -> Option<Vec<f64>> {
    crate::guard::state_after(vec![0.0; SUM_STATE], data.len(), |row, prev, next| {
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
        let max_full = maxindex(&data, period);
        let min_full = minindex(&data, period);
        // `from >= period` so the head `data[..from]` is long enough to seed a state
        // (a head of exactly `period - 1` rows never fills a window — see the guard test).
        for &from in &[period, period + 1, 40, 60, 119] {
            let st = maxindex_final_state(&data[..from], period).unwrap();
            let (tail, _) = maxindex_resume(&data, period, from, 0, &st).unwrap();
            assert_bits(&tail, &max_full[from..], "maxindex");

            let st = minindex_final_state(&data[..from], period).unwrap();
            let (tail, _) = minindex_resume(&data, period, from, 0, &st).unwrap();
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
        let max_full = maxindex(&data, period); // original-absolute ground truth
        let min_full = minindex(&data, period);

        // The carried state is original-absolute (computed over the original head), and
        // `origin == cut` rebases the sub-frame window while keeping emitted positions
        // original-absolute.
        let from_sub = 40usize; // a sub-frame row well past warm-up
        let head_orig = &data[..cut + from_sub];

        let st = maxindex_final_state(head_orig, period).unwrap();
        let (tail, end) = maxindex_resume(sub, period, from_sub, cut, &st).unwrap();
        assert_bits(&tail, &max_full[cut + from_sub..], "maxindex origin");
        assert_eq!(end[0], *max_full.last().unwrap(), "maxindex end pos");

        let st = minindex_final_state(head_orig, period).unwrap();
        let (tail, end) = minindex_resume(sub, period, from_sub, cut, &st).unwrap();
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
        assert!(maxindex_final_state(&data, 200).is_none()); // math_ops.rs:136
        assert!(minindex_final_state(&data, 200).is_none()); // math_ops.rs:215
                                                             // from < period - 1 -> tracker not yet warm.
        let st = maxindex_final_state(&data, 10).unwrap();
        assert!(maxindex_resume(&data, 10, 5, 0, &st).is_none()); // math_ops.rs:179
        let st = minindex_final_state(&data, 10).unwrap();
        assert!(minindex_resume(&data, 10, 5, 0, &st).is_none()); // math_ops.rs:256
    }
}
