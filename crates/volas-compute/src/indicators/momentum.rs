use crate::kernels;

// ---------------------------------------------------------------------------
// Momentum — change relative to the price `period` bars earlier
// ---------------------------------------------------------------------------

/// Momentum: `data[i] - data[i-period]` (TA-Lib MOM). NaN during warm-up.
pub fn mom(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    if period >= n {
        return vec![f64::NAN; n];
    }
    // Warm-up NaN, then compute the valid region once (no `vec![NaN; n]` re-write).
    let mut out = vec![f64::NAN; period];
    out.extend((period..n).map(|i| data[i] - data[i - period]));
    out
}

/// Shared shape for the rate-of-change ratios (ROC/ROCP/ROCR/ROCR100): relate
/// each row to the price `period` bars earlier via `f(current, prior)`, NaN
/// during warm-up. A prior price of exactly zero yields `0.0`, matching TA-Lib's
/// divide-by-zero guard (purely theoretical for a positive price series).
fn roc_ratio(data: &[f64], period: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let n = data.len();
    if period >= n {
        return vec![f64::NAN; n];
    }
    // Fill only the warm-up with NaN, then compute the valid region once via `extend`
    // rather than `vec![NaN; n]` followed by an index-overwrite of `[period, n)` — the
    // latter writes that range twice (NaN, then the result). Halves the result-buffer
    // writes; a meaningful share of these tiny division-bound kernels' cost.
    let mut out = vec![f64::NAN; period];
    out.extend(
        data[period..]
            .iter()
            .zip(&data[..(n - period)])
            .map(
                |(&current, &prior)| {
                    if prior == 0.0 {
                        0.0
                    } else {
                        f(current, prior)
                    }
                },
            ),
    );
    out
}

/// Rate of change: `100 * (data/data[period ago] - 1)` (TA-Lib ROC).
pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    if period >= n {
        return vec![f64::NAN; n];
    }
    let mut out = Vec::with_capacity(n);
    // ROC is a coverage hot spot by itself, so keep the canonical line specialized:
    // one explicit loop avoids the generic closure/iterator adapter used by ROCP/ROCR.
    unsafe {
        out.set_len(n);
    }
    out[..period].fill(f64::NAN);
    let data_ptr = data.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut cur_ptr = unsafe { data_ptr.add(period) };
    let mut prior_ptr = data_ptr;
    let mut out_write = unsafe { out_ptr.add(period) };
    let mut remaining = n - period;
    while remaining > 0 {
        // The three pointers advance together over `[period, n)`, `[0, n-period)`,
        // and the valid output region. This keeps the hot loop in TA-Lib's
        // trailing-index shape without recomputing `i` and `i-period` addresses.
        let prior = unsafe { *prior_ptr };
        let value = if prior == 0.0 {
            0.0
        } else {
            (unsafe { *cur_ptr } / prior - 1.0) * 100.0
        };
        unsafe {
            *out_write = value;
            cur_ptr = cur_ptr.add(1);
            prior_ptr = prior_ptr.add(1);
            out_write = out_write.add(1);
        }
        remaining -= 1;
    }
    out
}

/// Rate of change percentage: `data/data[period ago] - 1` (TA-Lib ROCP).
pub fn rocp(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| cur / prior - 1.0)
}

/// Rate of change ratio: `data/data[period ago]` (TA-Lib ROCR).
pub fn rocr(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| cur / prior)
}

/// Rate of change ratio ×100: `100 * data/data[period ago]` (TA-Lib ROCR100).
pub fn rocr100(data: &[f64], period: usize) -> Vec<f64> {
    roc_ratio(data, period, |cur, prior| cur / prior * 100.0)
}

/// Williams %R: `-100 * (HH - close) / (HH - LL)` over `period`, where HH/LL are
/// the highest high / lowest low (TA-Lib WILLR). A flat range (HH == LL) yields 0.
/// Lookback `period-1`; the tracker preserves TA-Lib's window and tie semantics.
pub fn willr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    if period == 0 || period > n {
        return vec![f64::NAN; n];
    }
    let lookback = period - 1;
    let mut out = Vec::with_capacity(n);
    // Every slot is written before return: warm-up rows are filled with NaN here,
    // and the loop below writes all valid rows `[lookback, n)`.
    unsafe {
        out.set_len(n);
    }
    out[..lookback].fill(f64::NAN);
    let mut today = lookback;
    let mut trailing = 0usize;
    let mut highest_idx = usize::MAX;
    let mut lowest_idx = usize::MAX;
    let mut highest = 0.0;
    let mut lowest = 0.0;
    while today < n {
        // `today < n`, `trailing <= today`, and rescan `idx <= today`, so all
        // unchecked loads below stay inside the input slices.
        let low_today = unsafe { *low.get_unchecked(today) };
        if lowest_idx == usize::MAX || lowest_idx < trailing {
            lowest_idx = trailing;
            lowest = unsafe { *low.get_unchecked(trailing) };
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *low.get_unchecked(idx) };
                if value < lowest {
                    lowest_idx = idx;
                    lowest = value;
                }
                idx += 1;
            }
        } else if low_today <= lowest {
            lowest_idx = today;
            lowest = low_today;
        }

        let high_today = unsafe { *high.get_unchecked(today) };
        if highest_idx == usize::MAX || highest_idx < trailing {
            highest_idx = trailing;
            highest = unsafe { *high.get_unchecked(trailing) };
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *high.get_unchecked(idx) };
                if value > highest {
                    highest_idx = idx;
                    highest = value;
                }
                idx += 1;
            }
        } else if high_today >= highest {
            highest_idx = today;
            highest = high_today;
        }

        // Same Williams %R formula as TA-Lib, but with one range reciprocal instead
        // of TA-Lib's `(range / -100)` followed by a second division.
        let range = highest - lowest;
        if range != 0.0 {
            out[today] = (highest - unsafe { *close.get_unchecked(today) }) * (-100.0 / range);
        } else {
            out[today] = 0.0;
        }
        trailing += 1;
        today += 1;
    }
    out
}

/// Balance of Power: `(close − open)/(high − low)` (TA-Lib BOP). A bar with no
/// range (`high − low < ε`) yields 0. Lookback 0.
pub fn bop(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..close.len())
        .map(|i| {
            let range = high[i] - low[i];
            if range < 1e-14 {
                0.0
            } else {
                (close[i] - open[i]) / range
            }
        })
        .collect()
}

/// Intraday Momentum Index (TA-Lib IMI): `100·upSum/(upSum+downSum)` over `period`,
/// where each up bar (`close>open`) adds `close-open` to upSum and each down bar adds
/// `open-close` to downSum. Lookback `period-1`. (An all-doji window yields NaN, as in
/// TA-Lib — no divide-by-zero guard. O(n·period), matching TA-Lib.)
pub fn imi(open: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let lookback = period - 1;
    // Each bar's up / down move; the window sums slide (add the entering bar, drop the leaving
    // one) -> O(n) instead of re-scanning the window each bar. The slide drifts ~1e-13, inside
    // the 1e-9 TA-Lib tolerance, as the var / cci / linear-regression slides already rely on.
    let updown = |i: usize| -> (f64, f64) {
        let (c, o) = (close[i], open[i]);
        if c > o {
            (c - o, 0.0)
        } else {
            (0.0, o - c)
        }
    };
    let (mut up_sum, mut down_sum) = (0.0, 0.0);
    for i in 0..lookback {
        let (u, d) = updown(i);
        up_sum += u;
        down_sum += d;
    }
    for today in lookback..n {
        let (u, d) = updown(today);
        up_sum += u;
        down_sum += d;
        out[today] = 100.0 * (up_sum / (up_sum + down_sum));
        let (lu, ld) = updown(today - lookback);
        up_sum -= lu;
        down_sum -= ld;
    }
    out
}

/// Commodity Channel Index (TA-Lib CCI): `(tp − SMA(tp)) / (0.015 · meanDev)` over
/// `period`, where `tp = (high+low+close)/3` and `meanDev` is the mean absolute
/// deviation of `tp` from its average. A zero numerator or zero deviation yields 0
/// (TA-Lib's guard). Lookback `period-1`. O(n·period), as in TA-Lib.
pub fn cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let p = period as f64;
    let tp: Vec<f64> = (0..n)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect();
    // The window mean is a sliding sum (add the entering bar, drop the leaving one) — O(1)
    // per bar instead of re-summing the window. The mean-absolute-deviation still scans the
    // window (each `|x - avg|` depends on the just-computed mean, so it cannot slide). The
    // sliding sum drifts ~1e-13 relative, well inside the 1e-9 TA-Lib parity tolerance, as the
    // var / linear-regression slides already rely on.
    let mut sum: f64 = tp[..period - 1].iter().sum();
    for i in (period - 1)..n {
        sum += tp[i];
        let avg = sum / p;
        let window = &tp[i + 1 - period..=i];
        let sum_dev: f64 = window.iter().map(|x| (x - avg).abs()).sum();
        let num = tp[i] - avg;
        out[i] = if num != 0.0 && sum_dev != 0.0 {
            num / (0.015 * (sum_dev / p))
        } else {
            0.0
        };
        sum -= tp[i + 1 - period];
    }
    out
}

/// TRIX (TA-Lib): the 1-period percent rate-of-change of a triple SMA-seeded EMA of
/// `close`. Lookback `3·(period-1) + 1`. The three cascaded EMAs are fused into a single
/// pass via [`kernels::ema_cascade`] (one traversal, one allocation), then the verified
/// `roc` finishes it. Bit-identical to chaining three `ema_seeded` calls.
pub fn trix(close: &[f64], period: usize) -> Vec<f64> {
    let e3 = kernels::ema_cascade::<3>(close, period);
    roc(&e3, 1)
}

/// Final TRIX state `[e0, e1, e2]` — the three cascaded EMAs as of the last row (the
/// 3rd stage `e2` is the prior cascade output the 1-period ROC needs). `None` if the
/// cascade never seeds (`3·(period−1) >= n` → all-NaN, keep the fallback). Pairs with
/// [`trix_resume`].
pub fn trix_final_state(close: &[f64], period: usize) -> Option<Vec<f64>> {
    let e = kernels::ema_cascade_final::<3>(close, period)?;
    Some(e.to_vec())
}

/// Resume [`trix`] from `state = [e0, e1, e2]` over rows `[from, n)`, bit-identical to a
/// full recompute: advance the 3-deep cascade per bar, then the 1-period ROC of the 3rd
/// stage, `(e2_new / e2_prev − 1)·100` — the exact `roc(., 1)` arithmetic, with `e2_prev`
/// the carried 3rd-stage value (= the cascade output at `from−1`). Reads only
/// `close[from..]`.
pub fn trix_resume(
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let k = 2.0 / (period as f64 + 1.0);
    let n = close.len();
    let mut e = [state[0], state[1], state[2]];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &close[from..n] {
        let prev = e[2]; // the cascade output at the previous row (ROC's denominator)
        kernels::ema_cascade_step(&mut e, x, k);
        out.push((e[2] / prev - 1.0) * 100.0);
    }
    (out, e.to_vec())
}

/// Aroon Up (TA-Lib AROON, up output). Lookback `period`.
pub fn aroon_up(high: &[f64], _low: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    if period == 0 || period >= n {
        return vec![f64::NAN; n];
    }
    let mut out = Vec::with_capacity(n);
    out.resize(period, f64::NAN);
    let factor = 100.0 / period as f64;
    let pf = period as f64;
    let mut today = period;
    let mut trailing = 0usize;
    let mut highest_idx = usize::MAX;
    let mut highest = 0.0;
    while today < n {
        let value = high[today];
        if highest_idx == usize::MAX || highest_idx < trailing {
            highest_idx = trailing;
            highest = high[trailing];
            for (off, &candidate) in high[(trailing + 1)..=today].iter().enumerate() {
                if candidate >= highest {
                    highest_idx = trailing + 1 + off;
                    highest = candidate;
                }
            }
        } else if value >= highest {
            highest_idx = today;
            highest = value;
        }
        out.push(factor * (pf - (today - highest_idx) as f64));
        trailing += 1;
        today += 1;
    }
    out
}

/// Aroon Down (TA-Lib AROON, down output). Lookback `period`.
pub fn aroon_down(_high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let n = low.len();
    if period == 0 || period >= n {
        return vec![f64::NAN; n];
    }
    let mut out = Vec::with_capacity(n);
    out.resize(period, f64::NAN);
    let factor = 100.0 / period as f64;
    let pf = period as f64;
    let mut today = period;
    let mut trailing = 0usize;
    let mut lowest_idx = usize::MAX;
    let mut lowest = 0.0;
    while today < n {
        let value = low[today];
        if lowest_idx == usize::MAX || lowest_idx < trailing {
            lowest_idx = trailing;
            lowest = low[trailing];
            for (off, &candidate) in low[(trailing + 1)..=today].iter().enumerate() {
                if candidate <= lowest {
                    lowest_idx = trailing + 1 + off;
                    lowest = candidate;
                }
            }
        } else if value <= lowest {
            lowest_idx = today;
            lowest = value;
        }
        out.push(factor * (pf - (today - lowest_idx) as f64));
        trailing += 1;
        today += 1;
    }
    out
}

/// Aroon Oscillator (TA-Lib AROONOSC): `factor·(hiIdx − loIdx)` — the algebraic identity
/// of `aroonUp − aroonDown` that TA-Lib evaluates directly, so it materialises neither
/// line nor a subtraction pass. Lookback `period`.
pub fn aroonosc(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    if period == 0 || period >= n {
        return vec![f64::NAN; n];
    }
    let mut out = Vec::with_capacity(n);
    // Every valid slot is written exactly once below. The loop invariants mirror
    // `willr`: `today < n`, `trailing <= today`, and rescans stay inside
    // `[trailing, today]`, so the hot tracker can use unchecked slice access.
    unsafe {
        out.set_len(n);
    }
    out[..period].fill(f64::NAN);
    let factor = 100.0 / period as f64;
    let high_ptr = high.as_ptr();
    let low_ptr = low.as_ptr();
    let out_ptr = out.as_mut_ptr();
    // Seed the first `[0, period]` window exactly as TA-Lib's first rescan does.
    // After that the hot loop only needs the expiry check, not a sentinel state.
    let mut highest_idx = 0usize;
    let mut lowest_idx = 0usize;
    let mut highest = unsafe { *high_ptr };
    let mut lowest = unsafe { *low_ptr };
    let mut idx = 1usize;
    while idx <= period {
        let low_value = unsafe { *low_ptr.add(idx) };
        if low_value <= lowest {
            lowest_idx = idx;
            lowest = low_value;
        }
        let high_value = unsafe { *high_ptr.add(idx) };
        if high_value >= highest {
            highest_idx = idx;
            highest = high_value;
        }
        idx += 1;
    }
    unsafe {
        *out_ptr.add(period) = factor * (highest_idx as isize - lowest_idx as isize) as f64;
    }
    let mut today = period + 1;
    let mut trailing = 1usize;
    let mut out_write = unsafe { out_ptr.add(today) };
    while today < n {
        let low_today = unsafe { *low_ptr.add(today) };
        if lowest_idx < trailing {
            lowest_idx = trailing;
            lowest = unsafe { *low_ptr.add(trailing) };
            // Manual index loops keep the rare rescan path close to TA-Lib's C loop
            // and avoid iterator setup cost on this short-window kernel.
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *low_ptr.add(idx) };
                if value <= lowest {
                    lowest_idx = idx;
                    lowest = value;
                }
                idx += 1;
            }
        } else if low_today <= lowest {
            lowest_idx = today;
            lowest = low_today;
        }

        let high_today = unsafe { *high_ptr.add(today) };
        if highest_idx < trailing {
            highest_idx = trailing;
            highest = unsafe { *high_ptr.add(trailing) };
            let mut idx = trailing + 1;
            while idx <= today {
                let value = unsafe { *high_ptr.add(idx) };
                if value >= highest {
                    highest_idx = idx;
                    highest = value;
                }
                idx += 1;
            }
        } else if high_today >= highest {
            highest_idx = today;
            highest = high_today;
        }
        unsafe {
            *out_write = factor * (highest_idx as isize - lowest_idx as isize) as f64;
            out_write = out_write.add(1);
        }
        trailing += 1;
        today += 1;
    }
    out
}

/// Money Flow Index (TA-Lib MFI): `100·posMF/(posMF+negMF)` over `period`, where each
/// bar's money flow `tp·volume` (tp = (h+l+c)/3) is signed by `tp`'s direction vs the
/// prior bar. A total flow below 1.0 yields 0 (TA-Lib's guard). First value at index
/// `period` (lookback = period). The sliding sums recompute the leaving bar's flow,
/// which is bit-identical to TA-Lib's stored value.
pub fn mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    if period == 0 || period + 1 > n {
        return vec![f64::NAN; n];
    }
    let mut out = Vec::with_capacity(n);
    // Warm-up slots are filled below and every valid slot `[period, n)` is written
    // exactly once by the two loops, so this avoids a full-buffer NaN initialization.
    unsafe {
        out.set_len(n);
    }
    out[..period].fill(f64::NAN);
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
    let mut pos_sum = 0.0;
    let mut neg_sum = 0.0;
    // Warm up the first window — bars `1..=period` fill ring slots `0..period`.
    let mut i = 1usize;
    while i <= period {
        let tp = (unsafe { *high.get_unchecked(i) }
            + unsafe { *low.get_unchecked(i) }
            + unsafe { *close.get_unchecked(i) })
            / 3.0;
        let flow = tp * unsafe { *volume.get_unchecked(i) };
        // Match TA-Lib's signed-flow branch shape; on real price series adjacent
        // typical-price direction is predictable enough to beat boolean-to-float masks.
        let (p, ng) = if tp > prev_tp {
            (flow, 0.0)
        } else if tp < prev_tp {
            (0.0, flow)
        } else {
            (0.0, 0.0)
        };
        ring_pos[i - 1] = p;
        ring_neg[i - 1] = ng;
        pos_sum += p;
        neg_sum += ng;
        prev_tp = tp;
        i += 1;
    }
    let total = pos_sum + neg_sum;
    out[period] = if total < 1.0 {
        0.0
    } else {
        100.0 * pos_sum / total
    };
    let mut slot = 0usize; // ring slot holding the oldest in-window bar (to evict next)
    i = period + 1;
    while i < n {
        pos_sum -= ring_pos[slot]; // bar leaving the window
        neg_sum -= ring_neg[slot];
        let tp = (unsafe { *high.get_unchecked(i) }
            + unsafe { *low.get_unchecked(i) }
            + unsafe { *close.get_unchecked(i) })
            / 3.0;
        let flow = tp * unsafe { *volume.get_unchecked(i) };
        let (p, ng) = if tp > prev_tp {
            (flow, 0.0)
        } else if tp < prev_tp {
            (0.0, flow)
        } else {
            (0.0, 0.0)
        };
        ring_pos[slot] = p; // bar entering
        ring_neg[slot] = ng;
        pos_sum += p;
        neg_sum += ng;
        let total = pos_sum + neg_sum;
        out[i] = if total < 1.0 {
            0.0
        } else {
            100.0 * pos_sum / total
        };
        prev_tp = tp;
        slot += 1;
        if slot == period {
            slot = 0;
        }
        i += 1;
    }
    out
}

/// Ultimate Oscillator (TA-Lib ULTOSC): a weighted blend of buying-pressure / true-
/// range ratios over three periods. Per bar, BP = `close − min(low, prevClose)` and
/// TR = the true range; with sliding sums `aₖ/bₖ` over `pₖ` bars, the value is
/// `100·(4·a₁/b₁ + 2·a₂/b₂ + a₃/b₃)/7` (a term is dropped if its TR-sum is ~0). The
/// 4/2/1 weights follow argument position, not magnitude. Default 7/14/28; lookback
/// = max period.
pub fn ultosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    let max_p = p1.max(p2).max(p3);
    if p1 == 0 || p2 == 0 || p3 == 0 || max_p >= n {
        return out;
    }
    // (buying pressure, true range) for bar i (needs the prior close, so i >= 1).
    let term = |i: usize| -> (f64, f64) {
        let prev_close = close[i - 1];
        let true_low = low[i].min(prev_close);
        let bp = close[i] - true_low;
        let mut tr = high[i] - low[i];
        tr = tr.max((prev_close - high[i]).abs());
        tr = tr.max((prev_close - low[i]).abs());
        (bp, tr)
    };
    let start = max_p; // first output index
    let mut bp_ring = vec![0.0; max_p];
    let mut tr_ring = vec![0.0; max_p];
    let (mut a1, mut b1, mut a2, mut b2, mut a3, mut b3) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // Cache BP/TR terms in one max-period ring so the hot loop computes today's
    // term once instead of recomputing three trailing terms after every output.
    for i in (start - max_p + 1)..start {
        let (a, b) = term(i);
        bp_ring[i % max_p] = a;
        tr_ring[i % max_p] = b;
        if i >= start - p1 + 1 {
            a1 += a;
            b1 += b;
        }
        if i >= start - p2 + 1 {
            a2 += a;
            b2 += b;
        }
        a3 += a;
        b3 += b;
    }
    let (mut t1, mut t2, mut t3) = (start - p1 + 1, start - p2 + 1, start - p3 + 1);
    for today in start..n {
        let (a, b) = term(today);
        let slot = today % max_p;
        bp_ring[slot] = a;
        tr_ring[slot] = b;
        a1 += a;
        a2 += a;
        a3 += a;
        b1 += b;
        b2 += b;
        b3 += b;
        let mut output = 0.0;
        if b1.abs() >= 1e-14 {
            output += 4.0 * (a1 / b1);
        }
        if b2.abs() >= 1e-14 {
            output += 2.0 * (a2 / b2);
        }
        if b3.abs() >= 1e-14 {
            output += a3 / b3;
        }
        let at = bp_ring[t1 % max_p];
        let bt = tr_ring[t1 % max_p];
        a1 -= at;
        b1 -= bt;
        t1 += 1;
        let at = bp_ring[t2 % max_p];
        let bt = tr_ring[t2 % max_p];
        a2 -= at;
        b2 -= bt;
        t2 += 1;
        let at = bp_ring[t3 % max_p];
        let bt = tr_ring[t3 % max_p];
        a3 -= at;
        b3 -= bt;
        t3 += 1;
        out[today] = 100.0 * (output / 7.0);
    }
    out
}
