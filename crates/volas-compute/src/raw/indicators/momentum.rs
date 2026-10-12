use crate::accum::Csum;
use crate::guard::{all_finite, own_sums, state_after, window_sums, Order, Slide, Window};
use crate::missing;
use crate::raw::kernels;

// ---------------------------------------------------------------------------
// Momentum — change relative to the price `period` bars earlier
// ---------------------------------------------------------------------------

/// The momentum / rate-of-change family: each row against the row `period` bars back.
/// The one owner of the family's per-row rule — the batch kernels, TRIX's 1-period ROC
/// and the live resumes all evaluate [`Momentum::value`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Momentum {
    /// `current − prior` (TA-Lib MOM).
    Mom,
    /// `100·(current / prior − 1)` (TA-Lib ROC).
    Roc,
    /// `current / prior − 1` (TA-Lib ROCP).
    Rocp,
    /// `current / prior` (TA-Lib ROCR).
    Rocr,
    /// `100·current / prior` (TA-Lib ROCR100).
    Rocr100,
}

impl Momentum {
    /// The family member named `name` (`mom`, `roc`, `rocp`, `rocr`, `rocr100`).
    pub fn of(name: &str) -> Option<Momentum> {
        Some(match name {
            "mom" => Momentum::Mom,
            "roc" => Momentum::Roc,
            "rocp" => Momentum::Rocp,
            "rocr" => Momentum::Rocr,
            "rocr100" => Momentum::Rocr100,
            _ => return None,
        })
    }

    /// The value of `current` against `prior`. A ratio over a prior of exactly zero is
    /// `0.0` (TA-Lib's divide-by-zero guard) — for present values only: a missing input
    /// is missing (the per-row term rule, [`missing::term`]), so the zero-base branch
    /// never turns a missing value into `0`.
    #[inline(always)]
    pub fn value(self, current: f64, prior: f64) -> f64 {
        let v = match self {
            Momentum::Mom => current - prior,
            _ if prior == 0.0 => 0.0,
            Momentum::Roc => (current / prior - 1.0) * 100.0,
            Momentum::Rocp => current / prior - 1.0,
            Momentum::Rocr => current / prior,
            Momentum::Rocr100 => current / prior * 100.0,
        };
        missing::term(&[current, prior], v)
    }

    /// The value at row `i` of `data` (NaN while `i < period`).
    #[inline]
    pub fn at(self, data: &[f64], period: usize, i: usize) -> f64 {
        if i < period { f64::NAN } else { self.value(data[i], data[i - period]) }
    }

    /// The whole series: NaN warm-up `[0, period)`, then [`Momentum::value`] per row —
    /// the valid region written once (no `vec![NaN; n]` re-write; these tiny
    /// division-bound kernels are dominated by their result-buffer writes).
    fn series(self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if period >= n {
            return vec![f64::NAN; n];
        }
        let mut out = vec![f64::NAN; period];
        out.extend(data[period..].iter().zip(&data[..n - period]).map(|(&c, &p)| self.value(c, p)));
        out
    }
}

/// Momentum: `data[i] - data[i-period]` (TA-Lib MOM). NaN during warm-up.
pub fn mom(data: &[f64], period: usize) -> Vec<f64> {
    Momentum::Mom.series(data, period)
}

/// One rate of change `100·(current/prior − 1)` ([`Momentum::Roc`]), for TRIX's full
/// computation and resumes.
#[inline(always)]
fn roc_value(current: f64, prior: f64) -> f64 {
    Momentum::Roc.value(current, prior)
}

/// Rate of change: `100 * (data/data[period ago] - 1)` (TA-Lib ROC).
pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    if period >= n {
        return vec![f64::NAN; n];
    }
    // ROC is a coverage hot spot by itself, so keep the canonical line specialized:
    // one explicit loop avoids the generic closure/iterator adapter used by ROCP/ROCR.
    // Single-write (D2): NaN warm-up `[0, period)`, valid region written once via the
    // walking output pointer (no prefill memset); `out.finish()` proves it was filled.
    let mut out = crate::buf::OutBuf::warmup(n, period);
    let data_ptr = data.as_ptr();
    let out_ptr = out.ptr();
    let mut cur_ptr = unsafe { data_ptr.add(period) };
    let mut prior_ptr = data_ptr;
    let mut out_write = unsafe { out_ptr.add(period) };
    let mut remaining = n - period;
    while remaining > 0 {
        // The three pointers advance together over `[period, n)`, `[0, n-period)`,
        // and the valid output region. This keeps the hot loop in TA-Lib's
        // trailing-index shape without recomputing `i` and `i-period` addresses.
        let value = roc_value(unsafe { *cur_ptr }, unsafe { *prior_ptr });
        unsafe {
            *out_write = value;
            cur_ptr = cur_ptr.add(1);
            prior_ptr = prior_ptr.add(1);
            out_write = out_write.add(1);
        }
        remaining -= 1;
    }
    out.finish()
}

/// Rate of change percentage: `data/data[period ago] - 1` (TA-Lib ROCP).
pub fn rocp(data: &[f64], period: usize) -> Vec<f64> {
    Momentum::Rocp.series(data, period)
}

/// Rate of change ratio: `data/data[period ago]` (TA-Lib ROCR).
pub fn rocr(data: &[f64], period: usize) -> Vec<f64> {
    Momentum::Rocr.series(data, period)
}

/// Rate of change ratio ×100: `100 * data/data[period ago]` (TA-Lib ROCR100).
pub fn rocr100(data: &[f64], period: usize) -> Vec<f64> {
    Momentum::Rocr100.series(data, period)
}

/// Williams %R: `-100 * (HH - close) / (HH - LL)` over `period`, where HH/LL are
/// the highest high / lowest low (TA-Lib WILLR). A flat range (HH == LL) yields 0; a
/// window holding a missing value is NaN. Lookback `period-1`; the tracker preserves
/// TA-Lib's window and tie semantics.
pub fn willr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    // The tracker reads its inputs unchecked: inputs of unequal length line up no row.
    if !super::same_len(&[high, low, close]) {
        return Vec::new();
    }
    let n = close.len();
    if period == 0 || period > n {
        return vec![f64::NAN; n];
    }
    let lookback = period - 1;
    // Single-write (D2): NaN warm-up, valid region written once via `out.set`.
    let mut out = crate::buf::OutBuf::warmup(n, lookback);
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

        out.set(today, willr_value(highest, lowest, unsafe { *close.get_unchecked(today) }));
        trailing += 1;
        today += 1;
    }
    let mut out = out.finish();
    // The finite-window rule: the tracker steps over a missing high / low, and a flat
    // range reads no close — so blank every window holding one, and a missing close.
    missing::blank_windows(&mut out, high, period);
    missing::blank_windows(&mut out, low, period);
    missing::blank_windows(&mut out, close, 1);
    out
}

/// Williams %R from a window's extremes and the close: `(HH − close) · (−100 / (HH − LL))`,
/// `0` for a flat range — TA-Lib's formula with one range reciprocal instead of its
/// `(range / −100)` and second division. The extremes' zeros are made `+0` first
/// ([`kernels::canonical_zero`]): which signed zero the tracker kept depends on where it
/// last rebuilt, and a recomputed window (the live refresh) rebuilds elsewhere.
#[inline(always)]
fn willr_value(highest: f64, lowest: f64, close: f64) -> f64 {
    let (highest, lowest) = (kernels::canonical_zero(highest), kernels::canonical_zero(lowest));
    let range = highest - lowest;
    if range != 0.0 { (highest - close) * (-100.0 / range) } else { 0.0 }
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
/// TA-Lib — no divide-by-zero guard.) Row 0 is at absolute position `origin`.
pub fn imi(open: &[f64], close: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let n = close.len();
    if period == 0 || period > n {
        return vec![f64::NAN; n];
    }
    let lookback = period - 1;
    // Each bar's up / down move; the window sums slide (add the entering bar, drop the leaving
    // one) -> O(n) instead of re-scanning the window each bar, exact ([`Csum`], re-anchored
    // at absolute positions) so a window's sums are its own however long the slide.
    let updown = |i: usize| -> (f64, f64) {
        let [u, d] = imi_updown(open[i], close[i]);
        (u, d)
    };
    let (mut up_sum, mut down_sum) = (Csum::default(), Csum::default());
    for i in 0..lookback {
        let (u, d) = updown(i);
        up_sum.add(u);
        down_sum.add(d);
    }
    let mut out = crate::buf::OutBuf::warmup(n, lookback);
    let mut clean = true;
    for today in lookback..n {
        let (u, d) = updown(today);
        up_sum.add(u);
        down_sum.add(d);
        if !(up_sum.is_exact() && down_sum.is_exact()) {
            clean &= up_sum.value().is_finite() && down_sum.value().is_finite();
            if clean {
                [up_sum, down_sum] = Csum::of_rows((today - lookback..=today).map(|j| {
                    let (u, d) = updown(j);
                    [u, d]
                }));
            }
        }
        out.set(today, imi_value(up_sum.value(), down_sum.value()));
        let (lu, ld) = updown(today - lookback);
        up_sum.sub(lu);
        down_sum.sub(ld);
    }
    if clean && up_sum.value().is_finite() && down_sum.value().is_finite() {
        return out.finish();
    }
    // A non-finite bar entered: window-local careful path (see `crate::guard`).
    let mut out = crate::buf::OutBuf::warmup(n, lookback);
    window_sums(n, period, Order::AddReadDrop, origin, |i| {
        let (u, d) = updown(i);
        [u, d]
    }, |i, [u, d]| out.set(i, imi_value(u, d)));
    out.finish()
}

/// Commodity Channel Index (TA-Lib CCI): `(tp − SMA(tp)) / (0.015 · meanDev)` over
/// `period`, where `tp = (high+low+close)/3` and `meanDev` is the mean absolute
/// deviation of `tp` from its average. A zero numerator or zero deviation yields 0
/// (TA-Lib's guard). Row 0 is at absolute position `origin`. Lookback `period-1`.
/// O(n·period), as in TA-Lib.
pub fn cci(high: &[f64], low: &[f64], close: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let n = close.len();
    if period == 0 || period > n {
        return vec![f64::NAN; n];
    }
    let tp: Vec<f64> = (0..n)
        .map(|i| typical(high, low, close, i))
        .collect();
    // The window mean is a sliding sum (add the entering bar, drop the leaving one) — O(1)
    // per bar instead of re-summing the window. The mean-absolute-deviation still scans the
    // window (each `|x - avg|` depends on the just-computed mean, so it cannot slide). The
    // sliding sum is exact ([`Csum`], re-summed where it could not hold its sum), so a window's
    // mean is its own however long the slide. The value at row `i` from its window mean.
    let p = period as f64;
    let value = |i: usize, avg: f64| cci_value(|j| tp[j], period, i, avg);
    let mut sum = Csum::default();
    for &x in &tp[..period - 1] {
        sum.add(x);
    }
    let mut out = crate::buf::OutBuf::warmup(n, period - 1);
    let mut clean = true;
    for i in (period - 1)..n {
        sum.add(tp[i]);
        if !sum.is_exact() {
            clean &= sum.value().is_finite();
            if clean {
                sum = Csum::of(tp[i + 1 - period..=i].iter().copied());
            }
        }
        out.set(i, value(i, sum.mean(p)));
        sum.sub(tp[i + 1 - period]);
    }
    if clean && sum.value().is_finite() {
        return out.finish();
    }
    // A non-finite bar entered: window-local careful path (see `crate::guard`).
    let mut slide = Slide::<1>::at(origin);
    let tpf = |j: usize| tp[j];
    (0..n).map(|i| cci_row(&mut slide, period, i, tpf)).collect()
}

/// One row of [`cci`] over the typical prices `tp`, resumed from `slide` (add, read,
/// drop) — the careful slide, which the fast loop follows on clean windows.
fn cci_row(slide: &mut Slide<1>, period: usize, i: usize, tp: impl Fn(usize) -> f64) -> f64 {
    let p = period as f64;
    match slide.row(Order::AddReadDrop, period, i, |j| all_finite([tp(j)])) {
        Window::Partial => f64::NAN,
        Window::Sums([sum]) => cci_value(&tp, period, i, sum.mean(p)),
        Window::Bad => cci_value(&tp, period, i, own_sums(i + 1 - period..=i, |j| [tp(j)])[0] / p),
    }
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
/// stage — [`roc`]'s own rule (`roc_value`, its zero-base guard included), with `e2_prev`
/// the carried 3rd-stage value (= the cascade output at `from−1`). Reads only
/// `close[from..]`.
pub fn trix_resume(
    close: &[f64],
    period: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut e = [state[0], state[1], state[2]];
    let mut out = Vec::with_capacity(n.saturating_sub(from));
    for &x in &close[from..n] {
        let prev = e[2]; // the cascade output at the previous row (ROC's denominator)
        kernels::ema_cascade_step(&mut e, x, period);
        out.push(roc_value(e[2], prev));
    }
    (out, e.to_vec())
}

/// Aroon Up (TA-Lib AROON, up output): NaN while the `period + 1`-bar window holds a
/// missing high. Lookback `period`.
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
    missing::blank_windows(&mut out, high, period + 1);
    out
}

/// Aroon Down (TA-Lib AROON, down output): NaN while the `period + 1`-bar window holds
/// a missing low. Lookback `period`.
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
    missing::blank_windows(&mut out, low, period + 1);
    out
}

/// Aroon Oscillator (TA-Lib AROONOSC): `factor·(hiIdx − loIdx)` — the algebraic identity
/// of `aroonUp − aroonDown` that TA-Lib evaluates directly, so it materialises neither
/// line nor a subtraction pass. NaN while the window holds a missing high or low.
/// Lookback `period`.
pub fn aroonosc(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    // The tracker reads its inputs unchecked: inputs of unequal length line up no row.
    if !super::same_len(&[high, low]) {
        return Vec::new();
    }
    let n = high.len();
    if period == 0 || period >= n {
        return vec![f64::NAN; n];
    }
    // The loop invariants mirror `willr`: `today < n`, `trailing <= today`, and
    // rescans stay inside `[trailing, today]`, so the hot tracker can use
    // unchecked slice access.
    // Single-write (D2): NaN warm-up `[0, period)`, valid region written once via the
    // output pointer below (no prefill memset); `out.finish()` proves it was filled.
    let mut out = crate::buf::OutBuf::warmup(n, period);
    let factor = 100.0 / period as f64;
    let high_ptr = high.as_ptr();
    let low_ptr = low.as_ptr();
    let out_ptr = out.ptr();
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
    let mut out = out.finish();
    missing::blank_windows(&mut out, high, period + 1);
    missing::blank_windows(&mut out, low, period + 1);
    out
}

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
    // The flow loop reads its inputs unchecked: inputs of unequal length line up no row.
    if !super::same_len(&[high, low, close, volume]) {
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
    window_sums(n - 1, period, Order::DropAdd, origin, flow, |j, s| out.set(j + 1, mfi_value(s)));
    out.finish()
}

/// Ultimate Oscillator (TA-Lib ULTOSC): a weighted blend of buying-pressure / true-
/// range ratios over three periods. Per bar, BP = `close − min(low, prevClose)` and
/// TR = the true range; with sliding sums `aₖ/bₖ` over `pₖ` bars, the value is
/// `100·(4·a₁/b₁ + 2·a₂/b₂ + a₃/b₃)/7` (a term is dropped if its TR-sum is ~0). The
/// 4/2/1 weights follow argument position, not magnitude. Row 0 is at absolute position
/// `origin` (term `j`, of bar `j + 1`, at `origin + j`). Default 7/14/28; lookback
/// = max period. Each period's sums slide window-locally ([`Slide`], see
/// `crate::guard`), one row at a time exactly as [`ultosc_step`] resumes them.
pub fn ultosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    origin: usize,
) -> Vec<f64> {
    let n = close.len();
    let periods = [p1, p2, p3];
    if periods.contains(&0) || p1.max(p2).max(p3) >= n {
        return vec![f64::NAN; n];
    }
    // The terms once (term `j` is bar `j + 1`'s): every window reads each twice.
    let terms: Vec<[f64; 2]> = (1..n).map(|i| ultosc_term(high, low, close, i)).collect();
    let mut slides = [Slide::<2>::at(origin); 3];
    (0..n).map(|row| ultosc_row(|j| terms[j], periods, row, &mut slides)).collect()
}

/// ULTOSC's `[buying pressure, true range]` of bar `i >= 1`: `close − min(low, prev
/// close)` and [`super::tr_at`]. Both are missing when an input the bar reads is (the
/// per-row term rule, [`missing::term`]): `f64::min` would step over a missing prior
/// close.
#[inline]
fn ultosc_term(high: &[f64], low: &[f64], close: &[f64], i: usize) -> [f64; 2] {
    let (c, l, prev_close) = (close[i], low[i], close[i - 1]);
    let bp = c - l.min(prev_close);
    missing::terms(&[c, l, prev_close], [bp, super::tr_at(high, low, close, i)])
}

/// One row of [`ultosc`] (its terms are term `row - 1`; add, read, drop per period),
/// advancing the three periods' slides.
fn ultosc_row(term: impl Fn(usize) -> [f64; 2], periods: [usize; 3], row: usize, slides: &mut [Slide<2>; 3]) -> f64 {
    if row == 0 {
        return f64::NAN;
    }
    let j = row - 1;
    let mut sums = [0.0; 6];
    for (k, (slide, p)) in slides.iter_mut().zip(periods).enumerate() {
        let w = match slide.row(Order::AddReadDrop, p, j, |i| all_finite(term(i))) {
            Window::Partial => [f64::NAN; 2],
            Window::Sums(s) => s.map(|c| c.value()),
            Window::Bad => own_sums(j + 1 - p..=j, &term),
        };
        sums[2 * k..2 * k + 2].copy_from_slice(&w);
    }
    if row >= periods[0].max(periods[1]).max(periods[2]) { ultosc_value(sums) } else { f64::NAN }
}

/// [`ultosc`]'s state after the rows (row 0 at absolute position `origin`): its three
/// periods' [`Slide`]s.
#[allow(clippy::too_many_arguments)]
pub fn ultosc_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    origin: usize,
) -> Option<Vec<f64>> {
    state_after(Slide::<2>::initial(origin).repeat(3), close.len(), |row, prev, next| {
        ultosc_step(high, low, close, p1, p2, p3, row, prev, next)
    })
}

/// One row of [`ultosc`] at `row`, from `state` into `next`.
#[allow(clippy::too_many_arguments)]
pub fn ultosc_step(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    const S: usize = Slide::<2>::STATE;
    let periods = [p1, p2, p3];
    if periods.contains(&0) || row >= close.len() || state.len() != 3 * S || next.len() != 3 * S {
        return None;
    }
    let mut slides = [Slide::<2>::at(0); 3];
    for (k, slide) in slides.iter_mut().enumerate() {
        *slide = Slide::from_state(&state[k * S..(k + 1) * S])?;
    }
    let value = ultosc_row(|j| ultosc_term(high, low, close, j + 1), periods, row, &mut slides);
    for (k, slide) in slides.iter().enumerate() {
        slide.write_state(&mut next[k * S..(k + 1) * S]);
    }
    Some(value)
}

/// [`imi`]'s state after the rows (row 0 at absolute position `origin`): the [`Slide`]
/// of its up / down sums.
pub fn imi_state(open: &[f64], close: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<2>::initial(origin), close.len(), |row, prev, next| {
        imi_step(open, close, period, row, prev, next)
    })
}

/// One row of [`imi`] at `row` (add, read, drop), from `state` into `next`.
pub fn imi_step(open: &[f64], close: &[f64], period: usize, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    if period == 0 || row >= close.len() || open.len() != close.len() || next.len() != Slide::<2>::STATE {
        return None;
    }
    let term = |i: usize| imi_updown(open[i], close[i]);
    let mut slide = Slide::<2>::from_state(state)?;
    let value = match slide.row(Order::AddReadDrop, period, row, |j| all_finite(term(j))) {
        Window::Partial => f64::NAN,
        Window::Sums([u, d]) => imi_value(u.value(), d.value()),
        Window::Bad => {
            let [u, d] = own_sums(row + 1 - period..=row, term);
            imi_value(u, d)
        }
    };
    slide.write_state(next);
    Some(value)
}

/// IMI's `[up, down]` move of a bar: `close − open` on an up bar, `open − close` else.
#[inline]
fn imi_updown(o: f64, c: f64) -> [f64; 2] {
    if c > o { [c - o, 0.0] } else { [0.0, o - c] }
}

/// IMI from a window's up / down sums: `100·up/(up+down)`.
#[inline]
fn imi_value(up: f64, down: f64) -> f64 {
    100.0 * (up / (up + down))
}

/// ULTOSC from its three window sums `[a1, b1, a2, b2, a3, b3]`: `100·(4·a₁/b₁ +
/// 2·a₂/b₂ + a₃/b₃)/7`, dropping a term whose TR-sum is ~0. A NaN sum is kept (it is
/// not "~0"), so a window holding a bad bar is NaN rather than silently losing a term.
#[inline]
fn ultosc_value([a1, b1, a2, b2, a3, b3]: [f64; 6]) -> f64 {
    let mut output = 0.0;
    if b1.abs() >= 1e-14 || b1.is_nan() {
        output += 4.0 * (a1 / b1);
    }
    if b2.abs() >= 1e-14 || b2.is_nan() {
        output += 2.0 * (a2 / b2);
    }
    if b3.abs() >= 1e-14 || b3.is_nan() {
        output += a3 / b3;
    }
    100.0 * (output / 7.0)
}


/// Scalar single-row twin of [`trix_resume`]: the value at `row` only, from
/// `state = [e0, e1, e2]` (the 3-deep cascade as of `row-1`), zero-alloc and
/// bit-identical to the `trix_resume` loop body at index `row`. Advances the
/// 3-deep cascade one bar with `close[row]` by [`kernels::ema_cascade_step`], then the
/// 1-period ROC of the 3rd stage by [`roc`]'s own rule (`roc_value`), with
/// `e2_prev = state[2]`. The advanced
/// cascade (the state as of `row`) is written into `next`.
pub fn trix_resume_one(
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if state.len() != 3 || next.len() != 3 || row >= close.len() {
        return None;
    }
    let mut e = [state[0], state[1], state[2]];
    let prev = e[2]; // cascade output at row-1 (ROC's denominator)
    kernels::ema_cascade_step(&mut e, close[row], period);
    next.copy_from_slice(&e);
    Some(roc_value(e[2], prev))
}


// ---------------------------------------------------------------------------
// Resumable forms: the running sums of each full computation, in its exact operation
// order, so a resume of one row at a time is bit-identical to recomputing the column.
// ---------------------------------------------------------------------------

/// The typical price `(high + low + close) / 3` at `i`.
#[inline]
fn typical(high: &[f64], low: &[f64], close: &[f64], i: usize) -> f64 {
    (high[i] + low[i] + close[i]) / 3.0
}

/// CCI's value at row `i` from its window's mean typical price.
#[inline]
fn cci_value(tp: impl Fn(usize) -> f64, period: usize, i: usize, avg: f64) -> f64 {
    let p = period as f64;
    let sum_dev: f64 = (i + 1 - period..=i).map(|j| (tp(j) - avg).abs()).sum();
    let num = tp(i) - avg;
    if num != 0.0 && sum_dev != 0.0 {
        num / (0.015 * (sum_dev / p))
    } else {
        0.0
    }
}

/// [`cci`]'s state after the rows: the [`Slide`] of the typical-price sum.
pub fn cci_state(high: &[f64], low: &[f64], close: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<1>::initial(origin), close.len(), |row, prev, next| {
        cci_step(high, low, close, period, row, prev, next)
    })
}

/// One row of [`cci`] at `row` (add, read, drop), from `state` into `next`.
pub fn cci_step(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || row >= close.len() || next.len() != Slide::<1>::STATE {
        return None;
    }
    let tp = |j: usize| typical(high, low, close, j);
    let mut slide = Slide::<1>::from_state(state)?;
    let value = cci_row(&mut slide, period, row, tp);
    slide.write_state(next);
    Some(value)
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
