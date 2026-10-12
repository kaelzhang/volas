// ---------------------------------------------------------------------------
// Directional movement — Wilder's +DM/-DM/TR, ±DI, DX, ADX, ADXR
// ---------------------------------------------------------------------------
//
// All seven derive from one-period directional movement and true range, smoothed by
// Wilder's *sum* recurrence (not the SMA-seeded average used by ATR): the seed is the
// sum of the first `period-1` one-period values, then `s = s - s/period + term`.
// Faithful port of TA-Lib (which performs no integer rounding).

/// One-period +DM / -DM at bar `i` (needs the prior bar). Per Wilder: the larger of
/// the up-move (`high-prevHigh`) and down-move (`prevLow-low`) wins, the other is 0;
/// ties and non-positive moves yield 0 for both.
#[inline]
fn dm1(high: &[f64], low: &[f64], i: usize) -> (f64, f64) {
    let diff_p = high[i] - high[i - 1];
    let diff_m = low[i - 1] - low[i];
    let plus = if diff_p > 0.0 && diff_p > diff_m {
        diff_p
    } else {
        0.0
    };
    let minus = if diff_m > 0.0 && diff_p < diff_m {
        diff_m
    } else {
        0.0
    };
    (plus, minus)
}

use super::tr_at; // the one true-range term (missing when an input it reads is)

/// Wilder "sum" smoothing of a one-period `term`: seed `= Σ term(i)` over bars
/// `[1, period-1]`, placed at index `period-1`; then `s = s - s/period + term(i)`.
/// NaN before index `period-1` — present from `period` rows on, like TA-Lib.
fn wilder_sum(n: usize, period: usize, term: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut s = 0.0;
    for i in 1..period {
        s += term(i);
    }
    out[period - 1] = s;
    // `s - s/period + term` == `s*(1 - 1/period) + term`: precompute the factor and
    // fuse (mul_add), taking the per-element division off the recurrence's critical
    // path. Wilder smoothing is contractive, so the ~1e-16 reassociation decays —
    // within the 1e-9 parity tolerance. Speeds every directional indicator.
    let a = 1.0 - 1.0 / period as f64;
    #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
    for i in period..n {
        s = s.mul_add(a, term(i));
        out[i] = s;
    }
    out
}

/// Wilder-smoothed +DM (TA-Lib PLUS_DM). Lookback `period-1`.
pub fn plus_dm(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    wilder_sum(high.len(), period, |i| dm1(high, low, i).0)
}

/// Wilder-smoothed -DM (TA-Lib MINUS_DM). Lookback `period-1`.
pub fn minus_dm(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    wilder_sum(high.len(), period, |i| dm1(high, low, i).1)
}

/// `100 · smoothedDM / smoothedTR`, emitted from index `period` (one bar after the DM
/// seed), with a ~0 TR yielding 0. Shared by ±DI.
fn di(dm_sm: &[f64], tr_sm: &[f64], period: usize) -> Vec<f64> {
    let n = dm_sm.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        out[i] = di_val(dm_sm[i], tr_sm[i]);
    }
    out
}

/// `100·dm_sm/tr_sm` with TA-Lib's ~0-TR guard — one row's ±DI, the one owner of that
/// rule for the full computation and the resumes.
#[inline]
pub(super) fn di_val(dm_sm: f64, tr_sm: f64) -> f64 {
    if tr_sm.abs() < 1e-14 {
        0.0
    } else {
        100.0 * dm_sm / tr_sm
    }
}

/// One row's DX from the +DM / −DM / TR sums, with TA-Lib's ~0-TR and ~0-DI-sum guards —
/// the one owner of that rule for the full computation and the resumes.
#[inline]
pub(super) fn dx_val(sp: f64, sm: f64, st: f64) -> f64 {
    if st.abs() < 1e-14 {
        0.0
    } else {
        let plus_di = 100.0 * sp / st;
        let minus_di = 100.0 * sm / st;
        let sum = plus_di + minus_di;
        if sum.abs() < 1e-14 {
            0.0
        } else {
            100.0 * (minus_di - plus_di).abs() / sum
        }
    }
}

/// +DI (TA-Lib PLUS_DI). Lookback `period`.
pub fn plus_di(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let sp = wilder_sum(high.len(), period, |i| dm1(high, low, i).0);
    let st = wilder_sum(high.len(), period, |i| tr_at(high, low, close, i));
    di(&sp, &st, period)
}

/// -DI (TA-Lib MINUS_DI). Lookback `period`.
pub fn minus_di(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let sm = wilder_sum(high.len(), period, |i| dm1(high, low, i).1);
    let st = wilder_sum(high.len(), period, |i| tr_at(high, low, close, i));
    di(&sm, &st, period)
}

/// Fused Wilder sums of +DM, −DM and TR in a single pass — `dm1` (which yields both
/// directional moves) is evaluated once per bar instead of twice, and the three
/// division-free recurrences share one traversal. Bit-identical to three
/// [`wilder_sum`] calls.
fn dm_tr_sums(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let (mut sp, mut sm, mut st) = (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    if period == 0 || period >= n {
        return (sp, sm, st);
    }
    let (mut p, mut m, mut t) = (0.0, 0.0, 0.0);
    for i in 1..period {
        let (dp, dm) = dm1(high, low, i);
        p += dp;
        m += dm;
        t += tr_at(high, low, close, i);
    }
    sp[period - 1] = p;
    sm[period - 1] = m;
    st[period - 1] = t;
    let a = 1.0 - 1.0 / period as f64;
    for i in period..n {
        let (dp, dm) = dm1(high, low, i);
        p = p.mul_add(a, dp);
        m = m.mul_add(a, dm);
        t = t.mul_add(a, tr_at(high, low, close, i));
        sp[i] = p;
        sm[i] = m;
        st[i] = t;
    }
    (sp, sm, st)
}

/// Directional Movement Index `DX = 100·|+DI − −DI| / (+DI + −DI)` (TA-Lib DX); a ~0
/// TR or ~0 DI-sum yields 0. Lookback `period`.
pub fn dx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let (sp, sm, st) = dm_tr_sums(high, low, close, period);
    let n = sp.len();
    let mut out = vec![f64::NAN; n];
    for i in period..n {
        out[i] = dx_val(sp[i], sm[i], st[i]);
    }
    out
}

/// Average Directional Movement Index (TA-Lib ADX): SMA-seeded Wilder *average* of DX
/// (`ADX[2p-1] = mean(DX[p..2p-1])`, then `ADX = (ADX_prev·(p-1) + DX)/p`).
/// Lookback `2·period-1`.
pub fn adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let dxv = dx(high, low, close, period);
    let n = dxv.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 {
        return out;
    }
    let first = 2 * period - 1;
    if first >= n {
        return out;
    }
    let pf = period as f64;
    let mut sum = 0.0;
    for v in dxv.iter().take(2 * period).skip(period) {
        sum += v;
    }
    let mut prev = sum / pf;
    out[first] = prev;
    // `(prev*(period-1) + dx)/period` as `prev*a + dx*b` (precomputed reciprocals,
    // fused): no per-element division on the ADX recurrence. Contractive -> within
    // the 1e-9 parity tolerance.
    let a = (pf - 1.0) / pf;
    let b = 1.0 / pf;
    for i in (2 * period)..n {
        prev = prev.mul_add(a, dxv[i] * b);
        out[i] = prev;
    }
    out
}

/// Average Directional Movement Index Rating: `(ADX[i] + ADX[i−(period−1)]) / 2`
/// (TA-Lib ADXR). Lookback `3·period-2`.
pub fn adxr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let a = adx(high, low, close, period);
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 {
        return out;
    }
    let off = period - 1;
    for i in (3 * period - 2)..n {
        out[i] = (a[i] + a[i - off]) / 2.0;
    }
    out
}

mod resume;
pub use resume::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::indicators::test_support::*;

    /// Every directional resume, fed the carried Wilder accumulator(s) of a full compute
    /// over the head, reproduces the tail of a full compute over the whole input —
    /// bit-for-bit. Iterating `from` over an oscillating OHLC series advances the fused
    /// `dm1`/`tr_at` recurrences through their steady state for all seven families.
    #[test]
    fn directional_resume_is_bit_identical_to_full() {
        let (high, low, close) = ohlc(120);
        let p = 14usize;

        // ±DM warm up at `period-1`; ±DI/DX at `period`; ADX at `2p-1`; ADXR at `3p-2`.
        // Pick `from` values comfortably past each family's seed (the largest seed is
        // ADXR's 3*14-2 = 40), so every head produces a valid carried state.
        for &from in &[41usize, 45, 60, 90, 119] {
            let h = &high[..from];
            let l = &low[..from];
            let c = &close[..from];

            let st = plus_dm_final_state(h, l, p).unwrap();
            let (tail, _) = plus_dm_resume(&high, &low, p, from, &st).unwrap();
            assert_bits(&tail, &plus_dm(&high, &low, p)[from..], "plus_dm");

            let st = minus_dm_final_state(h, l, p).unwrap();
            let (tail, _) = minus_dm_resume(&high, &low, p, from, &st).unwrap();
            assert_bits(&tail, &minus_dm(&high, &low, p)[from..], "minus_dm");

            let st = plus_di_final_state(h, l, c, p).unwrap();
            let (tail, _) = plus_di_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &plus_di(&high, &low, &close, p)[from..], "plus_di");

            let st = minus_di_final_state(h, l, c, p).unwrap();
            let (tail, _) = minus_di_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &minus_di(&high, &low, &close, p)[from..], "minus_di");

            let st = dx_final_state(h, l, c, p).unwrap();
            let (tail, _) = dx_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &dx(&high, &low, &close, p)[from..], "dx");

            let st = adx_final_state(h, l, c, p).unwrap();
            let (tail, _) = adx_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &adx(&high, &low, &close, p)[from..], "adx");

            let st = adxr_final_state(h, l, c, p).unwrap();
            let (tail, _) = adxr_resume(&high, &low, &close, p, from, &st).unwrap();
            assert_bits(&tail, &adxr(&high, &low, &close, p)[from..], "adxr");
        }
    }

    /// `from == 0` resumes decline (no prior bar to read the one-period term from) — the
    /// early `None` arm of every `*_resume`.
    #[test]
    fn directional_resume_declines_at_from_zero() {
        let (high, low, close) = ohlc(60);
        let p = 14usize;
        let dummy = vec![0.0; 8]; // longer than any family's state, unread on the None path
        assert!(plus_dm_resume(&high, &low, p, 0, &dummy).is_none()); // :284
        assert!(minus_dm_resume(&high, &low, p, 0, &dummy).is_none()); // :305
        assert!(plus_di_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :378
        assert!(minus_di_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :409
        assert!(dx_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :457
        assert!(adx_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :525
        assert!(adxr_resume(&high, &low, &close, p, 0, &dummy).is_none()); // :605
    }

    /// `*_final_state` declines before the indicator seeds, so the caller keeps the
    /// full-recompute fallback: `period >= n` for the Wilder-sum families, `period == 0`
    /// and the staged `2p-1 >= n` / `3p-2 >= n` seeds for ADX / ADXR.
    #[test]
    fn directional_final_state_declines_before_seed() {
        let (high, low, close) = ohlc(120);
        // period >= n -> the Wilder sum never seeds.
        assert!(plus_dm_final_state(&high, &low, 200).is_none()); // :236
        assert!(minus_dm_final_state(&high, &low, 200).is_none());
        assert!(plus_di_final_state(&high, &low, &close, 200).is_none()); // :322 (via dm_tr_sums_final)
        assert!(minus_di_final_state(&high, &low, &close, 200).is_none());
        assert!(dx_final_state(&high, &low, &close, 200).is_none());
        // ADX: period == 0 (:476) and 2*period-1 >= n (:480).
        assert!(adx_final_state(&high, &low, &close, 0).is_none());
        assert!(adx_final_state(&high, &low, &close, 70).is_none());
        // ADXR: period == 0 (:550) and 3*period-2 >= n (:553).
        assert!(adxr_final_state(&high, &low, &close, 0).is_none());
        assert!(adxr_final_state(&high, &low, &close, 50).is_none());
    }

    /// The TA-Lib ~0-divisor guards in `di_val` / `dx_val` — which are reached only on the
    /// RESUME paths (the public `plus_di`/`dx` use their own inline `di`/loop). Drive the
    /// resumes with degenerate inputs so every zero arm fires.
    #[test]
    fn di_dx_zero_divisor_guards() {
        let p = 14usize;

        // Flat series (constant high == low == close): smoothed TR ~ 0, so di_val (:355)
        // and dx_val's outer guard (:427) both return 0 on resume.
        let flat = vec![50.0; 40];
        let from = 20usize;
        let st = plus_di_final_state(&flat[..from], &flat[..from], &flat[..from], p).unwrap();
        let (tail, _) = plus_di_resume(&flat, &flat, &flat, p, from, &st).unwrap();
        assert!(
            tail.iter().all(|&x| x == 0.0),
            "plus_di resume flat TR -> 0"
        ); // :355
        let st = minus_di_final_state(&flat[..from], &flat[..from], &flat[..from], p).unwrap();
        let (tail, _) = minus_di_resume(&flat, &flat, &flat, p, from, &st).unwrap();
        assert!(
            tail.iter().all(|&x| x == 0.0),
            "minus_di resume flat TR -> 0"
        );
        let st = dx_final_state(&flat[..from], &flat[..from], &flat[..from], p).unwrap();
        let (tail, _) = dx_resume(&flat, &flat, &flat, p, from, &st).unwrap();
        assert!(tail.iter().all(|&x| x == 0.0), "dx resume flat TR -> 0"); // :427

        // +DI + −DI ~ 0 while TR > 0 -> dx_val's inner sum guard (:433). Strictly shrinking
        // inside bars: each high lower than the prior high AND each low higher than the
        // prior low, so every one-period +DM and −DM is 0 (no new extreme), yet the
        // high-low range stays positive so the smoothed TR is non-zero.
        let n = 40usize;
        let high: Vec<f64> = (0..n).map(|i| 100.0 - i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..n).map(|i| 60.0 + i as f64 * 0.5).collect();
        let close: Vec<f64> = (0..n).map(|i| 80.0 + (i as f64 * 0.3).sin()).collect();
        // Sanity: the constructed bars really are inside bars with positive range.
        for i in 1..n {
            assert!(high[i] < high[i - 1] && low[i] > low[i - 1] && high[i] > low[i]);
        }
        let st = dx_final_state(&high[..from], &low[..from], &close[..from], p).unwrap();
        let (tail, _) = dx_resume(&high, &low, &close, p, from, &st).unwrap();
        assert!(tail.iter().all(|&x| x == 0.0), "dx resume zero DI-sum -> 0"); // :433
    }
}
