//! Group A non-TA-Lib indicators (gap report 2026-06-07).
//!
//! Pure O(n) kernels, verified against the source-pinned reference oracle
//! (`test/oracle_reference.py`). The recursive members (pvt / nvi / pvi / efi / tsi /
//! mass_index) additionally carry a small `*_final_state` / `*_resume` pair so the
//! directive engine refreshes them on `append` in O(new rows) and continues them
//! bit-exactly past a head-dropping slice (verified by `test/test_group_a_mutation.py`).

use crate::raw::indicators::av;
use crate::missing;
use crate::guard::Slide;
use crate::guard::state_after;
use crate::raw::kernels;

/// PSY (心理线): `100 * mean(up, n)`, where `up[i] = 1` when `close[i] > close[i-1]` (the
/// first bar is not rising). NaN until `n` bars are available.
/// Source: Eastmoney 心理线. Row 0 is at absolute position `origin`.
pub fn psy(close: &[f64], n: usize, origin: usize) -> Vec<f64> {
    let up: Vec<f64> = (0..close.len()).map(|i| psy_up(close, i)).collect();
    (kernels::sma(av(&up), n, origin) * 100.0).to_vec()
}

/// PSY's rising flag at bar `i` (`0` at the first bar). It reads `close[i]` and the
/// prior close, and is missing when one of them is (the per-row term rule,
/// [`missing::term`]): a comparison with a missing close would read as "not rising".
#[inline]
fn psy_up(close: &[f64], i: usize) -> f64 {
    if i == 0 {
        return missing::term(&[close[0]], 0.0);
    }
    missing::term(&[close[i], close[i - 1]], f64::from(close[i] > close[i - 1]))
}

/// PVT Price Volume Trend (cumulative): `PVT_i = PVT_{i-1} + (C_i - C_{i-1}) / C_{i-1} * V_i`,
/// with `PVT_0 = 0`.
/// Source: StockCharts / Investopedia — Price Volume Trend.
pub fn pvt(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; close.len()];
    let mut acc = 0.0;
    for i in 1..close.len() {
        acc += (close[i] - close[i - 1]) / close[i - 1] * volume[i];
        out[i] = acc;
    }
    out
}

/// NVI Negative Volume Index (base 1000): on a down-volume bar `*= 1 + ROC`, otherwise held.
/// Source: StockCharts — Negative Volume Index.
pub fn nvi(close: &[f64], volume: &[f64]) -> Vec<f64> {
    volume_index(close, volume, true)
}

/// PVI Positive Volume Index (base 1000): on an up-volume bar `*= 1 + ROC`, otherwise held.
/// Source: StockCharts — Positive Volume Index.
pub fn pvi(close: &[f64], volume: &[f64]) -> Vec<f64> {
    volume_index(close, volume, false)
}

/// Shared NVI / PVI engine. `on_down` selects the negative (volume-falling) variant.
fn volume_index(close: &[f64], volume: &[f64], on_down: bool) -> Vec<f64> {
    const BASE: f64 = 1000.0;
    let mut out = vec![BASE; close.len()];
    for i in 1..close.len() {
        let triggered = if on_down {
            volume[i] < volume[i - 1]
        } else {
            volume[i] > volume[i - 1]
        };
        out[i] = if triggered {
            out[i - 1] * (1.0 + (close[i] - close[i - 1]) / close[i - 1])
        } else {
            out[i - 1]
        };
    }
    out
}

/// DPO Detrended Price Oscillator: `Price[(n/2 + 1) ago] - SMA_n`.
/// Source: StockCharts — Detrended Price Oscillator (displaced form).
pub fn dpo(close: &[f64], n: usize, origin: usize) -> Vec<f64> {
    let sma = kernels::sma(av(close), n, origin);
    let shift = n / 2 + 1;
    (0..close.len())
        .map(|i| if i >= shift { close[i - shift] - sma[i] } else { f64::NAN })
        .collect()
}

/// CMF Chaikin Money Flow: `sum_n(MFV) / sum_n(volume)`, `MFV = ((C-L)-(H-C))/(H-L) * V`.
/// Source: StockCharts / Fidelity — Chaikin Money Flow.
pub fn cmf(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], n: usize, origin: usize) -> Vec<f64> {
    let mfv: Vec<f64> = (0..close.len()).map(|i| mfv(high, low, close, volume, i)).collect();
    let sum_mfv = super::sum(&mfv, n, origin);
    let sum_vol = super::sum(volume, n, origin);
    (0..close.len()).map(|i| sum_mfv[i] / sum_vol[i]).collect()
}

/// CHOP Choppiness Index: `100 * log10( sum_n(TR) / (HHV_n(high) - LLV_n(low)) ) / log10(n)`.
/// Source: TradingView — Choppiness Index.
pub fn chop(high: &[f64], low: &[f64], close: &[f64], n: usize, origin: usize) -> Vec<f64> {
    let tr = super::tr(high, low, close);
    // TR[0] is NaN; `sma` skips the leading-NaN prefix, so `sma * n` is the n-window TR sum.
    let sum_tr = kernels::sma(av(&tr), n, origin) * n as f64;
    let hh = super::hhv(high, n);
    let ll = super::llv(low, n);
    let denom = (n as f64).log10();
    (0..close.len())
        .map(|i| 100.0 * (sum_tr[i] / (hh[i] - ll[i])).log10() / denom)
        .collect()
}

/// KST Know Sure Thing (Pring): `1*RCMA1 + 2*RCMA2 + 3*RCMA3 + 4*RCMA4`, where
/// `RCMA1=SMA10(ROC10)`, `RCMA2=SMA10(ROC15)`, `RCMA3=SMA10(ROC20)`, `RCMA4=SMA15(ROC30)`.
/// Source: StockCharts — Know Sure Thing.
pub fn kst(close: &[f64], origin: usize) -> Vec<f64> {
    let leg = |(p, n): (usize, usize)| {
        let roc: Vec<f64> = (0..close.len()).map(|i| kst_roc(close, p, i)).collect();
        kernels::sma(av(&roc), n, origin)
    };
    let [r1, r2, r3, r4] = KST_LEGS.map(leg);
    (0..close.len())
        .map(|i| r1[i] + 2.0 * r2[i] + 3.0 * r3[i] + 4.0 * r4[i])
        .collect()
}

/// EMV Ease of Movement: `SMA_n(distance / box)`, `distance = mid - prev mid`,
/// `box = (volume / 1e8) / (high - low)`, `mid = (high + low) / 2`. The 1e8 volume scale is
/// StockCharts' presentation convention. Source: StockCharts — Ease of Movement.
pub fn emv(high: &[f64], low: &[f64], volume: &[f64], n: usize, origin: usize) -> Vec<f64> {
    let emv1: Vec<f64> = (0..high.len()).map(|i| emv_term(high, low, volume, i)).collect();
    kernels::sma(av(&emv1), n, origin).to_vec()
}

/// EFI Elder Force Index: `EMA_n((C - prev C) * volume)`.
/// Source: StockCharts / Investopedia — Force Index.
pub fn efi(close: &[f64], volume: &[f64], n: usize) -> Vec<f64> {
    super::ema(&efi_raw(close, volume), n)
}

/// TSI True Strength Index: `100 * EMA_short(EMA_long(m)) / EMA_short(EMA_long(|m|))`,
/// `m = C - prev C`. Source: StockCharts — True Strength Index.
pub fn tsi(close: &[f64], long: usize, short: usize) -> Vec<f64> {
    let m = tsi_momentum(close);
    let am: Vec<f64> = m.iter().map(|x| x.abs()).collect();
    let num = super::ema(&super::ema(&m, long), short);
    let den = super::ema(&super::ema(&am, long), short);
    (0..close.len()).map(|i| 100.0 * num[i] / den[i]).collect()
}

/// Mass Index: `sum_n( EMA9(H-L) / EMA9(EMA9(H-L)) )`.
/// Source: StockCharts — Mass Index. Row 0 is at absolute position `origin`.
pub fn mass_index(high: &[f64], low: &[f64], n: usize, origin: usize) -> Vec<f64> {
    let rng = mass_index_range(high, low);
    let single = super::ema(&rng, 9);
    let double = super::ema(&single, 9);
    let ratio: Vec<f64> = (0..high.len()).map(|i| single[i] / double[i]).collect();
    // `ratio` warms up with NaN, so use `sma * n` (which skips the leading NaN) rather than
    // the running `sum` (which would propagate it).
    (kernels::sma(av(&ratio), n, origin) * n as f64).to_vec()
}

/// CRSI Connors RSI: `mean( RSI(close, rsi_len), RSI(streak, streak_len),
/// PercentRank(1-period ROC%, rank_len) )`, where `streak` is the signed run length of
/// consecutive up / down closes and PercentRank is the share of the prior `rank_len` ROC
/// values strictly below the current one. Source: Connors Research / TradingView.
pub fn crsi(close: &[f64], rsi_len: usize, streak_len: usize, rank_len: usize) -> Vec<f64> {
    let len = close.len();
    let mut streak = vec![0.0; len];
    for i in 1..len {
        streak[i] = if close[i] > close[i - 1] {
            if streak[i - 1] > 0.0 { streak[i - 1] + 1.0 } else { 1.0 }
        } else if close[i] < close[i - 1] {
            if streak[i - 1] < 0.0 { streak[i - 1] - 1.0 } else { -1.0 }
        } else {
            0.0
        };
    }
    let rsi_close = super::rsi(close, rsi_len);
    let rsi_streak = super::rsi(&streak, streak_len);
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

// --- state-carry (append O(new rows) + bit-exact slice continuation) ----------
//
// Each recursive member compresses its whole history into a small fixed state: the
// cumulative line value (pvt / nvi / pvi), the single carried EMA (efi), the four
// nested-EMA stage values (tsi), or the two EMA-cascade stages plus the rolling-sum
// window (mass_index). `*_final_state` captures it after a full compute (returning
// `None` before the indicator seeds, so the caller keeps the correct full-recompute
// fallback); `*_resume` continues over only the new rows `[from, n)`, reading nothing
// before `from`, with arithmetic bit-identical to each kernel's steady-state loop.

/// EFI's raw force series `(C - prev C) * volume` (`NaN` at the first bar); the input the
/// `efi` EMA smooths. Shared by [`efi`] and [`efi_final_state`] so both seed identically.
fn efi_raw(close: &[f64], volume: &[f64]) -> Vec<f64> {
    (0..close.len())
        .map(|i| if i > 0 { (close[i] - close[i - 1]) * volume[i] } else { f64::NAN })
        .collect()
}

/// TSI's momentum series `m = C - prev C` (`NaN` at the first bar).
fn tsi_momentum(close: &[f64]) -> Vec<f64> {
    (0..close.len())
        .map(|i| if i > 0 { close[i] - close[i - 1] } else { f64::NAN })
        .collect()
}

/// Mass Index's high-low range series.
fn mass_index_range(high: &[f64], low: &[f64]) -> Vec<f64> {
    (0..high.len()).map(|i| high[i] - low[i]).collect()
}

/// `[pvt, prev_close]` after a full [`pvt`] — the seed a [`pvt_resume`] needs. `None` for
/// an empty series.
pub fn pvt_final_state(close: &[f64], volume: &[f64]) -> Option<Vec<f64>> {
    let n = close.len();
    if n == 0 {
        return None;
    }
    let mut acc = 0.0;
    for i in 1..n {
        acc += (close[i] - close[i - 1]) / close[i - 1] * volume[i];
    }
    Some(vec![acc, close[n - 1]])
}

/// Resume [`pvt`] from `[pvt_{from-1}, close_{from-1}]` over `[from, n)`, bit-identical to a
/// full recompute. Reads only `close[from..]` / `volume[from..]` plus the carried prev close.
pub fn pvt_resume(
    close: &[f64],
    volume: &[f64],
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut acc = state[0];
    let mut prev = state[1];
    let mut out = Vec::with_capacity(close.len().saturating_sub(from));
    for i in from..close.len() {
        acc += (close[i] - prev) / prev * volume[i];
        out.push(acc);
        prev = close[i];
    }
    (out, vec![acc, prev])
}

/// `[index, prev_close, prev_volume]` after a full [`volume_index`]. `None` for an empty
/// series. `on_down` selects the NVI (volume-falling) variant, matching the kernel.
fn volume_index_final_state(close: &[f64], volume: &[f64], on_down: bool) -> Option<Vec<f64>> {
    let n = close.len();
    if n == 0 {
        return None;
    }
    let mut val = 1000.0; // BASE
    for i in 1..n {
        let triggered = if on_down {
            volume[i] < volume[i - 1]
        } else {
            volume[i] > volume[i - 1]
        };
        if triggered {
            val *= 1.0 + (close[i] - close[i - 1]) / close[i - 1];
        }
    }
    Some(vec![val, close[n - 1], volume[n - 1]])
}

/// Resume [`volume_index`] from `[index_{from-1}, close_{from-1}, volume_{from-1}]` over
/// `[from, n)`, bit-identical to a full recompute.
fn volume_index_resume(
    close: &[f64],
    volume: &[f64],
    from: usize,
    state: &[f64],
    on_down: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut val = state[0];
    let mut prev_c = state[1];
    let mut prev_v = state[2];
    let mut out = Vec::with_capacity(close.len().saturating_sub(from));
    for i in from..close.len() {
        let triggered = if on_down {
            volume[i] < prev_v
        } else {
            volume[i] > prev_v
        };
        if triggered {
            val *= 1.0 + (close[i] - prev_c) / prev_c;
        }
        out.push(val);
        prev_c = close[i];
        prev_v = volume[i];
    }
    (out, vec![val, prev_c, prev_v])
}

/// `[nvi, prev_close, prev_volume]` after a full [`nvi`].
pub fn nvi_final_state(close: &[f64], volume: &[f64]) -> Option<Vec<f64>> {
    volume_index_final_state(close, volume, true)
}

/// Resume [`nvi`]; see [`volume_index_resume`].
pub fn nvi_resume(
    close: &[f64],
    volume: &[f64],
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    volume_index_resume(close, volume, from, state, true)
}

/// `[pvi, prev_close, prev_volume]` after a full [`pvi`].
pub fn pvi_final_state(close: &[f64], volume: &[f64]) -> Option<Vec<f64>> {
    volume_index_final_state(close, volume, false)
}

/// Resume [`pvi`]; see [`volume_index_resume`].
pub fn pvi_resume(
    close: &[f64],
    volume: &[f64],
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    volume_index_resume(close, volume, from, state, false)
}

/// `[efi_ema, prev_close]` after a full [`efi`], or `None` before the EMA seeds.
pub fn efi_final_state(close: &[f64], volume: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut state = super::ema_final_state(&efi_raw(close, volume), n)?;
    state.push(close[close.len() - 1]);
    Some(state)
}

/// Resume [`efi`] from `[efi_ema_{from-1}, close_{from-1}]` over `[from, n)`. Rebuilds the
/// raw force term from the carried prev close, then advances the EMA with the same fused
/// `(x-prev)·k+prev` step as the kernel.
pub fn efi_resume(
    close: &[f64],
    volume: &[f64],
    n: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let k = 2.0 / (n as f64 + 1.0);
    let mut e = state[0];
    let mut prev = state[1];
    let mut out = Vec::with_capacity(close.len().saturating_sub(from));
    for i in from..close.len() {
        let x = (close[i] - prev) * volume[i];
        e = (x - e).mul_add(k, e);
        out.push(e);
        prev = close[i];
    }
    (out, vec![e, prev])
}

/// `[inner_m, outer_m, inner_am, outer_am, prev_close]` after a full [`tsi`] — the two
/// nested EMA stages of the numerator (`m`) and denominator (`|m|`) chains as of the last
/// row. `None` before the outer EMA seeds.
pub fn tsi_final_state(close: &[f64], long: usize, short: usize) -> Option<Vec<f64>> {
    let n = close.len();
    let m = tsi_momentum(close);
    let am: Vec<f64> = m.iter().map(|x| x.abs()).collect();
    let inner_m = super::ema(&m, long);
    let inner_am = super::ema(&am, long);
    // The outer EMA seeding (`?`) gates the whole state: if it seeded, the inner EMA seeded
    // earlier, so `inner_*[n-1]` are finite steady-state values.
    let om = super::ema_final_state(&inner_m, short)?;
    let oa = super::ema_final_state(&inner_am, short)?;
    Some(vec![inner_m[n - 1], om[0], inner_am[n - 1], oa[0], close[n - 1]])
}

/// Resume [`tsi`] over `[from, n)`, advancing both nested EMA chains with the kernel's exact
/// fused steps.
pub fn tsi_resume(
    close: &[f64],
    long: usize,
    short: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let kl = 2.0 / (long as f64 + 1.0);
    let ks = 2.0 / (short as f64 + 1.0);
    let (mut im, mut om, mut ia, mut oa, mut prev) =
        (state[0], state[1], state[2], state[3], state[4]);
    let mut out = Vec::with_capacity(close.len().saturating_sub(from));
    #[allow(clippy::needless_range_loop)] // numeric kernel: index-loop kept for hot-path codegen stability
    for i in from..close.len() {
        let mi = close[i] - prev;
        let ai = mi.abs();
        im = (mi - im).mul_add(kl, im);
        om = (im - om).mul_add(ks, om);
        ia = (ai - ia).mul_add(kl, ia);
        oa = (ia - oa).mul_add(ks, oa);
        out.push(100.0 * om / oa);
        prev = close[i];
    }
    (out, vec![im, om, ia, oa, prev])
}

/// `[single_ema, double_ema, <the n-sum's slide>, ratio[len-n..len]…]` after a full
/// [`mass_index`] (row 0 at absolute position `origin`): the two EMA9-cascade stages, the
/// state of the guarded `n`-row sum of the ratios ([`kernels::sma_step`]'s [`Slide`]: the
/// exact sum, its count of non-finite rows and the next row's absolute position), and the
/// `n` ratios still in the window. `None` before the `n`-sum emits. The slide is
/// `kernels::sma`'s own, so a resume continues on identical bits — an infinite ratio
/// included.
pub fn mass_index_final_state(high: &[f64], low: &[f64], n: usize, origin: usize) -> Option<Vec<f64>> {
    let len = high.len();
    let rng = mass_index_range(high, low);
    let single = super::ema(&rng, 9);
    let double = super::ema(&single, 9);
    let ratio: Vec<f64> = (0..len).map(|i| single[i] / double[i]).collect();
    let start = ratio.iter().position(|x| !x.is_nan()).unwrap_or(len);
    if start + n > len {
        return None; // the n-window sum has not emitted a valid value yet
    }
    let slide = state_after(Slide::<1>::initial(origin), len, |row, prev, next| {
        kernels::sma_step(n, row, |j| ratio[j], prev, next)
    })?;
    let mut state = vec![single[len - 1], double[len - 1]];
    state.extend(slide);
    state.extend_from_slice(&ratio[len - n..len]);
    Some(state)
}

/// Where the carried ratios start in a [`mass_index`] state.
const MASS_RATIOS: usize = 2 + kernels::SMA_STATE;

/// One row of [`mass_index`] from the carried `state` (see [`mass_index_final_state`]):
/// advance the EMA9 cascade by the row's range, form its ratio, and slide the guarded
/// `n`-sum over the window of carried ratios plus the new one; the state as of the row is
/// written into `next`. Returns `sum` (`mean·n`). The one step of the Vec and scalar resumes.
fn mass_index_step(range: f64, n: usize, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    let k = 2.0 / (9.0 + 1.0); // EMA9
    let single = (range - state[0]).mul_add(k, state[0]);
    let double = (single - state[1]).mul_add(k, state[1]);
    let ratio = single / double;
    let ring = &state[MASS_RATIOS..MASS_RATIOS + n]; // the ratios of rows `row-n .. row-1`
    let term = |j: usize| if j == row { ratio } else { ring[j + n - row] };
    let mean = kernels::sma_step(n, row, term, &state[2..MASS_RATIOS], &mut next[2..MASS_RATIOS])?;
    next[0] = single;
    next[1] = double;
    next[MASS_RATIOS..MASS_RATIOS + n - 1].copy_from_slice(&ring[1..]);
    next[MASS_RATIOS + n - 1] = ratio;
    Some(mean * n as f64)
}

/// Resume [`mass_index`] over `[from, n)`: one [`mass_index_step`] per row — bit-identical
/// to `kernels::sma(ratio, n) * n`.
pub fn mass_index_resume(
    high: &[f64],
    low: &[f64],
    n: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let len = high.len();
    let mut prev = state.to_vec();
    let mut next = prev.clone();
    let mut out = Vec::with_capacity(len.saturating_sub(from));
    for i in from..len {
        let v = mass_index_step(high[i] - low[i], n, i, &prev, &mut next);
        out.push(v.expect("a mass_index state from mass_index_final_state steps"));
        std::mem::swap(&mut prev, &mut next);
    }
    (out, prev)
}


/// Scalar single-row twin of [`pvt_resume`]: PVT at `row` from
/// `[acc_{row-1}, close_{row-1}]`, the state as of `row` written into `next` (same
/// length), zero-alloc, bit-identical to one loop iteration.
pub fn pvt_resume_one(
    close: &[f64],
    volume: &[f64],
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row == 0 || state.len() < 2 || next.len() != state.len() || row >= close.len() {
        return None;
    }
    let acc = state[0];
    let prev = state[1];
    let acc = acc + (close[row] - prev) / prev * volume[row];
    next.copy_from_slice(state);
    next[0] = acc;
    next[1] = close[row];
    Some(acc)
}

/// Scalar single-row twin of [`volume_index_resume`]; `on_down` selects NVI (true) vs
/// PVI (false), matching the kernel. Reads close/volume at `row`, carries prev close/vol;
/// the state as of `row` is written into `next` (same length).
fn volume_index_resume_one(
    close: &[f64],
    volume: &[f64],
    row: usize,
    state: &[f64],
    next: &mut [f64],
    on_down: bool,
) -> Option<f64> {
    if row == 0 || state.len() < 3 || next.len() != state.len() || row >= close.len() {
        return None;
    }
    let val = state[0];
    let prev_c = state[1];
    let prev_v = state[2];
    let triggered = if on_down { volume[row] < prev_v } else { volume[row] > prev_v };
    let val = if triggered { val * (1.0 + (close[row] - prev_c) / prev_c) } else { val };
    next.copy_from_slice(state);
    next[..3].copy_from_slice(&[val, close[row], volume[row]]);
    Some(val)
}

/// Scalar single-row twin of [`nvi_resume`].
pub fn nvi_resume_one(
    close: &[f64],
    volume: &[f64],
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    volume_index_resume_one(close, volume, row, state, next, true)
}

/// Scalar single-row twin of [`pvi_resume`].
pub fn pvi_resume_one(
    close: &[f64],
    volume: &[f64],
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    volume_index_resume_one(close, volume, row, state, next, false)
}

/// Scalar single-row twin of [`efi_resume`]: rebuild the raw force term from the carried
/// prev close, then one fused EMA step; the state as of `row` is written into `next`
/// (same length). Bit-identical to one loop iteration.
pub fn efi_resume_one(
    close: &[f64],
    volume: &[f64],
    n: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row == 0 || state.len() < 2 || next.len() != state.len() || row >= close.len() {
        return None;
    }
    let k = 2.0 / (n as f64 + 1.0);
    let e = state[0];
    let prev = state[1];
    let x = (close[row] - prev) * volume[row];
    let e = (x - e).mul_add(k, e);
    next.copy_from_slice(state);
    next[0] = e;
    next[1] = close[row];
    Some(e)
}

/// Scalar single-row twin of [`tsi_resume`]: advance both nested EMA chains one step and
/// emit `100*outer_m/outer_am`; the state as of `row` is written into `next` (same
/// length). Bit-identical to one loop iteration.
pub fn tsi_resume_one(
    close: &[f64],
    long: usize,
    short: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row == 0 || state.len() < 5 || next.len() != state.len() || row >= close.len() {
        return None;
    }
    let kl = 2.0 / (long as f64 + 1.0);
    let ks = 2.0 / (short as f64 + 1.0);
    let (mut im, mut om, mut ia, mut oa, prev) =
        (state[0], state[1], state[2], state[3], state[4]);
    let mi = close[row] - prev;
    let ai = mi.abs();
    im = (mi - im).mul_add(kl, im);
    om = (im - om).mul_add(ks, om);
    ia = (ai - ia).mul_add(kl, ia);
    oa = (ia - oa).mul_add(ks, oa);
    next.copy_from_slice(state);
    next[..5].copy_from_slice(&[im, om, ia, oa, close[row]]);
    Some(100.0 * om / oa)
}

/// Scalar single-row twin of [`mass_index_resume`]: one [`mass_index_step`] from `state`,
/// the state as of `row` written into `next`. Bit-identical to one loop iteration of the
/// Vec resume.
pub fn mass_index_resume_one(
    high: &[f64],
    low: &[f64],
    n: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row >= high.len() || n == 0 || row < n || state.len() != MASS_RATIOS + n || next.len() != state.len() {
        return None;
    }
    mass_index_step(high[row] - low[row], n, row, state, next)
}


// ---------------------------------------------------------------------------
// Resumable forms: the running sums of each full computation, in its exact operation
// order, so a resume of one row at a time is bit-identical to recomputing the column.
// ---------------------------------------------------------------------------

/// One bar's money-flow volume, as [`cmf`] sums it.
#[inline]
fn mfv(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], i: usize) -> f64 {
    ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i]) * volume[i]
}

/// [`cmf`]'s state after the rows: its two sums' states.
pub fn cmf_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    n: usize,
    origin: usize,
) -> Option<Vec<f64>> {
    state_after(Slide::<1>::initial(origin).repeat(2), close.len(), |row, prev, next| {
        cmf_step(high, low, close, volume, n, row, prev, next)
    })
}

/// One row of [`cmf`] at `row`, from `state` into `next`.
#[allow(clippy::too_many_arguments)]
pub fn cmf_step(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    n: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    const S: usize = super::SUM_STATE;
    if row >= close.len() || state.len() != 2 * S || next.len() != 2 * S {
        return None;
    }
    let (n_mfv, n_vol) = next.split_at_mut(S);
    let flow = super::sum_row(n, row, |j| mfv(high, low, close, volume, j), &state[..S], n_mfv)?;
    let vol = super::sum_row(n, row, |j| volume[j], &state[S..], n_vol)?;
    Some(flow / vol)
}

/// [`dpo`]'s state after the rows of `close`: its SMA's.
pub fn dpo_state(close: &[f64], n: usize, origin: usize) -> Option<Vec<f64>> {
    super::ma_state(close, n, 0, origin)
}

/// One row of [`dpo`] at `row`, from `state` into `next`.
pub fn dpo_step(close: &[f64], n: usize, row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    let sma = super::ma_step(close, n, 0, row, state, next)?;
    let shift = n / 2 + 1;
    Some(if row >= shift { close[row - shift] - sma } else { f64::NAN })
}

/// One bar's ease of movement, as [`emv`] averages it.
#[inline]
fn emv_term(high: &[f64], low: &[f64], volume: &[f64], i: usize) -> f64 {
    const SCALE: f64 = 100_000_000.0;
    if i == 0 {
        return f64::NAN;
    }
    let mid = |k: usize| (high[k] + low[k]) / 2.0;
    (mid(i) - mid(i - 1)) / ((volume[i] / SCALE) / (high[i] - low[i]))
}

/// [`emv`]'s state after the rows: its SMA's.
pub fn emv_state(high: &[f64], low: &[f64], volume: &[f64], n: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<1>::initial(origin), high.len(), |row, prev, next| {
        emv_step(high, low, volume, n, row, prev, next)
    })
}

/// One row of [`emv`] at `row`, from `state` into `next`.
pub fn emv_step(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    n: usize,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row >= high.len() {
        return None;
    }
    kernels::sma_step(n, row, |j| emv_term(high, low, volume, j), state, next)
}

/// KST's four `(rate-of-change period, SMA period)` pairs.
const KST_LEGS: [(usize, usize); 4] = [(10, 10), (15, 10), (20, 10), (30, 15)];

/// The `p`-bar rate of change of `close` at `i`, as [`kst`] averages it.
#[inline]
fn kst_roc(close: &[f64], p: usize, i: usize) -> f64 {
    if i >= p { (close[i] / close[i - p] - 1.0) * 100.0 } else { f64::NAN }
}

/// [`kst`]'s state after the rows of `close`: its four SMAs'.
pub fn kst_state(close: &[f64], origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<1>::initial(origin).repeat(4), close.len(), |row, prev, next| {
        kst_step(close, row, prev, next)
    })
}

/// One row of [`kst`] at `row`, from `state` into `next`.
pub fn kst_step(close: &[f64], row: usize, state: &[f64], next: &mut [f64]) -> Option<f64> {
    const S: usize = kernels::SMA_STATE;
    if row >= close.len() || state.len() != 4 * S || next.len() != 4 * S {
        return None;
    }
    let mut r = [0.0; 4];
    for (k, (roc, sma)) in KST_LEGS.into_iter().enumerate() {
        let (s, n) = (&state[k * S..(k + 1) * S], &mut next[k * S..(k + 1) * S]);
        r[k] = kernels::sma_step(sma, row, |j| kst_roc(close, roc, j), s, n)?;
    }
    Some(r[0] + 2.0 * r[1] + 3.0 * r[2] + 4.0 * r[3])
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::indicators::test_support::*;

    /// The recursive `*_final_state` guards decline before the indicator can seed — the
    /// cumulative pair on an empty series, mass_index on a frame too short for its EMA9
    /// cascade + n-sum to emit — so the engine keeps the correct full-recompute fallback.
    #[test]
    fn final_state_declines_when_unseeded() {
        assert!(pvt_final_state(&[], &[]).is_none()); // n == 0
        assert!(nvi_final_state(&[], &[]).is_none()); // volume_index n == 0
        assert!(pvi_final_state(&[], &[]).is_none());
        let (high, low, _) = ohlc(20);
        assert!(mass_index_final_state(&high, &low, 25, 0).is_none()); // start + n > len
    }

    /// Each resume, fed the carried state of a full compute over the head `[0, from)`,
    /// reproduces the tail of a full compute over the whole input — bit-for-bit. This is
    /// the property that makes the append + slice-continuation fast path safe.
    #[test]
    fn resume_is_bit_identical_to_full() {
        let (high, low, close) = ohlc(160);
        // A volume track that rises and falls so NVI's / PVI's up- and down-volume arms
        // both fire across the run.
        let vol: Vec<f64> = (0..160)
            .map(|i| 1000.0 + 300.0 * ((i as f64) * 0.27).sin())
            .collect();

        let pvt_full = pvt(&close, &vol);
        let nvi_full = nvi(&close, &vol);
        let pvi_full = pvi(&close, &vol);
        let efi_full = efi(&close, &vol, 13);
        let tsi_full = tsi(&close, 25, 13);
        let mi_full = mass_index(&high, &low, 25, 0);

        // Every `from` is past the slowest warm-up (tsi 37, mass_index 40).
        for &from in &[41usize, 60, 100, 159] {
            let (c, v) = (&close[..from], &vol[..from]);

            let st = pvt_final_state(c, v).unwrap();
            assert_bits(&pvt_resume(&close, &vol, from, &st).0, &pvt_full[from..], "pvt");

            let st = nvi_final_state(c, v).unwrap();
            assert_bits(&nvi_resume(&close, &vol, from, &st).0, &nvi_full[from..], "nvi");

            let st = pvi_final_state(c, v).unwrap();
            assert_bits(&pvi_resume(&close, &vol, from, &st).0, &pvi_full[from..], "pvi");

            let st = efi_final_state(c, v, 13).unwrap();
            assert_bits(&efi_resume(&close, &vol, 13, from, &st).0, &efi_full[from..], "efi");

            let st = tsi_final_state(c, 25, 13).unwrap();
            assert_bits(&tsi_resume(&close, 25, 13, from, &st).0, &tsi_full[from..], "tsi");

            let st = mass_index_final_state(&high[..from], &low[..from], 25, 0).unwrap();
            assert_bits(
                &mass_index_resume(&high, &low, 25, from, &st).0,
                &mi_full[from..],
                "mass_index",
            );
        }
    }
}
