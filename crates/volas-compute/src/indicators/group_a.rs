//! Group A non-TA-Lib indicators (gap report 2026-06-07).
//!
//! Pure O(n) kernels, verified against the source-pinned reference oracle
//! (`test/oracle_reference.py`). The recursive members (pvt / nvi / pvi / efi / tsi /
//! mass_index) additionally carry a small `*_final_state` / `*_resume` pair so the
//! directive engine refreshes them on `append` in O(new rows) and continues them
//! bit-exactly past a head-dropping slice (verified by `test/test_group_a_mutation.py`).

use crate::indicators::av;
use crate::kernels;

/// PSY (心理线): `100 * mean(up, n)`, where `up[i] = 1` when `close[i] > close[i-1]` (the
/// first bar is not rising). NaN until `n` bars are available.
/// Source: Eastmoney 心理线.
pub fn psy(close: &[f64], n: usize) -> Vec<f64> {
    let up: Vec<f64> = (0..close.len())
        .map(|i| f64::from(i > 0 && close[i] > close[i - 1]))
        .collect();
    (kernels::sma(av(&up), n) * 100.0).to_vec()
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
pub fn dpo(close: &[f64], n: usize) -> Vec<f64> {
    let sma = kernels::sma(av(close), n);
    let shift = n / 2 + 1;
    (0..close.len())
        .map(|i| if i >= shift { close[i - shift] - sma[i] } else { f64::NAN })
        .collect()
}

/// CMF Chaikin Money Flow: `sum_n(MFV) / sum_n(volume)`, `MFV = ((C-L)-(H-C))/(H-L) * V`.
/// Source: StockCharts / Fidelity — Chaikin Money Flow.
pub fn cmf(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], n: usize) -> Vec<f64> {
    let mfv: Vec<f64> = (0..close.len())
        .map(|i| ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i]) * volume[i])
        .collect();
    let sum_mfv = super::sum(&mfv, n);
    let sum_vol = super::sum(volume, n);
    (0..close.len()).map(|i| sum_mfv[i] / sum_vol[i]).collect()
}

/// CHOP Choppiness Index: `100 * log10( sum_n(TR) / (HHV_n(high) - LLV_n(low)) ) / log10(n)`.
/// Source: TradingView — Choppiness Index.
pub fn chop(high: &[f64], low: &[f64], close: &[f64], n: usize) -> Vec<f64> {
    let tr = super::tr(high, low, close);
    // TR[0] is NaN; `sma` skips the leading-NaN prefix, so `sma * n` is the n-window TR sum.
    let sum_tr = kernels::sma(av(&tr), n) * n as f64;
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
pub fn kst(close: &[f64]) -> Vec<f64> {
    let roc = |p: usize| -> Vec<f64> {
        (0..close.len())
            .map(|i| if i >= p { (close[i] / close[i - p] - 1.0) * 100.0 } else { f64::NAN })
            .collect()
    };
    let (roc10, roc15, roc20, roc30) = (roc(10), roc(15), roc(20), roc(30));
    let r1 = kernels::sma(av(&roc10), 10);
    let r2 = kernels::sma(av(&roc15), 10);
    let r3 = kernels::sma(av(&roc20), 10);
    let r4 = kernels::sma(av(&roc30), 15);
    (0..close.len())
        .map(|i| r1[i] + 2.0 * r2[i] + 3.0 * r3[i] + 4.0 * r4[i])
        .collect()
}

/// EMV Ease of Movement: `SMA_n(distance / box)`, `distance = mid - prev mid`,
/// `box = (volume / 1e8) / (high - low)`, `mid = (high + low) / 2`. The 1e8 volume scale is
/// StockCharts' presentation convention. Source: StockCharts — Ease of Movement.
pub fn emv(high: &[f64], low: &[f64], volume: &[f64], n: usize) -> Vec<f64> {
    const SCALE: f64 = 100_000_000.0;
    let mid: Vec<f64> = (0..high.len()).map(|i| (high[i] + low[i]) / 2.0).collect();
    let emv1: Vec<f64> = (0..high.len())
        .map(|i| {
            if i == 0 {
                f64::NAN
            } else {
                (mid[i] - mid[i - 1]) / ((volume[i] / SCALE) / (high[i] - low[i]))
            }
        })
        .collect();
    kernels::sma(av(&emv1), n).to_vec()
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
/// Source: StockCharts — Mass Index.
pub fn mass_index(high: &[f64], low: &[f64], n: usize) -> Vec<f64> {
    let rng = mass_index_range(high, low);
    let single = super::ema(&rng, 9);
    let double = super::ema(&single, 9);
    let ratio: Vec<f64> = (0..high.len()).map(|i| single[i] / double[i]).collect();
    // `ratio` warms up with NaN, so use `sma * n` (which skips the leading NaN) rather than
    // the running `sum` (which would propagate it).
    (kernels::sma(av(&ratio), n) * n as f64).to_vec()
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

/// `[single_ema, double_ema, sum, ratio[len-n..len]…]` after a full [`mass_index`]: the two
/// EMA9-cascade stages, the exact rolling-sum accumulator, and the `n` ratio values still in
/// the sum window. `None` before the `n`-sum emits. The accumulator mirrors `kernels::sma`'s
/// fast-path sliding sum so a resume continues on identical bits.
pub fn mass_index_final_state(high: &[f64], low: &[f64], n: usize) -> Option<Vec<f64>> {
    let len = high.len();
    let rng = mass_index_range(high, low);
    let single = super::ema(&rng, 9);
    let double = super::ema(&single, 9);
    let ratio: Vec<f64> = (0..len).map(|i| single[i] / double[i]).collect();
    let start = ratio.iter().position(|x| !x.is_nan()).unwrap_or(len);
    if start + n > len {
        return None; // the n-window sum has not emitted a valid value yet
    }
    let mut sum = 0.0;
    for i in start..len {
        sum += ratio[i];
        if i >= start + n {
            sum -= ratio[i - n];
        }
    }
    let mut state = Vec::with_capacity(3 + n);
    state.push(single[len - 1]);
    state.push(double[len - 1]);
    state.push(sum);
    state.extend_from_slice(&ratio[len - n..len]);
    Some(state)
}

/// Resume [`mass_index`] over `[from, n)`: advance the EMA9 cascade, form each ratio, slide
/// the running sum (add the new ratio, drop the one leaving the window), and emit
/// `(sum / n) * n` — bit-identical to `kernels::sma(ratio, n) * n`.
pub fn mass_index_resume(
    high: &[f64],
    low: &[f64],
    n: usize,
    from: usize,
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let len = high.len();
    let k = 2.0 / (9.0 + 1.0); // EMA9
    let (mut single, mut double, mut sum) = (state[0], state[1], state[2]);
    let mut ratios: Vec<f64> = state[3..3 + n].to_vec(); // ratio[from-n .. from]
    let mut out = Vec::with_capacity(len.saturating_sub(from));
    for i in from..len {
        single = ((high[i] - low[i]) - single).mul_add(k, single);
        double = (single - double).mul_add(k, double);
        let ratio_i = single / double;
        sum += ratio_i;
        sum -= ratios[i - from]; // ratio[i - n]
        ratios.push(ratio_i);
        out.push((sum / n as f64) * n as f64);
    }
    let mut new_state = Vec::with_capacity(3 + n);
    new_state.push(single);
    new_state.push(double);
    new_state.push(sum);
    new_state.extend_from_slice(&ratios[ratios.len() - n..]);
    (out, new_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::test_support::*;

    /// The recursive `*_final_state` guards decline before the indicator can seed — the
    /// cumulative pair on an empty series, mass_index on a frame too short for its EMA9
    /// cascade + n-sum to emit — so the engine keeps the correct full-recompute fallback.
    #[test]
    fn final_state_declines_when_unseeded() {
        assert!(pvt_final_state(&[], &[]).is_none()); // n == 0
        assert!(nvi_final_state(&[], &[]).is_none()); // volume_index n == 0
        assert!(pvi_final_state(&[], &[]).is_none());
        let (high, low, _) = ohlc(20);
        assert!(mass_index_final_state(&high, &low, 25).is_none()); // start + n > len
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
        let mi_full = mass_index(&high, &low, 25);

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

            let st = mass_index_final_state(&high[..from], &low[..from], 25).unwrap();
            assert_bits(
                &mass_index_resume(&high, &low, 25, from, &st).0,
                &mi_full[from..],
                "mass_index",
            );
        }
    }
}
