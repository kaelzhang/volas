//! Group B indicators (gap report 2026-06-07, §9): convention-sensitive market
//! indicators. Each kernel pins ONE authoritative convention, cited inline and matched
//! bit-for-shape by the source-pinned oracle (`test/oracle_reference.py`). Finite-memory
//! members refresh on `append` through the engine's windowed fast-path (no state-carry);
//! recursive members carry a `*_final_state` / `*_resume` pair like Group A.

use crate::indicators::av;
use crate::kernels;

/// Rolling `n`-window sum that skips a leading-NaN prefix — so a series with an undefined
/// first bar (`tr`, prev-close diffs) sums from its first finite value — via `sma * n`.
fn rolling_sum(x: &[f64], n: usize) -> Vec<f64> {
    (kernels::sma(av(x), n) * n as f64).to_vec()
}

/// Vortex Indicator (+VI / −VI): `+VM = |high − prev low|`, `−VM = |low − prev high|`;
/// `+VI = Σₙ(+VM) / Σₙ(TR)`, `−VI = Σₙ(−VM) / Σₙ(TR)`. `plus` selects the +VI line.
/// Source: StockCharts ChartSchool / Wikipedia — Vortex Indicator.
pub fn vortex(high: &[f64], low: &[f64], close: &[f64], n: usize, plus: bool) -> Vec<f64> {
    let len = high.len();
    let tr = super::tr(high, low, close); // NaN at bar 0
    let vm: Vec<f64> = (0..len)
        .map(|i| {
            if i == 0 {
                f64::NAN
            } else if plus {
                (high[i] - low[i - 1]).abs()
            } else {
                (low[i] - high[i - 1]).abs()
            }
        })
        .collect();
    let sum_vm = rolling_sum(&vm, n);
    let sum_tr = rolling_sum(&tr, n);
    (0..len).map(|i| sum_vm[i] / sum_tr[i]).collect()
}

/// BRAR — AR (人气指标) = `Σₙ(H − O) / Σₙ(O − L) × 100`. `H − O` and `O − L` are always ≥ 0.
/// Source: 通达信 / MBA智库 — 人气意愿指标 (BRAR).
pub fn brar_ar(open: &[f64], high: &[f64], low: &[f64], n: usize) -> Vec<f64> {
    let len = high.len();
    let ho: Vec<f64> = (0..len).map(|i| high[i] - open[i]).collect();
    let ol: Vec<f64> = (0..len).map(|i| open[i] - low[i]).collect();
    let (s_ho, s_ol) = (rolling_sum(&ho, n), rolling_sum(&ol, n));
    (0..len).map(|i| s_ho[i] / s_ol[i] * 100.0).collect()
}

/// BRAR — BR (意愿指标) = `Σₙ max(0, H − Cᵧ) / Σₙ max(0, Cᵧ − L) × 100`, `Cᵧ` = prior close.
/// The `max(0, …)` clamp is the 通达信 convention (a high below — or low above — the prior
/// close contributes nothing). Source: 通达信 / MBA智库 — 人气意愿指标 (BRAR).
pub fn brar_br(high: &[f64], low: &[f64], close: &[f64], n: usize) -> Vec<f64> {
    let len = high.len();
    let up: Vec<f64> = (0..len)
        .map(|i| if i == 0 { f64::NAN } else { (high[i] - close[i - 1]).max(0.0) })
        .collect();
    let dn: Vec<f64> = (0..len)
        .map(|i| if i == 0 { f64::NAN } else { (close[i - 1] - low[i]).max(0.0) })
        .collect();
    let (s_up, s_dn) = (rolling_sum(&up, n), rolling_sum(&dn, n));
    (0..len).map(|i| s_up[i] / s_dn[i] * 100.0).collect()
}

/// VR 成交量比率 = `(UVS + ½·PVS) / (DVS + ½·PVS) × 100` over `n` bars, where UVS / DVS / PVS
/// are the summed volumes of up- / down- / flat-close days (vs the prior close).
/// Source: MBA智库 — 成交量比率 (VR).
pub fn vr(close: &[f64], volume: &[f64], n: usize) -> Vec<f64> {
    let len = close.len();
    let (mut uv, mut dv, mut pv) = (vec![f64::NAN; len], vec![f64::NAN; len], vec![f64::NAN; len]);
    for i in 1..len {
        let (mut u, mut d, mut p) = (0.0, 0.0, 0.0);
        if close[i] > close[i - 1] {
            u = volume[i];
        } else if close[i] < close[i - 1] {
            d = volume[i];
        } else {
            p = volume[i];
        }
        uv[i] = u;
        dv[i] = d;
        pv[i] = p;
    }
    let (suv, sdv, spv) = (rolling_sum(&uv, n), rolling_sum(&dv, n), rolling_sum(&pv, n));
    (0..len)
        .map(|i| (suv[i] + 0.5 * spv[i]) / (sdv[i] + 0.5 * spv[i]) * 100.0)
        .collect()
}
