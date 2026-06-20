//! The KDJ stochastic oscillator (`k` / `d` / `j` lines) and its append-resume
//! state — split out of `oscillators` to keep each indicator file focused.

use ndarray::Array1;

use super::av;
use crate::kernels;

fn kdj_rsv(high: &[f64], low: &[f64], close: &[f64], period_rsv: usize) -> Array1<f64> {
    let llv = kernels::rolling_min(av(low), period_rsv);
    let hhv = kernels::rolling_max(av(high), period_rsv);
    let n = close.len();
    let mut rsv = Array1::from_elem(n, 0.0);
    for i in 0..n {
        let denom = hhv[i] - llv[i];
        if denom.abs() > 1e-10 {
            rsv[i] = (close[i] - llv[i]) / denom * 100.0;
        }
    }
    rsv
}

/// KDJ %K line.
pub fn kdj_k(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    init: f64,
) -> Vec<f64> {
    let rsv = kdj_rsv(high, low, close, period_rsv);
    kernels::ewma_with_init(rsv.view(), period_k, init).to_vec()
}

/// KDJ %D line.
pub fn kdj_d(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    init: f64,
) -> Vec<f64> {
    let rsv = kdj_rsv(high, low, close, period_rsv);
    let k = kernels::ewma_with_init(rsv.view(), period_k, init);
    kernels::ewma_with_init(k.view(), period_d, init).to_vec()
}

/// KDJ %J line (`3K - 2D`).
pub fn kdj_j(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    init: f64,
) -> Vec<f64> {
    let rsv = kdj_rsv(high, low, close, period_rsv);
    let k = kernels::ewma_with_init(rsv.view(), period_k, init);
    let d = kernels::ewma_with_init(k.view(), period_d, init);
    (3.0 * &k - 2.0 * &d).to_vec()
}

/// The KDJ line a resume emits: %K, %D, or %J (`3K - 2D`).
#[derive(Clone, Copy)]
pub enum KdjLine {
    K,
    D,
    J,
}

/// Final KDJ recursive state after a full compute: `[k_last]` for `.k`, `[k_last, d_last]`
/// for `.d` / `.j` (the ⅓-weight SMA-smoothed %K and %D as of the last row). RSV is
/// finite-memory (a `period_rsv` window), so it is NOT carried — a [`kdj_resume`] recomputes
/// it from the windowed high/low/close tail. `want_d` also carries `d_last` (needed by
/// `.d` / `.j`). `None` for an empty series.
#[allow(clippy::too_many_arguments)]
pub fn kdj_final_state(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    init: f64,
    want_d: bool,
) -> Option<Vec<f64>> {
    let n = close.len();
    if n == 0 {
        return None;
    }
    let rsv = kdj_rsv(high, low, close, period_rsv);
    let k = kernels::ewma_with_init(rsv.view(), period_k, init);
    if !want_d {
        return Some(vec![k[n - 1]]);
    }
    let d = kernels::ewma_with_init(k.view(), period_d, init);
    Some(vec![k[n - 1], d[n - 1]])
}

/// Resume a KDJ `line` from `state` (`[k_last]` for `.k`, `[k_last, d_last]` for `.d`/`.j`,
/// as of row `from - 1`) over rows `[from, n)` — bit-identical to a full recompute. The %K /
/// %D SMA recurrences continue from the carried values (past the `init` seed) using the same
/// `base·prev + alpha·x` step as [`kernels::ewma_with_init`], while RSV is recomputed over the
/// windowed `high/low/close` tail `[from - period_rsv + 1, n)` (a windowed min/max gives the
/// identical value to the full series). `None` when there is not a full RSV window before
/// `from` (`from + 1 < period_rsv`), `from` is out of range, or the state is too short.
#[allow(clippy::too_many_arguments)]
pub fn kdj_resume(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_rsv: usize,
    period_k: usize,
    period_d: usize,
    line: KdjLine,
    from: usize,
    state: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = close.len();
    if period_rsv == 0 || from == 0 || from > n || from + 1 < period_rsv || state.is_empty() {
        return None;
    }
    let lo = from + 1 - period_rsv;
    // RSV over [from, n): compute on the windowed tail, then drop its `period_rsv - 1` warm-up.
    let rsv = kdj_rsv(&high[lo..n], &low[lo..n], &close[lo..n], period_rsv);
    let skip = from - lo; // == period_rsv - 1
    let (ak, bk) = (1.0 / period_k as f64, 1.0 - 1.0 / period_k as f64);
    let mut k_prev = state[0];
    match line {
        KdjLine::K => {
            let mut out = Vec::with_capacity(n - from);
            for &r in rsv.iter().skip(skip) {
                k_prev = bk * k_prev + ak * r;
                out.push(k_prev);
            }
            Some((out, vec![k_prev]))
        }
        KdjLine::D | KdjLine::J => {
            if state.len() < 2 {
                return None;
            }
            let (ad, bd) = (1.0 / period_d as f64, 1.0 - 1.0 / period_d as f64);
            let mut d_prev = state[1];
            let is_j = matches!(line, KdjLine::J);
            let mut out = Vec::with_capacity(n - from);
            for &r in rsv.iter().skip(skip) {
                k_prev = bk * k_prev + ak * r;
                d_prev = bd * d_prev + ad * k_prev;
                out.push(if is_j {
                    3.0 * k_prev - 2.0 * d_prev
                } else {
                    d_prev
                });
            }
            Some((out, vec![k_prev, d_prev]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::test_support::*;

    /// KDJ `.k` / `.d` / `.j` resumes, fed the carried %K (and %D) of a full compute over the
    /// head, reproduce the tail of a full compute over the whole input — bit-for-bit. RSV is
    /// recomputed from the windowed tail (a windowed min/max equals the full-series value),
    /// and the ⅓-weight %K/%D SMA recurrences continue from the carried values — so, unlike
    /// the windowed-SMA stochrsi `.d`, every KDJ line is exact.
    #[test]
    fn kdj_resume_is_bit_identical_to_full() {
        let (high, low, close) = ohlc(150);
        let (p, pk, pd, init) = (9usize, 3usize, 3usize, 50.0);
        let k_full = kdj_k(&high, &low, &close, p, pk, init);
        let d_full = kdj_d(&high, &low, &close, p, pk, pd, init);
        let j_full = kdj_j(&high, &low, &close, p, pk, pd, init);
        // `from` spans the first row a full RSV window exists (`p - 1`) through a generic
        // large offset; the carried %K/%D continue the recursion past the dropped head.
        for &from in &[p - 1, p, 30, 80, 149] {
            // `.k` carries just [%K].
            let st_k = kdj_final_state(
                &high[..from],
                &low[..from],
                &close[..from],
                p,
                pk,
                pd,
                init,
                false,
            )
            .unwrap();
            assert_eq!(st_k.len(), 1, "kdj.k carries [%K]");
            let (tail, ret) =
                kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, from, &st_k).unwrap();
            assert_bits(&tail, &k_full[from..], "kdj.k");
            assert_eq!(ret.len(), 1);

            // `.d` / `.j` carry [%K, %D].
            let st_d = kdj_final_state(
                &high[..from],
                &low[..from],
                &close[..from],
                p,
                pk,
                pd,
                init,
                true,
            )
            .unwrap();
            assert_eq!(st_d.len(), 2, "kdj.d/.j carry [%K, %D]");
            let (tail, ret) =
                kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::D, from, &st_d).unwrap();
            assert_bits(&tail, &d_full[from..], "kdj.d");
            assert_eq!(ret.len(), 2);
            let (tail, _) =
                kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::J, from, &st_d).unwrap();
            assert_bits(&tail, &j_full[from..], "kdj.j");
        }
    }

    /// KDJ guards: an empty series declines the final state; a zero RSV period, a zero /
    /// out-of-range / pre-window `from`, an empty state, or a single-element state for
    /// `.d`/`.j` all decline the resume (each then falls back to the correct full recompute).
    #[test]
    fn kdj_guards_decline() {
        let (high, low, close) = ohlc(60);
        let (p, pk, pd, init) = (9usize, 3usize, 3usize, 50.0);
        let n = close.len();

        assert!(kdj_final_state(&[], &[], &[], p, pk, pd, init, false).is_none()); // empty series

        let st =
            kdj_final_state(&high[..40], &low[..40], &close[..40], p, pk, pd, init, true).unwrap();
        assert!(kdj_resume(&high, &low, &close, 0, pk, pd, KdjLine::K, 40, &st).is_none()); // period_rsv == 0
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, 0, &st).is_none()); // from == 0
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, n + 1, &st).is_none()); // from > n
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, p - 2, &st).is_none()); // from + 1 < p
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::K, 40, &[]).is_none()); // empty state
        assert!(kdj_resume(&high, &low, &close, p, pk, pd, KdjLine::D, 40, &[1.0]).is_none());
        // `.d` needs [%K, %D]
    }
}
