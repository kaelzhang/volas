//! Incremental directive resume support for cached computed columns.

use crate::exec::{arg_f64, arg_usize, series_f64};
use crate::types::Node;
use volas_compute::indicators as ind;
use volas_core::{Column, DataFrame};

// --- state-carry resume (additive; fallback path stays correct) -------------
//
// A recursive indicator's whole history compresses into a small fixed-size state
// (a `Vec<f64>`). `initial_state` captures that state after a full compute;
// `execute_resume` continues the recursion over only the new tail rows, producing
// values bit-identical to a fresh full recompute. Both return `None` for any
// directive without a resume kernel, so the caller transparently falls back to the
// correct full-recompute path. Only the canonical no-operand forms (the directives
// volas auto-caches) are handled; an unusual `@`-operand override returns `None`
// and stays on the fallback.

/// Resolve a command node to `(name_lc, sub, args, series)` when it is a plain
/// `Node::Command` (or a bare `Node::Name` no-arg command, e.g. `obv`/`ad`); `None`
/// otherwise. The name is lower-cased and `cdl`→`style` aliased, matching
/// [`exec_command`]. A `Node::Name` carries no sub / args / series — the same way
/// [`execute`] dispatches it via `exec_command(df, name, None, &[], &[])`.
fn as_command(node: &Node) -> Option<(String, Option<String>, &[Option<String>], &[Node])> {
    let lc = |name: &str| {
        let name = name.to_ascii_lowercase();
        if name == "cdl" {
            "style".to_string()
        } else {
            name
        }
    };
    match node {
        Node::Command(cmd) => Some((lc(&cmd.name), cmd.sub.clone(), &cmd.args, &cmd.series)),
        Node::Name(name) if !name.is_empty() => Some((lc(name), None, &[], &[])),
        _ => None,
    }
}

/// The final recursive state after a full compute of `node` against `df`, matching
/// the just-computed `computed` column. `None` when `node` has no resume kernel
/// (the caller then keeps the full-recompute fallback). `computed` is accepted for
/// kernels that can read their state off the output column directly; the cumulative
/// family recomputes its (tiny) state from the raw inputs to stay bit-exact with
/// the canonical kernel.
pub fn initial_state(df: &DataFrame, node: &Node, _computed: &Column) -> Option<Vec<f64>> {
    let (name, sub, args, series) = as_command(node)?;
    let sub = sub.as_deref();
    match (name.as_str(), sub) {
        // Stateless finite-memory indicators need no carried values, but marking
        // them resumable lets append refresh compute only `[valid_rows, height)`.
        ("avgprice" | "medprice" | "typprice" | "wclprice" | "tr", _)
        | ("mom" | "roc" | "rocp" | "rocr" | "rocr100", _) => Some(Vec::new()),

        ("obv", _) => {
            let real = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::obv_final_state(&real, &volume)
        }
        ("ad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            ind::ad_final_state(&high, &low, &close, &volume)
        }
        ("adosc", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            let fast = arg_usize(args, 0, Some(3)).ok()?;
            let slow = arg_usize(args, 1, Some(10)).ok()?;
            ind::adosc_final_state(&high, &low, &close, &volume, fast, slow)
        }

        // Group A cumulative / EMA-recursion family (volas-exclusive).
        ("pvt", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::pvt_final_state(&close, &volume)
        }
        ("nvi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::nvi_final_state(&close, &volume)
        }
        ("pvi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::pvi_final_state(&close, &volume)
        }
        ("efi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            ind::efi_final_state(&close, &volume, arg_usize(args, 0, Some(13)).ok()?)
        }
        ("tsi", _) => ind::tsi_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(25)).ok()?,
            arg_usize(args, 1, Some(13)).ok()?,
        ),
        ("mass_index", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::mass_index_final_state(&high, &low, arg_usize(args, 0, Some(25)).ok()?)
        }

        // Keltner Channels (Group B): the middle line reuses the EMA state; the bands carry
        // the EMA + ATR pair.
        ("keltner", None) => ind::ema_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(20)).ok()?,
        ),
        ("keltner", Some(_)) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::keltner_band_final_state(
                &close,
                &high,
                &low,
                arg_usize(args, 0, Some(20)).ok()?,
                arg_usize(args, 1, Some(10)).ok()?,
            )
        }

        // Group B cumulative family (volas-exclusive): carry the running line + prior bar.
        ("wad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::wad_final_state(&high, &low, &close)
        }
        ("asi", _) => {
            let open = series_f64(df, series, 0, "open").ok()?;
            let high = series_f64(df, series, 1, "high").ok()?;
            let low = series_f64(df, series, 2, "low").ok()?;
            let close = series_f64(df, series, 3, "close").ok()?;
            ind::asi_final_state(&open, &high, &low, &close, arg_f64(args, 0, 3.0).ok()?)
        }

        // SAR family — carry the recurrence's loop state (trend, accel factor(s),
        // extreme point, current SAR, and the prior bar's high/low).
        ("sar", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::sar_final_state(
                &high,
                &low,
                arg_f64(args, 0, 0.02).ok()?,
                arg_f64(args, 1, 0.2).ok()?,
            )
        }
        ("sarext", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::sarext_final_state(
                &high,
                &low,
                arg_f64(args, 0, 0.0).ok()?,
                arg_f64(args, 1, 0.0).ok()?,
                arg_f64(args, 2, 0.02).ok()?,
                arg_f64(args, 3, 0.02).ok()?,
                arg_f64(args, 4, 0.2).ok()?,
                arg_f64(args, 5, 0.02).ok()?,
                arg_f64(args, 6, 0.02).ok()?,
                arg_f64(args, 7, 0.2).ok()?,
            )
        }

        // EMA-recursion family — carry the sub-EMA stage states (see exec's resume block).
        ("ema", _) => ind::ema_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, None).ok()?,
        ),
        ("smma", _) => ind::smma_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, None).ok()?,
        ),
        ("dema", _) => ind::dema_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(30)).ok()?,
        ),
        ("tema", _) => ind::tema_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(30)).ok()?,
        ),
        // T3's carried state is just the six EMA stages (vfactor only scales the combine,
        // not the cascade), so `t3_final_state` needs no vfactor.
        ("t3", _) => ind::t3_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(5)).ok()?,
        ),
        ("trix", _) => ind::trix_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(30)).ok()?,
        ),
        ("kama", _) => ind::kama_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(30)).ok()?,
        ),

        ("macd", None) => ind::macd_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(12)).ok()?,
            arg_usize(args, 1, Some(26)).ok()?,
        ),
        ("macd", Some("signal" | "histogram")) => ind::macd_signal_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(12)).ok()?,
            arg_usize(args, 1, Some(26)).ok()?,
            arg_usize(args, 2, Some(9)).ok()?,
        ),
        ("macdfix", None) => {
            ind::macd_final_state(&series_f64(df, series, 0, "close").ok()?, 12, 26)
        }
        ("macdfix", Some("signal" | "histogram")) => ind::macd_signal_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            12,
            26,
            arg_usize(args, 0, Some(9)).ok()?,
        ),

        // Wilder-smoothing family — carry the running average(s). RSI/CMO carry
        // [avg_gain, avg_loss]; ATR/NATR carry the running ATR; the directional ratios
        // carry the +DM/−DM/TR Wilder sums (and ADX/ADXR additionally the running ADX /
        // its trailing window).
        ("rsi", _) => ind::rsi_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, None).ok()?,
        ),
        ("cmo", _) => ind::cmo_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("atr", _) => ind::atr_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("natr", _) => ind::atr_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("plus_dm", _) => ind::plus_dm_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("minus_dm", _) => ind::minus_dm_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("plus_di", _) => ind::plus_di_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("minus_di", _) => ind::minus_di_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("dx", _) => ind::dx_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("adx", _) => ind::adx_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),
        ("adxr", _) => ind::adxr_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(args, 0, Some(14)).ok()?,
        ),

        // Hilbert-transform family — carry the shared core state (WMA smoother + 4
        // Hilbert channels + homodyne discriminator) plus each output's small tail
        // (DC-phase ring, trendline iTrend triple, MAMA/FAMA accumulators). All read
        // the single `close` series (default), matching their dispatch in `execute`.
        ("ht_dcperiod", _) | ("ht_phasor", _) => {
            ind::ht_core_state(&series_f64(df, series, 0, "close").ok()?)
        }
        ("ht_dcphase", _) => ind::ht_dcphase_state(&series_f64(df, series, 0, "close").ok()?),
        ("ht_sine", _) => ind::ht_sine_state(&series_f64(df, series, 0, "close").ok()?),
        ("ht_trendline", _) => ind::ht_trendline_state(&series_f64(df, series, 0, "close").ok()?),
        ("ht_trendmode", _) => ind::ht_trendmode_state(&series_f64(df, series, 0, "close").ok()?),
        ("mama", _) => ind::mama_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_f64(args, 0, 0.5).ok()?,
            arg_f64(args, 1, 0.05).ok()?,
        ),

        // Index family — carry the incremental tracker's final running extreme
        // `[idx_abs, value]`. Captured at first compute, where `origin == 0`, so the
        // stored index is original-absolute (stable across a later slice).
        ("maxindex", _) | ("minmaxindex", Some("max")) => ind::maxindex_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(30)).ok()?,
        ),
        ("minindex", _) | ("minmaxindex", Some("min")) => ind::minindex_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(args, 0, Some(30)).ok()?,
        ),

        // StochRSI — carry the RSI Wilder pair + the recent RSI tail feeding the windows.
        // A recursive-MA `.d` (matype != 0) keeps the fallback (no resume).
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let is_d = line == "d";
            if is_d && arg_usize(args, 3, Some(0)).ok()? != 0 {
                return None;
            }
            ind::stochrsi_final_state(
                &series_f64(df, series, 0, "close").ok()?,
                arg_usize(args, 0, Some(14)).ok()?,
                arg_usize(args, 1, Some(5)).ok()?,
                is_d,
                arg_usize(args, 2, Some(3)).ok()?,
            )
        }

        // KDJ — carry the recursive %K (and %D for `.d`/`.j`). RSV is finite-memory and is
        // recomputed on resume, not carried; the `init` seed only affects the warm-up.
        ("kdj", Some(line @ ("k" | "d" | "j"))) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let period_rsv = arg_usize(args, 0, Some(9)).ok()?;
            let period_k = arg_usize(args, 1, Some(3)).ok()?;
            let want_d = line != "k";
            let (period_d, init) = if want_d {
                (
                    arg_usize(args, 2, Some(3)).ok()?,
                    arg_f64(args, 3, 50.0).ok()?,
                )
            } else {
                (3, arg_f64(args, 2, 50.0).ok()?)
            };
            ind::kdj_final_state(
                &high, &low, &close, period_rsv, period_k, period_d, init, want_d,
            )
        }

        _ => None,
    }
}

fn default_period(directive: &str, name: &str, default: usize) -> Option<usize> {
    if directive == name {
        return Some(default);
    }
    directive
        .strip_prefix(name)?
        .strip_prefix(':')?
        .parse()
        .ok()
}

/// Parse-free scalar twin of [`execute_resume_default_series`] for the dominant
/// single-bar append case.
pub fn execute_resume_default_series_one(
    df: &DataFrame,
    directive: &str,
    row: usize,
) -> Option<f64> {
    if directive.contains('@') || row >= df.height() {
        return None;
    }
    match directive {
        "avgprice" => {
            let open = series_f64(df, &[], 0, "open").ok()?;
            let high = series_f64(df, &[], 1, "high").ok()?;
            let low = series_f64(df, &[], 2, "low").ok()?;
            let close = series_f64(df, &[], 3, "close").ok()?;
            Some((open[row] + high[row] + low[row] + close[row]) / 4.0)
        }
        "medprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            Some((high[row] + low[row]) / 2.0)
        }
        "typprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            Some((high[row] + low[row] + close[row]) / 3.0)
        }
        "wclprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            Some((high[row] + low[row] + 2.0 * close[row]) / 4.0)
        }
        "tr" => {
            if row == 0 {
                return Some(f64::NAN);
            }
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            let prev_close = close[row - 1];
            Some(
                (high[row] - low[row])
                    .max((high[row] - prev_close).abs())
                    .max((low[row] - prev_close).abs()),
            )
        }
        _ => {
            let (kind, period) = ["mom", "roc", "rocp", "rocr", "rocr100"]
                .into_iter()
                .find_map(|kind| {
                    default_period(directive, kind, 10).map(|period| (kind, period))
                })?;
            if row < period {
                return Some(f64::NAN);
            }
            let data = series_f64(df, &[], 0, "close").ok()?;
            let prior = data[row - period];
            Some(match kind {
                "mom" => data[row] - prior,
                _ if prior == 0.0 => 0.0,
                "roc" => (data[row] / prior - 1.0) * 100.0,
                "rocp" => data[row] / prior - 1.0,
                "rocr" => data[row] / prior,
                "rocr100" => data[row] / prior * 100.0,
                _ => unreachable!(), // LCOV_EXCL_LINE
            })
        }
    }
}

/// Parse-free resume for canonical default-series finite-memory directives.
/// This exists for the single-bar append hot path: the formula is cheaper than
/// reparsing the directive AST. Directives with explicit series (`@...`) or
/// expressions intentionally fall back to [`execute_resume`].
pub fn execute_resume_default_series(
    df: &DataFrame,
    directive: &str,
    from_row: usize,
) -> Option<(Column, Vec<f64>)> {
    if directive.contains('@') {
        return None;
    }
    match directive {
        "avgprice" => {
            let open = series_f64(df, &[], 0, "open").ok()?;
            let high = series_f64(df, &[], 1, "high").ok()?;
            let low = series_f64(df, &[], 2, "low").ok()?;
            let close = series_f64(df, &[], 3, "close").ok()?;
            let mut out = Vec::with_capacity(open.len().saturating_sub(from_row));
            for i in from_row..open.len() {
                out.push((open[i] + high[i] + low[i] + close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "medprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let mut out = Vec::with_capacity(high.len().saturating_sub(from_row));
            for i in from_row..high.len() {
                out.push((high[i] + low[i]) / 2.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "typprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + close[i]) / 3.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "wclprice" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + 2.0 * close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        "tr" => {
            let high = series_f64(df, &[], 0, "high").ok()?;
            let low = series_f64(df, &[], 1, "low").ok()?;
            let close = series_f64(df, &[], 2, "close").ok()?;
            let mut out = Vec::with_capacity(high.len().saturating_sub(from_row));
            for i in from_row..high.len() {
                if i == 0 {
                    out.push(f64::NAN);
                } else {
                    let prev_close = close[i - 1];
                    out.push(
                        (high[i] - low[i])
                            .max((high[i] - prev_close).abs())
                            .max((low[i] - prev_close).abs()),
                    );
                }
            }
            Some((Column::f64(out), Vec::new()))
        }
        _ => {
            let (kind, period) = ["mom", "roc", "rocp", "rocr", "rocr100"]
                .into_iter()
                .find_map(|kind| {
                    default_period(directive, kind, 10).map(|period| (kind, period))
                })?;
            let data = series_f64(df, &[], 0, "close").ok()?;
            let mut out = Vec::with_capacity(data.len().saturating_sub(from_row));
            for i in from_row..data.len() {
                if i < period {
                    out.push(f64::NAN);
                    continue;
                }
                let prior = data[i - period];
                let value = match kind {
                    "mom" => data[i] - prior,
                    _ if prior == 0.0 => 0.0,
                    "roc" => (data[i] / prior - 1.0) * 100.0,
                    "rocp" => data[i] / prior - 1.0,
                    "rocr" => data[i] / prior,
                    "rocr100" => data[i] / prior * 100.0,
                    _ => unreachable!(), // LCOV_EXCL_LINE
                };
                out.push(value);
            }
            Some((Column::f64(out), Vec::new()))
        }
    }
}

/// Resume `node` from `prev_state` over rows `[from_row, height)`, returning the
/// new-row [`Column`] and the updated state. `None` when `node` has no resume
/// kernel (caller falls back to a full recompute). The values are bit-identical to
/// a fresh full recompute, so writing them into the stale tail keeps the cached
/// column exact.
///
/// `origin` is the original-frame row this (possibly sliced) frame's row 0 maps to
/// (`ComputedMeta::origin`). Recursive *value* indicators ignore it; the
/// absolute-position index family adds it back so emitted positions stay
/// original-absolute across a head-dropping slice.
pub fn execute_resume(
    df: &DataFrame,
    node: &Node,
    prev_state: &[f64],
    from_row: usize,
    origin: usize,
) -> Option<(Column, Vec<f64>)> {
    let (name, sub, args, series) = as_command(node)?;
    let sub = sub.as_deref();
    let close = || series_f64(df, series, 0, "close");
    match (name.as_str(), sub) {
        // Stateless finite-memory resume. These kernels depend only on the row
        // being refreshed plus a fixed prior row, so append can produce exactly the
        // stale tail without probing a window or recomputing the full column.
        ("avgprice", _) => {
            let open = series_f64(df, series, 0, "open").ok()?;
            let high = series_f64(df, series, 1, "high").ok()?;
            let low = series_f64(df, series, 2, "low").ok()?;
            let close = series_f64(df, series, 3, "close").ok()?;
            let mut out = Vec::with_capacity(open.len().saturating_sub(from_row));
            for i in from_row..open.len() {
                out.push((open[i] + high[i] + low[i] + close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("medprice", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let mut out = Vec::with_capacity(high.len().saturating_sub(from_row));
            for i in from_row..high.len() {
                out.push((high[i] + low[i]) / 2.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("typprice", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + close[i]) / 3.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("wclprice", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let mut out = Vec::with_capacity(close.len().saturating_sub(from_row));
            for i in from_row..close.len() {
                out.push((high[i] + low[i] + 2.0 * close[i]) / 4.0);
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("tr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let mut out = Vec::with_capacity(high.len().saturating_sub(from_row));
            for i in from_row..high.len() {
                if i == 0 {
                    out.push(f64::NAN);
                } else {
                    let prev_close = close[i - 1];
                    out.push(
                        (high[i] - low[i])
                            .max((high[i] - prev_close).abs())
                            .max((low[i] - prev_close).abs()),
                    );
                }
            }
            Some((Column::f64(out), Vec::new()))
        }
        ("mom" | "roc" | "rocp" | "rocr" | "rocr100", _) => {
            let data = close().ok()?;
            let period = arg_usize(args, 0, Some(10)).ok()?;
            let mut out = Vec::with_capacity(data.len().saturating_sub(from_row));
            for i in from_row..data.len() {
                if i < period {
                    out.push(f64::NAN);
                    continue;
                }
                let prior = data[i - period];
                let value = match name.as_str() {
                    "mom" => data[i] - prior,
                    _ if prior == 0.0 => 0.0,
                    "roc" => (data[i] / prior - 1.0) * 100.0,
                    "rocp" => data[i] / prior - 1.0,
                    "rocr" => data[i] / prior,
                    "rocr100" => data[i] / prior * 100.0,
                    _ => unreachable!(), // LCOV_EXCL_LINE
                };
                out.push(value);
            }
            Some((Column::f64(out), Vec::new()))
        }

        ("obv", _) => {
            let real = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) = ind::obv_resume(&real, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("ad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            let (vals, st) = ind::ad_resume(&high, &low, &close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("adosc", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let volume = series_f64(df, series, 3, "volume").ok()?;
            let fast = arg_usize(args, 0, Some(3)).ok()?;
            let slow = arg_usize(args, 1, Some(10)).ok()?;
            let (vals, st) = ind::adosc_resume(
                &high, &low, &close, &volume, fast, slow, from_row, prev_state,
            );
            Some((Column::f64(vals), st))
        }

        // Group A cumulative / EMA-recursion family (volas-exclusive).
        ("pvt", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) = ind::pvt_resume(&close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("nvi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) = ind::nvi_resume(&close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("pvi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) = ind::pvi_resume(&close, &volume, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("efi", _) => {
            let close = series_f64(df, series, 0, "close").ok()?;
            let volume = series_f64(df, series, 1, "volume").ok()?;
            let (vals, st) =
                ind::efi_resume(&close, &volume, arg_usize(args, 0, Some(13)).ok()?, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("tsi", _) => {
            let (vals, st) = ind::tsi_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(25)).ok()?,
                arg_usize(args, 1, Some(13)).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("mass_index", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) =
                ind::mass_index_resume(&high, &low, arg_usize(args, 0, Some(25)).ok()?, from_row, prev_state);
            Some((Column::f64(vals), st))
        }

        // Keltner Channels (Group B): middle resumes the EMA; bands resume the EMA + ATR pair.
        ("keltner", None) => {
            let (vals, st) = ind::ema_resume(
                &series_f64(df, series, 0, "close").ok()?,
                arg_usize(args, 0, Some(20)).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("keltner", Some(sub)) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::keltner_band_resume(
                &close,
                &high,
                &low,
                arg_usize(args, 0, Some(20)).ok()?,
                arg_usize(args, 1, Some(10)).ok()?,
                arg_f64(args, 2, 2.0).ok()?,
                sub == "upper",
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // Group B cumulative family (volas-exclusive): resume the running line.
        ("wad", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::wad_resume(&high, &low, &close, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("asi", _) => {
            let open = series_f64(df, series, 0, "open").ok()?;
            let high = series_f64(df, series, 1, "high").ok()?;
            let low = series_f64(df, series, 2, "low").ok()?;
            let close = series_f64(df, series, 3, "close").ok()?;
            let (vals, st) = ind::asi_resume(
                &open,
                &high,
                &low,
                &close,
                arg_f64(args, 0, 3.0).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }

        // SAR family — resume the state machine from the carried tuple. A resume at
        // `from_row < 2` (the SAR bootstrap needs bars 0 and 1) returns `None` and falls back.
        ("sar", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) = ind::sar_resume(
                &high,
                &low,
                arg_f64(args, 0, 0.02).ok()?,
                arg_f64(args, 1, 0.2).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("sarext", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            // `start_value` (arg 0) only steers the bar-1 bootstrap, never re-run on resume.
            let (vals, st) = ind::sarext_resume(
                &high,
                &low,
                arg_f64(args, 1, 0.0).ok()?,
                arg_f64(args, 2, 0.02).ok()?,
                arg_f64(args, 3, 0.02).ok()?,
                arg_f64(args, 4, 0.2).ok()?,
                arg_f64(args, 5, 0.02).ok()?,
                arg_f64(args, 6, 0.02).ok()?,
                arg_f64(args, 7, 0.2).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // EMA-recursion family — resume each carried sub-EMA from its last value (skipping
        // the SMA seed), bit-identical to the full kernel's steady-state recurrence.
        ("ema", _) => {
            let (vals, st) = ind::ema_resume(
                &close().ok()?,
                arg_usize(args, 0, None).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("smma", _) => {
            let (vals, st) = ind::smma_resume(
                &close().ok()?,
                arg_usize(args, 0, None).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("dema", _) => {
            let (vals, st) = ind::dema_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(30)).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("tema", _) => {
            let (vals, st) = ind::tema_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(30)).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("t3", _) => {
            let (vals, st) = ind::t3_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(5)).ok()?,
                arg_f64(args, 1, 0.7).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("trix", _) => {
            let (vals, st) = ind::trix_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(30)).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        // KAMA's sliding-sum resume can decline (short retained head) → None falls back.
        ("kama", _) => {
            let (vals, st) = ind::kama_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(30)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        ("macd", None) => {
            let (vals, st) = ind::macd_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(12)).ok()?,
                arg_usize(args, 1, Some(26)).ok()?,
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("macd", Some(line @ ("signal" | "histogram"))) => {
            let (vals, st) = ind::macd_signal_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(12)).ok()?,
                arg_usize(args, 1, Some(26)).ok()?,
                arg_usize(args, 2, Some(9)).ok()?,
                line == "histogram",
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }
        ("macdfix", None) => {
            let (vals, st) = ind::macd_resume(&close().ok()?, 12, 26, from_row, prev_state);
            Some((Column::f64(vals), st))
        }
        ("macdfix", Some(line @ ("signal" | "histogram"))) => {
            let (vals, st) = ind::macd_signal_resume(
                &close().ok()?,
                12,
                26,
                arg_usize(args, 0, Some(9)).ok()?,
                line == "histogram",
                from_row,
                prev_state,
            );
            Some((Column::f64(vals), st))
        }

        // Wilder-smoothing family — resume the running average(s) over the new rows.
        // Each reads only `…[from_row-1..]` (the per-bar term needs the prior bar), so a
        // resume at `from_row == 0` returns `None` (falls back). DM/DI/DX/ADX/ADXR pull
        // high/low(/close); RSI/CMO/ATR/NATR their named series.
        ("rsi", _) => {
            let (vals, st) = ind::rsi_resume(
                &close().ok()?,
                arg_usize(args, 0, None).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("cmo", _) => {
            let (vals, st) = ind::cmo_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("atr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::atr_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("natr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::natr_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("plus_dm", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) = ind::plus_dm_resume(
                &high,
                &low,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minus_dm", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let (vals, st) = ind::minus_dm_resume(
                &high,
                &low,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("plus_di", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::plus_di_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minus_di", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::minus_di_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("dx", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::dx_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("adx", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::adx_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("adxr", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let (vals, st) = ind::adxr_resume(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // Hilbert-transform family — reconstruct the shared core + per-output tail and
        // step the recurrence over the new rows. A resume at/under the core warm-up
        // (or, for the price-windowed trendline/trendmode, before a full dominant-cycle
        // window is visible) returns `None` and falls back to the full recompute.
        ("ht_dcperiod", _) => {
            let (vals, st) = ind::ht_dcperiod_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_phasor", sub) => {
            let (vals, st) = ind::ht_phasor_resume(
                &close().ok()?,
                sub == Some("quadrature"),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("ht_dcphase", _) => {
            let (vals, st) = ind::ht_dcphase_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_sine", sub) => {
            let (vals, st) = ind::ht_sine_resume(
                &close().ok()?,
                sub == Some("leadsine"),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("ht_trendline", _) => {
            let (vals, st) = ind::ht_trendline_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("ht_trendmode", _) => {
            let (vals, st) = ind::ht_trendmode_resume(&close().ok()?, from_row, prev_state)?;
            Some((Column::f64(vals), st))
        }
        ("mama", sub) => {
            let (vals, st) = ind::mama_resume(
                &close().ok()?,
                arg_f64(args, 0, 0.5).ok()?,
                arg_f64(args, 1, 0.05).ok()?,
                sub == Some("fama"),
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // Index family — windowed arg-extreme emitting ABSOLUTE positions. The carried
        // state is the incremental tracker's running extreme `[idx_abs, value]`; `origin`
        // rebases sub-frame positions back to original-absolute. minmaxindex.max / .min
        // are exactly maxindex / minindex (see `execute`).
        ("maxindex", _) | ("minmaxindex", Some("max")) => {
            let (vals, st) = ind::maxindex_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(30)).ok()?,
                from_row,
                origin,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }
        ("minindex", _) | ("minmaxindex", Some("min")) => {
            let (vals, st) = ind::minindex_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(30)).ok()?,
                from_row,
                origin,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // StochRSI — a windowed %K (and SMA `.d`) of the Wilder-recursive RSI; resume by
        // carrying the RSI Wilder pair + the recent RSI values feeding the windows. Only
        // the canonical SMA `.d` (matype 0) resumes; a recursive-MA `.d` declines.
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let is_d = line == "d";
            if is_d && arg_usize(args, 3, Some(0)).ok()? != 0 {
                return None; // non-SMA `.d` smoothing is recursive — fall back.
            }
            let (vals, st) = ind::stochrsi_resume(
                &close().ok()?,
                arg_usize(args, 0, Some(14)).ok()?,
                arg_usize(args, 1, Some(5)).ok()?,
                is_d,
                arg_usize(args, 2, Some(3)).ok()?,
                from_row,
                prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        // KDJ — resume the recursive %K (+ %D for `.d`/`.j`) from the carried state; RSV is
        // finite-memory, recomputed over the windowed high/low/close tail.
        ("kdj", Some(line @ ("k" | "d" | "j"))) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let period_rsv = arg_usize(args, 0, Some(9)).ok()?;
            let period_k = arg_usize(args, 1, Some(3)).ok()?;
            let kline = match line {
                "k" => ind::KdjLine::K,
                "d" => ind::KdjLine::D,
                _ => ind::KdjLine::J,
            };
            let period_d = if line == "k" {
                3
            } else {
                arg_usize(args, 2, Some(3)).ok()?
            };
            let (vals, st) = ind::kdj_resume(
                &high, &low, &close, period_rsv, period_k, period_d, kline, from_row, prev_state,
            )?;
            Some((Column::f64(vals), st))
        }

        _ => None,
    }
}
