//! Capture a recursive indicator's carried state after a full compute, so an
//! append can resume over only the new tail rows (per-directive `initial_state`).

use crate::exec::{arg_f64, arg_usize, series_f64};
use crate::types::Ast;
use volas_compute::indicators as ind;
use volas_core::{Column, DataFrame};

use super::as_command;

/// The final recursive state after a full compute of `node` against `df`, matching
/// the just-computed `computed` column. `None` when `node` has no resume kernel
/// (the caller then keeps the full-recompute fallback). `computed` is accepted for
/// kernels that can read their state off the output column directly; the cumulative
/// family recomputes its (tiny) state from the raw inputs to stay bit-exact with
/// the canonical kernel.
pub fn initial_state(df: &DataFrame, node: &Ast, _computed: &Column) -> Option<Vec<f64>> {
    let (name, sub, args, series) = as_command(node)?;
    let sub = sub.as_deref();
    match (name.as_str(), sub) {
        // Stateless finite-memory indicators need no carried values, but marking
        // them resumable lets append refresh compute only `[valid_rows, height)`.
        ("avgprice" | "medprice" | "typprice" | "wclprice" | "tr" | "bop", _)
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
            let fast = arg_usize(&args, 0);
            let slow = arg_usize(&args, 1);
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
            ind::efi_final_state(&close, &volume, arg_usize(&args, 0))
        }
        ("tsi", _) => ind::tsi_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
            arg_usize(&args, 1),
        ),
        ("mass_index", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::mass_index_final_state(&high, &low, arg_usize(&args, 0))
        }

        // Keltner Channels (Group B): the middle line reuses the EMA state; the bands carry
        // the EMA + ATR pair.
        ("keltner", None) => ind::ema_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("keltner", Some(_)) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::keltner_band_final_state(
                &close,
                &high,
                &low,
                arg_usize(&args, 0),
                arg_usize(&args, 1),
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
            ind::asi_final_state(&open, &high, &low, &close, arg_f64(&args, 0))
        }
        ("supertrend", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            ind::supertrend_final_state(
                &high,
                &low,
                &close,
                arg_usize(&args, 0),
                arg_f64(&args, 1),
            )
        }

        // SAR family — carry the recurrence's loop state (trend, accel factor(s),
        // extreme point, current SAR, and the prior bar's high/low).
        ("sar", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::sar_final_state(
                &high,
                &low,
                arg_f64(&args, 0),
                arg_f64(&args, 1),
            )
        }
        ("sarext", _) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            ind::sarext_final_state(
                &high,
                &low,
                arg_f64(&args, 0),
                arg_f64(&args, 1),
                arg_f64(&args, 2),
                arg_f64(&args, 3),
                arg_f64(&args, 4),
                arg_f64(&args, 5),
                arg_f64(&args, 6),
                arg_f64(&args, 7),
            )
        }

        // EMA-recursion family — carry the sub-EMA stage states (see exec's resume block).
        ("ema", _) => ind::ema_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("smma", _) => ind::smma_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("dema", _) => ind::dema_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("tema", _) => ind::tema_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        // T3's carried state is just the six EMA stages (vfactor only scales the combine,
        // not the cascade), so `t3_final_state` needs no vfactor.
        ("t3", _) => ind::t3_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("trix", _) => ind::trix_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("kama", _) => ind::kama_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),

        ("macd", None) => ind::macd_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
            arg_usize(&args, 1),
        ),
        ("macd", Some("signal" | "histogram")) => ind::macd_signal_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
            arg_usize(&args, 1),
            arg_usize(&args, 2),
        ),
        ("macdfix", None) => {
            ind::macd_final_state(&series_f64(df, series, 0, "close").ok()?, 12, 26)
        }
        ("macdfix", Some("signal" | "histogram")) => ind::macd_signal_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            12,
            26,
            arg_usize(&args, 0),
        ),

        // Wilder-smoothing family — carry the running average(s). RSI/CMO carry
        // [avg_gain, avg_loss]; ATR/NATR carry the running ATR; the directional ratios
        // carry the +DM/−DM/TR Wilder sums (and ADX/ADXR additionally the running ADX /
        // its trailing window).
        ("rsi", _) => ind::rsi_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("cmo", _) => ind::cmo_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("atr", _) => ind::atr_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("natr", _) => ind::atr_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("plus_dm", _) => ind::plus_dm_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            arg_usize(&args, 0),
        ),
        ("minus_dm", _) => ind::minus_dm_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            arg_usize(&args, 0),
        ),
        ("plus_di", _) => ind::plus_di_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("minus_di", _) => ind::minus_di_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("dx", _) => ind::dx_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("adx", _) => ind::adx_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("adxr", _) => ind::adxr_final_state(
            &series_f64(df, series, 0, "high").ok()?,
            &series_f64(df, series, 1, "low").ok()?,
            &series_f64(df, series, 2, "close").ok()?,
            arg_usize(&args, 0),
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
            arg_f64(&args, 0),
            arg_f64(&args, 1),
        ),

        // Index family — carry the incremental tracker's final running extreme
        // `[idx_abs, value]`. Captured at first compute, where `origin == 0`, so the
        // stored index is original-absolute (stable across a later slice).
        ("maxindex", _) | ("minmaxindex", Some("max")) => ind::maxindex_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),
        ("minindex", _) | ("minmaxindex", Some("min")) => ind::minindex_final_state(
            &series_f64(df, series, 0, "close").ok()?,
            arg_usize(&args, 0),
        ),

        // StochRSI — carry the RSI Wilder pair + the recent RSI tail feeding the windows.
        // A recursive-MA `.d` (matype != 0) keeps the fallback (no resume).
        ("stochrsi", Some(line @ ("k" | "d"))) => {
            let is_d = line == "d";
            if is_d && arg_usize(&args, 3) != 0 {
                return None;
            }
            ind::stochrsi_final_state(
                &series_f64(df, series, 0, "close").ok()?,
                arg_usize(&args, 0),
                arg_usize(&args, 1),
                is_d,
                arg_usize(&args, 2),
            )
        }

        // KDJ — carry the recursive %K (and %D for `.d`/`.j`). RSV is finite-memory and is
        // recomputed on resume, not carried; the `init` seed only affects the warm-up.
        ("kdj", Some(line @ ("k" | "d" | "j"))) => {
            let high = series_f64(df, series, 0, "high").ok()?;
            let low = series_f64(df, series, 1, "low").ok()?;
            let close = series_f64(df, series, 2, "close").ok()?;
            let period_rsv = arg_usize(&args, 0);
            let period_k = arg_usize(&args, 1);
            let want_d = line != "k";
            let (period_d, init) = if want_d {
                (
                    arg_usize(&args, 2),
                    arg_f64(&args, 3),
                )
            } else {
                (3, arg_f64(&args, 2))
            };
            ind::kdj_final_state(
                &high, &low, &close, period_rsv, period_k, period_d, init, want_d,
            )
        }

        _ => None,
    }
}
