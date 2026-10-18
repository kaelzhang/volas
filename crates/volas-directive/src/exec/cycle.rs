//! Command dispatch — directional movement, regression / statistics, volume,
//! price transforms, and the Hilbert cycle suite (final group of the chain).

use super::*;
use volas_core::{Result, VolasError};
use crate::types::ArgValue;
use volas_compute::raw::indicators as ind;
use volas_core::Column;

pub(super) fn dispatch(
    ops: &Operands,
    name: &str,
    sub: Option<&str>,
    args: &[ArgValue],
) -> Result<Column> {
    let close = |i| ops.f64(i, "close");
    let f64col = |v: Vec<f64>| Ok(Column::f64(v));
    match (name, sub) {
        // Directional movement family (+DM/-DM need only high/low; the rest add close).
        ("plus_dm", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            f64col(ind::plus_dm(&high, &low, arg_usize(args, 0)))
        }
        ("minus_dm", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            f64col(ind::minus_dm(&high, &low, arg_usize(args, 0)))
        }
        ("plus_di", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::plus_di(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
            ))
        }
        ("minus_di", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::minus_di(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
            ))
        }
        ("dx", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::dx(&high, &low, &close, arg_usize(args, 0)))
        }
        ("adx", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::adx(&high, &low, &close, arg_usize(args, 0)))
        }
        ("adxr", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::adxr(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
            ))
        }
        ("cci", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::cci(&high, &low, &close, arg_usize(args, 0), ops.origin()))
        }
        ("imi", _) => {
            let open = ops.f64(0, "open")?;
            let close = ops.f64(1, "close")?;
            f64col(ind::imi(&open, &close, arg_usize(args, 0), ops.origin()))
        }
        ("psy", _) => { let c = close(0)?; f64col(ind::psy_with(&c, arg_usize(args, 0), ops.origin(), ops.clean())) },
        ("pvt", _) => {
            let close = ops.f64(0, "close")?;
            let volume = ops.f64(1, "volume")?;
            f64col(ind::pvt(&close, &volume))
        }
        ("nvi", _) => {
            let close = ops.f64(0, "close")?;
            let volume = ops.f64(1, "volume")?;
            f64col(ind::nvi(&close, &volume))
        }
        ("pvi", _) => {
            let close = ops.f64(0, "close")?;
            let volume = ops.f64(1, "volume")?;
            f64col(ind::pvi(&close, &volume))
        }
        ("dpo", _) => f64col(ind::dpo(&close(0)?, arg_usize(args, 0), ops.origin())),
        ("cmf", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            let volume = ops.f64(3, "volume")?;
            f64col(ind::cmf_pass_with(&high, &low, &close, &volume, arg_usize(args, 0), ops.origin(), ops.clean(), &mut ()))
        }
        ("chop", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::chop(&high, &low, &close, arg_usize(args, 0), ops.origin()))
        }
        ("kst", _) => { let c = close(0)?; f64col(ind::kst_pass_with(&c, ops.origin(), ops.clean(), &mut ())) },
        ("emv", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let volume = ops.f64(2, "volume")?;
            f64col(ind::emv_pass_with(&high, &low, &volume, arg_usize(args, 0), ops.origin(), ops.clean(), &mut ()))
        }
        ("mass_index", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            f64col(ind::mass_index(&high, &low, arg_usize(args, 0), ops.origin()))
        }
        ("efi", _) => {
            let close = ops.f64(0, "close")?;
            let volume = ops.f64(1, "volume")?;
            f64col(ind::efi(&close, &volume, arg_usize(args, 0)))
        }
        ("tsi", _) => f64col(ind::tsi(
            &close(0)?,
            arg_usize(args, 0),
            arg_usize(args, 1),
        )),
        ("crsi", _) => f64col(ind::crsi(
            &close(0)?,
            arg_usize(args, 0),
            arg_usize(args, 1),
            arg_usize(args, 2),
        )),
        ("trix", _) => f64col(ind::trix(&close(0)?, arg_usize(args, 0))),
        ("aroon", Some(dir @ ("up" | "down"))) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let period = arg_usize(args, 0);
            let v = if dir == "up" {
                ind::aroon_up(&high, &low, period)
            } else {
                ind::aroon_down(&high, &low, period)
            };
            f64col(v)
        }
        ("aroonosc", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            f64col(ind::aroonosc(&high, &low, arg_usize(args, 0)))
        }

        ("sum", _) => f64col(ind::sum(&close(0)?, arg_usize(args, 0), ops.origin())),
        ("maxindex", _) => {
            let close = close(0)?;
            f64col(ind::maxindex_with(&close, arg_usize(args, 0), ops.origin(), ops.all_finite()))
        },
        ("minindex", _) => {
            let close = close(0)?;
            f64col(ind::minindex_with(&close, arg_usize(args, 0), ops.origin(), ops.all_finite()))
        },
        // minmax / minmaxindex are the (min, max) value / index pair over the window;
        // their outputs are exactly llv/hhv (values) and minindex/maxindex (indices).
        ("minmax", Some("min")) => f64col(ind::llv(&close(0)?, arg_usize(args, 0))),
        ("minmax", Some("max")) => f64col(ind::hhv(&close(0)?, arg_usize(args, 0))),
        ("minmaxindex", Some("min")) => {
            {
            let close = close(0)?;
            f64col(ind::minindex_with(&close, arg_usize(args, 0), ops.origin(), ops.all_finite()))
        }
        }
        ("minmaxindex", Some("max")) => {
            {
            let close = close(0)?;
            f64col(ind::maxindex_with(&close, arg_usize(args, 0), ops.origin(), ops.all_finite()))
        }
        }
        ("natr", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::natr(
                &high,
                &low,
                &close,
                arg_usize(args, 0),
            ))
        }
        ("bop", _) => {
            let open = ops.f64(0, "open")?;
            let high = ops.f64(1, "high")?;
            let low = ops.f64(2, "low")?;
            let close = ops.f64(3, "close")?;
            f64col(ind::bop(&open, &high, &low, &close))
        }

        ("linearreg", _) => f64col(ind::linearreg(&close(0)?, arg_usize(args, 0), ops.origin())),
        ("linearreg_slope", _) => f64col(ind::linearreg_slope(
            &close(0)?,
            arg_usize(args, 0),
            ops.origin(),
        )),
        ("linearreg_intercept", _) => f64col(ind::linearreg_intercept(
            &close(0)?,
            arg_usize(args, 0),
            ops.origin(),
        )),
        ("linearreg_angle", _) => f64col(ind::linearreg_angle(
            &close(0)?,
            arg_usize(args, 0),
            ops.origin(),
        )),
        ("tsf", _) => f64col(ind::tsf(&close(0)?, arg_usize(args, 0), ops.origin())),
        // beta/correl relate two series: the first defaults to close, the second is required.
        ("correl", _) => {
            let x = close(0)?;
            let y = ops.f64_required(1)?;
            f64col(ind::correl(&x, &y, arg_usize(args, 0), ops.origin()))
        }
        ("beta", _) => {
            let x = close(0)?;
            let y = ops.f64_required(1)?;
            f64col(ind::beta(&x, &y, arg_usize(args, 0), ops.origin()))
        }

        ("var", _) => f64col(ind::var(&close(0)?, arg_usize(args, 0), ops.origin())),

        // pandas-window statistics as directives: one kernel source with the
        // rolling API (volas_compute::window), full-window semantics
        // (min_periods = period, so an NA-bearing window stays NA).
        ("median", _) => {
            let p = arg_usize(args, 0);
            f64col(volas_compute::window::median(&close(0)?, p, p).map_err(|e| volas_core::VolasError::Value(e.to_string()))?)
        }
        ("quantile", _) => {
            let p = arg_usize(args, 0);
            let q = arg_f64(args, 1);
            f64col(volas_compute::window::quantile(&close(0)?, p, p, q, "linear").map_err(|e| volas_core::VolasError::Value(e.to_string()))?)
        }
        // percent rank of the current bar within its window, in (0, 1].
        ("rank", _) => {
            let p = arg_usize(args, 0);
            f64col(volas_compute::window::rank(&close(0)?, p, p, "average", true, true).map_err(|e| volas_core::VolasError::Value(e.to_string()))?)
        }
        ("skew", _) => {
            let p = arg_usize(args, 0);
            f64col(volas_compute::window::skew(&close(0)?, p, p).map_err(|e| volas_core::VolasError::Value(e.to_string()))?)
        }
        ("kurt", _) => {
            let p = arg_usize(args, 0);
            f64col(volas_compute::window::kurt(&close(0)?, p, p).map_err(|e| volas_core::VolasError::Value(e.to_string()))?)
        }
        ("sem", _) => {
            let p = arg_usize(args, 0);
            f64col(volas_compute::window::sem(&close(0)?, p, p, 1).map_err(|e| volas_core::VolasError::Value(e.to_string()))?)
        }
        // TradingView dispersion / order statistics (window family).
        ("dev", _) => {
            let p = arg_usize(args, 0);
            f64col(volas_compute::window::dev(&close(0)?, p, p).map_err(|e| volas_core::VolasError::Value(e.to_string()))?)
        }
        ("mode", _) => {
            let p = arg_usize(args, 0);
            f64col(volas_compute::window::mode(&close(0)?, p, p).map_err(|e| volas_core::VolasError::Value(e.to_string()))?)
        }
        // TradingView oscillators (cog / rci) and the per-bar volume study iii.
        ("cog", _) => f64col(ind::cog(&close(0)?, arg_usize(args, 0), ops.origin())),
        ("rci", _) => f64col(ind::rci(&close(0)?, arg_usize(args, 0))),
        ("iii", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            let volume = ops.f64(3, "volume")?;
            f64col(if ops.clean() { ind::iii_complete(&high, &low, &close, &volume) } else { ind::iii(&high, &low, &close, &volume) })
        }
        ("kcw", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::kcw(
                &close,
                &high,
                &low,
                arg_usize(args, 0),
                arg_usize(args, 1),
                arg_f64(args, 2),
            ))
        }
        ("pivothigh" | "pivotlow", _) => {
            let high = name == "pivothigh";
            let source = ops.f64(0, if high { "high" } else { "low" })?;
            f64col(ind::pivot_with(&source, arg_usize(args, 0), arg_usize(args, 1), high, ops.all_finite()))
        }
        ("stddev", _) => f64col(ind::stddev(
            &close(0)?,
            arg_usize(args, 0),
            arg_f64(args, 1),
            ops.origin(),
        )),

        ("obv", _) => {
            let real = ops.f64(0, "close")?;
            let volume = ops.f64(1, "volume")?;
            f64col(ind::obv(&real, &volume))
        }
        ("ad", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            let volume = ops.f64(3, "volume")?;
            f64col(ind::ad(&high, &low, &close, &volume))
        }
        ("adosc", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            let volume = ops.f64(3, "volume")?;
            f64col(ind::adosc(
                &high,
                &low,
                &close,
                &volume,
                arg_usize(args, 0),
                arg_usize(args, 1),
            ))
        }

        ("avgprice", _) => {
            let open = ops.f64(0, "open")?;
            let high = ops.f64(1, "high")?;
            let low = ops.f64(2, "low")?;
            let close = ops.f64(3, "close")?;
            f64col(ind::avgprice_with(&open, &high, &low, &close, ops.clean()))
        }
        ("medprice", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            f64col(ind::medprice_with(&high, &low, ops.clean()))
        }
        ("typprice", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::typprice_with(&high, &low, &close, ops.clean()))
        }
        ("wclprice", _) => {
            let high = ops.f64(0, "high")?;
            let low = ops.f64(1, "low")?;
            let close = ops.f64(2, "close")?;
            f64col(ind::wclprice_with(&high, &low, &close, ops.clean()))
        }

        // Hilbert-transform cycle suite. Each takes a single price series (default
        // close); multi-output ones expose the secondary line via a sub-command (P3).
        ("ht_dcperiod", _) => f64col(ind::ht_dcperiod(&close(0)?)),
        ("ht_dcphase", _) => f64col(ind::ht_dcphase(&close(0)?)),
        ("ht_trendline", _) => f64col(ind::ht_trendline(&close(0)?)),
        ("ht_trendmode", _) => f64col(ind::ht_trendmode(&close(0)?)),
        ("ht_phasor", None) => f64col(ind::ht_phasor_line(&close(0)?, false)),
        ("ht_phasor", Some("quadrature")) => f64col(ind::ht_phasor_line(&close(0)?, true)),
        ("ht_sine", None) => f64col(ind::ht_sine(&close(0)?).0),
        ("ht_sine", Some("leadsine")) => f64col(ind::ht_sine(&close(0)?).1),
        ("mama", None) => f64col(ind::mama_line(
            &close(0)?,
            arg_f64(args, 0),
            arg_f64(args, 1),
            false,
        )),
        ("mama", Some("fama")) => f64col(ind::mama_line(
            &close(0)?,
            arg_f64(args, 0),
            arg_f64(args, 1),
            true,
        )),

        (other, _) => Err(VolasError::Value(format!("unknown command '{other}'"))), // LCOV_EXCL_LINE
    }
}
