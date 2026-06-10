//! Command dispatch — directional movement, regression / statistics, volume,
//! price transforms, and the Hilbert cycle suite (final group of the chain).

use super::*;
use volas_core::{Result, VolasError};
use crate::types::Node;
use volas_compute::indicators as ind;
use volas_core::{Column, DataFrame};

pub(super) fn dispatch(
    df: &DataFrame,
    name: &str,
    sub: Option<&str>,
    args: &[Option<String>],
    series: &[Node],
) -> Result<Column> {
    let close = |i| series_f64(df, series, i, "close");
    let f64col = |v: Vec<f64>| Ok(Column::f64(v));
    match (name, sub) {
        // Directional movement family (+DM/-DM need only high/low; the rest add close).
        ("plus_dm", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::plus_dm(&high, &low, arg_usize(args, 0, Some(14))?))
        }
        ("minus_dm", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::minus_dm(&high, &low, arg_usize(args, 0, Some(14))?))
        }
        ("plus_di", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::plus_di(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14))?,
            ))
        }
        ("minus_di", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::minus_di(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14))?,
            ))
        }
        ("dx", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::dx(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("adx", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::adx(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("adxr", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::adxr(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14))?,
            ))
        }
        ("cci", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::cci(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("imi", _) => {
            let open = series_f64(df, series, 0, "open")?;
            let close = series_f64(df, series, 1, "close")?;
            f64col(ind::imi(&open, &close, arg_usize(args, 0, Some(14))?))
        }
        ("psy", _) => f64col(ind::psy(&close(0)?, arg_usize(args, 0, Some(12))?)),
        ("pvt", _) => {
            let close = series_f64(df, series, 0, "close")?;
            let volume = series_f64(df, series, 1, "volume")?;
            f64col(ind::pvt(&close, &volume))
        }
        ("nvi", _) => {
            let close = series_f64(df, series, 0, "close")?;
            let volume = series_f64(df, series, 1, "volume")?;
            f64col(ind::nvi(&close, &volume))
        }
        ("pvi", _) => {
            let close = series_f64(df, series, 0, "close")?;
            let volume = series_f64(df, series, 1, "volume")?;
            f64col(ind::pvi(&close, &volume))
        }
        ("dpo", _) => f64col(ind::dpo(&close(0)?, arg_usize(args, 0, Some(20))?)),
        ("cmf", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let volume = series_f64(df, series, 3, "volume")?;
            f64col(ind::cmf(&high, &low, &close, &volume, arg_usize(args, 0, Some(20))?))
        }
        ("chop", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::chop(&high, &low, &close, arg_usize(args, 0, Some(14))?))
        }
        ("kst", _) => f64col(ind::kst(&close(0)?)),
        ("emv", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let volume = series_f64(df, series, 2, "volume")?;
            f64col(ind::emv(&high, &low, &volume, arg_usize(args, 0, Some(14))?))
        }
        ("mass_index", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::mass_index(&high, &low, arg_usize(args, 0, Some(25))?))
        }
        ("efi", _) => {
            let close = series_f64(df, series, 0, "close")?;
            let volume = series_f64(df, series, 1, "volume")?;
            f64col(ind::efi(&close, &volume, arg_usize(args, 0, Some(13))?))
        }
        ("tsi", _) => f64col(ind::tsi(
            &close(0)?,
            arg_usize(args, 0, Some(25))?,
            arg_usize(args, 1, Some(13))?,
        )),
        ("crsi", _) => f64col(ind::crsi(
            &close(0)?,
            arg_usize(args, 0, Some(3))?,
            arg_usize(args, 1, Some(2))?,
            arg_usize(args, 2, Some(100))?,
        )),
        ("trix", _) => f64col(ind::trix(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("aroon", Some(dir @ ("up" | "down"))) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let period = arg_usize(args, 0, Some(14))?;
            let v = if dir == "up" {
                ind::aroon_up(&high, &low, period)
            } else {
                ind::aroon_down(&high, &low, period)
            };
            f64col(v)
        }
        ("aroonosc", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::aroonosc(&high, &low, arg_usize(args, 0, Some(14))?))
        }

        ("sum", _) => f64col(ind::sum(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("maxindex", _) => f64col(ind::maxindex(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("minindex", _) => f64col(ind::minindex(&close(0)?, arg_usize(args, 0, Some(30))?)),
        // minmax / minmaxindex are the (min, max) value / index pair over the window;
        // their outputs are exactly llv/hhv (values) and minindex/maxindex (indices).
        ("minmax", Some("min")) => f64col(ind::llv(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("minmax", Some("max")) => f64col(ind::hhv(&close(0)?, arg_usize(args, 0, Some(30))?)),
        ("minmaxindex", Some("min")) => {
            f64col(ind::minindex(&close(0)?, arg_usize(args, 0, Some(30))?))
        }
        ("minmaxindex", Some("max")) => {
            f64col(ind::maxindex(&close(0)?, arg_usize(args, 0, Some(30))?))
        }
        ("natr", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::natr(
                &high,
                &low,
                &close,
                arg_usize(args, 0, Some(14))?,
            ))
        }
        ("bop", _) => {
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            let close = series_f64(df, series, 3, "close")?;
            f64col(ind::bop(&open, &high, &low, &close))
        }

        ("linearreg", _) => f64col(ind::linearreg(&close(0)?, arg_usize(args, 0, Some(14))?)),
        ("linearreg_slope", _) => f64col(ind::linearreg_slope(
            &close(0)?,
            arg_usize(args, 0, Some(14))?,
        )),
        ("linearreg_intercept", _) => f64col(ind::linearreg_intercept(
            &close(0)?,
            arg_usize(args, 0, Some(14))?,
        )),
        ("linearreg_angle", _) => f64col(ind::linearreg_angle(
            &close(0)?,
            arg_usize(args, 0, Some(14))?,
        )),
        ("tsf", _) => f64col(ind::tsf(&close(0)?, arg_usize(args, 0, Some(14))?)),
        // beta/correl relate two series: the first defaults to close, the second is required.
        ("correl", _) => {
            let x = close(0)?;
            let y = series_f64_required(df, series, 1)?;
            f64col(ind::correl(&x, &y, arg_usize(args, 0, Some(30))?))
        }
        ("beta", _) => {
            let x = close(0)?;
            let y = series_f64_required(df, series, 1)?;
            f64col(ind::beta(&x, &y, arg_usize(args, 0, Some(5))?))
        }

        ("var", _) => f64col(ind::var(&close(0)?, arg_usize(args, 0, Some(5))?)),
        ("stddev", _) => f64col(ind::stddev(
            &close(0)?,
            arg_usize(args, 0, Some(5))?,
            arg_f64(args, 1, 1.0)?,
        )),

        ("obv", _) => {
            let real = series_f64(df, series, 0, "close")?;
            let volume = series_f64(df, series, 1, "volume")?;
            f64col(ind::obv(&real, &volume))
        }
        ("ad", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let volume = series_f64(df, series, 3, "volume")?;
            f64col(ind::ad(&high, &low, &close, &volume))
        }
        ("adosc", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            let volume = series_f64(df, series, 3, "volume")?;
            f64col(ind::adosc(
                &high,
                &low,
                &close,
                &volume,
                arg_usize(args, 0, Some(3))?,
                arg_usize(args, 1, Some(10))?,
            ))
        }

        ("avgprice", _) => {
            let open = series_f64(df, series, 0, "open")?;
            let high = series_f64(df, series, 1, "high")?;
            let low = series_f64(df, series, 2, "low")?;
            let close = series_f64(df, series, 3, "close")?;
            f64col(ind::avgprice(&open, &high, &low, &close))
        }
        ("medprice", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            f64col(ind::medprice(&high, &low))
        }
        ("typprice", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::typprice(&high, &low, &close))
        }
        ("wclprice", _) => {
            let high = series_f64(df, series, 0, "high")?;
            let low = series_f64(df, series, 1, "low")?;
            let close = series_f64(df, series, 2, "close")?;
            f64col(ind::wclprice(&high, &low, &close))
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
            arg_f64(args, 0, 0.5)?,
            arg_f64(args, 1, 0.05)?,
            false,
        )),
        ("mama", Some("fama")) => f64col(ind::mama_line(
            &close(0)?,
            arg_f64(args, 0, 0.5)?,
            arg_f64(args, 1, 0.05)?,
            true,
        )),

        (other, _) => Err(VolasError::Value(format!("unknown command '{other}'"))), // LCOV_EXCL_LINE
    }
}
