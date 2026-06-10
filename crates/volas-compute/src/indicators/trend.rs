//! Trend-following indicators — moving averages (SMA/EMA/SMMA/WMA/BBI), the
//! cascaded EMA MAs (DEMA/TEMA/TRIMA/T3), KAMA, Parabolic SAR / SAREXT, and MACD
//! — plus true-range volatility (TR/ATR/NATR). Each family lives in its own
//! cohesive submodule; everything is re-exported flat so the public path stays
//! `indicators::<name>` (and `indicators::trend::<name>`).

mod cascade;
mod kama;
mod ma;
mod macd;
mod sar;
mod volatility;

pub use cascade::*;
pub use kama::*;
pub use ma::*;
pub use macd::*;
pub use sar::*;
pub use volatility::*;

#[cfg(test)]
mod tests;
