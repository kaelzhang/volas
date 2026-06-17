# volas

A Rust-backed, OHLCV-shaped `DataFrame` for candlestick / market time-series, with
a technical-indicator **directive** engine.

volas is intentionally narrow — not a general-purpose DataFrame. It targets live
OHLCV pipelines: append a new bar, keep indicator columns cached, and recompute only
the stale tail (`O(lookback + new rows)`, not `O(n)`).

This crate is the umbrella that re-exports the volas workspace (`volas-core`,
`volas-compute`, `volas-directive`, `volas-time`, `volas-io`) behind one dependency.

```toml
[dependencies]
volas = "1"
```

```rust
use volas::{Column, DataFrame};
use volas::directive::{execute, parse};

let df = DataFrame::new(
    vec!["close".to_string()],
    vec![Column::f64(vec![1.0, 2.0, 3.0, 4.0])],
    None,
)?;

// `ma:2` is a 2-period simple moving average over `close`.
let directive = parse("ma:2")?;
let ma = execute(&df, &directive)?;
assert_eq!(ma.to_f64_vec()[3], 3.5); // (3.0 + 4.0) / 2
# Ok::<(), volas::VolasError>(())
```

## Layout

- top level — the data model (`DataFrame`, `Series`, `Column`, `Index`, `DType`,
  `Scalar`, `Tz`, `Result`, `VolasError`), plus `read_csv` and `TimeFrame`;
- `volas::directive` — parse a directive string, then `execute` it;
- `volas::compute` — numeric kernels and technical indicators (pure functions);
- `volas::time` — time-frame cumulation (OHLCV resampling);
- `volas::core` — the full `volas-core` surface.

There is also a Python package named `volas` (Rust kernels via PyO3) on PyPI; it is a
separate distribution from this crate.

## License

[MIT](../../LICENSE)
