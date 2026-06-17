# volas

English | [简体中文](https://github.com/kaelzhang/volas/blob/main/crates/volas/README.zh-CN.md)

A Rust-backed, OHLCV-shaped `DataFrame` for candlestick / market time-series, with a
technical-indicator **directive** engine.

volas is intentionally narrow — not a general-purpose DataFrame. It targets live OHLCV
pipelines: compute an indicator by naming its *directive* (`"ma:20"`, `"macd.signal"`,
`"close > open"`), and resample / cumulate bars to a coarser time frame.

```toml
[dependencies]
volas = "1"
```

> There is also a Python package named `volas` (these same Rust kernels via PyO3) on
> PyPI. It is a separate distribution from this crate; the directive vocabulary is
> identical across both.

## Quick start

```rust
use volas::{Column, DataFrame};
use volas::directive::{execute, parse};

// Build a frame from named columns (here just `close`).
let df = DataFrame::new(
    vec!["close".to_string()],
    vec![Column::f64(vec![1.0, 2.0, 3.0, 4.0])],
    None,
).unwrap();

// Compute an indicator by its directive string. `ma:2` is the 2-period SMA.
let directive = parse("ma:2").unwrap();
let ma = execute(&df, &directive).unwrap();

assert!(!ma.is_valid(0));             // the warm-up row is NA (no value yet)
assert_eq!(ma.to_f64_vec()[3], 3.5);  // (3.0 + 4.0) / 2
```

## The data model

A `DataFrame` is a set of equal-length, named, typed `Column`s over a shared row
`Index`. A `Column` is a contiguous typed buffer plus a validity mask (so any cell
can be `NA` independently of its physical value).

```rust
use volas::{Column, DataFrame, DType};

let df = DataFrame::new(
    vec!["open".to_string(), "close".to_string()],
    vec![
        Column::f64(vec![10.0, 11.0, 12.0]),
        Column::f64(vec![10.5, 10.5, 13.0]),
    ],
    None, // None => a default 0..n RangeIndex
).unwrap();

assert_eq!(df.height(), 3);
assert_eq!(df.width(), 2);
assert_eq!(df.names(), &["open".to_string(), "close".to_string()]);

let close = df.column("close").unwrap();   // -> &Column
assert_eq!(close.dtype(), DType::F64);
assert_eq!(close.to_f64_vec()[2], 13.0);    // owned Vec<f64>
assert_eq!(close.as_f64(), Some(&[10.5, 10.5, 13.0][..])); // borrowed slice (F64 only)
```

Column constructors: `Column::f64`, `Column::i64`, `Column::bool`,
`Column::str`. Read values with `Column::to_f64_vec` (owned), `Column::as_f64`
(borrowed, `None` for non-`f64`), `Column::is_valid` (per-cell NA check),
`Column::len`, and `Column::dtype`.

## Indicators via directives

The directive engine is the primary way to compute indicators: `parse` a directive
string into an `Ast`, then `execute` it against a `DataFrame` to get a `Column`. This
covers **every** built-in indicator uniformly — see the full reference of all
directives in
[INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md).

```rust
use volas::{Column, DataFrame};
use volas::directive::{execute, parse, stringify};

let n = 40;
let df = DataFrame::new(
    vec!["open".to_string(), "close".to_string()],
    vec![
        Column::f64((1..=n).map(|x| x as f64).collect()),
        Column::f64((1..=n).map(|x| x as f64 + 0.5).collect()),
    ],
    None,
).unwrap();

// Single-output directive -> one Column.
let rsi = execute(&df, &parse("rsi:14").unwrap()).unwrap();
assert_eq!(rsi.len(), 40);

// Multi-output indicator: address each line by its sub-command.
let macd   = execute(&df, &parse("macd").unwrap()).unwrap();
let signal = execute(&df, &parse("macd.signal").unwrap()).unwrap();
let hist   = execute(&df, &parse("macd.histogram").unwrap()).unwrap();
assert_eq!((macd.len(), signal.len(), hist.len()), (40, 40, 40));

// A comparison directive -> a Bool column (usable as a row mask).
let bullish = execute(&df, &parse("close > open").unwrap()).unwrap();
assert_eq!(bullish.len(), 40);

// `@`-suffix picks input columns; canonical form drops a default input.
let ast = parse("ma:20@close").unwrap();
assert_eq!(stringify(&ast), "ma:20");
```

A directive has the shape `name:arg0,arg1,...@col0,col1,...` — `name` is the indicator,
`:` arguments are its parameters, and `@` columns are the inputs (each with a sensible
default, e.g. `close`). Multi-output indicators expose each line as `name.subcommand`
(`macd.signal`, `boll.upper`, `kdj.k`). The exact arguments, defaults, and sub-commands
for all directives are in
[INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md).

Warm-up length (rows before the first valid value) is available without computing:

```rust
use volas::directive::{lookback, parse};

let ast = parse("ma:20").unwrap();
assert_eq!(lookback(&ast), 19); // a 20-period SMA has 19 warm-up rows
```

### Raw kernels (advanced)

The directive engine is the recommended API. For direct, `DataFrame`-free access, the
`compute` module exposes the numeric kernels as pure functions over slices
(`&[f64] -> Vec<f64>`); most users should prefer directives, which are stable, complete,
and identical to the Python surface.

## Reading CSV

```rust,no_run
use volas::{read_csv, ReadCsvOptions};

// Defaults: comma-separated, the first row is the header, standard NA tokens.
let df = read_csv("ohlcv.csv", &ReadCsvOptions::default()).unwrap();
println!("{} rows x {} cols", df.height(), df.width());
```

`path` is any `AsRef<Path>` (`&str`, `String`, `PathBuf`). Tune parsing via
`ReadCsvOptions` (`delimiter`, `has_header`, `na_values`, `keep_default_na`).

## Time-frame cumulation (OHLCV resampling)

Aggregate finer bars into a coarser `TimeFrame` (the source must have a
`DatetimeIndex`). The default OHLCV spec is `open`=first, `high`=max, `low`=min,
`close`=last, `volume`=sum.

```rust,no_run
use volas::{Column, DataFrame, TimeFrame};
use volas::time::{cumulate, AggSpec};

// In practice `df` has a 1-minute DatetimeIndex; this is a placeholder frame.
let df = DataFrame::new(vec!["close".to_string()], vec![Column::f64(vec![1.0])], None).unwrap();

// Resample to 5-minute bars.
let five_min = cumulate(&df, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap();
let _ = five_min;
```

`TimeFrame` is an enum (`Min1`, `Min5`, `Hour1`, `Day1`, …) and also parses a label via
`TimeFrame::from_label("5m")`.

## Error handling

Fallible functions return `Result<_, VolasError>` (a single error enum) — no panics on
bad input. An unknown or malformed directive is reported at `parse`:

```rust
use volas::VolasError;
use volas::directive::parse;

let ok = parse("ma:2");
assert!(ok.is_ok());

let bad: Result<_, VolasError> = parse("definitely_not_an_indicator:5");
assert!(bad.is_err());
```

## Crate layout

- top level — the data model (`DataFrame`, `Series`, `Column`, `Index`,
  `DType`, `Scalar`, `Tz`, `Result`, `VolasError`), plus `read_csv` and
  `TimeFrame`;
- `directive` — `parse` / `execute` / `stringify` / `lookback`, and the `Ast`;
- `compute` — numeric kernels and technical indicators (pure functions);
- `time` — time-frame cumulation (OHLCV resampling);
- `core` — the full `volas-core` surface, for the less-common types.

This crate is a thin facade re-exporting the volas workspace (`volas-core`,
`volas-compute`, `volas-directive`, `volas-time`, `volas-io`) behind one dependency.

## License

[MIT](https://github.com/kaelzhang/volas/blob/main/LICENSE)
