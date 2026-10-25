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

> There is also a Python package named `volas` — the same Rust kernels exposed via
> PyO3. See the [volas Python project on GitHub](https://github.com/kaelzhang/volas)
> and [`volas` on PyPI](https://pypi.org/project/volas/). It is a separate
> distribution from this crate; the directive vocabulary is identical across both.

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
`Index`. A `Column` is a contiguous typed buffer; any cell can be `NA` (missing)
independently of its value — see [Missing values](#missing-values-na) below.

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

`Column` storage and `Series` structure are opaque. Inspect a column with typed
slice accessors or its borrowed `ColumnView`; constructing that view cannot create
a `Column`. `Series::new` checks value / index lengths in every build, while
`name()`, `data()` and `index()` borrow its parts. Checked `Series::scatter`
writes selected values while preserving the structure and labels.

Explicit index labels are built with `Index::new(name, IndexKind::...)`. Datetime
labels must strictly increase and contain no NaT; integer and string labels must
be unique (their order is unrestricted). `DataFrame::new` validates its index too.
`Index::take`, `extend` and frame row selection return an error if the resulting
labels violate these rules. Sorting a datetime index backwards therefore fails.

```rust
use std::sync::Arc;
use volas::{Column, Index, IndexKind, Series, Tz};

let index = Index::new(Some("time".into()), IndexKind::Datetime(vec![1, 2].into(), Tz::Utc)).unwrap();
let series = Series::new(Some("close".into()), Column::f64(vec![10.0, 11.0]), Arc::new(index)).unwrap();
assert_eq!(series.name(), Some("close"));
assert_eq!(series.data().len(), series.index().len());
assert!(series.index().take(&[1, 0]).is_err());
assert!(Index::new(None, IndexKind::Int64(vec![1, 1].into())).is_err());
```

Rust writes validate the complete selection before publication: `Column::scatter`
checks positions and value counts, and `DataFrame::append` checks schema, values
and one index admission plan before committing. `OpenPeriod::push` uses that same
checked append. Reduced rows use `check_folded_row(...).land()`; raw append and
fold publication helpers are private. `Series::from_column` and
`DataFrame::from_columns` build checked positional indexes from the admitted shape.
NumPy and DLPack exports dispatch the storage type once inside the Column owner.
Reduction admissions retain an immutable positions slice or an owned range.
`RowGroups::new` and `RowGroups::from_offsets` validate group counts, nonempty
groups and every position once; their immutable proof can reduce any column of
the admitted height via `reduce` or `reduce_with_spare`. Arbitrary iterators
cannot supply either admission proof.

Cell folds use `Column::check_fold_cell` or `check_reduced_cell`; a reduction
into an existing cell uses `check_reduction`. Each returns a target-bound plan
whose `land()` checks and publishes the operation. Raw fold commits are private.

```rust
use volas::Column;
use volas::time::Agg;
let input = Column::i64(vec![2, 3]);
let mut sum = Column::i64(vec![0]);
sum.check_reduction(0, Agg::Sum.as_combine_op(), &input, 0..2).unwrap().land().unwrap();
assert_eq!(sum.as_i64(), Some(&[5][..]));
```

### Missing values (NA)

A cell can be missing independently of its value. Float columns use `NaN` as the NA
marker; integer / bool / string columns carry an explicit validity mask. Build them
from optional cells (`Column::i64_from_options` / `bool_from_options` /
`str_from_options`), or from values plus a mask with the checked `Column::i64_with`
/ `i32_with` / `bool_with` / `str_with` (a mask of another length is an error, as is
any frame column whose mask does not cover it). Check missingness per cell with
`is_valid`, and count it with `null_count`.

```rust
use volas::Column;
use volas::Validity;

// Float columns: NaN is the missing marker.
let prices = Column::f64(vec![1.0, f64::NAN, 3.0]);
assert_eq!(prices.null_count(), 1);
assert!(prices.is_valid(0));
assert!(!prices.is_valid(1));         // cell 1 is NA
assert!(prices.get_f64(1).is_nan());  // reading an NA f64 yields NaN

// Integer / bool / string columns carry a validity mask instead of a sentinel.
let volume = Column::i64_from_options([Some(100), None, Some(300)]); // cell 1 is NA
assert_eq!(volume.null_count(), 1);
assert!(!volume.is_valid(1));

// An explicit mask must cover exactly the values, else a shape error.
let checked = Column::i64_with(vec![100, 0, 300], Validity::from_valid_iter(3, [true, false, true]));
assert_eq!(checked.unwrap(), volume);
assert!(Column::i64_with(vec![1], Validity::from_valid_iter(3, [true, false, true])).is_err());
```

Indicator warm-up rows are NA too — e.g. a 2-period SMA has one NA row at the head:

```rust
use volas::{Column, DataFrame};
use volas::directive::{execute, parse};

let df = DataFrame::new(
    vec!["close".to_string()],
    vec![Column::f64(vec![1.0, 2.0, 3.0])],
    None,
).unwrap();
let ma = execute(&df, &parse("ma:2").unwrap()).unwrap();

assert!(!ma.is_valid(0));        // the warm-up row is NA
assert_eq!(ma.null_count(), 1);
```

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
assert_eq!(stringify(&*ast), "ma:20");
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
and identical to the Python surface. The kernels follow the directive engine's
missing-row rule: a row where any input of a recursive indicator is NaN outputs NaN,
and the recursion continues from the last complete row as if that row did not exist
(see [Missing and non-finite inputs](https://github.com/kaelzhang/volas/blob/main/docs/indicator-inputs.md#missing-and-non-finite-inputs)).

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

For incremental Rust input, `volas::time::Cumulator::append` retains the open
period's fine bars. `frame(&self)` returns an independent snapshot while the
cumulator remains usable. When finished, `into_frame(self)` consumes the
cumulator and transfers its output buffers. `into_parts(self)` additionally
returns the open period and its running aggregate for live continuation.
The running aggregate admits one bar through `check_push(bar, resent).land()`.
`check_restart(bar, &spec)` reduces a later period's first bar once; its `current()`
can be appended to the output before `land()` publishes that restart.


`Cumulator::from_frame(&fine, tf, spec)` constructs and ingests an initial batch
with output capacity for consuming materialization and the first live append.
It validates like `new` followed by `append`, including deferred open-period errors.
Ordinary `new` / `append` builders retain their existing snapshot capacity budget.

For raw feeds that contain consecutive retransmissions, build a
`volas::time::RawFineInput` with `new(names, columns, timestamps, tz, index_name)`.
This input checks non-decreasing, NaT-free timestamps and keeps the last row in
each consecutive equal-timestamp group before creating a checked frame.
`Cumulator::from_raw(&input, tf, spec)` and `append_raw(&input)` ingest it.
The retransmission exception belongs to this raw cumulation entry; ordinary
`Index` and `DataFrame` construction always enforce unique labels.


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


This crate exposes a selected, validated interface from the volas workspace behind
one dependency. Low-level core storage types are not exported by the facade.

## License

[MIT](https://github.com/kaelzhang/volas/blob/main/LICENSE)

Raw missing-placeholder growth is internal to admitted frame writes; public column extension uses `Column::append` with a typed missing column.
