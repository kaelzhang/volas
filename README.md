# volas

> High-performance, Rust-backed columnar kernel for stock / candlestick (OHLCV) time-series data.

**Status:** early development — version `0.0.0`. APIs are not yet stable and may change at any time.

## What is volas?

`volas` is a focused, DataFrame-like data structure purpose-built for financial
candlestick (OHLCV) time series and quantitative-trading workflows.

Rather than being a general-purpose DataFrame, `volas` supports only what this
domain needs — a time-indexed, two-dimensional table of numeric and boolean
columns — and implements its storage and compute core in **Rust** for low,
predictable latency on incremental ("append one bar") workloads.

### Goals

- **Live-trading first** — optimized for the incremental hot path: append a new
  bar, update indicators, read the result, with minimal per-bar latency.
- **Small, regular data model** — a time index plus 2-D columns of numbers and
  booleans; no multi-level indexes, no heterogeneous storage, no general-purpose
  reshaping.
- **Rust core** — storage and computation live in a compiled Rust extension.
- **First-class NumPy interop** — zero-copy `to_numpy` export, designed so data
  flows cheaply into NumPy and `torch.Tensor` for both training and inference.

## Installation

> Not yet published to PyPI. For a local build, see [Development](#development).

```sh
pip install volas
```

Requires Python >= 3.11.

## Usage

> Preview of the intended API. Not yet implemented at `0.0.0`; subject to change.

```py
from volas import Volas

v = Volas({
    'open':   [...],
    'high':   [...],
    'low':    [...],
    'close':  [...],
    'volume': [...],
})

# Zero-copy export to NumPy (where possible)
close = v['close'].to_numpy()
matrix = v.to_numpy()
```

## Development

Requires Python >= 3.11 and a Rust toolchain.

```sh
# Build the Rust extension and install the package in-place (development)
make build

# Build a release wheel + sdist into dist/
make build-pkg

# Publish to PyPI (build + upload)
make publish
```

## License

[MIT](LICENSE)
