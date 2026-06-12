# Developer Guide

## Design notes & non-goals

- **Not a general-purpose DataFrame.** volas models exactly what OHLCV
  quant workflows need; it deliberately omits multi-level indexes,
  heterogeneous per-cell storage, joins, and general reshaping.
- **pandas-independent at runtime.** pandas and TA-Lib are used only as
  test oracles and benchmark comparators, never imported at runtime.
- **External API cleanliness first.** The Python surface is kept clean
  and pandas-shaped; internal layering is secondary to per-bar latency.

## Development

Requires Python >= 3.11 and a Rust toolchain.

```sh
make install        # Rust toolchain + maturin + Python dev deps
make build          # build the Rust extension, install the package in-place
make test           # run the Python test suite
make coverage       # true cargo-test union pytest line coverage
make benchmark      # multi-library benchmark
make build-pkg      # build a release wheel + sdist into dist/
```

`make coverage` delegates to `scripts/coverage.sh`. `make benchmark`
compares pandas, stock-pandas, polars, TA-Lib, and volas where those
benchmark-only dependencies are installed.

### Dependency groups

- **`dev`** (`pip install -e .[dev]`) — everything the test suite needs;
  this is all CI installs. It includes pandas because the parity tests
  use it as an oracle. pandas is test-time only; volas has no pandas
  runtime dependency.
- **`benchmark`** (`pip install -e .[benchmark]`) — extra comparison
  libraries used only by the benchmark. `make benchmark` installs
  `.[dev,benchmark]`; a library that is only needed to benchmark, never
  to test, belongs here so CI test runs stay lean.

### Benchmark & web report

`make benchmark` times every candidate on batch indicator computation,
the incremental append-one-bar path, and the full volas-vs-TA-Lib
coverage rows. To optimize one indicator, pass `INDICATOR=<directive>`;
that scoped run prints only that indicator's coverage rows and never
writes the web report:

```sh
make benchmark INDICATOR=roc:10
make benchmark                  # full run; always writes ./benchmark-report.html
```

The locally generated `./benchmark-report.html` (and the always-current
published copy at <https://kaelzhang.github.io/volas/>, deployed by the
`pages` workflow) keeps the append and
batch sections as charts, then summarizes full coverage as one row per
TA-Lib indicator. Extra length fixtures and cached append-refresh
comparisons appear as additional `volas vs TA-Lib` columns instead of
duplicate indicator rows.
