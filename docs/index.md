# volas documentation

volas is a Rust-backed, pandas-shaped DataFrame for stock OHLCV pipelines,
with 254 built-in indicators and incremental refresh after appending new bars.

Start with the [project README](https://github.com/kaelzhang/volas#readme) for
installation and the quickstart. Use these guides for the API reference, live
cumulation and time frames, interop, migration, indicator directives, benchmark
methodology, and supported pandas behavior.

| Goal | Read |
| --- | --- |
| Look up the `DataFrame` / `Series` / `Row` API and the top-level functions | [DataFrame API](dataframe-api.md) |
| Cumulate bars, stream a live time frame, handle timezones | [Cumulation, time frames and timezones](cumulation.md) |
| Write indicator directives and handle directive errors | [Directives](directives.md) |
| Index, write and handle missing values the pandas way | [pandas-compatible API](pandas-compatibility.md) |
| Exchange data with pandas, Arrow and DLPack | [Interop](interop.md) |
| Check benchmark claims and methodology | [Benchmark FAQ](benchmark-faq.md) |
| Move TA-Lib calls into volas directives | [TA-Lib migration](migration-talib.md) |
| Move a pandas OHLCV pipeline into volas | [pandas migration](migration-pandas.md) |
| Find output names for multi-output indicators | [Directive cheat sheet](directive-cheatsheet.md) |
| See how indicators treat missing and infinite inputs and keep window sums exact | [Missing values and exact window sums](indicator-inputs.md) |
| Decide whether volas is the wrong tool | [When not to use volas](when-not-to-use.md) |

The complete indicator index and pandas-subset contract remain in
[INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md) and
[PANDAS-DIFFERENCES.md](https://github.com/kaelzhang/volas/blob/main/PANDAS-DIFFERENCES.md).

```{toctree}
:hidden:
:maxdepth: 2

dataframe-api
cumulation
directives
pandas-compatibility
interop
benchmark-faq
migration-talib
migration-pandas
directive-cheatsheet
indicator-inputs
when-not-to-use
```
