# volas documentation

volas is a Rust-backed, pandas-shaped DataFrame for stock OHLCV pipelines,
with 254 built-in indicators and incremental refresh after appending new bars.

Start with the [project README](https://github.com/kaelzhang/volas#readme) for
installation and the quickstart. Use these guides for migration, indicator
directives, benchmark methodology, and supported pandas behavior.

| Goal | Read |
| --- | --- |
| Check benchmark claims and methodology | [Benchmark FAQ](benchmark-faq.md) |
| Move TA-Lib calls into volas directives | [TA-Lib migration](migration-talib.md) |
| Move a pandas OHLCV pipeline into volas | [pandas migration](migration-pandas.md) |
| Find output names for multi-output indicators | [Directive cheat sheet](directive-cheatsheet.md) |
| Decide whether volas is the wrong tool | [When not to use volas](when-not-to-use.md) |

The complete indicator index and pandas-subset contract remain in
[INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md) and
[PANDAS-DIFFERENCES.md](https://github.com/kaelzhang/volas/blob/main/PANDAS-DIFFERENCES.md).

```{toctree}
:hidden:
:maxdepth: 2

benchmark-faq
migration-talib
migration-pandas
directive-cheatsheet
when-not-to-use
```
