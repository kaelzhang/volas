# When not to use volas

volas is narrow by design. It is built for live OHLCV pipelines where technical
indicator columns are cached on the frame and refreshed after new bars arrive.

Use another tool when that is not your problem.

## Keep pandas or polars for general dataframe work

Use pandas or polars for joins, group-bys, pivots, arbitrary reshaping,
multi-index analysis, heterogeneous business tables, exploratory notebooks, and
general analytics. volas intentionally does not implement the full pandas API.

## Keep TA-Lib for simple batch indicator calls

If you only need one-off array calculations such as:

```py
talib.RSI(close, timeperiod=14)
```

and you do not need an OHLCV frame, cached directive columns, or append refresh,
TA-Lib remains a mature and simple choice.

## Avoid volas for non-OHLCV schemas

volas works best when the data is naturally shaped around `open`, `high`, `low`,
`close`, `volume`, time indexes, and indicator features. If your data has no
bar-like structure, the directive engine and append semantics may not help.

## Avoid silent conversion expectations

volas has no `object` dtype and does not silently widen columns through lossy
implicit conversions. If your workflow depends on pandas accepting mixed Python
objects in a column, use pandas for that stage and cross into volas only for the
OHLCV indicator segment.

See [PANDAS-DIFFERENCES.md](../PANDAS-DIFFERENCES.md) for the detailed contract.
