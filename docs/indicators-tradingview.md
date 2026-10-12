# TradingView-compatible directives

The full indicator index is [INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md); the input rules every indicator follows are [Missing values and exact window sums](indicator-inputs.md).

These directives mirror TradingView **Pine Script** `ta.*` indicators that have
no TA-Lib equivalent. Each is validated against the canonical Pine formula
(and, where the convention agrees, against `pandas-ta`). Argument notation is
the same as the [TA-Lib-compatible](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md#ta-lib-compatible-directives) table:
`<name=value>` is an optional argument with its default; a `<name>` **without
an equals sign is required**.

## <a id="indicator-vwma"></a>`vwma`, Volume-Weighted Moving Average

```
vwma:<period>@<series=close>,<series_volume=volume>
```

`Σ(close·volume, period) / Σ(volume, period)` over each trailing window — a
moving average that weights each bar by its traded volume. A window whose
volume sums to zero yields `NA`.

- **period** `int` (required) The window length, `>= 1`.
- **series?** `str='close'` The price series.
- **series_volume?** `str='volume'` The volume series.

```py
df['vwma:20']              # 20-bar volume-weighted MA of close
df['vwma:10@(ma:5),volume'] # price = a nested directive, explicit volume
```

## <a id="indicator-alma"></a>`alma`, Arnaud Legoux Moving Average

```
alma:<period>,<offset=0.85>,<sigma=6>@<series=close>
```

A Gaussian-weighted window MA: weight `wᵢ = exp(-(i-m)² / 2s²)` with
`m = offset·(period-1)` and `s = period/sigma`, normalized over the window.
`offset` slides the Gaussian peak toward the most recent bar (1.0) or the
oldest (0.0); `sigma` controls its width.

- **period** `int` (required) The window length, `>= 1`.
- **offset?** `float=0.85` Peak position, in `[0, 1]`.
- **sigma?** `float=6` Gaussian width, `> 0`.

```py
df['alma:20']              # default ALMA (offset 0.85, sigma 6)
df['alma:9,0.5,3']         # centered, narrower
```

## <a id="indicator-hma"></a>`hma`, Hull Moving Average

```
hma:<period>@<series=close>
```

`WMA(2·WMA(close, period/2) − WMA(close, period), round(√period))` — a
low-lag moving average. Lookback is `period + round(√period) − 2`.

- **period** `int` (required) The window length, `>= 1`.
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['hma:20']
```

## <a id="indicator-swma"></a>`swma`, Symmetrically-Weighted Moving Average

```
swma@<series=close>
```

Fixed 4-bar weighted average with weights `[1/6, 2/6, 2/6, 1/6]` (oldest →
newest). No period argument.

- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['swma']
df['swma@(ema:5)']         # on a nested directive
```

## <a id="indicator-cog"></a>`cog`, Center of Gravity

```
cog:<period>@<series=close>
```

John Ehlers' Center of Gravity oscillator:
`-Σ((1+i)·source[i]) / Σ(source[i])` over the trailing `period` (the most
recent bar weighted 1, the oldest weighted `period`).

- **period** `int` (required) `>= 2`.
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['cog:10']
```

## <a id="indicator-dev"></a>`dev`, Mean Absolute Deviation

```
dev:<period>@<series=close>
```

The average absolute deviation about the window mean,
`mean(|source − mean(source)|)` — the dispersion measure CCI is built on.

- **period** `int` (required) `>= 2`.
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['dev:20']
```

## <a id="indicator-rci"></a>`rci`, Rank Correlation Index

```
rci:<period>@<series=close>
```

Spearman's rank correlation between the source and the bar index over `period`
bars, scaled to `[-100, 100]` — how monotonically (directionally consistently)
price is moving. `+100` is a perfectly rising window, `-100` a perfectly falling
one.

- **period** `int` (required) `>= 2`.
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['rci:9']
df['rci:9 > 80']           # strong up-trend signal
```

## <a id="indicator-iii"></a>`iii`, Intraday Intensity Index

```
iii@<series_high=high>,<series_low=low>,<series_close=close>,<series_volume=volume>
```

David Bostian's Intraday Intensity Index, a per-bar volume-pressure measure:
`((2·close − high − low) / (high − low)) · volume`. A zero-range bar
(`high == low`) yields `0`. No window parameter.

- **series_high? / series_low? / series_close? / series_volume?** the OHLCV columns.

```py
df['iii']
```

## <a id="indicator-kcw"></a>`kcw`, Keltner Channel Width

```
kcw:<ema_period=20>,<atr_period=10>,<mult=2>@<series_high=high>,<series_low=low>,<series_close=close>
```

The width of volas's Keltner Channel, normalized by the basis:
`(upper − lower) / middle = 2·mult·ATR(atr_period) / EMA(ema_period)`. It always
equals `(df['keltner.upper'] - df['keltner.lower']) / df['keltner']` for the same
parameters. (TradingView's `ta.kcw` uses an EMA-of-range basis instead of ATR —
a documented divergence that keeps volas's Keltner family internally consistent.)

- **ema_period?** `int=20` The EMA basis period.
- **atr_period?** `int=10` The ATR period.
- **mult?** `float=2` The channel multiplier.

```py
df['kcw']
df['kcw:20,10,2']
```

## <a id="indicator-mode"></a>`mode`, Rolling Mode

```
mode:<period>@<series=close>
```

The most frequent value in the trailing `period` window; on a tie, the smallest
value. Missing cells are ignored. Most useful on a discretized series (raw
floating-point prices rarely repeat exactly).

- **period** `int` (required) `>= 2`.
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['mode:20']
```

## <a id="indicator-pivothigh"></a><a id="indicator-pivotlow"></a>`pivothigh` / `pivotlow`, Fractal Pivots

```
pivothigh:<leftbars>,<rightbars>@<series=high>
pivotlow:<leftbars>,<rightbars>@<series=low>
```

Fractal swing-point detection. A bar is a **pivot high** when its value is the
strict maximum of the window `[bar-leftbars, bar+rightbars]` (every neighbour
strictly lower — a tie disqualifies it); `pivotlow` is the symmetric strict
minimum. The pivot's value is emitted at the **confirmation bar**
`pivot + rightbars` (it needs `rightbars` of future data to confirm), and `NA`
everywhere else — so the output is a sparse series of confirmed swing points.

> **This is a non-causal (look-ahead) indicator**: the value at a bar describes
> a pivot `rightbars` bars in the past. It is correct for **labeling / research**
> (where reading future bars is intended), but a live trading signal must never
> treat the emitted bar as the moment the pivot occurred.

- **leftbars** `int` (required) Bars of history required to the left, `>= 1`.
- **rightbars** `int` (required) Confirmation bars to the right, `>= 1`.
- **series?** `str='high'` (pivothigh) / `'low'` (pivotlow) — the series to scan.

```py
df['pivothigh:5,5']        # 5-left / 5-right swing highs of `high`
df['pivotlow:2,8']         # asymmetric swing lows of `low`
df['pivothigh:3,3@close']  # swing highs of close
```
