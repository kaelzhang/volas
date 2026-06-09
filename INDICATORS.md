# Built-in Indicators

This page is the complete directive reference for `volas`.

Volas supports indicators in two groups. The first group is native to Volas or
inherits stock-pandas directive names; TA-Lib either has no equivalent or no
first-class function with the same directive name and OHLCV defaults. The second
group follows TA-Lib's function surface: directive names are lowercase,
arguments are positional, and multi-output indicators expose each line as a
sub-command such as `macd.signal`, `boll.upper`, or `ht_sine.leadsine`.

## Volas-exclusive indicators

These directives are implemented by Volas itself. Many of them follow the
stock-pandas directive vocabulary, with the examples adapted to `volas.DataFrame`.

### `smma`, Smoothed Moving Average

```
smma:<period>@<on>
```

Gets the `period`-period smoothed moving average on column or directive `on`.
`SMA` is often confused between simple moving average and smoothed moving
average, so Volas uses `ma` for simple moving average and `smma` for smoothed
moving average.

- **period** `int` (required)
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Equivalent to df['smma:5@close']
df['smma:5']

df['smma:10@open']
```

### `bbi`, Bull and Bear Index (多空指标)

```
bbi:<a>,<b>,<c>,<d>@<on>
```

Calculates BBI (Bull and Bear Index), which is the average of `ma:3`, `ma:6`,
`ma:12`, and `ma:24` by default.

- **a?** `int=3`
- **b?** `int=6`
- **c?** `int=12`
- **d?** `int=24`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Uses default parameters
df['bbi']

# Custom parameters
df['bbi:5,10,20,30@close']
```

### `bbw`, Bollinger Band Width

```
bbw:<period>@<on>
```

Gets Bollinger Band Width for a series.

- **period?** `int=20`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Bollinger band width
df['bbw']

# Equivalent definition
(df['boll.upper'] - df['boll.lower']) / df['boll']

# Or as a directive expression
df['(boll.upper - boll.lower) / boll']
```

### `rsv`, Raw Stochastic Value (未成熟随机值)

```
rsv:<period>@<high>,<low>,<close>
```

Calculates the raw stochastic value, which is often used to calculate KDJ.

- **period** `int` (required)
- **high?** `str='high'` The column name for high prices.
- **low?** `str='low'` The column name for low prices.
- **close?** `str='close'` The column name for close prices.

```py
# Uses default columns: high, low, close
df['rsv:9']

# Specify custom columns
df['rsv:9@high,low,close']
```

### `kdj`, A Variety of Stochastic Oscillator (随机指标)

KDJ is a variety of the [Stochastic Oscillator](https://en.wikipedia.org/wiki/Stochastic_oscillator)
indicator created by [Dr. George Lane](https://en.wikipedia.org/wiki/George_Lane_(technical_analyst)),
which follows the formula:

```
RSV = rsv(period_rsv)
%K = ewma(RSV, period_k, init_value)
%D = ewma(%K, period_d, init_value)
%J = 3 * %K - 2 * %D
```

The EWMA here is seeded by `init_value`. Trading software from different vendors
usually uses one of `0.0`, `50.0`, or `100.0` as the initial value; Volas defaults
to `50.0`.

```
kdj.k:<period_rsv>,<period_k>,<init_value>@<high>,<low>,<close>
kdj.d:<period_rsv>,<period_k>,<period_d>,<init_value>@<high>,<low>,<close>
kdj.j:<period_rsv>,<period_k>,<period_d>,<init_value>@<high>,<low>,<close>
```

- **period_rsv?** `int=9` The period for calculating RSV.
- **period_k?** `int=3` The period for smoothing RSV into %K.
- **period_d?** `int=3` The period for smoothing %K into %D.
- **init_value?** `float=50.0` The initial value for smoothing.
- **high?** `str='high'` The column name for high prices.
- **low?** `str='low'` The column name for low prices.
- **close?** `str='close'` The column name for close prices.

```py
# The %D series of KDJ
df['kdj.d']

# Equivalent to default parameters and columns
df['kdj.d:9,3,3,50@high,low,close']

# KDJ lines with custom periods
df[['kdj.k:9,9,50', 'kdj.d:9,9,9,50', 'kdj.j:9,9,9,50']]
```

### `llv`, Lowest of Low Values

```
llv:<period>@<on>
```

Gets the lowest value in N periods. By default, it reads the `low` column.

- **period** `int` (required)
- **on?** `str='low'` Which column or directive the calculation is based on.

```py
# The 10-period lowest low prices
df['llv:10']

# The 10-period lowest close prices
df['llv:10@close']
```

### `hhv`, Highest of High Values

```
hhv:<period>@<on>
```

Gets the highest value in N periods. By default, it reads the `high` column.

- **period** `int` (required)
- **on?** `str='high'` Which column or directive the calculation is based on.

```py
# The 10-period highest high prices
df['hhv:10']

# The 10-period highest close prices
df['hhv:10@close']
```

### `donchian`, Donchian Channels

```
donchian:<period>@<high>,<low>
donchian.upper:<period>@<high>
donchian.lower:<period>@<low>
```

Gets Donchian channels, the historical view of price volatility by charting a
security's highest and lowest prices over a set period.

- **period** `int` (required)
- **high?** `str='high'` The column to calculate highest high values.
- **low?** `str='low'` The column to calculate lowest low values.

```py
# Donchian middle channel with default columns
df['donchian:20']

# Donchian upper and lower channels
df['donchian.upper:20']
df['donchian.lower:20']

# Short aliases
df['donchian.u:20']
df['donchian.l:20']
```

### `hv`, Historical Volatility

```
hv:<period>,<time_frame>,<trading_days>@<on>
```

Gets historical volatility, the statistical measure of the dispersion of returns
for a security or index over a period of time.

- **period** `int` (required)
- **time_frame?** `str='1d'` Time frame such as `1m`, `15m`, `1h`, or `1d`.
- **trading_days?** `int=252` Trading days in a year; crypto workflows often use
  `365`.
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# 10-period historical volatility for 15-minute data based on 365 yearly days
df['hv:10,15m,365']

# Uses default time_frame and trading_days
df['hv:10']
```

### `psy`, Psychological Line (心理线)

```
psy:<period>@<on>
```

The percentage of rising days (close above the previous close) over the last
`period` bars.

- **period?** `int=12`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['psy']
df['psy:6']
```

### `dpo`, Detrended Price Oscillator

```
dpo:<period>@<on>
```

The price `period/2 + 1` bars ago minus the `period`-bar SMA, removing the trend
to expose shorter cycles.

- **period?** `int=20`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['dpo']
df['dpo:10']
```

### `tsi`, True Strength Index

```
tsi:<long>,<short>@<on>
```

A double-EMA-smoothed momentum oscillator: `100 * EMA_short(EMA_long(Δclose)) /
EMA_short(EMA_long(|Δclose|))`.

- **long?** `int=25`
- **short?** `int=13`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['tsi']
df['tsi:25,13']
```

### `kst`, Know Sure Thing

```
kst@<on>
```

Pring's momentum oscillator: a weighted sum of four SMA-smoothed rate-of-change
terms (ROC 10/15/20/30, smoothed by SMA 10/10/10/15, weighted 1/2/3/4).

- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['kst']
```

### `crsi`, Connors RSI

```
crsi:<rsi>,<streak>,<rank>@<on>
```

Connors' composite: the average of `rsi:<rsi>`, the RSI of the consecutive up /
down streak length, and the percent-rank of the 1-bar return over the last `rank`
bars.

- **rsi?** `int=3`
- **streak?** `int=2`
- **rank?** `int=100`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['crsi']
df['crsi:3,2,100']
```

### `chop`, Choppiness Index

```
chop:<period>@<high>,<low>,<close>
```

How choppy versus trending the market is over `period` bars:
`100 * log10(sum(TR) / (HHV − LLV)) / log10(period)`. Higher is choppier.

- **period?** `int=14`
- **high? / low? / close?** `str` the input columns; default to the like-named frame columns.

```py
df['chop']
df['chop:14']
```

### `cmf`, Chaikin Money Flow

```
cmf:<period>@<high>,<low>,<close>,<volume>
```

The `period`-bar sum of money-flow volume divided by the sum of volume — positive
is buying pressure, negative is selling pressure.

- **period?** `int=20`
- **high? / low? / close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['cmf']
df['cmf:20']
```

### `emv`, Ease of Movement

```
emv:<period>@<high>,<low>,<volume>
```

The `period`-bar SMA of price displacement per unit of volume (StockCharts' 1e8
volume scale) — how easily price moves.

- **period?** `int=14`
- **high? / low? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['emv']
df['emv:14']
```

### `efi`, Elder Force Index

```
efi:<period>@<close>,<volume>
```

`EMA_period(Δclose * volume)` — the force of a move, combining its direction, size,
and volume.

- **period?** `int=13`
- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['efi']
df['efi:13']
```

### `pvt`, Price Volume Trend

```
pvt@<close>,<volume>
```

A cumulative volume line weighted by each bar's return:
`PVT += (Δclose / prev close) * volume`.

- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['pvt']
```

### `nvi`, Negative Volume Index

```
nvi@<close>,<volume>
```

A cumulative line (base 1000) that compounds the return only on bars where volume
fell — tracking the "smart money" days.

- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['nvi']
```

### `pvi`, Positive Volume Index

```
pvi@<close>,<volume>
```

A cumulative line (base 1000) that compounds the return only on bars where volume
rose — tracking the "crowd" days.

- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['pvi']
```

### `mass_index`, Mass Index

```
mass_index:<period>@<high>,<low>
```

The `period`-bar sum of the 9-EMA / double-9-EMA ratio of the high−low range; a
range "bulge" can flag a coming reversal.

- **period?** `int=25`
- **high? / low?** `str` the input columns; default to the like-named frame columns.

```py
df['mass_index']
df['mass_index:25']
```

### `bias`, Bias Ratio (乖离率)

```
bias:<period>@<on>
```

The percentage deviation of the series from its `period`-bar SMA,
`(close − SMA) / SMA × 100`. This is the China-market name for `ppo:1,<period>,0`; the
classic triple is `bias:6`, `bias:12`, `bias:24`.

- **period?** `int=6`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['bias']
df['bias:24']
```

### `dma`, Difference of Moving Average (平行线差)

```
dma:<fast>,<slow>@<on>
dma.ama:<fast>,<slow>,<m>@<on>
```

The DDD line is the difference of two SMAs, `SMA_fast − SMA_slow` — the China-market name for
`apo:<fast>,<slow>,0`. The AMA signal line is the `m`-bar SMA of the DDD line. `dma.ddd` is an
alias of the main `dma` line.

- **fast?** `int=10`
- **slow?** `int=50`
- **m?** `int=10` The AMA signal period (only on `dma.ama`).
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# DDD difference line
df['dma']
# , which is equivalent to
df['dma.ddd']

# AMA signal line
df['dma.ama']
```

### `vortex`, Vortex Indicator

```
vortex.plus:<period>@<high>,<low>,<close>
vortex.minus:<period>@<high>,<low>,<close>
```

Two trend lines built from directional movement relative to the true range:
`+VI = Σₙ|high − prev low| / Σₙ TR` and `−VI = Σₙ|low − prev high| / Σₙ TR`. `+VI` above
`−VI` signals an up-trend, the reverse a down-trend, and their crossings mark turns. Vortex
has no single primary line — request `vortex.plus` (alias `.p`) or `vortex.minus` (alias `.m`).

- **period?** `int=14`
- **high? / low? / close?** `str` the input columns.

```py
df['vortex.plus']
df['vortex.minus']
```

### `brar`, BRAR Sentiment (人气意愿指标)

```
brar.ar:<period>@<open>,<high>,<low>
brar.br:<period>@<high>,<low>,<close>
```

Two China-market sentiment lines. AR (人气, popularity) = `Σₙ(H−O) / Σₙ(O−L) × 100`; BR
(意愿, willingness) = `Σₙ max(0, H−Cᵧ) / Σₙ max(0, Cᵧ−L) × 100`, where `Cᵧ` is the prior close
and the `max(0, …)` clamp follows the 通达信 convention. Request `brar.ar` or `brar.br`.

- **period?** `int=26`
- **open? / high? / low? / close?** `str` the input columns (AR uses open/high/low; BR uses high/low/close).

```py
df['brar.ar']
df['brar.br']
```

### `vr`, Volume Ratio (成交量比率)

```
vr:<period>@<close>,<volume>
```

A volume-sentiment ratio over `period` bars: `(UVS + ½·PVS) / (DVS + ½·PVS) × 100`, where
UVS / DVS / PVS sum the volume of up- / down- / flat-close days (classified vs the prior close).

- **period?** `int=26`
- **close? / volume?** `str` the input columns.

```py
df['vr']
df['vr:26']
```

### `coppock`, Coppock Curve

```
coppock:<wma>,<roc_long>,<roc_short>@<on>
```

A long-term momentum oscillator: the `wma`-period weighted MA of the sum of two
rate-of-change terms, `WMA(ROC_long + ROC_short)`. A cross above zero is the classic buy
signal.

- **wma?** `int=10`
- **roc_long?** `int=14`
- **roc_short?** `int=11`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['coppock']
df['coppock:10,14,11']
```

### `relative_vigor`, Relative Vigor Index

```
relative_vigor:<period>@<open>,<high>,<low>,<close>
relative_vigor.signal:<period>@<open>,<high>,<low>,<close>
```

An oscillator built on the idea that price closes higher than it opens in up-trends:
`SMAₙ(swma(C−O)) / SMAₙ(swma(H−L))`, where `swma` is the 4-bar symmetric weighted average
`[1, 2, 2, 1] / 6`. `relative_vigor.signal` is the `swma` of the index. (Named in full
because `RVI` also denotes the Relative Volatility Index.)

- **period?** `int=10`
- **open? / high? / low? / close?** `str` the input columns.

```py
df['relative_vigor']
df['relative_vigor.signal']
```

### `dkx`, Bull-Bear Line (多空线)

```
dkx@<open>,<high>,<low>,<close>
dkx.ma:<m>@<open>,<high>,<low>,<close>
```

The DKX line is the 20-period weighted MA of a typical price `MID = (3C + L + O + H) / 6`
(a China-market trend line); `dkx.ma` is its `m`-bar SMA signal. DKX crossing above its
signal is bullish.

- **m?** `int=10` The MADKX signal period (only on `dkx.ma`).
- **open? / high? / low? / close?** `str` the input columns.

```py
df['dkx']
df['dkx.ma']
```

### `wvad`, Williams Variable Accumulation/Distribution (威廉变异离散量)

```
wvad:<period>@<open>,<high>,<low>,<close>,<volume>
```

A volume-weighted gauge of intraday buying vs selling power:
`Σₙ( (C − O) / (H − L) × V )`. Positive means buyers dominated the open-to-close range,
negative means sellers.

- **period?** `int=24`
- **open? / high? / low? / close? / volume?** `str` the input columns.

```py
df['wvad']
df['wvad:24']
```

### `cdp`, Counter-Trend Operation (逆势操作)

```
cdp@<high>,<low>,<close>
cdp.ah@<high>,<low>,<close>
cdp.nh@<high>,<low>,<close>
cdp.nl@<high>,<low>,<close>
cdp.al@<high>,<low>,<close>
```

Five intraday levels computed from the **prior** bar: `CDP = (H + L + 2C) / 4` (the center),
with `cdp.ah = CDP + (H−L)`, `cdp.nh = 2·CDP − L`, `cdp.nl = 2·CDP − H`,
`cdp.al = CDP − (H−L)`, ordered AH > NH > CDP > NL > AL — a short-term reversal system for
range-bound days.

- **high? / low? / close?** `str` the input columns (read from the prior bar).

```py
df['cdp']        # the CDP center
df['cdp.ah']     # AH (highest)
df['cdp.al']     # AL (lowest)
```

### `mike`, MIKE Support/Resistance (麦克指标)

```
mike.weakr:<period>@<high>,<low>,<close>
mike.midr:<period>@<high>,<low>,<close>
mike.strongr:<period>@<high>,<low>,<close>
mike.weaks:<period>@<high>,<low>,<close>
mike.mids:<period>@<high>,<low>,<close>
mike.strongs:<period>@<high>,<low>,<close>
```

Six support / resistance bands around the typical price `TYP = (H+L+C)/3`, using the
`period`-bar high-of-high `HH` and low-of-low `LL`. Resistance: weak `TYP+(TYP−LL)`, mid
`TYP+(HH−LL)`, strong `2·HH−LL`; support mirrors them — weak `TYP−(HH−TYP)`, mid
`TYP−(HH−LL)`, strong `2·LL−HH`. MIKE has no single line — request one of the six.

- **period?** `int=12`
- **high? / low? / close?** `str` the input columns.

```py
df['mike.strongr']   # strong resistance
df['mike.weaks']     # weak support
```

### `keltner`, Keltner Channels

```
keltner:<ema_period>@<close>
keltner.upper:<ema_period>,<atr_period>,<mult>@<high>,<low>,<close>
keltner.lower:<ema_period>,<atr_period>,<mult>@<high>,<low>,<close>
```

An EMA with an ATR-scaled envelope (the modern convention). The middle line is
`EMA(close, ema_period)`; the bands are `middle ± mult · ATR(atr_period)`. Price riding the
upper band signals strength, the lower band weakness.

- **ema_period?** `int=20`
- **atr_period?** `int=10` (bands only)
- **mult?** `float=2.0` (bands only)
- **high? / low? / close?** `str` the input columns (the middle line uses only close).

```py
df['keltner']          # middle (EMA)
df['keltner.upper']
df['keltner.lower']
```

### `stoch_momentum`, Stochastic Momentum Index

```
stoch_momentum:<k>,<d>,<signal>@<high>,<low>,<close>
stoch_momentum.signal:<k>,<d>,<signal>@<high>,<low>,<close>
```

A refined stochastic measuring the close relative to the **midpoint** of the `k`-bar range,
double-EMA smoothed: `SMI = Ds / (Dhl/2) × 100`, where `Ds = EMA_d(EMA_d(C − (HH+LL)/2))` and
`Dhl = EMA_d(EMA_d(HH − LL))`. `stoch_momentum.signal` is the `signal`-period EMA of SMI.
(Named in full because `SMI` collides with SMI Ergodic on some platforms.)

- **k?** `int=10` (the high/low range)
- **d?** `int=3` (double-EMA smoothing)
- **signal?** `int=3` (signal EMA; only on `.signal`)
- **high? / low? / close?** `str` the input columns.

```py
df['stoch_momentum']
df['stoch_momentum.signal']
```

### `ttm_squeeze`, TTM Squeeze

```
ttm_squeeze:<period>,<bb_mult>,<kc_mult>@<high>,<low>,<close>
ttm_squeeze.on:<period>,<bb_mult>,<kc_mult>@<high>,<low>,<close>
```

John Carter's volatility-breakout indicator. `ttm_squeeze` is the momentum histogram — the
linear regression over `period` of `close − ((HH+LL)/2 + SMA(close))/2`. `ttm_squeeze.on` is
`1.0` while the market is "squeezed" (the Bollinger Bands sit inside the Keltner Channels —
low volatility, a breakout is brewing), else `0.0`.

- **period?** `int=20`
- **bb_mult?** `float=2.0` The Bollinger σ multiplier (squeeze flag only).
- **kc_mult?** `float=1.5` The Keltner range multiplier (squeeze flag only).
- **high? / low? / close?** `str` the input columns.

```py
df['ttm_squeeze']       # momentum histogram
df['ttm_squeeze.on']    # 1.0 = squeeze on
```

### `pivot_points`, Pivot Points

```
pivot_points@<high>,<low>,<close>
pivot_points.r1@<high>,<low>,<close>   (also .r2, .r3)
pivot_points.s1@<high>,<low>,<close>   (also .s2, .s3)
```

The floor-trader support / resistance levels computed from the **prior** bar: `PP = (H+L+C)/3`
(the pivot), then `r1 = 2·PP − L`, `s1 = 2·PP − H`, `r2 = PP + (H−L)`, `s2 = PP − (H−L)`,
`r3 = H + 2·(PP − L)`, `s3 = L − 2·(H − PP)`. `pivot_points.p` is an alias of the pivot.

- **high? / low? / close?** `str` the input columns (read from the prior bar).

```py
df['pivot_points']      # the pivot
df['pivot_points.r1']   # first resistance
df['pivot_points.s1']   # first support
```

### `ichimoku`, Ichimoku Cloud

```
ichimoku.tenkan:<t>,<k>,<sb>@<high>,<low>,<close>
ichimoku.kijun:<t>,<k>,<sb>@<high>,<low>,<close>
ichimoku.senkou_a:<t>,<k>,<sb>@<high>,<low>,<close>
ichimoku.senkou_b:<t>,<k>,<sb>@<high>,<low>,<close>
ichimoku.chikou:<t>,<k>,<sb>@<high>,<low>,<close>
```

The five Ichimoku lines (no primary — request one):

- **tenkan** (alias `conversion`) = `(HH_t + LL_t) / 2`.
- **kijun** (`base`) = `(HH_k + LL_k) / 2`.
- **senkou_a** (`span_a`) = `(tenkan + kijun) / 2`, displaced forward `k` bars.
- **senkou_b** (`span_b`) = `(HH_sb + LL_sb) / 2`, displaced forward `k` bars.
- **chikou** (`lagging`) = the close — the lagging span. It is returned **causally** (so it
  never depends on a future bar); plot it shifted `−k` for charting, or compare `close` to
  `close.shift(k)` for the chikou-vs-price signal.

- **t?** `int=9` · **k?** `int=26` · **sb?** `int=52` (the kijun period `k` is also the forward displacement).
- **high? / low? / close?** `str` the input columns.

```py
df['ichimoku.tenkan']
df['ichimoku.kijun']
df['ichimoku.senkou_a']   # the cloud's leading edge, displaced to each bar
```

### `wad`, Williams Accumulation/Distribution (威廉多空力度线)

```
wad@<high>,<low>,<close>
```

Larry Williams' cumulative accumulation/distribution line (distinct from the TA-Lib `ad`): on
an up close it adds `C − min(prev C, L)`, on a down close `C − max(prev C, H)`, and is held
flat otherwise.

- **high? / low? / close?** `str` the input columns.

```py
df['wad']
```

### `asi`, Accumulative Swing Index (振动升降指标)

```
asi:<limit_move>@<open>,<high>,<low>,<close>
```

Welles Wilder's cumulative Swing Index, `ASI = Σ SI`, where each
`SI = 50 · (N / R) · (K / limit_move)` weighs the bar's directional move (`N`) against a
Wilder true-range denominator (`R`). `limit_move` is Wilder's per-market limit-move scaling.

- **limit_move?** `float=3.0`
- **open? / high? / low? / close?** `str` the input columns.

```py
df['asi']
df['asi:3']
```

### `supertrend`, Supertrend

```
supertrend:<period>,<mult>@<high>,<low>,<close>
supertrend.direction:<period>,<mult>@<high>,<low>,<close>
```

An ATR trailing-stop trend indicator: `hl2 ± mult · ATR` bands, recursively tightened against
the prior bar and flipped into a single trailing line. `supertrend` is that line (support in an
up-trend, resistance in a down-trend); `supertrend.direction` is the `+1` (up) / `−1` (down)
trend.

- **period?** `int=10` (the ATR period)
- **mult?** `float=3.0` (the band multiplier)
- **high? / low? / close?** `str` the input columns.

```py
df['supertrend']            # the trailing line
df['supertrend.direction']  # +1 up / -1 down
```

## Built-in Commands for Statistics

### `change`, Percentage Change

```
change:<period>@<on>
```

Percentage change between the current and a prior element on a certain series.
It computes the percentage change from the immediately previous element by
default, which is useful when comparing percentage change in a time series of
prices.

- **period?** `int=2` `2` means the start value and the end value of a two-period
  window are compared.
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Percentage change of the close column
df['change']

# Percentage change with a custom period
df['change:5@close']

# Percentage change of a nested directive
df['change@(ma:20)']
```

### `increase`, Consecutive Increase or Decrease

```
increase:<repeat>,<direction>@<on>
```

Gets a `bool` series where each item is `True` if the value of `on` increases in
the last `repeat` steps. Use `direction=-1` to detect repeated decreases.

- **repeat?** `int=1`
- **direction?** `int=1` `1` means increasing; `-1` means decreasing.
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Whether the 20-period moving average has increased repeatedly for 3 bars
df['increase:3@(ma:20@close)']

# Whether close has decreased repeatedly for 5 bars
df['increase:5,-1@close']
```

### `style`, Candle Color

```
style.<style>@<open>,<close>
```

Gets a `bool` series indicating whether the candlestick of a period is of the
given style. This native form is for candle color only; TA-Lib candlestick
patterns are exposed as `cdl.<pattern>` in the table below.

- **style** `'bullish'` or `'bearish'` (required)
- **open?** `str='open'` The column name for open prices.
- **close?** `str='close'` The column name for close prices.

```py
# Uses default open and close columns
df['style.bullish']
df['style.bearish']

# Specify custom columns
df['style.bearish@open,close']
```

### `repeat`, Consecutive Boolean Condition

```
repeat:<repeat>@<bool_directive>
```

The `repeat` command first gets the result of `bool_directive`, then detects
whether `True` repeats for `repeat` consecutive periods.

- **repeat?** `int=1` Must be larger than `0`.
- **bool_directive** `str | (Directive)` A column name or a directive wrapped in
  parentheses.

```py
# Whether bullish candlesticks repeat for 3 periods
df['repeat:3@(style.bullish)']

# Repeat check on a directive expression
df['repeat:5@(close > ma:20)']
```

## TA-Lib-compatible directives

TA-Lib-related directives use lowercase Volas names, but the `TA-Lib original`
column below lists the upstream TA-Lib function they correspond to. Arguments
before `@` are positional; input series after `@` override the default columns.
Square brackets mean an argument has a default. Required arguments are written
without brackets. Empty argument slots keep earlier defaults, so
`macd.signal:,,5` means fast period `12`, slow period `26`, and signal period `5`.

`matype` follows TA-Lib's integer convention: `0=SMA`, `1=EMA`, `2=WMA`,
`3=DEMA`, `4=TEMA`, `5=TRIMA`, `6=KAMA`, `7=MAMA`, and `8=T3`. Multi-output
indicators also accept short aliases where documented by the parser, for example
`macd.s` for `macd.signal`, `boll.u` for `boll.upper`, `aroon.d` for
`aroon.down`, and `style.<pattern>` as an alias for `cdl.<pattern>`.

```py
# Ordinary defaulted positional arguments
df['macd.signal:12,26,9@close']

# Directive command names are case-insensitive; column names after @ stay as written.
df['RSI:14@close']

# Skip fast and slow defaults, override only the signal period
df['macd.signal:,,5']

# A directive with multiple input series
df['stoch.d:5,3,0,3,0@high,low,close']

# MAVP needs a second series for the variable period
df['mavp:2,30,0@close,periods']
```

The TA-Lib Math Transform group is exposed on `Series` rather than as directive
strings: `acos`, `asin`, `atan`, `ceil`, `cos`, `cosh`, `exp`, `floor`, `ln`,
`log10`, `sin`, `sinh`, `sqrt`, `tan`, and `tanh`.

| Volas directive | TA-Lib original | Meaning | Parameters |
| --- | --- | --- | --- |
| `ma` | `MA` | Generic moving average selected by MA type. | `:<period>[,<matype=0>]@series=close` |
| `ema` | `EMA` | Exponential moving average. | `:<period>@series=close` |
| `wma` | `WMA` | Weighted moving average. | `[:period=30]@series=close` |
| `dema` | `DEMA` | Double exponential moving average. | `[:period=30]@series=close` |
| `tema` | `TEMA` | Triple exponential moving average. | `[:period=30]@series=close` |
| `trima` | `TRIMA` | Triangular moving average. | `[:period=30]@series=close` |
| `kama` | `KAMA` | Kaufman adaptive moving average. | `[:period=30]@series=close` |
| `t3` | `T3` | T3 moving average. | `[:period=5,vfactor=0.7]@series=close` |
| `mama` | `MAMA` | MESA adaptive moving average main line. | `[:fast_limit=0.5,slow_limit=0.05]@series=close` |
| `mama.fama` | `MAMA` | Following adaptive moving average line. | `[:fast_limit=0.5,slow_limit=0.05]@series=close` |
| `mavp` | `MAVP` | Moving average with per-row variable periods. | `[:min=2,max=30,matype=0]@series=close,periods` |
| `sar` | `SAR` | Parabolic SAR. | `[:acceleration=0.02,maximum=0.2]@high,low` |
| `sarext` | `SAREXT` | Extended Parabolic SAR. | `[:start=0,offset=0,long_init=0.02,long_step=0.02,long_max=0.2,short_init=0.02,short_step=0.02,short_max=0.2]@high,low` |
| `boll` | `BBANDS` | Bollinger middle band. | `[:period=20]@series=close` |
| `boll.upper` | `BBANDS` | Bollinger upper band. | `[:period=20,times=2]@series=close` |
| `boll.lower` | `BBANDS` | Bollinger lower band. | `[:period=20,times=2]@series=close` |
| `accbands` | `ACCBANDS` | Acceleration Bands middle line. | `[:period=20]@series=close` |
| `accbands.upper` | `ACCBANDS` | Acceleration Bands upper line. | `[:period=20]@high,low` |
| `accbands.lower` | `ACCBANDS` | Acceleration Bands lower line. | `[:period=20]@high,low` |
| `midpoint` | `MIDPOINT` | Midpoint over a rolling period. | `[:period=14]@series=close` |
| `midprice` | `MIDPRICE` | Midpoint price over high and low. | `[:period=14]@high,low` |
| `ht_trendline` | `HT_TRENDLINE` | Hilbert Transform instantaneous trendline. | `@series=close` |
| `macd` | `MACD` | MACD line; Volas uses standalone EMA fast minus EMA slow. | `[:fast=12,slow=26]@series=close` |
| `macd.signal` | `MACD` | Signal line of the Volas MACD line. | `[:fast=12,slow=26,signal=9]@series=close` |
| `macd.histogram` | `MACD` | MACD histogram: line minus signal. | `[:fast=12,slow=26,signal=9]@series=close` |
| `macdext` | `MACDEXT` | MACD line with independent MA types. | `[:fast=12,fast_matype=0,slow=26,slow_matype=0]@series=close` |
| `macdext.signal` | `MACDEXT` | MACDEXT signal line. | `[:fast=12,fast_matype=0,slow=26,slow_matype=0,signal=9,signal_matype=0]@series=close` |
| `macdext.histogram` | `MACDEXT` | MACDEXT histogram. | `[:fast=12,fast_matype=0,slow=26,slow_matype=0,signal=9,signal_matype=0]@series=close` |
| `macdfix` | `MACDFIX` | Fixed 12/26 MACD line; Volas uses standalone EMA fast minus EMA slow. | `@series=close` |
| `macdfix.signal` | `MACDFIX` | Signal line of the Volas fixed 12/26 MACD line. | `[:signal=9]@series=close` |
| `macdfix.histogram` | `MACDFIX` | Histogram of the Volas fixed 12/26 MACD line. | `[:signal=9]@series=close` |
| `apo` | `APO` | Absolute price oscillator. | `[:fast=12,slow=26,matype=0]@series=close` |
| `ppo` | `PPO` | Percentage price oscillator. | `[:fast=12,slow=26,matype=0]@series=close` |
| `rsi` | `RSI` | Relative Strength Index. | `:<period>@series=close` |
| `cmo` | `CMO` | Chande Momentum Oscillator. | `[:period=14]@series=close` |
| `cci` | `CCI` | Commodity Channel Index. | `[:period=14]@high,low,close` |
| `imi` | `IMI` | Intraday Momentum Index. | `[:period=14]@open,close` |
| `mfi` | `MFI` | Money Flow Index. | `[:period=14]@high,low,close,volume` |
| `bop` | `BOP` | Balance of Power. | `@open,high,low,close` |
| `willr` | `WILLR` | Williams Percent Range. | `[:period=14]@high,low,close` |
| `mom` | `MOM` | Momentum. | `[:period=10]@series=close` |
| `roc` | `ROC` | Rate of change. | `[:period=10]@series=close` |
| `rocp` | `ROCP` | Rate of change percentage. | `[:period=10]@series=close` |
| `rocr` | `ROCR` | Rate of change ratio. | `[:period=10]@series=close` |
| `rocr100` | `ROCR100` | Rate of change ratio multiplied by 100. | `[:period=10]@series=close` |
| `stoch.k` | `STOCH` | Slow stochastic percent K. | `[:fastk=5,slowk=3,slowk_matype=0,slowd=3,slowd_matype=0]@high,low,close` |
| `stoch.d` | `STOCH` | Slow stochastic percent D. | `[:fastk=5,slowk=3,slowk_matype=0,slowd=3,slowd_matype=0]@high,low,close` |
| `stochf.k` | `STOCHF` | Fast stochastic percent K. | `[:fastk=5,fastd=3,fastd_matype=0]@high,low,close` |
| `stochf.d` | `STOCHF` | Fast stochastic percent D. | `[:fastk=5,fastd=3,fastd_matype=0]@high,low,close` |
| `stochrsi.k` | `STOCHRSI` | Fast stochastic RSI percent K. | `[:rsi=14,fastk=5,fastd=3,fastd_matype=0]@series=close` |
| `stochrsi.d` | `STOCHRSI` | Fast stochastic RSI percent D. | `[:rsi=14,fastk=5,fastd=3,fastd_matype=0]@series=close` |
| `trix` | `TRIX` | One-period ROC of a triple EMA. | `[:period=30]@series=close` |
| `ultosc` | `ULTOSC` | Ultimate Oscillator. | `[:short=7,medium=14,long=28]@high,low,close` |
| `aroon.up` | `AROON` | Aroon up line. | `[:period=14]@high,low` |
| `aroon.down` | `AROON` | Aroon down line. | `[:period=14]@high,low` |
| `aroonosc` | `AROONOSC` | Aroon oscillator. | `[:period=14]@high,low` |
| `plus_dm` | `PLUS_DM` | Plus directional movement. | `[:period=14]@high,low` |
| `minus_dm` | `MINUS_DM` | Minus directional movement. | `[:period=14]@high,low` |
| `plus_di` | `PLUS_DI` | Plus directional indicator. | `[:period=14]@high,low,close` |
| `minus_di` | `MINUS_DI` | Minus directional indicator. | `[:period=14]@high,low,close` |
| `dx` | `DX` | Directional Movement Index. | `[:period=14]@high,low,close` |
| `adx` | `ADX` | Average Directional Movement Index. | `[:period=14]@high,low,close` |
| `adxr` | `ADXR` | Average Directional Movement Index Rating. | `[:period=14]@high,low,close` |
| `obv` | `OBV` | On-Balance Volume. | `@close,volume` |
| `ad` | `AD` | Chaikin Accumulation Distribution line. | `@high,low,close,volume` |
| `adosc` | `ADOSC` | Chaikin Accumulation Distribution oscillator. | `[:fast=3,slow=10]@high,low,close,volume` |
| `tr` | `TRANGE` | True Range. | `@high,low,close` |
| `atr` | `ATR` | Average True Range. | `[:period=14]@high,low,close` |
| `natr` | `NATR` | Normalized Average True Range. | `[:period=14]@high,low,close` |
| `avgprice` | `AVGPRICE` | Average price. | `@open,high,low,close` |
| `medprice` | `MEDPRICE` | Median price. | `@high,low` |
| `typprice` | `TYPPRICE` | Typical price. | `@high,low,close` |
| `wclprice` | `WCLPRICE` | Weighted close price. | `@high,low,close` |
| `ht_dcperiod` | `HT_DCPERIOD` | Hilbert Transform dominant cycle period. | `@series=close` |
| `ht_dcphase` | `HT_DCPHASE` | Hilbert Transform dominant cycle phase. | `@series=close` |
| `ht_phasor` | `HT_PHASOR` | Hilbert Transform phasor in-phase line. | `@series=close` |
| `ht_phasor.quadrature` | `HT_PHASOR` | Hilbert Transform phasor quadrature line. | `@series=close` |
| `ht_sine` | `HT_SINE` | Hilbert Transform sine wave. | `@series=close` |
| `ht_sine.leadsine` | `HT_SINE` | Hilbert Transform lead sine wave. | `@series=close` |
| `ht_trendmode` | `HT_TRENDMODE` | Hilbert Transform trend versus cycle mode. | `@series=close` |
| `linearreg` | `LINEARREG` | Linear regression value. | `[:period=14]@series=close` |
| `linearreg_slope` | `LINEARREG_SLOPE` | Linear regression slope. | `[:period=14]@series=close` |
| `linearreg_intercept` | `LINEARREG_INTERCEPT` | Linear regression intercept. | `[:period=14]@series=close` |
| `linearreg_angle` | `LINEARREG_ANGLE` | Linear regression angle. | `[:period=14]@series=close` |
| `tsf` | `TSF` | Time Series Forecast. | `[:period=14]@series=close` |
| `var` | `VAR` | Variance. | `[:period=5]@series=close` |
| `stddev` | `STDDEV` | Standard deviation. | `[:period=5,nbdev=1]@series=close` |
| `correl` | `CORREL` | Pearson correlation coefficient. | `[:period=30]@series0=close,series1` |
| `beta` | `BETA` | Beta. | `[:period=5]@series0=close,series1` |
| `sum` | `SUM` | Rolling sum. | `[:period=30]@series=close` |
| `maxindex` | `MAXINDEX` | Index of the rolling maximum. | `[:period=30]@series=close` |
| `minindex` | `MININDEX` | Index of the rolling minimum. | `[:period=30]@series=close` |
| `minmax.min` | `MINMAX` | Rolling minimum from the MINMAX pair. | `[:period=30]@series=close` |
| `minmax.max` | `MINMAX` | Rolling maximum from the MINMAX pair. | `[:period=30]@series=close` |
| `minmaxindex.min` | `MINMAXINDEX` | Index of the rolling minimum from the pair. | `[:period=30]@series=close` |
| `minmaxindex.max` | `MINMAXINDEX` | Index of the rolling maximum from the pair. | `[:period=30]@series=close` |
| `cdl.2crows` | `CDL2CROWS` | Two Crows | `@open,high,low,close` |
| `cdl.3blackcrows` | `CDL3BLACKCROWS` | Three Black Crows | `@open,high,low,close` |
| `cdl.3inside` | `CDL3INSIDE` | Three Inside Up/Down | `@open,high,low,close` |
| `cdl.3linestrike` | `CDL3LINESTRIKE` | Three-Line Strike  | `@open,high,low,close` |
| `cdl.3outside` | `CDL3OUTSIDE` | Three Outside Up/Down | `@open,high,low,close` |
| `cdl.3starsinsouth` | `CDL3STARSINSOUTH` | Three Stars In The South | `@open,high,low,close` |
| `cdl.3whitesoldiers` | `CDL3WHITESOLDIERS` | Three Advancing White Soldiers | `@open,high,low,close` |
| `cdl.abandonedbaby` | `CDLABANDONEDBABY` | Abandoned Baby | `[:penetration=0.3]@open,high,low,close` |
| `cdl.advanceblock` | `CDLADVANCEBLOCK` | Advance Block | `@open,high,low,close` |
| `cdl.belthold` | `CDLBELTHOLD` | Belt-hold | `@open,high,low,close` |
| `cdl.breakaway` | `CDLBREAKAWAY` | Breakaway | `@open,high,low,close` |
| `cdl.closingmarubozu` | `CDLCLOSINGMARUBOZU` | Closing Marubozu | `@open,high,low,close` |
| `cdl.concealbabyswall` | `CDLCONCEALBABYSWALL` | Concealing Baby Swallow | `@open,high,low,close` |
| `cdl.counterattack` | `CDLCOUNTERATTACK` | Counterattack | `@open,high,low,close` |
| `cdl.darkcloudcover` | `CDLDARKCLOUDCOVER` | Dark Cloud Cover | `[:penetration=0.5]@open,high,low,close` |
| `cdl.doji` | `CDLDOJI` | Doji | `@open,high,low,close` |
| `cdl.dojistar` | `CDLDOJISTAR` | Doji Star | `@open,high,low,close` |
| `cdl.dragonflydoji` | `CDLDRAGONFLYDOJI` | Dragonfly Doji | `@open,high,low,close` |
| `cdl.engulfing` | `CDLENGULFING` | Engulfing Pattern | `@open,high,low,close` |
| `cdl.eveningdojistar` | `CDLEVENINGDOJISTAR` | Evening Doji Star | `[:penetration=0.3]@open,high,low,close` |
| `cdl.eveningstar` | `CDLEVENINGSTAR` | Evening Star | `[:penetration=0.3]@open,high,low,close` |
| `cdl.gapsidesidewhite` | `CDLGAPSIDESIDEWHITE` | Up/Down-gap side-by-side white lines | `@open,high,low,close` |
| `cdl.gravestonedoji` | `CDLGRAVESTONEDOJI` | Gravestone Doji | `@open,high,low,close` |
| `cdl.hammer` | `CDLHAMMER` | Hammer | `@open,high,low,close` |
| `cdl.hangingman` | `CDLHANGINGMAN` | Hanging Man | `@open,high,low,close` |
| `cdl.harami` | `CDLHARAMI` | Harami Pattern | `@open,high,low,close` |
| `cdl.haramicross` | `CDLHARAMICROSS` | Harami Cross Pattern | `@open,high,low,close` |
| `cdl.highwave` | `CDLHIGHWAVE` | High-Wave Candle | `@open,high,low,close` |
| `cdl.hikkake` | `CDLHIKKAKE` | Hikkake Pattern | `@open,high,low,close` |
| `cdl.hikkakemod` | `CDLHIKKAKEMOD` | Modified Hikkake Pattern | `@open,high,low,close` |
| `cdl.homingpigeon` | `CDLHOMINGPIGEON` | Homing Pigeon | `@open,high,low,close` |
| `cdl.identical3crows` | `CDLIDENTICAL3CROWS` | Identical Three Crows | `@open,high,low,close` |
| `cdl.inneck` | `CDLINNECK` | In-Neck Pattern | `@open,high,low,close` |
| `cdl.invertedhammer` | `CDLINVERTEDHAMMER` | Inverted Hammer | `@open,high,low,close` |
| `cdl.kicking` | `CDLKICKING` | Kicking | `@open,high,low,close` |
| `cdl.kickingbylength` | `CDLKICKINGBYLENGTH` | Kicking - bull/bear determined by the longer marubozu | `@open,high,low,close` |
| `cdl.ladderbottom` | `CDLLADDERBOTTOM` | Ladder Bottom | `@open,high,low,close` |
| `cdl.longleggeddoji` | `CDLLONGLEGGEDDOJI` | Long Legged Doji | `@open,high,low,close` |
| `cdl.longline` | `CDLLONGLINE` | Long Line Candle | `@open,high,low,close` |
| `cdl.marubozu` | `CDLMARUBOZU` | Marubozu | `@open,high,low,close` |
| `cdl.matchinglow` | `CDLMATCHINGLOW` | Matching Low | `@open,high,low,close` |
| `cdl.mathold` | `CDLMATHOLD` | Mat Hold | `[:penetration=0.5]@open,high,low,close` |
| `cdl.morningdojistar` | `CDLMORNINGDOJISTAR` | Morning Doji Star | `[:penetration=0.3]@open,high,low,close` |
| `cdl.morningstar` | `CDLMORNINGSTAR` | Morning Star | `[:penetration=0.3]@open,high,low,close` |
| `cdl.onneck` | `CDLONNECK` | On-Neck Pattern | `@open,high,low,close` |
| `cdl.piercing` | `CDLPIERCING` | Piercing Pattern | `@open,high,low,close` |
| `cdl.rickshawman` | `CDLRICKSHAWMAN` | Rickshaw Man | `@open,high,low,close` |
| `cdl.risefall3methods` | `CDLRISEFALL3METHODS` | Rising/Falling Three Methods | `@open,high,low,close` |
| `cdl.separatinglines` | `CDLSEPARATINGLINES` | Separating Lines | `@open,high,low,close` |
| `cdl.shootingstar` | `CDLSHOOTINGSTAR` | Shooting Star | `@open,high,low,close` |
| `cdl.shortline` | `CDLSHORTLINE` | Short Line Candle | `@open,high,low,close` |
| `cdl.spinningtop` | `CDLSPINNINGTOP` | Spinning Top | `@open,high,low,close` |
| `cdl.stalledpattern` | `CDLSTALLEDPATTERN` | Stalled Pattern | `@open,high,low,close` |
| `cdl.sticksandwich` | `CDLSTICKSANDWICH` | Stick Sandwich | `@open,high,low,close` |
| `cdl.takuri` | `CDLTAKURI` | Takuri (Dragonfly Doji with very long lower shadow) | `@open,high,low,close` |
| `cdl.tasukigap` | `CDLTASUKIGAP` | Tasuki Gap | `@open,high,low,close` |
| `cdl.thrusting` | `CDLTHRUSTING` | Thrusting Pattern | `@open,high,low,close` |
| `cdl.tristar` | `CDLTRISTAR` | Tristar Pattern | `@open,high,low,close` |
| `cdl.unique3river` | `CDLUNIQUE3RIVER` | Unique 3 River | `@open,high,low,close` |
| `cdl.upsidegap2crows` | `CDLUPSIDEGAP2CROWS` | Upside Gap Two Crows | `@open,high,low,close` |
| `cdl.xsidegap3methods` | `CDLXSIDEGAP3METHODS` | Upside/Downside Gap Three Methods | `@open,high,low,close` |
