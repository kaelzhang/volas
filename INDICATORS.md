# Built-in Indicators

This page is the complete directive reference for `volas` — **254** indicators
(each main command, every multi-output sub-command line, and each candlestick
pattern; counted by `scripts/count_indicators.py` from the Rust source).

## Indicator index

Directive names are case-insensitive. Alias notes list extra command or sub-command spellings accepted by the parser; multi-output indicators list each selectable output line.

- **Volas-exclusive indicators:** [`smma`](#user-content-indicator-smma), [`bbi`](#user-content-indicator-bbi), [`bbw`](#user-content-indicator-bbw), [`rsv`](#user-content-indicator-rsv), [KDJ](#user-content-indicator-kdj) (sub-commands: [`kdj.k`](#user-content-indicator-kdj), [`kdj.d`](#user-content-indicator-kdj), [`kdj.j`](#user-content-indicator-kdj); sub-commands only), [`llv`](#user-content-indicator-llv), [`hhv`](#user-content-indicator-hhv), [`donchian`](#user-content-indicator-donchian) (sub-commands: [`donchian.upper`](#user-content-indicator-donchian), [`donchian.lower`](#user-content-indicator-donchian); aliases: `.middle`/`.m`, `.u`, `.l`), [`hv`](#user-content-indicator-hv), [`psy`](#user-content-indicator-psy), [`dpo`](#user-content-indicator-dpo), [`tsi`](#user-content-indicator-tsi), [`kst`](#user-content-indicator-kst), [`crsi`](#user-content-indicator-crsi), [`chop`](#user-content-indicator-chop), [`cmf`](#user-content-indicator-cmf), [`emv`](#user-content-indicator-emv), [`efi`](#user-content-indicator-efi), [`pvt`](#user-content-indicator-pvt), [`nvi`](#user-content-indicator-nvi), [`pvi`](#user-content-indicator-pvi), [`mass_index`](#user-content-indicator-mass_index), [`bias`](#user-content-indicator-bias), [`dma`](#user-content-indicator-dma) (sub-commands: [`dma.ama`](#user-content-indicator-dma); aliases: `dma.ddd`), [Vortex](#user-content-indicator-vortex) (sub-commands: [`vortex.plus`](#user-content-indicator-vortex), [`vortex.minus`](#user-content-indicator-vortex); aliases: `.p`, `.m`; sub-commands only), [BRAR](#user-content-indicator-brar) (sub-commands: [`brar.ar`](#user-content-indicator-brar), [`brar.br`](#user-content-indicator-brar); sub-commands only), [`vr`](#user-content-indicator-vr), [`coppock`](#user-content-indicator-coppock), [`relative_vigor`](#user-content-indicator-relative_vigor) (sub-commands: [`relative_vigor.signal`](#user-content-indicator-relative_vigor)), [`dkx`](#user-content-indicator-dkx) (sub-commands: [`dkx.ma`](#user-content-indicator-dkx)), [`wvad`](#user-content-indicator-wvad), [`cdp`](#user-content-indicator-cdp) (sub-commands: [`cdp.ah`](#user-content-indicator-cdp), [`cdp.nh`](#user-content-indicator-cdp), [`cdp.nl`](#user-content-indicator-cdp), [`cdp.al`](#user-content-indicator-cdp)), [MIKE](#user-content-indicator-mike) (sub-commands: [`mike.weakr`](#user-content-indicator-mike), [`mike.midr`](#user-content-indicator-mike), [`mike.strongr`](#user-content-indicator-mike), [`mike.weaks`](#user-content-indicator-mike), [`mike.mids`](#user-content-indicator-mike), [`mike.strongs`](#user-content-indicator-mike); sub-commands only), [`keltner`](#user-content-indicator-keltner) (sub-commands: [`keltner.upper`](#user-content-indicator-keltner), [`keltner.lower`](#user-content-indicator-keltner); aliases: `.middle`/`.m`, `.u`, `.l`), [`stoch_momentum`](#user-content-indicator-stoch_momentum) (sub-commands: [`stoch_momentum.signal`](#user-content-indicator-stoch_momentum)), [`ttm_squeeze`](#user-content-indicator-ttm_squeeze) (sub-commands: [`ttm_squeeze.on`](#user-content-indicator-ttm_squeeze)), [`pivot_points`](#user-content-indicator-pivot_points) (sub-commands: [`pivot_points.r1`](#user-content-indicator-pivot_points), [`pivot_points.r2`](#user-content-indicator-pivot_points), [`pivot_points.r3`](#user-content-indicator-pivot_points), [`pivot_points.s1`](#user-content-indicator-pivot_points), [`pivot_points.s2`](#user-content-indicator-pivot_points), [`pivot_points.s3`](#user-content-indicator-pivot_points); aliases: `.p`/`.pp`), [Ichimoku](#user-content-indicator-ichimoku) (sub-commands: [`ichimoku.tenkan`](#user-content-indicator-ichimoku), [`ichimoku.kijun`](#user-content-indicator-ichimoku), [`ichimoku.senkou_a`](#user-content-indicator-ichimoku), [`ichimoku.senkou_b`](#user-content-indicator-ichimoku), [`ichimoku.chikou`](#user-content-indicator-ichimoku); aliases: `.conversion`, `.base`, `.span_a`, `.span_b`, `.lagging`; sub-commands only), [`wad`](#user-content-indicator-wad), [`asi`](#user-content-indicator-asi), [`supertrend`](#user-content-indicator-supertrend) (sub-commands: [`supertrend.direction`](#user-content-indicator-supertrend); aliases: `.line`, `.trend`, `.d`).
- **Built-in Commands for Statistics:** [`change`](#user-content-indicator-change), [`increase`](#user-content-indicator-increase), [Candle color](#user-content-indicator-style) (sub-commands: [`style.bullish`](#user-content-indicator-style), [`style.bearish`](#user-content-indicator-style); sub-commands only), [`repeat`](#user-content-indicator-repeat), [`median`](#user-content-indicator-median), [`quantile`](#user-content-indicator-quantile), [`rank`](#user-content-indicator-rank), [`skew`](#user-content-indicator-skew), [`kurt`](#user-content-indicator-kurt), [`sem`](#user-content-indicator-sem).
- **TradingView-compatible directives:** [`vwma`](#user-content-indicator-vwma), [`alma`](#user-content-indicator-alma), [`hma`](#user-content-indicator-hma), [`swma`](#user-content-indicator-swma), [`cog`](#user-content-indicator-cog), [`dev`](#user-content-indicator-dev), [`rci`](#user-content-indicator-rci), [`iii`](#user-content-indicator-iii), [`kcw`](#user-content-indicator-kcw), [`mode`](#user-content-indicator-mode), [`pivothigh`](#user-content-indicator-pivothigh), [`pivotlow`](#user-content-indicator-pivotlow).
- **TA-Lib-compatible directives:** [`ma`](#user-content-indicator-ma), [`ema`](#user-content-indicator-ema), [`wma`](#user-content-indicator-wma), [`dema`](#user-content-indicator-dema), [`tema`](#user-content-indicator-tema), [`trima`](#user-content-indicator-trima), [`kama`](#user-content-indicator-kama), [`t3`](#user-content-indicator-t3), [`mama`](#user-content-indicator-mama) (sub-commands: [`mama.fama`](#user-content-indicator-mama-fama); aliases: `.mama`), [`mavp`](#user-content-indicator-mavp), [`sar`](#user-content-indicator-sar), [`sarext`](#user-content-indicator-sarext), [`boll`](#user-content-indicator-boll) (sub-commands: [`boll.upper`](#user-content-indicator-boll-upper), [`boll.lower`](#user-content-indicator-boll-lower); aliases: `.middle`/`.m`, `.u`, `.l`), [`accbands`](#user-content-indicator-accbands) (sub-commands: [`accbands.upper`](#user-content-indicator-accbands-upper), [`accbands.lower`](#user-content-indicator-accbands-lower); aliases: `.middle`/`.m`, `.u`, `.l`), [`midpoint`](#user-content-indicator-midpoint), [`midprice`](#user-content-indicator-midprice), [`ht_trendline`](#user-content-indicator-ht_trendline), [`macd`](#user-content-indicator-macd) (sub-commands: [`macd.signal`](#user-content-indicator-macd-signal), [`macd.histogram`](#user-content-indicator-macd-histogram); aliases: `.dif`, `.s`/`.dea`, `.h`/`.macd`), [`macdext`](#user-content-indicator-macdext) (sub-commands: [`macdext.signal`](#user-content-indicator-macdext-signal), [`macdext.histogram`](#user-content-indicator-macdext-histogram); aliases: `.dif`, `.s`/`.dea`, `.h`/`.macd`), [`macdfix`](#user-content-indicator-macdfix) (sub-commands: [`macdfix.signal`](#user-content-indicator-macdfix-signal), [`macdfix.histogram`](#user-content-indicator-macdfix-histogram); aliases: `.dif`, `.s`/`.dea`, `.h`/`.macd`), [`apo`](#user-content-indicator-apo), [`ppo`](#user-content-indicator-ppo), [`rsi`](#user-content-indicator-rsi), [`cmo`](#user-content-indicator-cmo), [`cci`](#user-content-indicator-cci), [`imi`](#user-content-indicator-imi), [`mfi`](#user-content-indicator-mfi), [`bop`](#user-content-indicator-bop), [`willr`](#user-content-indicator-willr), [`mom`](#user-content-indicator-mom), [`roc`](#user-content-indicator-roc), [`rocp`](#user-content-indicator-rocp).
- **TA-Lib-compatible directives, continued:** [`rocr`](#user-content-indicator-rocr), [`rocr100`](#user-content-indicator-rocr100), [STOCH](#user-content-indicator-stoch-k) (sub-commands: [`stoch.k`](#user-content-indicator-stoch-k), [`stoch.d`](#user-content-indicator-stoch-d); aliases: `.slowk`, `.slowd`; sub-commands only), [STOCHF](#user-content-indicator-stochf-k) (sub-commands: [`stochf.k`](#user-content-indicator-stochf-k), [`stochf.d`](#user-content-indicator-stochf-d); aliases: `.fastk`, `.fastd`; sub-commands only), [STOCHRSI](#user-content-indicator-stochrsi-k) (sub-commands: [`stochrsi.k`](#user-content-indicator-stochrsi-k), [`stochrsi.d`](#user-content-indicator-stochrsi-d); aliases: `.fastk`, `.fastd`; sub-commands only), [`trix`](#user-content-indicator-trix), [`ultosc`](#user-content-indicator-ultosc), [AROON](#user-content-indicator-aroon-up) (sub-commands: [`aroon.up`](#user-content-indicator-aroon-up), [`aroon.down`](#user-content-indicator-aroon-down); aliases: `.u`, `.d`; sub-commands only), [`aroonosc`](#user-content-indicator-aroonosc), [`plus_dm`](#user-content-indicator-plus_dm), [`minus_dm`](#user-content-indicator-minus_dm), [`plus_di`](#user-content-indicator-plus_di), [`minus_di`](#user-content-indicator-minus_di), [`dx`](#user-content-indicator-dx), [`adx`](#user-content-indicator-adx), [`adxr`](#user-content-indicator-adxr), [`obv`](#user-content-indicator-obv), [`ad`](#user-content-indicator-ad), [`adosc`](#user-content-indicator-adosc), [`tr`](#user-content-indicator-tr), [`atr`](#user-content-indicator-atr), [`natr`](#user-content-indicator-natr), [`avgprice`](#user-content-indicator-avgprice), [`medprice`](#user-content-indicator-medprice), [`typprice`](#user-content-indicator-typprice), [`wclprice`](#user-content-indicator-wclprice), [`ht_dcperiod`](#user-content-indicator-ht_dcperiod), [`ht_dcphase`](#user-content-indicator-ht_dcphase), [`ht_phasor`](#user-content-indicator-ht_phasor) (sub-commands: [`ht_phasor.quadrature`](#user-content-indicator-ht_phasor-quadrature); aliases: `.i`/`.inphase`, `.q`/`.quad`), [`ht_sine`](#user-content-indicator-ht_sine) (sub-commands: [`ht_sine.leadsine`](#user-content-indicator-ht_sine-leadsine); aliases: `.sine`, `.lead`), [`ht_trendmode`](#user-content-indicator-ht_trendmode), [`linearreg`](#user-content-indicator-linearreg), [`linearreg_slope`](#user-content-indicator-linearreg_slope), [`linearreg_intercept`](#user-content-indicator-linearreg_intercept), [`linearreg_angle`](#user-content-indicator-linearreg_angle), [`tsf`](#user-content-indicator-tsf), [`var`](#user-content-indicator-var), [`stddev`](#user-content-indicator-stddev), [`correl`](#user-content-indicator-correl), [`beta`](#user-content-indicator-beta), [`sum`](#user-content-indicator-sum), [`maxindex`](#user-content-indicator-maxindex), [`minindex`](#user-content-indicator-minindex), [MINMAX](#user-content-indicator-minmax-min) (sub-commands: [`minmax.min`](#user-content-indicator-minmax-min), [`minmax.max`](#user-content-indicator-minmax-max); sub-commands only), [MINMAXINDEX](#user-content-indicator-minmaxindex-min) (sub-commands: [`minmaxindex.min`](#user-content-indicator-minmaxindex-min), [`minmaxindex.max`](#user-content-indicator-minmaxindex-max); sub-commands only).
- **Candlestick pattern directives:** every `cdl.<pattern>` entry also accepts `style.<pattern>` as an alias: [`cdl.2crows`](#user-content-indicator-cdl-2crows), [`cdl.3blackcrows`](#user-content-indicator-cdl-3blackcrows), [`cdl.3inside`](#user-content-indicator-cdl-3inside), [`cdl.3linestrike`](#user-content-indicator-cdl-3linestrike), [`cdl.3outside`](#user-content-indicator-cdl-3outside), [`cdl.3starsinsouth`](#user-content-indicator-cdl-3starsinsouth), [`cdl.3whitesoldiers`](#user-content-indicator-cdl-3whitesoldiers), [`cdl.abandonedbaby`](#user-content-indicator-cdl-abandonedbaby), [`cdl.advanceblock`](#user-content-indicator-cdl-advanceblock), [`cdl.belthold`](#user-content-indicator-cdl-belthold), [`cdl.breakaway`](#user-content-indicator-cdl-breakaway), [`cdl.closingmarubozu`](#user-content-indicator-cdl-closingmarubozu), [`cdl.concealbabyswall`](#user-content-indicator-cdl-concealbabyswall), [`cdl.counterattack`](#user-content-indicator-cdl-counterattack), [`cdl.darkcloudcover`](#user-content-indicator-cdl-darkcloudcover), [`cdl.doji`](#user-content-indicator-cdl-doji), [`cdl.dojistar`](#user-content-indicator-cdl-dojistar), [`cdl.dragonflydoji`](#user-content-indicator-cdl-dragonflydoji), [`cdl.engulfing`](#user-content-indicator-cdl-engulfing), [`cdl.eveningdojistar`](#user-content-indicator-cdl-eveningdojistar), [`cdl.eveningstar`](#user-content-indicator-cdl-eveningstar), [`cdl.gapsidesidewhite`](#user-content-indicator-cdl-gapsidesidewhite), [`cdl.gravestonedoji`](#user-content-indicator-cdl-gravestonedoji), [`cdl.hammer`](#user-content-indicator-cdl-hammer), [`cdl.hangingman`](#user-content-indicator-cdl-hangingman), [`cdl.harami`](#user-content-indicator-cdl-harami), [`cdl.haramicross`](#user-content-indicator-cdl-haramicross), [`cdl.highwave`](#user-content-indicator-cdl-highwave), [`cdl.hikkake`](#user-content-indicator-cdl-hikkake), [`cdl.hikkakemod`](#user-content-indicator-cdl-hikkakemod), [`cdl.homingpigeon`](#user-content-indicator-cdl-homingpigeon), [`cdl.identical3crows`](#user-content-indicator-cdl-identical3crows), [`cdl.inneck`](#user-content-indicator-cdl-inneck), [`cdl.invertedhammer`](#user-content-indicator-cdl-invertedhammer), [`cdl.kicking`](#user-content-indicator-cdl-kicking), [`cdl.kickingbylength`](#user-content-indicator-cdl-kickingbylength), [`cdl.ladderbottom`](#user-content-indicator-cdl-ladderbottom), [`cdl.longleggeddoji`](#user-content-indicator-cdl-longleggeddoji), [`cdl.longline`](#user-content-indicator-cdl-longline), [`cdl.marubozu`](#user-content-indicator-cdl-marubozu), [`cdl.matchinglow`](#user-content-indicator-cdl-matchinglow), [`cdl.mathold`](#user-content-indicator-cdl-mathold), [`cdl.morningdojistar`](#user-content-indicator-cdl-morningdojistar), [`cdl.morningstar`](#user-content-indicator-cdl-morningstar), [`cdl.onneck`](#user-content-indicator-cdl-onneck), [`cdl.piercing`](#user-content-indicator-cdl-piercing), [`cdl.rickshawman`](#user-content-indicator-cdl-rickshawman), [`cdl.risefall3methods`](#user-content-indicator-cdl-risefall3methods), [`cdl.separatinglines`](#user-content-indicator-cdl-separatinglines), [`cdl.shootingstar`](#user-content-indicator-cdl-shootingstar), [`cdl.shortline`](#user-content-indicator-cdl-shortline), [`cdl.spinningtop`](#user-content-indicator-cdl-spinningtop), [`cdl.stalledpattern`](#user-content-indicator-cdl-stalledpattern), [`cdl.sticksandwich`](#user-content-indicator-cdl-sticksandwich), [`cdl.takuri`](#user-content-indicator-cdl-takuri), [`cdl.tasukigap`](#user-content-indicator-cdl-tasukigap), [`cdl.thrusting`](#user-content-indicator-cdl-thrusting), [`cdl.tristar`](#user-content-indicator-cdl-tristar), [`cdl.unique3river`](#user-content-indicator-cdl-unique3river), [`cdl.upsidegap2crows`](#user-content-indicator-cdl-upsidegap2crows), [`cdl.xsidegap3methods`](#user-content-indicator-cdl-xsidegap3methods).

Volas supports indicators in two groups. The first group is native to Volas or
inherits stock-pandas directive names; TA-Lib either has no equivalent or no
first-class function with the same directive name and OHLCV defaults. The second
group follows TA-Lib's function surface: directive names are lowercase,
arguments are positional, and multi-output indicators expose each line as a
sub-command such as `macd.signal`, `boll.upper`, or `ht_sine.leadsine`.

**Reading the signatures.** A parameter written `<name=value>` keeps that value as
its default — it is the one dominant industry standard for the indicator (e.g.
Bollinger's `20`, MACD's `12,26,9`). A parameter written `<name>` (no `=value`) is
**required**: it has no single dominant default, so volas does not invent one — you
supply the value your strategy uses. Common choices are indicator-specific — RSI is
most often `14` but `9` / `21` / `25` are all used; ATR / ADX usually `14`; Donchian
`20`, also `10` / `55`; a moving-average period or a rolling `sum` / `var` window has
no standard at all. (`df[d]` has always required these; only `directive_stringify` /
`directive_lookback` now reject a bare required form.)

## Volas-exclusive indicators

These directives are implemented by Volas itself. Many of them follow the
stock-pandas directive vocabulary, with the examples adapted to `volas.DataFrame`.

### <a id="indicator-smma"></a>`smma`, Smoothed Moving Average

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

### <a id="indicator-bbi"></a>`bbi`, Bull and Bear Index (多空指标)

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

### <a id="indicator-bbw"></a>`bbw`, Bollinger Band Width

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

### <a id="indicator-rsv"></a>`rsv`, Raw Stochastic Value (未成熟随机值)

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

### <a id="indicator-kdj"></a>`kdj`, A Variety of Stochastic Oscillator (随机指标)

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

### <a id="indicator-llv"></a>`llv`, Lowest of Low Values

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

### <a id="indicator-hhv"></a>`hhv`, Highest of High Values

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

### <a id="indicator-donchian"></a>`donchian`, Donchian Channels

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

### <a id="indicator-hv"></a>`hv`, Historical Volatility

```
hv:<period>,<time_frame>,<trading_days>@<on>
```

Gets historical volatility, the statistical measure of the dispersion of returns
for a security or index over a period of time.

- **period** `int` (required)
- **time_frame?** `str='1d'` Time frame such as `1m`, `15m`, `1h`, or `1d`.
- **trading_days?** `int=254` Trading days in a year; pass an exchange-specific
  yearly session count when your stock universe uses a different calendar.
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# 10-period historical volatility for 15-minute data based on 252 US equity sessions
df['hv:10,15m,252']

# Uses default time_frame and trading_days
df['hv:10']
```

### <a id="indicator-psy"></a>`psy`, Psychological Line (心理线)

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

### <a id="indicator-dpo"></a>`dpo`, Detrended Price Oscillator

```
dpo:<period>@<on>
```

The price `period/2 + 1` bars ago minus the `period`-bar SMA, removing the trend
to expose shorter cycles.

- **period** `int` (required)
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['dpo']
df['dpo:10']
```

### <a id="indicator-tsi"></a>`tsi`, True Strength Index

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

### <a id="indicator-kst"></a>`kst`, Know Sure Thing

```
kst@<on>
```

Pring's momentum oscillator: a weighted sum of four SMA-smoothed rate-of-change
terms (ROC 10/15/20/30, smoothed by SMA 10/10/10/15, weighted 1/2/3/4).

- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['kst']
```

### <a id="indicator-crsi"></a>`crsi`, Connors RSI

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

### <a id="indicator-chop"></a>`chop`, Choppiness Index

```
chop:<period>@<high>,<low>,<close>
```

How choppy versus trending the market is over `period` bars:
`100 * log10(sum(TR) / (HHV − LLV)) / log10(period)`. Higher is choppier.

- **period** `int` (required)
- **high? / low? / close?** `str` the input columns; default to the like-named frame columns.

```py
df['chop']
df['chop:14']
```

### <a id="indicator-cmf"></a>`cmf`, Chaikin Money Flow

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

### <a id="indicator-emv"></a>`emv`, Ease of Movement

```
emv:<period>@<high>,<low>,<volume>
```

The `period`-bar SMA of price displacement per unit of volume (StockCharts' 1e8
volume scale) — how easily price moves.

- **period** `int` (required)
- **high? / low? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['emv']
df['emv:14']
```

### <a id="indicator-efi"></a>`efi`, Elder Force Index

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

### <a id="indicator-pvt"></a>`pvt`, Price Volume Trend

```
pvt@<close>,<volume>
```

A cumulative volume line weighted by each bar's return:
`PVT += (Δclose / prev close) * volume`.

- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['pvt']
```

### <a id="indicator-nvi"></a>`nvi`, Negative Volume Index

```
nvi@<close>,<volume>
```

A cumulative line (base 1000) that compounds the return only on bars where volume
fell — tracking the "smart money" days.

- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['nvi']
```

### <a id="indicator-pvi"></a>`pvi`, Positive Volume Index

```
pvi@<close>,<volume>
```

A cumulative line (base 1000) that compounds the return only on bars where volume
rose — tracking the "crowd" days.

- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['pvi']
```

### <a id="indicator-mass_index"></a>`mass_index`, Mass Index

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

### <a id="indicator-bias"></a>`bias`, Bias Ratio (乖离率)

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

### <a id="indicator-dma"></a>`dma`, Difference of Moving Average (平行线差)

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

### <a id="indicator-vortex"></a>`vortex`, Vortex Indicator

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

### <a id="indicator-brar"></a>`brar`, BRAR Sentiment (人气意愿指标)

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

### <a id="indicator-vr"></a>`vr`, Volume Ratio (成交量比率)

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

### <a id="indicator-coppock"></a>`coppock`, Coppock Curve

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

### <a id="indicator-relative_vigor"></a>`relative_vigor`, Relative Vigor Index

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

### <a id="indicator-dkx"></a>`dkx`, Bull-Bear Line (多空线)

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

### <a id="indicator-wvad"></a>`wvad`, Williams Variable Accumulation/Distribution (威廉变异离散量)

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

### <a id="indicator-cdp"></a>`cdp`, Counter-Trend Operation (逆势操作)

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

### <a id="indicator-mike"></a>`mike`, MIKE Support/Resistance (麦克指标)

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

- **period** `int` (required)
- **high? / low? / close?** `str` the input columns.

```py
df['mike.strongr']   # strong resistance
df['mike.weaks']     # weak support
```

### <a id="indicator-keltner"></a>`keltner`, Keltner Channels

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

### <a id="indicator-stoch_momentum"></a>`stoch_momentum`, Stochastic Momentum Index

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

### <a id="indicator-ttm_squeeze"></a>`ttm_squeeze`, TTM Squeeze

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

### <a id="indicator-pivot_points"></a>`pivot_points`, Pivot Points

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

### <a id="indicator-ichimoku"></a>`ichimoku`, Ichimoku Cloud

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

### <a id="indicator-wad"></a>`wad`, Williams Accumulation/Distribution (威廉多空力度线)

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

### <a id="indicator-asi"></a>`asi`, Accumulative Swing Index (振动升降指标)

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

### <a id="indicator-supertrend"></a>`supertrend`, Supertrend

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

### <a id="indicator-change"></a>`change`, Percentage Change

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

### <a id="indicator-increase"></a>`increase`, Consecutive Increase or Decrease

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

### <a id="indicator-style"></a>`style`, Candle Color

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

### <a id="indicator-repeat"></a>`repeat`, Consecutive Boolean Condition

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

### <a id="indicator-median"></a>`median`, Rolling Median

```
median:<period>@<series>
```

The median of the trailing `period` values. A window containing a missing
value yields `NA` (the full-window warm-up discipline shared by the TA family).

- **period** `int` (required) Must be `>= 2`.
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['median:20']            # 20-bar rolling median of close
df['median:50@volume']     # on another column
```

### <a id="indicator-quantile"></a>`quantile`, Rolling Quantile

```
quantile:<period>,<q>@<series>
```

The `q`-quantile (linear interpolation) of the trailing `period` values — a
percentile channel in one directive.

- **period** `int` (required) Must be `>= 2`.
- **q?** `float=0.5` The quantile level, in `[0, 1]`.
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['quantile:20,0.9']      # the 90th percentile of the last 20 closes
df['quantile:20,0.1']      # the 10th percentile (lower channel edge)
```

### <a id="indicator-rank"></a>`rank`, Rolling Percent Rank

```
rank:<period>@<series>
```

The percent rank of the **current** bar within its own trailing window, in
`(0, 1]` — "where does today sit inside the last `period` bars?" (`1.0` = the
highest value of the window).

- **period** `int` (required) Must be `>= 2`.
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['rank:254']             # percentile of today's close within the last year
df['rank:20 > 0.95']       # near the top of its 20-bar range (bool signal)
```

### <a id="indicator-skew"></a>`skew`, Rolling Skewness

```
skew:<period>@<series>
```

The bias-corrected sample skewness of the trailing `period` values.

- **period** `int` (required) Must be `>= 3` (skewness is undefined below 3 samples).
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['skew:60@(change)']     # skew of returns over the last 60 bars
```

### <a id="indicator-kurt"></a>`kurt`, Rolling Kurtosis

```
kurt:<period>@<series>
```

The bias-corrected **excess** kurtosis of the trailing `period` values
(computed with two-pass central moments — numerically exact even when the
window mean dwarfs its spread).

- **period** `int` (required) Must be `>= 4` (kurtosis is undefined below 4 samples).
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['kurt:60@(change)']     # tail-heaviness of recent returns
```

### <a id="indicator-sem"></a>`sem`, Rolling Standard Error of the Mean

```
sem:<period>@<series>
```

The standard error of the mean (`std / sqrt(count)`, sample `ddof=1`) of the
trailing `period` values.

- **period** `int` (required) Must be `>= 2`.
- **series?** `str='close'` Which column or directive the calculation is based on.

```py
df['sem:20']               # uncertainty of the 20-bar mean
```

## TradingView-compatible directives

These directives mirror TradingView **Pine Script** `ta.*` indicators that have
no TA-Lib equivalent. Each is validated against the canonical Pine formula
(and, where the convention agrees, against `pandas-ta`). Argument notation is
the same as the [TA-Lib-compatible](#ta-lib-compatible-directives) table:
`<name=value>` is an optional argument with its default; a `<name>` **without
an equals sign is required**.

### <a id="indicator-vwma"></a>`vwma`, Volume-Weighted Moving Average

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

### <a id="indicator-alma"></a>`alma`, Arnaud Legoux Moving Average

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

### <a id="indicator-hma"></a>`hma`, Hull Moving Average

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

### <a id="indicator-swma"></a>`swma`, Symmetrically-Weighted Moving Average

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

### <a id="indicator-cog"></a>`cog`, Center of Gravity

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

### <a id="indicator-dev"></a>`dev`, Mean Absolute Deviation

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

### <a id="indicator-rci"></a>`rci`, Rank Correlation Index

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

### <a id="indicator-iii"></a>`iii`, Intraday Intensity Index

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

### <a id="indicator-kcw"></a>`kcw`, Keltner Channel Width

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

### <a id="indicator-mode"></a>`mode`, Rolling Mode

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

### <a id="indicator-pivothigh"></a><a id="indicator-pivotlow"></a>`pivothigh` / `pivotlow`, Fractal Pivots

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

## TA-Lib-compatible directives

TA-Lib-related directives use lowercase Volas names, but the `TA-Lib original`
column below lists the upstream TA-Lib function they correspond to. Arguments
before `@` are positional scalars; input series after `@` are column names (or
nested directives in parentheses) overriding the default columns. Empty
argument slots keep earlier defaults, so `macd.signal:,,5` means fast period
`12`, slow period `26`, and signal period `5`.

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

The `Parameters` column uses this notation:

```
:<min=2>,<max=30>,<matype=0>@<series=close>,<series_periods>
│  │                         │              │
│  │                         │              └ no `=` sign: the input is REQUIRED
│  │                         └ input series with its default column — may be omitted
│  └ scalar argument with its default value — may be omitted (or left empty)
└ `:` introduces the scalar arguments; `@` introduces the input series
```

Every `<name=value>` argument has a default and can be omitted; a `<name>`
**without an equals sign is required**.

| Volas directive | TA-Lib original | Meaning | Parameters |
| --- | --- | --- | --- |
| <a id="indicator-ma"></a>`ma` | `MA` | Generic moving average selected by MA type. | `:<period>,<matype=0>@<series=close>` |
| <a id="indicator-ema"></a>`ema` | `EMA` | Exponential moving average. | `:<period>@<series=close>` |
| <a id="indicator-wma"></a>`wma` | `WMA` | Weighted moving average. | `:<period>@<series=close>` |
| <a id="indicator-dema"></a>`dema` | `DEMA` | Double exponential moving average. | `:<period>@<series=close>` |
| <a id="indicator-tema"></a>`tema` | `TEMA` | Triple exponential moving average. | `:<period>@<series=close>` |
| <a id="indicator-trima"></a>`trima` | `TRIMA` | Triangular moving average. | `:<period>@<series=close>` |
| <a id="indicator-kama"></a>`kama` | `KAMA` | Kaufman adaptive moving average. | `:<period>@<series=close>` |
| <a id="indicator-t3"></a>`t3` | `T3` | T3 moving average. | `:<period>,<vfactor=0.7>@<series=close>` |
| <a id="indicator-mama"></a>`mama` | `MAMA` | MESA adaptive moving average main line. | `:<fast_limit=0.5>,<slow_limit=0.05>@<series=close>` |
| <a id="indicator-mama-fama"></a>`mama.fama` | `MAMA` | Following adaptive moving average line. | `:<fast_limit=0.5>,<slow_limit=0.05>@<series=close>` |
| <a id="indicator-mavp"></a>`mavp` | `MAVP` | Moving average with per-row variable periods; the REQUIRED second input series supplies each row's period (clamped to `[min, max]`). | `:<min>,<max>,<matype=0>@<series_close=close>,<series_periods>` |
| <a id="indicator-sar"></a>`sar` | `SAR` | Parabolic SAR. | `:<acceleration=0.02>,<maximum=0.2>@<series_high=high>,<series_low=low>` |
| <a id="indicator-sarext"></a>`sarext` | `SAREXT` | Extended Parabolic SAR. | `:<start=0>,<offset=0>,<long_init=0.02>,<long_step=0.02>,<long_max=0.2>,<short_init=0.02>,<short_step=0.02>,<short_max=0.2>@<series_high=high>,<series_low=low>` |
| <a id="indicator-boll"></a>`boll` | `BBANDS` | Bollinger middle band. | `:<period=20>@<series=close>` |
| <a id="indicator-boll-upper"></a>`boll.upper` | `BBANDS` | Bollinger upper band. | `:<period=20>,<times=2>@<series=close>` |
| <a id="indicator-boll-lower"></a>`boll.lower` | `BBANDS` | Bollinger lower band. | `:<period=20>,<times=2>@<series=close>` |
| <a id="indicator-accbands"></a>`accbands` | `ACCBANDS` | Acceleration Bands middle line. | `:<period=20>@<series=close>` |
| <a id="indicator-accbands-upper"></a>`accbands.upper` | `ACCBANDS` | Acceleration Bands upper line. | `:<period=20>@<series_high=high>,<series_low=low>` |
| <a id="indicator-accbands-lower"></a>`accbands.lower` | `ACCBANDS` | Acceleration Bands lower line. | `:<period=20>@<series_high=high>,<series_low=low>` |
| <a id="indicator-midpoint"></a>`midpoint` | `MIDPOINT` | Midpoint over a rolling period. | `:<period>@<series=close>` |
| <a id="indicator-midprice"></a>`midprice` | `MIDPRICE` | Midpoint price over high and low. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-ht_trendline"></a>`ht_trendline` | `HT_TRENDLINE` | Hilbert Transform instantaneous trendline. | `@<series=close>` |
| <a id="indicator-macd"></a>`macd` | `MACD` | MACD line; Volas uses standalone EMA fast minus EMA slow. | `:<fast=12>,<slow=26>@<series=close>` |
| <a id="indicator-macd-signal"></a>`macd.signal` | `MACD` | Signal line of the Volas MACD line. | `:<fast=12>,<slow=26>,<signal=9>@<series=close>` |
| <a id="indicator-macd-histogram"></a>`macd.histogram` | `MACD` | MACD histogram: line minus signal. | `:<fast=12>,<slow=26>,<signal=9>@<series=close>` |
| <a id="indicator-macdext"></a>`macdext` | `MACDEXT` | MACD line with independent MA types. | `:<fast=12>,<fast_matype=0>,<slow=26>,<slow_matype=0>@<series=close>` |
| <a id="indicator-macdext-signal"></a>`macdext.signal` | `MACDEXT` | MACDEXT signal line. | `:<fast=12>,<fast_matype=0>,<slow=26>,<slow_matype=0>,<signal=9>,<signal_matype=0>@<series=close>` |
| <a id="indicator-macdext-histogram"></a>`macdext.histogram` | `MACDEXT` | MACDEXT histogram. | `:<fast=12>,<fast_matype=0>,<slow=26>,<slow_matype=0>,<signal=9>,<signal_matype=0>@<series=close>` |
| <a id="indicator-macdfix"></a>`macdfix` | `MACDFIX` | Fixed 12/26 MACD line; Volas uses standalone EMA fast minus EMA slow. | `@<series=close>` |
| <a id="indicator-macdfix-signal"></a>`macdfix.signal` | `MACDFIX` | Signal line of the Volas fixed 12/26 MACD line. | `:<signal=9>@<series=close>` |
| <a id="indicator-macdfix-histogram"></a>`macdfix.histogram` | `MACDFIX` | Histogram of the Volas fixed 12/26 MACD line. | `:<signal=9>@<series=close>` |
| <a id="indicator-apo"></a>`apo` | `APO` | Absolute price oscillator. | `:<fast=12>,<slow=26>,<matype=0>@<series=close>` |
| <a id="indicator-ppo"></a>`ppo` | `PPO` | Percentage price oscillator. | `:<fast=12>,<slow=26>,<matype=0>@<series=close>` |
| <a id="indicator-rsi"></a>`rsi` | `RSI` | Relative Strength Index. | `:<period>@<series=close>` |
| <a id="indicator-cmo"></a>`cmo` | `CMO` | Chande Momentum Oscillator. | `:<period>@<series=close>` |
| <a id="indicator-cci"></a>`cci` | `CCI` | Commodity Channel Index. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-imi"></a>`imi` | `IMI` | Intraday Momentum Index. | `:<period>@<series_open=open>,<series_close=close>` |
| <a id="indicator-mfi"></a>`mfi` | `MFI` | Money Flow Index. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>,<series_volume=volume>` |
| <a id="indicator-bop"></a>`bop` | `BOP` | Balance of Power. | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-willr"></a>`willr` | `WILLR` | Williams Percent Range. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-mom"></a>`mom` | `MOM` | Momentum. | `:<period>@<series=close>` |
| <a id="indicator-roc"></a>`roc` | `ROC` | Rate of change. | `:<period>@<series=close>` |
| <a id="indicator-rocp"></a>`rocp` | `ROCP` | Rate of change percentage. | `:<period>@<series=close>` |
| <a id="indicator-rocr"></a>`rocr` | `ROCR` | Rate of change ratio. | `:<period>@<series=close>` |
| <a id="indicator-rocr100"></a>`rocr100` | `ROCR100` | Rate of change ratio multiplied by 100. | `:<period>@<series=close>` |
| <a id="indicator-stoch-k"></a>`stoch.k` | `STOCH` | Slow stochastic percent K. | `:<fastk=5>,<slowk=3>,<slowk_matype=0>,<slowd=3>,<slowd_matype=0>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-stoch-d"></a>`stoch.d` | `STOCH` | Slow stochastic percent D. | `:<fastk=5>,<slowk=3>,<slowk_matype=0>,<slowd=3>,<slowd_matype=0>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-stochf-k"></a>`stochf.k` | `STOCHF` | Fast stochastic percent K. | `:<fastk=5>,<fastd=3>,<fastd_matype=0>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-stochf-d"></a>`stochf.d` | `STOCHF` | Fast stochastic percent D. | `:<fastk=5>,<fastd=3>,<fastd_matype=0>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-stochrsi-k"></a>`stochrsi.k` | `STOCHRSI` | Fast stochastic RSI percent K. | `:<rsi=14>,<fastk=5>,<fastd=3>,<fastd_matype=0>@<series=close>` |
| <a id="indicator-stochrsi-d"></a>`stochrsi.d` | `STOCHRSI` | Fast stochastic RSI percent D. | `:<rsi=14>,<fastk=5>,<fastd=3>,<fastd_matype=0>@<series=close>` |
| <a id="indicator-trix"></a>`trix` | `TRIX` | One-period ROC of a triple EMA. | `:<period>@<series=close>` |
| <a id="indicator-ultosc"></a>`ultosc` | `ULTOSC` | Ultimate Oscillator. | `:<short=7>,<medium=14>,<long=28>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-aroon-up"></a>`aroon.up` | `AROON` | Aroon up line. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-aroon-down"></a>`aroon.down` | `AROON` | Aroon down line. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-aroonosc"></a>`aroonosc` | `AROONOSC` | Aroon oscillator. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-plus_dm"></a>`plus_dm` | `PLUS_DM` | Plus directional movement. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-minus_dm"></a>`minus_dm` | `MINUS_DM` | Minus directional movement. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-plus_di"></a>`plus_di` | `PLUS_DI` | Plus directional indicator. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-minus_di"></a>`minus_di` | `MINUS_DI` | Minus directional indicator. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-dx"></a>`dx` | `DX` | Directional Movement Index. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-adx"></a>`adx` | `ADX` | Average Directional Movement Index. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-adxr"></a>`adxr` | `ADXR` | Average Directional Movement Index Rating. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-obv"></a>`obv` | `OBV` | On-Balance Volume. | `@<series_close=close>,<series_volume=volume>` |
| <a id="indicator-ad"></a>`ad` | `AD` | Chaikin Accumulation Distribution line. | `@<series_high=high>,<series_low=low>,<series_close=close>,<series_volume=volume>` |
| <a id="indicator-adosc"></a>`adosc` | `ADOSC` | Chaikin Accumulation Distribution oscillator. | `:<fast=3>,<slow=10>@<series_high=high>,<series_low=low>,<series_close=close>,<series_volume=volume>` |
| <a id="indicator-tr"></a>`tr` | `TRANGE` | True Range. | `@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-atr"></a>`atr` | `ATR` | Average True Range. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-natr"></a>`natr` | `NATR` | Normalized Average True Range. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-avgprice"></a>`avgprice` | `AVGPRICE` | Average price. | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-medprice"></a>`medprice` | `MEDPRICE` | Median price. | `@<series_high=high>,<series_low=low>` |
| <a id="indicator-typprice"></a>`typprice` | `TYPPRICE` | Typical price. | `@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-wclprice"></a>`wclprice` | `WCLPRICE` | Weighted close price. | `@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-ht_dcperiod"></a>`ht_dcperiod` | `HT_DCPERIOD` | Hilbert Transform dominant cycle period. | `@<series=close>` |
| <a id="indicator-ht_dcphase"></a>`ht_dcphase` | `HT_DCPHASE` | Hilbert Transform dominant cycle phase. | `@<series=close>` |
| <a id="indicator-ht_phasor"></a>`ht_phasor` | `HT_PHASOR` | Hilbert Transform phasor in-phase line. | `@<series=close>` |
| <a id="indicator-ht_phasor-quadrature"></a>`ht_phasor.quadrature` | `HT_PHASOR` | Hilbert Transform phasor quadrature line. | `@<series=close>` |
| <a id="indicator-ht_sine"></a>`ht_sine` | `HT_SINE` | Hilbert Transform sine wave. | `@<series=close>` |
| <a id="indicator-ht_sine-leadsine"></a>`ht_sine.leadsine` | `HT_SINE` | Hilbert Transform lead sine wave. | `@<series=close>` |
| <a id="indicator-ht_trendmode"></a>`ht_trendmode` | `HT_TRENDMODE` | Hilbert Transform trend versus cycle mode. | `@<series=close>` |
| <a id="indicator-linearreg"></a>`linearreg` | `LINEARREG` | Linear regression value. | `:<period>@<series=close>` |
| <a id="indicator-linearreg_slope"></a>`linearreg_slope` | `LINEARREG_SLOPE` | Linear regression slope. | `:<period>@<series=close>` |
| <a id="indicator-linearreg_intercept"></a>`linearreg_intercept` | `LINEARREG_INTERCEPT` | Linear regression intercept. | `:<period>@<series=close>` |
| <a id="indicator-linearreg_angle"></a>`linearreg_angle` | `LINEARREG_ANGLE` | Linear regression angle. | `:<period>@<series=close>` |
| <a id="indicator-tsf"></a>`tsf` | `TSF` | Time Series Forecast. | `:<period>@<series=close>` |
| <a id="indicator-var"></a>`var` | `VAR` | Variance. | `:<period>@<series=close>` |
| <a id="indicator-stddev"></a>`stddev` | `STDDEV` | Standard deviation. | `:<period>,<nbdev>@<series=close>` |
| <a id="indicator-correl"></a>`correl` | `CORREL` | Pearson correlation coefficient. | `:<period>@<series=close>,<series_other>` |
| <a id="indicator-beta"></a>`beta` | `BETA` | Beta. | `:<period>@<series=close>,<series_other>` |
| <a id="indicator-sum"></a>`sum` | `SUM` | Rolling sum. | `:<period>@<series=close>` |
| <a id="indicator-maxindex"></a>`maxindex` | `MAXINDEX` | Index of the rolling maximum. | `:<period>@<series=close>` |
| <a id="indicator-minindex"></a>`minindex` | `MININDEX` | Index of the rolling minimum. | `:<period>@<series=close>` |
| <a id="indicator-minmax-min"></a>`minmax.min` | `MINMAX` | Rolling minimum from the MINMAX pair. | `:<period>@<series=close>` |
| <a id="indicator-minmax-max"></a>`minmax.max` | `MINMAX` | Rolling maximum from the MINMAX pair. | `:<period>@<series=close>` |
| <a id="indicator-minmaxindex-min"></a>`minmaxindex.min` | `MINMAXINDEX` | Index of the rolling minimum from the pair. | `:<period>@<series=close>` |
| <a id="indicator-minmaxindex-max"></a>`minmaxindex.max` | `MINMAXINDEX` | Index of the rolling maximum from the pair. | `:<period>@<series=close>` |
| <a id="indicator-cdl-2crows"></a>`cdl.2crows` | `CDL2CROWS` | Two Crows | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-3blackcrows"></a>`cdl.3blackcrows` | `CDL3BLACKCROWS` | Three Black Crows | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-3inside"></a>`cdl.3inside` | `CDL3INSIDE` | Three Inside Up/Down | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-3linestrike"></a>`cdl.3linestrike` | `CDL3LINESTRIKE` | Three-Line Strike  | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-3outside"></a>`cdl.3outside` | `CDL3OUTSIDE` | Three Outside Up/Down | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-3starsinsouth"></a>`cdl.3starsinsouth` | `CDL3STARSINSOUTH` | Three Stars In The South | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-3whitesoldiers"></a>`cdl.3whitesoldiers` | `CDL3WHITESOLDIERS` | Three Advancing White Soldiers | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-abandonedbaby"></a>`cdl.abandonedbaby` | `CDLABANDONEDBABY` | Abandoned Baby | `:<penetration=0.3>@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-advanceblock"></a>`cdl.advanceblock` | `CDLADVANCEBLOCK` | Advance Block | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-belthold"></a>`cdl.belthold` | `CDLBELTHOLD` | Belt-hold | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-breakaway"></a>`cdl.breakaway` | `CDLBREAKAWAY` | Breakaway | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-closingmarubozu"></a>`cdl.closingmarubozu` | `CDLCLOSINGMARUBOZU` | Closing Marubozu | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-concealbabyswall"></a>`cdl.concealbabyswall` | `CDLCONCEALBABYSWALL` | Concealing Baby Swallow | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-counterattack"></a>`cdl.counterattack` | `CDLCOUNTERATTACK` | Counterattack | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-darkcloudcover"></a>`cdl.darkcloudcover` | `CDLDARKCLOUDCOVER` | Dark Cloud Cover | `:<penetration=0.5>@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-doji"></a>`cdl.doji` | `CDLDOJI` | Doji | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-dojistar"></a>`cdl.dojistar` | `CDLDOJISTAR` | Doji Star | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-dragonflydoji"></a>`cdl.dragonflydoji` | `CDLDRAGONFLYDOJI` | Dragonfly Doji | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-engulfing"></a>`cdl.engulfing` | `CDLENGULFING` | Engulfing Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-eveningdojistar"></a>`cdl.eveningdojistar` | `CDLEVENINGDOJISTAR` | Evening Doji Star | `:<penetration=0.3>@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-eveningstar"></a>`cdl.eveningstar` | `CDLEVENINGSTAR` | Evening Star | `:<penetration=0.3>@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-gapsidesidewhite"></a>`cdl.gapsidesidewhite` | `CDLGAPSIDESIDEWHITE` | Up/Down-gap side-by-side white lines | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-gravestonedoji"></a>`cdl.gravestonedoji` | `CDLGRAVESTONEDOJI` | Gravestone Doji | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-hammer"></a>`cdl.hammer` | `CDLHAMMER` | Hammer | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-hangingman"></a>`cdl.hangingman` | `CDLHANGINGMAN` | Hanging Man | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-harami"></a>`cdl.harami` | `CDLHARAMI` | Harami Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-haramicross"></a>`cdl.haramicross` | `CDLHARAMICROSS` | Harami Cross Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-highwave"></a>`cdl.highwave` | `CDLHIGHWAVE` | High-Wave Candle | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-hikkake"></a>`cdl.hikkake` | `CDLHIKKAKE` | Hikkake Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-hikkakemod"></a>`cdl.hikkakemod` | `CDLHIKKAKEMOD` | Modified Hikkake Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-homingpigeon"></a>`cdl.homingpigeon` | `CDLHOMINGPIGEON` | Homing Pigeon | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-identical3crows"></a>`cdl.identical3crows` | `CDLIDENTICAL3CROWS` | Identical Three Crows | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-inneck"></a>`cdl.inneck` | `CDLINNECK` | In-Neck Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-invertedhammer"></a>`cdl.invertedhammer` | `CDLINVERTEDHAMMER` | Inverted Hammer | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-kicking"></a>`cdl.kicking` | `CDLKICKING` | Kicking | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-kickingbylength"></a>`cdl.kickingbylength` | `CDLKICKINGBYLENGTH` | Kicking - bull/bear determined by the longer marubozu | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-ladderbottom"></a>`cdl.ladderbottom` | `CDLLADDERBOTTOM` | Ladder Bottom | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-longleggeddoji"></a>`cdl.longleggeddoji` | `CDLLONGLEGGEDDOJI` | Long Legged Doji | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-longline"></a>`cdl.longline` | `CDLLONGLINE` | Long Line Candle | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-marubozu"></a>`cdl.marubozu` | `CDLMARUBOZU` | Marubozu | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-matchinglow"></a>`cdl.matchinglow` | `CDLMATCHINGLOW` | Matching Low | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-mathold"></a>`cdl.mathold` | `CDLMATHOLD` | Mat Hold | `:<penetration=0.5>@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-morningdojistar"></a>`cdl.morningdojistar` | `CDLMORNINGDOJISTAR` | Morning Doji Star | `:<penetration=0.3>@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-morningstar"></a>`cdl.morningstar` | `CDLMORNINGSTAR` | Morning Star | `:<penetration=0.3>@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-onneck"></a>`cdl.onneck` | `CDLONNECK` | On-Neck Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-piercing"></a>`cdl.piercing` | `CDLPIERCING` | Piercing Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-rickshawman"></a>`cdl.rickshawman` | `CDLRICKSHAWMAN` | Rickshaw Man | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-risefall3methods"></a>`cdl.risefall3methods` | `CDLRISEFALL3METHODS` | Rising/Falling Three Methods | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-separatinglines"></a>`cdl.separatinglines` | `CDLSEPARATINGLINES` | Separating Lines | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-shootingstar"></a>`cdl.shootingstar` | `CDLSHOOTINGSTAR` | Shooting Star | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-shortline"></a>`cdl.shortline` | `CDLSHORTLINE` | Short Line Candle | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-spinningtop"></a>`cdl.spinningtop` | `CDLSPINNINGTOP` | Spinning Top | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-stalledpattern"></a>`cdl.stalledpattern` | `CDLSTALLEDPATTERN` | Stalled Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-sticksandwich"></a>`cdl.sticksandwich` | `CDLSTICKSANDWICH` | Stick Sandwich | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-takuri"></a>`cdl.takuri` | `CDLTAKURI` | Takuri (Dragonfly Doji with very long lower shadow) | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-tasukigap"></a>`cdl.tasukigap` | `CDLTASUKIGAP` | Tasuki Gap | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-thrusting"></a>`cdl.thrusting` | `CDLTHRUSTING` | Thrusting Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-tristar"></a>`cdl.tristar` | `CDLTRISTAR` | Tristar Pattern | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-unique3river"></a>`cdl.unique3river` | `CDLUNIQUE3RIVER` | Unique 3 River | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-upsidegap2crows"></a>`cdl.upsidegap2crows` | `CDLUPSIDEGAP2CROWS` | Upside Gap Two Crows | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-cdl-xsidegap3methods"></a>`cdl.xsidegap3methods` | `CDLXSIDEGAP3METHODS` | Upside/Downside Gap Three Methods | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
