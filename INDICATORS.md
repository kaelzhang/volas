# Built-in Indicators

This page is the complete directive reference for `volas` — **254** indicators
(each main command, every multi-output sub-command line, and each candlestick
pattern; counted by `scripts/count_indicators.py` from the Rust source).

## Indicator index

Directive names are case-insensitive. Alias notes list extra command or sub-command spellings accepted by the parser; multi-output indicators list each selectable output line.

- **Volas-exclusive indicators:** [`smma`](#user-content-indicator-smma), [`bbi`](#user-content-indicator-bbi), [`bbw`](#user-content-indicator-bbw), [`rsv`](#user-content-indicator-rsv), [KDJ](#user-content-indicator-kdj) (sub-commands: [`kdj.k`](#user-content-indicator-kdj), [`kdj.d`](#user-content-indicator-kdj), [`kdj.j`](#user-content-indicator-kdj); sub-commands only), [`llv`](#user-content-indicator-llv), [`hhv`](#user-content-indicator-hhv), [`donchian`](#user-content-indicator-donchian) (sub-commands: [`donchian.upper`](#user-content-indicator-donchian), [`donchian.lower`](#user-content-indicator-donchian); aliases: `.middle`/`.m`, `.u`, `.l`), [`hv`](#user-content-indicator-hv), [`psy`](#user-content-indicator-psy), [`dpo`](#user-content-indicator-dpo), [`tsi`](#user-content-indicator-tsi), [`kst`](#user-content-indicator-kst), [`crsi`](#user-content-indicator-crsi), [`chop`](#user-content-indicator-chop), [`cmf`](#user-content-indicator-cmf), [`emv`](#user-content-indicator-emv), [`efi`](#user-content-indicator-efi), [`pvt`](#user-content-indicator-pvt), [`nvi`](#user-content-indicator-nvi), [`pvi`](#user-content-indicator-pvi), [`mass_index`](#user-content-indicator-mass_index), [`bias`](#user-content-indicator-bias), [`dma`](#user-content-indicator-dma) (sub-commands: [`dma.ama`](#user-content-indicator-dma); aliases: `dma.ddd`), [Vortex](#user-content-indicator-vortex) (sub-commands: [`vortex.plus`](#user-content-indicator-vortex), [`vortex.minus`](#user-content-indicator-vortex); aliases: `.p`, `.m`; sub-commands only), [BRAR](#user-content-indicator-brar) (sub-commands: [`brar.ar`](#user-content-indicator-brar), [`brar.br`](#user-content-indicator-brar); sub-commands only), [`vr`](#user-content-indicator-vr), [`coppock`](#user-content-indicator-coppock), [`relative_vigor`](#user-content-indicator-relative_vigor) (sub-commands: [`relative_vigor.signal`](#user-content-indicator-relative_vigor)), [`dkx`](#user-content-indicator-dkx) (sub-commands: [`dkx.ma`](#user-content-indicator-dkx)), [`wvad`](#user-content-indicator-wvad), [`cdp`](#user-content-indicator-cdp) (sub-commands: [`cdp.ah`](#user-content-indicator-cdp), [`cdp.nh`](#user-content-indicator-cdp), [`cdp.nl`](#user-content-indicator-cdp), [`cdp.al`](#user-content-indicator-cdp)), [MIKE](#user-content-indicator-mike) (sub-commands: [`mike.weakr`](#user-content-indicator-mike), [`mike.midr`](#user-content-indicator-mike), [`mike.strongr`](#user-content-indicator-mike), [`mike.weaks`](#user-content-indicator-mike), [`mike.mids`](#user-content-indicator-mike), [`mike.strongs`](#user-content-indicator-mike); sub-commands only), [`keltner`](#user-content-indicator-keltner) (sub-commands: [`keltner.upper`](#user-content-indicator-keltner), [`keltner.lower`](#user-content-indicator-keltner); aliases: `.middle`/`.m`, `.u`, `.l`), [`stoch_momentum`](#user-content-indicator-stoch_momentum) (sub-commands: [`stoch_momentum.signal`](#user-content-indicator-stoch_momentum)), [`ttm_squeeze`](#user-content-indicator-ttm_squeeze) (sub-commands: [`ttm_squeeze.on`](#user-content-indicator-ttm_squeeze)), [`pivot_points`](#user-content-indicator-pivot_points) (sub-commands: [`pivot_points.r1`](#user-content-indicator-pivot_points), [`pivot_points.r2`](#user-content-indicator-pivot_points), [`pivot_points.r3`](#user-content-indicator-pivot_points), [`pivot_points.s1`](#user-content-indicator-pivot_points), [`pivot_points.s2`](#user-content-indicator-pivot_points), [`pivot_points.s3`](#user-content-indicator-pivot_points); aliases: `.p`/`.pp`), [Ichimoku](#user-content-indicator-ichimoku) (sub-commands: [`ichimoku.tenkan`](#user-content-indicator-ichimoku), [`ichimoku.kijun`](#user-content-indicator-ichimoku), [`ichimoku.senkou_a`](#user-content-indicator-ichimoku), [`ichimoku.senkou_b`](#user-content-indicator-ichimoku), [`ichimoku.chikou`](#user-content-indicator-ichimoku); aliases: `.conversion`, `.base`, `.span_a`, `.span_b`, `.lagging`; sub-commands only), [`wad`](#user-content-indicator-wad), [`asi`](#user-content-indicator-asi), [`supertrend`](#user-content-indicator-supertrend) (sub-commands: [`supertrend.direction`](#user-content-indicator-supertrend); aliases: `.line`, `.trend`, `.d`).
- **Built-in Commands for Statistics:** [`change`](#user-content-indicator-change), [`increase`](#user-content-indicator-increase), [Candle color](#user-content-indicator-style) (sub-commands: [`style.bullish`](#user-content-indicator-style), [`style.bearish`](#user-content-indicator-style); sub-commands only), [`repeat`](#user-content-indicator-repeat), [`median`](#user-content-indicator-median), [`quantile`](#user-content-indicator-quantile), [`rank`](#user-content-indicator-rank), [`skew`](#user-content-indicator-skew), [`kurt`](#user-content-indicator-kurt), [`sem`](#user-content-indicator-sem).
- **TradingView-compatible directives:** [`vwma`](docs/indicators-tradingview.md#user-content-indicator-vwma), [`alma`](docs/indicators-tradingview.md#user-content-indicator-alma), [`hma`](docs/indicators-tradingview.md#user-content-indicator-hma), [`swma`](docs/indicators-tradingview.md#user-content-indicator-swma), [`cog`](docs/indicators-tradingview.md#user-content-indicator-cog), [`dev`](docs/indicators-tradingview.md#user-content-indicator-dev), [`rci`](docs/indicators-tradingview.md#user-content-indicator-rci), [`iii`](docs/indicators-tradingview.md#user-content-indicator-iii), [`kcw`](docs/indicators-tradingview.md#user-content-indicator-kcw), [`mode`](docs/indicators-tradingview.md#user-content-indicator-mode), [`pivothigh`](docs/indicators-tradingview.md#user-content-indicator-pivothigh), [`pivotlow`](docs/indicators-tradingview.md#user-content-indicator-pivotlow).
- **TA-Lib-compatible directives:** [`ma`](docs/indicators-talib.md#user-content-indicator-ma), [`ema`](docs/indicators-talib.md#user-content-indicator-ema), [`wma`](docs/indicators-talib.md#user-content-indicator-wma), [`dema`](docs/indicators-talib.md#user-content-indicator-dema), [`tema`](docs/indicators-talib.md#user-content-indicator-tema), [`trima`](docs/indicators-talib.md#user-content-indicator-trima), [`kama`](docs/indicators-talib.md#user-content-indicator-kama), [`t3`](docs/indicators-talib.md#user-content-indicator-t3), [`mama`](docs/indicators-talib.md#user-content-indicator-mama) (sub-commands: [`mama.fama`](docs/indicators-talib.md#user-content-indicator-mama-fama); aliases: `.mama`), [`mavp`](docs/indicators-talib.md#user-content-indicator-mavp), [`sar`](docs/indicators-talib.md#user-content-indicator-sar), [`sarext`](docs/indicators-talib.md#user-content-indicator-sarext), [`boll`](docs/indicators-talib.md#user-content-indicator-boll) (sub-commands: [`boll.upper`](docs/indicators-talib.md#user-content-indicator-boll-upper), [`boll.lower`](docs/indicators-talib.md#user-content-indicator-boll-lower); aliases: `.middle`/`.m`, `.u`, `.l`), [`accbands`](docs/indicators-talib.md#user-content-indicator-accbands) (sub-commands: [`accbands.upper`](docs/indicators-talib.md#user-content-indicator-accbands-upper), [`accbands.lower`](docs/indicators-talib.md#user-content-indicator-accbands-lower); aliases: `.middle`/`.m`, `.u`, `.l`), [`midpoint`](docs/indicators-talib.md#user-content-indicator-midpoint), [`midprice`](docs/indicators-talib.md#user-content-indicator-midprice), [`ht_trendline`](docs/indicators-talib.md#user-content-indicator-ht_trendline), [`macd`](docs/indicators-talib.md#user-content-indicator-macd) (sub-commands: [`macd.signal`](docs/indicators-talib.md#user-content-indicator-macd-signal), [`macd.histogram`](docs/indicators-talib.md#user-content-indicator-macd-histogram); aliases: `.dif`, `.s`/`.dea`, `.h`/`.macd`), [`macdext`](docs/indicators-talib.md#user-content-indicator-macdext) (sub-commands: [`macdext.signal`](docs/indicators-talib.md#user-content-indicator-macdext-signal), [`macdext.histogram`](docs/indicators-talib.md#user-content-indicator-macdext-histogram); aliases: `.dif`, `.s`/`.dea`, `.h`/`.macd`), [`macdfix`](docs/indicators-talib.md#user-content-indicator-macdfix) (sub-commands: [`macdfix.signal`](docs/indicators-talib.md#user-content-indicator-macdfix-signal), [`macdfix.histogram`](docs/indicators-talib.md#user-content-indicator-macdfix-histogram); aliases: `.dif`, `.s`/`.dea`, `.h`/`.macd`), [`apo`](docs/indicators-talib.md#user-content-indicator-apo), [`ppo`](docs/indicators-talib.md#user-content-indicator-ppo), [`rsi`](docs/indicators-talib.md#user-content-indicator-rsi), [`cmo`](docs/indicators-talib.md#user-content-indicator-cmo), [`cci`](docs/indicators-talib.md#user-content-indicator-cci), [`imi`](docs/indicators-talib.md#user-content-indicator-imi), [`mfi`](docs/indicators-talib.md#user-content-indicator-mfi), [`bop`](docs/indicators-talib.md#user-content-indicator-bop), [`willr`](docs/indicators-talib.md#user-content-indicator-willr), [`mom`](docs/indicators-talib.md#user-content-indicator-mom), [`roc`](docs/indicators-talib.md#user-content-indicator-roc), [`rocp`](docs/indicators-talib.md#user-content-indicator-rocp).
- **TA-Lib-compatible directives, continued:** [`rocr`](docs/indicators-talib.md#user-content-indicator-rocr), [`rocr100`](docs/indicators-talib.md#user-content-indicator-rocr100), [STOCH](docs/indicators-talib.md#user-content-indicator-stoch-k) (sub-commands: [`stoch.k`](docs/indicators-talib.md#user-content-indicator-stoch-k), [`stoch.d`](docs/indicators-talib.md#user-content-indicator-stoch-d); aliases: `.slowk`, `.slowd`; sub-commands only), [STOCHF](docs/indicators-talib.md#user-content-indicator-stochf-k) (sub-commands: [`stochf.k`](docs/indicators-talib.md#user-content-indicator-stochf-k), [`stochf.d`](docs/indicators-talib.md#user-content-indicator-stochf-d); aliases: `.fastk`, `.fastd`; sub-commands only), [STOCHRSI](docs/indicators-talib.md#user-content-indicator-stochrsi-k) (sub-commands: [`stochrsi.k`](docs/indicators-talib.md#user-content-indicator-stochrsi-k), [`stochrsi.d`](docs/indicators-talib.md#user-content-indicator-stochrsi-d); aliases: `.fastk`, `.fastd`; sub-commands only), [`trix`](docs/indicators-talib.md#user-content-indicator-trix), [`ultosc`](docs/indicators-talib.md#user-content-indicator-ultosc), [AROON](docs/indicators-talib.md#user-content-indicator-aroon-up) (sub-commands: [`aroon.up`](docs/indicators-talib.md#user-content-indicator-aroon-up), [`aroon.down`](docs/indicators-talib.md#user-content-indicator-aroon-down); aliases: `.u`, `.d`; sub-commands only), [`aroonosc`](docs/indicators-talib.md#user-content-indicator-aroonosc), [`plus_dm`](docs/indicators-talib.md#user-content-indicator-plus_dm), [`minus_dm`](docs/indicators-talib.md#user-content-indicator-minus_dm), [`plus_di`](docs/indicators-talib.md#user-content-indicator-plus_di), [`minus_di`](docs/indicators-talib.md#user-content-indicator-minus_di), [`dx`](docs/indicators-talib.md#user-content-indicator-dx), [`adx`](docs/indicators-talib.md#user-content-indicator-adx), [`adxr`](docs/indicators-talib.md#user-content-indicator-adxr), [`obv`](docs/indicators-talib.md#user-content-indicator-obv), [`ad`](docs/indicators-talib.md#user-content-indicator-ad), [`adosc`](docs/indicators-talib.md#user-content-indicator-adosc), [`tr`](docs/indicators-talib.md#user-content-indicator-tr), [`atr`](docs/indicators-talib.md#user-content-indicator-atr), [`natr`](docs/indicators-talib.md#user-content-indicator-natr), [`avgprice`](docs/indicators-talib.md#user-content-indicator-avgprice), [`medprice`](docs/indicators-talib.md#user-content-indicator-medprice), [`typprice`](docs/indicators-talib.md#user-content-indicator-typprice), [`wclprice`](docs/indicators-talib.md#user-content-indicator-wclprice), [`ht_dcperiod`](docs/indicators-talib.md#user-content-indicator-ht_dcperiod), [`ht_dcphase`](docs/indicators-talib.md#user-content-indicator-ht_dcphase), [`ht_phasor`](docs/indicators-talib.md#user-content-indicator-ht_phasor) (sub-commands: [`ht_phasor.quadrature`](docs/indicators-talib.md#user-content-indicator-ht_phasor-quadrature); aliases: `.i`/`.inphase`, `.q`/`.quad`), [`ht_sine`](docs/indicators-talib.md#user-content-indicator-ht_sine) (sub-commands: [`ht_sine.leadsine`](docs/indicators-talib.md#user-content-indicator-ht_sine-leadsine); aliases: `.sine`, `.lead`), [`ht_trendmode`](docs/indicators-talib.md#user-content-indicator-ht_trendmode), [`linearreg`](docs/indicators-talib.md#user-content-indicator-linearreg), [`linearreg_slope`](docs/indicators-talib.md#user-content-indicator-linearreg_slope), [`linearreg_intercept`](docs/indicators-talib.md#user-content-indicator-linearreg_intercept), [`linearreg_angle`](docs/indicators-talib.md#user-content-indicator-linearreg_angle), [`tsf`](docs/indicators-talib.md#user-content-indicator-tsf), [`var`](docs/indicators-talib.md#user-content-indicator-var), [`stddev`](docs/indicators-talib.md#user-content-indicator-stddev), [`correl`](docs/indicators-talib.md#user-content-indicator-correl), [`beta`](docs/indicators-talib.md#user-content-indicator-beta), [`sum`](docs/indicators-talib.md#user-content-indicator-sum), [`maxindex`](docs/indicators-talib.md#user-content-indicator-maxindex), [`minindex`](docs/indicators-talib.md#user-content-indicator-minindex), [MINMAX](docs/indicators-talib.md#user-content-indicator-minmax-min) (sub-commands: [`minmax.min`](docs/indicators-talib.md#user-content-indicator-minmax-min), [`minmax.max`](docs/indicators-talib.md#user-content-indicator-minmax-max); sub-commands only), [MINMAXINDEX](docs/indicators-talib.md#user-content-indicator-minmaxindex-min) (sub-commands: [`minmaxindex.min`](docs/indicators-talib.md#user-content-indicator-minmaxindex-min), [`minmaxindex.max`](docs/indicators-talib.md#user-content-indicator-minmaxindex-max); sub-commands only).
- **Candlestick pattern directives:** every `cdl.<pattern>` entry also accepts `style.<pattern>` as an alias: [`cdl.2crows`](docs/indicators-talib.md#user-content-indicator-cdl-2crows), [`cdl.3blackcrows`](docs/indicators-talib.md#user-content-indicator-cdl-3blackcrows), [`cdl.3inside`](docs/indicators-talib.md#user-content-indicator-cdl-3inside), [`cdl.3linestrike`](docs/indicators-talib.md#user-content-indicator-cdl-3linestrike), [`cdl.3outside`](docs/indicators-talib.md#user-content-indicator-cdl-3outside), [`cdl.3starsinsouth`](docs/indicators-talib.md#user-content-indicator-cdl-3starsinsouth), [`cdl.3whitesoldiers`](docs/indicators-talib.md#user-content-indicator-cdl-3whitesoldiers), [`cdl.abandonedbaby`](docs/indicators-talib.md#user-content-indicator-cdl-abandonedbaby), [`cdl.advanceblock`](docs/indicators-talib.md#user-content-indicator-cdl-advanceblock), [`cdl.belthold`](docs/indicators-talib.md#user-content-indicator-cdl-belthold), [`cdl.breakaway`](docs/indicators-talib.md#user-content-indicator-cdl-breakaway), [`cdl.closingmarubozu`](docs/indicators-talib.md#user-content-indicator-cdl-closingmarubozu), [`cdl.concealbabyswall`](docs/indicators-talib.md#user-content-indicator-cdl-concealbabyswall), [`cdl.counterattack`](docs/indicators-talib.md#user-content-indicator-cdl-counterattack), [`cdl.darkcloudcover`](docs/indicators-talib.md#user-content-indicator-cdl-darkcloudcover), [`cdl.doji`](docs/indicators-talib.md#user-content-indicator-cdl-doji), [`cdl.dojistar`](docs/indicators-talib.md#user-content-indicator-cdl-dojistar), [`cdl.dragonflydoji`](docs/indicators-talib.md#user-content-indicator-cdl-dragonflydoji), [`cdl.engulfing`](docs/indicators-talib.md#user-content-indicator-cdl-engulfing), [`cdl.eveningdojistar`](docs/indicators-talib.md#user-content-indicator-cdl-eveningdojistar), [`cdl.eveningstar`](docs/indicators-talib.md#user-content-indicator-cdl-eveningstar), [`cdl.gapsidesidewhite`](docs/indicators-talib.md#user-content-indicator-cdl-gapsidesidewhite), [`cdl.gravestonedoji`](docs/indicators-talib.md#user-content-indicator-cdl-gravestonedoji), [`cdl.hammer`](docs/indicators-talib.md#user-content-indicator-cdl-hammer), [`cdl.hangingman`](docs/indicators-talib.md#user-content-indicator-cdl-hangingman), [`cdl.harami`](docs/indicators-talib.md#user-content-indicator-cdl-harami), [`cdl.haramicross`](docs/indicators-talib.md#user-content-indicator-cdl-haramicross), [`cdl.highwave`](docs/indicators-talib.md#user-content-indicator-cdl-highwave), [`cdl.hikkake`](docs/indicators-talib.md#user-content-indicator-cdl-hikkake), [`cdl.hikkakemod`](docs/indicators-talib.md#user-content-indicator-cdl-hikkakemod), [`cdl.homingpigeon`](docs/indicators-talib.md#user-content-indicator-cdl-homingpigeon), [`cdl.identical3crows`](docs/indicators-talib.md#user-content-indicator-cdl-identical3crows), [`cdl.inneck`](docs/indicators-talib.md#user-content-indicator-cdl-inneck), [`cdl.invertedhammer`](docs/indicators-talib.md#user-content-indicator-cdl-invertedhammer), [`cdl.kicking`](docs/indicators-talib.md#user-content-indicator-cdl-kicking), [`cdl.kickingbylength`](docs/indicators-talib.md#user-content-indicator-cdl-kickingbylength), [`cdl.ladderbottom`](docs/indicators-talib.md#user-content-indicator-cdl-ladderbottom), [`cdl.longleggeddoji`](docs/indicators-talib.md#user-content-indicator-cdl-longleggeddoji), [`cdl.longline`](docs/indicators-talib.md#user-content-indicator-cdl-longline), [`cdl.marubozu`](docs/indicators-talib.md#user-content-indicator-cdl-marubozu), [`cdl.matchinglow`](docs/indicators-talib.md#user-content-indicator-cdl-matchinglow), [`cdl.mathold`](docs/indicators-talib.md#user-content-indicator-cdl-mathold), [`cdl.morningdojistar`](docs/indicators-talib.md#user-content-indicator-cdl-morningdojistar), [`cdl.morningstar`](docs/indicators-talib.md#user-content-indicator-cdl-morningstar), [`cdl.onneck`](docs/indicators-talib.md#user-content-indicator-cdl-onneck), [`cdl.piercing`](docs/indicators-talib.md#user-content-indicator-cdl-piercing), [`cdl.rickshawman`](docs/indicators-talib.md#user-content-indicator-cdl-rickshawman), [`cdl.risefall3methods`](docs/indicators-talib.md#user-content-indicator-cdl-risefall3methods), [`cdl.separatinglines`](docs/indicators-talib.md#user-content-indicator-cdl-separatinglines), [`cdl.shootingstar`](docs/indicators-talib.md#user-content-indicator-cdl-shootingstar), [`cdl.shortline`](docs/indicators-talib.md#user-content-indicator-cdl-shortline), [`cdl.spinningtop`](docs/indicators-talib.md#user-content-indicator-cdl-spinningtop), [`cdl.stalledpattern`](docs/indicators-talib.md#user-content-indicator-cdl-stalledpattern), [`cdl.sticksandwich`](docs/indicators-talib.md#user-content-indicator-cdl-sticksandwich), [`cdl.takuri`](docs/indicators-talib.md#user-content-indicator-cdl-takuri), [`cdl.tasukigap`](docs/indicators-talib.md#user-content-indicator-cdl-tasukigap), [`cdl.thrusting`](docs/indicators-talib.md#user-content-indicator-cdl-thrusting), [`cdl.tristar`](docs/indicators-talib.md#user-content-indicator-cdl-tristar), [`cdl.unique3river`](docs/indicators-talib.md#user-content-indicator-cdl-unique3river), [`cdl.upsidegap2crows`](docs/indicators-talib.md#user-content-indicator-cdl-upsidegap2crows), [`cdl.xsidegap3methods`](docs/indicators-talib.md#user-content-indicator-cdl-xsidegap3methods).

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
no standard at all. `df[d]` and `directive_lookback` reject a directive that omits a
required argument; `directive_stringify` accepts it, because it only names the
indicator. A trailing colon with nothing after it (`obv:`) means no arguments.

**Input data.** How an indicator treats a missing (NaN) or infinite input, and how its
sliding sums stay exact over any history, is one set of rules for every indicator — see
[Missing values and exact window sums](docs/indicator-inputs.md).

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

Calculates the raw stochastic value, which is often used to calculate KDJ:
`100·(close − LLV(low, period)) / (HHV(high, period) − LLV(low, period))`, `0` for a flat
range. The rows before the first full window are NaN (warm-up).

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

Before the first full `period_rsv` window, KDJ's RSV covers the bars available so far
(the 通达信 convention: `LLV` / `HHV` of fewer bars than the period reduce what there is),
so %K is seeded from real values from the very first bar. This differs from
stock-pandas, which feeds an RSV of `0` into the first `period_rsv − 1` bars.

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
hv:<period>,<time_frame>,<trading_days>,<session_minutes>@<on>
```

Gets annualized historical volatility: the sample standard deviation of one-bar
log returns over `period` bars, scaled to a year.

```
r_t = ln(close_t / close_{t-1})
hv  = stdev(r, over the last `period` bars, ddof = 1) × sqrt(bars_per_year)
```

`bars_per_year` depends on the unit of `time_frame`:

| `time_frame` | `bars_per_year` |
| --- | --- |
| sub-day bars: `1s`, `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h` | `trading_days × session_minutes / bar_minutes` |
| `1d` | `trading_days` |
| `3d` | `trading_days / 3` |
| `1w` | `52` |
| `1M` | `12` |
| `1y` | `1` |

- **period** `int` (required) The number of returns in the window; at least 2.
- **time_frame?** `str='1d'` The bar time frame. It must be one of the
  [`TimeFrame`](docs/cumulation.md#timeframe) labels. Upper-case aliases such as
  `1W` and `1D` are also accepted. Any other text is a `DirectiveValueError`.
- **trading_days?** `int=252` Trading days per year. It applies to sub-day and
  day bars only. 252 is the US equity convention (thinkorswim and the Cboe
  realized-volatility formula use it); use 365 for a market that trades every
  day.
- **session_minutes?** `int=1440` Minutes of regular trading per trading day,
  from 1 to 1440. It applies to sub-day bars only. The default 1440 fits a
  24-hour market. A sub-day bar longer than the session is a
  `DirectiveValueError`; for example, `hv:10,8h,252,240` raises. A bar whose
  length does not divide the session counts as a fraction of a bar per day. For
  example, `4h` bars on a 390-minute session give 1.625 bars per day.
- **on?** `str='close'` Which column or directive the calculation is based on.

An argument that the time frame does not use is never silently ignored. Passing
a `session_minutes` other than 1440 with a frame of a day or longer, or a
`trading_days` other than 252 with a frame of a week or longer, is a
`DirectiveValueError`. Examples are `hv:10,1d,252,390` and `hv:10,1w,365`. The
default values stay legal on every frame and drop out of the canonical column
name, so `hv:10,1d,252,1440` is cached as `hv:10`.

Second bars are exact: a `1s` bar counts 60 bars per minute of session. The
log returns are demeaned and divided by `n − 1`, as in thinkorswim's
`HistoricalVolatility`; TradingView's built-in HV uses the population form
and does not scale intraday bars by their length. A session-only intraday
series also includes the overnight gap in each session's first return, which
raises the result.

| Market | Regular session | `session_minutes` | 15-minute bars per day |
| --- | --- | --- | --- |
| US equities (NYSE) | 09:30–16:00 ET, 6.5 h | 390 | 26 |
| China A-shares (SSE, SZSE) | 09:30–11:30 and 13:00–15:00, 4 h | 240 | 16 |
| Hong Kong (HKEX) | 09:30–12:00 and 13:00–16:00, 5.5 h | 330 | 22 |
| CME E-mini S&P 500 futures | 18:00–17:00 ET, with a 1-hour daily break, 23 h | 1380 | 92 |
| Crypto | 24 h, every day | 1440 (the default) | 96 |

`bars_per_year` is `trading_days` times the bars per day. For example, US
equities on 15-minute bars give 252 × 26 = 6552, and crypto on 15-minute bars
with `trading_days = 365` gives 365 × 96 = 35040. Pass your exchange's own
trading-day count, which is the number of sessions in its yearly calendar.

```py
# Daily bars, 252 trading days (all defaults)
df['hv:10']

# US equities, 15-minute bars over the 390-minute regular session
df['hv:10,15m,252,390']

# China A-shares, 5-minute bars over the 240-minute session
df['hv:20,5m,252,240']

# CME E-mini futures, hourly bars over the 1380-minute session
df['hv:20,1h,252,1380']

# Crypto, hourly bars: 365 days, the default 24-hour session
df['hv:24,1h,365']

# Weekly and monthly bars annualize by 52 and 12
df['hv:10,1w']
df['hv:12,1M']
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
df['dpo:20']
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

## TradingView- and TA-Lib-compatible directives

The directives that mirror TradingView Pine `ta.*` indicators are documented in
[TradingView-compatible directives](docs/indicators-tradingview.md), and those
corresponding to TA-Lib functions in [TA-Lib-compatible directives](docs/indicators-talib.md).
