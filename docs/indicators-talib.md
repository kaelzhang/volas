# TA-Lib-compatible directives

The full indicator index is [INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md); the input rules every indicator follows are [Missing values and exact window sums](indicator-inputs.md).

TA-Lib-related directives use lowercase Volas names, but the `TA-Lib original`
column below lists the upstream TA-Lib function they correspond to. Arguments
before `@` are positional scalars; input series after `@` are column names (or
nested directives in parentheses) overriding the default columns. Empty
argument slots keep earlier defaults, so `macd.signal:,,5` means fast period
`12`, slow period `26`, and signal period `5`.

Every recurrence evaluates TA-Lib's C source in plain arithmetic, without fused
multiply-add: the same value on every platform, bit-identical to a TA-Lib build that does
not fuse, and within rounding of one that does (see
[Deliberate divergences](migration-talib.md#deliberate-divergences) for which functions
a fusing build changes).

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
| <a id="indicator-ma"></a>`ma` | `MA` | Generic moving average selected by MA type. A `period` of 1 is the input itself for every type (TA-Lib `TA_MA`), with no warm-up; the commands that smooth with a generic MA (`apo`, `ppo`, `macdext`, `mavp`, `stoch`, `stochf`, `stochrsi`) inherit it. | `:<period>,<matype=0>@<series=close>` |
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
| <a id="indicator-macd"></a>`macd` | `MACD` | MACD line; Volas uses standalone EMA fast minus EMA slow. As in TA-Lib, a `slow` shorter than `fast` swaps the two (`macd:26,12` is `macd:12,26`); the same holds for `apo`, `ppo` and `macdext` (with the MA types). | `:<fast=12>,<slow=26>@<series=close>` |
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
| <a id="indicator-rsi"></a>`rsi` | `RSI` | Relative Strength Index, TA-Lib's `100·avgGain/(avgGain+avgLoss)`. A flat window (`avgGain + avgLoss` exactly 0, a constant or all-falling series) gives 0 as in TA-Lib; TA-Lib's absolute `1e-14` zero test is not used, so a tiny-priced series keeps the value it has at any other scale (a deliberate divergence). | `:<period>@<series=close>` |
| <a id="indicator-cmo"></a>`cmo` | `CMO` | Chande Momentum Oscillator. A flat window gives 0, with RSI's exact zero test. | `:<period>@<series=close>` |
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
| <a id="indicator-ultosc"></a>`ultosc` | `ULTOSC` | Ultimate Oscillator. The 4/2/1 weights follow argument position (short, medium, long); TA-Lib sorts the three periods ascending first, so the two agree for in-order arguments and differ by design otherwise. | `:<short=7>,<medium=14>,<long=28>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-aroon-up"></a>`aroon.up` | `AROON` | Aroon up line. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-aroon-down"></a>`aroon.down` | `AROON` | Aroon down line. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-aroonosc"></a>`aroonosc` | `AROONOSC` | Aroon oscillator. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-plus_dm"></a>`plus_dm` | `PLUS_DM` | Plus directional movement. At `period=1`, the raw one-bar movement from bar 1 (lookback 1), as in TA-Lib. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-minus_dm"></a>`minus_dm` | `MINUS_DM` | Minus directional movement. At `period=1`, the raw one-bar movement from bar 1 (lookback 1), as in TA-Lib. | `:<period>@<series_high=high>,<series_low=low>` |
| <a id="indicator-plus_di"></a>`plus_di` | `PLUS_DI` | Plus directional indicator, Wilder's `100·+DM/TR`. At `period=1` volas keeps the ×100 on the one-bar ratio; TA-Lib's period-1 branch omits it (`+DM/TR`), a deliberate divergence that keeps period 1 on the same scale as every other period. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-minus_di"></a>`minus_di` | `MINUS_DI` | Minus directional indicator, Wilder's `100·−DM/TR`. At `period=1` volas keeps the ×100 on the one-bar ratio; TA-Lib's period-1 branch omits it (`−DM/TR`), a deliberate divergence that keeps period 1 on the same scale as every other period. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-dx"></a>`dx` | `DX` | Directional Movement Index. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-adx"></a>`adx` | `ADX` | Average Directional Movement Index. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-adxr"></a>`adxr` | `ADXR` | Average Directional Movement Index Rating. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-obv"></a>`obv` | `OBV` | On-Balance Volume: adds the bar's volume on an up close, subtracts it on a down close, and leaves the total unchanged on a flat one. A non-finite close or volume skips the row under the missing-input rule. | `@<series_close=close>,<series_volume=volume>` |
| <a id="indicator-ad"></a>`ad` | `AD` | Chaikin Accumulation Distribution line. | `@<series_high=high>,<series_low=low>,<series_close=close>,<series_volume=volume>` |
| <a id="indicator-adosc"></a>`adosc` | `ADOSC` | Chaikin Accumulation Distribution oscillator. | `:<fast=3>,<slow=10>@<series_high=high>,<series_low=low>,<series_close=close>,<series_volume=volume>` |
| <a id="indicator-tr"></a>`tr` | `TRANGE` | True Range. | `@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-atr"></a>`atr` | `ATR` | Average True Range. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-natr"></a>`natr` | `NATR` | Normalized Average True Range, `ATR/close·100`; a close of exactly 0 gives 0 at its own row (TA-Lib's guard, with an exact zero test). At `period=1` volas normalises as at every other period, where TA-Lib returns the raw true range. | `:<period>@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-avgprice"></a>`avgprice` | `AVGPRICE` | Average price. | `@<series_open=open>,<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-medprice"></a>`medprice` | `MEDPRICE` | Median price. | `@<series_high=high>,<series_low=low>` |
| <a id="indicator-typprice"></a>`typprice` | `TYPPRICE` | Typical price. | `@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-wclprice"></a>`wclprice` | `WCLPRICE` | Weighted close price. | `@<series_high=high>,<series_low=low>,<series_close=close>` |
| <a id="indicator-ht_dcperiod"></a>`ht_dcperiod` | `HT_DCPERIOD` | Hilbert Transform dominant cycle period. | `@<series=close>` |
| <a id="indicator-ht_dcphase"></a>`ht_dcphase` | `HT_DCPHASE` | Hilbert Transform dominant cycle phase. Every Hilbert-transform output (`ht_*`, `mama`) evaluates TA-Lib's C recurrences in source order without fused multiply-add, so it is the same on every platform and bit-identical, from its first published row, to a TA-Lib build that does not fuse (x86-64 without FMA); against a fused build (clang on arm64) it agrees to rounding. On flat or step input the phase is rounding noise and may differ between TA-Lib builds and platforms. | `@<series=close>` |
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
| <a id="indicator-maxindex"></a>`maxindex` | `MAXINDEX` | Index of the rolling maximum; a tie reports the latest equal high, so a value depends on its window alone (TA-Lib reports the earliest after its tracker rebuilds, a deliberate divergence). | `:<period>@<series=close>` |
| <a id="indicator-minindex"></a>`minindex` | `MININDEX` | Index of the rolling minimum; a tie reports the latest equal low (see `maxindex`). | `:<period>@<series=close>` |
| <a id="indicator-minmax-min"></a>`minmax.min` | `MINMAX` | Rolling minimum from the MINMAX pair. | `:<period>@<series=close>` |
| <a id="indicator-minmax-max"></a>`minmax.max` | `MINMAX` | Rolling maximum from the MINMAX pair. | `:<period>@<series=close>` |
| <a id="indicator-minmaxindex-min"></a>`minmaxindex.min` | `MINMAXINDEX` | Index of the rolling minimum from the pair; a tie reports the latest equal low (see `maxindex`). | `:<period>@<series=close>` |
| <a id="indicator-minmaxindex-max"></a>`minmaxindex.max` | `MINMAXINDEX` | Index of the rolling maximum from the pair; a tie reports the latest equal high (see `maxindex`). | `:<period>@<series=close>` |
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
