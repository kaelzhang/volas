# Directive cheat sheet

Directive names are stable column names. Multi-output indicators expose each
line as its own directive, so code should request the exact line it needs rather
than rely on vague aliases.

```py
df["rsi:14"]                         # one output
df[["macd", "macd.signal"]]          # several outputs
df["rsi:14@settle"]                  # override input series
```

## Common multi-output indicators

| Family | Outputs | Example |
| --- | --- | --- |
| MACD | `macd`, `macd.signal`, `macd.histogram` | `df[["macd", "macd.signal", "macd.histogram"]]` |
| Fixed MACD | `macdfix`, `macdfix.signal`, `macdfix.histogram` | `df[["macdfix", "macdfix.signal", "macdfix.histogram"]]` |
| MACDEXT | `macdext`, `macdext.signal`, `macdext.histogram` | `df[["macdext", "macdext.signal", "macdext.histogram"]]` |
| Bollinger Bands | `boll.upper`, `boll`, `boll.lower` | `df[["boll.upper", "boll", "boll.lower"]]` |
| Stochastic | `stoch.k`, `stoch.d` | `df[["stoch.k", "stoch.d"]]` |
| Fast Stochastic | `stochf.k`, `stochf.d` | `df[["stochf.k", "stochf.d"]]` |
| Stochastic RSI | `stochrsi.k`, `stochrsi.d` | `df[["stochrsi.k", "stochrsi.d"]]` |
| Aroon | `aroon.up`, `aroon.down` | `df[["aroon.up:14", "aroon.down:14"]]` |
| Hilbert phasor | `ht_phasor`, `ht_phasor.quadrature` | `df[["ht_phasor", "ht_phasor.quadrature"]]` |
| Hilbert sine | `ht_sine`, `ht_sine.leadsine` | `df[["ht_sine", "ht_sine.leadsine"]]` |
| Keltner Channel | `keltner.upper`, `keltner`, `keltner.lower` | `df[["keltner.upper", "keltner", "keltner.lower"]]` |
| Relative Vigor | `relative_vigor`, `relative_vigor.signal` | `df[["relative_vigor", "relative_vigor.signal"]]` |

## Arguments and input columns

Arguments before `@` are positional. Empty slots keep defaults:

```py
df["macd.signal:12,26,9@close"]
df["macd.signal:,,5"]        # keep fast=12 and slow=26, set signal=5
df["stoch.d:5,3,0,3,0@high,low,close"]
```

Short aliases such as `macd.s` and `boll.u` are accepted for compatibility, but
public examples should prefer the stable full names: `macd.signal`,
`boll.upper`, `aroon.down`, and so on.

The complete reference is [INDICATORS.md](../INDICATORS.md).
