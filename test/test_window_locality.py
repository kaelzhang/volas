"""Window locality: an output depends only on the rows its definition reads.

A non-finite input (NaN / +inf / -inf) must not outlive the window that holds it. A
**finite-window** indicator, once the bad row has left its window, must give exactly
what it gives on clean data. A **recursive** indicator (infinite memory: EMA / Wilder /
cumulative / Hilbert families) skips a missing row altogether (`test_missing_rows.py`);
here, a leading run of missing rows is a series that simply starts later.

Every command in the Rust registry (spec.rs COMMANDS) must be classified here, so a
new indicator cannot skip this contract.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import volas
from volas import DataFrame

ROOT = Path(__file__).resolve().parent.parent

# command -> a runnable directive exercising it.
FINITE = {
    "ma": "ma:5",
    "apo": "apo",
    "ppo": "ppo",
    "macdext": "macdext",
    "wma": "wma:5",
    "trima": "trima:5",
    "mavp": "mavp:2,30,0@close,periods",
    "accbands": "accbands",
    "boll": "boll",
    "bbw": "bbw",
    "rsv": "rsv:5",
    "bbi": "bbi",
    "psy": "psy",
    "dpo": "dpo:5",
    "cmf": "cmf",
    "chop": "chop:5",
    "kst": "kst",
    "emv": "emv:5",
    "bias": "bias",
    "dma": "dma",
    "vortex": "vortex.minus",
    "brar": "brar.ar",
    "vr": "vr",
    "coppock": "coppock",
    "relative_vigor": "relative_vigor",
    "dkx": "dkx",
    "wvad": "wvad",
    "cdp": "cdp",
    "ttm_squeeze": "ttm_squeeze",
    "mike": "mike.midr:5",
    "pivot_points": "pivot_points",
    "ichimoku": "ichimoku.chikou",
    "tr": "tr",
    "llv": "llv:5",
    "hhv": "hhv:5",
    "donchian": "donchian:5",
    "hv": "hv:5",
    "increase": "increase",
    "style": "style.bearish",
    "repeat": "repeat:3@(close > ma:5)",
    "change": "change",
    "mom": "mom:5",
    "roc": "roc:5",
    "rocp": "rocp:5",
    "rocr": "rocr:5",
    "rocr100": "rocr100:5",
    "willr": "willr:5",
    "cci": "cci:5",
    "imi": "imi:5",
    "mfi": "mfi:5",
    "ultosc": "ultosc",
    "stoch": "stoch.d",
    "stochf": "stochf.d",
    "aroon": "aroon.d:5",
    "aroonosc": "aroonosc:5",
    "sum": "sum:5",
    "maxindex": "maxindex:5",
    "minindex": "minindex:5",
    "minmax": "minmax.max:5",
    "minmaxindex": "minmaxindex.max:5",
    "bop": "bop",
    "midpoint": "midpoint:5",
    "midprice": "midprice:5",
    "linearreg": "linearreg:5",
    "linearreg_slope": "linearreg_slope:5",
    "linearreg_intercept": "linearreg_intercept:5",
    "linearreg_angle": "linearreg_angle:5",
    "tsf": "tsf:5",
    "var": "var:5",
    "stddev": "stddev:5",
    "correl": "correl:5@close,volume",
    "beta": "beta:5@close,volume",
    "median": "median:5",
    "quantile": "quantile:5",
    "rank": "rank:5",
    "skew": "skew:5",
    "kurt": "kurt:5",
    "sem": "sem:5",
    "vwma": "vwma:5",
    "alma": "alma:5",
    "hma": "hma:5",
    "swma": "swma",
    "cog": "cog:5",
    "dev": "dev:5",
    "rci": "rci:5",
    "iii": "iii",
    "mode": "mode:5",
    "pivothigh": "pivothigh:5,3",
    "pivotlow": "pivotlow:5,3",
    "avgprice": "avgprice",
    "medprice": "medprice",
    "typprice": "typprice",
    "wclprice": "wclprice",
}
RECURSIVE = {
    "ema": "ema:5",
    "smma": "smma:5",
    "macdfix": "macdfix",
    "dema": "dema:5",
    "tema": "tema:5",
    "t3": "t3:5",
    "kama": "kama:5",
    "sar": "sar",
    "sarext": "sarext",
    "macd": "macd",
    "kdj": "kdj.d",
    "rsi": "rsi:5",
    "pvt": "pvt",
    "nvi": "nvi",
    "pvi": "pvi",
    "mass_index": "mass_index",
    "efi": "efi",
    "tsi": "tsi",
    "crsi": "crsi",
    "keltner": "keltner",
    "stoch_momentum": "stoch_momentum",
    "wad": "wad",
    "asi": "asi",
    "supertrend": "supertrend",
    "atr": "atr:5",
    "cmo": "cmo:5",
    "trix": "trix:5",
    "stochrsi": "stochrsi.d",
    "plus_dm": "plus_dm:5",
    "minus_dm": "minus_dm:5",
    "plus_di": "plus_di:5",
    "minus_di": "minus_di:5",
    "dx": "dx:5",
    "adx": "adx:5",
    "adxr": "adxr:5",
    "natr": "natr:5",
    "kcw": "kcw",
    "obv": "obv",
    "ad": "ad",
    "adosc": "adosc",
    "ht_dcperiod": "ht_dcperiod",
    "ht_dcphase": "ht_dcphase",
    "ht_phasor": "ht_phasor",
    "ht_sine": "ht_sine",
    "ht_trendline": "ht_trendline",
    "ht_trendmode": "ht_trendmode",
    "mama": "mama",
}

# Indicators that emit their first value from a partial first difference (as 通达信
# does): they read one row more than `directive_lookback` reports.
EXTRA_REACH = {"psy": 1}
# Indicators whose output is an absolute row position.
ABSOLUTE_POSITION = {"maxindex", "minindex", "minmaxindex"}

# Every selectable sub-command line, which may run through its own kernel entry point.
SUBCOMMANDS = {
    "accbands.lower": "accbands.lower",
    "accbands.upper": "accbands.upper",
    "aroon.d": "aroon.d:5",
    "aroon.down": "aroon.down:5",
    "aroon.up": "aroon.up:5",
    "boll.lower": "boll.lower",
    "boll.u": "boll.u",
    "boll.upper": "boll.upper",
    "brar.ar": "brar.ar",
    "brar.br": "brar.br",
    "cdp.ah": "cdp.ah",
    "cdp.al": "cdp.al",
    "cdp.nh": "cdp.nh",
    "cdp.nl": "cdp.nl",
    "dkx.ma": "dkx.ma",
    "dma.ama": "dma.ama",
    "dma.ddd": "dma.ddd",
    "donchian.lower": "donchian.lower:5",
    "donchian.upper": "donchian.upper:5",
    "ht_phasor.quadrature": "ht_phasor.quadrature",
    "ht_sine.leadsine": "ht_sine.leadsine",
    "ichimoku.chikou": "ichimoku.chikou",
    "ichimoku.kijun": "ichimoku.kijun",
    "ichimoku.senkou_a": "ichimoku.senkou_a",
    "ichimoku.senkou_b": "ichimoku.senkou_b",
    "ichimoku.tenkan": "ichimoku.tenkan",
    "kdj.d": "kdj.d",
    "kdj.j": "kdj.j",
    "kdj.k": "kdj.k",
    "keltner.lower": "keltner.lower",
    "keltner.upper": "keltner.upper",
    "macd.histogram": "macd.histogram",
    "macd.s": "macd.s",
    "macd.signal": "macd.signal",
    "macdext.histogram": "macdext.histogram",
    "macdext.signal": "macdext.signal",
    "macdfix.histogram": "macdfix.histogram",
    "macdfix.signal": "macdfix.signal",
    "mama.fama": "mama.fama",
    "mike.midr": "mike.midr:5",
    "mike.mids": "mike.mids:5",
    "mike.strongr": "mike.strongr:5",
    "mike.strongs": "mike.strongs:5",
    "mike.weakr": "mike.weakr:5",
    "mike.weaks": "mike.weaks:5",
    "minmax.max": "minmax.max:5",
    "minmax.min": "minmax.min:5",
    "minmaxindex.max": "minmaxindex.max:5",
    "minmaxindex.min": "minmaxindex.min:5",
    "pivot_points.p": "pivot_points.p",
    "pivot_points.r1": "pivot_points.r1",
    "pivot_points.r2": "pivot_points.r2",
    "pivot_points.r3": "pivot_points.r3",
    "pivot_points.s1": "pivot_points.s1",
    "pivot_points.s2": "pivot_points.s2",
    "pivot_points.s3": "pivot_points.s3",
    "relative_vigor.signal": "relative_vigor.signal",
    "stoch.d": "stoch.d",
    "stoch.k": "stoch.k",
    "stoch_momentum.signal": "stoch_momentum.signal",
    "stochf.d": "stochf.d",
    "stochf.k": "stochf.k",
    "stochrsi.d": "stochrsi.d",
    "stochrsi.k": "stochrsi.k",
    "style.bearish": "style.bearish",
    "style.bullish": "style.bullish",
    "supertrend.direction": "supertrend.direction",
    "ttm_squeeze.on": "ttm_squeeze.on",
    "vortex.minus": "vortex.minus",
    "vortex.plus": "vortex.plus",
}


def _command(key):
    return key.split(".", 1)[0]


def _directive(key):
    return {**FINITE, **RECURSIVE, **SUBCOMMANDS}[key]


N, K, LEAD = 200, 40, 3
_rng = np.random.default_rng(0)
_c = 100.0 + np.cumsum(_rng.standard_normal(N))
BASE = dict(
    open=_c + 0.1, high=_c + 1.0, low=_c - 1.0, close=_c.copy(),
    volume=_rng.integers(100, 900, N).astype(float), periods=np.full(N, 5.0),
)
PRICE_VOLUME = ("open", "high", "low", "close", "volume")


def _frame(mutate=None, start=0):
    data = {k: v.copy() for k, v in BASE.items()}
    if mutate is not None:
        for k in PRICE_VOLUME:
            mutate(data[k])
    return DataFrame({k: v[start:] for k, v in data.items()})


def _exec(directive, mutate=None, start=0):
    return np.asarray(_frame(mutate, start).exec(directive, dtype='float64'), dtype=float)


def test_registry_fully_classified():
    src = (ROOT / "crates/volas-directive/src/spec.rs").read_text()
    body = re.search(r"pub const COMMANDS[^=]*=\s*&\[(.*?)\];", src, re.S).group(1)
    commands = set(re.findall(r'"([a-z0-9_]+)"', body))
    assert not set(FINITE) & set(RECURSIVE)
    assert commands == set(FINITE) | set(RECURSIVE)


_FINITE_KEYS = sorted(FINITE) + sorted(k for k in SUBCOMMANDS if _command(k) in FINITE)
_ALL_KEYS = sorted(FINITE) + sorted(RECURSIVE) + sorted(SUBCOMMANDS)


def test_subcommands_belong_to_classified_commands():
    assert all(_command(k) in FINITE or _command(k) in RECURSIVE for k in SUBCOMMANDS)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("key", _FINITE_KEYS)
def test_finite_window_recovers_after_bad_value_leaves(key, bad):
    directive, command = _directive(key), _command(key)
    clean = _exec(directive)

    def put(v):
        v[K] = bad

    got = _exec(directive, put)
    after = K + volas.directive_lookback(directive) + 1 + EXTRA_REACH.get(command, 0)
    np.testing.assert_allclose(got[after:], clean[after:], rtol=1e-9, atol=1e-9)
    # rows before the bad value are untouched
    np.testing.assert_allclose(got[:K], clean[:K], rtol=1e-9, atol=1e-9)


# --- a missing row is missing for exactly the outputs whose definition reads it ---

# The rows (offsets from the missing row) whose definition reads it, for keys whose
# reach is not the contiguous trailing window `[0, lookback + extra]`: a lag reads one
# row `period` back, a prior-bar line reads only the bar before, a displaced line reads
# its window `displacement` rows back. (`bbi`'s reported lookback is one row long.)
_LAG = {k: [0, 5] for k in ("mom", "roc", "rocp", "rocr", "rocr100")}
_PRIOR_BAR = {k: [1] for k in ("cdp", "pivot_points", *(s for s in SUBCOMMANDS if s.startswith(("cdp.", "pivot_points."))))}
READS = {
    **_LAG,
    **_PRIOR_BAR,
    "bbi": list(range(24)),
    "coppock": [*range(10), *range(11, 24)],
    "mavp": list(range(5)),  # periods = 5
    "ichimoku": [0],  # the chikou line is the close itself
    "ichimoku.chikou": [0],
    "ichimoku.senkou_a": list(range(26, 52)),
    "ichimoku.senkou_b": list(range(26, 78)),
}


def _reads(key):
    if key in READS:
        return READS[key]
    span = volas.directive_lookback(_directive(key)) + 1 + EXTRA_REACH.get(_command(key), 0)
    return list(range(span))


def _base_for(key):
    """The data a key is checked on: `BASE`, except the candle-colour lines, whose `open`
    must not sit a fixed step above `close` (else the colour never changes and a
    laundered missing value would go unseen) — they take the candle data."""
    return {**BASE, **CDL_BASE} if _command(key) == "style" else BASE


def _exec_missing(directive, columns, rows=(K,), base=None):
    data = {k: v.copy() for k, v in (BASE if base is None else base).items()}
    for k in columns:
        data[k][list(rows)] = np.nan
    return np.asarray(DataFrame(data).exec(directive, dtype='float64'), dtype=float)


@pytest.mark.parametrize("key", _FINITE_KEYS)
def test_a_window_holding_a_missing_row_is_missing(key):
    """Every output whose definition reads the missing row is NaN — an extremum, a
    position, a count or a per-row term never steps over it."""
    got = _exec_missing(_directive(key), PRICE_VOLUME, base=_base_for(key))
    rows = [K + d for d in _reads(key)]
    assert np.isnan(got[rows]).all(), f"finite at {[r for r in rows if not np.isnan(got[r])]}"


@pytest.mark.parametrize("key,column", [(k, c) for k in _FINITE_KEYS for c in PRICE_VOLUME])
def test_a_missing_input_is_never_laundered_into_a_value(key, column):
    """With one input missing at one row, every output is either NaN or exactly what
    clean data gives: a missing value never turns into a different finite one."""
    directive, base = _directive(key), _base_for(key)
    clean = np.asarray(DataFrame(base).exec(directive, dtype='float64'), dtype=float)
    # a candle-colour line: one white and one black bar missing, so either colour shows
    rows = (K, K + 1) if _command(key) == "style" else (K,)
    got = _exec_missing(directive, [column], rows, base=base)
    laundered = ~np.isnan(got) & ~np.isclose(got, clean, rtol=1e-9, atol=1e-9, equal_nan=True)
    assert not laundered.any(), f"rows {np.flatnonzero(laundered)[:5]}: {got[laundered][:3]}"


def test_a_missing_period_is_a_missing_mavp_row():
    """`mavp` reads its period series at the row only: a missing period is NaN there."""
    data = {k: v.copy() for k, v in BASE.items()}
    data["periods"][K] = np.nan
    got = np.asarray(DataFrame(data).exec(FINITE["mavp"]), float)
    clean = _exec(FINITE["mavp"])
    assert np.isnan(got[K])
    np.testing.assert_array_equal(np.delete(got, K), np.delete(clean, K))


@pytest.mark.parametrize("directive", ["roc:1", "rocp:1", "rocr:1", "rocr100:1"])
def test_a_missing_value_over_a_zero_base_is_missing(directive):
    # the zero-base rule (0) applies to a present value only
    got = np.asarray(_closes([0.0, np.nan, 1.0, 2.0]).exec(directive), float)
    assert np.isnan(got[:3]).all() and np.isfinite(got[3])


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_leading_missing_rows_mean_a_later_start(key):
    """Once a series that starts later has warmed up, a series whose first rows are
    missing must agree with it — a leading gap never poisons what follows."""
    directive, command = _directive(key), _command(key)

    def lead(v):
        v[:LEAD] = np.nan

    got = _exec(directive, lead)[LEAD:]
    later = _exec(directive, start=LEAD)
    if command in ABSOLUTE_POSITION:
        later = later + LEAD
    warm = volas.directive_lookback(directive) + EXTRA_REACH.get(command, 0)
    np.testing.assert_allclose(got[warm:], later[warm:], rtol=1e-9, atol=1e-9)


# --- the value a window holding a bad row takes: the formula on that window ---

def _closes(xs):
    return DataFrame({"close": np.asarray(xs, dtype=float)})


@pytest.mark.parametrize("directive,want", [
    ("sum:2", [np.nan, 3.0, np.inf, np.inf, 9.0, 11.0]),
    ("ma:2", [np.nan, 1.5, np.inf, np.inf, 4.5, 5.5]),
    ("var:2", [np.nan, 0.25, np.nan, np.nan, 0.25, 0.25]),
    ("stddev:2", [np.nan, 0.5, np.nan, np.nan, 0.5, 0.5]),
])
def test_window_holding_inf_takes_the_formula_value(directive, want):
    got = _closes([1, 2, np.inf, 4, 5, 6]).exec(directive).tolist()
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("directive", ["sum:3", "ma:3", "var:3", "stddev:3", "wma:3", "linearreg:3"])
def test_window_holding_nan_is_nan_then_recovers(directive):
    got = np.asarray(_closes([1, 2, np.nan, 4, 5, 6]).exec(directive, dtype='float64'), float)
    assert np.isnan(got[2:5]).all()
    assert np.isfinite(got[5])


def test_opposite_infinities_in_one_window_are_nan():
    got = _closes([1, np.inf, -np.inf, 4, 5]).exec("sum:2").tolist()
    np.testing.assert_array_equal(got, [np.nan, np.inf, np.nan, -np.inf, 9.0])


# --- pandas-semantics rolling: NaN is missing (skipped), ±inf is a value ---

@pytest.mark.parametrize("method,want", [
    ("sum", [np.nan, 3.0, np.inf, np.inf, 9.0, 11.0]),
    ("mean", [np.nan, 1.5, np.inf, np.inf, 4.5, 5.5]),
])
def test_rolling_recovers_after_inf_leaves(method, want):
    s = DataFrame({"x": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0]})["x"]
    got = getattr(s.rolling(2), method)().to_numpy().tolist()
    np.testing.assert_array_equal(got, want)


def test_rolling_mixed_infinities_are_nan_then_recover():
    s = DataFrame({"x": [1.0, np.inf, -np.inf, 4.0, 5.0]})["x"]
    got = s.rolling(2).sum().to_numpy().tolist()
    np.testing.assert_array_equal(got, [np.nan, np.inf, np.nan, -np.inf, 9.0])


@pytest.mark.parametrize("key", _FINITE_KEYS)
def test_bad_rows_one_window_apart_both_leave(key):
    """A bad row entering as another leaves — the window holds exactly one throughout."""
    directive, command = _directive(key), _command(key)
    clean = _exec(directive)
    span = volas.directive_lookback(directive) + 1 + EXTRA_REACH.get(command, 0)

    def put(v):
        v[K] = np.nan
        v[K + span] = np.nan

    got = _exec(directive, put)
    after = K + 2 * span
    np.testing.assert_allclose(got[after:], clean[after:], rtol=1e-9, atol=1e-9)


def test_mfi_flat_bars_next_to_an_infinity():
    n = 30
    tp = np.r_[np.full(10, 100.0), np.linspace(100, 110, n - 10)]
    data = dict(high=tp + 1, low=tp - 1, close=tp.copy(), volume=np.full(n, 50.0))
    clean = np.asarray(DataFrame(data).exec("mfi:5", dtype='float64'), float)
    data["close"] = data["close"].copy()
    data["close"][2] = np.inf
    got = np.asarray(DataFrame(data).exec("mfi:5", dtype='float64'), float)
    # Only bar 2's flow is non-finite (bar 3's typical price is below +inf, a finite
    # negative flow); the 5-flow windows holding it end at rows 5 and 6 (row 5 is the
    # first output), and the flat bars around it take the zero-flow branch.
    assert np.isnan(got[5:7]).all()
    np.testing.assert_allclose(got[7:], clean[7:], rtol=1e-9, atol=1e-9)


# --- the incremental refresh continues the same recursion as a full recompute ---

_RECURSIVE_KEYS = sorted(RECURSIVE) + sorted(k for k in SUBCOMMANDS if _command(k) in RECURSIVE)


@pytest.mark.parametrize("key", _RECURSIVE_KEYS)
def test_incremental_refresh_after_a_leading_gap_matches_a_full_recompute(key):
    directive = _directive(key)
    data = {k: v.copy() for k, v in BASE.items()}
    for k in PRICE_VOLUME:
        data[k][:LEAD] = np.nan
    seed = 150
    df = DataFrame({k: v[:seed] for k, v in data.items()})
    df[directive]  # cache it: the carried state is captured here
    for i in range(seed, N):
        df.append(DataFrame({k: v[i:i + 1] for k, v in data.items()}))
        df.fulfill()
    got = np.asarray(df[directive], dtype=float)
    full = np.asarray(DataFrame(data).exec(directive, dtype='float64'), dtype=float)
    np.testing.assert_allclose(got, full, rtol=1e-9, atol=1e-9)


# --- candlestick patterns: a missing candle reaches only the bars that read it ---

_CDL_SRC = (ROOT / "crates/volas-compute/src/raw/indicators/candles/mod.rs").read_text()
CANDLES = re.findall(r'"([a-z0-9]+)"', re.search(r"pub const CANDLE_PATTERNS[^=]*=\s*&\[(.*?)\];", _CDL_SRC, re.S).group(1))
_r = np.random.default_rng(7)
_cc = 100.0 + np.cumsum(_r.standard_normal(N))
_co = _cc + _r.normal(0.0, 0.8, N)
CDL_BASE = dict(open=_co, high=np.maximum(_co, _cc) + _r.exponential(0.6, N),
                low=np.minimum(_co, _cc) - _r.exponential(0.6, N), close=_cc, volume=np.ones(N))


def _cdl(directive, column=None, bad=None):
    data = {k: v.copy() for k, v in CDL_BASE.items()}
    if column is not None:
        data[column][K] = bad
    return np.asarray(DataFrame(data).exec(directive, dtype='float64'), dtype=float)


def test_every_candle_pattern_is_covered():
    assert len(CANDLES) == 61 and np.count_nonzero(sum(_cdl(f"cdl.{p}") != 0 for p in CANDLES)) > 50


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("column", ("open", "high", "low", "close"))
@pytest.mark.parametrize("pattern", CANDLES)
def test_a_candle_pattern_is_missing_exactly_while_it_reads_a_missing_candle(pattern, column, bad):
    """A candle with a missing or infinite price makes a pattern NaN on the bars whose
    lookback holds it; every other bar is exactly what clean data gives (a candle
    average never carries it further)."""
    directive = f"cdl.{pattern}"
    clean, got = _cdl(directive), _cdl(directive, column, bad)
    span = volas.directive_lookback(directive) + 1
    assert np.isnan(got[K:K + span]).all()
    np.testing.assert_array_equal(np.delete(got, range(K, K + span)), np.delete(clean, range(K, K + span)))
