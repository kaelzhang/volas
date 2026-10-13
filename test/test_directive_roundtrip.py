"""A directive's canonical name re-parses to the same directive.

A cached directive column is stored — and, after an append, refreshed — under its
canonical name (`directive_stringify`), which drops the spaces around operators
(`rsi:14 > 50` -> `rsi:14>50`). So an argument must end at a binary operator, or the
refresh of a compound column would read `14>50` as one argument.
"""

import re

import numpy as np
import pytest

from volas import DataFrame, directive_lookback, directive_stringify

from test.exact import assert_exact

COMPOUND = [
    "rsi:14 > 50",
    "natr:14 + cmo:14",
    "ma:5 - ma:10",
    "ma:5 * 2",
    "ma:5 / ma:10",
    "ma:5 >= ma:10",
    "ma:5 <= ma:10",
    "ma:5 == ma:10",
    "ma:5 != ma:10",
    "ma:5 > ma:10 & close > open",
    "ma:5 > ma:10 | close > open",
    "ma:5 \\ ma:10",
    "ma:5 // ma:10",
    "ma:5 >< ma:10",
    "boll.upper:20,2 - boll.lower:20,2",
    "increase:3,-1 & close > open",
    "hv:10,15m > 0",
    "ma:5@(close>open) > 0",
    "ma:5 ^ ma:10",
    # a series operand other than a bare column keeps its parentheses
    "ema:5@(rsi:5)",
    "ema:5@(macd.signal:5,10,3@(close*2))",
    "atr:5@high,(low-1),(ema:3@open)",
    "correl:5@close,(3)",
]


@pytest.mark.parametrize("directive", COMPOUND)
def test_canonical_name_round_trips(directive):
    canonical = directive_stringify(directive)
    assert directive_stringify(canonical) == canonical
    assert directive_lookback(canonical) == directive_lookback(directive)


@pytest.mark.parametrize("directive", [
    "increase:3,-1",        # a sign opening an argument
    "boll.upper:20,1e-1",   # an exponent sign
    "boll.upper:20,1E+1",
])
def test_a_sign_inside_a_number_stays_in_the_argument(directive):
    directive_lookback(directive)  # one argument, not `… - 1` / `… + 1`


def test_an_operator_after_an_argument_is_an_operator():
    df = DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert_exact(df.exec("ma:2-1"), df.exec("ma:2 - 1"))


def _frame(n):
    rng = np.random.default_rng(0)
    c = 100.0 + np.cumsum(rng.standard_normal(n))
    return dict(open=c + 0.1, high=c + 1.0, low=c - 1.0, close=c, volume=np.full(n, 10.0))


@pytest.mark.parametrize("directive", COMPOUND)
def test_cached_compound_column_refreshes_after_append(directive):
    data, seed, n = _frame(80), 50, 80
    df = DataFrame({k: v[:seed] for k, v in data.items()})
    df[directive]
    for i in range(seed, n):
        df.append(DataFrame({k: v[i : i + 1] for k, v in data.items()}))
        df.fulfill()
    got = np.asarray(df[directive], dtype=float)
    full = np.asarray(DataFrame(data).exec(directive, dtype='float64'), dtype=float)
    np.testing.assert_allclose(got, full, rtol=1e-9, atol=1e-9)


# --- a cached column refreshes exactly, whatever its memory ------------------------
# A composite carries no resume state, so its refresh recomputes the stale tail: over a
# trailing window when every command in it has finite memory, from row 0 otherwise. A
# boolean composite of a recursive indicator (`rsi:14 > 50`) must not be refreshed over a
# window, where the indicator re-warms and its threshold crossings move.

def _all_forms():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "wl", Path(__file__).with_name("test_window_locality.py")
    )
    wl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wl)
    return wl, {**wl.FINITE, **wl.RECURSIVE, **wl.SUBCOMMANDS}


_WL, _FORMS = _all_forms()


@pytest.mark.parametrize("key", sorted(_FORMS))
def test_cached_composite_refreshes_exactly(key):
    base = {k: v for k, v in _WL.BASE.items()}
    seed, n = 120, 140
    # Threshold midway between the two middle values of the indicator's own tail, so a
    # drifted (re-warmed) value flips the comparison while float rounding cannot.
    values = np.asarray(DataFrame({k: v[:n] for k, v in base.items()}).exec(_FORMS[key], dtype='float64'), float)
    tail = np.unique(values[seed:][np.isfinite(values[seed:])])
    middle = float((tail[len(tail) // 2 - 1] + tail[len(tail) // 2]) / 2) if len(tail) > 1 else 0.0
    directive = f"({_FORMS[key]}) > {middle!r}"
    df = DataFrame({k: v[:seed] for k, v in base.items()})
    df[directive]
    for i in range(seed, n):
        df.append(DataFrame({k: v[i : i + 1] for k, v in base.items()}))
        df.fulfill()
    got = np.asarray(df[directive], dtype=float)
    full = np.asarray(DataFrame({k: v[:n] for k, v in base.items()}).exec(directive, dtype='float64'), dtype=float)
    assert_exact(got, full)


def _candle_patterns():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "crates/volas-compute/src/raw/indicators/candles/mod.rs").read_text()
    body = re.search(r"pub const CANDLE_PATTERNS[^=]*=\s*&\[(.*?)\];", src, re.S).group(1)
    return re.findall(r'"([a-z0-9]+)"', body)


@pytest.mark.parametrize("pattern", _candle_patterns())
def test_cached_candle_pattern_refreshes_exactly(pattern):
    directive = f"cdl.{pattern} != 0"
    base = {k: v for k, v in _WL.BASE.items()}
    seed, n = 120, 160
    df = DataFrame({k: v[:seed] for k, v in base.items()})
    df[directive]
    for i in range(seed, n):
        df.append(DataFrame({k: v[i : i + 1] for k, v in base.items()}))
        df.fulfill()
    got = np.asarray(df[directive], dtype=float)
    full = np.asarray(DataFrame({k: v[:n] for k, v in base.items()}).exec(directive, dtype='float64'), dtype=float)
    assert_exact(got, full)



@pytest.mark.parametrize("directive", [
    "-ema:10 > -100",               # a unary operator around a recursive command
    "~(rsi:14 > 50)",
    "ma:10,1 > 100",                # ma with an EMA matype is recursive
    "ma:10,2 > 100",                # ... with a WMA matype it is not
    "apo:12,26,3 > 0",              # DEMA matype
    "stoch.k:5,3,1,3,0 > 50",       # EMA-smoothed slow %K
    "macdext:12,1,26,1 > 0",
])
def test_cached_composite_refreshes_exactly_by_its_structure(directive):
    base = {k: v for k, v in _WL.BASE.items()}
    seed, n = 120, 160
    df = DataFrame({k: v[:seed] for k, v in base.items()})
    df[directive]
    for i in range(seed, n):
        df.append(DataFrame({k: v[i : i + 1] for k, v in base.items()}))
        df.fulfill()
    got = np.asarray(df[directive], dtype=float)
    full = np.asarray(DataFrame({k: v[:n] for k, v in base.items()}).exec(directive, dtype='float64'), dtype=float)
    assert_exact(got, full)
