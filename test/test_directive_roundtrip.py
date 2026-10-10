"""A directive's canonical name re-parses to the same directive.

A cached directive column is stored — and, after an append, refreshed — under its
canonical name (`directive_stringify`), which drops the spaces around operators
(`rsi:14 > 50` -> `rsi:14>50`). So an argument must end at a binary operator, or the
refresh of a compound column would read `14>50` as one argument.
"""

import numpy as np
import pytest

from volas import DataFrame, directive_lookback, directive_stringify

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
    np.testing.assert_array_equal(df.exec("ma:2-1"), df.exec("ma:2 - 1"))


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
    full = np.asarray(DataFrame(data).exec(directive), dtype=float)
    np.testing.assert_allclose(got, full, rtol=1e-9, atol=1e-9)
