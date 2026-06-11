"""Systematic audit — P8 layer-1 signature sublayer (review NF-C).

The method-name surface differential is blind to *parameter*-level gaps: a method
volas has, but missing a pandas kwarg. volas exposes reflectable signatures
(pyo3 text sigs), so the gap is machine-detectable. The verified-important
missing kwargs are align-backlog; the bulk of pandas's kwargs are out-of-scope.

Cell IDs:  T-sig.<receiver>.<method>/kw=<param>
"""

from __future__ import annotations

import inspect

import pytest

import volas

_S = volas.DataFrame({"x": [3.0, 1.0, 2.0]})["x"]
_DF = volas.DataFrame({"a": [3.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0]})


def _params(fn):
    try:
        return set(inspect.signature(fn).parameters)
    except (ValueError, TypeError):  # pragma: no cover
        return None


def test_signature_reflection_is_feasible():
    """The sublayer is possible: volas methods expose reflectable signatures,
    so a pandas-vs-volas parameter differential is machine-generable (P8)."""
    assert _params(_S.sort_values) == {"ascending", "na_position"}   # F44 landed


# F44 (FIXED): the verified-important pandas kwargs are accepted and behave.
def test_sort_values_na_position():
    s = volas.DataFrame({"x": [3.0, None, 1.0]})["x"]
    assert s.sort_values(na_position="first").isna().to_list()[0] is True
    assert s.sort_values(na_position="last").isna().to_list()[-1] is True
    with pytest.raises(ValueError):
        s.sort_values(na_position="middle")


def test_rank_na_option():
    s = volas.DataFrame({"x": [3.0, None, 1.0]})["x"]
    assert s.rank(na_option="keep").isna().to_list() == [False, True, False]
    top = s.rank(na_option="top").to_list()
    assert top == [3.0, 1.0, 2.0]                  # NA takes rank 1, others shift
    bottom = s.rank(na_option="bottom").to_list()
    assert bottom == [2.0, 3.0, 1.0]               # NA takes the last rank


def test_fillna_limit():
    s = volas.DataFrame({"x": [None, 1.0, None, None]})["x"]
    assert s.fillna(0.0, limit=2).isna().to_list() == [False, False, False, True]


def test_drop_errors_ignore():
    df = volas.DataFrame({"a": [1.0], "b": [2.0]})
    assert list(df.drop(["z"], axis=1, errors="ignore").columns) == ["a", "b"]
    with pytest.raises(KeyError):
        df.drop(["z"], axis=1)                     # default stays fail-loud (F37)
