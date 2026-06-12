"""Systematic audit — the R-1..R-6 census round (the kwargs-debt recurrence).

The cross-review's aggressive per-parameter census on the NEWLY implemented
APIs found the same family of gaps F44 fixed on the old ones. Pinned here:
negative ints are clean ValueErrors (never pyo3's OverflowError leak), the
core kwargs exist and behave, and the intentional divergences are waived.
"""

from __future__ import annotations

import pytest

import volas

_s = lambda: volas.DataFrame({"x": [3.0, 1.0, 2.0, 2.0]})["x"]
_df = lambda: volas.DataFrame({"a": [1.0, 1.0, 2.0], "b": [9.0, 9.0, 1.0]})


# R-1: negative ints -> clean ValueError (the head(-1) OverflowError lesson)
@pytest.mark.parametrize("call", [
    lambda: _s().rolling(-1),
    lambda: _s().rolling(2, min_periods=-1),
    lambda: _s().expanding(-1),
    lambda: _s().nlargest(-1),
    lambda: _s().nsmallest(-1),
    lambda: _s().fillna(0.0, limit=-1),
    lambda: _df().rolling(-1),
    lambda: _df().expanding(-1),
    lambda: _df().nlargest(-1, "a"),
    lambda: _df().nsmallest(-1, "a"),
    lambda: _df().fillna(0.0, limit=-1),
], ids=["s.rolling", "s.min_periods", "s.expanding", "s.nlargest", "s.nsmallest",
        "s.fillna_limit", "df.rolling", "df.expanding", "df.nlargest",
        "df.nsmallest", "df.fillna_limit"])
def test_negative_int_params_raise_valueerror(call):
    with pytest.raises(ValueError):
        call()


# R-3: between(inclusive=)
def test_between_inclusive():
    s = volas.DataFrame({"x": [1.0, 2.0, 3.0]})["x"]
    assert s.between(1, 3).to_list() == [True, True, True]            # both
    assert s.between(1, 3, inclusive="left").to_list() == [True, True, False]
    assert s.between(1, 3, inclusive="right").to_list() == [False, True, True]
    assert s.between(1, 3, inclusive="neither").to_list() == [False, True, False]
    with pytest.raises(ValueError):
        s.between(1, 3, inclusive="middle")


# R-4: drop_duplicates/duplicated keep=
def test_keep_last():
    s = volas.DataFrame({"x": [1.0, 2.0, 1.0]})["x"]
    assert s.duplicated(keep="first").to_list() == [False, False, True]
    assert s.duplicated(keep="last").to_list() == [True, False, False]
    assert s.drop_duplicates(keep="last").to_list() == [2.0, 1.0]
    df = _df()
    assert df.duplicated(keep="last").to_list() == [True, False, False]
    assert df.drop_duplicates(keep="last").shape == (2, 2)
    with pytest.raises(ValueError):
        s.duplicated(keep="middle")


# R-2: value_counts kwargs
def test_value_counts_kwargs():
    s = volas.DataFrame({"x": ["a", "b", "a", "a"]})["x"]
    norm = s.value_counts(normalize=True)
    assert norm.loc["a"] == pytest.approx(0.75)
    asc = s.value_counts(ascending=True)
    assert asc.to_list() == [1, 3]                       # least frequent first
    # dropna=False: a volas index has no missing-label slot -> clean error,
    # never a silent drop (documented divergence; use isna().sum()).
    with pytest.raises(ValueError):
        s.value_counts(dropna=False)


# DataFrame/Series symmetry (review-2 F3): rank na_option + fillna limit
def test_frame_kwarg_symmetry():
    df = volas.DataFrame({"a": [3.0, None, 1.0]})
    assert df.rank(na_option="top")["a"].to_list() == [3.0, 1.0, 2.0]
    assert df.fillna(0.0, limit=0)["a"].isna().to_list() == [False, True, False]


# R-5 (waiver): rolling(0) is a fail-loud ValueError — pandas instead returns
# all-NaN, which silently produces a useless column. Documented divergence.
def test_rolling_zero_waiver():
    with pytest.raises(ValueError):
        _s().rolling(0)


# R-6 (waiver): ewm takes span only — the quant convention; pandas's
# com/halflife/alpha quartet is out-of-scope (convert: span = 2/alpha - 1).
def test_ewm_span_only_waiver():
    import inspect
    assert set(inspect.signature(_s().ewm).parameters) == {"span"}
