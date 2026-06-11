"""Systematic audit — cross-cutting findings from the v0.3 cross-reviews (P8).

Bugs the reviewers' aggressive census surfaced that the matrix missed, plus the
new DataFrame(index=) requirement. All pinned as strict-xfail per owner
decisions (decision 8); no volas runtime change.

  F36  where/mask align by position, not by label -> silent wrong filtering
  F37  drop silently ignores a missing label (should KeyError)
  F39  reset_index / rename can produce duplicate column names (violates the
       unique-column contract)
  F45  DataFrame(data, index=...) is not supported (new align-backlog)
"""

from __future__ import annotations

import pytest

import volas


# F36 (FIXED): where/mask pairs the condition frame by NAME, never by position.
# A same-set/different-order condition is reordered to match (the correct
# result); a different name set or different index is an error.
def test_where_aligns_by_name():
    prices = volas.DataFrame({"AAPL": [150.0, 152.0], "MSFT": [300.0, 305.0]})
    signal = volas.DataFrame({"MSFT": [True, False], "AAPL": [False, True]})  # order differs
    out = prices.where(signal)
    # AAPL uses AAPL's signal [False, True]; MSFT uses MSFT's [True, False].
    assert out["AAPL"].isna().to_list() == [True, False]
    assert out["MSFT"].isna().to_list() == [False, True]


def test_where_different_column_set_raises():
    prices = volas.DataFrame({"AAPL": [150.0, 152.0], "MSFT": [300.0, 305.0]})
    cond = volas.DataFrame({"AAPL": [True, False], "TSLA": [True, False]})
    with pytest.raises(ValueError):
        prices.where(cond)


# F37 (decision 3, FIXED): drop of a missing label raises KeyError (was a silent
# no-op). fail-loud (C4).
def test_drop_missing_label_raises():
    df = volas.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(KeyError):
        df.drop(["z"], axis=1)


# F39 (decision 3, FIXED): three entry points must not breach the unique-column
# contract — reset_index name collision and rename collision now raise.
def test_reset_index_name_collision_raises():
    df = volas.DataFrame({"index": [9.0], "a": [1.0]})   # 'index' collides with the reset name
    with pytest.raises((ValueError, KeyError)):
        df.reset_index()


def test_rename_collision_raises():
    df = volas.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises((ValueError, KeyError)):
        df.rename({"b": "a"})                            # b -> a collides with existing a


# F45 (FIXED): DataFrame(data, index=...) attaches explicit row labels at
# construction (same kinds + uniqueness rules as set_index).
def test_dataframe_index_kwarg():
    df = volas.DataFrame({"a": [1.0, 2.0]}, index=[10, 20])
    assert df.reset_index().iloc[:, 0].to_list() == [10, 20]


# F40 (decision 4, FIXED): a non-binary numeric fill into a bool column raises
# (a bool column stays bool; C3/C4) — was a silent promotion to float64.
def test_bool_fillna_non_binary_raises():
    s = volas.DataFrame({"x": [True, None, False]})["x"]
    with pytest.raises((TypeError, ValueError)):
        s.fillna(2)


# F42 (FIXED): fillna accepts pd.NA like np.nan / volas.NA — the NA synonyms
# are uniform at the boundary.
def test_fillna_pd_na_accepted():
    import pandas as pd
    s = volas.DataFrame({"x": [1.0, None]})["x"]
    assert s.fillna(pd.NA).isna().to_list() == [False, True]   # NA fill -> identity
