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


# F36 (decision 2): where/mask must align the condition frame by column/label,
# not by position. A condition frame with the same names in a DIFFERENT ORDER
# currently mis-applies each column's mask. Decision: fail-loud unless the
# condition's columns match exactly (full label-alignment is a larger backlog).
@pytest.mark.xfail(reason="F36: where aligns by position, mis-applies mismatched-order condition", strict=True)
def test_where_mismatched_columns_raises():
    prices = volas.DataFrame({"AAPL": [150.0, 152.0], "MSFT": [300.0, 305.0]})
    signal = volas.DataFrame({"MSFT": [True, False], "AAPL": [False, True]})  # order differs
    with pytest.raises((ValueError, KeyError)):
        prices.where(signal)


# F37 (decision 3): drop of a missing label is a silent no-op; pandas raises
# KeyError. Should fail-loud (C4).
@pytest.mark.xfail(reason="F37: drop silently ignores a missing label (should KeyError)", strict=True)
def test_drop_missing_label_raises():
    df = volas.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(KeyError):
        df.drop(["z"], axis=1)


# F39 (decision 3): three entry points can breach the unique-column contract.
@pytest.mark.xfail(reason="F39: reset_index name collision produces duplicate columns", strict=True)
def test_reset_index_name_collision_raises():
    df = volas.DataFrame({"index": [9.0], "a": [1.0]})   # 'index' collides with the reset name
    with pytest.raises((ValueError, KeyError)):
        df.reset_index()


@pytest.mark.xfail(reason="F39: rename into an existing name produces duplicate columns", strict=True)
def test_rename_collision_raises():
    df = volas.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises((ValueError, KeyError)):
        df.rename({"b": "a"})                            # b -> a collides with existing a


# F45 (new requirement): DataFrame(data, index=...) — explicit index at
# construction (pandas standard). Currently TypeError: unexpected keyword.
@pytest.mark.xfail(reason="F45: DataFrame(data, index=...) not supported", strict=True)
def test_dataframe_index_kwarg():
    df = volas.DataFrame({"a": [1.0, 2.0]}, index=[10, 20])
    assert df.reset_index().iloc[:, 0].to_list() == [10, 20]


# F40 (decision 4): a non-binary numeric fill into a bool column must raise (a
# bool column stays bool; C3/C4 dtype honesty). Currently fillna(2) silently
# becomes float64 [1.0, 2.0, 0.0], destroying the bool semantics. xfail(strict).
@pytest.mark.xfail(reason="F40: bool.fillna(non-binary) should raise, volas -> float64", strict=True)
def test_bool_fillna_non_binary_raises():
    s = volas.DataFrame({"x": [True, None, False]})["x"]
    with pytest.raises((TypeError, ValueError)):
        s.fillna(2)


# F42 (NA-synonym gap): fillna accepts np.nan and volas.NA but rejects pd.NA
# with TypeError. pd.NA is a legitimate missing-value synonym and should
# propagate (all-NA), like the others. xfail(strict).
@pytest.mark.xfail(reason="F42: fillna(pd.NA) rejected (np.nan/volas.NA accepted)", strict=True)
def test_fillna_pd_na_accepted():
    import pandas as pd
    s = volas.DataFrame({"x": [1.0, None]})["x"]
    assert s.fillna(pd.NA).isna().to_list() == [False, True]   # NA fill -> identity
