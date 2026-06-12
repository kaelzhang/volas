"""Systematic audit — T6 (order): sort_values / sort_index / rank / unique /
head / tail / drop / reset_index / set_index.

The high-value invariants are value-faithful ordering and NA *position*: a sort
sinks NA to the end in both directions (V9-class), rank leaves NA as NA, unique
keeps a single NA.

Cell IDs:  T6.<method>[/D=<d>] · T6.sort/na-last
"""

from __future__ import annotations

import volas


def _present(s):
    return [x for x, m in zip(s.to_list(), s.isna().to_list()) if not m]


# --- sort_values: ordered values + NA sunk last ----------------------------
def test_sort_values_sinks_na_both_directions():
    s = volas.DataFrame({"x": [3.0, None, 1.0, 2.0]})["x"]
    asc = s.sort_values()
    assert _present(asc) == [1.0, 2.0, 3.0]
    assert asc.isna().to_list()[-1] is True            # NA last regardless of order
    desc = s.sort_values(ascending=False)
    assert _present(desc) == [3.0, 2.0, 1.0]
    assert desc.isna().to_list()[-1] is True


def test_sort_values_str_datetime():
    ss = volas.DataFrame({"x": ["c", "a", "b"]})["x"].sort_values()
    assert ss.to_list() == ["a", "b", "c"]


# --- rank: NA stays NA, ties/order otherwise --------------------------------
def test_rank_leaves_na():
    r = volas.DataFrame({"x": [3.0, None, 1.0, 2.0]})["x"].rank()
    assert r.isna().to_list() == [False, True, False, False]
    assert [x for x, m in zip(r.to_list(), r.isna().to_list()) if not m] == [3.0, 1.0, 2.0]


# --- unique: distinct values, a single NA ----------------------------------
def test_unique_keeps_single_na():
    u = volas.DataFrame({"x": ["a", "a", "b", None]})["x"].unique()
    vals = [x for x, m in zip(u.to_list(), u.isna().to_list()) if not m]
    assert vals == ["a", "b"]
    assert u.isna().to_list().count(True) == 1         # exactly one NA kept


# --- head / tail ------------------------------------------------------------
def test_head_tail():
    s = volas.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})["x"]
    assert s.head(2).to_list() == [1.0, 2.0]
    assert s.tail(2).to_list() == [3.0, 4.0]


# --- drop / reset_index / set_index (DataFrame structure) ------------------
def test_drop_column():
    df = volas.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    dropped = df.drop(["a"], axis=1)
    assert list(dropped.columns) == ["b"]
    assert list(df.columns) == ["a", "b"]              # original untouched


def test_set_then_reset_index_roundtrip():
    # set_index accepts int64 / datetime / string keys (a float key is a clean
    # TypeError — only valid index kinds, never a panic).
    df = volas.DataFrame({"a": [10, 20], "b": [3.0, 4.0]})
    si = df.set_index("a")
    assert list(si.columns) == ["b"]                   # 'a' became the index
    back = si.reset_index()
    assert "a" in list(back.columns)                   # 'a' restored as a column


def test_set_index_rejects_float_key():
    import pytest
    df = volas.DataFrame({"a": [10.0, 20.0], "b": [3.0, 4.0]})
    with pytest.raises(TypeError):
        df.set_index("a")                              # float64 is not a valid index kind
