"""Spec-to-spec port of pandas's DataFrame method tests.

Ported from pandas/tests/frame/methods/ (head/tail, dropna, sort_index,
reset_index, rename, set_index, astype, copy) — restricted to the all-numeric
frames volas models. Each case builds the same frame in volas and in pandas and
asserts the results agree, so pandas itself is the oracle.

volas detail difference (allowed by the porting brief, root cause noted):
  * ``reset_index(drop=False)`` restores the former index under the conventional
    name ``"index"`` (which is exactly what pandas uses for an *unnamed* index);
    volas's ``Index`` carries no name field, so a name given by ``set_index`` is
    not round-tripped. Pinned in ``test_reset_index_uses_conventional_name``.
"""

import numpy as np
import pandas as pd
import pytest

import volas

nan = float("nan")


def _pair(cols):
    data = {k: list(v) for k, v in cols.items()}
    return volas.DataFrame(data), pd.DataFrame(data)


def _same(vframe, pframe):
    va = np.asarray(vframe.to_numpy(), dtype=float)
    pa = np.asarray(pframe.to_numpy(), dtype=float)
    assert va.shape == pa.shape, f"{va.shape} != {pa.shape}"
    assert np.array_equal(va, pa, equal_nan=True), f"{va.tolist()} != {pa.tolist()}"


D = {"a": [3.0, 1.0, 2.0, nan], "b": [10.0, nan, 30.0, 40.0], "c": [5.0, 6.0, 7.0, 8.0]}


@pytest.mark.parametrize("n", [0, 1, 2, 4, 10])
def test_head_matches_pandas(n):
    v, p = _pair(D)
    _same(v.head(n), p.head(n))


@pytest.mark.parametrize("n", [1, 2, 4, 10])
def test_tail_matches_pandas(n):
    v, p = _pair(D)
    _same(v.tail(n), p.tail(n))


@pytest.mark.parametrize("how", ["any", "all"])
def test_dropna_matches_pandas(how):
    v, p = _pair(D)
    _same(v.dropna(how), p.dropna(how=how))


def test_dropna_all_only_drops_fully_nan_rows():
    cols = {"a": [1.0, nan, nan], "b": [2.0, nan, 5.0]}
    v, p = _pair(cols)
    _same(v.dropna("all"), p.dropna(how="all"))


@pytest.mark.parametrize("ascending", [True, False])
def test_sort_index_matches_pandas(ascending):
    v, p = _pair(D)
    _same(v.sort_index(ascending), p.sort_index(ascending=ascending))


def test_rename_columns_matches_pandas():
    v, p = _pair(D)
    vr = v.rename({"a": "A", "c": "Z"})
    pr = p.rename(columns={"a": "A", "c": "Z"})
    assert vr.columns == list(pr.columns)
    _same(vr, pr)


def test_set_index_moves_column_to_index():
    cols = {"A": [1.0, 2.0, 3.0], "k": [7, 8, 9]}
    v, p = _pair(cols)
    vi, pi = v.set_index("k"), p.set_index("k")
    assert vi.columns == list(pi.columns)  # 'k' removed from the columns
    assert np.asarray(vi.index).tolist() == list(pi.index)
    _same(vi, pi)


def test_astype_to_int_matches_pandas():
    v, p = _pair({"a": [1.0, 2.0, 3.0]})
    vi = v.astype({"a": "int64"})
    assert dict(vi.dtypes)["a"] == "int64"
    _same(vi, p.astype({"a": "int64"}))


def test_dtypes_reports_float_and_bool():
    v = volas.DataFrame({"a": [1.0, 2.0], "f": [True, False]})
    assert dict(v.dtypes) == {"a": "float64", "f": "bool"}


def test_to_numpy_matches_pandas():
    v, p = _pair(D)
    assert np.array_equal(
        np.asarray(v.to_numpy()), np.asarray(p.to_numpy()), equal_nan=True
    )
    assert v.to_numpy().dtype == np.float64


def test_copy_is_independent():
    df = volas.DataFrame({"a": [1.0, 2.0, 3.0]})
    c = df.copy()
    df["a"] = [9.0, 9.0, 9.0]
    assert np.asarray(c["a"].to_numpy(), dtype=float).tolist() == [1.0, 2.0, 3.0]


def test_reset_index_drop_true_matches_pandas():
    cols = {"A": [1.0, 2.0, 3.0], "k": [7, 8, 9]}
    v, p = _pair(cols)
    vr = v.set_index("k").reset_index(True)
    pr = p.set_index("k").reset_index(drop=True)
    assert vr.columns == list(pr.columns)
    _same(vr, pr)


def test_reset_index_uses_conventional_name():
    # volas's Index has no name field, so reset_index restores the former index
    # under the literal "index" label (pandas does the same for an *unnamed*
    # index; the difference only shows after a named set_index).
    df = volas.DataFrame({"A": [1.0, 2.0, 3.0], "k": [7, 8, 9]})
    restored = df.set_index("k").reset_index(False)
    assert restored.columns == ["index", "A"]
    assert np.asarray(restored["index"].to_numpy(), dtype=float).tolist() == [7.0, 8.0, 9.0]
