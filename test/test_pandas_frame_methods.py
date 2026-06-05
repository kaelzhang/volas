"""Spec-to-spec port of pandas's DataFrame method tests.

Ported from pandas/tests/frame/methods/ (head/tail, dropna, sort_index,
reset_index, rename, set_index, astype, copy) — restricted to the all-numeric
frames volas models. Expected values follow pandas semantics and are inlined
(derived from the fixed ``ROWS`` matrix), so the suite is pandas-free.

volas detail difference (allowed by the porting brief, root cause noted):
  * ``reset_index(drop=False)`` restores the former index under the conventional
    name ``"index"`` (which is exactly what pandas uses for an *unnamed* index);
    volas's ``Index`` carries no name field, so a name given by ``set_index`` is
    not round-tripped. Pinned in ``test_reset_index_uses_conventional_name``.
"""

import numpy as np
import pytest

import volas

nan = float("nan")

# The shared fixture, and its value matrix (row-major: a, b, c).
D = {"a": [3.0, 1.0, 2.0, nan], "b": [10.0, nan, 30.0, 40.0], "c": [5.0, 6.0, 7.0, 8.0]}
ROWS = [[3.0, 10.0, 5.0], [1.0, nan, 6.0], [2.0, 30.0, 7.0], [nan, 40.0, 8.0]]


def _df():
    return volas.DataFrame({k: list(v) for k, v in D.items()})


def _eq(frame, expected):
    got = np.asarray(frame.to_numpy(), dtype=float).reshape(-1)
    exp = np.asarray(expected, dtype=float).reshape(-1)
    assert got.shape == exp.shape, f"{got.shape} != {exp.shape}"
    assert np.array_equal(got, exp, equal_nan=True), f"{got.tolist()} != {exp.tolist()}"


@pytest.mark.parametrize("n", [0, 1, 2, 4, 10])
def test_head(n):
    _eq(_df().head(n), ROWS[:n])


@pytest.mark.parametrize("n", [1, 2, 4, 10])
def test_tail(n):
    _eq(_df().tail(n), ROWS[-n:])


def test_dropna_any_drops_rows_with_any_nan():
    _eq(_df().dropna("any"), [ROWS[0], ROWS[2]])   # rows 1 & 3 have a NaN


def test_dropna_all_keeps_partial_rows():
    _eq(_df().dropna("all"), ROWS)                  # no row is entirely NaN


def test_dropna_all_drops_only_fully_nan_rows():
    df = volas.DataFrame({"a": [1.0, nan, nan], "b": [2.0, nan, 5.0]})
    _eq(df.dropna("all"), [[1.0, 2.0], [nan, 5.0]])  # only row 1 (all NaN) dropped


def test_sort_index_ascending_is_identity_on_rangeindex():
    _eq(_df().sort_index(True), ROWS)


def test_sort_index_descending_reverses_rows():
    _eq(_df().sort_index(False), ROWS[::-1])


def test_rename_columns():
    r = _df().rename({"a": "A", "c": "Z"})
    assert r.columns == ["A", "b", "Z"]
    _eq(r, ROWS)                                    # values unchanged


def test_set_index_moves_column_to_index():
    v = volas.DataFrame({"A": [1.0, 2.0, 3.0], "k": [7, 8, 9]}).set_index("k")
    assert v.columns == ["A"]                       # 'k' removed from the columns
    assert np.asarray(v.index).tolist() == [7, 8, 9]
    _eq(v, [[1.0], [2.0], [3.0]])


def test_astype_to_int():
    v = volas.DataFrame({"a": [1.0, 2.0, 3.0]}).astype({"a": "int64"})
    assert dict(v.dtypes)["a"] == "int64"
    _eq(v, [[1.0], [2.0], [3.0]])


def test_dtypes_reports_float_and_bool():
    v = volas.DataFrame({"a": [1.0, 2.0], "f": [True, False]})
    assert dict(v.dtypes) == {"a": "float64", "f": "bool"}


def test_to_numpy_is_float64_matrix():
    v = _df()
    assert v.to_numpy().dtype == np.float64
    _eq(v, ROWS)


def test_copy_is_independent():
    df = volas.DataFrame({"a": [1.0, 2.0, 3.0]})
    c = df.copy()
    df["a"] = [9.0, 9.0, 9.0]
    assert np.asarray(c["a"].to_numpy(), dtype=float).tolist() == [1.0, 2.0, 3.0]


def test_reset_index_drop_true():
    v = volas.DataFrame({"A": [1.0, 2.0, 3.0], "k": [7, 8, 9]}).set_index("k")
    r = v.reset_index(True)
    assert r.columns == ["A"]
    assert np.asarray(r.index).tolist() == [0, 1, 2]
    _eq(r, [[1.0], [2.0], [3.0]])


def test_reset_index_uses_conventional_name():
    # volas's Index has no name field, so reset_index restores the former index
    # under the literal "index" label (pandas does the same for an *unnamed*
    # index; the difference only shows after a named set_index).
    df = volas.DataFrame({"A": [1.0, 2.0, 3.0], "k": [7, 8, 9]})
    restored = df.set_index("k").reset_index(False)
    assert restored.columns == ["index", "A"]
    assert np.asarray(restored["index"].to_numpy(), dtype=float).tolist() == [7.0, 8.0, 9.0]
