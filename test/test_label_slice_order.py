"""Label slices (`loc[lo:hi]`) follow the index's label order (pandas 3.0.3).

int and str indexes only need unique labels, so they may be in any order. A
slice searches by order only when the index is monotonic: increasing labels
select ``lo <= label <= hi``, decreasing labels select ``lo >= label >= hi``.
On an index that is neither, each endpoint is located by its exact position
and a missing (or repeated) endpoint raises ``KeyError``, so a slice never
selects or overwrites rows outside the requested labels.
"""

import numpy as np
import pandas as pd
import pytest

import volas

CASES = [
    # unordered int index: endpoints by position
    ([2, 0, 1], 0, 1),
    ([2, 0, 1], 1, 0),
    ([2, 0, 1], None, 0),
    ([2, 0, 1], 0, None),
    ([2, 0, 1], 2, 2),
    ([2, 0, 1], 0, 5),
    ([2, 0, 1], -1, None),
    ([2, 0, 1], None, 9),
    # decreasing int index: ordered search, reversed direction
    ([3, 2, 1], 2, 1),
    ([3, 2, 1], 1, 2),
    ([3, 2, 1], 5, 0),
    ([3, 2, 1], None, 2),
    ([3, 2, 1], 2, None),
    # increasing int index: ordered search, missing endpoints allowed
    ([1, 2, 3], 0, 5),
    ([1, 2, 3], 3, 1),
    ([1, 2, 3], 2, None),
    ([5], 9, 10),
    # str indexes, same three orders
    (["c", "a", "b"], "a", "b"),
    (["c", "a", "b"], "a", "z"),
    (["c", "a", "b"], "0", None),
    (["c", "b", "a"], "b", "a"),
    (["c", "b", "a"], "bb", ""),
    (["a", "b", "c"], "aa", "z"),
]


def _ids(case):
    idx, lo, hi = case
    return f"{idx}[{lo!r}:{hi!r}]"


def _oracle(labels, lo, hi):
    """The pandas selection as a row-position list, or ``KeyError``."""
    s = pd.Series(np.arange(len(labels)), index=labels)
    try:
        return s.loc[lo:hi].to_list()
    except KeyError:
        return KeyError


def _frame(labels):
    x = list(range(10, 10 + len(labels)))
    return volas.DataFrame({"x": x}, index=labels)


def _positions(labels, x):
    """Map selected ``x`` values (10 + position) back to row positions."""
    return [v - 10 for v in x]


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_frame_loc_slice_matches_pandas(case):
    labels, lo, hi = case
    want = _oracle(labels, lo, hi)
    df = _frame(labels)
    if want is KeyError:
        with pytest.raises(KeyError, match="slice bound"):
            df.loc[lo:hi]
        return
    got = df.loc[lo:hi]
    assert _positions(labels, got["x"].to_list()) == want
    assert got.index.tolist() == [labels[p] for p in want]


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_frame_loc_2d_slice_matches_pandas(case):
    labels, lo, hi = case
    want = _oracle(labels, lo, hi)
    df = _frame(labels)
    if want is KeyError:
        with pytest.raises(KeyError, match="slice bound"):
            df.loc[lo:hi, "x"]
        return
    assert _positions(labels, df.loc[lo:hi, "x"].to_list()) == want


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_series_loc_slice_matches_pandas(case):
    labels, lo, hi = case
    want = _oracle(labels, lo, hi)
    s = _frame(labels)["x"]
    if want is KeyError:
        with pytest.raises(KeyError, match="slice bound"):
            s.loc[lo:hi]
        return
    assert _positions(labels, s.loc[lo:hi].to_list()) == want


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_loc_slice_assignment_writes_only_selected_rows(case):
    labels, lo, hi = case
    want = _oracle(labels, lo, hi)
    df = _frame(labels)
    before = df["x"].to_list()
    if want is KeyError:
        with pytest.raises(KeyError, match="slice bound"):
            df.loc[lo:hi, "x"] = 99
        assert df["x"].to_list() == before  # nothing written
        return
    df.loc[lo:hi, "x"] = 99
    expected = [99 if p in want else v for p, v in enumerate(before)]
    assert df["x"].to_list() == expected


@pytest.mark.parametrize(
    "case", [c for c in CASES if isinstance(c[0][0], str)], ids=_ids
)
def test_frame_getitem_string_slice_matches_pandas(case):
    labels, lo, hi = case
    want = _oracle(labels, lo, hi)
    df = _frame(labels)
    if want is KeyError:
        with pytest.raises(KeyError, match="slice bound"):
            df[lo:hi]
        return
    assert _positions(labels, df[lo:hi]["x"].to_list()) == want


def test_reported_unsorted_overwrite():
    # the reported case: every row was overwritten, pandas writes two
    d = volas.DataFrame({"x": [20, 0, 10]}, index=[2, 0, 1])
    assert d.loc[0:1].index.tolist() == [0, 1]
    d.loc[0:1, "x"] = 99
    assert d["x"].to_list() == [20, 99, 99]
    with pytest.raises(KeyError, match="non-monotonic index with a missing label 5"):
        d.loc[0:5]


def test_order_follows_appended_labels():
    # the index keeps its order fact as rows are appended
    d = volas.DataFrame({"x": [1, 2]}, index=[1, 2])
    assert d.loc[0:9]["x"].to_list() == [1, 2]  # increasing: missing bounds fine
    d.append(volas.DataFrame({"x": [3]}, index=[0]))
    assert d.index.tolist() == [1, 2, 0]
    with pytest.raises(KeyError, match="missing label 9"):
        d.loc[0:9]
    assert d.loc[2:0]["x"].to_list() == [2, 3]


def test_datetime_index_increasing_search():
    t = volas.to_datetime(volas.DataFrame({"t": ["2020-01-01", "2020-01-02", "2020-01-04"]})["t"])
    d = volas.DataFrame({"x": [1, 2, 3]}, index=t)
    assert d.loc["2020-01-02":"2020-01-03"]["x"].to_list() == [2]
    assert d.loc["2019-12-31":"2020-01-09"]["x"].to_list() == [1, 2, 3]
