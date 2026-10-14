"""Label slices (`loc[lo:hi]`) follow the index's label order (pandas 3.0.3).

int and str indexes only need unique labels, so they may be in any order. A
slice searches by order only when the index is monotonic: increasing labels
select ``lo <= label <= hi``, decreasing labels select ``lo >= label >= hi``.
On an index that is neither, each endpoint is located by its exact position
and a missing (or repeated) endpoint raises ``KeyError``, so a slice never
selects or overwrites rows outside the requested labels.
"""

import re

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


def _positions(x):
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
    assert _positions(got["x"].to_list()) == want
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
    assert _positions(df.loc[lo:hi, "x"].to_list()) == want


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_series_loc_slice_matches_pandas(case):
    labels, lo, hi = case
    want = _oracle(labels, lo, hi)
    s = _frame(labels)["x"]
    if want is KeyError:
        with pytest.raises(KeyError, match="slice bound"):
            s.loc[lo:hi]
        return
    assert _positions(s.loc[lo:hi].to_list()) == want


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
    assert _positions(df[lo:hi]["x"].to_list()) == want


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


def _day(d):
    return pd.Timestamp(f"2024-01-0{d}")


DT_INDEXES = {
    "increasing": [1, 2, 4, 6],
    "decreasing": [6, 4, 2, 1],
    "unordered": [4, 1, 6, 2],
    "nat": [None, 1, 2, 4],
}
DT_BOUNDS = [None, 1, 2, 3, 4, 6, 9]


def _dt_frames(days):
    labels = pd.DatetimeIndex([pd.NaT if d is None else _day(d) for d in days])
    vals = list(range(10, 10 + len(days)))
    return pd.DataFrame({"x": vals}, index=labels), volas.DataFrame({"x": vals}, index=labels)


def _outcome(f):
    try:
        return f()
    except KeyError:
        return KeyError


@pytest.mark.parametrize("name", DT_INDEXES)
def test_timestamp_bounds_on_datetime_indexes_match_pandas(name):
    # an exact Timestamp bound follows the index's order, like an int label
    for lo in DT_BOUNDS:
        for hi in DT_BOUNDS:
            lo_t = None if lo is None else _day(lo)
            hi_t = None if hi is None else _day(hi)
            pdf, vdf = _dt_frames(DT_INDEXES[name])
            want = _outcome(lambda: pdf.loc[lo_t:hi_t, "x"].to_list())
            assert _outcome(lambda: vdf.loc[lo_t:hi_t, "x"].to_list()) == want, (lo, hi)
            assert _outcome(lambda: vdf["x"].loc[lo_t:hi_t].to_list()) == want, (lo, hi)
            if want is not KeyError:
                pdf.loc[lo_t:hi_t, "x"] = 99
                vdf.loc[lo_t:hi_t, "x"] = 99
                assert vdf["x"].to_list() == pdf["x"].to_list(), (lo, hi)


@pytest.mark.parametrize("days", [(1, 2, 2, 4), (2, 4, 2, 1)])
def test_a_datetime_index_never_repeats_a_label(days):
    # a label names one row: any repeat raises, consecutive or not
    with pytest.raises(ValueError, match="duplicate labels"):
        volas.DataFrame({"x": [10, 11, 12, 13]}, index=pd.DatetimeIndex([_day(d) for d in days]))


PARTIAL = "Value based partial slicing on non-monotonic DatetimeIndexes"


@pytest.mark.parametrize("name", ["decreasing", "unordered", "nat"])
def test_string_bounds_on_a_non_increasing_datetime_index_fail_closed(name):
    # pandas reads a string bound there as a partial-date value and may select
    # non-contiguous rows; volas raises rather than select different rows
    _, vdf = _dt_frames(DT_INDEXES[name])
    before = vdf["x"].to_list()
    for lo, hi in [("2024-01-02", None), (None, "2024-01-04"), ("2024-01-02", "2024-01-04")]:
        with pytest.raises(KeyError, match=PARTIAL):
            vdf.loc[lo:hi]
        with pytest.raises(KeyError, match=PARTIAL):
            vdf["x"].loc[lo:hi]
        with pytest.raises(KeyError, match=PARTIAL):
            vdf[lo:hi]
        with pytest.raises(KeyError, match=PARTIAL):
            vdf.loc[lo:hi, "x"] = 99
    # a Timestamp with a string on the other side is still a text slice
    with pytest.raises(KeyError, match=PARTIAL):
        vdf.loc[_day(1):"2024-01-04"]
    assert vdf["x"].to_list() == before  # nothing written


def test_reported_partial_string_overwrite():
    # the reviewed case: volas wrote one row where pandas selects three
    _, vdf = _dt_frames([4, 1, 6, 2])
    with pytest.raises(KeyError, match=PARTIAL):
        vdf.loc["2024-01-02":, "x"] = 0
    assert vdf["x"].to_list() == [10, 11, 12, 13]


def test_string_bounds_on_an_increasing_datetime_index_still_search_by_order():
    pdf, vdf = _dt_frames([1, 2, 4, 6])
    assert vdf.loc["2024-01-02":"2024-01-05"]["x"].to_list() == [11, 12]
    assert vdf["2024-01-02":]["x"].to_list() == pdf["2024-01-02":]["x"].to_list()


@pytest.mark.parametrize(
    "labels, lo, hi, indexer",
    [
        (["c", "a", "b"], 0, 1, "[0] of type int"),
        (["a", "b", "c"], None, 2, "[2] of type int"),
        ([2, 0, 1], "a", "b", "[a] of type str"),
        ([0, 1, 2], None, "b", "[b] of type str"),
        (None, "a", None, "[a] of type str"),  # RangeIndex
    ],
)
def test_a_bound_of_the_other_label_kind_is_a_type_error(labels, lo, hi, indexer):
    df = volas.DataFrame({"x": [1, 2, 3]}, index=labels)
    if isinstance(lo or hi, int):
        # pandas raises this TypeError for an int bound on a string index; for a
        # string bound on an int index it is inconsistent (KeyError when unordered,
        # an empty frame on a RangeIndex, a ufunc TypeError when increasing), so
        # volas raises the one TypeError there
        with pytest.raises(TypeError, match="cannot do slice indexing"):
            pd.DataFrame({"x": [1, 2, 3]}, index=labels).loc[lo:hi]
    text = "cannot do slice indexing on .* with these indexers " + re.escape(indexer)
    for read in (lambda: df.loc[lo:hi], lambda: df["x"].loc[lo:hi], lambda: df.loc[lo:hi, "x"]):
        with pytest.raises(TypeError, match=text):
            read()
    with pytest.raises(TypeError, match=text):
        df.loc[lo:hi, "x"] = 0
    assert df["x"].to_list() == [1, 2, 3]
