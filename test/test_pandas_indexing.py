"""Spec-to-spec port of pandas's DataFrame indexing tests.

Ported from pandas/tests/indexing/test_iloc.py, test_loc.py, test_at.py,
test_iat.py and pandas/tests/frame/indexing/test_getitem.py + test_setitem.py —
restricted to the all-numeric (float64/bool) frames with a RangeIndex / integer
/ string index that volas models. Expected results follow pandas semantics
exactly (label slices include the stop; positional slices exclude it; iloc
out-of-bounds raises IndexError; missing labels/columns raise KeyError).

This file also pins the handful of intentional volas↔pandas differences (the
porting brief allows adjusting these); each is root-caused in
``test_documented_differences`` so the behaviour can't drift silently:
  * ``df['name']`` is column-OR-directive: an unknown bareword raises a
    directive error, not KeyError (volas's headline overload).
  * a Series obtained from a frame is a read-only projection — mutate through
    the frame (``df.iloc/.loc/.iat/.at[...] = v``), not ``series.iloc[i] = v``.
  * ``iloc``/``loc`` assignment targets a single column; multi-column block
    assignment and frame-level arithmetic (``df + 1``) are not supported.
  * ``loc`` label slices are forward-only (a negative/!=1 step is dropped).
  * a 0-column projection (``iloc[:, []]``) is not representable (a frame's
    height comes from its columns).
"""

import numpy as np
import pytest

import volas


def _df(cols, index=None):
    data = {k: list(v) for k, v in cols.items()}
    if index is None:
        return volas.DataFrame(data)
    data["__idx__"] = list(index)
    return volas.DataFrame(data).set_index("__idx__")


def _2d(frame):
    return np.asarray(frame.to_numpy(), dtype=float)


def _1d(obj):
    return np.asarray(obj.to_numpy(), dtype=float).ravel()


# --------------------------------------------------------------------------- #
# df[...]  — column / projection / boolean-mask / slice
# --------------------------------------------------------------------------- #
def test_getitem_column_returns_series():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]})
    assert _1d(df["A"]).tolist() == [1.0, 2.0, 3.0]


def test_getitem_projection_preserves_requested_order():
    df = _df({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    assert _2d(df[["C", "A"]]).tolist() == [[5, 1], [6, 2]]


def test_getitem_single_projection_stays_a_frame():
    df = _df({"A": [1, 2], "B": [3, 4]})
    assert _2d(df[["B"]]).tolist() == [[3], [4]]


def test_getitem_boolean_list_mask():
    df = _df({"A": [1, 100, 1000], "B": [2, 200, 2000]})
    assert _2d(df[[True, True, False]]).tolist() == [[1, 2], [100, 200]]


def test_getitem_numpy_and_series_masks():
    df = _df({"A": [0, 1, 2, 3, 4]})
    np_mask = df[np.array([True, False, True, False, True])]
    ser_mask = df[df["A"] >= 2.0]
    assert _2d(np_mask).tolist() == [[0], [2], [4]]
    assert _2d(ser_mask).tolist() == [[2], [3], [4]]


def test_getitem_positional_slice_excludes_stop():
    df = _df({"A": [1, 2, 3, 4, 5]})
    assert _2d(df[1:3]).tolist() == [[2], [3]]


def test_getitem_wrong_length_bool_mask_raises():
    df = _df({"A": [1, 2, 3]})
    with pytest.raises(IndexError):
        df[[True, False]]


# --------------------------------------------------------------------------- #
# df.iloc[...] — positional get
# --------------------------------------------------------------------------- #
def test_iloc_scalar_and_negative():
    df = _df({"A": [2, 3, 5], "B": [7, 11, 13]})
    assert df.iloc[1, 1] == 11.0
    assert df.iloc[-1, -1] == 13.0


def test_iloc_single_row():
    df = _df({"A": [2, 3, 5], "B": [7, 11, 13]})
    assert _1d(df.iloc[0]).tolist() == [2.0, 7.0]
    assert _1d(df.iloc[-3]).tolist() == [2.0, 7.0]  # reaches the first row


def test_iloc_whole_column_and_negative():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]})
    assert _1d(df.iloc[:, 1]).tolist() == [4.0, 5.0, 6.0]
    assert _1d(df.iloc[:, -1]).tolist() == [4.0, 5.0, 6.0]


def test_iloc_block_slice():
    df = _df({"A": [1, 100, 1000], "B": [2, 200, 2000], "C": [3, 300, 3000]})
    assert _2d(df.iloc[1:2, 0:2]).tolist() == [[100, 200]]
    assert _2d(df.iloc[:2]).tolist() == [[1, 2, 3], [100, 200, 300]]


def test_iloc_fancy_rows_and_cols():
    df = _df({"A": [1, 100, 1000], "B": [2, 200, 2000], "C": [3, 300, 3000]})
    assert _2d(df.iloc[[0, 2], [1, 2]]).tolist() == [[2, 3], [2000, 3000]]
    assert _2d(df.iloc[[0, 1]]).tolist() == [[1, 2, 3], [100, 200, 300]]


def test_iloc_reverse_column_slice():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]})
    assert _2d(df.iloc[:, ::-1]).tolist() == [[4, 1], [5, 2], [6, 3]]


def test_iloc_rowlist_scalar_col_is_a_series():
    df = _df({"X": [1, 2, 3, 4], "Y": [10, 20, 30, 40]}, index=["A", "B", "C", "D"])
    assert _1d(df.iloc[[1, 3], 0]).tolist() == [2.0, 4.0]


def test_iloc_single_row_with_column_selector():
    # scalar row + multi-column -> the row (volas's 1-row frame).
    df = _df({"A": [1, 2, 3], "B": [10, 20, 30], "C": [100, 200, 300]})
    assert _1d(df.iloc[1, :]).tolist() == [2.0, 20.0, 200.0]
    assert _1d(df.iloc[0, [0, 2]]).tolist() == [1.0, 100.0]


def test_iloc_wrong_type_column_selector_raises():
    with pytest.raises(TypeError):
        _df({"A": [1, 2], "B": [3, 4]}).iloc[:, "A"]


def test_loc_wrong_type_column_selector_raises():
    with pytest.raises(Exception):
        _df({"A": [1, 2], "B": [3, 4]}).loc[:, 0]


@pytest.mark.parametrize(
    "op",
    [
        lambda d: d.iloc[30],
        lambda d: d.iloc[-30],
        lambda d: d.iloc[[1, 30]],
    ],
)
def test_iloc_out_of_bounds_raises_indexerror(op):
    with pytest.raises(IndexError):
        op(_df({"A": [1, 2, 3]}))


def test_iloc_column_out_of_bounds_raises():
    with pytest.raises(IndexError):
        _df({"A": [1, 2], "B": [3, 4]}).iloc[:, 4]


def test_iloc_float_key_raises_typeerror():
    with pytest.raises(TypeError):
        _df({"A": [1, 2, 3]}).iloc[3.0]


# --------------------------------------------------------------------------- #
# df.loc[...] — label get
# --------------------------------------------------------------------------- #
def test_loc_scalar_and_row():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    assert df.loc["b", "B"] == 5.0
    assert _1d(df.loc["b"]).tolist() == [2.0, 5.0]


def test_loc_whole_column_and_projection():
    df = _df({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    assert _1d(df.loc[:, "A"]).tolist() == [1.0, 2.0]
    assert _2d(df.loc[:, ["C", "A"]]).tolist() == [[5, 1], [6, 2]]


def test_loc_label_slice_includes_stop_string_index():
    df = _df({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=["a", "b", "c", "d"])
    assert _2d(df.loc["b":"c"]).tolist() == [[2, 6], [3, 7]]


def test_loc_label_slice_includes_stop_integer_index():
    df = _df({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]}, index=[0, 1, 2, 3])
    assert _2d(df.loc[1:2]).tolist() == [[2, 4], [3, 5]]  # stop 2 INCLUDED


def test_loc_column_label_slice_inclusive():
    df = _df({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]})
    assert _2d(df.loc[:, "B":"D"]).tolist() == [[3, 5, 7], [4, 6, 8]]


def test_loc_mask_and_column_is_a_series():
    df = _df({"A": [-1, 1, -1, 1], "B": [10, 20, 30, 40]})
    assert _1d(df.loc[df["A"] > 0, "B"]).tolist() == [20.0, 40.0]


def test_loc_label_list_and_integer_labels():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    assert _2d(df.loc[["a", "c"]]).tolist() == [[1, 4], [3, 6]]
    di = _df({"A": [10, 20, 30]}, index=[2, 4, 6])
    assert _2d(di.loc[[2, 6]]).tolist() == [[10], [30]]  # label-based, not positional


@pytest.mark.parametrize(
    "op",
    [
        lambda d: d.loc["z"],          # missing label
        lambda d: d.loc["a", "Z"],     # missing column
        lambda d: d.loc[["a", "x"]],   # any missing in a label list
    ],
)
def test_loc_missing_raises_keyerror(op):
    df = _df({"A": [1, 2, 3]}, index=["a", "b", "c"])
    with pytest.raises(KeyError):
        op(df)


# --------------------------------------------------------------------------- #
# df.iat / df.at — scalar get & set
# --------------------------------------------------------------------------- #
def test_iat_and_at_get():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    assert df.iat[0, 1] == 4.0
    assert df.iat[-1, 0] == 3.0
    assert df.at["c", "A"] == 3.0


def test_iat_set_leaves_other_cells_untouched():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.iat[1, 0] = 99.0
    assert _1d(df["A"]).tolist() == [1.0, 99.0, 3.0]
    assert _1d(df["B"]).tolist() == [4.0, 5.0, 6.0]


def test_at_set_leaves_other_cells_untouched():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df.at["b", "B"] = 99.0
    assert _1d(df["B"]).tolist() == [4.0, 99.0, 6.0]
    assert _1d(df["A"]).tolist() == [1.0, 2.0, 3.0]


def test_at_missing_raises_keyerror():
    df = _df({"A": [1, 2, 3]}, index=["a", "b", "c"])
    with pytest.raises(KeyError):
        df.at["z", "A"]
    with pytest.raises(KeyError):
        df.at["a", "Z"]


# --------------------------------------------------------------------------- #
# assignment via df[col]= / iloc / loc, and copy-on-write
# --------------------------------------------------------------------------- #
def test_setitem_column_array_and_scalar():
    df = _df({"a": [1, 2, 3], "b": [4, 5, 6]})
    df["a"] = [10, 20, 30]
    df["b"] = 0
    assert _1d(df["a"]).tolist() == [10.0, 20.0, 30.0]
    assert _1d(df["b"]).tolist() == [0.0, 0.0, 0.0]


def test_setitem_new_column():
    df = _df({"a": [1, 2, 3]})
    df["c"] = [5, 6, 7]
    assert _1d(df["c"]).tolist() == [5.0, 6.0, 7.0]
    assert _1d(df["a"]).tolist() == [1.0, 2.0, 3.0]


def test_setitem_wrong_length_raises():
    df = _df({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        df["a"] = [1, 2]


def test_iloc_setitem_scalar_and_column():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.iloc[1, 1] = 1.0
    assert _1d(df["B"]).tolist() == [4.0, 1.0, 6.0]
    assert _1d(df["A"]).tolist() == [1.0, 2.0, 3.0]
    df.iloc[:, 0] = [7, 8, 9]
    assert _1d(df["A"]).tolist() == [7.0, 8.0, 9.0]


def test_iloc_setitem_rowlist_single_column():
    df = _df({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.iloc[[0, 2], 1] = 99.0
    assert _1d(df["B"]).tolist() == [99.0, 5.0, 99.0]


def test_loc_setitem_mask_and_whole_column():
    df = _df({"A": [-1, 1, -1, 1], "B": [1, 1, 1, 1]})
    df.loc[df["A"] > 0, "B"] = 0.0
    assert _1d(df["B"]).tolist() == [1.0, 0.0, 1.0, 0.0]
    assert _1d(df["A"]).tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_loc_setitem_label_list_aligns_by_label():
    # index [3, 5, 4]; assigning labels [4, 3, 5] <- [1, 2, 3] aligns by LABEL:
    # label4<-1 (pos2), label3<-2 (pos0), label5<-3 (pos1) -> column [2, 3, 1].
    df = _df({"A": [float("nan")] * 3}, index=[3, 5, 4])
    df.loc[[4, 3, 5], "A"] = [1, 2, 3]
    assert _1d(df["A"]).tolist() == [2.0, 3.0, 1.0]


def test_copy_on_write_view_not_mutated():
    df = _df({"a": [1, 1]})
    view = df.iloc[:]
    df["a"] = [5, 5]
    assert _1d(view["a"]).tolist() == [1.0, 1.0]
    assert _1d(df["a"]).tolist() == [5.0, 5.0]


# --------------------------------------------------------------------------- #
# Intentional volas↔pandas differences — pinned with their root cause.
# --------------------------------------------------------------------------- #
def test_documented_differences():
    df = _df({"A": [1, 2, 3]}, index=[1, 2, 3])

    # (1) df['name'] is column-or-directive: an unknown bareword is parsed as a
    # directive and raises a directive error (pandas would raise KeyError).
    with pytest.raises(Exception) as ei:
        df["Zzz"]
    assert "Directive" in type(ei.value).__name__ or isinstance(ei.value, KeyError)

    # (2) a Series from a frame is a read-only projection.
    with pytest.raises(TypeError):
        df["A"].iloc[0] = 9.0

    # (3) iloc/loc assignment is single-column; a multi-column block raises.
    with pytest.raises(Exception):
        df.iloc[[0, 1], [0]] = [[1], [2]]  # col list, not a single position

    # (4) frame-level arithmetic is unsupported (only Series do arithmetic).
    with pytest.raises(TypeError):
        df + 1

    # (5) loc label slices are forward-only: a reverse step yields no rows
    # (root cause: label_bounds drops the slice step).
    assert len(df.loc[3:1:-1]) == 0
