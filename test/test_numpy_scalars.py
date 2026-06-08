"""Boundary scalar returns are numpy-typed, matching pandas 3.0.

Direct scalar access (`s[i]`, `iloc`, `loc`, `iat`, `at`, a Row's `[col]`) returns
a numpy scalar of the column dtype (np.float64 / np.int64 / np.bool_). Bulk
materialization (`to_list` / iteration / `to_dict`) stays native Python, like
pandas tolist().
"""

import numpy as np

import volas


def _df():
    return volas.DataFrame({"a": [1.5, 2.5, 3.5], "b": np.array([10, 20, 30], dtype=np.int64)})


# --- direct scalar access -> numpy scalar -----------------------------------

def test_series_getitem_is_numpy_scalar():
    df = _df()
    assert isinstance(df["a"][0], np.float64) and df["a"][0] == 1.5
    assert isinstance(df["b"][0], np.int64) and df["b"][0] == 10


def test_iloc_loc_are_numpy_scalars():
    df = _df()
    assert isinstance(df["a"].iloc[1], np.float64)
    assert isinstance(df["b"].iloc[1], np.int64)
    # loc by the (range) label
    assert isinstance(df["a"].loc[1], np.float64)


def test_frame_iat_at_are_numpy_scalars():
    df = _df()
    assert isinstance(df.iat[0, 0], np.float64)
    assert isinstance(df.iat[0, 1], np.int64)
    assert isinstance(df.at[df.index[0], "a"], np.float64)


def test_row_getitem_is_numpy_scalar():
    df = _df()
    row = df.iloc[0]
    assert isinstance(row["a"], np.float64)
    assert isinstance(row["b"], np.int64)


def test_bool_scalar_is_numpy_bool():
    s = volas.DataFrame({"a": [1.0, 0.0]})["a"] > 0.5
    assert isinstance(s[0], np.bool_)


# --- bulk materialization stays native Python -------------------------------

def test_tolist_stays_python_scalars():
    df = _df()
    assert type(df["a"].to_list()[0]) is float        # not np.float64
    assert type(df["b"].to_list()[0]) is int           # not np.int64
    assert type(df["a"].tolist()[0]) is float


def test_row_to_dict_stays_python():
    row = _df().iloc[0]
    d = row.to_dict()
    assert type(d["a"]) is float and type(d["b"]) is int
