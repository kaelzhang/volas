"""Boundary scalar returns are numpy-typed, matching pandas 3.0.

Direct scalar access (`s[i]`, `iloc`, `loc`, `iat`, `at`, a Row's `[col]`) returns
a numpy scalar of the column dtype (np.float64 / np.int64 / np.bool_). Bulk
materialization (`to_list` / iteration / `to_dict`) stays native Python, like
pandas tolist().
"""

import numpy as np
import pandas as pd

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

def test_to_list_stays_python_scalars():
    df = _df()
    assert type(df["a"].to_list()[0]) is float        # not np.float64
    assert type(df["b"].to_list()[0]) is int           # not np.int64
    assert type(df["a"].to_list()[0]) is float


def test_row_to_dict_stays_python():
    row = _df().iloc[0]
    d = row.to_dict()
    assert type(d["a"]) is float and type(d["b"]) is int


# --- reductions -> numpy scalars, dtype-aware (pandas 3.0) -------------------

def _si(values):
    return volas.DataFrame({"a": np.array(values, dtype=np.int64)})["a"]


def _sf(values):
    return volas.DataFrame({"a": [float(v) for v in values]})["a"]


def test_int_reduction_dtypes():
    s = _si([3, 1, 2, 5])
    # sum/prod/min/max keep int -> np.int64
    for op in ("sum", "prod", "min", "max"):
        assert isinstance(getattr(s, op)(), np.int64), op
    assert s.sum() == 11 and s.min() == 1 and s.max() == 5
    # mean/std/var/median/sem/skew/kurt/quantile -> np.float64
    for op in ("mean", "std", "var", "median", "sem", "skew", "kurt"):
        assert isinstance(getattr(s, op)(), np.float64), op
    assert isinstance(s.quantile(0.5), np.float64)


def test_float_reduction_dtypes():
    s = _sf([3, 1, 2, 5])
    for op in ("sum", "prod", "min", "max", "mean", "std"):
        assert isinstance(getattr(s, op)(), np.float64), op


def test_bool_reduction_dtypes():
    b = _sf([1, 0, 1]) > 0.5
    assert isinstance(b.sum(), np.int64) and b.sum() == 2  # counts trues
    assert isinstance(b.any(), np.bool_) and isinstance(b.all(), np.bool_)


def test_reduction_values_match_pandas():
    data = [3, 1, 4, 1, 5, 9]
    s, p = _si(data), pd.Series(data, dtype="int64")
    assert s.sum() == p.sum() and s.min() == p.min() and s.max() == p.max()
    assert abs(s.mean() - p.mean()) < 1e-12 and abs(s.std() - p.std()) < 1e-12
