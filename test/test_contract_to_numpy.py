"""Contract D5 + Q3 — DataFrame.to_numpy() must be an honest boundary, not route
datetime through the f64 channel by default. Pandas-aligned:
  * numeric/bool-only frame -> fast float64 matrix
  * datetime-only frame     -> 2-D datetime64[ns]
  * any other mix (datetime+numeric / datetime+str / str+numeric / str-only)
    -> object array, so no column is funnelled to f64
An EXPLICIT dtype='float64' over datetime is an allowed lossy export (Q3 'a'),
documented as such; over a str column it still raises."""

import numpy as np
import volas
from volas import DataFrame


def _dt_frame():
    df = DataFrame({"t": ["2021-01-01", "2021-01-02"], "u": ["2021-06-01", "2021-06-02"]})
    return df.astype({"t": "datetime64[ns]", "u": "datetime64[ns]"})


# --- default (no dtype) ------------------------------------------------------

def test_datetime_only_frame_exports_datetime64():
    arr = _dt_frame().to_numpy()
    assert arr.dtype == np.dtype("datetime64[ns]")
    assert arr[0, 0] == np.datetime64("2021-01-01")
    assert arr[1, 1] == np.datetime64("2021-06-02")


def test_datetime_plus_numeric_frame_exports_object():
    df = DataFrame({"t": ["2021-01-01", "2021-01-02"], "x": [1.0, 2.0]})
    df = df.astype({"t": "datetime64[ns]"})
    arr = df.to_numpy()
    assert arr.dtype == object
    assert arr[0, 0] == np.datetime64("2021-01-01")   # datetime kept, not f64 epoch
    assert arr[0, 1] == 1.0


def test_all_numeric_frame_still_float64():
    arr = DataFrame({"a": [1.0, 2.0], "b": [3, 4]}).to_numpy()
    assert arr.dtype == np.float64
    assert arr.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_datetime_plus_str_frame_object_unchanged():
    df = DataFrame({"t": ["2021-01-01"], "s": ["x"]})
    df = df.astype({"t": "datetime64[ns]"})
    arr = df.to_numpy()
    assert arr.dtype == object
    assert arr[0, 0] == np.datetime64("2021-01-01") and arr[0, 1] == "x"


# --- explicit dtype (Q3) -----------------------------------------------------

def test_explicit_float64_over_datetime_allowed_lossy():
    # explicit cast is the user opting into the lossy epoch-ns export
    arr = _dt_frame().to_numpy(dtype="float64")
    assert arr.dtype == np.float64
    assert arr.shape == (2, 2)


def test_explicit_float64_over_str_raises():
    import pytest
    with pytest.raises(Exception):
        DataFrame({"s": ["a", "b"]}).to_numpy(dtype="float64")
