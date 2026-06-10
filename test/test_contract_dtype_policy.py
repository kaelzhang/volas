"""Contract C3/D5 — dtype policy at the API boundary.

* volas has no `object` dtype, so `"object"`/`"O"` must be rejected (not silently
  aliased to str) — both as a constructor `dtype=` and in `astype`.
* A datetime column cannot be cast to float (epoch-ns > 2^53 quantizes to ~256ns);
  `astype('float'/'float32')` raises, pointing at the exact `int64` channel. The
  explicit `to_numpy(dtype='float64')` export remains the opt-in lossy path.
"""

import numpy as np
import pytest
from volas import DataFrame


def test_constructor_dtype_object_rejected():
    for dt in ("object", "O"):
        with pytest.raises(Exception):
            DataFrame({"x": [1, 2]}, dtype=dt)


def test_astype_object_rejected():
    for dt in ("object", "O"):
        with pytest.raises(Exception):
            DataFrame({"x": [1.5]}).astype({"x": dt})


def test_astype_str_still_works():
    out = DataFrame({"x": [1, 2]}).astype({"x": "str"})["x"]
    assert out.dtype == "str"
    assert out.to_list() == ["1", "2"]


def _dt():
    return DataFrame({"t": ["2021-01-01", "2021-01-02"]}).astype({"t": "datetime64[ns]"})


@pytest.mark.parametrize("dt", ["float", "float64", "float32"])
def test_astype_datetime_to_float_raises(dt):
    with pytest.raises(Exception):
        _dt().astype({"t": dt})


def test_astype_datetime_to_int64_still_exact():
    # int64 is the lossless epoch-ns channel and must keep working
    out = _dt().astype({"t": "int64"})["t"]
    assert out.dtype == "int64"
    assert out.to_list()[0] == np.datetime64("2021-01-01").astype("datetime64[ns]").astype("int64")
