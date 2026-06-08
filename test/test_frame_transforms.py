"""档3: DataFrame column-wise numeric methods, parity with pandas 3.0.

Transforms return a frame (dtype-preserving per column, like the Series ops);
reductions (sem/skew/kurt) return a Series indexed by column name over the numeric
columns. Checked against pandas for dtype + values.
"""

import numpy as np
import pandas as pd
import pytest

import volas


def _frames():
    data = {"f": [1.0, 2.0, 3.0, 4.0], "i": np.array([3, 1, 4, 1], dtype=np.int64)}
    return volas.DataFrame(dict(data)), pd.DataFrame(dict(data))


def _assert_frame(vfr, pfr):
    assert list(vfr.columns) == list(pfr.columns)
    for c in pfr.columns:
        assert vfr[c].dtype == str(pfr[c].dtype), f"{c}: {vfr[c].dtype} != {pfr[c].dtype}"
        va = np.asarray(vfr[c].to_numpy()).astype(float)
        pa = pfr[c].to_numpy().astype(float)
        np.testing.assert_allclose(np.nan_to_num(va, nan=-9e9), np.nan_to_num(pa, nan=-9e9))


@pytest.mark.parametrize("op", ["cumsum", "cummax", "cummin", "cumprod", "abs", "diff", "shift", "rank"])
def test_frame_transform_parity(op):
    v, p = _frames()
    _assert_frame(getattr(v, op)(), getattr(p, op)())


def test_frame_transforms_preserve_int_dtype():
    v, _ = _frames()
    # cumsum/cummax/cummin/cumprod/abs keep the int column int64; diff/shift/rank float
    assert v.cumsum()["i"].dtype == "int64"
    assert v.abs()["i"].dtype == "int64"
    assert v.diff()["i"].dtype == "float64"
    assert v.rank()["i"].dtype == "float64"


def test_frame_clip_parity():
    v, p = _frames()
    _assert_frame(v.clip(1.5, 3.5), p.clip(1.5, 3.5))
    # integer column with integral bounds stays int
    assert v.clip(2, 3)["i"].dtype == "int64"


@pytest.mark.parametrize("op", ["sem", "skew", "kurt"])
def test_frame_reduction_parity(op):
    v, p = _frames()
    vs, ps = getattr(v, op)(), getattr(p, op)()
    assert list(np.asarray(vs.index)) == list(ps.index)  # indexed by column name
    np.testing.assert_allclose(np.asarray(vs.to_numpy()).astype(float), ps.to_numpy().astype(float))


def test_frame_reduction_skips_non_numeric():
    v = volas.DataFrame({"a": [1.0, 2.0, 3.0], "s": ["x", "y", "z"]})
    out = v.sem()
    assert list(np.asarray(out.index)) == ["a"]  # the string column is skipped
