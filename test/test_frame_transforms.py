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


# shift / diff diverge from pandas on an int column (volas keeps int64 + NA where
# pandas upcasts to float64), so they are checked separately in test_int_shift_diff_na.
@pytest.mark.parametrize("op", ["cumsum", "cummax", "cummin", "cumprod", "abs", "rank"])
def test_frame_transform_parity(op):
    v, p = _frames()
    _assert_frame(getattr(v, op)(), getattr(p, op)())


def test_frame_transforms_preserve_int_dtype():
    v, _ = _frames()
    # cumsum/cummax/cummin/cumprod/abs/diff/shift keep the int column int64 (with NA
    # for the diff/shift gap); rank is always float.
    assert v.cumsum()["i"].dtype == "int64"
    assert v.abs()["i"].dtype == "int64"
    assert v.diff()["i"].dtype == "int64"
    assert v.shift()["i"].dtype == "int64"
    assert v.rank()["i"].dtype == "float64"


def test_int_shift_diff_na():
    # volas keeps the int dtype and fills the gap with volas.NA (PDEP-16 aligned),
    # where pandas 3.0 still upcasts an int shift/diff to float64.
    v, _ = _frames()
    sh = v.shift()["i"]
    assert sh.dtype == "int64"
    assert sh[0] is volas.NA and sh[1] == 3
    assert sh.isna().to_numpy().tolist() == [True, False, False, False]
    df = v.diff()["i"]
    assert df.dtype == "int64"
    assert df[0] is volas.NA and df[1] == -2  # 1 - 3


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
