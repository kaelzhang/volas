"""Transform-surface dtype parity with pandas 3.0 (typed-kernel rebuild).

Every dtype-preserving transform (cum*, abs, round, clip, where/mask, +-*) is
checked against pandas for both dtype AND values, on int and float inputs,
including the promotion cases (a non-integral bound/fill or a float operand
promotes int -> float; true division is always float).
"""

import numpy as np
import pandas as pd
import pytest

import volas

nan = float("nan")


def _vi(values):
    return volas.DataFrame({"a": np.array(values, dtype=np.int64)})["a"]


def _vf(values):
    return volas.DataFrame({"a": [float(v) for v in values]})["a"]


def _assert_parity(vser, pser):
    """volas series matches a pandas series in both dtype and values."""
    assert vser.dtype == str(pser.dtype), f"dtype: volas {vser.dtype} != pandas {pser.dtype}"
    va = np.asarray(vser.to_numpy()).astype(float)
    pa = pser.to_numpy().astype(float)
    np.testing.assert_allclose(np.nan_to_num(va, nan=-9e99), np.nan_to_num(pa, nan=-9e99))


# --- cumulatives: preserve int, native ---------------------------------------

@pytest.mark.parametrize("op", ["cumsum", "cummax", "cummin", "cumprod"])
def test_cumulative_dtype_parity(op):
    data = [3, 1, 4, 1, 5]
    _assert_parity(getattr(_vi(data), op)(), getattr(pd.Series(data, dtype="int64"), op)())
    fdata = [3.0, nan, 4.0, 1.0]
    _assert_parity(getattr(_vf(fdata), op)(), getattr(pd.Series(fdata), op)())


def test_cumsum_int_is_exact_beyond_2_53():
    # native i64 (no f64 round-trip) keeps full precision past 2**53
    big = 2**60 + 1
    out = _vi([big, big]).cumsum()
    assert out.dtype == "int64"
    assert int(np.asarray(out.to_numpy())[1]) == 2 * big  # exact, not 2**61


def test_bool_cumsum_is_int64():
    # pandas treats bool as int in cumsum (counts True)
    b = _vf([1.0, 2.0]) > 1.5  # a bool Series [False, True]
    out = b.cumsum()
    assert out.dtype == "int64"
    np.testing.assert_array_equal(out.to_numpy(), [0, 1])


# --- abs / round -------------------------------------------------------------

def test_abs_dtype_parity():
    _assert_parity(_vi([-3, 4, -5]).abs(), pd.Series([-3, 4, -5], dtype="int64").abs())
    _assert_parity(_vf([-3.0, nan, 5.0]).abs(), pd.Series([-3.0, nan, 5.0]).abs())


def test_round_dtype_parity():
    _assert_parity(_vi([15, 25, 35, 45]).round(-1), pd.Series([15, 25, 35, 45], dtype="int64").round(-1))
    _assert_parity(_vi([15, 25]).round(0), pd.Series([15, 25], dtype="int64").round(0))  # identity
    _assert_parity(_vf([1.27, 2.83]).round(1), pd.Series([1.27, 2.83]).round(1))


# --- clip: stay int with integral bounds, promote on a non-integral bound -----

def test_clip_dtype_parity():
    _assert_parity(_vi([1, 5, 9]).clip(2, 8), pd.Series([1, 5, 9], dtype="int64").clip(2, 8))
    _assert_parity(_vi([1, 5, 9]).clip(2.5, None), pd.Series([1, 5, 9], dtype="int64").clip(2.5, None))
    _assert_parity(_vf([-1.0, 1.0, 3.0]).clip(0.0, 2.0), pd.Series([-1.0, 1.0, 3.0]).clip(0.0, 2.0))


# --- where / mask ------------------------------------------------------------

def test_where_dtype_parity():
    vi, pi = _vi([1, 2, 3, 4]), pd.Series([1, 2, 3, 4], dtype="int64")
    _assert_parity(vi.where(vi > 2, 0), pi.where(pi > 2, 0))          # int fill -> int64
    _assert_parity(vi.where(vi > 2, 2.5), pi.where(pi > 2, 2.5))      # non-integral -> float64
    _assert_parity(vi.where(vi > 2), pi.where(pi > 2))                # default NaN -> float64
    _assert_parity(vi.mask(vi > 2, 0), pi.mask(pi > 2, 0))


# --- arithmetic --------------------------------------------------------------

def test_arithmetic_dtype_parity():
    vi, pi = _vi([5, 7, 9]), pd.Series([5, 7, 9], dtype="int64")
    _assert_parity(vi + vi, pi + pi)        # int64
    _assert_parity(vi - vi, pi - pi)        # int64
    _assert_parity(vi * vi, pi * pi)        # int64
    _assert_parity(vi + 2, pi + 2)          # int scalar -> int64
    _assert_parity(vi * 2.0, pi * 2.0)      # float scalar -> float64
    _assert_parity(vi / vi, pi / pi)        # true division -> float64
    _assert_parity(2 - vi, 2 - pi)          # reflected, int64
    _assert_parity(10.0 / vi, 10.0 / pi)    # reflected division -> float64


def test_int_arithmetic_wraps_like_pandas():
    big = np.iinfo(np.int64).max
    out = _vi([big]) + _vi([1])
    assert out.dtype == "int64"
    assert int(np.asarray(out.to_numpy())[0]) == np.iinfo(np.int64).min  # wraps


# --- bool: per-operation, matching pandas 3.0 -------------------------------

def _vbool(values):
    # a real bool Series (from a comparison)
    return _vf([1.0 if v else 0.0 for v in values]) > 0.5


def test_bool_dtype_matches_pandas():
    b = _vbool([True, False, True])
    # cumsum/cumprod -> int64; cummax/cummin/abs/round/clip -> bool
    assert b.cumsum().dtype == "int64"
    assert b.cumprod().dtype == "int64"
    assert b.cummax().dtype == "bool"
    assert b.cummin().dtype == "bool"
    assert b.abs().dtype == "bool"
    assert b.round().dtype == "bool"
    assert b.clip(False, True).dtype == "bool"
    np.testing.assert_array_equal(b.cummax().to_numpy(), [True, True, True])   # running OR
    np.testing.assert_array_equal(b.cummin().to_numpy(), [True, False, False])  # running AND


def test_bool_arithmetic_matches_pandas():
    b = _vbool([True, False, True])
    c = _vbool([True, True, False])
    assert (b + c).dtype == "bool"   # OR
    assert (b * c).dtype == "bool"   # AND
    np.testing.assert_array_equal((b + c).to_numpy(), [True, True, True])
    np.testing.assert_array_equal((b * c).to_numpy(), [True, False, False])
    assert (b + 1).dtype == "int64"     # bool ∘ int -> int
    assert (b + 1.0).dtype == "float64"  # bool ∘ float -> float
    with pytest.raises(Exception):
        _ = b - c   # bool subtraction unsupported (pandas raises)
    with pytest.raises(Exception):
        _ = b / c   # bool division unsupported (pandas raises)
