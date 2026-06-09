"""volas.NA — the native missing-value model (PDEP-16 aligned, no object dtype).

int/bool columns keep their dtype and carry a `volas.NA` in vacated cells
(shift / diff gap), where pandas 3.0 upcasts to float/object. `volas.NA` is a
pure user-facing symbol; storage stays dtype-optimal underneath.
"""

import numpy as np
import pytest

import volas


def _int_na():
    # int64 column with a leading NA from shift: [NA, 1, 2]
    return volas.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})["a"].shift(1)


def test_na_singleton_repr_and_bool():
    assert repr(volas.NA) == "<NA>"
    assert volas.NA is volas.NA
    with pytest.raises(TypeError):
        bool(volas.NA)


def test_int_shift_keeps_dtype_with_na():
    s = _int_na()
    assert s.dtype == "int64"
    assert s[0] is volas.NA
    assert s[1] == 1 and isinstance(s[1], np.int64)
    assert s.isna().to_numpy().tolist() == [True, False, False]
    assert s.notna().to_numpy().tolist() == [False, True, True]


def test_na_element_access_per_dtype():
    si = volas.DataFrame({"a": np.array([1, 2, 3], dtype=np.int32)})["a"].shift(1)
    assert si.dtype == "int32" and si[0] is volas.NA and si[1] == 1
    sb = (volas.DataFrame({"a": np.array([1, 0, 1], dtype=np.int64)})["a"] > 0).shift(1)
    assert sb.dtype == "bool" and sb[0] is volas.NA
    assert bool(sb[1]) is True  # the present cell is a real np.bool_


def test_na_to_numpy_and_to_list():
    s = _int_na()  # NA, 1, 2
    arr = s.to_numpy()
    assert arr.dtype == np.float64  # numpy can't hold NA -> float64 + nan
    assert np.isnan(arr[0]) and arr[1] == 1.0
    lst = s.to_list()
    assert lst[0] is volas.NA and lst[1] == 1


def test_na_reductions_skip_missing():
    s = _int_na()  # NA, 1, 2
    assert s.sum() == 3 and isinstance(s.sum(), np.int64)
    assert s.min() == 1 and s.max() == 2
    assert s.dropna().to_list() == [1, 2]


def test_na_fillna_promotes_only_when_needed():
    s = _int_na()  # NA, 1, 2
    assert s.fillna(0).to_list() == [0, 1, 2]
    assert s.fillna(0).dtype == "int64"
    assert s.fillna(2.5).dtype == "float64"  # a non-integral fill promotes to float


def test_na_directional_fill():
    s = _int_na()  # NA, 1, 2
    assert s.bfill().to_list() == [1, 1, 2]  # backfill the leading gap
    assert s.ffill()[0] is volas.NA  # nothing to carry forward into the leading gap
    assert s.ffill().dtype == "int64"


def test_na_diff_keeps_int():
    s = volas.DataFrame({"a": np.array([1, 3, 6], dtype=np.int64)})["a"].diff()
    assert s.dtype == "int64"
    assert s[0] is volas.NA and s[1] == 2 and s[2] == 3
