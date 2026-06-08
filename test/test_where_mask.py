"""Conditional selection / assignment parity with pandas 3.0:

``Series.where`` / ``Series.mask`` / ``DataFrame.where`` / ``DataFrame.mask``,
boolean-mask reads (``s[mask]``) and boolean-mask assignment (``s[mask] = v``,
``df[mask] = v``). Values are cross-checked against pandas where an oracle is
cheap; the float-coercion of volas's numeric model is asserted explicitly.
"""

import numpy as np
import pandas as pd
import pytest

import volas

nan = float("nan")


def _s(values):
    return volas.DataFrame({"a": list(values)})["a"]


# --- Series.where / mask ----------------------------------------------------

def test_series_where_keeps_true_else_nan():
    x = _s([1.0, -2.0, 3.0, -4.0])
    np.testing.assert_array_equal(x.where(x > 0).to_numpy(), [1, nan, 3, nan])


def test_series_where_scalar_other():
    x = _s([1.0, -2.0, 3.0, -4.0])
    np.testing.assert_array_equal(x.where(x > 0, 0.0).to_numpy(), [1, 0, 3, 0])


def test_series_mask_is_inverse_of_where():
    x = _s([1.0, -2.0, 3.0, -4.0])
    np.testing.assert_array_equal(x.mask(x > 0, 0.0).to_numpy(), [0, -2, 0, -4])


def test_series_where_series_other():
    # other as a Series fills element-wise -> where(x>0, -x) == abs(x)
    x = _s([1.0, -2.0, 3.0, -4.0])
    np.testing.assert_array_equal(x.where(x > 0, _s([-1.0, 2.0, -3.0, 4.0])).to_numpy(),
                                  [1, 2, 3, 4])


def test_series_where_matches_pandas():
    data = [5.0, -1.0, nan, 2.0, -3.0]
    x = _s(data)
    p = pd.Series(data)
    np.testing.assert_array_equal(x.where(x > 0, 0.0).to_numpy(),
                                  p.where(p > 0, 0.0).to_numpy())
    np.testing.assert_array_equal(x.mask(x > 0, 0.0).to_numpy(),
                                  p.mask(p > 0, 0.0).to_numpy())


def _bool_s(values):
    return volas.DataFrame({"m": list(values)})["m"]


def test_series_where_length_mismatch_raises():
    with pytest.raises(ValueError):
        _s([1.0, 2.0, 3.0]).where(_bool_s([True, False]))


# --- Series boolean read ----------------------------------------------------

def test_series_boolean_read_filters_rows_and_labels():
    y = _s([10.0, 20.0, 30.0, 40.0])
    sub = y[y > 20]
    np.testing.assert_array_equal(sub.to_numpy(), [30, 40])
    # a RangeIndex keeps the surviving original labels (-> Int64), like pandas
    np.testing.assert_array_equal(np.asarray(sub.index), [2, 3])


def test_series_boolean_read_empty():
    y = _s([1.0, 2.0])
    sub = y[y > 5]
    assert sub.shape == (0,)


# --- Series boolean / positional assignment ---------------------------------

def test_series_mask_assignment():
    z = _s([1.0, -2.0, 3.0, -4.0])
    z[z < 0] = 0.0
    np.testing.assert_array_equal(z.to_numpy(), [1, 0, 3, 0])


def test_series_positional_assignment():
    z = _s([1.0, 2.0, 3.0])
    z[1] = 9.0
    z[-1] = 7.0
    np.testing.assert_array_equal(z.to_numpy(), [1, 9, 7])


def test_series_mask_assignment_length_mismatch_raises():
    with pytest.raises(ValueError):
        _s([1.0, 2.0, 3.0])[_bool_s([True, False])] = 0.0


# --- assignment dtype rules (pandas 3.0: fit->keep, NaN->upcast, lossy->raise) -

def _si(values):
    return volas.DataFrame({"a": np.array(list(values), dtype=np.int64)})["a"]


def test_int_series_assignment_keeps_int64():
    z = _si([1, 2, 3, 4])
    z[z > 2] = 0  # integral fill stays int64
    assert z.dtype == "int64"
    np.testing.assert_array_equal(z.to_numpy(), [1, 2, 0, 0])
    z2 = _si([1, 2, 3, 4])
    z2[z2 > 2] = 0.0  # an integral float is still lossless -> int64
    assert z2.dtype == "int64"
    z3 = _si([1, 2, 3, 4])
    z3[1] = 9  # positional, same rule
    assert z3.dtype == "int64"
    np.testing.assert_array_equal(z3.to_numpy(), [1, 9, 3, 4])


def test_int_series_assignment_nan_upcasts_to_float():
    z = _si([1, 2, 3, 4])
    z[z > 2] = float("nan")
    assert z.dtype == "float64"
    np.testing.assert_array_equal(z.to_numpy(), [1, 2, nan, nan])


def test_int_series_lossy_assignment_raises():
    z = _si([1, 2, 3, 4])
    with pytest.raises(TypeError):
        z[z > 2] = 2.5  # non-integral cannot fit int64 -> raise (no silent upcast)


def test_float_series_assignment_stays_float():
    z = _s([1.0, 2.0, 3.0, 4.0])
    z[z > 2] = 0  # int fill into float column stays float
    assert z.dtype == "float64"
    np.testing.assert_array_equal(z.to_numpy(), [1, 2, 0, 0])


def test_frame_mixed_dtype_row_assignment_keeps_per_column():
    d = volas.DataFrame({"i": np.array([1, 2, 3], dtype=np.int64), "f": [1.0, 2.0, 3.0]})
    d[d["i"] > 1] = 0  # 0 fits both columns -> each keeps its dtype
    assert d["i"].dtype == "int64" and d["f"].dtype == "float64"
    np.testing.assert_array_equal(d["i"].to_numpy(), [1, 0, 0])
    np.testing.assert_array_equal(d["f"].to_numpy(), [1, 0, 0])


def test_frame_lossy_row_assignment_raises_and_leaves_frame_unchanged():
    d = volas.DataFrame({"i": np.array([1, 2, 3], dtype=np.int64), "f": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError):
        d[d["i"] > 1] = 0.5  # lossy for the int column
    # atomic: nothing was written
    assert d["i"].dtype == "int64"
    np.testing.assert_array_equal(d["i"].to_numpy(), [1, 2, 3])


# --- DataFrame.where / mask -------------------------------------------------

def test_frame_where_and_mask_via_isna():
    df = volas.DataFrame({"a": [1.0, nan, 3.0], "b": [nan, 5.0, 6.0]})
    np.testing.assert_array_equal(df.where(df.notna(), 0.0).to_numpy(),
                                  [[1, 0], [0, 5], [3, 6]])
    np.testing.assert_array_equal(df.mask(df.isna(), -1.0).to_numpy(),
                                  [[1, -1], [-1, 5], [3, 6]])


def test_frame_where_default_other_is_nan():
    df = volas.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    cond = df.isna()  # all-False -> where keeps nothing
    np.testing.assert_array_equal(df.where(cond).to_numpy(), [[nan, nan], [nan, nan]])


def test_frame_where_shape_mismatch_raises():
    df = volas.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    one = volas.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        df.where(one.isna())


# --- DataFrame boolean assignment -------------------------------------------

def test_frame_row_mask_assignment_sets_whole_rows():
    d = volas.DataFrame({"a": [1.0, -2.0, 3.0], "b": [-1.0, 2.0, -3.0]})
    d[d["a"] > 0] = 99.0  # rows 0 and 2
    np.testing.assert_array_equal(d.to_numpy(), [[99, 99], [-2, 2], [99, 99]])


def test_frame_cell_mask_assignment():
    d = volas.DataFrame({"a": [1.0, nan, 3.0], "b": [nan, 5.0, 6.0]})
    d[d.isna()] = 0.0
    np.testing.assert_array_equal(d.to_numpy(), [[1, 0], [0, 5], [3, 6]])


def test_frame_column_assignment_still_works():
    # the boolean-mask dispatch must not break ordinary column set
    d = volas.DataFrame({"a": [1.0, 2.0]})
    d["b"] = 5.0
    d["c"] = d["a"]
    assert d.columns == ["a", "b", "c"]
    np.testing.assert_array_equal(d["b"].to_numpy(), [5, 5])
    np.testing.assert_array_equal(d["c"].to_numpy(), [1, 2])
