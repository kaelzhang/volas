"""Contract D5 + Q3 — DataFrame.to_numpy() must be an honest boundary, not route
datetime through the f64 channel by default. Pandas-aligned:
  * numeric/bool-only frame -> fast float64 matrix
  * datetime-only frame     -> 2-D datetime64[ns]
  * any other mix (datetime+numeric / datetime+str / str+numeric / str-only)
    -> object array, so no column is funnelled to f64
An EXPLICIT dtype='float64' over datetime is an allowed lossy export (Q3 'a'),
documented as such; over a str column it still raises."""

import numpy as np
import pytest
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


# --- P1-01: datetime-only default keeps ns + NaT (built from the raw i64 buffer,
# not via per-cell Python objects NumPy re-coerces to seconds / fails on NaT) ----

NS123 = 1609459200000000123  # 2021-01-01T00:00:00.000000123
NS456 = 1609545600000000456  # 2021-01-02T00:00:00.000000456


def _dt_ns():
    return DataFrame({"t": np.array([NS123, NS456], dtype="datetime64[ns]")})


def test_datetime_only_default_preserves_nanoseconds():
    arr = _dt_ns().to_numpy()
    assert arr.dtype == np.dtype("datetime64[ns]")
    assert arr.astype("int64").reshape(-1).tolist() == [NS123, NS456]  # ns kept, not truncated


def test_datetime_only_with_nat_default_is_datetime64_not_error():
    arr = DataFrame({"t": np.array([NS123, "NaT"], dtype="datetime64[ns]")}).to_numpy()
    assert arr.dtype == np.dtype("datetime64[ns]")
    assert np.isnat(arr[1, 0]) and arr[0, 0].astype("int64") == NS123


# --- P2-01: explicit dtype is honored per cell (export boundary), like pandas ---

def test_dtype_object_is_lossless_typed_cells():
    # datetime + numeric: each cell its own typed value, never funnelled to f64
    df = DataFrame({"t": np.array([NS123, NS456], dtype="datetime64[ns]"), "x": [1.0, 2.0]})
    arr = df.to_numpy(dtype="object")
    assert arr.dtype == object
    assert isinstance(arr[0, 0], volas.Timestamp) and arr[0, 0].value == NS123
    assert arr[0, 1] == 1.0
    # datetime + str: strings survive (the old f64 funnel turned them into NaN)
    arrs = DataFrame({"t": np.array([NS123], dtype="datetime64[ns]"), "s": ["x"]}).to_numpy(dtype="object")
    assert arrs[0, 0].value == NS123 and arrs[0, 1] == "x"


def test_dtype_int64_over_datetime_is_exact_epoch_ns():
    arr = _dt_ns().to_numpy(dtype="int64")
    assert arr.dtype == np.int64
    assert arr.reshape(-1).tolist() == [NS123, NS456]  # EXACT ns, not via f64
    # NaT maps to the i64::MIN sentinel (pandas-aligned), not 0
    nat = DataFrame({"t": np.array([NS123, "NaT"], dtype="datetime64[ns]")}).to_numpy(dtype="int64")
    assert nat[0, 0] == NS123 and nat[1, 0] == np.iinfo(np.int64).min


def test_dtype_float64_over_datetime_stays_lossy_optin():
    # the explicit float export remains the caller's sanctioned lossy opt-in
    arr = _dt_ns().to_numpy(dtype="float64")
    assert arr.dtype == np.float64 and arr.shape == (2, 1)


def test_dtype_int64_over_str_raises():
    import pytest
    with pytest.raises(Exception):
        DataFrame({"s": ["a", "b"]}).to_numpy(dtype="int64")


def test_int64_column_exact_past_2_53():
    # the integer channel never round-trips through f64, so a large i64 survives
    big = np.array([2 ** 60, 2 ** 60 + 1], dtype=np.int64)
    arr = DataFrame({"a": big}).to_numpy(dtype="int64")
    assert arr.reshape(-1).tolist() == [2 ** 60, 2 ** 60 + 1]


# --- the export target is NumPy's resolved dtype, never its spelling ------------

@pytest.mark.parametrize(
    "spelling", ["f8", "d", "<f8", "double", "f4", "e", "complex128", "longdouble"]
)
def test_every_float_spelling_keeps_fractions_and_nan(spelling):
    arr = DataFrame({"a": [1.5, 2.25, None]}).to_numpy(dtype=spelling)
    expected = np.array([[1.5], [2.25], [np.nan]]).astype(spelling)
    assert arr.dtype == np.dtype(spelling)
    np.testing.assert_array_equal(arr, expected)


@pytest.mark.parametrize("spelling", ["object_", "|O"])
def test_every_object_spelling_keeps_typed_cells(spelling):
    arr = DataFrame({"a": [1.5, 2.25]}).to_numpy(dtype=spelling)
    assert arr.dtype == object
    assert arr.ravel().tolist() == [1.5, 2.25]


def test_bool_target_tests_the_float_value_not_a_truncated_int():
    arr = DataFrame({"a": [0.5, -0.5, 0.0]}).to_numpy(dtype="bool")
    assert arr.ravel().tolist() == [True, True, False]


def test_bool_target_over_missing_raises_like_an_integer_target():
    with pytest.raises(ValueError, match="missing"):
        DataFrame({"a": [1, None]}).to_numpy(dtype="bool")
    with pytest.raises(ValueError, match="missing"):
        DataFrame({"a": [1, None]})["a"].to_numpy(dtype="bool")
    filled = DataFrame({"a": [1, None]}).to_numpy(dtype="bool", na_value=0)
    assert filled.ravel().tolist() == [True, False]


def test_text_target_formats_the_float_values():
    arr = DataFrame({"a": [1.5, 2.25]}).to_numpy(dtype="U")
    assert arr.ravel().tolist() == ["1.5", "2.25"]
