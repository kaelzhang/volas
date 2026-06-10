"""P2 Series methods added for pandas parity (OHLCV-relevant convenience):
shape, fillna, ffill, bfill, astype, cumsum, round, abs, clip, quantile, idxmax/idxmin.

Numeric results are checked against pandas where a parity oracle is cheap.
"""

import numpy as np
import pandas as pd
import pytest

import volas
from volas import DataFrame

nan = float("nan")


def _s(values):
    return DataFrame({"a": list(values)})["a"]


# --- shape ------------------------------------------------------------------

def test_shape_is_one_tuple():
    assert _s([1.0, 2.0, 3.0]).shape == (3,)
    assert _s([]).shape == (0,)


# --- fillna / ffill / bfill -------------------------------------------------

def test_fillna_value():
    np.testing.assert_array_equal(_s([3.0, nan, 1.0]).fillna(0.0).to_numpy(), [3, 0, 1])


def test_ffill():
    src = [nan, 3.0, nan, nan, 1.0]
    np.testing.assert_array_equal(_s(src).ffill().to_numpy(), [nan, 3, 3, 3, 1])


def test_bfill():
    src = [3.0, nan, nan, 1.0, nan]
    np.testing.assert_array_equal(_s(src).bfill().to_numpy(), [3, 1, 1, 1, nan])


def test_fillna_requires_a_value():
    # pandas 3.0 removed fillna(method=); fillna now needs a value
    with pytest.raises(TypeError):
        _s([nan]).fillna()


def test_fillna_on_non_float_column_is_passthrough():
    mask = _s([1.0, 2.0]) > 1.5  # a bool Series
    np.testing.assert_array_equal(mask.fillna(0.0).to_numpy(), mask.to_numpy())


# --- astype -----------------------------------------------------------------

def test_astype_numeric():
    assert _s([1.0, 2.0]).astype("int64").dtype == "int64"
    assert _s([1.0, 0.0]).astype("bool").dtype == "bool"


def test_astype_int_with_non_finite_raises():
    # pandas raises IntCastingNaNError; volas raises rather than silently NaN->0 / inf->i64::MAX
    with pytest.raises(ValueError):
        _s([1.0, nan, 3.0]).astype("int64")
    with pytest.raises(ValueError):
        _s([1.0, float("inf"), 3.0]).astype("int64")


def test_astype_string_to_datetime():
    out = DataFrame({"a": ["2020-01-01", "2020-01-02"]})["a"].astype("datetime64[ns]")
    assert out.dtype == "datetime64[ns]"


def test_astype_datetime_to_datetime_roundtrips():
    dt = volas.to_datetime(DataFrame({"a": ["2020-01-01"]})["a"])
    assert dt.astype("datetime64[ns]").dtype == "datetime64[ns]"


def test_astype_epoch_int_to_datetime():
    out = DataFrame({"a": [1_577_836_800]})["a"].astype("datetime64[s]")
    assert out.dtype == "datetime64[ns]"


# --- cumsum / round / abs / clip --------------------------------------------

def test_cumsum_skips_nan_in_place():
    np.testing.assert_array_equal(_s([1.0, nan, 2.0, 3.0]).cumsum().to_numpy(), [1, nan, 3, 6])


def test_cummax_cummin_cumprod_skip_nan():
    np.testing.assert_array_equal(_s([1.0, nan, 2.0, 4.0]).cummax().to_numpy(), [1, nan, 2, 4])
    np.testing.assert_array_equal(_s([3.0, nan, 1.0, 2.0]).cummin().to_numpy(), [3, nan, 1, 1])
    np.testing.assert_array_equal(_s([1.0, nan, 2.0, 4.0]).cumprod().to_numpy(), [1, nan, 2, 8])


def test_prod_skips_nan_and_empty_is_one():
    assert _s([1.0, nan, 2.0, 3.0]).prod() == 6.0
    assert _s([nan, nan]).prod() == 1.0   # all-NaN -> 1.0 (pandas)
    assert _s([]).prod() == 1.0           # empty -> 1.0


def test_sem_skew_kurt_match_pandas():
    import math
    s = _s([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])  # values cross-checked vs pandas 3.0
    assert abs(s.sem() - 0.7559289460184544) < 1e-9
    assert abs(s.skew() - 0.8184875533567997) < 1e-9
    assert abs(s.kurt() - 0.9406250000000004) < 1e-9
    assert math.isnan(_s([1.0]).sem())          # n < 2
    assert math.isnan(_s([1.0, 2.0]).skew())    # n < 3
    assert math.isnan(_s([1.0, 2.0, 3.0]).kurt())  # n < 4


def test_rank_methods_and_pct():
    src = [3.0, 1.0, 1.0, 2.0, nan]
    np.testing.assert_array_equal(_s(src).rank().to_numpy(), [4, 1.5, 1.5, 3, nan])
    np.testing.assert_array_equal(_s(src).rank(method="min").to_numpy(), [4, 1, 1, 3, nan])
    np.testing.assert_array_equal(_s(src).rank(method="dense").to_numpy(), [3, 1, 1, 2, nan])
    np.testing.assert_array_equal(_s(src).rank(ascending=False).to_numpy(), [1, 3.5, 3.5, 2, nan])
    np.testing.assert_array_equal(_s(src).rank(pct=True).to_numpy(), [1.0, 0.375, 0.375, 0.75, nan])


def test_rank_unknown_method_raises():
    with pytest.raises(ValueError):
        _s([1.0, 2.0]).rank(method="bogus")


def test_describe_index_and_values():
    d = _s([1.0, 2.0, nan, 4.0, 5.0]).describe()
    assert list(np.asarray(d.index)) == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    np.testing.assert_allclose(
        np.asarray(d.to_numpy(), float),
        [4.0, 3.0, 1.8257418583505538, 1.0, 1.75, 3.0, 4.25, 5.0],
    )


def test_corr_cov_positional_pairwise():
    a = _s([1.0, 2.0, 3.0, 4.0])
    b = _s([1.0, 2.0, 3.0, 5.0])
    assert abs(a.corr(b) - 0.9827076298239906) < 1e-9
    assert abs(a.cov(b) - 2.1666666666666665) < 1e-9
    # NaN pairs dropped: drops index 1 -> corr([1,3],[1,3]) == 1
    assert abs(_s([1.0, nan, 3.0]).corr(_s([1.0, 5.0, 3.0])) - 1.0) < 1e-9


def test_round_default_and_decimals_and_negative():
    np.testing.assert_array_equal(_s([1.4, 1.6]).round().to_numpy(), [1, 2])
    np.testing.assert_array_equal(_s([1.234, 2.567]).round(1).to_numpy(), [1.2, 2.6])
    np.testing.assert_array_equal(_s([14.0, 25.0]).round(-1).to_numpy(), [10, 20])  # 25 -> 20 (banker's)


def test_round_is_banker_half_to_even():
    # round-half-to-even (banker's), matching pandas / NumPy — ties go to the even neighbour
    np.testing.assert_array_equal(_s([0.5, 1.5, 2.5, 3.5]).round(0).to_numpy(), [0, 2, 2, 4])


def test_abs():
    np.testing.assert_array_equal(_s([-1.0, 2.0, -3.0]).abs().to_numpy(), [1, 2, 3])


def test_clip_bounds_and_nan_preserved():
    np.testing.assert_array_equal(_s([-1.0, 1.0, 3.0]).clip(0.0, 2.0).to_numpy(), [0, 1, 2])
    np.testing.assert_array_equal(_s([-1.0, 5.0]).clip(lower=0.0).to_numpy(), [0, 5])
    np.testing.assert_array_equal(_s([-1.0, 5.0]).clip(upper=2.0).to_numpy(), [-1, 2])
    np.testing.assert_array_equal(_s([nan, 5.0]).clip(0.0, 1.0).to_numpy(), [nan, 1])
    np.testing.assert_array_equal(_s([1.0, 5.0]).clip().to_numpy(), [1, 5])  # no-op


# --- quantile ---------------------------------------------------------------

@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0, 0.3])
def test_quantile_matches_pandas_linear(q):
    data = [5.0, 1.0, 4.0, 2.0, 3.0]
    assert _s(data).quantile(q) == pytest.approx(pd.Series(data).quantile(q))


def test_quantile_skips_nan():
    assert _s([1.0, nan, 3.0]).quantile(0.5) == 2.0


def test_quantile_empty_is_nan():
    assert np.isnan(_s([nan, nan]).quantile(0.5))


def test_quantile_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _s([1.0]).quantile(1.5)


# --- idxmax / idxmin --------------------------------------------------------

def test_idxmax_idxmin_rangeindex():
    assert _s([3.0, 1.0, 5.0, 2.0]).idxmax() == 2
    assert _s([3.0, 1.0, 5.0, 2.0]).idxmin() == 1


def test_idxmax_skips_nan_and_takes_first_max():
    assert _s([nan, 5.0, 5.0, 1.0]).idxmax() == 1  # first occurrence


def test_idxmax_returns_datetime_label():
    # On a DatetimeIndex the label is a volas.Timestamp (R6), like ``Row.name``. It
    # compares equal to a datetime string (a display / lookup convenience), but the
    # type is Timestamp — assert that explicitly so a regression to a bare string or
    # np.datetime64 is caught (Timestamp.__eq__ on a string would otherwise hide it).
    d = DataFrame({"v": [1.0, 9.0, 3.0], "t": ["2020-01-01", "2020-01-02", "2020-01-03"]})
    d["t"] = volas.to_datetime(d["t"])
    d = d.set_index("t")
    label = d["v"].idxmax()
    assert isinstance(label, volas.Timestamp)
    assert label == "2020-01-02 00:00:00"   # string equality still works (lookup)


def test_idxmax_all_nan_raises():
    with pytest.raises(ValueError, match="all NA"):
        _s([nan, nan]).idxmax()
    with pytest.raises(ValueError, match="all NA"):
        _s([nan, nan]).idxmin()
