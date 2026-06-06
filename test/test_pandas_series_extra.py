"""P2 Series methods added for pandas parity (OHLCV-relevant convenience):
shape, fillna(method=), astype, cumsum, round, abs, clip, quantile, idxmax/idxmin.

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


# --- fillna(value=, method=) ------------------------------------------------

def test_fillna_value():
    np.testing.assert_array_equal(_s([3.0, nan, 1.0]).fillna(0.0).to_numpy(), [3, 0, 1])


def test_fillna_ffill_and_pad_alias():
    src = [nan, 3.0, nan, nan, 1.0]
    np.testing.assert_array_equal(_s(src).fillna(method="ffill").to_numpy(), [nan, 3, 3, 3, 1])
    np.testing.assert_array_equal(_s(src).fillna(method="pad").to_numpy(), [nan, 3, 3, 3, 1])


def test_fillna_bfill_and_backfill_alias():
    src = [3.0, nan, nan, 1.0, nan]
    np.testing.assert_array_equal(_s(src).fillna(method="bfill").to_numpy(), [3, 1, 1, 1, nan])
    np.testing.assert_array_equal(_s(src).fillna(method="backfill").to_numpy(), [3, 1, 1, 1, nan])


def test_fillna_both_value_and_method_raises():
    with pytest.raises(ValueError, match="not both"):
        _s([nan]).fillna(0.0, method="ffill")


def test_fillna_neither_raises():
    with pytest.raises(ValueError):
        _s([nan]).fillna()


def test_fillna_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        _s([nan]).fillna(method="interpolate")


def test_fillna_on_non_float_column_is_passthrough():
    mask = _s([1.0, 2.0]) > 1.5  # a bool Series
    np.testing.assert_array_equal(mask.fillna(0.0).to_numpy(), mask.to_numpy())


# --- astype -----------------------------------------------------------------

def test_astype_numeric():
    assert _s([1.0, 2.0]).astype("int64").dtype == "int64"
    assert _s([1.0, 0.0]).astype("bool").dtype == "bool"


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


def test_round_default_and_decimals_and_negative():
    np.testing.assert_array_equal(_s([1.4, 1.6]).round().to_numpy(), [1, 2])
    np.testing.assert_array_equal(_s([1.234, 2.567]).round(1).to_numpy(), [1.2, 2.6])
    np.testing.assert_array_equal(_s([14.0, 25.0]).round(-1).to_numpy(), [10, 30])


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
    # On a DatetimeIndex the label is rendered as volas does everywhere (a string,
    # like ``Row.name``), not a pandas Timestamp.
    d = DataFrame({"v": [1.0, 9.0, 3.0], "t": ["2020-01-01", "2020-01-02", "2020-01-03"]})
    d["t"] = volas.to_datetime(d["t"])
    d = d.set_index("t")
    assert d["v"].idxmax() == "2020-01-02 00:00:00"


def test_idxmax_all_nan_raises():
    with pytest.raises(ValueError, match="all NA"):
        _s([nan, nan]).idxmax()
    with pytest.raises(ValueError, match="all NA"):
        _s([nan, nan]).idxmin()
