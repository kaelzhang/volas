"""epoch -> datetime conversion maps a missing input to ``NaT``, never to
``1970-01-01``.

Regression for silent data corruption: ``to_datetime`` / ``DataFrame.astype`` /
``Series.astype`` used to truncate a float ``NaN`` (or ignore an int ``volas.NA``
bit) to epoch ``0``, turning a missing timestamp into ``1970-01-01``. A missing
input must become ``NaT`` for every entry path, dtype, and unit — matching
``pandas.to_datetime``. A genuine conversion error (bad unit / overflow) is a
plain ``ValueError``, not a directive-specific exception.
"""

import numpy as np
import pytest

import volas
from volas import DataFrame, to_datetime

nan = float("nan")
EPOCH_S = 1609770600  # 2021-01-04 14:30:00 UTC


def _isnat(series):
    return np.isnat(series.to_numpy()).tolist()


def test_to_datetime_float_nan_maps_to_nat():
    s = to_datetime([nan, float(EPOCH_S)], unit="s")
    assert s.dtype == "datetime64[ns]"
    assert s.isna().to_list() == [True, False]
    assert _isnat(s) == [True, False]
    assert s.to_numpy()[1] == np.datetime64("2021-01-04T14:30:00")


def test_to_datetime_int_na_maps_to_nat():
    # an int column with volas.NA (from a None in construction) keeps int64
    si = DataFrame({"t": [EPOCH_S, None]})["t"]
    assert si.dtype == "int64"
    out = to_datetime(si, unit="s")
    assert out.isna().to_list() == [False, True]  # the NA bit -> NaT, not epoch 0
    assert _isnat(out) == [False, True]


def test_astype_datetime_float_nan_maps_to_nat():
    out = DataFrame({"t": [nan, float(EPOCH_S)]}).astype({"t": "datetime64[s]"})["t"]
    assert out.isna().to_list() == [True, False]
    assert _isnat(out) == [True, False]


def test_astype_datetime_int_na_maps_to_nat():
    out = DataFrame({"t": [EPOCH_S, None]}).astype({"t": "datetime64[s]"})["t"]
    assert out.isna().to_list() == [False, True]


def test_series_astype_datetime_na():
    out = DataFrame({"t": [EPOCH_S, None]})["t"].astype("datetime64[s]")
    assert out.isna().to_list() == [False, True]
    fout = DataFrame({"t": [nan, float(EPOCH_S)]})["t"].astype("datetime64[s]")
    assert fout.isna().to_list() == [True, False]


def test_to_datetime_all_na_and_mixed_positions():
    assert to_datetime([nan, nan], unit="s").isna().to_list() == [True, True]
    # leading / trailing / interior NaN all survive as NaT, present values convert
    s = to_datetime([nan, float(EPOCH_S), nan, float(EPOCH_S + 60), nan], unit="s")
    assert s.isna().to_list() == [True, False, True, False, True]


def test_to_datetime_every_unit_handles_na():
    for unit in ["s", "ms", "us", "ns"]:
        s = to_datetime([nan, 1.0], unit=unit)
        assert s.isna().to_list() == [True, False], unit
        # the present value still scales by the unit
        assert not np.isnat(s.to_numpy()[1])


def test_to_datetime_matches_pandas_nat():
    import pandas as pd

    for data in ([nan, float(EPOCH_S)], [float(EPOCH_S), nan, float(EPOCH_S + 1)]):
        got = to_datetime(data, unit="s").to_numpy()
        want = pd.to_datetime(data, unit="s").to_numpy()
        assert np.isnat(got).tolist() == np.isnat(want).tolist()
        assert (got[~np.isnat(got)] == want[~np.isnat(want)]).all()


def test_to_datetime_bad_unit_is_plain_valueerror_not_directive():
    # a real conversion error is a plain ValueError (these conversions are NOT
    # directives, so they must not raise the directive-specific subclass)
    with pytest.raises(ValueError) as ei:
        to_datetime([1.0, 2.0], unit="weeks")
    assert type(ei.value).__name__ == "ValueError"
    with pytest.raises(ValueError) as ei2:
        DataFrame({"t": [1, 2]}).astype({"t": "datetime64[weeks]"})
    assert type(ei2.value).__name__ == "ValueError"


def test_to_datetime_non_numeric_column_is_typeerror():
    # reading a bool column as an epoch is a type error, surfaced as TypeError
    with pytest.raises(TypeError):
        DataFrame({"t": [True, False]}).astype({"t": "datetime64[s]"})
