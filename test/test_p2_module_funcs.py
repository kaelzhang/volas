"""P2 module-level / IO: to_datetime(format=) and to_csv float fidelity +
float_format."""

import numpy as np
import pytest

import volas
from volas import DataFrame

nan = float("nan")


# --- to_datetime(format=) ---------------------------------------------------

def test_to_datetime_format_date_only_is_midnight():
    got = volas.to_datetime(DataFrame({"t": ["2020-01-15", "2020-02-20"]})["t"], format="%Y-%m-%d")
    np.testing.assert_array_equal(
        got.to_numpy(), np.array(["2020-01-15", "2020-02-20"], dtype="datetime64[ns]")
    )


def test_to_datetime_format_with_time():
    got = volas.to_datetime(DataFrame({"t": ["15/01/2020 09:30"]})["t"], format="%d/%m/%Y %H:%M")
    assert got.to_numpy()[0] == np.datetime64("2020-01-15T09:30", "ns")


def test_to_datetime_format_mismatch_raises():
    with pytest.raises(ValueError, match="does not match format"):
        volas.to_datetime(DataFrame({"t": ["not a date"]})["t"], format="%Y-%m-%d")


def test_to_datetime_no_format_still_autoparses():
    got = volas.to_datetime(DataFrame({"t": ["2020-01-15"]})["t"])
    assert got.dtype == "datetime64[ns]"


# --- to_csv float fidelity --------------------------------------------------

def _df(a):
    return DataFrame({"a": list(a)})


def test_to_csv_keeps_float_decimal_point():
    # the bug: 1.0 used to be written as "1"
    assert _df([1.0, 2.5, 3.0]).to_csv(index=False) == "a\n1.0\n2.5\n3.0\n"


def test_to_csv_na_rep_for_nan():
    assert _df([1.0, nan]).to_csv(index=False, na_rep="NA") == "a\n1.0\nNA\n"


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("%.2f", "a\n1.00\n2.56\n"),
        ("%f", "a\n1.000000\n2.555000\n"),
        ("%.3e", "a\n1.000e0\n2.555e0\n"),
        ("%e", "a\n1e0\n2.555e0\n"),
        ("%.2g", "a\n1.00\n2.56\n"),
        ("%g", "a\n1.0\n2.555\n"),
    ],
)
def test_to_csv_float_format(fmt, expected):
    assert _df([1.0, 2.555]).to_csv(index=False, float_format=fmt) == expected


@pytest.mark.parametrize("bad", ["%.2x", "%2f", "%.xf", "%"])
def test_to_csv_bad_float_format_raises(bad):
    with pytest.raises(ValueError, match="float_format"):
        _df([1.0]).to_csv(float_format=bad)
