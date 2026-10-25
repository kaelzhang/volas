"""A CSV column declared as dates (``parse_dates``, or ``index_col`` with ``tz`` /
``date_unit``) is parsed from its own text tokens, before — and instead of —
generic dtype inference: an all-missing date column is a NaT column, epoch
tokens are read exactly (never through float64), and an epoch past the range
raises.
"""

import os
import tempfile

import pandas as pd
import pytest
import volas


def _read(text, **kw):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv")
        with open(p, "w") as f:
            f.write(text)
        return volas.read_csv(p, **kw)


def _ns(df):
    return df.index.view("i8").tolist()


def test_all_missing_parse_dates_column_is_nat():
    s = _read("time,x\nNA,1\n,2\n", parse_dates=["time"])["time"]
    assert str(s.dtype).startswith("datetime64") and s.isna().to_list() == [True, True]


def test_parse_dates_keeps_missing_and_parses_present_tokens():
    s = _read("time,x\nNA,1\n2024-01-01,2\n", parse_dates=["time"])["time"]
    assert s.isna().to_list() == [True, False]
    assert s.to_numpy().view("i8")[1] == pd.Timestamp("2024-01-01").value


@pytest.mark.parametrize("tokens, unit, want", [
    (["1704067200000000001", "1704067200000000500.0"], "ns", [1704067200000000001, 1704067200000000500]),
    (["1704067200.123456789", "1704067200.5"], "s", [1704067200123456789, 1704067200500000000]),
    (["1704067200250", "1.7040672005e12"], "ms", [1704067200250000000, 1704067200500000000]),
    (["-1.5", "0.0000000005"], "s", [-1500000000, 0]),
])
def test_epoch_tokens_are_read_exactly(tokens, unit, want):
    df = _read("ts,x\n" + "".join(f"{t},{i}\n" for i, t in enumerate(tokens)), index_col="ts", date_unit=unit)
    assert _ns(df) == want


@pytest.mark.parametrize("token", ["9223372036854775808", "1e400", "-9223372036854775808"])
def test_epoch_tokens_past_the_range_raise(token):
    with pytest.raises(OverflowError):
        _read(f"ts,x\n{token},1\n", index_col="ts", date_unit="ns")


@pytest.mark.parametrize("token", ["inf", "-inf", "nan", "2024-01-01", "1e", "1.2.3"])
def test_non_epoch_tokens_raise(token):
    with pytest.raises(ValueError):
        _read(f"ts,x\n{token},1\n", index_col="ts", date_unit="s", keep_default_na=False)


def test_missing_epoch_token_is_nat():
    text = "ts,x\n1,1\nNA,2\n"
    column = volas.to_datetime(_read(text)["ts"], unit="s")
    assert column.to_numpy().view("i8").tolist() == [10**9, -(2**63)]
    with pytest.raises(ValueError, match="without NaT"):
        _read(text, index_col="ts", date_unit="s")


def test_index_column_also_in_parse_dates_is_read_in_tz():
    df = _read("time,x\n2024-01-01 12:00,1\n", parse_dates=["time"], index_col="time", tz="America/New_York")
    assert df.tz == "America/New_York"
    assert _ns(df) == [pd.Timestamp("2024-01-01 12:00", tz="America/New_York").value]


def test_index_col_position_and_unknown_parse_dates():
    df = _read("x,time\n1,1704067200\n", index_col=-1, date_unit="s", tz="UTC")
    assert df.tz == "UTC" and _ns(df) == [1704067200 * 10**9]
    with pytest.raises(KeyError):
        _read("time,x\n2024-01-01,1\n", parse_dates=["nope"])
