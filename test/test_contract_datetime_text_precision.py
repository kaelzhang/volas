"""Contract D1 at the text boundary — datetime formatting (repr / str / to_csv /
astype('str')) preserves sub-second precision instead of truncating to whole
seconds, and parsing accepts the same fractional spelling so text output
round-trips at full nanosecond precision. Digits come in pandas-style groups of
three (.fff ms / .ffffff µs / .fffffffff ns); a whole second prints bare."""

import os
import tempfile

import numpy as np
import volas
from volas import DataFrame, Timestamp

NS = 1609459200000000123          # 2021-01-01 00:00:00.000000123 UTC


# --- formatting: 3/6/9-digit groups, whole seconds bare ----------------------

def test_repr_str_show_nanoseconds():
    ts = Timestamp(NS)
    assert repr(ts) == "Timestamp('2021-01-01 00:00:00.000000123')"
    assert str(ts) == "2021-01-01 00:00:00.000000123"


def test_fraction_digit_groups():
    assert str(Timestamp(1609459200123000000)) == "2021-01-01 00:00:00.123"      # ms
    assert str(Timestamp(1609459200000123000)) == "2021-01-01 00:00:00.000123"   # us
    assert str(Timestamp(1609459200000000123)) == "2021-01-01 00:00:00.000000123"  # ns
    assert str(Timestamp(1609459200000000000)) == "2021-01-01 00:00:00"          # whole


def test_tz_formatting_keeps_fraction():
    assert str(Timestamp(NS, tz="+08:00")) == "2021-01-01 08:00:00.000000123"


def test_astype_str_keeps_fraction():
    s = DataFrame({"t": np.array([NS], dtype="datetime64[ns]")})["t"]
    assert s.astype("str").to_list() == ["2021-01-01 00:00:00.000000123"]


# --- parsing: the emitted spelling parses back (round-trip) ------------------

def test_parse_fractional_string_roundtrip():
    assert Timestamp("2021-01-01 00:00:00.000000123").value == NS
    assert Timestamp("2021-01-01 00:00:00.123").value == 1609459200123000000
    assert Timestamp("2021-01-01 00:00:00.5+00:00").value == 1609459200500000000  # offset-aware


def test_parse_minute_resolution():
    # the everyday intraday spelling 'YYYY-MM-DD HH:MM' (pandas-parity)
    ts = Timestamp("2021-01-04 14:30")
    assert (ts.hour, ts.minute, ts.second) == (14, 30, 0)
    assert Timestamp("2021-01-04T14:30") == ts


def test_str_label_lookup_with_fraction():
    df = DataFrame({"t": np.array([NS, NS + 1], dtype="datetime64[ns]"), "v": [1.0, 2.0]}).set_index("t")
    assert df.loc["2021-01-01 00:00:00.000000123"]["v"] == 1.0


# --- CSV: a full ns round-trip through the text boundary ---------------------

def test_csv_roundtrip_preserves_nanoseconds():
    df = DataFrame({"t": np.array([NS, NS + 1], dtype="datetime64[ns]"), "v": [1.0, 2.0]})
    path = tempfile.mktemp(suffix=".csv")
    try:
        df.to_csv(path, index=False)
        text = open(path).read()
        assert "2021-01-01 00:00:00.000000123" in text
        back = volas.read_csv(path).astype({"t": "datetime64[ns]"})
        assert list(back["t"].to_numpy().astype("int64")) == [NS, NS + 1]
    finally:
        os.remove(path)


def test_csv_whole_second_format_unchanged():
    # whole-second values keep the bare spelling (no spurious '.000')
    df = DataFrame({"t": np.array([1609459200000000000], dtype="datetime64[ns]")})
    assert "2021-01-01 00:00:00\n" in df.to_csv(index=False)
