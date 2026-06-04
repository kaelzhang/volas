"""Per-frame timezone ingestion + tz_localize/convert + matching (audit PD-20)."""

import numpy as np
import pytest
import volas
from volas import DataFrame


def test_naive_string_ingested_in_tz():
    # A-share / HK style: naive local strings interpreted as UTC+8, stored UTC.
    df = DataFrame({'t': ['2020-01-01 09:30:00'], 'c': [1.0]},
                   date_col='t', tz='+08:00')
    assert df.tz == '+08:00'
    # the stored instant is 01:30 UTC (09:30 - 8h)
    assert df.index.astype('datetime64[ns]')[0] == np.datetime64('2020-01-01 01:30:00')


def test_offset_aware_strings_are_absolute():
    df = DataFrame({'t': ['2020-01-01T09:30:00+08:00'], 'c': [1.0]}, date_col='t')
    assert df.index.astype('datetime64[ns]')[0] == np.datetime64('2020-01-01 01:30:00')


def test_epoch_ms_ingestion():
    # exchange API returns epoch-ms; most robust path.
    df = DataFrame({'t': np.array([1577836800000], dtype=np.int64), 'c': [1.0]},
                   date_col='t', date_unit='ms')
    assert df.index.astype('datetime64[ns]')[0] == np.datetime64('2020-01-01 00:00:00')


def test_named_zone_dst_ingestion():
    # US equities in New York time, across the DST boundary.
    df = DataFrame(
        {'t': ['2021-01-04 09:30:00', '2021-07-01 09:30:00'], 'c': [1.0, 2.0]},
        date_col='t', tz='America/New_York')
    inst = df.index.astype('datetime64[ns]')
    # winter EST = UTC-5 -> 14:30 UTC; summer EDT = UTC-4 -> 13:30 UTC
    assert inst[0] == np.datetime64('2021-01-04 14:30:00')
    assert inst[1] == np.datetime64('2021-07-01 13:30:00')


def test_tz_localize_then_convert():
    # ingested without tz (treated UTC), then attach NY (wall-clock unchanged)
    df = DataFrame({'t': ['2021-01-04 09:30:00'], 'c': [1.0]}, date_col='t')
    assert df.tz is None
    ny = df.tz_localize('America/New_York')
    assert ny.tz == 'America/New_York'
    # the wall-clock 09:30 now means NY -> 14:30 UTC
    assert ny.index.astype('datetime64[ns]')[0] == np.datetime64('2021-01-04 14:30:00')
    # convert keeps the instant, only changes display tag
    shanghai = ny.tz_convert('+08:00')
    assert shanghai.index.astype('datetime64[ns]')[0] == np.datetime64('2021-01-04 14:30:00')
    assert shanghai.tz == '+08:00'


def test_bare_string_loc_matches_in_index_tz():
    df = DataFrame(
        {'t': ['2021-01-04 09:30:00', '2021-01-04 09:31:00'], 'c': [1.0, 2.0]},
        date_col='t', tz='America/New_York')
    # a bare string label is interpreted in the frame's tz
    assert df.at['2021-01-04 09:30:00', 'c'] == 1.0


def test_cross_market_align_on_utc_axis():
    # crypto (UTC) and US (NY) bars at the same instant align on the UTC index
    crypto = DataFrame({'t': ['2021-01-04 14:30:00'], 'p': [100.0]}, date_col='t')
    us = DataFrame({'t': ['2021-01-04 09:30:00'], 'p': [200.0]},
                   date_col='t', tz='America/New_York')
    assert (crypto.index.astype('datetime64[ns]')[0]
            == us.index.astype('datetime64[ns]')[0])


def test_tz_requires_date_col():
    with pytest.raises(ValueError):
        DataFrame({'c': [1.0]}, tz='+08:00')


def test_unknown_tz_errors():
    with pytest.raises(ValueError):
        DataFrame({'t': ['2020-01-01'], 'c': [1.0]}, date_col='t', tz='Not/AZone')
