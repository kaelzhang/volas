"""Per-frame timezone ingestion + tz_localize/convert + matching (audit PD-20).

Timezone handling is pandas-aligned: parse a column to UTC instants with
:func:`volas.to_datetime`, promote it to the index with ``set_index``, then tag the display
zone with ``tz_localize`` (reinterpret naive wall-clock as that zone) or ``tz_convert`` (keep
the instant, restate the zone). ``_ingest`` below is exactly that idiom; the tests exercise
it across the A-share / HK / US / crypto cases.
"""

import numpy as np
import pytest
import volas
from volas import DataFrame, to_datetime


def _ingest(data, col='t', tz=None, unit='ns'):
    """parse ``col`` to UTC instants, set it as the index, optionally localize a display zone.

    The canonical ingestion chain: ``to_datetime`` -> ``set_index`` -> ``tz_localize``."""
    df = DataFrame(data)
    df[col] = to_datetime(df[col], unit=unit)
    df = df.set_index(col)
    return df.tz_localize(tz) if tz is not None else df


def test_naive_string_ingested_in_tz():
    # A-share / HK style: naive local strings interpreted as UTC+8, stored UTC.
    df = _ingest({'t': ['2020-01-01 09:30:00'], 'c': [1.0]}, tz='+08:00')
    assert df.tz == '+08:00'
    # the stored instant is 01:30 UTC (09:30 - 8h)
    assert df.index.astype('datetime64[ns]')[0] == np.datetime64('2020-01-01 01:30:00')


def test_offset_aware_strings_are_absolute():
    df = _ingest({'t': ['2020-01-01T09:30:00+08:00'], 'c': [1.0]})
    assert df.index.astype('datetime64[ns]')[0] == np.datetime64('2020-01-01 01:30:00')


def test_epoch_ms_ingestion():
    # exchange API returns epoch-ms; most robust path.
    df = _ingest({'t': np.array([1577836800000], dtype=np.int64), 'c': [1.0]}, unit='ms')
    assert df.index.astype('datetime64[ns]')[0] == np.datetime64('2020-01-01 00:00:00')


def test_named_zone_dst_ingestion():
    # US equities in New York time, across the DST boundary.
    df = _ingest(
        {'t': ['2021-01-04 09:30:00', '2021-07-01 09:30:00'], 'c': [1.0, 2.0]},
        tz='America/New_York')
    inst = df.index.astype('datetime64[ns]')
    # winter EST = UTC-5 -> 14:30 UTC; summer EDT = UTC-4 -> 13:30 UTC
    assert inst[0] == np.datetime64('2021-01-04 14:30:00')
    assert inst[1] == np.datetime64('2021-07-01 13:30:00')


def test_tz_localize_then_convert():
    # ingested without tz (treated UTC), then attach NY (wall-clock unchanged)
    df = _ingest({'t': ['2021-01-04 09:30:00'], 'c': [1.0]})
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
    df = _ingest(
        {'t': ['2021-01-04 09:30:00', '2021-01-04 09:31:00'], 'c': [1.0, 2.0]},
        tz='America/New_York')
    # a bare string label is interpreted in the frame's tz
    assert df.at['2021-01-04 09:30:00', 'c'] == 1.0


def test_cross_market_align_on_utc_axis():
    # crypto (UTC) and US (NY) bars at the same instant align on the UTC index
    crypto = _ingest({'t': ['2021-01-04 14:30:00'], 'p': [100.0]})
    us = _ingest({'t': ['2021-01-04 09:30:00'], 'p': [200.0]}, tz='America/New_York')
    assert (crypto.index.astype('datetime64[ns]')[0]
            == us.index.astype('datetime64[ns]')[0])


def test_constructor_rejects_retired_tz_kwarg():
    # tz / date_col / date_unit were retired; the constructor no longer accepts them.
    with pytest.raises(TypeError):
        DataFrame({'c': [1.0]}, tz='+08:00')


def test_unknown_tz_errors():
    with pytest.raises(ValueError):
        _ingest({'t': ['2020-01-01'], 'c': [1.0]}, tz='Not/AZone')


# --- volas.Timestamp (typed, cross-tz label) -------------------------------

def test_timestamp_resolves_to_utc():
    ts = volas.Timestamp('2021-01-04 09:30:00', tz='America/New_York')
    assert ts.tz == 'America/New_York'
    assert ts.to_numpy()[0] == np.datetime64('2021-01-04 14:30:00')   # 09:30 NY -> 14:30 UTC


def test_timestamp_cross_tz_loc_match():
    # frame displayed in NY; query with a Shanghai Timestamp at the same instant
    df = _ingest(
        {'t': ['2021-01-04 09:30:00', '2021-01-04 09:31:00'], 'c': [1.0, 2.0]},
        tz='America/New_York')
    # 2021-01-04 22:30 +08:00 == 14:30 UTC == 09:30 NY -> matches row 0
    ts = volas.Timestamp('2021-01-04 22:30:00', tz='+08:00')
    assert df.at[ts, 'c'] == 1.0


def test_timestamp_slice_bounds():
    df = _ingest(
        {'t': ['2021-01-04 09:30:00', '2021-01-04 09:31:00', '2021-01-04 09:32:00'],
         'c': [1.0, 2.0, 3.0]},
        tz='America/New_York')
    lo = volas.Timestamp('2021-01-04 09:31:00', tz='America/New_York')
    hi = volas.Timestamp('2021-01-04 09:32:00', tz='America/New_York')
    sub = df.loc[lo:hi]
    assert sub['c'].to_list() == [2.0, 3.0]


def test_timestamp_compare():
    a = volas.Timestamp('2021-01-01 00:00:00')
    b = volas.Timestamp('2021-01-02 00:00:00')
    assert a < b and b > a and a == volas.Timestamp('2021-01-01')


# --- cumulate aligns daily buckets to the frame tz -------------------------

def test_cumulate_daily_buckets_align_to_ny_day():
    # two bars on different NY calendar days that fall on the SAME UTC day
    # (23:00 NY -> 04:00 UTC next day; 01:00 NY next day -> 06:00 UTC same day)
    df_ny = _ingest(
        {'t': ['2021-01-04 23:00:00', '2021-01-05 01:00:00'],
         'open': [1.0, 2.0], 'high': [1.0, 2.0], 'low': [1.0, 2.0],
         'close': [1.0, 2.0], 'volume': [10.0, 20.0]},
        tz='America/New_York')
    daily_ny = df_ny.cumulate('1d')
    # NY-day grouping -> two distinct trading days
    assert daily_ny.shape[0] == 2

    # same instants as a UTC frame collapse into one UTC day
    df_utc = _ingest(
        {'t': ['2021-01-05 04:00:00', '2021-01-05 06:00:00'],
         'open': [1.0, 2.0], 'high': [1.0, 2.0], 'low': [1.0, 2.0],
         'close': [1.0, 2.0], 'volume': [10.0, 20.0]})
    daily_utc = df_utc.cumulate('1d')
    assert daily_utc.shape[0] == 1
