"""Time-frame periods across DST transitions and zones.

Sub-day time frames bucket by real instants on the frame's local grid: an hour a
fall-back repeats is two bars with distinct instant labels, and a grid point a
spring-forward gap skips starts its bar where the gap ends. Day-and-longer frames
keep the local civil calendar. The grid belongs to the frame's own zone, never to
an appended bar's. Live folding (``append``) and one-shot ``cumulate`` agree bar
for bar, and the 1h bars match pandas ``resample('1h')`` (the pinned oracle).
"""

import numpy as np
import pandas as pd
import pytest

import volas

NY = 'America/New_York'


def _minute_bars(start_utc, end_utc, zone):
    """1-minute bars [start, end) in UTC, as a pandas frame indexed in ``zone``."""
    idx = pd.date_range(start_utc, end_utc, freq='1min', inclusive='left', tz='UTC').tz_convert(zone)
    n = len(idx)
    price = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame({'open': price, 'high': price + 1, 'low': price - 1, 'close': price,
                         'volume': np.ones(n)}, index=idx)


def _oracle(pdf, rule):
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return pdf.resample(rule).agg(agg).dropna()


def _assert_matches(vdf, expected):
    got = vdf.to_pandas()
    assert list(got.index) == list(expected.index)
    for c in expected.columns:
        assert got[c].tolist() == expected[c].tolist(), c


def _fold(pdf, tf):
    """Live fold: seed with the first bar, append the rest one bar at a time."""
    fine = volas.DataFrame.from_pandas(pdf)
    df = fine.iloc[0:1].cumulate(tf)
    for i in range(1, fine.shape[0]):
        df.append(fine.iloc[i:i + 1])
    return df


@pytest.mark.parametrize('day', ['2026-11-01', '2026-03-08'])
def test_new_york_hourly_bars_match_pandas_and_fold_matches_cumulate(day):
    # 2026-11-01: 01:00-02:00 repeats (05:00Z EDT, 06:00Z EST) -> two 01:00 bars;
    # 2026-03-08: 02:00 EST springs to 03:00 EDT (07:00Z) -> no 02:00 bar.
    pdf = _minute_bars(f'{day} 04:00', f'{day} 09:00', NY)
    expected = _oracle(pdf, '1h')
    one_shot = volas.DataFrame.from_pandas(pdf).cumulate('1h')
    _assert_matches(one_shot, expected)
    live = _fold(pdf, '1h')
    assert live.equals(one_shot)
    if day == '2026-11-01':
        labels = one_shot.to_pandas().index
        ones = [t for t in labels if t.hour == 1]
        assert len(ones) == 2 and ones[0] != ones[1]  # two distinct 01:00 instants
        assert one_shot['volume'].to_list() == [60.0] * 5


def test_fall_back_day_frames_stay_on_the_civil_calendar():
    pdf = _minute_bars('2026-11-01 04:00', '2026-11-02 05:00', NY)  # the whole 25-hour day
    daily = volas.DataFrame.from_pandas(pdf).cumulate('1d')
    _assert_matches(daily, _oracle(pdf, '1D'))
    assert daily['volume'].to_list() == [25 * 60.0]
    assert _fold(pdf, '1d').equals(daily)


def test_two_hour_bars_keep_the_local_grid_across_the_fold():
    # the 2h grid has no point inside the repeated hour: 00:00-02:00 local is one
    # 3-hour bar, and the bars keep starting at even local hours afterwards
    pdf = _minute_bars('2026-11-01 04:00', '2026-11-01 11:00', NY)
    bars = volas.DataFrame.from_pandas(pdf).cumulate('2h')
    labels = bars.to_pandas().index
    assert [t.hour for t in labels] == [0, 2, 4]
    assert bars['volume'].to_list() == [180.0, 120.0, 120.0]
    assert _fold(pdf, '2h').equals(bars)


def test_period_grid_is_the_frames_zone_not_the_bars():
    # a Shanghai daily frame folds a bar carried in UTC by its Shanghai day:
    # 2024-01-02 10:00Z is 18:00 on 01-02 in Shanghai -> the open 01-02 bar.
    day = pd.DataFrame({'close': [1.0], 'volume': [1.0]},
                       index=pd.DatetimeIndex(['2024-01-02 00:00'], tz='Asia/Shanghai'))
    df = volas.DataFrame.from_pandas(day).cumulate('1d')
    bar = pd.DataFrame({'close': [2.0], 'volume': [1.0]},
                       index=pd.DatetimeIndex(['2024-01-02 10:00'], tz='UTC'))
    df.append(volas.DataFrame.from_pandas(bar))
    assert df.shape[0] == 1
    assert df['volume'].to_list() == [2.0]
    # 2024-01-02 17:00Z is 01:00 on 01-03 in Shanghai: a new day, labelled at its
    # Shanghai midnight (2024-01-02 16:00Z) in the frame's zone
    nxt = pd.DataFrame({'close': [3.0], 'volume': [1.0]},
                       index=pd.DatetimeIndex(['2024-01-02 17:00'], tz='UTC'))
    df.append(volas.DataFrame.from_pandas(nxt))
    out = df.to_pandas()
    assert str(out.index.tz) == 'Asia/Shanghai'
    assert list(out.index) == list(pd.DatetimeIndex(['2024-01-02', '2024-01-03'], tz='Asia/Shanghai'))


def test_a_period_start_outside_the_range_raises():
    # the first instants of the datetime64[ns] axis have no whole-week start
    first = np.array([np.iinfo(np.int64).min + 1], dtype='int64').view('datetime64[ns]')
    pdf = pd.DataFrame({'volume': [1.0]}, index=pd.DatetimeIndex(first))
    with pytest.raises(OverflowError, match='the 1w period holding'):
        volas.DataFrame.from_pandas(pdf).cumulate('1w')
