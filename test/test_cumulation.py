"""volas cumulation (TimeFrame resampling) tests.

Ported / adapted from stock-pandas's ``test_time_frame.py`` and
``test_cum_append.py`` to volas's native API (decision B): the stateless
``df.cumulate(tf)`` one-shot and the explicit ``volas.Cumulator(tf)`` live
aggregator (``.append`` / ``.frame`` / ``.last``). 1-minute Tencent bars are
re-stamped and aggregated to coarser frames.
"""

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import volas
from volas import DataFrame, TimeFrame, Cumulator

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())
COLUMNS = ['open', 'high', 'low', 'close', 'volume']
LENGTH = 20


def get_1m(n=LENGTH):
    """First `n` Tencent bars re-stamped at a 1-minute interval, as a volas
    DataFrame with a DatetimeIndex."""
    raw = volas.read_csv(TENCENT)
    date = datetime(2020, 1, 1)
    step = timedelta(minutes=1)
    times = []
    for _ in range(n):
        times.append(date.strftime('%Y-%m-%d %H:%M:%S'))
        date += step
    data = {c: raw[c].to_numpy()[:n].astype(float) for c in COLUMNS}
    data['time_key'] = times  # a list of str -> parsed to the DatetimeIndex
    return DataFrame(data, date_col='time_key')


# --- TimeFrame --------------------------------------------------------------

def test_time_frame_str():
    assert str(TimeFrame.m1) == '1m'
    assert str(TimeFrame.s1) == '1s'
    assert str(TimeFrame.D1) == '1d'
    assert str(TimeFrame.M1) == '1M'
    assert f'{TimeFrame.m5}' == '5m'


def test_time_frame_minutes():
    assert TimeFrame.m5.minutes == 5
    assert TimeFrame.H1.minutes == 60


def test_time_frame_unify_second():
    assert TimeFrame.s1.unify('2020-01-02 03:04:05') == 20200102030405


def test_time_frame_invalid_label_raises():
    with pytest.raises(ValueError, match='invalid'):
        volas.read_csv(TENCENT).cumulate('1')


# --- expectations -----------------------------------------------------------

def expect_period(idx, coarse, source):
    assert coarse['open'].to_numpy()[idx] == source['open'].to_numpy()[0]
    assert coarse['high'].to_numpy()[idx] == source['high'].to_numpy().max()
    assert coarse['low'].to_numpy()[idx] == source['low'].to_numpy().min()
    assert coarse['close'].to_numpy()[idx] == source['close'].to_numpy()[-1]
    assert coarse['volume'].to_numpy()[idx] == source['volume'].to_numpy().sum()


def expect_cumulated(origin, coarse, n, step=5):
    assert len(coarse) == math.ceil(n / step)
    if n == 0:
        return
    rest = n % step
    for i in range(len(coarse)):
        if i == len(coarse) - 1 and rest:
            src = origin.iloc[i * step: i * step + rest]
        else:
            src = origin.iloc[i * step: (i + 1) * step]
        expect_period(i, coarse, src)


# --- cumulate (one-shot) ----------------------------------------------------

def test_cumulate_5m():
    fine = get_1m()
    coarse = fine.cumulate('5m')
    expect_cumulated(fine, coarse, LENGTH)
    assert coarse.equals(coarse.cumulate('5m'))  # idempotent


def test_cumulate_accepts_timeframe_object():
    fine = get_1m()
    assert fine.cumulate(TimeFrame.m5).equals(fine.cumulate('5m'))


def test_cumulate_various_frames_last_period():
    fine = get_1m()
    for tf, start in [('3m', -2), ('5m', -5), ('15m', -5)]:
        coarse = fine.cumulate(tf)
        expect_period(len(coarse) - 1, coarse, fine.iloc[start:])
    for tf in ['1h', '1d', '1M', '1y']:
        coarse = fine.cumulate(tf)
        assert len(coarse) == 1
        expect_period(0, coarse, fine)


def test_cumulate_custom_cumulator_override():
    fine = get_1m()
    coarse = fine.cumulate('5m', cumulators={'volume': 'last'})
    base = fine.cumulate('5m')
    np.testing.assert_array_equal(
        coarse['open'].to_numpy(), base['open'].to_numpy()
    )
    # volume now takes the last bar of the period rather than the sum
    assert coarse['volume'].to_numpy()[0] == fine.iloc[0:5]['volume'].to_numpy()[-1]


def test_cumulate_requires_datetime_index():
    plain = volas.read_csv(TENCENT)  # RangeIndex
    with pytest.raises(Exception):
        plain.cumulate('5m')


# --- Cumulator (live incremental) -------------------------------------------

def test_cumulator_incremental_matches_one_shot():
    fine = get_1m()
    one_shot = fine.cumulate('5m')
    cum = Cumulator('5m')
    for i in range(len(fine)):
        cum.append(fine.iloc[i:i + 1])
    assert cum.frame.equals(one_shot)


def test_cumulator_batch_matches_one_shot():
    fine = get_1m()
    cum = Cumulator('5m')
    cum.append(fine)
    assert cum.frame.equals(fine.cumulate('5m'))


def test_cumulator_step_lengths():
    fine = get_1m()
    cum = Cumulator('5m')
    for i in range(len(fine)):
        cum.append(fine.iloc[i:i + 1])
        expect_cumulated(fine, cum.frame, i + 1)


def test_cumulator_last_is_open_period():
    fine = get_1m()
    cum = Cumulator('5m')
    cum.append(fine.iloc[:3])
    last = cum.last
    assert last.shape[0] == 1
    assert last['volume'].to_numpy()[0] == fine.iloc[:3]['volume'].to_numpy().sum()


def test_cumulator_empty_append_raises():
    cum = Cumulator('5m')
    empty = volas.read_csv(TENCENT).iloc[0:0]
    with pytest.raises(Exception):
        cum.append(empty)


def test_cumulator_dedups_duplicate_timestamps():
    fine = get_1m(3)
    feed = (
        fine.iloc[0:1]
        .append(fine.iloc[1:2])
        .append(fine.iloc[1:2])  # re-sent bar 1
        .append(fine.iloc[2:3])
    )
    cum = Cumulator('5m')
    cum.append(feed)
    # all in one 5m period; the duplicate updates, not accumulates
    assert cum.frame['volume'].to_numpy()[0] == fine['volume'].to_numpy().sum()
