"""volas cumulation (TimeFrame resampling) tests.

Ported / adapted from stock-pandas's ``test_time_frame.py`` and
``test_cum_append.py`` to volas's native API: the one-shot ``df.cumulate(tf)``
(which returns a **tf-aware** DataFrame) and live folding via ``df.append`` —
a finer bar in the open period updates the forming last row, a bar in a new
period rolls over. 1-minute Tencent bars are re-stamped and aggregated to
coarser frames.
"""

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import volas
from volas import DataFrame, TimeFrame, to_datetime

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
    df = DataFrame(data)
    df['time_key'] = to_datetime(df['time_key'])
    return df.set_index('time_key')


# --- TimeFrame --------------------------------------------------------------

def test_time_frame_str():
    assert str(TimeFrame.m1) == '1m'
    assert str(TimeFrame.s1) == '1s'
    assert str(TimeFrame.D1) == '1d'
    assert str(TimeFrame.M1) == '1M'
    assert f'{TimeFrame.m5}' == '5m'


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


# --- tf-aware DataFrame (live folding) --------------------------------------

def _seed(fine, tf='5m'):
    """A tf-aware frame seeded with the first fine bar (the rest get folded)."""
    return fine.iloc[0:1].cumulate(tf)


def test_tf_append_incremental_matches_one_shot():
    fine = get_1m()
    df = _seed(fine)
    for i in range(1, len(fine)):
        df.append(fine.iloc[i:i + 1])
    assert df.equals(fine.cumulate('5m'))


def test_tf_append_batch_matches_one_shot():
    fine = get_1m()
    df = _seed(fine)
    df.append(fine.iloc[1:])  # fold the rest in one call
    assert df.equals(fine.cumulate('5m'))


def test_cumulate_keeps_index_name():
    # pandas resample keeps the source index name; volas matches
    fine = get_1m()  # index named 'time_key'
    assert fine.cumulate('5m').reset_index().columns[0] == 'time_key'


def test_tf_append_step_lengths():
    fine = get_1m()
    df = _seed(fine)
    expect_cumulated(fine, df, 1)
    for i in range(1, len(fine)):
        df.append(fine.iloc[i:i + 1])
        expect_cumulated(fine, df, i + 1)


def test_tf_forming_bar_is_last_row():
    fine = get_1m()
    df = _seed(fine)
    df.append(fine.iloc[1:3])  # bars 0,1,2 all in the first 5m period
    assert df.shape[0] == 1  # still the single, forming period
    assert df['volume'].to_numpy()[-1] == fine.iloc[0:3]['volume'].to_numpy().sum()


def test_tf_empty_append_raises():
    df = _seed(get_1m())
    with pytest.raises(Exception):
        df.append(volas.read_csv(TENCENT).iloc[0:0])


def test_tf_dedups_resent_forming_bar():
    fine = get_1m(3)
    df = _seed(fine)
    df.append(fine.iloc[1:2])
    df.append(fine.iloc[1:2])  # re-sent bar 1
    df.append(fine.iloc[2:3])
    # all in one 5m period; the re-sent bar updates, not accumulates
    assert df['volume'].to_numpy()[0] == fine['volume'].to_numpy().sum()


def test_tf_construct_requires_datetime_index():
    with pytest.raises(Exception):
        volas.DataFrame({'close': [1.0, 2.0]}, time_frame='5m')  # RangeIndex


def test_tf_cumulators_requires_time_frame():
    with pytest.raises(Exception):
        volas.DataFrame({'close': [1.0]}, cumulators={'volume': 'sum'})


def test_tf_illegal_coarsen_raises():
    df3 = get_1m().cumulate('3m')
    with pytest.raises(Exception):  # 5 is not a whole multiple of 3
        df3.cumulate('5m')
    assert df3.cumulate('15m').shape[0] >= 1  # 15 = 5 * 3 is legal


def test_tf_same_frame_cumulate_is_copy():
    df5 = get_1m().cumulate('5m')
    assert df5.cumulate('5m').equals(df5)


def test_tf_copy_preserves_folding():
    fine = get_1m()
    df = fine.iloc[0:5].cumulate('5m')  # one full 5m period (bars 0-4), forming
    c = df.copy()
    c.append(fine.iloc[5:6])  # bar 5 starts a new period in the copy
    assert c.shape[0] == 2 and df.shape[0] == 1  # original untouched


def test_tf_fir_indicator_incremental_matches_one_shot():
    # A finite-window (FIR) indicator cached across folds matches the one-shot.
    fine = get_1m()
    df = _seed(fine)
    _ = df['ma:2']  # cache the directive column
    for i in range(1, len(fine)):
        df.append(fine.iloc[i:i + 1])
    np.testing.assert_allclose(
        df['ma:2'].to_numpy(), fine.cumulate('5m').exec('ma:2'), equal_nan=True
    )


def test_tf_iir_indicator_incremental_matches_one_shot():
    # Recursive indicators (EMA, ...) also match one-shot under incremental
    # folding: the directive cache carries the recursive state across appends,
    # so the forming bar recomputes correctly rather than re-seeding the window.
    fine = get_1m()
    df = _seed(fine)
    _ = df['ema:2']
    for i in range(1, len(fine)):
        df.append(fine.iloc[i:i + 1])
    np.testing.assert_allclose(
        df['ema:2'].to_numpy(), fine.cumulate('5m').exec('ema:2'), equal_nan=True
    )
