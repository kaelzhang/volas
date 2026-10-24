"""Publishing a live aggregate commits its frame and both checkpoints together."""

import numpy as np
import pandas as pd
import pytest

from volas import DataFrame


def held_history(mode, order=('close', 'volume')):
    stamps = pd.date_range('2024-01-01', periods=4, freq='D')
    columns = {'close': np.arange(4, dtype=float),
               'volume': np.full(4, 2**63 - 1, dtype='int64')}
    source = DataFrame({name: columns[name] for name in order}, index=stamps,
                       time_frame='1d', cumulators={name: 'sum' for name in order})
    source['ma:2']
    live = source.iloc[3:]
    live['close'] = np.array([2**53], dtype='int64')
    live.fulfill()
    stamp = stamps[-1] + pd.Timedelta(minutes=1)
    if mode != 'refold':
        live.append(DataFrame({'close': np.array([0], dtype='int64'),
                               'volume': np.array([0], dtype='int64')}, index=[stamp]))
        live.fulfill()
        if mode == 'fold':
            stamp += pd.Timedelta(minutes=1)
    return live, stamp


def incoming(stamp, close, volume=0):
    return DataFrame({'close': np.array([close], dtype='int64'),
                      'volume': np.array([volume], dtype='int64')}, index=[stamp])


@pytest.mark.parametrize('mode', ['refold', 'fold', 'resend'])
def test_rejected_publish_then_identical_retry_keeps_checkpoints(mode):
    live, stamp = held_history(mode)
    before = live.copy()
    for _ in range(2):
        with pytest.raises(TypeError, match='close'):
            live.append(incoming(stamp, 1))
        assert live['close'].to_list() == [2**53]
        assert live.equals(before)
    # A later legal correction must not include either rejected input.
    live.append(incoming(stamp, -1))
    assert live['close'].to_list() == [2**53 - 1]
    live.append(incoming(stamp, -2))
    assert live['close'].to_list() == [2**53 - 2]


@pytest.mark.parametrize('mode', ['refold', 'fold', 'resend'])
def test_rejected_publish_then_legal_correction_keeps_checkpoints(mode):
    live, stamp = held_history(mode)
    with pytest.raises(TypeError, match='close'):
        live.append(incoming(stamp, 1))
    live.append(incoming(stamp, -1))
    assert live['close'].to_list() == [9007199254740991]
    live.append(incoming(stamp, -1))
    assert live['close'].to_list() == [9007199254740991]


@pytest.mark.parametrize('mode', ['refold', 'fold', 'resend'])
@pytest.mark.parametrize('order,error', [
    (('close', 'volume'), TypeError), (('volume', 'close'), OverflowError),
])
def test_publish_validates_all_columns_in_destination_order(mode, order, error):
    live, stamp = held_history(mode, order)
    before = live.copy()
    for _ in range(2):
        with pytest.raises(error):
            live.append(incoming(stamp, 1, 1))
        assert live['close'].to_list() == [2**53]
        assert live.equals(before)
    live.append(incoming(stamp, -1))
    assert live['close'].to_list() == [2**53 - 1]
    assert live['volume'].to_list() == [2**63 - 1]


@pytest.mark.parametrize('mode', ['refold', 'fold', 'resend'])
def test_multibar_publish_validates_history_before_committing_any_bar(mode):
    live, stamp = held_history(mode)
    before = live.copy()
    stamps = [stamp, stamp + pd.Timedelta(minutes=1)]
    bars = DataFrame({'close': np.array([-2, 3], dtype='int64'),
                      'volume': np.array([0, 0], dtype='int64')}, index=stamps)
    for _ in range(2):
        with pytest.raises(TypeError, match='close'):
            live.append(bars)
        assert live['close'].to_list() == [2**53]
        assert live.equals(before)
    live.append(incoming(stamp, -1))
    assert live['close'].to_list() == [2**53 - 1]



def test_failed_resend_does_not_become_the_next_prefix():
    live, stamp = held_history('resend')
    with pytest.raises(TypeError, match='close'):
        live.append(incoming(stamp, 1))
    later = stamp + pd.Timedelta(minutes=1)
    live.append(incoming(later, -1))
    assert live['close'].to_list() == [2**53 - 1]
    live.append(incoming(later, -1))
    assert live['close'].to_list() == [2**53 - 1]
