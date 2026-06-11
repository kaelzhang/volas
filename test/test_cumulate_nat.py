"""Time-bucketed ingest discipline: a DatetimeIndex feeding cumulate / a
time_frame constructor / the live fold must carry present (no-NaT), time-sorted
timestamps — a violation raises instead of silently corrupting the OHLCV; and
NaT must not render as a garbage civil date."""

import numpy as np
import pytest
from volas import DataFrame


def _ohlcv_with_nat():
    return DataFrame({
        'date': ['2021-01-04T14:30:00', None, '2021-01-04T14:31:00'],
        'open': [1.0, 2.0, 3.0], 'high': [1.0, 2.0, 3.0], 'low': [1.0, 2.0, 3.0],
        'close': [1.0, 2.0, 3.0], 'volume': [10.0, 20.0, 30.0],
    }).astype({'date': 'datetime64[ns]'}).set_index('date')


def test_cumulate_rejects_nat_index():
    # P1-03: a NaT index row used to be bucketed as its own period (silent wrong
    # OHLCV, shape (3,5) instead of an aggregated daily bar). It must now raise.
    df = _ohlcv_with_nat()
    with pytest.raises(ValueError):
        df.cumulate('1d')


def test_cumulate_clean_index_still_works():
    # a clean DatetimeIndex still aggregates (regression guard for the new check)
    df = DataFrame({
        'date': ['2021-01-04T14:30:00', '2021-01-04T14:31:00'],
        'open': [1.0, 3.0], 'high': [1.0, 3.0], 'low': [1.0, 3.0],
        'close': [1.0, 3.0], 'volume': [10.0, 30.0],
    }).astype({'date': 'datetime64[ns]'}).set_index('date')
    out = df.cumulate('1d')
    assert out.shape == (1, 5) and out['close'].to_list() == [3.0]


def test_nat_astype_str_is_nat_not_garbage_date():
    # P1-03 (civil_parts/format): NaT must stringify as 'NaT', not 1677-09-21
    s = DataFrame({'d': np.array(['2021-03-15', 'NaT'], dtype='datetime64[ns]')})['d']
    assert s.astype('str').to_list() == ['2021-03-15 00:00:00', 'NaT']


# --- tf-aware append guards: the live fold is symmetric with cumulate ---------
# cumulate() already rejects NaT (above); the live append/fold path must enforce
# the same monotonic, present-timestamp discipline (R4-P1-01 / R4-P1-02).

def _forming_5m():
    # a 5m frame with an open (forming) 14:30 period
    return DataFrame(
        {'t': np.array(['2021-01-04T14:30:00'], dtype='datetime64[ns]'), 'c': [1.0]}
    ).set_index('t').cumulate('5m')


def _bar(ts, c):
    return DataFrame({'t': np.array([ts], dtype='datetime64[ns]'), 'c': [c]}).set_index('t')


def test_tf_append_nat_bar_rejected():
    # R4-P1-02: a NaT bar has no period — symmetric with cumulate()'s NaT rejection
    nat = DataFrame({'t': np.array(['NaT'], dtype='datetime64[ns]'), 'c': [7.0]}).set_index('t')
    with pytest.raises(ValueError):
        _forming_5m().append(nat)


def test_tf_append_out_of_order_bar_rejected():
    # R4-P1-01: a bar earlier than the forming period's latest bar would produce a
    # non-monotonic index / fold later bars into the wrong period
    tf = _forming_5m()
    tf.append(_bar('2021-01-04T14:31:00', 2.0))  # same period -> folds (height stays 1)
    assert tf.shape == (1, 1) and tf['c'].to_list() == [2.0]
    with pytest.raises(ValueError):
        tf.append(_bar('2021-01-04T14:20:00', 9.0))  # earlier than 14:31 -> rejected


# --- batch cumulate / tf constructor: same ordering discipline (V15) ----------
# group_runs is consecutive-run grouping and OHLCV open/close is position-
# ordered, so an out-of-order batch input silently produced duplicate period
# rows and time-wrong open/close. The batch path now rejects it like the live
# fold (sorting stays an explicit caller action — never implicit).

def _bars(times, closes):
    t = np.array(times, dtype='datetime64[ns]')
    return DataFrame({'t': t, 'close': closes}).set_index('t')


def test_cumulate_rejects_reversed_same_period_bars():
    # reversed [14:31, 14:30]: position-last (14:30) used to become close = 1.0
    with pytest.raises(ValueError):
        _bars(['2021-01-04T14:31', '2021-01-04T14:30'], [2.0, 1.0]).cumulate('5m')


def test_cumulate_rejects_interleaved_periods():
    # [14:30, 14:35, 14:31] used to emit 3 rows with a duplicate 14:30 period
    with pytest.raises(ValueError):
        _bars(['2021-01-04T14:30', '2021-01-04T14:35', '2021-01-04T14:31'],
              [1.0, 5.0, 2.0]).cumulate('5m')


def test_cumulate_sorted_and_duplicate_ts_still_work():
    out = _bars(['2021-01-04T14:30', '2021-01-04T14:31', '2021-01-04T14:35'],
                [1.0, 2.0, 5.0]).cumulate('5m')
    assert out.shape == (2, 1) and out['close'].to_list() == [2.0, 5.0]
    # equal timestamps are non-decreasing (a re-sent bar), still accepted
    dup = _bars(['2021-01-04T14:30', '2021-01-04T14:30'], [1.0, 2.0]).cumulate('5m')
    assert dup['close'].to_list() == [2.0]


def test_time_frame_constructor_rejects_unsorted_or_nat_bars():
    # the constructor takes the rows as final bars and the last row as the
    # forming-period cursor — the same invariants must hold from the start
    with pytest.raises(ValueError):
        DataFrame(_bars(['2021-01-04T14:31', '2021-01-04T14:30'], [2.0, 1.0]),
                  time_frame='5m')
    nat_bars = DataFrame(
        {'t': np.array(['2021-01-04T14:30', 'NaT'], dtype='datetime64[ns]'),
         'close': [1.0, 2.0]}).set_index('t')
    with pytest.raises(ValueError):
        DataFrame(nat_bars, time_frame='5m')


def test_tf_append_in_order_and_resent_bar_still_work():
    # the guard rejects only strictly-earlier bars: same-ts re-sends and later bars
    # (same period or a roll-over) are accepted.
    tf = _forming_5m()
    tf.append(_bar('2021-01-04T14:31:00', 2.0))     # later, same period -> fold
    tf.append(_bar('2021-01-04T14:31:00', 2.5))     # re-sent forming bar (==) -> ok
    assert tf.shape == (1, 1)
    tf.append(_bar('2021-01-04T14:36:00', 3.0))     # later, new period -> roll over
    assert tf.shape == (2, 1)
