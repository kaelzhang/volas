"""Column folds and live checkpoints preserve dtype, order, errors and raw bars."""

import numpy as np
import pandas as pd
import pytest

from volas import DataFrame


AGGS = ['first', 'last', 'sum', 'max', 'min']
DTYPES = ['float64', 'float32', 'Int64', 'Int32', 'boolean', 'string', 'datetime64[ns]']


def make(values, dtype, stamps):
    series = pd.Series(values, dtype=dtype)
    return DataFrame.from_pandas(pd.DataFrame({'v': series.array}, index=stamps))


def values_for(dtype):
    if dtype == 'string':
        return ['a', 'longer text', '中文', '', 'last']
    if dtype == 'boolean':
        return [True, False, True, False, True]
    if dtype == 'datetime64[ns]':
        return list(pd.date_range('2020-01-01', periods=5, freq='D'))
    return [1, 4, 2, 7, 3]


@pytest.mark.parametrize('agg', AGGS)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('missing', [None, 0, 2, 4])
def test_all_aggregators_and_dtypes_restore_a_revised_missing_or_extreme_bar(agg, dtype, missing):
    stamps = pd.date_range('2024-01-01', periods=5, freq='min')
    vals = values_for(dtype)
    if missing is not None:
        vals[missing] = None
    fine = make(vals, dtype, stamps)
    spec = {'v': agg}
    supported = agg in ('first', 'last') or (dtype not in ('boolean', 'string') and not (dtype == 'datetime64[ns]' and agg == 'sum'))
    if not supported:
        with pytest.raises(TypeError, match='cumulators: cannot'):
            fine.cumulate('1d', cumulators=spec)
        return
    live = fine.iloc[:1].cumulate('1d', cumulators=spec)
    for i in range(5):
        live.append(fine.iloc[i:i + 1])
        # Each of the first, middle and last bars is correctable while latest.
        for value in (None, values_for(dtype)[1], vals[i]):
            live.append(make([value], dtype, stamps[i:i + 1]))
            expected = make(vals[:i] + [value], dtype, stamps[:i + 1]).cumulate('1d', cumulators=spec)
            assert live.equals(expected), (dtype, agg, missing, i, value)
            assert live.dtypes == expected.dtypes
    held = live.copy()
    for i in (0, 2):
        with pytest.raises(ValueError, match='out-of-order'):
            live.append(fine.iloc[i:i + 1])
        assert live.equals(held)
    # Reused checkpoints must forget the old period's missing inputs and prefix.
    later = stamps + pd.Timedelta(days=1)
    for i in range(2):
        live.append(make([vals[i]], dtype, later[i:i + 1]))
        for value in (None, values_for(dtype)[1], vals[i]):
            live.append(make([value], dtype, later[i:i + 1]))
            expected = make(vals + vals[:i] + [value], dtype, stamps.append(later[:i + 1])).cumulate('1d', cumulators=spec)
            assert live.equals(expected), (dtype, agg, missing, i, value)


@pytest.mark.parametrize('agg', AGGS)
@pytest.mark.parametrize('tf', ['1h', '1d'])
@pytest.mark.parametrize('start', ['2026-03-08 05:00', '2026-11-01 04:00'])
def test_dst_gaps_and_folds_keep_the_frames_grid_during_rollover_and_resend(agg, tf, start):
    stamps = pd.date_range(start, periods=32, freq='15min', tz='UTC')
    vals = np.arange(len(stamps), dtype=float) + 1
    raw = make(vals, 'float64', stamps.tz_convert('America/New_York'))
    live = raw.iloc[:1].cumulate(tf, cumulators={'v': agg})
    incoming = make(vals, 'float64', stamps)
    for i in range(1, len(stamps)):
        live.append(incoming.iloc[i:i + 1])
        live.append(incoming.iloc[i:i + 1])
        expected = raw.iloc[:i + 1].cumulate(tf, cumulators={'v': agg})
        assert live.equals(expected)
        np.testing.assert_array_equal(live.index, expected.index)


@pytest.mark.parametrize('dtype,limit', [('int32', 2**31 - 1), ('int64', 2**63 - 1)])
def test_failed_correction_and_multibar_overflow_leave_both_checkpoints_unchanged(dtype, limit):
    stamps = pd.date_range('2024-01-01', periods=4, freq='min')
    fine = make([limit - 2, 1], dtype, stamps[:2])
    live = fine.cumulate('1d', cumulators={'v': 'sum'})
    before = live.copy()
    for values, index in [([3], stamps[1:2]), ([0, 4], stamps[2:4])]:
        with pytest.raises(OverflowError):
            live.append(make(values, dtype, index))
        assert live.equals(before)
    live.append(make([2], dtype, stamps[1:2]))
    assert live['v'].to_list() == [limit]
    live.append(make([0], dtype, stamps[1:2]))
    assert live['v'].to_list() == [limit - 2]


@pytest.mark.parametrize('unsupported', ['count', 'median', lambda values: values.sum()])
def test_cumulator_names_and_callable_rejection_do_not_expand_the_contract(unsupported):
    fine = make([1.0], 'float64', pd.date_range('2024-01-01', periods=1))
    with pytest.raises((ValueError, TypeError)):
        fine.cumulate('1d', cumulators={'v': unsupported})


@pytest.mark.parametrize('agg', AGGS)
def test_live_fold_and_resend_preserve_their_nan_bits(agg):
    stamps = np.arange(3).astype('datetime64[s]').astype('datetime64[ns]')
    nan = np.array([0x7ff8000000000042], dtype='uint64').view('float64')[0]
    live = DataFrame({'v': [1.]}, index=stamps[:1], time_frame='1d', cumulators={'v': agg})
    bar = DataFrame({'v': [nan]}, index=stamps[1:2])
    live.append(bar)
    assert live.to_numpy().view('uint64')[0, 0] == 0x7ff8000000000042
    live.append(bar)
    bits = 0x7ff8000000000000 if agg in ('first', 'last') else 0x7ff8000000000042
    assert live.to_numpy().view('uint64')[0, 0] == bits


@pytest.mark.parametrize('middle,tail,overflow', [(1, 2**31 - 1, False), (192, 2**31 - 1 - 224, True)])
def test_dtype_write_keeps_live_rounding_and_refold_error_timing(middle, tail, overflow):
    stamps = pd.date_range('2024-01-01', periods=4, freq='min')
    live = make([2**30, middle, -2**30], 'float32', stamps[:3]).cumulate('1d', cumulators={'v': 'sum'})
    live['v'] = np.asarray(live['v'].to_numpy(), dtype='int32')
    before = live['v'].to_list()[0]
    bar = make([tail], 'int32', stamps[3:])
    if overflow:
        with pytest.raises(OverflowError):
            live.append(bar)
        assert live['v'].to_list() == [before]
        live.append(make([0], 'int32', stamps[3:]))
        assert live['v'].to_list() == [before]
    else:
        live.append(bar)
        assert live['v'].to_list() == [tail]
        with pytest.raises(OverflowError):
            live.append(bar)
        assert live['v'].to_list() == [tail]
    live.append(make([0], 'int32', stamps[3:]))
    assert live['v'].to_list() == [middle]
