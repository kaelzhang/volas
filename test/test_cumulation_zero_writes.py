"""Forming writes compare present float bits while missing payloads stay opaque."""

import numpy as np
import pandas as pd
import pytest

from volas import DataFrame
from test.exact import assert_exact


AGGS = ['first', 'last', 'max', 'min', 'sum']


def fine(values, dtype, stamps):
    return DataFrame({'v': np.array(values, dtype=dtype)}, index=stamps)


def write(frame, values, dtype, entry):
    if entry == 'cell':
        frame.iloc[-1, 0] = np.array(values, dtype=dtype)[-1]
    else:
        frame['v'] = np.array(values, dtype=dtype)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('old', [-0.0, +0.0], ids=['negative-to-positive', 'positive-to-negative'])
@pytest.mark.parametrize('agg', AGGS)
@pytest.mark.parametrize('entry', ['cell', 'column'])
@pytest.mark.parametrize('seed', ['constructed', 'cumulated'])
@pytest.mark.parametrize('copy', [False, True], ids=['original', 'same-tf-copy'])
def test_single_fine_bar_zero_write_refreshes_raw_bar_and_checkpoint(dtype, old, agg, entry, seed, copy):
    stamps = pd.date_range('2024-01-01', periods=3, freq='min')
    spec = {'v': agg}
    raw = fine([old], dtype, stamps[:1])
    live = (DataFrame(raw, time_frame='5m', cumulators=spec) if seed == 'constructed'
            else raw.cumulate('5m', cumulators=spec))
    new = -old
    write(live, [new], dtype, entry)
    assert_exact(live['v'], np.array([new], dtype=dtype))
    if copy:
        live = live.cumulate('5m', cumulators=spec)
    # Same-sign zeros make first/min/max/sum depend on the rewritten raw bar.
    live.append(fine([new], dtype, stamps[1:2]))
    expected = fine([new, new], dtype, stamps[:2]).cumulate('5m', cumulators=spec)
    assert_exact(live['v'], expected['v'])
    # Re-sends restore before-last: they must never recover the pre-write sign.
    for tail in (1.0, new, new):
        live.append(fine([tail], dtype, stamps[1:2]))
        expected = fine([new, tail], dtype, stamps[:2]).cumulate('5m', cumulators=spec)
        assert_exact(live['v'], expected['v'])
    live.append(fine([new], dtype, stamps[2:]))
    expected = fine([new, new, new], dtype, stamps).cumulate('5m', cumulators=spec)
    assert_exact(live['v'], expected['v'])


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('old', [-0.0, +0.0], ids=['negative-to-positive', 'positive-to-negative'])
@pytest.mark.parametrize('agg', AGGS)
@pytest.mark.parametrize('entry', ['cell', 'column'])
def test_multibar_zero_rewrite_is_atomic_and_later_resends_keep_fine_history(dtype, old, agg, entry):
    stamps = pd.date_range('2024-01-01', periods=3, freq='min')
    spec = {'v': agg}
    live = fine([old, old], dtype, stamps[:2]).cumulate('5m', cumulators=spec)
    before = live.copy()
    for _ in range(2):
        with pytest.raises(ValueError, match='2 fine bars'):
            write(live, [-old], dtype, entry)
        assert_exact(live.to_numpy(), before.to_numpy())
        assert_exact(live.index, before.index)
        assert live.dtypes == before.dtypes
    for frame in (live, live.cumulate('5m', cumulators=spec)):
        for tail in (1.0, old, old):
            frame.append(fine([tail], dtype, stamps[1:2]))
            expected = fine([old, tail], dtype, stamps[:2]).cumulate('5m', cumulators=spec)
            assert_exact(frame['v'], expected['v'])
        frame.append(fine([old], dtype, stamps[2:]))
        expected = fine([old, old, old], dtype, stamps).cumulate('5m', cumulators=spec)
        assert_exact(frame['v'], expected['v'])


@pytest.mark.parametrize('dtype,word,payload', [
    ('float32', 'uint32', 0x7fc00042), ('float64', 'uint64', 0x7ff8000000000042),
])
@pytest.mark.parametrize('agg', AGGS)
@pytest.mark.parametrize('entry', ['cell', 'column'])
def test_missing_payload_rewrite_keeps_multibar_history(dtype, word, payload, agg, entry):
    stamps = pd.date_range('2024-01-01', periods=2, freq='min')
    spec = {'v': agg}
    live = fine([np.nan, 1.0], dtype, stamps).cumulate('5m', cumulators=spec)
    nan = np.array([payload], dtype=word).view(dtype)[0]
    write(live, [nan], dtype, entry)
    live.append(fine([2.0], dtype, stamps[1:]))
    expected = fine([np.nan, 2.0], dtype, stamps).cumulate('5m', cumulators=spec)
    assert_exact(live['v'], expected['v'])


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_public_equals_keeps_value_semantics_for_signed_zero(dtype):
    stamps = pd.date_range('2024-01-01', periods=1)
    assert fine([-0.0], dtype, stamps).equals(fine([+0.0], dtype, stamps))
