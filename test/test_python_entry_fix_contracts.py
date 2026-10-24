"""Construction commits and inferred NumPy text widths include the whole request."""
import numpy as np
import pytest

from volas import DataFrame


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('dtype', ['U', 'str', 'S', 'U0', 'S0'])
@pytest.mark.parametrize('values, fill', [([1, None], 0.5), (['a', None], 'missing'),
                                         ([None, 'longer valid'], 'NA'), ([None, None], 'missing')])
def test_unsized_text_export_sizes_valid_values_and_fill(entry, dtype, values, fill):
    df = DataFrame({'x': values})
    if entry == 'frame':
        got = df.to_numpy(dtype=dtype, na_value=fill)
    elif entry == 'series':
        got = df['x'].to_numpy(dtype=dtype, na_value=fill)
    else:
        got = df.exec('x', dtype=dtype, na_value=fill)
    expected = np.array([str(fill) if v is None else str(v) for v in values], dtype=dtype)
    source = df['x'].dtype
    resolved = np.empty(0, dtype='object' if source == 'str' else source).astype(dtype).dtype
    expected = expected.astype(np.result_type(resolved, expected.dtype))
    if entry == 'frame':
        expected = expected.reshape(-1, 1)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('dtype', ['U2', 'S2'])
def test_fixed_width_text_export_keeps_requested_width(entry, dtype):
    df = DataFrame({'x': ['abcd', None]})
    got = (df.to_numpy(dtype=dtype, na_value='missing') if entry == 'frame' else
           df['x'].to_numpy(dtype=dtype, na_value='missing') if entry == 'series' else
           df.exec('x', dtype=dtype, na_value='missing'))
    expected = np.array(['abcd', 'missing'], dtype=dtype)
    np.testing.assert_array_equal(got.reshape(-1), expected)
    assert got.dtype == expected.dtype


@pytest.mark.parametrize('source', ['plain', 'window', 'tf'])
@pytest.mark.parametrize('overrides, message', [
    ({'dtype': 'invalid'}, 'dtype'),
    ({'index': [10]}, 'length'),
    ({'index': [10, 10, 12]}, 'duplicate'),
    ({'time_frame': 'invalid'}, 'time.frame|TimeFrame'),
    ({'time_frame': '1m', 'cumulators': {'close': 'invalid'}}, 'agg|cumulat'),
    ({'cumulators': {'close': 'last'}}, 'requires time_frame'),
    ({'window': 2, 'max_lookback': ['missing + 1']}, 'missing'),
])
def test_constructor_commits_source_caches_only_after_all_steps(source, overrides, message):
    index = np.array([0, 60, 120], dtype='datetime64[s]').astype('datetime64[ns]')
    df = DataFrame({'close': [1., 2., 3.]}, index=index,
                   window=3 if source == 'window' else None,
                   max_lookback=2 if source == 'window' else None,
                   time_frame='1m' if source == 'tf' else None)
    before = df.copy()
    with pytest.raises((ValueError, TypeError), match=message):
        DataFrame(df, columns=['close', 'hhv:2@close'], **overrides)
    assert df.columns == before.columns
    assert df.equals(before)
    np.testing.assert_array_equal(df.index, before.index)
    assert df.dtypes == before.dtypes
    assert df._physical_height == before._physical_height
    assert df.ready == before.ready
    # Compare continued behavior too: a tf source folds into the same open bar,
    # while plain/windowed sources append a new row.
    bar = DataFrame({'close': [4.]}, index=np.array([150], dtype='datetime64[s]').astype('datetime64[ns]'))
    df.append(bar)
    before.append(bar)
    assert df.equals(before)
    np.testing.assert_array_equal(df.index, before.index)


def test_constructor_publishes_caches_on_success_and_continues_source():
    df = DataFrame({'close': [1., 2., 3.]})
    got = DataFrame(df, columns=['ema:2'], dtype='float32', index=[10, 11, 12],
                    window=2, max_lookback=0)
    assert df.columns == ['close', 'ema:2']
    assert df.is_computed('ema:2')
    assert not got.is_computed('ema:2')
    assert got.dtypes == {'ema:2': 'float32'}
    assert got.index.tolist() == [11, 12]
    df.append({'close': 4.})
    expected = DataFrame({'close': [1., 2., 3., 4.]})['ema:2']
    assert df['ema:2'].equals(expected)


@pytest.mark.parametrize('finding', ['alignment', 'unknown'])
def test_series_atomicity_regressions_detect_target_corruption(monkeypatch, finding):
    from test import test_python_entry_contracts as cases

    def corrupt(target, mask, entry, write=False):
        series = target['x'] if isinstance(target, DataFrame) else target
        series[0] = 99
        raise ValueError('different indexes' if finding == 'alignment' else 'unknown signal')

    monkeypatch.setattr(cases, 'select', corrupt)
    with pytest.raises(AssertionError):
        if finding == 'alignment':
            cases.test_boolean_series_key_requires_visible_label_alignment('series', True, None)
        else:
            cases.test_masked_array_key_rejects_unknown_signal('series', True)


def test_failed_constructor_preserves_source_staleness_as_well_as_schema():
    df = DataFrame({'close': [1., 2., 3.]})
    df['ema:2']
    df.append({'close': 4.})
    before = df.copy()
    with pytest.raises(ValueError, match='requires time_frame'):
        DataFrame(df, columns=['close + 1', 'ema:2'], cumulators={})
    assert df.columns == before.columns
    for frame in (df, before):
        with pytest.raises(ValueError, match='stale'):
            frame.to_numpy()
        frame.fulfill()
    assert df.equals(before)
    df.append({'close': 5.})
    before.append({'close': 5.})
    df.fulfill()
    before.fulfill()
    assert df.equals(before)
