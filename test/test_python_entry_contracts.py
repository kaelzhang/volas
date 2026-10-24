"""Labels, projections and target representations survive Python entry points."""
import os
import subprocess
import sys

import numpy as np
import pytest

from volas import DataFrame, DirectiveValueError


def select(df, mask, entry, write=False):
    if entry == 'frame':
        if write:
            df[mask] = 0
        else:
            return df[mask]
    elif entry == 'series':
        s = df['x']
        if write:
            s[mask] = 0
        else:
            return s[mask]
    else:
        ix = getattr(df, entry)
        if write:
            ix[mask, 0 if entry == 'iloc' else 'x'] = 0
        else:
            return ix[mask]


@pytest.mark.parametrize('entry', ['frame', 'series', 'iloc', 'loc'])
@pytest.mark.parametrize('write', [False, True])
@pytest.mark.parametrize('window', [None, 2])
def test_boolean_series_key_requires_visible_label_alignment(entry, write, window):
    df = DataFrame({'x': [1, 2, 3]}, index=[10, 11, 12],
                   window=window, max_lookback=0 if window else None)
    mask = DataFrame({'m': [True] * len(df)}, index=list(range(len(df))))['m']
    before = df.copy()
    with pytest.raises(ValueError, match='different indexes'):
        select(df, mask, entry, write)
    assert df.equals(before)


@pytest.mark.parametrize('entry', ['frame', 'series', 'iloc', 'loc'])
@pytest.mark.parametrize('write', [False, True])
def test_masked_array_key_rejects_unknown_signal(entry, write):
    df = DataFrame({'x': [1, 2, 3]})
    mask = np.ma.array([True, False, True], mask=[False, True, False])
    before = df.copy()
    with pytest.raises(ValueError, match='unknown signal'):
        select(df, mask, entry, write)
    assert df.equals(before)


@pytest.mark.parametrize('series', [False, True])
def test_equals_rejects_different_datetime_awareness(series):
    a = DataFrame({'x': [1, 2]}, index=np.array([0, 1], dtype='datetime64[ns]'))
    b = a.tz_localize('UTC')
    if series:
        a, b = a['x'], b['x']
    with pytest.raises(ValueError, match='tz-naive.*tz-aware'):
        a.equals(b)


def test_series_equals_compares_labels_and_ignores_names():
    a = DataFrame({'x': [1, 2]}, index=[10, 11])['x']
    b = DataFrame({'y': [1, 2]}, index=[11, 12])['y']
    assert not a.equals(b)
    assert a.equals(DataFrame({'y': [1, 2]}, index=[10, 11])['y'])


@pytest.mark.parametrize('window', [None, 2])
@pytest.mark.parametrize('directive, canonical', [('close + 1', 'close+1'), ('HHV:2@close', 'hhv:2@close')])
def test_constructor_materializes_uncached_directive_through_projection(window, directive, canonical):
    a = DataFrame({'close': [1., 2., 3., 4.]}, window=window,
                  max_lookback=['ema:2'] if window else None)
    # A valid class-1 directive is not already declared or cached.
    names = ['close', directive]
    got = DataFrame(a, columns=names)
    assert got.equals(a[names])
    assert not got.is_computed(canonical)
    got.append({'close': 5.})
    assert np.isnan(got[canonical].to_numpy()[-1])


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'complex128', 'U20', 'bool'])
def test_na_fill_uses_explicit_target_representation(dtype):
    s = DataFrame({'x': [1, None]})['x']
    result = s.to_numpy(dtype=dtype, na_value=0.5)
    np.testing.assert_array_equal(result, np.array([1, 0.5], dtype=dtype))


@pytest.mark.parametrize('boundary', ['array', 'import', 'lookup', 'box'])
def test_failing_python_call_propagates_original_error(boundary):
    code = '''
import builtins
import numpy as np
from volas import DataFrame
s = DataFrame({'x': ['a', None] if BOUNDARY == 'array' else [1, 2]})['x']
def fail(*args, **kwargs):
    raise MemoryError('boundary sentinel')
if BOUNDARY == 'array':
    np.array = fail
elif BOUNDARY == 'box':
    np.int64 = fail
elif BOUNDARY == 'lookup':
    del np.float32
else:
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == 'numpy':
            fail()
        return original(name, *args, **kwargs)
    builtins.__import__ = importing
try:
    s.to_numpy() if BOUNDARY == 'array' else s.sum()
except (MemoryError, AttributeError) as exc:
    assert 'boundary sentinel' in str(exc) or BOUNDARY == 'lookup'
else:
    raise AssertionError('call did not propagate its failure')
'''.replace('BOUNDARY', repr(boundary))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('entry', ['iloc', 'loc'])
@pytest.mark.parametrize('write', [False, True])
@pytest.mark.parametrize('kind', ['series', 'masked'])
def test_column_axis_masks_keep_labels_and_missingness(entry, write, kind):
    df = DataFrame({'x': [1, 2], 'y': [3, 4]})
    if kind == 'series':
        mask = DataFrame({'m': [True, False]}, index=['y', 'x'])['m']
        message = 'different indexes'
    else:
        mask = np.ma.array([True, True], mask=[False, True])
        message = 'unknown signal'
    with pytest.raises(ValueError, match=message):
        if write:
            getattr(df, entry)[:, mask] = 0
        else:
            getattr(df, entry)[:, mask]


@pytest.mark.parametrize('entry', ['frame', 'series', 'iloc', 'loc'])
@pytest.mark.parametrize('stride', [1, -1, 2])
def test_known_masked_keys_keep_numpy_stride_order(entry, stride):
    bits = np.ma.array([True, False, True, False, True, False], mask=False)[::stride]
    df = DataFrame({'x': list(range(len(bits)))})
    got = select(df, bits, entry)
    got = got if entry == 'series' else got['x']
    assert got.to_list() == np.arange(len(bits))[np.asarray(bits)].tolist()


def test_explicit_integer_and_object_exports_stay_exact():
    s = DataFrame({'x': [2**53 + 1, None]})['x']
    assert s.to_numpy(dtype='int64', na_value=-1).tolist() == [2**53 + 1, -1]
    assert s.to_numpy(dtype='object', na_value=0.5).tolist() == [2**53 + 1, 0.5]


def test_constructor_invalid_projection_is_atomic():
    df = DataFrame({'close': [1., 2., 3.]})
    before = df.copy()
    with pytest.raises(DirectiveValueError):
        DataFrame(df, columns=['ema:2', 'ema:0'])
    assert df.equals(before)
    assert df.columns == ['close']


def test_equality_accepts_same_instants_in_different_zones_and_names():
    a = DataFrame({'x': [1, 2]}, index=np.array([0, 1], dtype='datetime64[ns]')).tz_localize('UTC')
    b = a.tz_convert('Asia/Shanghai')
    assert a.equals(b)
    assert a['x'].equals(b['x'])
    assert not DataFrame({'x': [1, 2]}, index=[10, 11]).equals(DataFrame({'x': [1, 2]}, index=[11, 12]))


@pytest.mark.parametrize('source', [[1., float('nan')], ['1', None]])
def test_na_fill_never_casts_an_unspecified_missing_payload(source):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        got = DataFrame({'x': source})['x'].to_numpy(dtype='int64', na_value=7)
    assert got.tolist() == [1, 7]


def test_exec_fills_in_target_dtype_too():
    np.testing.assert_array_equal(
        DataFrame({'x': [1, None]}).exec('x', dtype='float64', na_value=0.5),
        [1., 0.5],
    )



def test_float32_na_fill_widens_before_filling():
    s = DataFrame({'x': np.array([1, np.nan], dtype='float32')})['x']
    np.testing.assert_array_equal(s.to_numpy(dtype='float64', na_value=1e-100), [1., 1e-100])


@pytest.mark.parametrize('series', [False, True])
def test_unsigned_na_fill_uses_the_unsigned_target(series):
    df = DataFrame({'x': [2**53 + 1, None]})
    source = df['x'] if series else df
    got = source.to_numpy(dtype='uint64', na_value=2**63 + 1)
    assert got.ravel().tolist() == [2**53 + 1, 2**63 + 1]


@pytest.mark.parametrize('series', [False, True])
def test_datetime_na_fill_uses_target_units(series):
    df = DataFrame({'x': [1, None]})
    source = df['x'] if series else df
    got = source.to_numpy(dtype='datetime64[ns]', na_value=np.datetime64('2026-01-01', 'D'))
    np.testing.assert_array_equal(got.ravel(), np.array([1, np.datetime64('2026-01-01')], dtype='datetime64[ns]'))
