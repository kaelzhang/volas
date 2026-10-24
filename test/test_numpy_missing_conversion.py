"""Only valid cells participate in explicit target conversion before NA filling."""
import warnings

import numpy as np
import pytest

from volas import DataFrame


def export(frame, entry, dtype, fill):
    if entry == 'frame':
        return frame.to_numpy(dtype=dtype, na_value=fill).reshape(-1)
    if entry == 'series':
        return frame['x'].to_numpy(dtype=dtype, na_value=fill)
    return frame.exec('x', dtype=dtype, na_value=fill)


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('error_mode', ['raise', 'warn'])
@pytest.mark.parametrize('values,dtype,fill,expected', [
    (np.ma.array([1, 2**63 - 1], dtype='int64', mask=[0, 1]), 'float16', 0, [1, 0]),
    (np.ma.array([1, -(2**63)], dtype='int64', mask=[0, 1]), 'float16', .5, [1, .5]),
    (np.ma.array([1, 2**31 - 1], dtype='int32', mask=[0, 1]), 'float16', 0, [1, 0]),
    (np.array([1, 'NaT'], dtype='datetime64[ns]'), 'float16', 0, [1, 0]),
    (np.ma.array([1, np.inf, -np.inf, np.nan], dtype='float64', mask=[0, 1, 1, 1]), 'float16', .5, [1, .5, .5, .5]),
    (np.ma.array([1, np.inf, -np.inf, np.nan], dtype='float32', mask=[0, 1, 1, 1]), 'int64', 0, [1, 0, 0, 0]),
    (np.ma.array(['a', 'é'], mask=[0, 1]), 'S2', b'NA', [b'a', b'NA']),
    (np.ma.array(['a', 'é'], mask=[0, 1]), 'S', b'NA', [b'a', b'NA']),
    (np.ma.array([2**53 + 1, 2**63 - 1], dtype='int64', mask=[0, 1]),
     'int64', -1, [2**53 + 1, -1]),
    (np.ma.array([2**53 + 1, 2**63 - 1], dtype='int64', mask=[0, 1]),
     'uint64', 2**64 - 1, [2**53 + 1, 2**64 - 1]),
    (np.ma.array([2**53 + 1, 2**63 - 1], dtype='int64', mask=[0, 1]),
     'U', 'missing', [str(2**53 + 1), 'missing']),
])
def test_missing_payload_never_casts(entry, error_mode, values, dtype, fill, expected):
    frame = DataFrame({'x': values})
    with warnings.catch_warnings(), np.errstate(all=error_mode):
        warnings.simplefilter('error')
        got = export(frame, entry, dtype, fill)
    expected = np.array(expected, dtype=dtype)
    if dtype in ('S', 'U'):
        source = frame['x'].dtype
        resolved = np.empty(0, dtype='object' if source == 'str' else source).astype(dtype).dtype
        expected = expected.astype(np.result_type(resolved, expected.dtype))
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('values', [
    [None, None],
    np.ma.array([2**63 - 1, -(2**63)], dtype='int64', mask=True),
    np.array(['NaT', 'NaT'], dtype='datetime64[ns]'),
    np.ma.array([np.inf, np.nan], dtype='float32', mask=True),
    np.ma.array(['é', 'invalid'], mask=True),
])
@pytest.mark.parametrize('dtype,fill', [('V2', b'ab'), ('S2', b'ab'), ('U2', 'ab'),
                                        ('S', b'long fill'), ('U', 'long fill'),
                                        ('float16', .5), ('int64', 2**53 + 1)])
def test_all_missing_converts_only_fill(entry, values, dtype, fill):
    frame = DataFrame({'x': values})
    # String -> numeric is a deliberate Frame contract, even with all NA.
    if entry == 'frame' and isinstance(values, np.ma.MaskedArray) and values.dtype.kind == 'U' and dtype in ('float16', 'int64'):
        with pytest.raises(ValueError, match='string column'):
            export(frame, entry, dtype, fill)
        return
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        got = export(frame, entry, dtype, fill)
    expected = np.array([fill, fill], dtype=dtype)
    if dtype in ('S', 'U'):
        source = frame['x'].dtype
        resolved = np.empty(0, dtype='object' if source == 'str' else source).astype(dtype).dtype
        expected = expected.astype(np.result_type(resolved, expected.dtype))
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
def test_valid_overflow_and_invalid_fill_still_raise(entry):
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError, match='overflow'):
            export(DataFrame({'x': [2**63 - 1, None]}), entry, 'float16', 0)
        with pytest.raises((ValueError, TypeError)):
            export(DataFrame({'x': [1, None]}), entry, 'int64', 'invalid')


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int64', 'uint64', 'S', 'U'])
def test_arbitrary_integer_missing_bits_never_affect_export(entry, dtype):
    rng = np.random.default_rng(47)
    values = rng.integers(-(2**63), 2**63 - 1, 64, dtype=np.int64)
    values[::4] = 1
    mask = np.arange(64) % 4 != 0
    frame = DataFrame({'x': np.ma.array(values, mask=mask)})
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        got = export(frame, entry, dtype, 0)
    expected = np.array(np.where(mask, 0, 1), dtype=dtype)
    np.testing.assert_array_equal(got, expected)
