"""Tests for ``volas.to_datetime`` and the datetime ``astype`` units.

Both mirror pandas:

* ``to_datetime(x, unit=...)`` reads numeric epochs **preserving** sub-unit
  fractions, exactly like ``pandas.to_datetime(..., unit=...)``.
* ``df.astype({col: 'datetime64[s]'})`` **truncates** to the dtype's unit, like a
  NumPy / pandas ``astype`` cast.

Parity is checked against pandas directly (a dev-time oracle).
"""

import numpy as np
import pandas as pd
import pytest

from volas import DataFrame, Series, to_datetime


def ns(obj):
    """UTC epoch-nanoseconds (int64) of a volas Series / datetime64 ndarray."""
    arr = obj.to_numpy() if hasattr(obj, 'to_numpy') else np.asarray(obj)
    return arr.astype('datetime64[ns]').astype('int64')


def pd_ns(values, **kw):
    return pd.to_datetime(np.asarray(values), **kw).to_numpy().astype('datetime64[ns]').astype('int64')


# --- to_datetime: numeric epochs -------------------------------------------

@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_to_datetime_numeric_matches_pandas(unit):
    vals = [1609770600.0, 1609770660.0, 2.0, 0.0]
    got = to_datetime(vals, unit=unit)
    assert got.dtype == 'datetime64[ns]'
    np.testing.assert_array_equal(ns(got), pd_ns(vals, unit=unit))


def test_to_datetime_preserves_fraction():
    # 0.5 s == 500_000_000 ns, kept (not truncated) — matches pd.to_datetime
    got = to_datetime([1609770660.5], unit='s')
    assert ns(got)[0] == 1609770660_500_000_000
    assert ns(got)[0] == pd_ns([1609770660.5], unit='s')[0]


def test_to_datetime_int_epoch_matches_pandas():
    vals = [1609770600, 1609770660]
    np.testing.assert_array_equal(ns(to_datetime(vals, unit='s')), pd_ns(vals, unit='s'))


def test_to_datetime_default_unit_is_ns():
    vals = [1609770600_000_000_000, 0]
    np.testing.assert_array_equal(ns(to_datetime(vals)), pd_ns(vals, unit='ns'))


# --- to_datetime: strings ---------------------------------------------------

def test_to_datetime_strings_utc_matches_pandas():
    strs = ['2021-01-04 09:30:00', '2021-01-04 10:00:00']
    np.testing.assert_array_equal(ns(to_datetime(strs)), pd_ns(strs))


def test_to_datetime_offset_aware_string_is_absolute():
    got = to_datetime(['2021-01-04T09:30:00+08:00'])
    assert ns(got)[0] == pd.Timestamp('2021-01-04 01:30:00').value


# --- to_datetime: input kinds, name / index, idempotence -------------------

def test_to_datetime_series_preserves_name_and_index():
    df = DataFrame({'t': [1609770600.0, 1609770660.0]})
    s = to_datetime(df['t'], unit='s')
    assert isinstance(s, Series)
    assert s.name == 't'
    np.testing.assert_array_equal(s.index, df['t'].index)


def test_to_datetime_numpy_array_input():
    arr = np.array([1609770600.0, 1609770660.0])
    got = to_datetime(arr, unit='s')
    assert got.name is None
    np.testing.assert_array_equal(ns(got), pd_ns(arr, unit='s'))


def test_to_datetime_already_datetime_is_idempotent():
    base = to_datetime([1609770600.0], unit='s')
    np.testing.assert_array_equal(ns(to_datetime(base)), ns(base))


# --- to_datetime: errors ----------------------------------------------------

def test_to_datetime_bad_unit_raises():
    with pytest.raises(ValueError):
        to_datetime([1.0], unit='weeks')


def test_to_datetime_unparseable_string_raises():
    with pytest.raises(ValueError):
        to_datetime(['not-a-date'])


def test_to_datetime_bad_input_type_raises():
    with pytest.raises(TypeError):
        to_datetime(object())


def test_to_datetime_bool_input_raises():
    with pytest.raises((TypeError, ValueError)):
        to_datetime([True, False], unit='s')


# --- astype: datetime units (truncating) -----------------------------------

@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_astype_datetime_unit_matches_pandas(unit):
    vals = [1609770600.0, 1609770660.5]
    out = DataFrame({'t': vals, 'close': [1.0, 2.0]}).astype({'t': f'datetime64[{unit}]'})
    assert out['t'].dtype == 'datetime64[ns]'
    pser = pd.DataFrame({'t': vals}).astype({'t': f'datetime64[{unit}]'})['t']
    np.testing.assert_array_equal(ns(out['t']), ns(pser.to_numpy()))
    # other columns are untouched
    np.testing.assert_array_equal(out['close'].to_numpy(), np.array([1.0, 2.0]))


def test_astype_datetime_truncates_fraction():
    out = DataFrame({'t': [1609770660.5]}).astype({'t': 'datetime64[s]'})
    assert ns(out['t'])[0] == 1609770660_000_000_000  # the 0.5 s is dropped


@pytest.mark.parametrize('dt', ['datetime', 'datetime64', 'datetime64[ns]'])
def test_astype_bare_datetime_is_ns(dt):
    out = DataFrame({'t': [1609770600]}).astype({'t': dt})  # int read as ns
    assert ns(out['t'])[0] == 1609770600


def test_astype_datetime_on_string_column():
    strs = ['2021-01-04 09:30:00', '2021-01-04 10:00:00']
    out = DataFrame({'t': strs}).astype({'t': 'datetime64[s]'})  # unit ignored; parsed
    assert out['t'].dtype == 'datetime64[ns]'
    np.testing.assert_array_equal(ns(out['t']), pd_ns(strs))


def test_astype_datetime_idempotent():
    df = DataFrame({'t': [1609770600.0]}).astype({'t': 'datetime64[s]'})
    out = df.astype({'t': 'datetime'})  # already datetime -> kept
    np.testing.assert_array_equal(ns(out['t']), ns(df['t']))


def test_astype_mixed_datetime_and_numeric():
    out = DataFrame({'t': [1609770600.0], 'n': [1.5]}).astype({'t': 'datetime64[s]', 'n': 'int'})
    assert out['t'].dtype == 'datetime64[ns]'
    assert out['n'].dtype == 'int64'
    assert out['n'].to_numpy()[0] == 1


def test_astype_non_datetime_only_still_works():
    out = DataFrame({'a': [1.0, 2.0]}).astype({'a': 'int'})
    assert out['a'].dtype == 'int64'


def test_astype_datetime_on_bool_raises():
    with pytest.raises((TypeError, ValueError)):
        DataFrame({'b': [True, False]}).astype({'b': 'datetime64[s]'})


def test_astype_unknown_dtype_raises():
    with pytest.raises(ValueError):
        DataFrame({'a': [1.0]}).astype({'a': 'complex128'})
