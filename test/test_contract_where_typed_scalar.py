"""Contract consistency — where/mask must accept a same-dtype scalar fill (str
into str, datetime string into datetime, bool into bool), like the assignment
path already does. An incompatible scalar stays a TypeError."""

import numpy as np
import pytest
import volas
from volas import DataFrame

NA = volas.NA


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


def _mask(bits):
    return DataFrame({'a': bits})['a']


def test_str_where_str_scalar():
    s = _s(['a', 'b', 'c'])
    out = s.where(_mask([True, False, True]), 'z')   # keep where True, 'z' where False
    assert out.dtype == 'str' and out.to_list() == ['a', 'z', 'c']


def test_datetime_where_string_scalar():
    s = _s(['2024-01-01', '2024-01-02'], 'datetime64[ns]')
    out = s.where(_mask([True, False]), '2021-01-03')
    assert out.dtype == 'datetime64[ns]'
    assert out.to_list()[1] == np.datetime64('2021-01-03')


def test_bool_where_bool_scalar():
    s = _s([True, True, True])
    out = s.where(_mask([True, False, True]), False)
    assert out.dtype == 'bool' and out.to_list() == [True, False, True]


def test_str_where_incompatible_scalar_raises():
    with pytest.raises(Exception):
        _s(['a', 'b']).where(_mask([True, False]), 5)   # a number into a str column
