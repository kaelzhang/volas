"""Contract C4 + Q2 — explicit `astype` from str to a numeric dtype PARSES each
cell (astype is an explicit cast, not an implicit op): a valid numeric string
converts, an empty / missing cell becomes NA, and any other non-empty string
raises. It must never silently funnel a str column to all-NaN (the old
to_f64_vec path), which destroys data."""

import numpy as np
import pytest
import volas
from volas import DataFrame


def _s(data, dtype=None):
    return DataFrame({'a': np.array(data, dtype=dtype) if dtype is not None else data})['a']


# --- str -> float: parse -----------------------------------------------------

def test_str_to_float_parses():
    out = _s(['1.5', '2.5', '-3.0']).astype('float64')
    assert out.dtype == 'float64'
    assert list(out.to_numpy()) == [1.5, 2.5, -3.0]


def test_str_to_float_invalid_raises():
    with pytest.raises(Exception):
        _s(['1.5', 'abc']).astype('float64')


def test_str_to_float_na_and_empty_become_na():
    out = _s(['1.5', None, '']).astype('float64')
    got = out.to_numpy()
    assert got[0] == 1.5
    assert np.isnan(got[1]) and np.isnan(got[2])  # NA and "" -> NaN (missing)


# --- str -> int: parse, reject non-integral ----------------------------------

def test_str_to_int_parses():
    out = _s(['1', '2', '-3']).astype('int64')
    assert out.dtype == 'int64'
    assert list(out.to_numpy()) == [1, 2, -3]


def test_str_to_int_non_integral_raises():
    # "1.5" is not an int literal -> raise (pandas astype(int) does too)
    with pytest.raises(Exception):
        _s(['1', '1.5']).astype('int64')


def test_str_to_int_invalid_raises():
    with pytest.raises(Exception):
        _s(['1', 'abc']).astype('int64')


def test_str_to_int_na_preserved():
    out = _s(['1', None, '2']).astype('int64')
    assert out.dtype == 'int64'
    assert out[0] == 1 and out[1] is volas.NA and out[2] == 2


# --- round-trip float -> str -> float (review's 3rd repro) -------------------

def test_float_str_float_roundtrip():
    s = _s([1.5, 2.0, -3.25])
    back = s.astype('str').astype('float64')
    assert list(back.to_numpy()) == [1.5, 2.0, -3.25]
