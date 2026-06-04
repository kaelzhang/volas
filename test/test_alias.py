"""volas column/directive alias tests.

Ported / adapted from stock-pandas's ``test_basic.py::test_aliases``. ``alias``
maps a new name to an existing column (keeping both), resolves everywhere a
column is looked up (including inside directives), and propagates to derived
frames.
"""

import numpy as np
import pytest

from volas import DataFrame

SIMPLE = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def make():
    return DataFrame({'open': SIMPLE, 'close': [x + 1 for x in SIMPLE]})


def test_alias_resolves():
    df = make()
    df.alias('Open', 'open')
    np.testing.assert_array_equal(df['Open'].to_numpy(), SIMPLE)


def test_alias_resolves_inside_directive():
    df = make()
    df.alias('Open', 'open')
    np.testing.assert_allclose(
        df['ma:2@Open'].to_numpy(), df['ma:2@open'].to_numpy(), equal_nan=True
    )


def test_alias_survives_drop():
    df = make()
    df.alias('Open', 'open')
    dropped = df.drop([0])
    np.testing.assert_array_equal(dropped['Open'].to_numpy(), SIMPLE[1:])


def test_alias_survives_copy_and_slice():
    df = make()
    df.alias('Open', 'open')
    assert np.array_equal(df.copy()['Open'].to_numpy(), SIMPLE)
    assert np.array_equal(df.iloc[1:]['Open'].to_numpy(), SIMPLE[1:])


def test_alias_already_exists_raises():
    df = make()
    with pytest.raises(ValueError, match='already exists'):
        df.alias('open', 'close')


def test_alias_src_not_exists_raises():
    df = make()
    with pytest.raises(ValueError, match='not exists'):
        df.alias('some_column', 'not-exists')
