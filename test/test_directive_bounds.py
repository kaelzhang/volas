"""Bounded directive input: the parser rejects hostile input with a
``DirectiveSyntaxError`` instead of crashing the process or caching it forever."""

import numpy as np
import pytest

import volas
from volas import DirectiveSyntaxError

DEEP = {
    'paren': '(' * 20_000 + 'close' + ')' * 20_000,
    'not': '~' * 30_000 + 'close',
    'chain': '+'.join(['close'] * 30_000),
    'series': 'ema:2@(' * 5_000 + 'close' + ')' * 5_000,
}


@pytest.fixture
def df():
    return volas.DataFrame({'close': [1.0, 2.0, 3.0]})


@pytest.mark.parametrize('kind', list(DEEP))
def test_deep_directive_is_a_syntax_error_not_a_crash(df, kind):
    directive = DEEP[kind]
    with pytest.raises(DirectiveSyntaxError, match='nests deeper than 128 levels'):
        volas.directive_stringify(directive)
    with pytest.raises(DirectiveSyntaxError, match='nests deeper than 128 levels'):
        df.exec(directive)
    with pytest.raises(DirectiveSyntaxError, match='nests deeper than 128 levels'):
        volas.directive_lookback(directive)


def test_directive_at_the_nesting_bound_still_runs(df):
    directive = '(' * 127 + 'close' + ')' * 127
    np.testing.assert_array_equal(np.asarray(df.exec(directive)), [1.0, 2.0, 3.0])


BIG = '1' + '0' * 400  # rounds to inf as a 64-bit float


@pytest.mark.parametrize(
    'directive', [f'close < {BIG}', f'-{BIG}', f'ma:2@({BIG})'], ids=['compare', 'negated', 'series']
)
def test_literal_overflowing_f64_is_a_syntax_error(df, directive):
    with pytest.raises(DirectiveSyntaxError, match='out of range'):
        volas.directive_stringify(directive)
    with pytest.raises(DirectiveSyntaxError, match='out of range'):
        df[directive]


def test_overflowing_literal_never_rebinds_to_a_column_named_inf():
    # It used to cache as `close<inf`, whose refresh re-parsed `inf` as a Name.
    frame = volas.DataFrame({'close': [1.0, 2.0], 'inf': [0.0, 0.0]})
    with pytest.raises(DirectiveSyntaxError):
        frame[f'close < {BIG}']
    assert list(frame.columns) == ['close', 'inf']
