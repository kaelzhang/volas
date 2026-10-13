"""The sign of a zero in a directive is one rule, whatever the spelling.

An integer zero has no sign: the literal ``-0`` and the negation ``-(0)`` are both
``0``. A float zero written with its decimal point keeps its sign (``-0.0``), and unary
minus flips the sign of a float zero. The canonical name of every form re-parses to the
same value, bit for bit, so a cached column refreshed from its name never changes sign.
"""

import itertools

import numpy as np
import pytest

import volas

from test.exact import same_exact

FRAME = volas.DataFrame({'close': [1.0, -2.0, 0.0], 'n': np.array([3, 0, -4], dtype=np.int64)})


def _same_bits(a, b):
    return same_exact(a, b)


def _values(directive):
    return FRAME.exec(directive, dtype='float64')


@pytest.mark.parametrize('directive, want', [
    ('close/-0', np.inf),        # -0 is the integer zero: no sign
    ('close/-(0)', np.inf),      # negating an integer zero is still zero
    ('close/-0.0', -np.inf),     # a float zero literal keeps its sign
    ('close/-(0.5-0.5)', -np.inf),  # unary minus flips a float zero
    ('close/-(-0.0)', np.inf),
])
def test_one_rule_for_the_sign_of_zero(directive, want):
    assert FRAME.exec(directive, dtype='float64')[0] == want


@pytest.mark.parametrize('directive', ['-0', '-(0)', '-n', '-(-n)'])
def test_an_integer_zero_negates_to_positive_zero(directive):
    values = np.asarray(FRAME.exec(directive, dtype='float64'), dtype=float)
    assert not np.signbit(values[values == 0]).any()


LEAVES = ['0', '-0', '0.0', '-0.0', '-(0)', '-(0.0)', '-(-0.0)', '2', '-2.5', '-(2)', 'n', '-n', 'close']


@pytest.mark.parametrize('a, op, b', itertools.product(LEAVES, ['/', '*', '-'], LEAVES[:7]))
def test_the_canonical_name_evaluates_to_the_same_bits(a, op, b):
    directive = f'{a}{op}({b})'
    name = volas.directive_stringify(directive)
    assert _same_bits(_values(name), _values(directive)), (directive, name)
    assert volas.directive_stringify(name) == name


@pytest.mark.parametrize('directive', ['dev:2/-0', 'emv:1/-0', 'close/-0.0', 'close*-(0)'])
def test_a_cached_column_keeps_its_sign_across_a_refresh(directive):
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.standard_normal(12))
    data = {'open': c, 'high': c + 1, 'low': c - 1, 'close': c, 'volume': rng.uniform(1, 9, 12)}
    df = volas.DataFrame({k: v[:10] for k, v in data.items()})
    df[directive]
    df.append({k: v[10] for k, v in data.items()})
    full = volas.DataFrame({k: v[:11] for k, v in data.items()}).exec(directive, dtype='float64')
    assert _same_bits(df[directive], full)
