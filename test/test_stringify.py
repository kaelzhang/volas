"""volas directive_stringify tests.

Ported / adapted from stock-pandas's ``test_parser.py::test_stringify`` and
``test_basic.py::test_directive_stringify``. The canonical form is volas-native:
whitespace removed, minimal parens by operator priority, default args/series
dropped. (Unlike stock-pandas it does not coerce numeric args to a typed display,
so `100` stays `100` rather than `100.0`.)
"""

import pytest

from volas import directive_stringify as stringify


@pytest.mark.parametrize('directive,expected', [
    # default args / series dropped (test_basic)
    ('boll', 'boll'),
    ('boll:20@close', 'boll'),
    ('boll:30@close', 'boll:30'),
    ('macd:12,26', 'macd'),
    ('ma:5@close', 'ma:5'),
    # operator priority + minimal parens
    ('close + open * high', 'close+open*high'),
    ('3 * (high - low)', '3*(high-low)'),
    ('(kdj.j > 100) | (kdj.j <= 100)', 'kdj.j>100|kdj.j<=100'),
    ('kdj.j > 100 | kdj.j <= 100', 'kdj.j>100|kdj.j<=100'),
    ('~ ( kdj.j < 0 )', '~(kdj.j<0)'),
    ('- a * ((b + c) > 0)', '-a*(b+c>0)'),
    ('boll > -1', 'boll>-1'),
    # unary / special operator spacing
    ('close + - close', 'close+-close'),
    ('a +- b', 'a+-b'),
    ('a + - b', 'a+-b'),
    ('a +~ b', 'a+~b'),
    # sub-command canonicalization
    ('macd.dif', 'macd'),
    ('macd.s', 'macd.signal'),
    ('boll.u:20', 'boll.upper'),
    ('(kdj.j)&(kdj.d)', 'kdj.j&kdj.d'),
    # default argument / series slots kept as empty placeholders
    ('boll:30,', 'boll:30'),
    ('kdj.j:,4', 'kdj.j:,4'),
    ('kdj.j:@,high', 'kdj.j@,high'),
])
def test_stringify(directive, expected):
    assert stringify(directive) == expected
