"""volas typed directive errors.

Adapted (lean) from stock-pandas's ``test_parse.py`` / ``test_parser.py``. volas
raises ``DirectiveSyntaxError`` for parse failures (annotated with line/column)
and ``DirectiveValueError`` for an unknown command / sub-command, too many
arguments, or a bad argument value. Both subclass ``ValueError`` (so ``except
ValueError`` keeps working) and ``DirectiveError``. Messages are volas-native.
"""

import pytest

import volas
from volas import DirectiveError, DirectiveSyntaxError, DirectiveValueError


@pytest.fixture
def df():
    return volas.read_csv(str(__import__('pathlib').Path(__file__).parent / 'data' / 'tencent.csv'))


# --- syntax errors (with line/column) ---------------------------------------

@pytest.mark.parametrize('directive', [
    'a >',          # trailing operator -> unexpected end
    '>',            # leading operator
    'ma:5 > 0)',    # extra close paren
    'ma@(abc',      # unclosed group
])
def test_syntax_error(df, directive):
    with pytest.raises(DirectiveSyntaxError) as ei:
        df[directive]
    assert 'line' in str(ei.value) and 'column' in str(ei.value)
    assert isinstance(ei.value, (DirectiveError, ValueError))


# --- value errors -----------------------------------------------------------

@pytest.mark.parametrize('directive,fragment', [
    ('invalid_cmd:5', 'unknown command'),
    ('macd.unknown', 'sub-command'),       # known command, bad sub
    ('kdj', 'requires a sub-command'),     # sub required
    ('ema:2,3', 'at most'),                # too many arguments (ema takes one period)
    ('ema:2,3@close', 'at most'),
    ('style:cartoon', 'bullish'),          # bad style value
])
def test_value_error(df, directive, fragment):
    with pytest.raises(DirectiveValueError) as ei:
        df[directive]
    assert fragment in str(ei.value)
    assert isinstance(ei.value, (DirectiveError, ValueError))


def test_directive_errors_are_value_errors(df):
    # backward-compatible: `except ValueError` still catches directive errors
    with pytest.raises(ValueError):
        df['invalid_cmd:5']
    with pytest.raises(ValueError):
        df['a >']


def test_exec_also_raises_typed(df):
    with pytest.raises(DirectiveValueError):
        df.exec('invalid_cmd:5')
    with pytest.raises(DirectiveSyntaxError):
        df.exec('a >')
