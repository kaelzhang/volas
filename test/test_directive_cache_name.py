"""A cached directive's column name is ``directive_stringify`` of it — one renderer —
and it keeps a command apart from a bare name.

``directive_stringify(d)`` is documented as "the actual column name volas caches it
under": every argument is written as its bound value (``ma:05`` caches as ``ma:5``), and
an explicit command whose arguments are all defaults keeps its colon (``boll:20`` caches
as ``boll:``), so its name never re-reads as a bare name — which a data column of that
name would take. A cached column refreshes from the directive it was bound to, never
from a re-read of its name.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

import volas
from test.harness.inventory import executable_lines, sample_value

from test.exact import same_exact

N = 60


def _data(n=N, seed=5):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.standard_normal(n))
    o = c + 0.4 * rng.standard_normal(n)
    return {'open': o, 'high': np.maximum(o, c) + rng.uniform(0.1, 1.5, n),
            'low': np.minimum(o, c) - rng.uniform(0.1, 1.5, n), 'close': c,
            'volume': rng.uniform(1, 100, n)}


DATA = _data()


def _frame(rows=N, **extra):
    return volas.DataFrame({**{k: v[:rows] for k, v in DATA.items()}, **{k: v[:rows] for k, v in extra.items()}})


def _same(a, b):
    return same_exact(a, b)


def _spellings(line):
    """Spellings of one line: its default form, its aliases and an upper-case head, and
    argument tuples written with leading zeros, a `+` sign and float forms."""
    rng = np.random.default_rng(zlib.crc32(line.head.encode()))
    heads = [line.head, *line.aliases, line.head.upper()]
    out = [line.form + line.series_suffix]
    for k in range(3):
        toks = []
        for a in line.args:
            if a.kind == 'int':
                v = int(rng.integers(int(a.lo), int(min(a.hi, a.lo + 9)) + 1))
                toks.append(['0' + str(v), '+' + str(v), str(v)][k] if a.lo >= 0 else str(v))
            else:
                v = sample_value(a, k)
                toks.append(f'{float(v):.3f}' if a.kind == 'float' and k == 1 else str(v))
        head = heads[k % len(heads)]
        out.append(f"{head}:{','.join(toks)}{line.series_suffix}" if toks else head + line.series_suffix)
    return list(dict.fromkeys(out))


CASES = [pytest.param(d, id=d) for line in executable_lines() for d in _spellings(line)]


@pytest.mark.parametrize('directive', CASES)
def test_the_cached_column_is_named_by_directive_stringify(directive):
    try:
        name = volas.directive_stringify(directive)
    except Exception:
        pytest.skip('not a valid spelling of this line')
    df = _frame()
    before = set(df.columns)
    try:
        values = df[directive]
    except Exception:
        pytest.skip('does not execute on this frame')
    added = set(df.columns) - before
    assert added == {name}, (directive, added, name)
    assert volas.directive_stringify(name) == name
    # the name reads back the cached column, and re-binds to the same directive
    assert _same(df[name], values)
    assert _same(_frame().exec(name, dtype='float64'), _frame().exec(directive, dtype='float64'))


def test_an_all_default_command_and_its_bare_name_are_one_column():
    assert volas.directive_stringify('boll:20@close') == 'boll'
    assert volas.directive_stringify('MACD:12,26') == 'macd'
    assert volas.directive_stringify('macd:') == 'macd'
    assert volas.directive_stringify('ma:05') == 'ma:5'
    assert volas.directive_stringify('hv:10,1W') == 'hv:10,1w'
    assert volas.directive_stringify('donchian.upper') == 'donchian.upper'  # a form, period absent
    df = _frame()
    for d in ('boll:20', 'boll', 'BOLL:', 'boll:20@close'):
        df[d]
    assert [c for c in df.columns if c.startswith('boll')] == ['boll']


@pytest.mark.parametrize('line', executable_lines(), ids=lambda line: line.head)
def test_every_default_spelling_of_a_line_is_one_column(line):
    """A line's default spellings — head and aliases, upper / title case, with and
    without an empty argument list, and its default form — cache as one column."""
    heads = [line.head, *line.aliases, line.head.upper(), line.head.title()]
    spellings = {h + suffix + line.series_suffix for h in heads for suffix in ('', ':')}
    spellings.add(line.form + line.series_suffix)
    df = _frame()
    names = set()
    for d in sorted(spellings):
        try:
            name = volas.directive_stringify(d)
            df[d]
        except Exception:
            continue  # not a spelling that executes (a required argument absent)
        names.add(name)
    assert len(names) <= 1, (line.head, names)
    assert all(df.is_computed(n) for n in names)


def test_a_bare_command_is_one_column_whatever_its_case():
    """`OBV`, `obv` and `Obv` written bare all name the command `obv`: one cached column."""
    df = _frame()
    df['OBV']
    df['Obv']
    df['obv']
    assert [c for c in df.columns if c.lower() == 'obv'] == ['obv']
    assert volas.directive_stringify('close - OBV') == 'close-obv'
    # a column spelled so is read as data where it is not a command (`rsi` needs a period)
    assert volas.directive_stringify('Close + RSI') == 'Close+RSI'


def test_a_cached_name_does_not_bypass_the_ambiguity_rule():
    """A directive spelled so that it names a cached column is still bound against the
    frame: once a data column `obv` exists, `close - obv` is ambiguous (R17) — it does not
    silently read the column cached as `close - OBV` before that data column existed."""
    df = _frame()
    df['close - OBV']
    df['obv'] = np.zeros(N)  # `OBV` is spelled otherwise, so the column may be added
    with pytest.raises(volas.DirectiveValueError, match=(
            "ambiguous name 'obv' in 'close - obv': it is both a column of this DataFrame and the "
            "built-in command 'obv'. Write 'obv:' to use the command")):
        df['close - obv']
    assert np.asarray(df['close-obv']).shape == (N,)  # the cached column by its own name
    assert np.asarray(df['close - obv:']).shape == (N,)  # the command, explicitly


def _appended(directive, extra=None, rows=N - 1):
    """`directive` cached on the first `rows` rows, then one row appended and read."""
    extra = extra or {}
    df = _frame(rows, **extra)
    df[directive]
    row = {k: v[rows] for k, v in DATA.items()}
    row.update({k: v[rows] for k, v in extra.items()})
    df.append(row)
    return df


def test_a_command_written_with_a_colon_beside_a_data_column_of_its_name():
    """With a data column ``change``, ``change: * 2`` is the command: it caches as
    ``change*2`` and keeps reading the command after an append (the column keeps the
    directive it was bound to); ``change:`` alone returns the command's values without
    caching them over the data column of that name."""
    change = {'change': np.linspace(-1, 1, N)}
    df = _appended('change: * 2', change)
    assert 'change*2' in df.columns and df.is_computed('change*2')
    want = _frame().exec('change:2 * 2', dtype='float64')
    assert _same(df['change*2'], want)
    assert _same(df['change: * 2'], want)
    # the bare name beside that data column is ambiguous (R17)
    with pytest.raises(volas.DirectiveValueError, match="ambiguous name 'change'"):
        df['close * change']
    with pytest.raises(volas.DirectiveValueError, match="ambiguous name 'change'"):
        df['change * 2']
    df = _frame(**change)
    assert _same(df['change:'], _frame().exec('change', dtype='float64'))
    assert not df.is_computed('change') and _same(df['change'], change['change'])


def test_a_data_column_named_like_a_command_that_needs_an_operand():
    """``repeat`` is no command usable bare (its operand is required), so a data column
    of that name is read as data — and the cached name re-reads it."""
    extra = {'repeat': np.linspace(1, 2, N)}
    df = _appended('ACCBANDS@repeat', extra)
    name = volas.directive_stringify('ACCBANDS@repeat')
    assert name == 'accbands@repeat' and df.is_computed(name)
    want = _frame(**extra).exec('accbands@repeat', dtype='float64')
    assert _same(df[name], want)
    assert _same(_frame(**extra).exec(name, dtype='float64'), want)


def test_a_folded_forming_bar_refreshes_a_displaced_ichimoku_span():
    """A tf frame's forming bar folded from a finer bar: the leading span displaced by
    a one-bar kijun refreshes to the full recompute of the frame's rows."""
    rng = np.random.default_rng(9)
    n = 9
    c = 100 + np.cumsum(rng.standard_normal(n + 1))
    labels = np.datetime64('2026-03-08T07:00') + np.arange(n) * np.timedelta64(5, 'm')
    rows = {'open': c[:n], 'high': c[:n] + 1, 'low': c[:n] - 1, 'close': c[:n], 'volume': np.full(n, 10.0)}
    df = volas.DataFrame(rows, index=volas.to_datetime(labels.astype('datetime64[ns]').astype(np.int64)),
                         time_frame='5m')
    d = 'ichimoku.span_a:,1,'
    df[d]
    bar = volas.DataFrame({'open': [c[n]], 'high': [c[n] + 3], 'low': [c[n] - 3], 'close': [c[n]], 'volume': [5.0]},
                          index=volas.to_datetime([int((labels[-1] + np.timedelta64(1, 'm')).astype('datetime64[ns]').astype(np.int64))]))
    df.append(bar)
    assert len(df) == n
    assert _same(df[d], df.exec(d, dtype='float64'))
