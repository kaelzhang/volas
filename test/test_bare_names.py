"""A bare name in a directive is either a data column or a built-in command, never a
cached column and never both: a name that could be read either way raises at bind
time, a data column that would change what a cached directive reads cannot be added,
and `exec` / a new directive read the data columns only — so a stale cached column's
placeholder is never computed with."""

import numpy as np
import pytest

import volas

rng = np.random.default_rng(3)
N = 40
CLOSE = 100 + np.cumsum(rng.standard_normal(N))
VOLUME = rng.uniform(1, 100, N)
CHANGE = rng.standard_normal(N)

AMBIGUOUS = (
    "ambiguous name 'change' in 'close * change': it is both a column of this DataFrame "
    "and the built-in command 'change'. Write 'change:' to use the command, or rename the "
    "column — df.rename(columns={'change': 'change_pct'}) — to use the data."
)
CANNOT_ADD = (
    "cannot add column 'change': the cached directive 'close*change' uses the built-in "
    "command 'change' under that name. Rename the new column, or drop the cached directive "
    "first — df.drop(['close*change'], axis=1)."
)


def frame(n=N, change=True):
    data = {'close': CLOSE[:n], 'volume': VOLUME[:n]}
    if change:
        data['change'] = CHANGE[:n]
    return volas.DataFrame(data)


def state(df):
    def cells(c):
        return [None if v != v else v for v in df.get_column(c).to_numpy().tolist()]
    return list(df.columns), {c: cells(c) for c in df.columns}


def test_a_name_that_is_a_column_and_a_command_raises_at_bind_time():
    df = frame()
    before = state(df)
    for read in (lambda: df['close * change'], lambda: df.exec('close * change')):
        with pytest.raises(volas.DirectiveValueError) as err:
            read()
        assert str(err.value) == AMBIGUOUS
        assert state(df) == before
    with pytest.raises(volas.DirectiveValueError, match="ambiguous name 'change' in 'ma:3@change'"):
        df['ma:3@change']


def test_the_command_or_the_renamed_column_resolve_the_ambiguity():
    df = frame()
    command = frame(change=False).exec('close * change')
    assert np.array_equal(df['close * change:'].to_numpy(), command, equal_nan=True)
    assert np.array_equal(df.exec('close * change:'), command, equal_nan=True)
    assert np.array_equal(df['change:'].to_numpy(), frame(change=False).exec('change'), equal_nan=True)
    renamed = df.rename(columns={'change': 'change_pct'})
    assert np.array_equal(renamed['close * change_pct'].to_numpy(), CLOSE * CHANGE)
    # the command form keeps reading the command after an append, never the column
    df['close * change:']
    assert df.is_computed('close*change')  # cached under its name, refreshed as the command
    df.append(volas.DataFrame({'close': [CLOSE[-1] + 1], 'volume': [1.0], 'change': [5.0]}))
    want = volas.DataFrame({'close': list(CLOSE) + [CLOSE[-1] + 1], 'volume': list(VOLUME) + [1.0]})
    assert np.array_equal(df['close * change:'].to_numpy(), want.exec('close * change'),
                          equal_nan=True)


def test_a_column_named_like_a_command_a_cached_directive_uses_cannot_be_added():
    df = frame(change=False)
    df['close * change']
    before = state(df)
    with pytest.raises(ValueError) as err:
        df['change'] = CHANGE
    assert str(err.value) == CANNOT_ADD
    assert state(df) == before
    df['change_pct'] = CHANGE  # another name is fine


def test_a_cached_command_column_overwritten_with_data_is_checked_too():
    df = frame(change=False)
    df['obv']
    df['close - obv']
    with pytest.raises(ValueError, match="cannot add column 'obv'"):
        df['obv'] = np.zeros(N)
    assert df.is_computed('obv')


def test_a_new_composite_never_reads_a_stale_cached_column():
    df = frame(N - 2, change=False)
    df['obv']
    df.append(volas.DataFrame({'close': CLOSE[N - 2:N - 1], 'volume': VOLUME[N - 2:N - 1]}))
    df['close - obv']
    df.append(volas.DataFrame({'close': CLOSE[N - 1:], 'volume': VOLUME[N - 1:]}))
    df.fulfill()
    full = frame(change=False)
    assert np.array_equal(df['close - obv'].to_numpy(), full.exec('close - obv'), equal_nan=True)


def test_exec_reads_data_columns_only_and_the_visible_rows():
    df = frame(N - 1, change=False)
    df['obv']
    df.append(volas.DataFrame({'close': CLOSE[-1:], 'volume': VOLUME[-1:]}))
    assert np.array_equal(df.exec('obv'), frame(change=False).exec('obv'), equal_nan=True)
    assert np.array_equal(df.exec('close - obv'), frame(change=False).exec('close - obv'),
                          equal_nan=True)
    win = volas.DataFrame({'close': CLOSE, 'volume': VOLUME}, window=5, max_lookback=3)
    assert len(win.exec('mom:3')) == 5
    assert np.array_equal(win.exec('mom:3'), frame(change=False).exec('mom:3')[-5:])



def test_the_suggested_drop_works_with_the_directive_as_written():
    df = frame(change=False)
    df['close * change']
    with pytest.raises(ValueError) as err:
        df['change'] = CHANGE
    assert str(err.value) == CANNOT_ADD
    for spelling in ('close * change', 'close*change'):  # the owner's and the message's
        assert 'close*change' not in df.drop([spelling], axis=1).columns
    assert df.is_computed('close * change')
    assert np.array_equal(df.get_column('close * change').to_numpy(), df['close*change'].to_numpy(),
                          equal_nan=True)
    dropped = df.drop(['close * change'], axis=1)
    assert 'close*change' not in dropped.columns
    dropped['change'] = CHANGE
    assert df.is_computed('close * change')


def test_several_cached_directives_are_named_together():
    df = frame(change=False)
    df['close * change']
    df['change + 1']
    with pytest.raises(ValueError) as err:
        df['change'] = CHANGE
    assert "the cached directives 'change+1', 'close*change' use" in str(err.value) or \
        "the cached directives 'close*change', 'change+1' use" in str(err.value)
    assert 'drop the cached directives first' in str(err.value)


def test_a_command_with_a_required_series_is_not_a_bare_command():
    df = volas.DataFrame({'close': CLOSE, 'repeat': CHANGE})
    assert np.array_equal(df.exec('close + repeat'), CLOSE + CHANGE)
