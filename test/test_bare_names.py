"""A bare name in a directive is either a data column or a built-in command, never a
cached column: a data column owns every spelling at bind time; a data column that
would change what an existing cached directive reads cannot be added,
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


def test_a_name_that_is_a_column_and_a_command_reads_the_data():
    df = frame()
    for spelling in ('change', 'CHANGE', 'change:'):
        np.testing.assert_array_equal(df.exec('close * ' + spelling), CLOSE * CHANGE)
        np.testing.assert_array_equal(df['close * ' + spelling], CLOSE * CHANGE)
    np.testing.assert_array_equal(df['ma:3@change'],
                                  volas.DataFrame({'x': CHANGE}).exec('ma:3@x'))


def test_every_command_spelling_and_the_renamed_column_read_the_data():
    df = frame()
    for d in ('close * change', 'close * change:'):
        np.testing.assert_array_equal(df[d], CLOSE * CHANGE)
        np.testing.assert_array_equal(df.exec(d), CLOSE * CHANGE)
    np.testing.assert_array_equal(df['change:'], CHANGE)
    renamed = df.rename(columns={'change': 'change_pct'})
    np.testing.assert_array_equal(renamed['close * change_pct'], CLOSE * CHANGE)
    df.append(volas.DataFrame({'close': [CLOSE[-1] + 1], 'volume': [1.0], 'change': [5.0]}))
    want = np.append(CLOSE * CHANGE, (CLOSE[-1] + 1) * 5.0)
    np.testing.assert_array_equal(df['close * change:'], want)


@pytest.mark.parametrize('spelling', ['change', 'CHANGE', 'ChAnGe'])
def test_a_column_named_like_a_command_a_cached_directive_uses_cannot_be_added(spelling):
    df = frame(change=False)
    df[f'close * {spelling}']
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


@pytest.mark.parametrize('spelling', ['change', 'CHANGE', 'ChAnGe'])
def test_binding_reads_the_physical_data_column_under_every_command_spelling(spelling):
    df = volas.DataFrame({'close': CLOSE, spelling: CHANGE})
    before_columns = list(df.columns)
    for name in ('change', 'CHANGE', 'ChAnGe', 'change:', 'CHANGE:'):
        directive = f'close * {name}'
        np.testing.assert_array_equal(df.exec(directive), CLOSE * CHANGE)
        np.testing.assert_array_equal(df[directive], CLOSE * CHANGE)
        np.testing.assert_array_equal(df[name], CHANGE)
    assert list(df.columns) == [*before_columns, f'close*{spelling}']
    assert not df.is_computed(spelling)
    assert df.is_computed(f'close*{spelling}')
    np.testing.assert_array_equal(df.get_column(spelling), CHANGE)


@pytest.mark.parametrize('spelling', ['change', 'CHANGE', 'ChAnGe'])
def test_added_equivalent_column_spelling_requires_dropping_the_cached_command(spelling):
    df = frame(change=False)
    df[f'close * {spelling}']
    before = state(df)
    with pytest.raises(ValueError) as err:
        df['CHANGE'] = CHANGE
    assert str(err.value) == CANNOT_ADD.replace("column 'change'", "column 'CHANGE'")
    assert state(df) == before
    assert df.is_computed('close*change')
    np.testing.assert_array_equal(df['close*change'],
                                  frame(change=False).exec('close * change'))
    dropped = df.drop([f'close * {spelling}'], axis=1)
    assert 'close*change' not in dropped.columns
    dropped['CHANGE'] = CHANGE
    assert not dropped.is_computed('CHANGE')
    for name in ('change', 'CHANGE', 'ChAnGe', 'change:', 'CHANGE:'):
        np.testing.assert_array_equal(dropped[name], CHANGE)
        np.testing.assert_array_equal(dropped.exec(f'close * {name}'), CLOSE * CHANGE)
        np.testing.assert_array_equal(dropped[f'close * {name}'], CLOSE * CHANGE)
    assert df.is_computed('close*change')
    assert state(df) == before


@pytest.mark.parametrize('spelling', ['CHANGE', 'ChAnGe'])
def test_declared_bare_commands_use_canonical_identity_for_added_columns(spelling):
    df = volas.DataFrame({'close': CLOSE, 'volume': VOLUME},
                         window=5, max_lookback=[f'close * {spelling}'])
    before = state(df)
    with pytest.raises(ValueError) as err:
        df['change'] = CHANGE[-5:]
    assert str(err.value) == CANNOT_ADD
    assert state(df) == before
