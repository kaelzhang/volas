"""Directive spellings bind to the physical data label and keep that binding live."""

import numpy as np
import pytest

from volas import DataFrame


@pytest.mark.parametrize('physical', ['atr:14', 'ATR:014', 'ATR:14', 'atr:0014'])
@pytest.mark.parametrize('spelling', ['atr:14', 'ATR:014', 'ATR:14', 'atr:0014'])
def test_every_directive_spelling_reads_the_actual_data_label(physical, spelling):
    values = np.arange(1., 7.)
    df = DataFrame({physical: values})
    assert df[spelling].to_numpy().tolist() == values.tolist()
    assert df.exec(spelling).tolist() == values.tolist()
    assert df.exec(f'{spelling} + 0').tolist() == values.tolist()
    assert df.exec(f'ema:2@({spelling})').tolist()[1:] == DataFrame({'close': values}).exec('ema:2').tolist()[1:]
    assert df[[spelling]].columns == [physical]
    assert df.get_column(spelling).to_numpy().tolist() == values.tolist()
    assert DataFrame(df, columns=[spelling]).columns == [physical]


@pytest.mark.parametrize('physical', ['change', 'CHANGE', 'CHANGE:', 'CHANGE:02'])
def test_bare_command_spellings_read_data_before_command_binding(physical):
    values = np.arange(1., 7.)
    df = DataFrame({'close': values * 3, physical: values})
    # Default-argument spellings refer to the same data series.
    for spelling in ['change', 'CHANGE', 'change:', 'CHANGE:02']:
        assert df[spelling].to_numpy().tolist() == values.tolist()
        assert df.exec(f'{spelling} * 2').tolist() == (values * 2).tolist()


@pytest.mark.parametrize('name', ['change', 'CHANGE', 'CHANGE:', 'CHANGE:02'])
@pytest.mark.parametrize('cached', ['close * change', 'close * change:'])
def test_adding_alias_of_a_cached_command_is_atomic(name, cached):
    df = DataFrame({'close': np.arange(1., 7.)})
    df[cached]
    columns = df.columns
    with pytest.raises(ValueError, match='cannot add column'):
        df[name] = np.ones(6)
    assert df.columns == columns


@pytest.mark.parametrize('cached', ['close + ema:2', 'ema:3@(ema:2)'])
def test_adding_alias_of_a_nested_cached_command_is_atomic(cached):
    df = DataFrame({'close': np.arange(1., 7.)})
    df[cached]
    columns = df.columns
    with pytest.raises(ValueError, match='cannot add column'):
        df['EMA:02'] = np.ones(6)
    assert df.columns == columns


@pytest.mark.parametrize('physical', ['atr:14', 'ATR:014', 'ATR:14', 'atr:0014'])
@pytest.mark.parametrize('directive', ['atr:14 + 0', 'ema:2@(atr:14)'])
def test_alias_getter_refreshes_its_existing_cached_expression(physical, directive):
    values = np.arange(1., 7.)
    df = DataFrame({physical: values})
    cached = df[directive].name
    changed = values + 10
    df[physical] = changed
    want = changed if '+' in directive else DataFrame({'close': changed}).exec('ema:2')
    np.testing.assert_array_equal(df[directive].to_numpy(), want)
    # The read must refresh the existing cache cursor as well as return correct values.
    np.testing.assert_array_equal(df.to_numpy()[:, df.columns.index(cached)], want)
