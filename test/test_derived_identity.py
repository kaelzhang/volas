"""Regression tests for projection identity and continuation boundaries."""
import numpy as np
import pytest
from volas import DataFrame


def bits(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_array_equal(a[~np.isnan(a)].view('u8'), b[~np.isnan(b)].view('u8'))


@pytest.mark.parametrize('form', ['getitem', 'constructor', 'loc', 'iloc', 'window'])
def test_projection_starts_own_coordinates(form):
    f = DataFrame({'close': np.arange(1., 21.), 'volume': np.ones(20)}).iloc[17:]
    p = {'getitem': lambda: f[['close']],
         'constructor': lambda: DataFrame(f, columns=['close']),
         'loc': lambda: f.loc[:, ['close']], 'iloc': lambda: f.iloc[:, [0]],
         'window': lambda: DataFrame({'close': np.arange(1., 21.)}, window=3, max_lookback=2)[['close']]}[form]()
    own = DataFrame({'close': p['close'].to_numpy()}, index=p.index)
    bits(p.exec('maxindex:2'), own.exec('maxindex:2'))


@pytest.mark.parametrize('reverse', [False, True])
def test_equivalent_physical_owners_rejected_at_construction(reverse):
    pairs = [('ema:2', [10., 20., 30.]), ('EMA:02', [40., 50., 60.])]
    with pytest.raises(ValueError, match='ema:2.*EMA:02|EMA:02.*ema:2'):
        DataFrame(dict(pairs[::-1] if reverse else pairs))


@pytest.mark.parametrize('operation', ['add', 'rename'])
def test_equivalent_physical_owners_rejected_atomically(operation):
    f = DataFrame({'ema:2': [10., 20., 30.], 'x': [1., 2., 3.]})
    before = f.to_numpy().copy()
    with pytest.raises(ValueError, match='ema:2.*EMA:02|EMA:02.*ema:2'):
        if operation == 'add':
            f['EMA:02'] = [40., 50., 60.]
        else:
            f.rename(columns={'x': 'EMA:02'})
    np.testing.assert_array_equal(f.to_numpy(), before)
    assert f.columns == ['ema:2', 'x']


@pytest.mark.parametrize('expr', ['ema:2@(EMA:2)', 'EMA:2 + 0'])
def test_cold_projection_carries_bound_storage_name(expr):
    f = DataFrame({'close': [1., 2., 3.], 'EMA:2': [10., 20., 30.]})
    want = f.exec(expr)
    p = f[[expr]]
    bits(p.to_numpy().ravel(), want)
    bits(f[[expr]].to_numpy().ravel(), want)


def test_failed_projection_leaves_source_unchanged():
    f = DataFrame({'close': [1., 2., 3.], 'EMA:2': [10., 20., 30.]})
    before = f.to_numpy().copy()
    with pytest.raises(ValueError):
        f[['ema:2@(EMA:2)', 'ema:2@(EMA:2)']]
    assert f.columns == ['close', 'EMA:2']
    np.testing.assert_array_equal(f.to_numpy(), before)


@pytest.mark.parametrize('directive', ['ma:5', 'maxindex:5', 'ema:2'])
@pytest.mark.parametrize('tf', [False, True])
def test_window_column_drop_keeps_continuation(directive, tf):
    idx = np.datetime64('2026-01-01', 'ns') + np.arange(11) * np.timedelta64(5, 'm')
    kw = {'time_frame': '5m'} if tf else {}
    f = DataFrame({'close': np.arange(1., 11.), 'volume': np.ones(10)}, index=idx[:10], window=2, max_lookback=[directive], **kw)
    s = f.drop(['volume'], axis=1)
    s.append(DataFrame({'close': [11.]}, index=idx[10:]))
    bits(s[directive].to_numpy(dtype='float64'), DataFrame({'close': np.arange(1., 12.)}, index=idx).exec(directive)[-3:])


@pytest.mark.parametrize('physical', ['atr:14', 'ATR:14', 'atr:014'])
def test_ready_uses_bound_declaration(physical):
    f = DataFrame({'close': [1., 2.], physical: [10., 20.]}, window=1, max_lookback=['ema:5@(atr:14)'])
    assert not f.ready


@pytest.mark.parametrize('accessor', ['iloc', 'loc'])
@pytest.mark.parametrize('window', [False, True])
@pytest.mark.parametrize('tf', [False, True])
@pytest.mark.parametrize('step', [2, -2])
@pytest.mark.parametrize('mask', [[False, False, True, True, True, True], [True, False, True, False, True, False]])
def test_strided_bool_mask_keeps_boolean_identity(accessor, step, mask, window, tf):
    idx = np.datetime64('2026-01-01', 'ns') + np.arange(10) * np.timedelta64(5, 'm')
    kw = {'time_frame': '5m'} if tf else {}
    if window:
        kw.update(window=6, max_lookback=['ema:2'])
    f = DataFrame({'close': np.arange(10.)}, index=idx, **kw)
    if not window:
        f = f.iloc[-6:]
    f['ema:2']
    storage = np.zeros(12, dtype=bool)
    view = storage[::step]
    view[:] = mask
    got = getattr(f, accessor)[view]
    want = f[np.array(mask)]
    assert got.columns == want.columns
    np.testing.assert_array_equal(np.asarray(got.index), np.asarray(want.index))
    bits(got['close'].to_numpy(), want['close'].to_numpy())


@pytest.mark.parametrize('form', ['loc', 'iloc'])
def test_explicit_all_column_list_is_projection(form):
    f = DataFrame({'close': np.arange(1., 21.)}).iloc[17:]
    p = f.loc[:, ['close']] if form == 'loc' else f.iloc[:, [0]]
    bits(p.exec('maxindex:2'), [np.nan, 1., 2.])


@pytest.mark.parametrize('physical, aliases', [('EMA:02', ['ema:2', 'EMA:2', 'ema:02']), ('BOLL:20@close', ['boll', 'BOLL', 'boll:'])])
def test_unique_physical_owner_used_by_all_bindings(physical, aliases):
    values = [10., 20., 30.]
    f = DataFrame({'close': [1., 2., 3.], physical: values})
    for alias in aliases:
        bits(f[alias].to_numpy(), values)
        bits(f.exec(f'({alias})+0'), values)
        bits(f.exec(f'ema:2@({alias})'), [np.nan, 15., 25.])
        assert f[[alias]].columns == [physical]


@pytest.mark.parametrize('source', ['arrow', 'pandas', 'csv'])
def test_import_rejects_equivalent_physical_owners(source, tmp_path):
    data = {'ema:2': [1., 2.], 'EMA:02': [3., 4.]}
    with pytest.raises(ValueError, match='ema:2.*EMA:02'):
        if source == 'arrow':
            import pyarrow as pa
            DataFrame.from_arrow(pa.table(data))
        elif source == 'pandas':
            import pandas as pd
            DataFrame.from_pandas(pd.DataFrame(data))
        else:
            from volas import read_csv
            path = tmp_path / 'owners.csv'
            path.write_text('ema:2,EMA:02\n1,3\n2,4\n')
            read_csv(str(path))


@pytest.mark.parametrize('accessor', ['iloc', 'loc'])
def test_bool_mask_wrong_length_never_becomes_positions(accessor):
    f = DataFrame({'close': np.arange(6.)})
    with pytest.raises(ValueError, match='boolean mask length'):
        getattr(f, accessor)[np.array([True, False, True])[::-1]]


@pytest.mark.parametrize('directive', ['var:5', 'stddev:5'])
def test_projection_reanchors_numeric_kernels_in_own_coordinates(directive):
    x = 100 + np.random.default_rng(42).normal(0, .001, 100)
    p = DataFrame({'close': x}).iloc[1:][['close']]
    bits(p.exec(directive), DataFrame({'close': x[1:]}).exec(directive))


@pytest.mark.parametrize('directive', ['ma:5', 'maxindex:5', 'ema:2', 'ema:3@(ma:5)'])
@pytest.mark.parametrize('mutation', ['replace', 'write'])
@pytest.mark.parametrize('tf', [False, True])
def test_window_column_drop_keeps_history_for_mutations(directive, mutation, tf):
    idx = np.datetime64('2026-01-01', 'ns') + np.arange(40) * np.timedelta64(5, 'm')
    values = 100 + np.sin(np.arange(40) * .3)
    kw = {'time_frame': '5m'} if tf else {}
    f = DataFrame({'close': values, 'volume': np.ones(40)}, index=idx, window=2, max_lookback=[directive], **kw)
    s = f.drop(['volume'], axis=1)
    expected = values.copy()
    if mutation == 'replace':
        expected[-1] = 123.
        s.append(DataFrame({'close': [123.]}, index=idx[-1:]))
    else:
        expected[-2] = 123.
        s.iat[0, 0] = 123.
    bits(s[directive].to_numpy(dtype='float64'), DataFrame({'close': expected}, index=idx).exec(directive)[-2:])


@pytest.mark.parametrize('physical', ['ATR:14', 'atr:014'])
def test_bound_declaration_ready_survives_copy_and_rewindow(physical):
    expression = 'ema:5@(atr:14)'
    f = DataFrame({'close': [1., 2.], physical: [10., 20.]}, window=1, max_lookback=[expression, expression])
    for source in [f, f.copy(), DataFrame(f), DataFrame(f, window=1, max_lookback=[expression])]:
        assert not source.ready
        for value in [30., 40.]:
            source.append({'close': 1., physical: value})
            source.fulfill()
            assert not source.ready
        source.append({'close': 1., physical: 50.})
        source.fulfill()
        assert source.ready


@pytest.mark.parametrize('accessor', ['iloc', 'loc'])
@pytest.mark.parametrize('step', [2, -2])
def test_strided_column_bool_mask_uses_boolean_identity(accessor, step):
    f = DataFrame({'close': [1., 2., 3.], 'volume': [10., 20., 30.]})
    storage = np.zeros(4, dtype=bool)
    mask = storage[::step]
    mask[:] = [False, True]
    selected = getattr(f, accessor)[:, mask]
    assert selected.columns == ['volume']
    bits(selected.to_numpy().ravel(), [10., 20., 30.])
    getattr(f, accessor)[:, mask] = 7.
    bits(f['close'].to_numpy(), [1., 2., 3.])
    bits(f['volume'].to_numpy(), [7., 7., 7.])


@pytest.mark.parametrize('accessor', ['iloc', 'loc'])
def test_non_vector_bool_dtype_never_becomes_integer_positions(accessor):
    f = DataFrame({'close': [1., 2., 3.]})
    with pytest.raises(TypeError):
        getattr(f, accessor)[np.array([[True], [False], [True]])]
    with pytest.raises(TypeError):
        getattr(f, accessor)[np.bool_(True)]


@pytest.mark.parametrize('accessor', ['iloc', 'loc'])
def test_empty_lists_remain_empty_selections(accessor):
    f = DataFrame({'close': [1., 2., 3.], 'volume': [10., 20., 30.]})
    assert getattr(f, accessor)[[]].shape == (0, 2)
    assert getattr(f, accessor)[:, []].shape == (3, 0)


def test_loc_projection_uses_canonical_physical_owner():
    f = DataFrame({'close': [1., 2., 3.], 'EMA:02': [10., 20., 30.]})
    assert f.loc[:, ['ema:2']].columns == ['EMA:02']
    expression = 'ema:2@(ema:2)'
    expected = f[expression].to_numpy()
    bits(f.loc[:, [expression]].to_numpy().ravel(), expected)


def test_named_cell_and_exports_use_physical_identity():
    f = DataFrame({'close': [1., 2., 3.], 'EMA:02': [10., 20., 30.]})
    assert f.at[0, 'ema:2'] == 10.
    f.at[0, 'ema:2'] = 11.
    assert f['EMA:02'].to_list() == [11., 20., 30.]
    out = np.empty((3, 1))
    f.fill_into(out, columns=['ema:2'])
    bits(out.ravel(), [11., 20., 30.])
    assert 'EMA:02' in f.to_csv(columns=['ema:2'], index=False)
    assert 'EMA:02' in f.to_string(columns=['ema:2'])


@pytest.mark.parametrize('accessor', ['iloc', 'loc'])
def test_column_mask_scalar_assignment_uses_write_transaction(accessor):
    f = DataFrame({'close': [1., 2., 3.], 'volume': [10., 20., 30.]})
    getattr(f, accessor)[:, np.array([True, True])] = 7.
    bits(f.to_numpy().ravel(), [7.] * 6)
    mixed = DataFrame({'close': [1., 2., 3.], 'text': ['a', 'b', 'c']})
    before = mixed.to_numpy().copy()
    with pytest.raises(TypeError):
        getattr(mixed, accessor)[:, np.array([True, True])] = 7.
    np.testing.assert_array_equal(mixed.to_numpy(), before)


@pytest.mark.parametrize('form', ['getitem', 'constructor'])
def test_failed_tf_projection_preserves_source_freshness(form):
    idx = np.datetime64('2026-01-01', 'ns') + np.arange(4) * np.timedelta64(5, 'm')
    seed = DataFrame({'open': [0., 0., 0.], 'close': [1., 2., 3.]}, index=idx[:3])
    seed['close>open']
    f = DataFrame(seed, time_frame='5m', cumulators={'close>open': 'sum'})
    f.append(DataFrame({'open': [0.], 'close': [4.]}, index=idx[3:]))
    with pytest.raises(ValueError, match='stale'):
        f.to_numpy()
    with pytest.raises(TypeError, match='cannot sum the bool'):
        if form == 'getitem':
            f[['close>open']]
        else:
            DataFrame(f, columns=['close>open'])
    with pytest.raises(ValueError, match='stale'):
        f.to_numpy()
