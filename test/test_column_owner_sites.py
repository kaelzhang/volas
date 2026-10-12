"""Every column write, read and fold goes through its one owner.

- a validity mask must cover exactly its values (construction never reads past it);
- equality reads presence, never the payload under a missing cell;
- a value of another dtype is stored only when it converts losslessly (assignment,
  frame append, row replacement, the time-frame fold);
- aggregation: ``first`` / ``last`` are literal endpoints, ``max`` / ``min`` / ``sum``
  are missing as soon as an input is — fold and batch alike, every numeric dtype;
- column-wise reductions keep bool columns, and float64 statistics raise past 2^53;
- directive numbers stay exact (integer literals, comparisons) and float64 math
  raises past 2^53; directive booleans carry presence (Kleene logic) through the
  cache, appends and refreshes.
"""

import numpy as np
import pyarrow as pa
import pytest

import volas
from volas import NA

BIG = 2**53 + 1  # the first integer float64 cannot hold


def _ohlcv(n, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.standard_normal(n))
    return {'open': c + 0.1, 'high': c + 1, 'low': c - 1, 'close': c, 'volume': rng.uniform(1, 9, n)}


def _ts(n, start='2024-01-02 09:30'):
    return volas.to_datetime(np.arange(n) * 60_000_000_000 + volas.Timestamp(start).value)


def _cells(values):
    """The cells of a Series or an exported array, a missing one as ``NA``."""
    cells = values.to_list() if hasattr(values, 'to_list') else values.tolist()
    return [NA if v is NA or v is None or (isinstance(v, float) and np.isnan(v)) else v for v in cells]


# --- the validity layout ------------------------------------------------------------


@pytest.mark.parametrize('dtype', ['int64', 'int32', 'bool', 'str', 'float64'])
def test_a_missing_mask_of_another_length_is_rejected(dtype):
    values = {'int64': [1, 2, 3], 'int32': [1, 2, 3], 'bool': [True, False, True],
              'str': ['a', 'b', 'c'], 'float64': [1.0, 2.0, 3.0]}[dtype]
    for mask in ([False] * 5, [True] * 5, [False]):
        with pytest.raises(ValueError, match='missing mask has'):
            volas.DataFrame({'x': (values, mask, dtype)})
    ok = volas.DataFrame({'x': (values, [False, True, False], dtype)})
    assert ok['x'].isna().to_list() == [False, True, False]


def test_a_masked_array_keeps_its_mask_for_every_dtype():
    for data in ([1, 2, 3], [1.5, 2.5, 3.5], ['a', 'b', 'c'], [True, False, True]):
        arr = np.ma.masked_array(np.array(data), mask=[False, True, False])
        s = volas.DataFrame({'x': arr})['x']
        assert s.isna().to_list() == [False, True, False]


# --- equality reads presence --------------------------------------------------------


def _int_with_payload(payload):
    data = pa.py_buffer(np.array([payload, 5], dtype='int64').tobytes())
    validity = pa.py_buffer(np.packbits([0, 1], bitorder='little').tobytes())
    return pa.Array.from_buffers(pa.int64(), 2, [validity, data])


def test_equality_never_reads_a_missing_payload():
    a = volas.DataFrame({'x': volas.Series.from_arrow(_int_with_payload(0))})
    b = volas.DataFrame({'x': volas.Series.from_arrow(_int_with_payload(7))})
    assert a.equals(b)
    assert not a.equals(volas.DataFrame({'x': [3, 5]}))


# --- lossless storage -------------------------------------------------------------


def test_assigning_values_converts_losslessly():
    df = volas.DataFrame({'f': [0.0, 0.0], 'g': np.array([0.0, 0.0], dtype='float32'), 'n': [0, 0]})
    with pytest.raises(TypeError, match='not exactly representable'):
        df.iloc[0:2, 0] = np.array([BIG, 1], dtype='int64')
    with pytest.raises(OverflowError, match='out of the float32 range'):
        df.iloc[0:2, 1] = np.array([1e300, 1.0])
    with pytest.raises(TypeError, match='not exactly representable'):
        df.iloc[0:2, 2] = np.array([2.5, 1.0])
    assert df['f'].to_list() == [0.0, 0.0] and df['n'].to_list() == [0, 0]
    df.iloc[0:2, 0] = np.array([2**53, 1], dtype='int64')  # exact: stored
    df.iloc[0:2, 2] = np.array([2.0, np.nan])  # integral -> int, NaN -> NA
    assert df['f'].to_list() == [float(2**53), 1.0]
    assert _cells(df['n']) == [2, NA]


def test_a_frame_append_converts_losslessly_like_a_dict_append():
    df = volas.DataFrame({'n': np.array([1, 2], dtype='int64'), 'f': [0.5, 1.5]})
    df.append(volas.DataFrame({'n': [3.0], 'f': np.array([4], dtype='int64')}))
    df.append({'n': 4.0, 'f': 5})
    assert df['n'].dtype == 'int64' and df['n'].to_list() == [1, 2, 3, 4]
    assert df['f'].to_list() == [0.5, 1.5, 4.0, 5.0]
    with pytest.raises(TypeError, match='not exactly representable'):
        df.append(volas.DataFrame({'n': [2.5], 'f': [1.0]}))
    with pytest.raises(TypeError, match='not exactly representable'):
        df.append(volas.DataFrame({'n': [5], 'f': np.array([BIG], dtype='int64')}))


def test_a_replacing_row_converts_losslessly():
    idx = _ts(2)
    df = volas.DataFrame({'time': idx, 'n': np.array([1, 2], dtype='int64')}).set_index('time')
    df.append(volas.DataFrame({'time': idx[1:], 'n': [7.0]}).set_index('time'))
    assert len(df) == 2 and df['n'].to_list() == [1, 7]
    with pytest.raises(TypeError):
        df.append(volas.DataFrame({'time': idx[1:], 'n': [7.5]}).set_index('time'))


# --- aggregation: fold == batch, a missing input is a missing aggregate ----------------


def _fine(cols, n=6):
    return volas.DataFrame({'time': _ts(n), **cols}).set_index('time')


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'int64', 'int32'])
def test_fold_equals_cumulate_with_a_missing_input(dtype):
    vals = [3, 9, None, 4, 7, 1]
    col = ([v or 0 for v in vals], [v is None for v in vals], dtype) if dtype.startswith('int') else \
        np.array([np.nan if v is None else v for v in vals], dtype=dtype)
    fine = _fine({'open': [1.0] * 6, 'high': [1.0] * 6, 'low': [1.0] * 6, 'close': [1.0] * 6, 'x': col})
    for agg in ('max', 'min', 'sum', 'first', 'last'):
        batch = fine.cumulate('3m', cumulators={'x': agg})
        live = fine.iloc[:1].cumulate('3m', cumulators={'x': agg})
        for i in range(1, 6):
            live.append(fine.iloc[i:i + 1])
        assert live['x'].dtype == batch['x'].dtype == dtype, agg
        got, want = _cells(live['x']), _cells(batch['x'])
        assert got == want, (agg, got, want)
        assert got[0] is NA  # the missing input decides, first / last too
        assert got[1] == {'max': 7, 'min': 1, 'sum': 12, 'first': 4, 'last': 1}[agg]


def test_bool_and_str_columns_fold_like_the_batch():
    flags = [True, None, False, True, True, False]
    syms = ['a', 'b', None, 'c', 'd', 'e']
    fine = _fine({'close': [1.0] * 6, 'f': ([bool(v) for v in flags], [v is None for v in flags], 'bool'),
                  's': ([v or '' for v in syms], [v is None for v in syms], 'str')})
    for agg in ('first', 'last'):
        cum = {'f': agg, 's': agg}
        batch = fine.cumulate('3m', cumulators=cum)
        live = fine.iloc[:1].cumulate('3m', cumulators=cum)
        for i in range(1, 6):
            live.append(fine.iloc[i:i + 1])
            if i == 4:  # a re-sent bar re-cumulates the forming period
                live.append(fine.iloc[i:i + 1])
        for col in ('f', 's'):
            assert _cells(live[col]) == _cells(batch[col]), (agg, col)
            assert _cells(batch[col])[0] is NA, (agg, col)  # the period holds a missing value
        assert _cells(batch['f'])[1] == {'first': True, 'last': False}[agg]
        assert _cells(batch['s'])[1] == {'first': 'c', 'last': 'e'}[agg]


def test_an_unsupported_aggregation_is_rejected_up_front():
    fine = _fine({'close': [1.0] * 6, 't': _ts(6, '2020-01-01'), 'sym': ['x'] * 6})
    with pytest.raises(TypeError, match='cannot sum the datetime'):
        fine.cumulate('3m', cumulators={'t': 'sum'})
    with pytest.raises(TypeError, match='cannot max the str column "sym"'):
        fine.cumulate('3m', cumulators={'sym': 'max'})
    seed = volas.DataFrame({'time': _ts(2), 'close': [1.0, 2.0], 'sym': ['x', 'y']}).set_index('time')
    with pytest.raises(TypeError, match='cannot max the str column "sym"'):
        volas.DataFrame(seed, time_frame='1m', cumulators={'sym': 'max'})


# --- column-wise reductions ----------------------------------------------------------


def test_every_frame_reduction_keeps_bool_columns():
    df = volas.DataFrame({'b': [True, False, True], 'x': [1.0, 2.0, 4.0]})
    for op in ('sum', 'prod', 'min', 'max', 'mean', 'var', 'std', 'median', 'sem', 'skew', 'kurt'):
        assert list(getattr(df, op)().index) == ['b', 'x'], op
    assert df.sum().to_list() == [2.0, 7.0]
    bools = volas.DataFrame({'b': [True, False], 'c': [True, True]})
    assert bools.max().to_list() == [True, True] and bools.min().to_list() == [False, True]
    assert list(bools.corr().index) == ['b', 'c']


@pytest.mark.parametrize('op', ['corr', 'cov', 'sem', 'skew', 'kurt'])
def test_float64_statistics_raise_past_2_53(op):
    df = volas.DataFrame({'n': np.array([BIG, 1, 2], dtype='int64'), 'x': [1.0, 2.0, 3.0]})
    with pytest.raises(OverflowError, match='2\\^53'):
        getattr(df, op)()
    getattr(volas.DataFrame({'n': np.array([2**53, 1, 2], dtype='int64')}), op)()


# --- directive numbers ----------------------------------------------------------------


def test_directive_integers_compare_exactly():
    df = volas.DataFrame({'n': np.array([BIG, 2**53, 7], dtype='int64'), 'm': np.array([2**53, BIG, 7], dtype='int64')})
    assert df.exec('n > 9007199254740992').tolist() == [True, False, False]
    assert df.exec('n == 9007199254740993').tolist() == [True, False, False]
    assert df.exec('n > m').tolist() == [True, False, False]
    assert df.exec('n == m').tolist() == [False, False, True]
    assert volas.directive_stringify('n>9007199254740993') == 'n>9007199254740993'
    assert volas.directive_stringify('n>2.0') == 'n>2'


@pytest.mark.parametrize('d', ['n - 1', 'n * 2', 'n / 2', '-n', 'ma:1@n', 'sum:1@n', 'minmax.max:1@n'])
def test_directive_float64_math_raises_past_2_53(d):
    df = volas.DataFrame({'n': np.array([BIG, 1], dtype='int64'), 'close': [1.0, 2.0]})
    with pytest.raises(OverflowError, match='2\\^53'):
        df.exec(d)
    small = volas.DataFrame({'n': np.array([2**53, 1], dtype='int64'), 'close': [1.0, 2.0]})
    small.exec(d)


# --- directive booleans carry presence ---------------------------------------------------


def test_a_comparison_with_a_missing_operand_is_missing():
    df = volas.DataFrame({'close': [1.0, 2.0, 3.0, 2.0, 5.0]})
    assert _cells(df.exec('close > ma:3')) == [NA, NA, True, False, True]
    # `~` keeps a missing truth missing (it used to turn warm-up NaN into True)
    assert _cells(df.exec('~(close > ma:3)')) == [NA, NA, False, True, False]
    # Kleene: a present false decides `&`, a present true decides `|`
    assert _cells(df.exec('(close > ma:3) & (close < 0)')) == [False] * 5
    assert _cells(df.exec('(close > ma:3) | (close > 0)')) == [True] * 5
    assert _cells(df.exec('(close > ma:3) ^ (close > 0)')) == [NA, NA, False, True, False]
    assert _cells(df.exec('ma:3 != 0')) == [NA, NA, True, True, True]
    # a cross needs both rows: the first row and warm-up rows are missing unless false
    assert _cells(df.exec('close // ma:3')) == [NA, NA, NA, False, True]


def test_the_boolean_commands_carry_presence():
    df = volas.DataFrame({'open': [1.0, np.nan, 1.0, 1.0], 'close': [2.0, 2.0, 0.5, 3.0]})
    assert _cells(df.exec('style.bullish')) == [True, NA, False, True]
    assert _cells(df.exec('increase:1@close')) == [NA, False, False, True]
    assert _cells(df.exec('repeat:2@(close > open)')) == [NA, NA, False, False]


BOOLS = ('close > ma:5', '~(close > ma:5)', 'increase:2@(ma:3)', 'close // ma:3', 'repeat:2@(close > ma:4)')


def test_a_cached_boolean_keeps_missing_through_appends_and_refreshes():
    cols = _ohlcv(30)
    full = volas.DataFrame(cols)
    live = volas.DataFrame({k: v[:3] for k, v in cols.items()})
    for d in BOOLS:
        live[d]
    for i in range(3, 30):
        live.append(volas.DataFrame({k: v[i:i + 1] for k, v in cols.items()}))
        if i % 4 == 0:
            live.fulfill()
    for d in BOOLS:
        assert _cells(live[d]) == _cells(full[d]), d
        assert _cells(live[d])[:2] == [NA] * 2, d  # warm-up truths are missing
