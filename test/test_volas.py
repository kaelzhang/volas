"""volas behaviour + correctness tests.

Indicator/operator correctness is verified by 1:1 parity against stock-pandas on
real Tencent data (`test_parity_*`). Core API behaviour is covered directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_pandas import StockDataFrame

from volas import DataFrame, Series

from .common import create_stock, get_last, get_tencent

_TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


def _read_tencent_pandas() -> pd.DataFrame:
    """Raw Tencent CSV as a pandas frame — used only to build the stock-pandas
    parity oracle below (the sole pandas data path left in the non-interop tests).
    """
    return pd.read_csv(_TENCENT)


@pytest.fixture
def stock():
    return get_tencent()


@pytest.fixture(scope='module')
def spd():
    """A stock-pandas StockDataFrame on the same data, as the parity oracle."""
    return StockDataFrame(_read_tencent_pandas())


# ---------------------------------------------------------------------------
# Core DataFrame / Series API
# ---------------------------------------------------------------------------

def test_construct_from_dict():
    df = create_stock()
    assert df.shape == (6, 5)
    assert df.columns == ['open', 'close', 'high', 'low', 'volume']
    assert len(df) == 6


def test_construct_from_numpy():
    df = DataFrame({'close': np.array([1.0, 2.0, 3.0])})
    assert df.shape == (3, 1)
    assert df['close'].to_numpy().tolist() == [1.0, 2.0, 3.0]


def test_construct_unequal_lengths_raises():
    with pytest.raises(ValueError):
        DataFrame({'a': [1.0, 2.0], 'b': [1.0]})


def test_construct_bad_value_raises():
    with pytest.raises(TypeError):
        DataFrame({'a': 'not-an-array'})


def test_getitem_column_returns_series(stock):
    s = stock['close']
    assert isinstance(s, Series)
    assert s.name == 'close'
    assert s.dtype == 'float64'
    assert len(s) == len(stock)


def test_getitem_list_returns_dataframe(stock):
    sub = stock[['open', 'close']]
    assert isinstance(sub, DataFrame)
    assert sub.columns == ['open', 'close']
    assert sub.shape == (len(stock), 2)


def test_getitem_missing_column_raises(stock):
    with pytest.raises((KeyError, ValueError)):
        stock['definitely_missing_column_xyz']


def test_get_column(stock):
    s = stock.get_column('close')
    assert isinstance(s, Series)
    np.testing.assert_array_equal(s.to_numpy(), stock['close'].to_numpy())


def test_get_column_missing_raises(stock):
    with pytest.raises(KeyError):
        stock.get_column('Close')


def test_to_numpy_2d(stock):
    arr = stock.to_numpy()
    assert arr.shape == (len(stock), len(stock.columns))
    np.testing.assert_array_equal(arr[:, 0], stock['open'].to_numpy())


def test_append():
    df = create_stock()
    other = DataFrame({
        'open': [8.0], 'close': [9.0], 'high': [18.0], 'low': [7.0], 'volume': [800.0],
    })
    out = df.append(other)
    assert out.shape == (7, 5)
    assert get_last(out['close']) == 9.0


def test_bool_mask_filter_via_series(stock):
    mask = stock['close > open']
    filtered = stock[mask]
    assert isinstance(filtered, DataFrame)
    closes = stock['close'].to_numpy()
    opens = stock['open'].to_numpy()
    assert len(filtered) == int((closes > opens).sum())


def test_bool_mask_filter_via_numpy(stock):
    mask = stock['close'].to_numpy() > stock['open'].to_numpy()
    filtered = stock[mask]
    assert len(filtered) == int(mask.sum())


def test_series_repr(stock):
    # pandas-style vertical repr with a Name/dtype footer (not the old summary)
    last = repr(stock['close']).splitlines()[-1]
    assert last.startswith('Name: close,') and last.endswith('dtype: float64')


def test_dataframe_repr(stock):
    # pandas-style aligned table, not the old `DataFrame(columns=..., shape=...)`
    r = repr(stock)
    assert 'shape=' not in r and 'columns=' not in r
    assert 'close' in r


def test_series_dtype_bool(stock):
    assert stock['close > open'].dtype == 'bool'


# ---------------------------------------------------------------------------
# Indicator parity vs stock-pandas (1:1 on real Tencent data)
# ---------------------------------------------------------------------------

PARITY = [
    'ma:5', 'ma:20', 'ma:10@open',
    'boll', 'boll.upper', 'boll.lower', 'boll.u', 'boll.l', 'bbw',
    'rsv:9',
    'bbi',
    'llv:10', 'hhv:10', 'donchian:20', 'donchian.upper:20', 'donchian.lower:20',
    'hv:20,1d,252',
    'change@close', 'change:3@(boll)',
    'increase:3@close', 'increase:3,-1@close',
    'ma:2@(boll.upper)',
    'close > open', 'close >= open', 'close < open', 'close == open',
    'ma:5 > ma:10',
    'ma:5 // ma:10', 'ma:5 \\ ma:10', 'ma:5 >< ma:10',
]
# Rows where volas deliberately differs from stock-pandas: stock-pandas writes an RSV of
# 0 before the first full window, volas leaves the warm-up missing (NaN) — so `rsv` is
# compared from its first full window on. (KDJ, which stock-pandas seeds with those 0s,
# is checked against an independent reference below instead.)
_PARITY_FROM = {'rsv:9': 8}


@pytest.mark.parametrize('directive', PARITY)
def test_parity_with_stock_pandas(stock, spd, directive):
    start = _PARITY_FROM.get(directive, 0)
    expected = np.asarray(spd.exec(directive, create_column=False), dtype=float)[start:]
    actual = stock.exec(directive, dtype='float64')  # a missing truth -> NaN
    assert np.isnan(actual[:start]).all()
    actual = actual[start:]
    if stock[directive].dtype == 'bool':
        # A directive boolean is missing where an operand is (the warm-up rows);
        # stock-pandas reads those rows as False — the documented divergence
        # (docs/directives.md#operators). Every row volas knows matches exactly.
        unknown = np.isnan(actual)
        assert not expected[unknown].any(), f'a missing truth where stock-pandas is True: {directive!r}'
        assert unknown.sum() < len(actual), directive
        expected, actual = expected[~unknown], actual[~unknown]
    assert np.allclose(expected, actual, rtol=1e-6, atol=1e-6, equal_nan=True), (
        f'mismatch for {directive!r}'
    )


def _kdj_reference(high, low, close, n=9, m1=3, m2=3, init=50.0):
    """通达信 KDJ: RSV = 100·(C − LLV(L, n))/(HHV(H, n) − LLV(L, n)), where before the first
    full window LLV / HHV reduce the bars available so far; K = SMA(RSV, m1, 1),
    D = SMA(K, m2, 1), both seeded with `init`; J = 3K − 2D."""
    k_prev = d_prev = init
    out = []
    for i in range(len(close)):
        lo_, hi_ = low[max(0, i - n + 1):i + 1].min(), high[max(0, i - n + 1):i + 1].max()
        rsv = (close[i] - lo_) / (hi_ - lo_) * 100.0 if abs(hi_ - lo_) > 1e-10 else 0.0
        k_prev = (m1 - 1) / m1 * k_prev + rsv / m1
        d_prev = (m2 - 1) / m2 * d_prev + k_prev / m2
        out.append((k_prev, d_prev, 3.0 * k_prev - 2.0 * d_prev))
    return np.array(out)


@pytest.mark.parametrize('directive,line,args', [
    ('kdj.k', 0, {}), ('kdj.d', 1, {}), ('kdj.j', 2, {}),
    ('kdj.k:9,3,50', 0, {}), ('kdj.j:9,3,3,50', 2, {}), ('kdj.d:5,4,2,20', 1, dict(n=5, m1=4, m2=2, init=20.0)),
])
def test_kdj_follows_the_tdx_convention_from_the_first_bar(stock, directive, line, args):
    """KDJ's RSV covers the bars available before its first full window (通达信), so %K is
    seeded from real RSVs from the very first bar — never from a warm-up placeholder."""
    h, lo, c = (np.asarray(stock[k], dtype=float) for k in ('high', 'low', 'close'))
    want = _kdj_reference(h, lo, c, **args)[:, line]
    np.testing.assert_allclose(np.asarray(stock.exec(directive), dtype=float), want, rtol=1e-12, atol=1e-9)
    # a boolean of it sees the same values
    got = np.asarray(stock.exec(f'{directive} < 0'), dtype=bool)
    np.testing.assert_array_equal(got, want < 0)


# ---------------------------------------------------------------------------
# Directive behaviour
# ---------------------------------------------------------------------------

def test_exec_returns_ndarray(stock):
    assert isinstance(stock.exec('ma:5'), np.ndarray)
    assert 'ma:5' not in stock.columns  # exec is stateless — it never caches a column


def test_ma_values():
    df = create_stock()
    ma = df['ma:2'].to_numpy()
    assert np.isnan(ma[0])
    assert ma[1] == pytest.approx((2 + 3 + 1 + 1) / 2)  # (open close based) -> use directly
    # ma:2 on close [3,4,5,6,7,8]
    close = df['close'].to_numpy()
    assert ma[1] == pytest.approx((close[0] + close[1]) / 2)
    assert ma[-1] == pytest.approx((close[-1] + close[-2]) / 2)


def test_period_larger_than_size_all_nan():
    df = create_stock()
    ma = df['ma:100'].to_numpy()
    assert np.isnan(ma).all()


def test_directive_on_column(stock):
    a = stock['ma:10@open'].to_numpy()
    b = stock.exec('ma:10@open')
    np.testing.assert_array_equal(a, b)


def test_multi_directive_frame(stock):
    sub = stock[['ma:5', 'ma:10']]
    assert sub.columns == ['ma:5', 'ma:10']
    np.testing.assert_allclose(
        sub['ma:5'].to_numpy(), stock['ma:5'].to_numpy(), equal_nan=True
    )


def test_unknown_command_raises(stock):
    with pytest.raises((ValueError, KeyError)):
        stock.exec('definitely_not_a_command:5')


def test_empty_directive_raises(stock):
    with pytest.raises((ValueError, KeyError)):
        stock.exec('')


def test_style_directive_invalid_subcommand_raises(stock):
    with pytest.raises((ValueError, KeyError)):
        stock.exec('style.sideways')


def test_bbw_matches_definition(stock):
    a = stock['bbw'].to_numpy()
    upper = stock['boll.upper'].to_numpy()
    lower = stock['boll.lower'].to_numpy()
    middle = stock['boll'].to_numpy()
    b = (upper - lower) / middle
    np.testing.assert_allclose(a, b, rtol=1e-8, equal_nan=True)
