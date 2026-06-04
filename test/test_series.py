"""volas Series arithmetic / reductions.

The pandas Series-operation subset volas implements: ``+ - * /`` (and their
reflected forms), ``equals``, ``mean``. Verified against NumPy on real data.
"""

from pathlib import Path

import numpy as np
import pytest

import volas

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


@pytest.fixture
def stock():
    return volas.read_csv(TENCENT, parse_dates=['time_key'], index_col='time_key')


def test_add_scalar(stock):
    s = stock['close']
    np.testing.assert_allclose((s + 1.0).to_numpy(), s.to_numpy() + 1.0)


def test_radd_scalar(stock):
    s = stock['close']
    np.testing.assert_allclose((1.0 + s).to_numpy(), s.to_numpy() + 1.0)


def test_sub_series(stock):
    hl = stock['high'] - stock['low']
    np.testing.assert_allclose(
        hl.to_numpy(), stock['high'].to_numpy() - stock['low'].to_numpy()
    )


def test_rsub_scalar(stock):
    s = stock['close']
    np.testing.assert_allclose((100.0 - s).to_numpy(), 100.0 - s.to_numpy())


def test_mul_and_div_scalar(stock):
    s = stock['close']
    np.testing.assert_allclose((s * 2.0).to_numpy(), s.to_numpy() * 2.0)
    np.testing.assert_allclose((s / 2.0).to_numpy(), s.to_numpy() / 2.0)


def test_series_times_series(stock):
    r = stock['close'] * stock['open']
    np.testing.assert_allclose(
        r.to_numpy(), stock['close'].to_numpy() * stock['open'].to_numpy()
    )


def test_equals(stock):
    assert stock['close'].equals(stock['close'])
    assert not stock['close'].equals(stock['open'])


def test_mean(stock):
    s = stock['close']
    assert s.mean() == pytest.approx(np.nanmean(s.to_numpy()))
