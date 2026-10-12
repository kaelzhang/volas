"""The reference oracle itself follows the documented missing-row rule.

docs/numerics.md "Missing and non-finite inputs": a recursive indicator skips a row where
any of its inputs is NaN, so its output equals the indicator run on the series with
those rows deleted, NaN put back at their positions. ``oracle_reference`` applies that
rule at its input boundary; these cases pin it on hand-computed series, then check volas
agrees with the oracle on the same missing rows.
"""

import numpy as np
import pytest

from volas import DataFrame

from . import oracle_reference as oref

nan = np.nan


def test_ema_skips_missing_rows():
    # Rows deleted -> [1, 3, 5]; seed SMA(1, 3) = 2, then 2/3*5 + 1/3*2 = 4.
    got = oref._ema([1.0, nan, 3.0, 5.0], 2).to_numpy()
    np.testing.assert_allclose(got, [nan, nan, 2.0, 4.0], equal_nan=True)


def test_ema_skips_leading_and_inner_missing_rows():
    # Rows deleted -> [2, 4, 6]; seed SMA(2, 4) = 3, then 2/3*6 + 1/3*3 = 5.
    got = oref._ema([nan, 2.0, 4.0, nan, 6.0], 2).to_numpy()
    np.testing.assert_allclose(got, [nan, nan, 3.0, nan, 5.0], equal_nan=True)


def test_ema_without_enough_rows_is_all_nan():
    got = oref._ema([1.0, nan, 3.0], 3).to_numpy()
    assert np.isnan(got).all()


@pytest.mark.parametrize('close,volume', [
    ([1.0, 2.0, nan, 3.0, 5.0], [1.0, 1.0, 1.0, 1.0, 1.0]),  # missing close
    ([1.0, 2.0, 4.0, 3.0, 5.0], [1.0, 1.0, nan, 1.0, 1.0]),  # missing volume
])
def test_efi_deletes_incomplete_rows_before_differencing(close, volume):
    # Rows deleted -> close [1, 2, 3, 5], volume 1: raw force [nan, 1, 1, 2];
    # EMA2 seeds SMA(1, 1) = 1, then 2/3*2 + 1/3*1 = 5/3.
    got = oref.efi(None, None, None, close, volume, 2).to_numpy()
    np.testing.assert_allclose(got, [nan, nan, nan, 1.0, 5.0 / 3.0], equal_nan=True)


@pytest.mark.parametrize('directive,fn', [
    ('ema:2', lambda o, h, lo, c, v: oref._ema(c, 2)),
    ('efi:2', lambda o, h, lo, c, v: oref.efi(o, h, lo, c, v, 2)),
])
def test_volas_matches_oracle_on_missing_rows(directive, fn):
    cols = {
        'open': [1.0] * 6,
        'high': [2.0] * 6,
        'low': [0.0] * 6,
        'close': [1.0, 2.0, nan, 3.0, 5.0, 4.0],
        'volume': [1.0, 2.0, 1.0, nan, 3.0, 2.0],
    }
    got = DataFrame(cols)[directive].to_numpy()
    want = fn(*(cols[k] for k in ('open', 'high', 'low', 'close', 'volume'))).to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)
