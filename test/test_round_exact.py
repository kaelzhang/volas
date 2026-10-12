"""Rounding to decimal places gives the correctly rounded decimal result.

`round(x, k)` is the binary float nearest to the exact value of `x` rounded half-to-even
at the `k`-th decimal place — Python's `round` for floats, and for float32 the float32
nearest to that decimal. A value whose scaled form overflows (`round(1e300, 10)`) is
already a whole number of `10^-k`, so it is returned as is; a `k` past every float's
precision returns the value; a result past the float range raises, like Python's.
NumPy's `x·10^k` scaling (which pandas uses) is not exact: `2.675` rounds to `2.68`, and
`1e300` to `inf`. The same exactness holds where volas reads a scaled decimal: an Arrow
`Decimal128` / `Decimal256` imports as the float nearest to its value.
"""

from __future__ import annotations

import decimal
from decimal import Decimal

import numpy as np
import pytest

import volas

pa = pytest.importorskip('pyarrow')


def _values(rng, n):
    mant = rng.random(n) * 2 - 1
    exps = rng.integers(-330, 309, n)
    xs = list(mant * 10.0 ** exps.clip(-307, 308))
    xs += [float(f'{rng.integers(1, 10**6)}.{rng.integers(0, 10**4):04d}5') for _ in range(n)]  # near ties
    xs += [1e300, 1.7e308, -1.7e308, 5e-324, -5e-324, 2.675, 0.285, 1.0049999999999999, 0.5, 1.5, 2.5,
           -2.5, 0.125, 0.375, 2.0 ** -24, 2.0 ** 52 + 0.5, 2.0 ** 53, 1e22, 1e23, 123456789.987654321]
    return [float(x) for x in xs if np.isfinite(x)]


def _series(xs, dtype=np.float64):
    return volas.DataFrame({'x': np.asarray(xs, dtype=dtype)})['x']


def _bits_equal(a, b):
    return np.float64(a).view(np.uint64) == np.float64(b).view(np.uint64) or (np.isnan(a) and np.isnan(b))


KS = [0, 1, 2, 3, 5, 8, 10, 15, 16, 17, 22, 23, 30, 100, 300, 307, 308, 309, 320, 323, 324, 330, 400, 2**31 - 1,
      -1, -2, -5, -15, -16, -22, -23, -38, -39, -100, -300, -307, -308, -309, -400, -(2**31)]


def _python_round(x, k):
    try:
        return round(x, k)
    except OverflowError:
        return OverflowError


@pytest.mark.parametrize('k', KS)
def test_float64_round_is_pythons(k):
    xs = _values(np.random.default_rng(k % 1000 + 1), 300)
    want = [_python_round(x, k) for x in xs]
    if OverflowError in want:
        bad = [x for x, w in zip(xs, want) if w is OverflowError]
        for x in bad:
            with pytest.raises(OverflowError):
                _series([x]).round(k)
        xs = [x for x, w in zip(xs, want) if w is not OverflowError]
        want = [w for w in want if w is not OverflowError]
    got = _series(xs).round(k).to_list()
    for x, g, w in zip(xs, got, want):
        assert _bits_equal(g, w), f'round({x!r}, {k}) = {g!r}, want {w!r}'


def _nearest_f32(d: Decimal) -> np.float32:
    """The float32 nearest to the exact decimal `d`, ties to even."""
    with decimal.localcontext() as ctx:
        ctx.prec = 2000
        f = np.float32(float(d))
        cands = [f, np.nextafter(f, np.float32(np.inf)), np.nextafter(f, np.float32(-np.inf))]
        cands = [c for c in cands if np.isfinite(c)] or [f]
        dist = [abs(Decimal(float(c)) - d) for c in cands]
        best = min(dist)
        tied = [c for c, e in zip(cands, dist) if e == best]
        return min(tied, key=lambda c: int(np.float32(c).view(np.uint32)) & 1)


@pytest.mark.parametrize('k', [0, 1, 2, 3, 6, 8, 10, 11, 20, 40, 45, 46, 50, -1, -3, -10, -38, -39, -40])
def test_float32_round_is_the_nearest_float32_of_the_decimal(k):
    rng = np.random.default_rng(100 + k)
    xs = np.r_[(rng.random(400) * 2 - 1) * 10.0 ** rng.integers(-45, 38, 400),
               [2.675, 0.125, 0.375, 1.5, 2.5, 3.4e38, 1e-45, 16777216.5]].astype(np.float32)
    xs = xs[np.isfinite(xs)]
    got = _series(xs, np.float32).round(k).to_numpy()
    assert got.dtype == np.float32
    with decimal.localcontext() as ctx:
        ctx.prec = 2000
        ctx.rounding = decimal.ROUND_HALF_EVEN
        for x, g in zip(xs, got):
            d = Decimal(float(x)).quantize(Decimal(1).scaleb(-k)) if Decimal(float(x)) else Decimal(0)
            w = _nearest_f32(d)
            if d == 0:
                w = np.copysign(np.float32(0), x)
            assert np.float32(g).view(np.uint32) == np.float32(w).view(np.uint32), f'round({x!r}, {k}) = {g!r}, want {w!r}'


def test_round_overflow_raises_and_leaves_nan_and_inf():
    with pytest.raises(OverflowError):
        _series([1.7e308]).round(-308)
    # float32's largest value to a multiple of 10^35 is 3.403e38, past float32
    with pytest.raises(OverflowError):
        _series(np.array([np.finfo(np.float32).max]), np.float32).round(-35)
    got = _series([np.nan, np.inf, -np.inf, -0.0, 0.0]).round(2).to_list()
    assert np.isnan(got[0]) and got[1:3] == [np.inf, -np.inf]
    assert [np.signbit(v) for v in got[3:]] == [True, False]


@pytest.mark.parametrize('kind,precision', [('decimal128', 38), ('decimal256', 76)])
def test_arrow_decimals_import_as_the_nearest_float(kind, precision):
    rng = np.random.default_rng(precision)
    with decimal.localcontext() as ctx:
        ctx.prec = 200
        for scale in (0, 2, 9, 18, 23, 30, precision - 1):
            values = []
            for _ in range(200):
                digits = ''.join(map(str, rng.integers(0, 10, int(rng.integers(1, precision + 1)))))
                values.append(Decimal(('-' if rng.random() < 0.5 else '') + digits).scaleb(-scale))
            values += [Decimal('2.675'), Decimal('0.1'), Decimal(0)]
            values = [v.quantize(Decimal(1).scaleb(-scale)) for v in values]
            arr = pa.array(values, type=getattr(pa, kind)(precision, scale))
            got = volas.Series.from_arrow(arr).to_list()
            want = [float(v) for v in arr.to_pylist()]
            for v, g, w in zip(arr.to_pylist(), got, want):
                assert _bits_equal(g, w), f'{kind}({precision}, {scale}) {v}: {g!r} != {w!r}'
