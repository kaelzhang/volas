"""A directive's lookback is where its kernel publishes its first value.

``directive_lookback`` is operational metadata: ``ready``, windowed margins and the
resume paths trust it, so a lookback that disagrees with the kernel lets a cached
column be read (or continued) inside its warm-up. This property test checks every
output line of the directive language — its argument domains probed from the
validator (``test.harness.inventory``) — over several argument tuples per line,
including tuples whose periods are out of their usual order, against the first
non-missing row the kernel actually publishes on clean data.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

import volas
from test.harness.inventory import executable_lines, sample_value

N = 400


def _frame() -> volas.DataFrame:
    rng = np.random.default_rng(20261012)
    c = 100 + np.cumsum(rng.standard_normal(N))
    o = c + 0.4 * rng.standard_normal(N)
    return volas.DataFrame({'open': o, 'high': np.maximum(o, c) + rng.uniform(0.1, 1.5, N),
                            'low': np.minimum(o, c) - rng.uniform(0.1, 1.5, N), 'close': c,
                            'volume': rng.uniform(1, 100, N)})


FRAME = _frame()


def _tuples(line, count=8):
    """Deterministic argument tuples for ``line``: small integer periods (1 included,
    in every order), every choice, in-domain floats."""
    rng = np.random.default_rng(zlib.crc32(line.head.encode()))
    out = []
    for k in range(count):
        vals = []
        for a in line.args:
            if a.kind == 'int':
                lo, hi = int(a.lo), int(min(a.hi, a.lo + 24))
                vals.append(str(lo if k == 0 else int(rng.integers(lo, hi + 1))))
            else:
                vals.append(sample_value(a, k))
        out.append(f"{line.head}:{','.join(vals)}" if vals else line.head)
    return list(dict.fromkeys(out))


# Lines whose first value is not their warm-up, by documented design:
# * a pivot is an event — its first value is the first confirmed pivot, at or after
#   the lookback (the earliest row one can be confirmed);
# * KDJ smooths an RSV taken over the bars available so far (the 通达信 convention), so
#   it publishes from bar 0; its lookback is the smoothing's warm-up, after that;
# * `increase` / `repeat` are Kleene ANDs over their window: one False step already
#   decides a False before the window is complete, so a value may come before the lookback.
EVENT = {'pivothigh', 'pivotlow'}
EARLY = {'kdj.k', 'kdj.d', 'kdj.j', 'increase', 'repeat'}
# Open defects of another class this sweep surfaces (not lookback metadata):
OPEN_FORMS = {
    # a period of 1 is accepted but degenerate: log10(1) = 0 divides CHOP, and a one-bar
    # standard deviation leaves the squeeze undefined
    'chop:1': 'period 1 is degenerate (log10(1) = 0)',
    'ttm_squeeze:1,-5.0,-5.0': 'period 1 is degenerate (one-bar deviation)',
}


def _case(line, form):
    reason = OPEN_FORMS.get(form)
    marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
    return pytest.param(line, form, id=form, marks=marks)


CASES = [_case(line, form) for line in executable_lines() if line.args for form in _tuples(line)]


@pytest.mark.parametrize('line,form', CASES)
def test_lookback_is_the_first_published_row(line, form):
    directive = form + line.series_suffix
    try:
        values = FRAME.exec(directive, dtype='float64')  # NA -> NaN, booleans -> 0 / 1
    except volas.DirectiveError:
        pytest.skip(f'{directive}: not executable with these arguments')
    missing = np.isnan(values)
    first = int(np.argmin(missing)) if not missing.all() else N
    lookback = volas.directive_lookback(directive)
    if line.head in EVENT:
        assert first >= lookback, f'{directive}: lookback {lookback}, a pivot at row {first}'
    elif line.head in EARLY:
        assert first <= lookback, f'{directive}: lookback {lookback}, first value at row {first}'
    else:
        assert first == min(lookback, N), f'{directive}: lookback {lookback}, first value at row {first}'
