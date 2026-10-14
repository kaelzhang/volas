"""Every directive, cached ALONE, continues across every contiguous selection exactly as
a from-scratch computation of the same accepted rows (docs/dataframe-api.md "Derived
frames"): a tail's append and re-sent row, a middle selection, a selection of a
selection, a one-row tail's direct re-send, a missing row among the rows before the
selection, and a whole-column write of another dtype. Caching one directive at a time
keeps another column's history from standing in for a missing one."""

import numpy as np
import pytest

import volas
from volas import DataFrame

from .harness.inventory import executable_lines

N = 60
rng = np.random.default_rng(23)
_c = 100 + np.cumsum(rng.normal(0, 1, N + 4))
COLS = {'open': _c + rng.normal(0, .3, N + 4), 'high': _c + 1 + rng.random(N + 4),
        'low': _c - 1 - rng.random(N + 4), 'close': _c,
        'volume': rng.integers(1, 1000, N + 4).astype(float)}
LINES = [ln.form + ln.series_suffix for ln in executable_lines()]
LINES = [d for d in LINES if volas.directive_lookback(d) <= 30]


def same(got, want):
    def cell(v):
        return None if v is volas.NA or (isinstance(v, float) and v != v) else v
    return [cell(v) for v in np.asarray(got, dtype=object)] == [cell(v) for v in np.asarray(want, dtype=object)]


def frame(cols, a, b):
    return DataFrame({k: v[a:b] for k, v in cols.items()}, index=list(range(a, b)))


def cached(d, cols=COLS):
    df = frame(cols, 0, N)
    df[d]
    return df


def full(d, cols, b, extra=None):
    rows = {k: v[:b] if extra is None else np.concatenate([v[:b], extra[k]]) for k, v in cols.items()}
    return DataFrame(rows).exec(d, dtype='object')


def bar(cols, i, scale=1.0):
    return {k: v[i:i + 1] * scale for k, v in cols.items()}


def read(df, d):
    return df[volas.directive_stringify(d)].to_numpy(dtype='object')


@pytest.mark.parametrize('d', LINES)
def test_every_directive_continues_across_every_contiguous_selection(d):
    fails = []

    def check(label, got, want):
        if not same(got, want):
            fails.append(label)

    # a tail: an append, then a re-sent last row
    tail = cached(d).iloc[N - 3:]
    new = bar(COLS, N, 1.01)
    tail.append(DataFrame(new, index=[N]))
    check('tail append', read(tail, d), full(d, COLS, N, new)[N - 3:])
    resent = bar(COLS, N, 0.99)
    tail.append(DataFrame(resent, index=[N]))
    check('tail resend', read(tail, d), full(d, COLS, N, resent)[N - 3:])
    # a middle selection, and the same rows as a selection of a selection
    for label, sub in (('middle', cached(d).iloc[17:41]), ('nested', cached(d).iloc[10:].iloc[7:31])):
        new = bar(COLS, 41, 1.02)
        sub.append(DataFrame(new, index=[41]))
        check(label, read(sub, d), full(d, COLS, 41, new)[17:])
    # a one-row tail re-sending its row directly
    one = cached(d).tail(1)
    resent = bar(COLS, N - 1, 1.03)
    one.append(DataFrame(resent, index=[N - 1]))
    check('one-row resend', read(one, d), full(d, COLS, N - 1, resent)[N - 1:])
    # a missing row among the rows before the selection
    gappy = {k: v.copy() for k, v in COLS.items()}
    for v in gappy.values():
        v[N - 3] = np.nan
    short = cached(d, gappy).tail(2)
    new = bar(gappy, N)
    short.append(DataFrame(new, index=[N]))
    check('missing in lead', read(short, d), full(d, gappy, N, new)[N - 2:])
    # a whole column replaced by values of another dtype (float32 holds them exactly)
    typed = cached(d).iloc[N - 4:]
    values = np.round(COLS['close'][N - 4:N] * 4) / 4
    typed['close'] = values.astype('float32')
    rows = {k: v[:N].copy() for k, v in COLS.items()}
    rows['close'][N - 4:] = values
    check('typed write', read(typed, d), DataFrame(rows).exec(d, dtype='object')[N - 4:])
    assert not fails, (d, fails)
