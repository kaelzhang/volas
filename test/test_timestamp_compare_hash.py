"""`Timestamp` equality is hash-consistent: ``a == b`` implies ``hash(a) == hash(b)``.

A Timestamp equals only another instant (a Timestamp, a ``datetime`` /
``pd.Timestamp``, an ``np.datetime64``) of the same awareness; a string or an
integer is never equal (pandas' rule), so a Timestamp and a value it is equal to
always land in the same dict / set bucket. Selection by string (``df.loc['...']``,
index-vs-string comparison) is a separate, parsing surface and is unaffected.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import volas

# whole-second instants inside the datetime / datetime64[ns] ranges, plus a
# sub-microsecond part so both hashing regimes are exercised
_NS = st.integers(-(2 ** 62), 2 ** 62)
_ZONES = st.sampled_from([None, 'UTC', '+08:00', 'America/New_York'])


def _operands(ns, tz):
    ts = volas.Timestamp(np.datetime64(ns, 'ns'))
    ts = ts if tz is None else ts.tz_localize('UTC').tz_convert(tz)
    p = pd.Timestamp(ns, tz=None if tz is None else 'UTC')
    p = p if tz is None else p.tz_convert(tz)
    out = [ts, p, str(p), ns, np.int64(ns)]
    if ns % 1_000 == 0:
        # a sub-microsecond np.datetime64 hashes by NumPy's own rule, which pandas'
        # Timestamp (and so volas) does not share — the one documented exception
        out += [np.datetime64(ns, 'ns'), p.to_pydatetime()]
    return out


@settings(max_examples=300, deadline=None)
@given(_NS, _ZONES, _NS, _ZONES)
def test_equal_implies_equal_hash(a, tza, b, tzb):
    left = volas.Timestamp(np.datetime64(a, 'ns'))
    left = left if tza is None else left.tz_localize('UTC').tz_convert(tza)
    for rhs in _operands(a, tza) + _operands(b, tzb):
        if left == rhs:
            assert hash(left) == hash(rhs), (left, rhs)


@pytest.mark.parametrize('operand', ['2021-06-15 09:30:00', 'abc', 1623749400000000000, 5,
                                     np.int64(1623749400000000000), True])
def test_strings_and_integers_are_not_instants(operand):
    ts = volas.Timestamp('2021-06-15 09:30:00')
    assert (ts == operand) is False and (operand == ts) is False
    assert (ts != operand) is True
    for f in (lambda: ts < operand, lambda: ts <= operand, lambda: ts > operand,
              lambda: ts >= operand, lambda: operand < ts):
        with pytest.raises(TypeError):
            f()


def test_string_selection_still_parses():
    # selection and index comparison parse strings — only the scalar rule changed
    df = volas.DataFrame({'c': [1.0, 2.0]},
                         index=np.array(['2024-01-01', '2024-01-02'], 'datetime64[ns]'))
    assert df.loc['2024-01-02', 'c'] == 2.0
    assert df.index[1] == volas.Timestamp('2024-01-02')
    assert volas.Timestamp('2024-01-02') == df.index[1]
    assert volas.Timestamp('2024-01-02') == dt.datetime(2024, 1, 2)
