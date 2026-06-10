"""P2-03: DataFrame.equals is value-semantics (a RangeIndex equals the same
integer labels materialized). P2-04: to_datetime treats an empty/blank string as
missing (NaT), consistent with read_csv(parse_dates)."""

import numpy as np
import pandas as pd
import pytest
import volas
from volas import DataFrame


# --- P2-03: equals by index label value, not representation kind --------------

def test_equals_rangeindex_vs_materialized_int():
    df = DataFrame({'a': [1, 2, 3]})
    rt = volas.from_pandas(df.to_pandas())   # round-trip materializes RangeIndex -> Int64
    assert df.equals(rt)


def test_equals_distinguishes_different_labels():
    # same column values but different index labels -> NOT equal (the fix is value
    # semantics, not "ignore the index")
    a = DataFrame({'a': [1, 2, 3], 'i': [0, 1, 2]}).set_index('i')
    b = DataFrame({'a': [1, 2, 3], 'i': [5, 6, 7]}).set_index('i')
    assert not a.equals(b)


def test_equals_still_false_on_different_values():
    a = DataFrame({'a': [1, 2, 3]})
    c = DataFrame({'a': [1, 2, 9]})
    assert not a.equals(c)


# --- P2-04: empty / blank string -> NaT in to_datetime ------------------------

def test_to_datetime_empty_string_is_nat():
    out = volas.to_datetime(['2021-01-01', ''])
    assert out.isna().to_list() == [False, True]


def test_to_datetime_blank_string_is_nat():
    out = volas.to_datetime(['2021-01-01', '   '])
    assert out.isna().to_list() == [False, True]


def test_to_datetime_empty_with_format_is_nat():
    out = volas.to_datetime(['2021-01-01', ''], format='%Y-%m-%d')
    assert out.isna().to_list() == [False, True]


def test_to_datetime_nonempty_invalid_still_errors():
    with pytest.raises(Exception):
        volas.to_datetime(['2021-01-01', 'not-a-date'])


def test_equals_datetime_and_str_index_value_semantics():
    # datetime- and string-labelled indexes compare by value too (label_eq arms)
    a = DataFrame({'d': np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[ns]'),
                   'v': [1.0, 2.0]}).set_index('d')
    b = DataFrame({'d': np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[ns]'),
                   'v': [1.0, 2.0]}).set_index('d')
    assert a.equals(b)
    c = DataFrame({'d': np.array(['2021-01-01', '2021-01-09'], dtype='datetime64[ns]'),
                   'v': [1.0, 2.0]}).set_index('d')
    assert not a.equals(c)                       # different datetime labels
    sa = DataFrame({'k': ['x', 'y'], 'v': [1.0, 2.0]}).set_index('k')
    assert sa.equals(DataFrame({'k': ['x', 'y'], 'v': [1.0, 2.0]}).set_index('k'))
    assert not sa.equals(DataFrame({'k': ['x', 'z'], 'v': [1.0, 2.0]}).set_index('k'))


def test_tz_aware_nat_renders_nat():
    # a tz-aware DatetimeIndex containing NaT formats as 'NaT' (format_ns_tz guard),
    # not a 1677 civil date
    ps = pd.DataFrame({'v': [1, 2]},
                      index=pd.to_datetime(['2021-01-01', 'NaT']).tz_localize('America/New_York'))
    vf = volas.from_pandas(ps)
    assert 'NaT' in repr(vf)
