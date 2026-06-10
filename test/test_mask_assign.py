"""Boolean-mask assignment contract — df[mask] = scalar / df[bool_frame] = scalar.

The mask-assignment surface routes through the same `scalar_to_column` + `scatter`
primitive as the `.loc/.iloc/.at/.iat` indexers, so its dtype/validity rules are
identical:
- the condition frame must be boolean (a numeric / string mask is rejected, the
  same contract as `DataFrame.where`);
- a string / datetime-string fill targets str / datetime columns (Decision 2:
  typed scalar assignment is uniform across every surface);
- a value that cannot fit a *selected* column is a hard error and nothing is
  written (atomic), but a column with no selected cells is left untouched;
- `None` / `NaN` marks the selected cells NA, keeping each column's dtype.
"""

import numpy as np
import pytest
import volas as vs
from volas import DataFrame


# --- P1-02: the condition frame must be boolean -------------------------------

def test_cell_mask_rejects_numeric_condition_frame():
    df = DataFrame({'a': [1, 2, 3]})
    with pytest.raises(TypeError):
        df[DataFrame({'a': [1, 0, 1]})] = 0          # numeric mask -> reject
    assert df['a'].to_list() == [1, 2, 3]            # unchanged (atomic)


def test_cell_mask_rejects_string_condition_frame():
    df = DataFrame({'a': [1, 2]})
    with pytest.raises(TypeError):
        df[DataFrame({'a': ['x', '']})] = 0          # string mask -> reject
    assert df['a'].to_list() == [1, 2]


def test_cell_mask_accepts_bool_condition_frame():
    df = DataFrame({'a': [1, 2, 3]})
    df[DataFrame({'a': [False, True, True]})] = 0    # valid bool frame still works
    assert df['a'].to_list() == [1, 0, 0]


# --- P1-03 + Decision 2: typed string / datetime fill -------------------------

def test_cell_mask_fills_string_holes():
    df = DataFrame({'a': ['x', None, 'z']})
    df[df.isna()] = 'q'
    assert df['a'].to_list() == ['x', 'q', 'z']


def test_row_mask_fills_string_column():
    df = DataFrame({'a': ['x', 'y', 'z']})
    df[np.array([False, True, False])] = 'q'
    assert df['a'].to_list() == ['x', 'q', 'z']


def test_cell_mask_fills_datetime_holes_from_string():
    df = DataFrame({'a': np.array(['2021-01-01', 'NaT', '2021-01-03'], dtype='datetime64[ns]')})
    df[df.isna()] = '2021-01-02'
    got = df['a'].to_list()
    assert got[1] == np.datetime64('2021-01-02') and df.isna()['a'].to_list() == [False, False, False]


# --- mixed-frame atomicity ----------------------------------------------------

def test_string_fill_leaves_unselected_numeric_columns_untouched():
    # the numeric column has no NA, so isna() selects nothing in it -> untouched;
    # the str column's hole is filled. No error, partial frame is consistent.
    df = DataFrame({'s': ['x', None, 'z'], 'n': [1, 2, 3]})
    df[df.isna()] = 'q'
    assert df['s'].to_list() == ['x', 'q', 'z']
    assert df['n'].to_list() == [1, 2, 3] and df['n'].dtype == 'int64'


def test_string_fill_into_selected_numeric_cell_is_atomic_error():
    # a numeric column DOES have a selected (NA) cell -> a string cannot fit it,
    # so the whole assignment raises and nothing is written (atomic).
    df = DataFrame({'s': ['x', None, 'z'], 'n': [1, None, 3]})
    with pytest.raises(TypeError):
        df[df.isna()] = 'q'
    assert df['s'].to_list() == ['x', vs.NA, 'z']    # untouched
    assert df['n'].dtype == 'int64'


# --- None / NaN marks NA uniformly --------------------------------------------

def test_row_mask_none_marks_na_keeping_dtype():
    df = DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
    df[np.array([False, True, False])] = None
    assert df['a'].dtype == 'int64' and df['b'].dtype == 'int64'
    assert df.isna()['a'].to_list() == [False, True, False]
    assert df.isna()['b'].to_list() == [False, True, False]


def test_cell_mask_nan_marks_na_keeping_int():
    df = DataFrame({'a': [1, 2, 3]})
    df[DataFrame({'a': [False, True, True]})] = float('nan')
    assert df['a'].dtype == 'int64' and df.isna()['a'].to_list() == [False, True, True]
