"""Audit existing-API fixes: EX-3/4/5/7/12/13/14."""

import numpy as np
import volas
from volas import DataFrame, Row


def test_int_list_infers_int64():            # EX-7
    assert DataFrame({'i': [1, 2, 3]})['i'].dtype == 'int64'
    assert DataFrame({'f': [1.0, 2.0, 3.0]})['f'].dtype == 'float64'
    assert DataFrame({'b': [True, False, True]})['b'].dtype == 'bool'
    assert DataFrame({'m': [1, 2.5, 3]})['m'].dtype == 'float64'   # mixed -> float


def test_strict_equals():                    # EX-5
    a = DataFrame({'x': [1.0, 0.0]})['x']
    b = DataFrame({'x': np.array([1, 0], dtype=np.int64)})['x']
    assert not a.equals(b)                   # same values, different dtype -> not equal
    assert a.equals(DataFrame({'x': [1.0, 0.0]})['x'])


def test_exec_is_stateless():                # EX-3
    df = DataFrame({'open': [1., 2., 3., 4., 5.], 'close': [2., 3., 4., 5., 6.]})
    df.exec('ma:2')
    assert 'ma:2' not in df.columns          # exec never caches — it is stateless
    df['ma:2']                               # df[directive] is the caching path
    assert 'ma:2' in df.columns


def test_drop_columns_axis1():               # EX-4
    df = DataFrame({'a': [1., 2.], 'b': [3., 4.], 'c': [5., 6.]})
    assert df.drop(['b'], axis=1).columns == ['a', 'c']
    assert len(df.drop([0], axis=0)) == 1    # axis=0 still drops rows


def test_row_to_dict_and_export():           # EX-13, EX-14
    df = DataFrame({'a': [1.0, 2.0], 's': ['x', 'y']})
    row = df.iloc[1]
    assert isinstance(row, Row)
    assert row.to_dict() == {'a': 2.0, 's': 'y'}


def test_read_csv_header(tmp_path):          # EX-12
    p = tmp_path / "n.csv"
    p.write_text("1,2\n3,4\n")
    assert volas.read_csv(str(p), header=None).columns == ['0', '1']   # no header
    assert volas.read_csv(str(p)).columns == ['1', '2']                # first row as header
