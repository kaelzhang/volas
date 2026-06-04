"""pandas interop + to_csv (audit SP-6 / PD-19)."""

import numpy as np
import pandas as pd
import volas
from volas import DataFrame


def test_to_pandas():
    df = DataFrame({'a': [1., 2., 3.], 'b': np.array([10, 20, 30], dtype=np.int64)})
    pdf = df.to_pandas()
    assert isinstance(pdf, pd.DataFrame)
    assert pdf['a'].tolist() == [1., 2., 3.]
    assert pdf['b'].tolist() == [10, 20, 30]


def test_from_pandas_roundtrip():
    pdf = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    df = volas.from_pandas(pdf)
    assert df.columns == ['a', 'b']
    assert df['a'].to_list() == [1., 2.]


def test_from_pandas_datetime_index():
    pdf = pd.DataFrame({'close': [1., 2.]},
                       index=pd.to_datetime(['2020-01-01', '2020-01-02']))
    df = volas.from_pandas(pdf)
    assert str(df.index.dtype) == 'datetime64[ns]'
    assert df['close'].to_list() == [1., 2.]


def test_to_csv_string_and_file(tmp_path):
    df = DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    s = df.to_csv()
    assert s.splitlines()[0] == 'index,a,b'
    p = tmp_path / "o.csv"
    assert df.to_csv(str(p), index=False) is None
    lines = p.read_text().splitlines()
    assert lines[0] == 'a,b'
    df2 = volas.read_csv(str(p))            # round-trips
    assert df2.columns == ['a', 'b']
