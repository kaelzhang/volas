"""Lossless import metadata and pandas-compatible text boundaries."""

import csv
import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from volas import DataFrame


@pytest.mark.parametrize('index', [pd.RangeIndex(2, 5, name='n'), pd.RangeIndex(3, name='n'), pd.RangeIndex(2, 8, 2), pd.RangeIndex(5, 2, -1), pd.RangeIndex(0)])
def test_from_pandas_preserves_range_labels_and_name(index):
    pdf = pd.DataFrame({'x': np.arange(len(index))}, index=index)
    out = DataFrame.from_pandas(pdf).to_pandas()
    assert out.index.tolist() == index.tolist()
    assert out.index.name == index.name


@pytest.mark.parametrize('columns', [[1, '1'], ['x', 'x']])
def test_from_pandas_rejects_stringified_column_collisions(columns):
    with pytest.raises(ValueError, match='duplicate.*column.*name'):
        DataFrame.from_pandas(pd.DataFrame([[1, 2]], columns=columns))


@pytest.mark.parametrize('sep', [',', ';'])
def test_csv_escapes_index_name_labels_and_headers(sep):
    pdf = pd.DataFrame({'x,;\n"': [1, 2]}, index=pd.Index(['a,;1\nb', 'q"\r'], name='n,;\n"'))
    df = DataFrame.from_pandas(pdf)
    text = df.to_csv(sep=sep)
    assert list(csv.reader(io.StringIO(text), delimiter=sep)) == [
        [pdf.index.name, pdf.columns[0]], ['a,;1\nb', '1'], ['q"\r', '2']]
    assert text == pdf.to_csv(sep=sep)


@pytest.mark.parametrize('values', [['a', None], [None], []])
def test_numpy_nullable_string_uses_pd_na(values):
    df = DataFrame.from_pandas(pd.DataFrame({'s': pd.Series(values, dtype='string')}))
    out = df.to_pandas(dtype_backend='numpy_nullable')['s']
    assert out.dtype == pd.StringDtype()
    assert out.dtype.na_value is pd.NA
    if None in values:
        assert out.iloc[values.index(None)] is pd.NA


@pytest.mark.parametrize('dtype', ['Int32', 'int32[pyarrow]'])
@pytest.mark.parametrize('values', [[1, None, 3], [None], []])
def test_pandas_integer_extension_preserves_int32_width(dtype, values):
    df = DataFrame.from_pandas(pd.DataFrame({'x': pd.Series(values, dtype=dtype)}))
    assert str(df['x'].dtype) == 'int32'
    assert df.to_pandas(dtype_backend='numpy_nullable')['x'].dtype == pd.Int32Dtype()


@pytest.mark.parametrize('index', [pd.RangeIndex(3), pd.RangeIndex(2, 5, name='n'), pd.Index(['a', 'b', 'c'], name='n')])
def test_pandas_index_only_frame_preserves_row_count(index):
    df = DataFrame.from_pandas(pd.DataFrame(index=index))
    assert df.shape == (3, 0)
    assert df.to_numpy().shape == (3, 0)
    assert df.to_pandas().index.tolist() == index.tolist()


@pytest.mark.parametrize('chunks', [False, True])
def test_arrow_zero_column_frame_preserves_row_count(chunks):
    table = pa.table({'x': [1, 2, 3]}).select([])
    if chunks:
        table = pa.concat_tables([table, table])
    df = DataFrame.from_arrow(table)
    assert df.shape == (table.num_rows, 0)
    assert df.to_arrow().shape == table.shape


@pytest.mark.parametrize('min_rows', [5, 100, None, 0, 1])
@pytest.mark.parametrize('max_rows', [2, 1, 0, 3])
def test_to_string_clamps_min_rows_without_duplicate_rows(max_rows, min_rows):
    pdf = pd.DataFrame({'x': [0, 1, 2]})
    assert DataFrame.from_pandas(pdf).to_string(max_rows=max_rows, min_rows=min_rows) == pdf.to_string(max_rows=max_rows, min_rows=min_rows)


@pytest.mark.parametrize('fmt', ['%.2g', '%g', '%.0g', '%.5g', '%.2e', '%e', '%.2f'])
def test_printf_float_formats_use_significant_digits_and_exponents(fmt):
    values = [1234.567, 0.00001234, 0.0001234, 9.999, 999.9, 1.0, -0.0, 0.0, float('inf'), -float('inf')]
    df = DataFrame({'x': values})
    assert df.to_csv(index=False, header=False, float_format=fmt) == ''.join((fmt % x) + '\n' for x in values)
    rendered = df.to_string(index=False, header=False, float_format=fmt)
    assert [line.strip() for line in rendered.splitlines()] == [fmt % x for x in values]
    assert [line.split(maxsplit=1)[1].strip() for line in df['x'].to_string(float_format=fmt).splitlines()] == [fmt % x for x in values]


def test_unsigned_pandas_extension_still_widens_losslessly():
    pdf = pd.DataFrame({'x': pd.Series([2**32 - 1, None], dtype='uint32[pyarrow]')})
    out = DataFrame.from_pandas(pdf).to_pandas(dtype_backend='numpy_nullable')
    assert out['x'].dtype == pd.Int64Dtype()
    assert out['x'].iloc[0] == 2**32 - 1


@pytest.mark.parametrize('max_rows', [0, 1, 2])
def test_series_truncation_uses_the_same_row_budget(max_rows):
    values = [0, 1, 2]
    assert DataFrame({'x': values})['x'].to_string(max_rows=max_rows) == pd.Series(values).to_string(max_rows=max_rows)
