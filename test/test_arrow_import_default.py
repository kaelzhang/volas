"""Matching Arrow buffers are borrowed by every default import entry point."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from volas import DataFrame, Series


class Stream:
    def __init__(self, table):
        self.table = table

    def __arrow_c_stream__(self):
        return self.table.to_reader().__arrow_c_stream__()


@pytest.mark.parametrize('kind, borrowed', [('series', True), ('frame', True), ('stream', True),
                                          ('column', False), ('pandas_column', False), ('pandas', False)],
                         ids=['series', 'frame', 'stream', 'column', 'pandas_column', 'pandas'])
def test_arrow_default_borrows_buffer(kind, borrowed):
    source = np.arange(30., dtype='float64')
    array = pa.array(source)
    table = pa.table({'close': array})
    imports = {
        'series': lambda: Series.from_arrow(array),
        'frame': lambda: DataFrame.from_arrow(table)['close'],
        'stream': lambda: DataFrame.from_arrow(Stream(table))['close'],
        # Ordinary constructors convert sequences; their copying contract is separate
        # from the Arrow C Data / Stream import protocol.
        'column': lambda: DataFrame({'close': array})['close'],
        'pandas_column': lambda: DataFrame({'close': pd.Series(array, dtype=pd.ArrowDtype(array.type))})['close'],
        'pandas': lambda: DataFrame.from_pandas(table.to_pandas(types_mapper=pd.ArrowDtype))['close'],
    }
    got = imports[kind]()
    shares = got.to_arrow().buffers()[1].address == array.buffers()[1].address
    assert shares is borrowed
    np.testing.assert_array_equal(got.to_numpy(), source)


def test_arrow_signatures_publish_borrow_default():
    import inspect
    for method in [DataFrame.from_arrow, Series.from_arrow]:
        assert inspect.signature(method).parameters['copy'].default is False
