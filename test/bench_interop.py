"""Arrow and DLPack boundary costs; source construction stays outside timing."""
import numpy as np
import pandas as pd
import pyarrow as pa

from test.bench_workloads import interop as volas_interop

OPS = ('frame_from_arrow', 'frame_to_arrow', 'series_from_arrow', 'series_to_arrow',
       'dlpack_import', 'dlpack_export')


def build(length, polars):
    # Five dense float64 columns pin the incident's 1M x 5 workload.
    values = np.arange(length, dtype='float64')
    columns = {name: values.copy() for name in ('open', 'high', 'low', 'close', 'volume')}
    table = pa.table(columns)
    array = table['close'].chunk(0)
    pdf = table.to_pandas()
    ps = pd.Series(values)
    reg = {
        'volas': volas_interop(length),
        'pandas': {
            'frame_from_arrow': table.to_pandas,
            'frame_to_arrow': lambda: pa.Table.from_pandas(pdf, preserve_index=False),
            'series_from_arrow': array.to_pandas,
            'series_to_arrow': lambda: pa.Array.from_pandas(ps),
            'dlpack_import': lambda: pd.DataFrame({'close': np.from_dlpack(values)}),
        },
    }
    if hasattr(ps, '__dlpack__'):
        reg['pandas']['dlpack_export'] = lambda: np.from_dlpack(ps)
    if polars is not None:
        pf, pseries = polars.from_arrow(table), polars.from_arrow(array)
        reg['polars'] = {
            'frame_from_arrow': lambda: polars.from_arrow(table),
            'frame_to_arrow': pf.to_arrow,
            'series_from_arrow': lambda: polars.from_arrow(array),
            'series_to_arrow': pseries.to_arrow,
            'dlpack_import': lambda: polars.DataFrame({'close': np.from_dlpack(values)}),
        }
        if hasattr(pseries, '__dlpack__'):
            reg['polars']['dlpack_export'] = lambda: np.from_dlpack(pseries)
    return reg
