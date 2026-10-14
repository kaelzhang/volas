"""Interop measurements retain values, both sizes and performance-gate identities."""
import copy
import json

import numpy as np
import pytest

from scripts import benchmark_report, perf_gate
from scripts.perf_items import volas_key
from test.bench_interop import OPS, build


@pytest.mark.parametrize('with_polars', [False, True])
def test_interop_operations_preserve_values(with_polars):
    pl = pytest.importorskip('polars') if with_polars else None
    expected = np.arange(8, dtype='float64')
    for candidate, operations in build(8, pl).items():
        for operation, call in operations.items():
            result = call()
            if operation.endswith('to_arrow'):
                result = result.to_pandas()
            if operation == 'dlpack_export':
                values = result
            elif operation.startswith('series_'):
                values = result.to_numpy()
            else:
                values = result['close'].to_numpy()
            np.testing.assert_array_equal(values, expected)


def test_interop_gate_detects_large_import_copy_and_report_keeps_sizes(tmp_path):
    rows = []
    for length in [1999, 1_000_000]:
        for op in OPS:
            rows.append({'name': f'test_api_interop[{length}-{op}-volas]',
                         'params': {'candidate': 'volas', 'indicator': op},
                         'extra_info': {'length': length},
                         'stats': {'median': 20e-6, 'mean': 20e-6}})
    base = {'benchmarks': rows}
    head = copy.deepcopy(base)
    head['benchmarks'][6]['stats']['median'] = 3e-3
    assert len({volas_key(row) for row in rows}) == 12
    for side, data in [('base', base), ('head', head)]:
        (tmp_path / f'{side}.json').write_text(json.dumps(data))
    assert perf_gate.main([str(tmp_path / 'head.json'), '--base', str(tmp_path / 'base.json')]) == 1
    groups = benchmark_report.parse(base)
    assert len(groups['api']) == 12
    assert 'frame_from_arrow (1,000,000 rows)' in groups['api']
