"""Stable identities shared by paired measurement and the regression gate."""
from __future__ import annotations

import re


def benchmark_key(row: dict) -> tuple[str, str]:
    """Keep pytest function sections and explicit benchmark groups distinct."""
    name = row['name']
    section = name.split('[', 1)[0].removeprefix('test_')
    group = row.get('group')
    return (f'{section}/{group}' if group else section, name)


def volas_key(row: dict) -> tuple[str, str] | None:
    section, name = benchmark_key(row)
    params = row.get('params') or {}
    function = name.split('[', 1)[0]
    if params.get('candidate') != 'volas' and function not in ('test_tf_live', 'test_first_cache',
                                                            'test_range_batch_million', 'test_range_single_row'):
        match = re.search(r'\[(.+)-volas\]$', name)
        if not match:
            return None
    # The whole parametrized name retains length, phase and directive dimensions.
    return section, name
