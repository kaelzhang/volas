import json
import os
from pathlib import Path
import warnings

import pytest


_PANDAS_ORACLE_VERSION = None

_TARGETED_BENCHMARK_TESTS = {
    'test_coverage',
    'test_coverage_extended',
    'test_coverage_after_append',
    'test_tf_live',
    'test_first_cache',
}


def _configure_pandas_oracle_warnings():
    global _PANDAS_ORACLE_VERSION

    try:
        import pandas as pd
    except ModuleNotFoundError:
        _PANDAS_ORACLE_VERSION = 'not installed'
        return

    _PANDAS_ORACLE_VERSION = pd.__version__
    warning_cls = getattr(pd.errors, 'PandasChangeWarning', None)
    if warning_cls is not None:
        # pandas is an oracle for the supported pandas-shaped API. Its migration
        # warnings are compatibility failures, not background test noise.
        warnings.simplefilter('error', warning_cls)
    pandas4_warning_cls = getattr(pd.errors, 'Pandas4Warning', None)
    if pandas4_warning_cls is not None and pandas4_warning_cls is not warning_cls:
        warnings.simplefilter('error', pandas4_warning_cls)


def pytest_configure(config):
    _configure_pandas_oracle_warnings()


def pytest_report_header(config):
    if _PANDAS_ORACLE_VERSION is None:
        _configure_pandas_oracle_warnings()
    return f'pandas oracle: {_PANDAS_ORACLE_VERSION}'


def pytest_addoption(parser):
    parser.addoption("--volas-cost-json", default=None, help="Collect release cost bounds for paired gating.")
    parser.addoption(
        '--volas-benchmark-indicator',
        action='store',
        default=None,
        help='Run only the coverage benchmark rows for one directive, e.g. roc:10.',
    )


def _indicator_matches(requested, params):
    for key in ('directive', 'indicator'):
        value = params.get(key)
        if value == requested:
            return True
        if isinstance(value, str) and value.split('@n=', 1)[0] == requested:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    if config.getoption('--volas-cost-json'):
        kept = [item for item in items if 'cost_guard' in item.fixturenames]
        config.hook.pytest_deselected(items=[item for item in items if item not in kept])
        items[:] = kept
    requested = config.getoption('--volas-benchmark-indicator')
    if not requested:
        return

    kept = []
    deselected = []
    for item in items:
        callspec = getattr(item, 'callspec', None)
        test_name = getattr(item, 'originalname', None) or item.name.split('[', 1)[0]
        params = callspec.params if callspec is not None else {}
        if test_name in _TARGETED_BENCHMARK_TESTS and _indicator_matches(requested, params):
            kept.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = kept


@pytest.fixture
def cost_guard(request):
    """Correctness runs execute all operations; release pairs enforce timing bounds."""
    def record(actual, limit):
        path = request.config.getoption('--volas-cost-json')
        if path:
            checks = getattr(request.config, '_volas_cost_checks', {})
            if request.node.nodeid in checks:
                raise ValueError('duplicate release cost check')
            checks[request.node.nodeid] = actual / limit
            request.config._volas_cost_checks = checks
    def pair(first, second):
        if os.environ.get('VOLAS_COST_REVERSE') == '1':
            b, a = second(), first()
        else:
            a, b = first(), second()
        return a, b
    record.pair = pair
    return record


def pytest_sessionfinish(session, exitstatus):
    path = session.config.getoption('--volas-cost-json')
    if path and exitstatus == 0:
        Path(path).write_text(json.dumps(getattr(session.config, '_volas_cost_checks', {})))
