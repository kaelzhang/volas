_TARGETED_BENCHMARK_TESTS = {
    'test_coverage',
    'test_coverage_extended',
    'test_coverage_after_append',
}


def pytest_addoption(parser):
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
