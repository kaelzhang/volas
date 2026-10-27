"""The normal gate must fail closed when Rust budgets disappear or are ignored."""
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    'python_budgets', Path(__file__).resolve().parents[1] / 'scripts/test_python_budgets.py')
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)
NAMES = ('entry_budget::explicit_export_queries_missing_only_when_needed',
         'entry_budget::no_override_constructor_takes_one_snapshot')


def test_driver_accepts_both_required_tests():
    DRIVER.check_discovered('\n'.join(f'{name}: test' for name in NAMES))
    DRIVER.check_executed('\n'.join(f'test {name} ... ok' for name in NAMES))


@pytest.mark.parametrize('listing', ['', f'{NAMES[0]}: test',
                                     '\n'.join(f'{n}: benchmark' for n in NAMES)])
def test_driver_rejects_missing_tests(listing):
    with pytest.raises(RuntimeError, match='required'):
        DRIVER.check_discovered(listing)


@pytest.mark.parametrize('status', ['ignored', 'FAILED', ''])
def test_driver_rejects_ignored_failed_and_zero_execution(status):
    output = '\n'.join(f'test {name} ... {status}' for name in NAMES)
    with pytest.raises(RuntimeError, match='required'):
        DRIVER.check_executed(output)


@pytest.mark.parametrize('execution', [
    'running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored',
    '\n'.join(f'test {name} ... ignored' for name in NAMES),
])
def test_driver_main_rejects_silent_success(monkeypatch, tmp_path, execution):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DRIVER, 'interpreter_environment', lambda: {})
    monkeypatch.setattr(DRIVER, 'build', lambda *_: 'cargo-test-executable')
    outputs = iter(['\n'.join(f'{name}: test' for name in NAMES), execution])
    monkeypatch.setattr(DRIVER, 'run', lambda *a, **k: next(outputs))
    with pytest.raises(RuntimeError, match='required'):
        DRIVER.main()
