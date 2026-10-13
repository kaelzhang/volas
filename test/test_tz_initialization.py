"""First-use initialization supports callbacks without recursive deadlocks."""

import subprocess
import sys


def _run_child(source):
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_initialization_callback_can_resolve_a_different_key():
    _run_child("""
import volas
from volas import _timezone

original = _timezone._resolve
seen = []
def resolve(key):
    seen.append(key)
    if key == 'America/New_York':
        nested = volas.Timestamp('2024-01-01', tz='Asia/Shanghai')
        assert nested.tz == 'Asia/Shanghai'
    return original(key)
_timezone._resolve = resolve
value = volas.Timestamp('2024-01-01', tz='America/New_York')
assert value.tz == 'America/New_York'
assert seen == ['America/New_York', 'Asia/Shanghai']
""")


def test_same_key_recursion_raises_and_a_later_retry_succeeds():
    _run_child("""
import volas
from volas import _timezone

original = _timezone._resolve
def resolve(key):
    volas.Timestamp('2024-01-01', tz=key)
    return original(key)
_timezone._resolve = resolve
try:
    volas.Timestamp('2024-01-01', tz='America/New_York')
except RuntimeError as error:
    assert 'recursive timezone initialization' in str(error)
    assert 'America/New_York' in str(error)
else:
    raise AssertionError('same-key recursion must fail without hanging')
finally:
    _timezone._resolve = original
value = volas.Timestamp('2024-01-01', tz='America/New_York')
assert value.tz == 'America/New_York'
""")


def test_cross_thread_initialization_cycle_raises_and_allows_retry():
    _run_child("""
from concurrent.futures import ThreadPoolExecutor
import threading

import volas
from volas import _timezone

original = _timezone._resolve
keys = ('America/Chicago', 'Europe/London')
other = {keys[0]: keys[1], keys[1]: keys[0]}
barrier = threading.Barrier(2)
state = threading.local()
def resolve(key):
    if not getattr(state, 'entered', False):
        state.entered = True
        barrier.wait(timeout=3)
        volas.Timestamp('2024-01-01', tz=other[key])
    return original(key)
_timezone._resolve = resolve
def initialize(key):
    try:
        volas.Timestamp('2024-01-01', tz=key)
        return None
    except RuntimeError as error:
        return str(error)
with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(initialize, keys))
assert any(result is not None for result in results)
assert all(
    result is None or ('would block' in result and 'retry' in result)
    for result in results
), results
_timezone._resolve = original
for key in keys:
    assert volas.Timestamp('2024-01-01', tz=key).tz == key
""")
