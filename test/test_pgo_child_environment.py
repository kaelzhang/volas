"""Every tool boundary owns and records its actual child environment."""

import json
import os
import re
import subprocess
import sys
import zipfile

from scripts import build_release as B, pgo_dependencies as D
from test.pgo_toolchain import ROOT, HOST


def test_training_pip_owns_environment_and_preserves_identity(tmp_path, monkeypatch):
    wheel = tmp_path / 'env_fixture-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as z:
        z.writestr('env_fixture/__init__.py', 'value = 42\n')
        z.writestr('env_fixture-1.0.dist-info/METADATA',
                   'Metadata-Version: 2.1\nName: env_fixture\nVersion: 1.0\n')
        z.writestr('env_fixture-1.0.dist-info/WHEEL',
                   'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        z.writestr('env_fixture-1.0.dist-info/RECORD', '')
    lock = tmp_path / 'requirements.txt'
    lock.write_text('env_fixture==1.0 --hash=sha256:' + D.digest(wheel) + '\n')
    monkeypatch.setattr(D, 'LOCK', lock)
    before = B.identity(ROOT, HOST)[0]
    clean = D.install(tmp_path / 'clean', tmp_path / 'clean.log', tmp_path)
    injected = {'PIP_USER': '1', 'LDFLAGS': '-L/usr/local/opt/ruby/lib',
                'CFLAGS': '--invalid-ambient', 'RUSTFLAGS': '--invalid-ambient'}
    for key, value in injected.items():
        monkeypatch.setenv(key, value)
    actual = D.install(tmp_path / 'shell', tmp_path / 'shell.log', tmp_path)
    assert actual == clean
    assert B.identity(ROOT, HOST)[0] == before
    record = json.loads((tmp_path / 'shell.environment.json').read_text())
    assert set(injected) <= set(record['environment']['removed'])
    assert not set(injected) & set(record['environment']['retained'])
    assert '--require-hashes' in record['argv']
    assert '--only-binary=:all:' in record['argv']
    assert all(os.environ[key] == value for key, value in injected.items())


def test_coverage_rust_tests_observe_owned_allocator_checks(tmp_path):
    crate = tmp_path / 'probe'
    (crate / 'src').mkdir(parents=True)
    (crate / 'Cargo.toml').write_text(
        '[package]\nname="allocator-environment-probe"\nversion="0.0.0"\nedition="2021"\n'
        '[workspace]\n')
    (crate / 'src/lib.rs').write_text('''
#[test]
fn observes_owned_allocator_checks() {
    assert_eq!(std::env::var("MallocPreScribble").as_deref(), Ok("1"));
    assert_eq!(std::env::var("MALLOC_PERTURB_").as_deref(), Ok("170"));
    assert!(std::env::var("LDFLAGS").is_err());
    assert!(std::env::var("RUSTFLAGS").is_err());
}
''')
    source = (ROOT / 'scripts/coverage.sh').read_text()
    # Execute the production Rust coverage boundary, substituting only the
    # workspace with a small real Rust test that observes the runtime inputs.
    # Keep cargo llvm-cov itself, so its test subprocess must retain the guards.
    command = re.search(r'"\$PYTHON" scripts/build_env.py[^\n]*(?:\\\n[^\n]*)*',
                        source[source.index('echo ">> cargo test coverage..."'):])[0]
    command = command.replace('--workspace --exclude volas-python',
                              '--manifest-path "$PROBE/Cargo.toml"')
    command = command.replace('$PROF_DIR/cargo.lcov', str(tmp_path / 'probe.lcov'))
    record = tmp_path / 'rust.environment.json'
    command = command.replace('scripts/build_env.py',
                              f'"{ROOT}/scripts/build_env.py" --record "{record}"')
    result = subprocess.run(
        ['bash', '-eu', '-c', command], cwd=ROOT,
        env=dict(os.environ, PYTHON=sys.executable, PROBE=str(crate),
                 CARGO_TARGET_DIR=str(tmp_path / 'target'),
                 MallocPreScribble='1', MALLOC_PERTURB_='170',
                 LDFLAGS='-L/usr/local/opt/ruby/lib', RUSTFLAGS='--invalid-ambient'),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    environment = json.loads(record.read_text())['environment']
    assert environment['owned']['MallocPreScribble'] == '1'
    assert environment['owned']['MALLOC_PERTURB_'] == '170'
    assert {'LDFLAGS', 'RUSTFLAGS'} <= set(environment['removed'])


def test_owned_inputs_survive_by_provenance_in_real_child(tmp_path):
    keys = ['MallocPreScribble', 'MALLOC_PERTURB_', 'MallocScribble',
            'ASAN_OPTIONS', 'VOLAS_CUSTOM_HARNESS_GUARD']
    script = 'import json,os; print(json.dumps({k:os.environ.get(k) for k in ' + repr(keys) + '}))'
    for owned in (False, True):
        record = tmp_path / f'{owned}.json'
        additions = [arg for key in keys for arg in ('--set', key + '=owner')]
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/build_env.py'), '--record', str(record),
             *(additions if owned else []), '--', sys.executable, '-c', script],
            env=dict(os.environ, **dict.fromkeys(keys, 'ambient')),
            capture_output=True, text=True, check=True,
        )
        assert json.loads(result.stdout) == dict.fromkeys(keys, 'owner' if owned else None)
        data = json.loads(record.read_text())['environment']
        assert set(keys) <= set(data['removed'])
        assert data['owned'] == (dict.fromkeys(keys, 'owner') if owned else {})


def test_scripts_spawn_only_through_the_environment_owner():
    import ast

    bypasses = []
    for path in (ROOT / 'scripts').glob('*.py'):
        if path.name == 'build_config.py':
            continue
        tree = ast.parse(path.read_text())
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            module = call.func.value
            if isinstance(module, ast.Name) and (
                module.id == 'subprocess' or
                module.id == 'os' and call.func.attr in {'system', 'popen', 'execv', 'execvpe'}
            ):
                bypasses.append(f'{path.name}:{call.lineno}')
    assert not bypasses, bypasses
