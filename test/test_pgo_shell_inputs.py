"""Shell-owned inputs cannot be supplied by inherited implementation variables."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from test.pgo_toolchain import ROOT, clean_env


@pytest.mark.parametrize('mode', ['release', 'checked'])
def test_audit_shell_initializes_owned_inputs(tmp_path, mode):
    # Run the complete, unchanged production entry and real environment owner.
    # Observe downstream commands without doing a workspace build or installation.
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('audit_harness.sh', 'build_env.py', 'build_config.py'):
        shutil.copy2(ROOT / 'scripts' / name, scripts / name)
    observer = tmp_path / 'observe.py'
    observer.write_text(
        'import json,os,sys\n'
        'from pathlib import Path\n'
        'with Path(__file__).with_name("calls.jsonl").open("a") as f:\n'
        ' f.write(json.dumps({"argv":sys.argv[1:],"env":dict(os.environ)})+"\\n")\n'
    )
    python = tmp_path / 'python'
    python.write_text(
        '#!/bin/bash\n'
        'case "$*" in\n'
        f' *build_release.py*|"-m maturin"*|"-m pytest"*) exec "{sys.executable}" '
        f'"{observer}" "$@" ;;\n'
        f' *) exec "{sys.executable}" "$@" ;;\n'
        'esac\n'
    )
    cargo = tmp_path / 'cargo'
    cargo.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{observer}" cargo "$@"\n')
    python.chmod(0o755)
    cargo.chmod(0o755)
    result = subprocess.run(
        ['bash', str(scripts / 'audit_harness.sh'), mode], cwd=tmp_path,
        env=clean_env(
            PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
            VOLAS_PYTHON=str(python), CHECKED_FLAGS='-C opt-level=0',
            OWNED_ARGS='--set RUSTFLAGS=-Copt-level=0',
            RUSTFLAGS='--invalid-ambient', LDFLAGS='--invalid-ambient',
        ), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    matrix = next(c for c in calls if c['argv'][0] == 'cargo')
    expected = '-C debug-assertions -C overflow-checks' if mode == 'checked' else None
    assert matrix['env'].get('RUSTFLAGS') == expected
    assert 'CHECKED_FLAGS' not in matrix['env'] and 'LDFLAGS' not in matrix['env']
    records = [json.loads(p.read_text()) for p in
               (tmp_path / 'target/build-environments').glob('*.json')]
    record = next(r for r in records if r['argv'][0] == 'cargo')['environment']
    assert record['owned'] == ({'RUSTFLAGS': expected} if expected else {})
    assert {'CHECKED_FLAGS', 'RUSTFLAGS', 'LDFLAGS'} <= set(record['removed'])
    assert record['effective'].get('RUSTFLAGS') == expected
    if mode == 'checked':
        extension = next(c for c in calls if c['argv'][:2] == ['-m', 'maturin'])
        assert extension['env']['RUSTFLAGS'] == expected
        assert 'CHECKED_FLAGS' not in extension['env']


@pytest.mark.parametrize('setting', [
    'RUSTFLAGS=-C opt-level=0', 'CARGO_ENCODED_RUSTFLAGS=-Copt-level=0',
    'CFLAGS=--invalid-ambient', 'MallocPreScribble=0', 'LLVM_PROFILE_FILE=relative',
])
def test_shell_owner_rejects_unbound_owned_values(tmp_path, setting):
    marker = tmp_path / 'started'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/build_env.py'), '--set', setting,
         '--', sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()'],
        env=clean_env(CHECKED_FLAGS='-C opt-level=0'), capture_output=True, text=True,
    )
    assert result.returncode != 0, 'unbound shell input reached the child'
    assert 'unbound owned input' in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize('profile', ['checked', 'coverage'])
def test_shell_owner_constructs_check_flags_and_records_inputs(tmp_path, profile):
    record = tmp_path / 'environment.json'
    path = str(tmp_path / 'build-%m-%p.profraw')
    settings = ['--set', 'MallocPreScribble=1', '--set', 'MALLOC_PERTURB_=170']
    if profile == 'coverage':
        settings += ['--set', 'LLVM_PROFILE_FILE=' + path]
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/build_env.py'), '--record', str(record),
         '--rust-checks', profile, *settings, '--', sys.executable, '-c',
         'import json,os; print(json.dumps(dict(os.environ)))'],
        env=clean_env(RUSTFLAGS='--invalid-ambient', LLVM_PROFILE_FILE='ambient'),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    expected = {'RUSTFLAGS': '-C debug-assertions -C overflow-checks' if profile == 'checked'
                else '-Cinstrument-coverage', 'MallocPreScribble': '1', 'MALLOC_PERTURB_': '170'}
    if profile == 'coverage':
        expected['LLVM_PROFILE_FILE'] = path
    else:
        assert 'LLVM_PROFILE_FILE' not in actual
    assert all(actual[k] == v for k, v in expected.items())
    data = json.loads(record.read_text())['environment']
    assert data['owned'] == expected
    assert {'RUSTFLAGS', 'LLVM_PROFILE_FILE'} <= set(data['removed'])
