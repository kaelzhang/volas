"""Run existing Cargo budgets with CPython's embed flags.

No extension feature changes: only this test executable links the C initializer.
The Rust test measures process CPU time through the selected Python.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import sysconfig


REQUIRED = {
    'entry_budget::explicit_export_queries_missing_only_when_needed',
    'entry_budget::no_override_constructor_takes_one_snapshot',
}


def check_discovered(output):
    found = set(re.findall(r'^(\S+): test$', output, re.MULTILINE))
    if not REQUIRED <= found:
        raise RuntimeError(f'required budget tests not discovered: {sorted(REQUIRED - found)}')


def check_executed(output):
    passed = set(re.findall(r'^test (\S+) \.\.\. ok$', output, re.MULTILINE))
    if not REQUIRED <= passed:
        raise RuntimeError(f'required budget tests not executed and passed: {sorted(REQUIRED - passed)}')


def run(args, **kwargs):
    result = subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end='')
    return result.stdout


def interpreter_environment():
    import numpy as np
    env = os.environ.copy()
    env.pop('VOLAS_BUDGET_CPU_WORKER', None)
    for name in ('executable', 'prefix', 'exec_prefix', 'base_prefix', 'base_exec_prefix'):
        env[f'VOLAS_BUDGET_{name.upper()}'] = getattr(sys, name)
    paths = [str(Path(p).absolute()) for p in sys.path if p]
    env['VOLAS_BUDGET_PATH_COUNT'] = str(len(paths))
    env.update({f'VOLAS_BUDGET_PATH_{i}': p for i, p in enumerate(paths)})
    env['VOLAS_BUDGET_NUMPY'] = str(Path(np.__file__).resolve())
    env['PYO3_PYTHON'] = sys.executable
    return env


def build(env, target):
    # Run the selected interpreter's own config script. It owns library names,
    # static dependencies and macOS framework arguments; never trim LDLIBRARY.
    config = Path(sysconfig.get_config_var('LIBPL')) / 'python-config.py'
    includes = shlex.split(run([sys.executable, str(config), '--includes']))
    ldflags = shlex.split(run([sys.executable, str(config), '--embed', '--ldflags']))
    libdir = sysconfig.get_config_var('LIBDIR')
    ldflags.append(f'-L{libdir}')
    if sys.platform != 'win32':
        ldflags.append(f'-Wl,-rpath,{libdir}')
    source = Path('crates/volas-python/tests/entry_budget/python.c')
    identity = source.read_bytes() + sys.version.encode() + sys.executable.encode()
    obj = target / f'python-{hashlib.sha256(identity).hexdigest()[:16]}.o'
    subprocess.run([*shlex.split(sysconfig.get_config_var('CC')), *includes, '-c',
                    'crates/volas-python/tests/entry_budget/python.c', '-o', str(obj)], check=True)
    inherited = (env['CARGO_ENCODED_RUSTFLAGS'].split('\x1f')
                 if 'CARGO_ENCODED_RUSTFLAGS' in env else shlex.split(env.get('RUSTFLAGS', '')))
    flags = ['-C', f'link-arg={obj}']
    for flag in ldflags:
        flags += ['-C', f'link-arg={flag}']
    env['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join([*inherited, *flags])
    env['CARGO_TARGET_DIR'] = str(target)
    print(f'CPython embed link flags: {ldflags}', flush=True)
    output = run(['cargo', 'test', '-p', 'volas-python', '--lib', '--no-run',
                  '--message-format=json-render-diagnostics'], env=env)
    artifacts = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
    executables = [a['executable'] for a in artifacts
                   if a.get('reason') == 'compiler-artifact' and a.get('executable')]
    if len(executables) != 1:
        raise RuntimeError(f'expected one Cargo test executable, got {executables}')
    return executables[0]


def main():
    target = Path('target/python-budget-tests').resolve()
    target.mkdir(parents=True, exist_ok=True)
    env = interpreter_environment()
    executable = build(env, target)
    check_discovered(run([executable, 'entry_budget::', '--list'], env=env))
    output = run([executable, 'entry_budget::', '--test-threads=1', '--nocapture'], env=env)
    print(output, flush=True)
    check_executed(output)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stdout or '', file=sys.stderr)
        print(error.stderr or '', file=sys.stderr)
        raise
