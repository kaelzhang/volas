"""Link the PyO3 unit-test executable to this interpreter, keeping extension features."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig


def main():
    # The shipped cdylib uses extension-module (Python supplies its symbols).
    # A standalone Rust test executable instead needs libpython at link/run time.
    library = sysconfig.get_config_var('LDLIBRARY') or sysconfig.get_config_var('LIBRARY')
    libdir = sysconfig.get_config_var('LIBDIR')
    if sys.platform == 'win32':
        libdir = str(Path(sys.base_prefix) / 'libs')
        name = f'python{sys.version_info.major}{sys.version_info.minor}'
    else:
        name = library.removeprefix('lib').split('.so')[0].split('.dylib')[0].removesuffix('.a')
    flags = ['-L', f'native={libdir}', '-l', name]
    if sys.platform != 'win32':
        flags += ['-C', f'link-arg=-Wl,-rpath,{libdir}']
    env = os.environ.copy()
    env['PYO3_PYTHON'] = sys.executable
    inherited = (env['CARGO_ENCODED_RUSTFLAGS'].split('\x1f')
                 if 'CARGO_ENCODED_RUSTFLAGS' in env else shlex.split(env.get('RUSTFLAGS', '')))
    env['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join([*inherited, *flags])
    env['CARGO_TARGET_DIR'] = str(Path('target/python-budget-tests').resolve())
    subprocess.run(['cargo', 'test', '-p', 'volas-python', '--lib', 'entry_budget::',
                    '--', '--nocapture'], env=env, check=True)


if __name__ == '__main__':
    main()
