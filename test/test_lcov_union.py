import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'lcov_union.py'


def _write_lcov(path, source, hits):
    lines = [f'SF:{source}']
    lines.extend(f'DA:{line},{count}' for line, count in hits.items())
    lines.append(f'LF:{len(hits)}')
    lines.append(f'LH:{sum(1 for count in hits.values() if count)}')
    lines.append('end_of_record')
    path.write_text('\n'.join(lines) + '\n')


def _source(tmp_path, name='demo/src/lib.rs', text='fn main() {}\n'):
    path = tmp_path / 'crates' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_lcov_union_succeeds_when_union_covers_all_lines(tmp_path):
    source = _source(tmp_path)
    cargo = tmp_path / 'cargo.lcov'
    pytest = tmp_path / 'pytest.lcov'
    out = tmp_path / 'union.lcov'
    _write_lcov(cargo, source, {1: 1})
    _write_lcov(pytest, source, {1: 0})

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(cargo), str(pytest), str(out)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert '100.00%' in result.stdout
    assert out.exists()


def test_lcov_union_fails_when_any_line_remains_uncovered(tmp_path):
    source = _source(tmp_path, text='fn covered() {}\nfn missed() {}\n')
    cargo = tmp_path / 'cargo.lcov'
    pytest = tmp_path / 'pytest.lcov'
    _write_lcov(cargo, source, {1: 1, 2: 0})
    _write_lcov(pytest, source, {1: 0, 2: 0})

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(cargo), str(pytest)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert 'demo/src/lib.rs: 2' in result.stdout
    assert 'coverage is below 100%' in result.stderr


def test_lcov_union_honors_line_exclusion_markers(tmp_path):
    source = _source(
        tmp_path,
        text='fn covered() {}\npanic!("invariant"); // LCOV_EXCL_LINE\n',
    )
    cargo = tmp_path / 'cargo.lcov'
    pytest = tmp_path / 'pytest.lcov'
    _write_lcov(cargo, source, {1: 1, 2: 0})
    _write_lcov(pytest, source, {1: 0, 2: 0})

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(cargo), str(pytest)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert '100.00%' in result.stdout
