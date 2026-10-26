"""Existing CI entries provision the build owner's tools and execute its check."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from test.pgo_toolchain import ROOT, clean_env
from test.pgo_linked_imports import linked_dump
from test.pgo_toolchain import HOST, SYSROOT

sys.path.insert(0, str(ROOT / "scripts"))
from build_release import RUST, MATURIN  # noqa: E402


def job_text(name, workflow="ci"):
    text = (ROOT / f".github/workflows/{workflow}.yml").read_text()
    return re.search(r"^  " + name + r":\n.*?(?=^  \S|\Z)", text, re.M | re.S)[0]


@pytest.mark.parametrize("job", ["test", "coverage", "asm-diff", "types"])
def test_ci_provisions_pinned_toolchain(job, tmp_path):
    text = job_text(job)
    rust = re.search(r"uses: dtolnay/rust-toolchain@(\S+)", text)[1]
    if "toolchain: ${{ steps.toolchain.outputs.rust }}" in text:
        versions = subprocess.check_output(
            [sys.executable, "scripts/check_toolchain.py", "--versions"],
            cwd=ROOT,
            text=True,
        )
        values = dict(line.split("=", 1) for line in versions.splitlines())
        rust = values["rust"]
        assert values["maturin"] == MATURIN
        assert (
            'python scripts/check_toolchain.py --versions >> "$GITHUB_OUTPUT"' in text
        )
    assert rust == RUST, f"{job} selects {rust}; build owner requires {RUST}"
    assert "components: ${{ steps.toolchain.outputs.components }}" in text
    assert values["components"] == "llvm-tools-preview"
    assert '"maturin==${{ steps.toolchain.outputs.maturin }}"' in text
    assert "python scripts/check_toolchain.py" in text
    checked = subprocess.run(
        [sys.executable, "scripts/check_toolchain.py"],
        cwd=ROOT,
        env=clean_env(),
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert f"rustc {RUST}" in checked.stdout and f"maturin {MATURIN}" in checked.stdout
    # Those actual selected LLVM tools must parse a final linked import library.
    dump = linked_dump(tmp_path / "linked", HOST)
    assert "foreign" in dump and "Disassembly of section" in dump


def test_prerequisite_check_rejects_wrong_active_compiler(tmp_path):
    # Keep the real installed compiler/LLVM in place; change only active selection.
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    rustc = bin_ / "rustc"
    rustc.write_text(
        '#!/bin/sh\nprintf "rustc 0.0.0\\nhost: invalid\\nrelease: 0.0.0\\n"\n'
    )
    rustc.chmod(0o755)
    result = subprocess.run(
        [sys.executable, "scripts/check_toolchain.py"],
        cwd=ROOT,
        env=clean_env(PATH=str(bin_) + os.pathsep + os.environ["PATH"]),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert f"requires rustc {RUST}" in result.stderr


def test_ci_lint_reads_the_shared_rust_pin():
    text = job_text("lint")
    assert "toolchain: ${{ steps.toolchain.outputs.rust }}" in text
    assert "components: ${{ steps.toolchain.outputs.components }},clippy" in text
    assert "check_toolchain.py --rust-only" in text
    assert "1.96.0" not in text


@pytest.mark.parametrize("workflow,job,consumer", [
    ("perf", "perf", "Resolve the base commit"),
    ("pages", "build", "Build the benchmark report"),
])
def test_report_workflows_check_pinned_tools_before_build_or_base_skip(
    workflow, job, consumer, tmp_path
):
    text = job_text(job, workflow)
    assert 'python scripts/check_toolchain.py --versions >> "$GITHUB_OUTPUT"' in text
    assert "toolchain: ${{ steps.toolchain.outputs.rust }}" in text
    assert "components: ${{ steps.toolchain.outputs.components }}" in text
    assert '"maturin==${{ steps.toolchain.outputs.maturin }}"' in text
    check = re.search(
        r"- name: Check build prerequisites\n\s+run: (.+)", text
    )
    assert check and check.start() < text.index("- name: " + consumer)
    assert "continue-on-error" not in text[:check.end()]
    if workflow == "perf":
        assert "python -m pip install --no-build-isolation -e '.[dev,benchmark]'" in text
    # Execute the workflow's preflight with a wrong outer CLI. It must fail
    # before entering the BASE-skippable part of the workflow.
    module = tmp_path / "maturin"
    module.mkdir()
    (module / "__init__.py").touch()
    (module / "__main__.py").write_text("print('maturin 0.0.0')\n")
    marker = tmp_path / "base-started"
    result = subprocess.run(
        ["bash", "-e", "-c", check[1] + '\ntouch "$BASE_MARKER"'],
        cwd=ROOT,
        env=clean_env(
            PYTHONPATH=str(tmp_path), BASE_MARKER=str(marker),
            PATH=str(Path(sys.executable).parent)
            + os.pathsep + os.environ["PATH"],
        ),
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and f"requires maturin {MATURIN}" in result.stderr
    assert not marker.exists()


def test_test_matrix_provisions_coverage_plugin_before_prerequisite_check():
    text = job_text('test')
    coverage = job_text('coverage')
    install = '- name: Install cargo-llvm-cov\n'
    assert install in text
    # The coverage job owns the installation selector, including its version.
    def step(source):
        return source[source.index(install):].split('\n      - name:', 1)[0].strip()
    assert step(text) == step(coverage)
    check = 'python scripts/check_toolchain.py --cargo-llvm-cov'
    assert text.index(install) < text.index(check) < text.index('run: cargo test')
    assert text.index(check) < text.index('run: make test-quick')


def test_prerequisite_check_rejects_missing_coverage_plugin(tmp_path):
    bin_ = tmp_path / 'bin'
    bin_.mkdir()
    for tool in ('cargo', 'rustc'):
        (bin_ / tool).symlink_to(SYSROOT / 'bin' / tool)
    marker = tmp_path / 'tests-started'
    env = clean_env(PATH=str(bin_), CARGO_HOME=str(tmp_path / 'cargo-home'))
    absent = subprocess.run(['cargo', 'llvm-cov', '--version'], env=env,
                            capture_output=True, text=True)
    assert absent.returncode != 0 and 'no such command:' in absent.stderr and 'llvm-cov' in absent.stderr
    command = re.search(r'- name: Check build prerequisites\n\s+run: (.+)',
                        job_text('test'))[1]
    command = command.replace('python ', f'"{sys.executable}" ', 1)
    result = subprocess.run(
        ['/bin/bash', '-ec', command + f'\n/usr/bin/touch "{marker}"'], cwd=ROOT,
        env=env,
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'install cargo-llvm-cov' in result.stderr
    assert 'no such command:' in result.stderr and 'llvm-cov' in result.stderr
    assert not marker.exists()


def test_prerequisite_check_reports_active_coverage_plugin():
    result = subprocess.run(
        [sys.executable, 'scripts/check_toolchain.py', '--cargo-llvm-cov'],
        cwd=ROOT, env=clean_env(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual = subprocess.check_output(['cargo', 'llvm-cov', '--version'],
                                     cwd=ROOT, env=clean_env(), text=True).strip()
    assert actual.startswith('cargo-llvm-cov ') and actual in result.stdout
