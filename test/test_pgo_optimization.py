"""Optimized real interpreters cannot bypass release or training acceptance."""

import ast
import subprocess
import sys

import pytest

from test.pgo_toolchain import ROOT, clean_env


@pytest.mark.parametrize("optimize", ["1", "2"])
def test_optimized_training_rejects_missing_inventory(tmp_path, optimize):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
from scripts import pgo_train as T
original = Path.read_text
Path.read_text = lambda p, *a, **k: '{}' if p.name == 'pgo_inventory.json' else original(p, *a, **k)
T.train('batch')
""",
        ],
        cwd=ROOT,
        env=clean_env(PYTHONOPTIMIZE=optimize),
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0, result.stdout
    assert "training inventory drift" in result.stderr


@pytest.mark.parametrize("optimize", ["1", "2"])
def test_optimized_owner_never_publishes_broken_smoke(tmp_path, optimize):
    # Execute the actual final-smoke command and publication path in a separate
    # optimized owner process. Build preparation is immaterial to this boundary.
    site = tmp_path / "normal/site"
    site.mkdir(parents=True)
    (site / "volas.py").write_text(
        'class DataFrame:\n def __init__(self,*a,**k): raise RuntimeError("BROKEN_COMPUTATION")\n'
    )
    runner = """
import json, sys
from pathlib import Path
from scripts import build_release as B
from scripts import build_install
site=Path(sys.argv[1])/'normal/site'
B.identity=lambda *a: ({}, None)
wheel=site.parent/"test.whl"
wheel.write_bytes(b"wheel")
B.build=lambda *a: (site.parent, {"wheel":str(wheel)})
# Record publication rather than performing a second unrelated artifact check.
build_install.publish=lambda *a: (site/'volas-build.json').write_text('published')
sys.argv=['build_release','--target','fixture','--artifacts',sys.argv[1]]
B.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", runner, str(tmp_path)],
        cwd=ROOT,
        env=clean_env(PYTHONOPTIMIZE=optimize),
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0, result.stdout
    assert "BROKEN_COMPUTATION" in result.stderr or "optimisation" in result.stderr
    assert not (tmp_path / "final.json").exists()
    assert not (site / "volas-build.json").exists()


def test_scripts_do_not_use_removable_acceptance_asserts():
    found = []
    for path in (ROOT / "scripts").glob("*.py"):
        tree = ast.parse(path.read_text())
        found += [
            f"{path.name}:{n.lineno}"
            for n in ast.walk(tree)
            if isinstance(n, ast.Assert)
        ]
        # Embedded worker/smoke programs are Python too.
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                try:
                    program = ast.parse(n.value)
                except SyntaxError:
                    continue
                found += [
                    f"{path.name}:embedded:{n.lineno}"
                    for c in ast.walk(program)
                    if isinstance(c, ast.Assert)
                ]
    assert not found, found


@pytest.mark.parametrize("optimize", ["1", "2"])
def test_build_environment_rejects_inherited_optimization(optimize):
    from scripts.build_config import check_environment

    with pytest.raises(ValueError, match="optimisation"):
        check_environment({"PYTHONOPTIMIZE": optimize})


@pytest.mark.parametrize("optimize", ["1", "2"])
@pytest.mark.parametrize("damage", ["raise", "values", "mask"])
def test_smoke_computation_runs_under_optimization(tmp_path, optimize, damage):
    code = (
        'raise RuntimeError("BROKEN_COMPUTATION")'
        if damage == "raise"
        else 'return [float("nan"),1.5,99.]'
        if damage == "values"
        else "return [0.,1.5,2.5]"
    )
    (tmp_path / "volas.py").write_text(
        "class DataFrame:\n def __init__(self,*a,**k): pass\n def exec(self,*a,**k): "
        + code
        + "\n"
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/release_smoke.py")],
        cwd=tmp_path,
        env=clean_env(PYTHONOPTIMIZE=optimize, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "BROKEN_COMPUTATION" if damage == "raise" else "smoke computation mismatch"
    ) in result.stderr


def test_owner_child_interpreter_has_optimization_disabled(tmp_path, monkeypatch):
    from scripts import build_release as B

    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    log = tmp_path / "child.log"
    B.run(
        [sys.executable, "-c", "import sys; print(sys.flags.optimize)"], log, tmp_path
    )
    assert log.read_text().splitlines()[-1] == "0"
