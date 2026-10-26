"""Release identities must follow real pip/maturin installation lifetimes."""

import csv
import json
import hashlib
import zipfile
from pathlib import Path
import subprocess
import sys

import pytest

from test.pgo_toolchain import ROOT, clean_env


def run(cmd, root, env, log):
    result = subprocess.run(
        list(map(str, cmd)),
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with log.open("a") as f:
        f.write(result.stdout)
    assert result.returncode == 0, result.stdout
    return result.stdout.strip()


@pytest.mark.parametrize("entry", ["maturin", "make"])
def test_release_debug_invalidates_identity_and_coverage_preserves_it(tmp_path, entry):
    # Real tiny PyO3 wheel through the release owner, then unwrapped maturin
    # develop in a private venv. Never install into the interpreter running pytest.
    crate = tmp_path / "crates/volas-python"
    (crate / "src").mkdir(parents=True)
    (tmp_path / "volas").mkdir()
    (tmp_path / "volas/__init__.py").write_text("")
    (tmp_path / "Cargo.toml").write_text(
        '[workspace]\nmembers=["crates/volas-python"]\nresolver="2"\n'
        "[profile.release]\nlto=true\ncodegen-units=1\nopt-level=3\n"
    )
    (crate / "Cargo.toml").write_text(
        '[package]\nname="volas-python"\nversion="0.0.0"\nedition="2021"\n'
        '[lib]\nname="volas_rs"\ncrate-type=["cdylib"]\n'
        '[dependencies]\npyo3={version="=0.25.1",features=["extension-module"]}\n'
    )
    (crate / "src/lib.rs").write_text(
        "use pyo3::prelude::*;\n#[pymodule]\nfn volas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {"
        ' m.add("debug", cfg!(debug_assertions))?; Ok(()) }\n'
    )
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires=["maturin>=1.4,<2"]\nbuild-backend="maturin"\n'
        '[project]\nname="volas"\nversion="0.0.0"\n[tool.maturin]\n'
        'manifest-path="crates/volas-python/Cargo.toml"\n'
    )
    (crate / "volas_rs.pyi").write_text("debug: bool\n")
    log = tmp_path / "install.log"
    env = clean_env()
    for key in ("RUSTC_WRAPPER", "LLVM_PROFILE_FILE", "CARGO_TARGET_DIR", "SDKROOT"):
        env.pop(key, None)
    venv = tmp_path / "venv"
    run(
        [sys.executable, "-m", "venv", "--system-site-packages", venv],
        tmp_path,
        env,
        log,
    )
    python = venv / "bin/python"
    from maturin.__main__ import get_maturin_path

    (venv / "bin/maturin").symlink_to(get_maturin_path())
    env.update(VIRTUAL_ENV=str(venv), CONDA_PREFIX=str(venv))
    run(["cargo", "generate-lockfile", "--offline"], tmp_path, env, log)
    artifacts = tmp_path / "release"
    artifacts.mkdir()
    run(
        [python, "-m", "maturin", "build", "--release", "--locked", "--out", artifacts],
        tmp_path,
        env,
        log,
    )
    wheel = next(artifacts.glob("*.whl"))
    with zipfile.ZipFile(wheel) as z:
        native_hash = hashlib.sha256(
            z.read(next(n for n in z.namelist() if n.endswith(".so")))
        ).hexdigest()
    data = {
        "mode": "normal",
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "native_sha256": native_hash,
    }
    (artifacts / "final.json").write_text(json.dumps(data))
    env["PYTHONPATH"] = str(ROOT)
    run(
        [
            python,
            "-c",
            "from pathlib import Path; import json; "
            "from scripts.build_release import install_final; "
            "p=Path(" + repr(str(artifacts)) + "); "
            "install_final(json.loads((p/'final.json').read_text()),p)",
        ],
        tmp_path,
        env,
        log,
    )
    site = Path(
        run(
            [python, "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
            tmp_path,
            env,
            log,
        )
    )
    manifest = site / "volas-build.json"
    original = manifest.read_bytes()
    assert (
        run(
            [python, "-c", "import volas_rs; print(volas_rs.debug)"], tmp_path, env, log
        )
        == "False"
    )
    # This is the same preflight owner called by the full test/coverage entry.
    snapshot = tmp_path / "saved.json"
    install_owner = ROOT / "scripts/build_install.py"
    run([python, install_owner, "save", snapshot], tmp_path, env, log)
    develop = (
        [python, "-m", "maturin", "develop"]
        if entry == "maturin"
        else [
            "make",
            "-f",
            ROOT / "Makefile",
            "build-ext",
            "PYTHON=" + str(python),
            "VOLAS_PYTHON=" + str(python),
        ]
    )
    run(develop, tmp_path, env, log)
    assert not manifest.exists(), "debug installation retained a stale release manifest"
    assert (
        run(
            [python, "-c", "import volas_rs; print(volas_rs.debug)"], tmp_path, env, log
        )
        == "True"
    )
    debug_snapshot = tmp_path / "debug.json"
    assert (
        run([python, install_owner, "save", debug_snapshot], tmp_path, env, log) == ""
    )
    assert not debug_snapshot.exists()
    run([python, install_owner, "restore", snapshot], tmp_path, env, log)
    assert manifest.read_bytes() == original
    records = list(site.glob("volas-*.dist-info/RECORD"))
    assert len(records) == 1
    with records[0].open(newline="") as f:
        assert sum(row[0] == "volas-build.json" for row in csv.reader(f)) == 1
    # Hash failures remain fatal: an invalid release cannot become a debug install
    # merely because its sidecar exists. Snapshot bytes survive the rejection.
    wheel_bytes = wheel.read_bytes()
    wheel.write_bytes(wheel_bytes + b"tampered")
    rejected = subprocess.run(
        [str(python), str(install_owner), "save", str(tmp_path / "bad.json")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0 and "release wheel hash mismatch" in rejected.stderr
    wheel.write_bytes(wheel_bytes)
    native = next(site.rglob("volas_rs*.so"))
    native_bytes = native.read_bytes()
    native.write_bytes(native_bytes + b"tampered")
    rejected = subprocess.run(
        [str(python), str(install_owner), "save", str(tmp_path / "bad.json")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert (
        rejected.returncode != 0
        and "installed native library hash mismatch" in rejected.stderr
    )
    native.write_bytes(native_bytes)
    assert snapshot.read_bytes() == original
    # Restoring coverage must also restore ownership for the *next* develop.
    run(develop, tmp_path, env, log)
    assert not manifest.exists()
