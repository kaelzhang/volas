"""Developer shell overrides cannot alter or prevent an owned build."""

import json
import subprocess
import sys

import pytest

from scripts import build_release as B
from test.pgo_toolchain import ROOT, HOST, clean_env

AMBIENT = {
    "LDFLAGS": "-L/usr/local/opt/ruby/lib",
    "CFLAGS": "-fvolas-ambient-must-not-reach-compiler",
    "CPPFLAGS": "-DVOLAS_AMBIENT",
    "RUSTFLAGS": "--volas-ambient-must-not-reach-compiler",
    "CARGO_ENCODED_RUSTFLAGS": "--also-invalid",
    "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER": "/no/ambient/linker",
    "CXXFLAGS_aarch64_apple_darwin": "--invalid",
    "CPPFLAGS_aarch64_apple_darwin": "--invalid",
    "RUSTC_WRAPPER": "/no/ambient/wrapper",
    "VOLAS_UNRECORDED_BUILD": "would-change-build",
    "LIBRARY_PATH": "/no/ambient/library",
}


def test_real_identity_ignores_unbound_shell_flags(monkeypatch):
    for key in AMBIENT:
        monkeypatch.delenv(key, raising=False)
    before = B.identity(ROOT, HOST)[0]
    for key, value in AMBIENT.items():
        monkeypatch.setenv(key, value)
    assert B.identity(ROOT, HOST)[0] == before


def test_real_release_cli_sanitizes_before_resolution_and_compilation(tmp_path):
    root = tmp_path / "source"
    crate = root / "crates/volas-python"
    (crate / "src").mkdir(parents=True)
    (root / ".cargo").mkdir()
    (root / ".cargo/config.toml").write_text("")
    (root / "Cargo.toml").write_text(
        '[workspace]\nmembers=["crates/volas-python"]\nresolver="2"\n'
        '[profile.release]\nlto=true\ncodegen-units=1\nopt-level=3\n'
    )
    (crate / "Cargo.toml").write_text(
        '[package]\nname="volas-python"\nversion="0.0.0"\nedition="2021"\n'
        '[lib]\nname="volas_rs"\ncrate-type=["cdylib"]\n'
        '[dependencies]\npyo3={version="=0.25.1",features=["extension-module"]}\n'
    )
    (crate / "build.rs").write_text(
        'fn main() { if std::env::var("VOLAS_UNRECORDED_BUILD").is_ok() {'
        ' panic!("unrecorded input reached build script"); } }\n'
    )
    (crate / "src/lib.rs").write_text(
        'use pyo3::prelude::*;\n#[pyfunction]\nfn value()->usize {42}\n'
        '#[pymodule]\nfn volas_rs(m: &Bound<\'_, PyModule>) -> PyResult<()> {'
        'm.add_function(wrap_pyfunction!(value,m)?)?; Ok(()) }\n'
    )
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires=["maturin==1.13.3"]\nbuild-backend="maturin"\n'
        '[project]\nname="volas"\nversion="0.0.0"\n[tool.maturin]\n'
        'manifest-path="crates/volas-python/Cargo.toml"\ninclude=["volas_rs.pyi"]\n'
    )
    (root / "volas_rs.pyi").write_text("def value() -> int: ...\n")
    subprocess.run(["cargo", "generate-lockfile", "--offline"], cwd=root,
                   env=clean_env(), check=True, capture_output=True)
    identities = []
    for name, additions in (("clean", {}), ("shell", AMBIENT)):
        artifacts = tmp_path / name
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_release.py"),
             "--source", str(root), "--mode", "generate", "--artifacts", str(artifacts)],
            cwd=root, env=clean_env(**additions), capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        manifest = json.loads((artifacts / "generate/build.json").read_text())
        identities.append(manifest["identity"])
        trace = (artifacts / "generate/rustc-argv.jsonl").read_text()
        assert "volas-ambient" not in trace and "/no/ambient" not in trace
        if additions:
            assert set(additions) <= set(manifest["environment"]["removed"])
    assert identities[0] == identities[1]


def test_recorded_deployment_target_still_changes_identity(monkeypatch):
    monkeypatch.setenv("MACOSX_DEPLOYMENT_TARGET", "11.0")
    before = B.identity(ROOT, HOST)[0]
    monkeypatch.setenv("MACOSX_DEPLOYMENT_TARGET", "12.0")
    after = B.identity(ROOT, HOST)[0]
    assert before != after


def test_unrecorded_cargo_config_cannot_change_linker(tmp_path):
    from scripts.build_config import check_cargo_config

    (tmp_path / ".cargo").mkdir()
    (tmp_path / ".cargo/config.toml").write_text(
        f'[target.{HOST}]\nlinker="/no/unrecorded/linker"\n'
    )
    with pytest.raises(ValueError, match="unbound Cargo configuration"):
        check_cargo_config(tmp_path)


def test_shell_build_boundary_records_inputs_and_keeps_parent_environment(tmp_path):
    import os

    source = tmp_path / "probe.rs"
    source.write_text(
        'const _: () = assert!(option_env!("VOLAS_UNRECORDED_BUILD").is_none());\n'
        '#[cfg(not(owned))] compile_error!("missing owner flags");\n'
        'fn main() {}\n'
    )
    record = tmp_path / "environment.json"
    env = dict(os.environ, **AMBIENT)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_env.py"), "--record", str(record),
         "--set", "RUSTFLAGS=--cfg owned", "--", "cargo", "--version"],
        env=env, cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())["environment"]["owned"] == {
        "RUSTFLAGS": "--cfg owned"
    }
    # Invoke the real compiler too: rustc does not consume Cargo's RUSTFLAGS.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_env.py"), "--record", str(record),
         "--", "rustc", str(source), "--cfg", "owned", "-o", str(tmp_path / "probe")],
        env=env, cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(record.read_text())
    assert set(AMBIENT) <= set(data["environment"]["removed"])
    assert not set(AMBIENT) & set(data["environment"]["retained"])
    assert env["LDFLAGS"] == AMBIENT["LDFLAGS"]
    assert data["argv"][-4:] == ["--cfg", "owned", "-o", str(tmp_path / "probe")]


def test_publish_authentication_survives_without_recording_the_secret(tmp_path):
    import hashlib

    token = "local-test-credential-do-not-publish"
    record = tmp_path / "environment.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_env.py"), "--record", str(record),
         "--", sys.executable, "-c",
         "import hashlib,os; print(hashlib.sha256("
         "os.environ['CARGO_REGISTRY_TOKEN'].encode()).hexdigest())"],
        env=clean_env(CARGO_REGISTRY_TOKEN=token, **AMBIENT),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    fingerprint = hashlib.sha256(token.encode()).hexdigest()
    assert result.stdout.strip() == fingerprint
    assert token not in record.read_text()
    data = json.loads(record.read_text())["environment"]
    assert data["retained"]["CARGO_REGISTRY_TOKEN"] == {"sha256": fingerprint}
    assert set(AMBIENT) <= set(data["removed"])
