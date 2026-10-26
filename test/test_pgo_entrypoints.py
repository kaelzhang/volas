"""Exercise real manifest/schema/hash guards before any comparison worker."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def site(path, mode="normal", source="same", config=None):
    path.mkdir()
    native = path / "volas_rs.so"
    native.write_bytes(b"native fixture with a verifiable identity")
    config = config or {
        "flags": [],
        "release": {"lto": True, "codegen-units": 1, "opt-level": 3},
        "features": "extension-module",
        "tools": {"rustc": "pinned"},
    }
    data = {
        "schema": 2,
        "mode": mode,
        "flags": config["flags"],
        "identity": {"source": source, "configuration": config},
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    if mode == "use":
        data["profile_sha256"] = "trained-profile"
    (path / "volas-build.json").write_text(json.dumps(data))
    return path


def test_pair_contract_actually_exercises_native_guard(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_pair
    from test.test_pgo_perf import (
        test_pair_rejects_mixed_modes_and_cross_source_adoption,
    )

    calls = []
    original = build_pair.verify_library

    def verify(*a):
        calls.append(a)
        return original(*a)

    monkeypatch.setattr(build_pair, "verify_library", verify)
    test_pair_rejects_mixed_modes_and_cross_source_adoption(tmp_path, monkeypatch)
    assert calls, "pair tests bypassed real native hash validation"


@pytest.mark.parametrize("damage", ["native", "schema", "flags", "release", "mode"])
def test_cpu_entry_rejects_before_manifest_or_worker(tmp_path, damage):
    base, head = site(tmp_path / "base"), site(tmp_path / "head")
    meta = head / "volas-build.json"
    data = json.loads(meta.read_text())
    if damage == "native":
        (head / "volas_rs.so").write_bytes(b"corrupt")
    elif damage == "schema":
        data["schema"] = 900
    elif damage == "flags":
        data["identity"]["configuration"]["flags"] = ["-Ctarget-feature=-crc,+crc"]
    elif damage == "release":
        data["identity"]["configuration"]["release"]["lto"] = "thin"
    else:
        data["mode"] = "use"
        data["profile_sha256"] = "profile"
    meta.write_text(json.dumps(data))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/perf_cpu.py"),
            str(base),
            str(head),
            str(tmp_path / "output"),
            str(tmp_path / "NO_MANIFEST"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "NO_MANIFEST" not in result.stderr and "worker" not in result.stdout
    assert not (tmp_path / "output.builds.json").exists()


def test_valid_manifest_pair_cli(tmp_path):
    base, head = site(tmp_path / "base"), site(tmp_path / "head")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_pair.py"), str(base), str(head)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "validated normal/normal" in result.stdout


def test_training_load_rejects_real_wrong_library_before_work(tmp_path, monkeypatch):
    import importlib
    from scripts import pgo_train

    native = importlib.import_module("volas_rs.volas_rs")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    # Even a matching hash cannot authorize a native outside the selected site.
    with pytest.raises(ValueError, match="library identity mismatch"):
        pgo_train.load_site(
            tmp_path, hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
        )
    with pytest.raises(ValueError, match="library identity mismatch"):
        pgo_train.load_site(Path(native.__file__).parent, "wrong-native-hash")


@pytest.mark.parametrize("damage", ["native", "mode"])
def test_shell_entry_rejects_before_measurement(tmp_path, damage):
    from test.test_perf_sections import _shell_arguments

    _shell_arguments(tmp_path, ROOT / "scripts/perf_ab.sh", invalid=damage)


def test_training_cli_bootstraps_from_artifacts_without_global_packages(tmp_path):
    import os

    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"training": {}}))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts/pgo_train.py"),
            "--site",
            str(tmp_path / "site"),
            "--native-sha256",
            "wrong",
            "--identity",
            str(identity),
            "--dependencies",
            str(tmp_path / "deps"),
            "--output",
            str(tmp_path / "result.json"),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    assert "volas-training-dependencies.json" in result.stderr
    assert not (tmp_path / "result.json").exists()
