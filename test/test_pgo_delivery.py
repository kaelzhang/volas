"""Package delivery and dependency installs consume verified manifests."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_upload_only_consumes_manifest_and_preserves_other_artifacts(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    wheel = build / "selected.whl"
    wheel.write_bytes(b"selected distribution")
    stale = tmp_path / "dist/old-mode.whl"
    stale.parent.mkdir()
    stale.write_bytes(b"user owned")
    manifest = build / "delivery.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "artifacts": [
                    {
                        "path": str(wheel),
                        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    (tmp_path / "twine.py").write_text(
        'import json,os,sys\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n'
    )
    capture = tmp_path / "upload.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/publish_artifacts.py"),
        "--manifest",
        str(manifest),
    ]
    env = dict(os.environ, PYTHONPATH=str(tmp_path), CAPTURE=str(capture))
    subprocess.run(command, env=env, check=True)
    assert json.loads(capture.read_text())[-1] == str(wheel)
    assert str(stale) not in capture.read_text()
    assert stale.read_bytes() == b"user owned"
    capture.unlink()
    wheel.write_bytes(b"changed")
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "hash mismatch" in result.stderr
    assert not capture.exists()


def test_dependency_install_checks_distribution_hash(tmp_path, monkeypatch):
    from scripts import pgo_dependencies as D

    wheel = tmp_path / "pgo_fixture-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("pgo_fixture/__init__.py", "value = 42\n")
        z.writestr(
            "pgo_fixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: pgo_fixture\nVersion: 1.0\n",
        )
        z.writestr(
            "pgo_fixture-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        z.writestr("pgo_fixture-1.0.dist-info/RECORD", "")
    lock = tmp_path / "requirements.txt"
    lock.write_text("pgo_fixture==1.0 --hash=sha256:" + "0" * 64 + "\n")
    monkeypatch.setattr(D, "LOCK", lock)
    log = tmp_path / "install.log"
    with pytest.raises(subprocess.CalledProcessError):
        D.install(tmp_path / "bad", log, tmp_path)
    assert "DO NOT MATCH THE HASHES" in log.read_text()
    assert not (tmp_path / "bad/volas-training-dependencies.json").exists()
    lock.write_text("pgo_fixture==1.0 --hash=sha256:" + D.digest(wheel) + "\n")
    installed = D.install(tmp_path / "good", log, tmp_path)
    assert installed["distributions"]["pgo_fixture"]["sha256"] == D.digest(wheel)
    monkeypatch.syspath_prepend(str(tmp_path / "good"))
    import types

    monkeypatch.setitem(
        sys.modules,
        "pgo_fixture",
        types.SimpleNamespace(__file__=str(tmp_path / "foreign.py")),
    )
    with pytest.raises(ValueError, match="loaded dependency identity"):
        D.verify(tmp_path / "good")
    monkeypatch.delitem(sys.modules, "pgo_fixture")
    assert D.verify(tmp_path / "good") == installed["distributions"]
    (tmp_path / "good/pgo_fixture/__init__.py").write_text("value = 0\n")
    with pytest.raises(ValueError, match="payload mismatch"):
        D.verify(tmp_path / "good")
