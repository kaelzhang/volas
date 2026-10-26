"""Hash-locked training distributions and verified, isolated installations."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess

import sys

if __package__:
    from . import build_config as processes
else:
    import build_config as processes

LOCK = Path(__file__).with_name("pgo-requirements.txt")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locked():
    result = {}
    for line in LOCK.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        requirement, *hashes = shlex.split(line)
        name, version = requirement.split("==")
        if not hashes or any(
            not h.startswith("--hash=sha256:") or len(h[14:]) != 64 for h in hashes
        ):
            raise ValueError(
                f"training dependency missing distribution SHA-256: {name}"
            )
        result[name] = {"version": version, "sha256": [h[14:] for h in hashes]}
    return result


def install(destination, log, wheels=None):
    if destination.exists():
        raise ValueError("training dependency directory must be fresh")
    report = destination.with_suffix(".pip.json")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--require-hashes",
        "--only-binary=:all:",
        "--no-deps",
        "--no-compile",
        "--ignore-installed",
        "--target",
        str(destination),
        "--report",
        str(report),
        "-r",
        str(LOCK),
    ]
    if wheels:
        cmd += ["--no-index", "--find-links", str(wheels)]
    with Path(log).open("w") as f:
        processes.run_child(cmd, record=Path(log).with_suffix(".environment.json"), stdout=f, stderr=subprocess.STDOUT, check=True)
    expected = locked()
    actual = {}
    for item in json.loads(report.read_text())["install"]:
        name, version = item["metadata"]["name"].lower(), item["metadata"]["version"]
        sha = item["download_info"]["archive_info"]["hashes"]["sha256"]
        if (
            name not in expected
            or version != expected[name]["version"]
            or sha not in expected[name]["sha256"]
        ):
            raise ValueError(f"training distribution identity mismatch: {name}")
        actual[name] = {
            "version": version,
            "sha256": sha,
            "filename": item["download_info"]["url"].rsplit("/", 1)[-1],
        }
    if set(actual) != set(expected):
        raise ValueError("training distribution set mismatch")
    payload = {
        str(p.relative_to(destination)): digest(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    data = {"distributions": actual, "payload": payload}
    (destination / "volas-training-dependencies.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )
    return data


def verify(destination):
    data = json.loads((destination / "volas-training-dependencies.json").read_text())
    expected = locked()
    if set(data["distributions"]) != set(expected):
        raise ValueError("training distribution set mismatch")
    for name, item in data["distributions"].items():
        distribution = importlib.metadata.distribution(name)
        if (
            item["version"] != expected[name]["version"]
            or item["sha256"] not in expected[name]["sha256"]
            or distribution.version != item["version"]
            or not Path(distribution.locate_file(""))
            .resolve()
            .is_relative_to(destination.resolve())
        ):
            raise ValueError(f"training distribution identity mismatch: {name}")
    for relative, sha in data["payload"].items():
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination.resolve()) or digest(path) != sha:
            raise ValueError(f"training dependency payload mismatch: {relative}")
    for name in expected:
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(destination.resolve()):
            raise ValueError(
                f"loaded dependency identity mismatch: {name}: {module.__file__}"
            )
    return data["distributions"]
