"""Release tooling contracts, independent of compiler work."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def owner():
    spec = importlib.util.spec_from_file_location(
        "build_release", ROOT / "scripts/build_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "key",
    [
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
        "MATURIN_PGO",
    ],
)
def test_flags_cannot_mask_target_configuration(key):
    with pytest.raises(ValueError, match="ambient"):
        owner().check_environment({key: "bad"})


def test_profile_provenance_mismatch(tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        owner().validate_profile(
            tmp_path / "p", {"source": "new"}, {"identity": {"source": "old"}}, None
        )


def test_empty_profile_rejected(tmp_path):
    p = tmp_path / "p"
    p.touch()
    with pytest.raises(ValueError, match="empty"):
        owner().validate_profile(p, {}, {"identity": {}}, None)


def test_corrupt_profile_rejected(tmp_path):
    p = tmp_path / "p"
    p.write_text("corrupt")
    with pytest.raises(ValueError, match="hash"):
        owner().validate_profile(
            p, {}, {"identity": {}, "profile_sha256": "other"}, None
        )


def test_training_wrong_library_rejected(tmp_path):
    p = tmp_path / "wrong.so"
    p.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="library"):
        owner().verify_library(p, tmp_path / "site", "wrong-hash")


def test_empty_raw_rejected(tmp_path):
    with pytest.raises(ValueError, match="raw"):
        owner().raw_files(tmp_path)


def test_corrupt_profile_with_matching_hash_is_rejected_by_llvm(tmp_path):
    import subprocess

    m = owner()
    info = m.output(["rustc", "-Vv"])
    host = next(
        line.split(": ")[1] for line in info.splitlines() if line.startswith("host:")
    )
    llvm = (
        Path(m.output(["rustc", "--print", "sysroot"]))
        / "lib/rustlib"
        / host
        / "bin/llvm-profdata"
    )
    p = tmp_path / "profile.profdata"
    p.write_bytes(b"not llvm profile data")
    with pytest.raises(subprocess.CalledProcessError):
        m.validate_profile(p, {}, {"identity": {}, "profile_sha256": m.digest(p)}, llvm)


def test_append_factory_returns_fresh_state_and_actual_append():
    import numpy as np
    from test.bench_workloads import append, generated_ohlcv
    from volas import DataFrame

    data = generated_ohlcv(128)
    run, setup = append(data, "ema:20")
    (a,), _ = setup()
    (b,), _ = setup()
    assert a is not b
    assert len(a) == len(b) == 127
    run(a)
    assert len(a) == 128 and len(b) == 127
    np.testing.assert_array_equal(
        a["ema:20"].to_numpy(), DataFrame(data).exec("ema:20")
    )


def test_training_inventory_is_complete_and_executable():
    import json
    from scripts.count_indicators import count
    from test.bench_workloads import generated_ohlcv
    from volas import DataFrame

    inventory = json.loads((ROOT / "test/pgo_inventory.json").read_text())
    assert set(inventory) == set(count()["forms"])
    df = DataFrame(generated_ohlcv(257))
    for directive in inventory.values():
        assert len(df.exec(directive, dtype="float64")) == 257


def test_package_recipe_keeps_one_artifact_directory(tmp_path):
    import os
    import re
    import subprocess
    import sys

    clock = tmp_path / "date"
    counter = tmp_path / "counter"
    clock.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\np=Path({str(counter)!r})\nn=int(p.read_text())+1 if p.exists() else 1\np.write_text(str(n))\nprint(n)\n"
    )
    clock.chmod(0o755)
    result = subprocess.check_output(
        ["make", "-n", "build-pkg", "PYTHON=" + sys.executable],
        cwd=ROOT,
        env=dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"]),
        text=True,
    )
    paths = re.findall(r"/target/release-build-\d+", result)
    assert paths and len(set(paths)) == 1, result


def test_profile_paths_with_spaces_use_encoded_flags(monkeypatch):
    monkeypatch.delenv("RUSTFLAGS", raising=False)
    flags = ["-Cprofile-use=/a path/profile.profdata", "-Ctarget-feature=+fma"]
    env = owner().flag_environment(flags)
    assert env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") == flags
    assert "RUSTFLAGS" not in env
