"""The repository pin provisions every local and CI Rust invocation."""

import subprocess
import sys
import tomllib

from test.pgo_toolchain import ROOT


def test_repository_pin_is_the_only_version_source():
    from scripts import build_release as B

    sys.path.insert(0, str(ROOT / "scripts"))
    import check_toolchain as C

    pin = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pin["channel"] == "1.96.0"
    assert "llvm-tools-preview" in pin["components"]
    assert package["build-system"]["requires"] == ["maturin==1.13.3"]
    assert not any(
        d.startswith("maturin")
        for d in package["project"]["optional-dependencies"]["dev"]
    )
    values = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [sys.executable, "-S", "scripts/check_toolchain.py", "--versions"],
            text=True,
            cwd=ROOT,
        ).splitlines()
    )
    assert values["rust"] == pin["channel"]
    assert values["maturin"] == package["build-system"]["requires"][0].split("==")[1]
    assert 'RUST = "' not in (ROOT / "scripts/build_release.py").read_text()
    assert 'MATURIN = "' not in (ROOT / "scripts/build_release.py").read_text()
    assert B.RUST == C.RUST == values["rust"]
    for script in ("asm_diff.sh", "hot_path_asm.sh"):
        text = (ROOT / "scripts" / script).read_text()
        assert "check_toolchain.py --rust-only" in text
        assert "1.96.0" not in text


def test_clean_install_dry_run_selects_pinned_tools():
    dry = subprocess.check_output(
        ["make", "-n", "install", f"PYTHON={sys.executable}"],
        cwd=ROOT,
        text=True,
    )
    assert "rustup update stable" not in dry
    assert "rustup toolchain install 1.96.0" in dry
    assert "--component llvm-tools-preview" in dry
    assert f'{sys.executable} -m pip install "maturin==1.13.3"' in dry
    assert "scripts/check_toolchain.py --rust-only" in dry
    assert "scripts/check_toolchain.py\n" in dry
    assert "--no-build-isolation" in dry
    rust = subprocess.check_output(
        ["make", "-n", "install-rust", f"PYTHON={sys.executable}"],
        cwd=ROOT,
        text=True,
    )
    assert "rustup toolchain install 1.96.0" in rust
