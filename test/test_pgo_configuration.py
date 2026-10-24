"""Regression contracts for effective Cargo configuration and actual invocations."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import build_release as B

from test.pgo_toolchain import ROOT, HOST, extension_link_flags


@pytest.mark.parametrize(
    "key",
    [
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CARGO_BUILD_RUSTC",
        "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER",
        "CC",
        "SDKROOT",
    ],
)
def test_unbound_tool_override_is_rejected(key):
    with pytest.raises(ValueError, match="ambient"):
        B.check_environment({key: "/bin/false"})


def test_real_resolver_preserves_concrete_target_prefix_and_order(
    tmp_path, monkeypatch
):
    (tmp_path / ".cargo").mkdir()
    flags = ["-Ctarget-feature=+crc,-crc", "-Ctarget-feature=-crc,+crc", "--cfg=prefix"]
    (tmp_path / ".cargo/config.toml").write_text(
        f"[target.{HOST}]\nrustflags={json.dumps(flags)}\n"
    )
    monkeypatch.delenv("LDFLAGS", raising=False)
    monkeypatch.delenv("RUSTFLAGS", raising=False)
    monkeypatch.delenv("CARGO_ENCODED_RUSTFLAGS", raising=False)
    resolved = B.resolved_flags(
        tmp_path, HOST, ["--cfg=suffix"], tmp_path / "resolve.log"
    )
    assert resolved == [*flags, "--cfg=suffix"]


@pytest.mark.parametrize(
    "mode,flags",
    [
        ("normal", ["-Cprofile-use=/tmp/wrong"]),
        ("generate", ["-Cprofile-use=/tmp/wrong"]),
        ("use", ["-Cprofile-generate=/tmp/wrong"]),
        ("generate", ["-Cprofile-generate=/tmp/a", "-Cprofile-generate=/tmp/b"]),
    ],
)
@pytest.mark.parametrize(
    "target", list(dict.fromkeys([HOST, "x86_64-unknown-linux-gnu"]))
)
def test_real_argv_rejects_wrong_feedback_mode(tmp_path, mode, flags, target):
    log = tmp_path / "argv.log"
    args = [
        subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
        + "/bin/rustc",
        "--crate-name",
        "volas_rs",
        "--target",
        target,
        *flags,
        *extension_link_flags(target),
    ]
    import shlex

    log.write_text("     Running `" + shlex.join(args) + "`\n")
    with pytest.raises(ValueError, match="feedback"):
        B.validate_argv(log, mode, target)


def test_real_profile_rejects_ordered_configuration_mismatch(tmp_path):
    p = tmp_path / "p"
    p.write_bytes(b"profile")
    a = {"configuration": {"flags": ["+crc", "-crc"]}}
    b = {"configuration": {"flags": ["-crc", "+crc"]}}
    with pytest.raises(ValueError, match="provenance"):
        B.validate_profile(p, a, {"identity": b, "profile_sha256": B.digest(p)}, None)


def test_identity_changes_with_ancestor_flags(tmp_path, monkeypatch):
    monkeypatch.delenv("LDFLAGS", raising=False)
    monkeypatch.delenv("RUSTFLAGS", raising=False)
    monkeypatch.delenv("CARGO_ENCODED_RUSTFLAGS", raising=False)
    # Keep the real source but change a Cargo ancestor configuration in an archive.
    import shutil

    source = tmp_path / "source"
    source.mkdir()
    for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml"):
        shutil.copy2(ROOT / name, source / name)
    for name in ("crates", "volas", ".cargo"):
        shutil.copytree(ROOT / name, source / name)
    before = B.identity(source, HOST)[0]
    (tmp_path / ".cargo").mkdir()
    (tmp_path / ".cargo/config.toml").write_text(
        f'[target.{HOST}]\nrustflags=["--cfg=changed"]\n'
    )
    after = B.identity(source, HOST)[0]
    assert before != after, "effective configuration must bind ancestor flags"


def test_publish_uses_this_build_manifest_only(tmp_path):
    result = subprocess.check_output(
        [
            "make",
            "-n",
            "publish",
            f"BUILD_ARTIFACTS={tmp_path}",
            "PYTHON=" + sys.executable,
        ],
        cwd=ROOT,
        text=True,
    )
    assert "dist/*" not in result
    assert "--manifest" in result and "delivery.json" in result


@pytest.mark.parametrize(
    "target", list(dict.fromkeys([HOST, "x86_64-unknown-linux-gnu"]))
)
def test_build_blocks_missing_trained_function_diagnostic(
    tmp_path, monkeypatch, target
):
    import shlex
    import zipfile

    profile = tmp_path / "profile.profdata"
    profile.write_bytes(b"profile")
    symbol = "_ZN13volas_compute8dispatch17h0123456789abcdefE"
    profile.with_suffix(".json").write_text(json.dumps({"coverage": {symbol: 42}}))
    config = {
        "flags": [],
        "tools": {
            "rustc": {"path": "/bin/rustc"},
            "cargo": {"path": "/bin/cargo"},
            "linker": {"path": "/bin/cc"},
            "sdk": None,
        },
    }
    flags = [f"-Cprofile-use={profile}", "-Cllvm-args=-pgo-warn-missing-function"]

    def run(cmd, log, root, env=None, **kw):
        a = [
            "/bin/rustc",
            "--crate-name",
            "volas_rs",
            "--target",
            target,
            *flags,
            "-Clinker=/bin/cc",
            *extension_link_flags(target),
        ]
        Path(log).write_text(
            "Running `"
            + shlex.join(a)
            + "`\nwarning: volas_compute.abc-cgu.0: no profile data available for function "
            + symbol
            + " Hash = 123 up to 0 count discarded\n"
        )
        if env and "VOLAS_RUSTC_TRACE" in env:
            trace = Path(env["VOLAS_RUSTC_TRACE"])
            trace.write_text(json.dumps(a) + "\n")
            # Prove the platform's argv precondition independently of the
            # missing-trained-function diagnostic this test is about.
            B.validate_argv(log, "use", target, profile, config, trace)
        wheels = tmp_path / "use/wheels"
        wheels.mkdir(exist_ok=True)
        with zipfile.ZipFile(wheels / "test.whl", "w") as z:
            z.writestr("volas_rs.pyi", "stub")

    def install(w, site, root, log):
        site.mkdir()
        native = site / "volas_rs.so"
        native.write_bytes(b"native")
        return native

    monkeypatch.setattr(B, "run", run)
    monkeypatch.setattr(B, "install", install)
    monkeypatch.setattr(B, "resolved_flags", lambda *a: [])
    with pytest.raises(ValueError, match="trained.*profile|profile.*trained"):
        B.build(ROOT, tmp_path, "use", target, {"configuration": config}, profile)


def test_training_requirements_have_distribution_hashes():
    lines = (ROOT / "scripts/pgo-requirements.txt").read_text().splitlines()
    assert all("--hash=sha256:" in l for l in lines if l and not l.startswith("#"))


@pytest.mark.parametrize(
    "flags,mode,path",
    [
        (["-Cprofile-use=/tmp/other"], "use", "/tmp/expected"),
        (
            ["-C", "profile-generate=/tmp/raw", "-Cprofile-use=/tmp/p"],
            "generate",
            "/tmp/raw",
        ),
        (["-C", "profile-use=/tmp/p"], "normal", None),
    ],
)
def test_feedback_validation_uses_exact_argument_and_path(flags, mode, path):
    from scripts.build_config import validate_flags

    with pytest.raises(ValueError, match="feedback"):
        validate_flags(flags, mode, path)


def test_argv_rejects_actual_compiler_substitution(tmp_path):
    from scripts.build_config import validate_argv

    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(["/bin/false", "--crate-name", "volas_rs", "--target", HOST]) + "\n"
    )
    with pytest.raises(ValueError, match="executable mismatch"):
        validate_argv(
            None,
            "normal",
            HOST,
            config={"tools": {"rustc": {"path": "/bin/rustc"}}},
            trace=trace,
        )


def test_diagnostics_distinguishes_cold_monomorph_and_unknown_format():
    from scripts.pgo_diagnostics import audit

    hot = "_ZN13volas_compute8dispatch17h0123456789abcdefE"
    cold = "_ZN13volas_compute8dispatch17hfedcba9876543210E"
    line = f"warning: volas_compute.abc-cgu.0: no profile data available for function {cold} Hash = 123 up to 0 count discarded"
    data = audit(line, {hot: 42, cold: 0})
    assert data["allowed_cold_misses"][0]["training_count"] == 0
    assert "zero training counters" in data["allowed_cold_misses"][0]["reason"]
    with pytest.raises(ValueError, match="unrecognized"):
        audit("warning: no profile data available: changed LLVM format", {hot: 42})


def test_real_resolver_preserves_build_rustflags_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("RUSTFLAGS", raising=False)
    monkeypatch.delenv("CARGO_ENCODED_RUSTFLAGS", raising=False)
    (tmp_path / ".cargo").mkdir()
    (tmp_path / ".cargo/config.toml").write_text(
        '[build]\nrustflags=["--cfg=build_fallback"]\n'
    )
    assert B.resolved_flags(
        tmp_path, HOST, ["--cfg=suffix"], tmp_path / "flags.log"
    ) == ["--cfg=build_fallback", "--cfg=suffix"]


def test_real_profile_coverage_cannot_be_replaced_by_sidecar(tmp_path):
    from scripts.pgo_diagnostics import coverage

    source = tmp_path / "probe.rs"
    source.write_text("fn main() { std::hint::black_box(42); }\n")
    binary = tmp_path / "probe"
    raw = tmp_path / "raw"
    raw.mkdir()
    subprocess.run(
        [
            "rustc",
            str(source),
            "--target",
            HOST,
            "-Cprofile-generate=" + str(raw),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    import os

    subprocess.run(
        [str(binary)],
        check=True,
        env=dict(os.environ, LLVM_PROFILE_FILE=str(raw / "%m-%p.profraw")),
    )
    info = B.output(["rustc", "-Vv"])
    host = info.split("host: ")[1].splitlines()[0]
    llvm = (
        Path(B.output(["rustc", "--print", "sysroot"]))
        / "lib/rustlib"
        / host
        / "bin/llvm-profdata"
    )
    profile = tmp_path / "probe.profdata"
    subprocess.run(
        [str(llvm), "merge", "-o", str(profile), *map(str, raw.glob("*.profraw"))],
        check=True,
    )
    counters = coverage(
        B.output([str(llvm), "show", "--all-functions", "--counts", str(profile)])
    )
    meta = {"identity": {}, "profile_sha256": B.digest(profile), "coverage": {}}
    with pytest.raises(ValueError, match="coverage"):
        B.validate_profile(profile, {}, meta, llvm)
    meta["coverage"] = counters
    B.validate_profile(profile, {}, meta, llvm)
