"""Rustc argument policies checked against real compiler acceptance and traces."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import build_config as C
from test.pgo_toolchain import ROOT, HOST, SYSROOT, LLVM, clean_env


SPELLINGS = ("short_joined", "short_split", "long_joined", "long_split")


def spell(kind, option):
    return {
        "short_joined": ["-C" + option],
        "short_split": ["-C", option],
        "long_joined": ["--codegen=" + option],
        "long_split": ["--codegen", option],
    }[kind]


@pytest.fixture(scope="module")
def real_profile(tmp_path_factory):
    p = tmp_path_factory.mktemp("compiler-profile")
    (p / "probe.rs").write_text("fn main() { std::hint::black_box(42); }")
    subprocess.run(
        [
            "rustc",
            str(p / "probe.rs"),
            "-Cprofile-generate=" + str(p / "raw"),
            "-o",
            str(p / "probe"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(p / "probe")],
        env=clean_env(LLVM_PROFILE_FILE=str(p / "%m-%p.profraw")),
        check=True,
    )
    profile = p / "probe.profdata"
    subprocess.run(
        [
            str(LLVM / "llvm-profdata"),
            "merge",
            "-o",
            str(profile),
            *map(str, p.glob("*.profraw")),
        ],
        check=True,
    )
    return profile


def real_trace(tmp_path, flags, mixed=False, host=False):
    trace = tmp_path / "rustc.jsonl"
    # rustc itself validates these spellings; the real wrapper records the argv.
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/rustc_record.py"),
            str(SYSROOT / "bin/rustc"),
            "--crate-name",
            "volas_rs",
            "--crate-type",
            "cdylib",
            *([] if host else ["--target", HOST]),
            "--print",
            "cfg",
            "-Clink-arg=-undefined",
            "-Clink-arg=dynamic_lookup",
            *flags,
        ],
        env=clean_env(VOLAS_RUSTC_TRACE=str(trace)),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if mixed:
        assert result.returncode != 0 and "are exclusive" in result.stderr, (
            result.stderr
        )
    else:
        assert result.returncode == 0, result.stderr
    return trace


@pytest.mark.parametrize("kind", SPELLINGS)
@pytest.mark.parametrize(
    "option",
    [
        "profile-use",
        "profile_use",
        "profile-generate",
        "profile_generate",
        "opt-level",
        "opt_level",
        "codegen-units",
        "codegen_units",
        "lto",
        "linker",
        "link-arg",
        "link_arg",
        "link-args",
        "link_args",
        "llvm-args",
        "llvm_args",
    ],
)
@pytest.mark.parametrize("entry", ["config", "trace"])
def test_rustc_accepted_spelling_cannot_bypass_policy(
    tmp_path, kind, option, entry, real_profile
):
    name = option.replace("_", "-")
    value = (
        "0"
        if name == "opt-level"
        else "2"
        if name == "codegen-units"
        else "thin"
        if name == "lto"
        else str(tmp_path / "foreign")
    )
    if name == "llvm-args":
        value = "-pgo-warn-missing-function"
    if name == "profile-use":
        Path(value).write_bytes(real_profile.read_bytes())
    flags = spell(kind, option + "=" + value)
    trace = real_trace(tmp_path, flags)
    if entry == "config":
        (tmp_path / ".cargo").mkdir()
        (tmp_path / ".cargo/config.toml").write_text(
            "[build]\nrustflags=" + json.dumps(flags) + "\n"
        )
        with pytest.raises(ValueError, match="feedback|codegen override"):
            C.check_cargo_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="feedback|codegen override"):
            C.validate_argv(None, "normal", HOST, trace=trace)


@pytest.mark.parametrize("kind", SPELLINGS)
@pytest.mark.parametrize("mode", ["generate", "use"])
@pytest.mark.parametrize(
    "damage", ["valid", "wrong-path", "duplicate", "relative", "mixed", "host"]
)
def test_real_feedback_trace_mode_count_and_path(
    tmp_path, kind, mode, damage, real_profile
):
    path = (
        "relative" if damage == "relative" else str(tmp_path / "feedback with spaces")
    )
    (tmp_path / path).write_bytes(real_profile.read_bytes())
    flags = spell(kind, "profile_" + mode + "=" + path)
    if damage == "duplicate":
        flags *= 2
    if damage == "mixed":
        flags += spell(
            kind,
            "profile_" + ("use" if mode == "generate" else "generate") + "=" + path,
        )
    trace = real_trace(tmp_path, flags, mixed=damage == "mixed")
    if damage == "host":
        real_trace(tmp_path, flags, host=True)
    expected = str(tmp_path / "other") if damage == "wrong-path" else path
    if damage == "valid":
        C.validate_argv(None, mode, HOST, expected, trace=trace)
    else:
        with pytest.raises(ValueError, match="feedback"):
            C.validate_argv(
                None,
                mode,
                HOST,
                expected,
                trace=trace,
            )


@pytest.mark.parametrize("kind", SPELLINGS)
def test_real_bare_generate_is_rejected_without_explicit_path(tmp_path, kind):
    trace = real_trace(tmp_path, spell(kind, "profile_generate"))
    with pytest.raises(ValueError, match="feedback"):
        C.validate_argv(None, "generate", HOST, str(tmp_path / "raw"), trace=trace)
