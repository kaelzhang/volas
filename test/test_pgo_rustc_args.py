"""Rustc argument policies checked against real compiler acceptance and traces."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import build_config as C
from test.pgo_toolchain import (
    ROOT,
    HOST,
    SYSROOT,
    LLVM,
    clean_env,
    extension_link_flags,
)


@pytest.fixture(params=list(dict.fromkeys([HOST, "x86_64-unknown-linux-gnu"])))
def target(request):
    return request.param


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


def real_trace(tmp_path, flags, target, mixed=False, host=False):
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
            *([] if host else ["--target", target]),
            "--print",
            "cfg",
            *extension_link_flags(target),
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
    tmp_path, kind, option, entry, real_profile, target
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
    trace = real_trace(tmp_path, flags, target)
    if entry == "config":
        (tmp_path / ".cargo").mkdir()
        (tmp_path / ".cargo/config.toml").write_text(
            "[build]\nrustflags=" + json.dumps(flags) + "\n"
        )
        with pytest.raises(ValueError, match="feedback|codegen override"):
            C.check_cargo_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="feedback|codegen override"):
            C.validate_argv(trace, "normal", target)


@pytest.mark.parametrize("kind", SPELLINGS)
@pytest.mark.parametrize("mode", ["generate", "use"])
@pytest.mark.parametrize(
    "damage", ["valid", "wrong-path", "duplicate", "relative", "mixed", "host"]
)
def test_real_feedback_trace_mode_count_and_path(
    tmp_path, kind, mode, damage, real_profile, target
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
    trace = real_trace(tmp_path, flags, target, mixed=damage == "mixed")
    if damage == "host":
        real_trace(tmp_path, flags, target, host=True)
    expected = str(tmp_path / "other") if damage == "wrong-path" else path
    if damage == "valid":
        C.validate_argv(trace, mode, target, expected)
    else:
        with pytest.raises(ValueError, match="feedback"):
            C.validate_argv(
                trace,
                mode,
                target,
                expected,
            )


@pytest.mark.parametrize("kind", SPELLINGS)
def test_real_bare_generate_is_rejected_without_explicit_path(tmp_path, kind, target):
    trace = real_trace(tmp_path, spell(kind, "profile_generate"), target)
    with pytest.raises(ValueError, match="feedback"):
        C.validate_argv(trace, "generate", target, str(tmp_path / "raw"))


@pytest.mark.parametrize("entry", ["config", "resolved", "trace"])
@pytest.mark.parametrize("kind", SPELLINGS)
def test_real_response_file_rejected_by_every_entry(
    tmp_path, entry, kind, real_profile, target
):
    response = tmp_path / "rustc response args"
    # rustc reads one argument per line, including split codegen spellings.
    response.write_text(
        "\n".join(
            spell(kind, "profile-use=" + str(real_profile)) + spell(kind, "opt-level=0")
        )
        + "\n"
    )
    flags = ["@" + str(response)]
    trace = real_trace(tmp_path, flags, target)
    if entry == "trace":
        with pytest.raises(ValueError, match="response.file"):
            C.validate_argv(trace, "normal", target)
    elif entry == "resolved":
        with pytest.raises(ValueError, match="response.file"):
            C.resolved_flags(tmp_path, target, flags, tmp_path / "resolve.log")
    else:
        (tmp_path / ".cargo").mkdir()
        (tmp_path / ".cargo/config.toml").write_text(
            "[build]\nrustflags=" + json.dumps(flags) + "\n"
        )
        with pytest.raises(ValueError, match="response.file"):
            C.check_cargo_config(tmp_path)


def test_real_response_file_cannot_hide_an_invocation_from_trace_filter(
    tmp_path, target
):
    trace = real_trace(tmp_path, [], target)  # A separate legitimate cdylib invocation.
    response = tmp_path / "hidden args"
    response.write_text("--crate-name\nhidden\n--print\ncfg\n--codegen=opt-level=0\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/rustc_record.py"),
            str(SYSROOT / "bin/rustc"),
            "@" + str(response),
        ],
        env=clean_env(VOLAS_RUSTC_TRACE=str(trace)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match="response.file"):
        C.validate_argv(trace, "normal", target)
