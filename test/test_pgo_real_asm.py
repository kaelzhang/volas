"""Assembly entry regressions using unmodified LLVM instruction text and real Cargo."""

import json
import os
from pathlib import Path
import shutil
import platform
import subprocess
import sys

import pytest

from test.pgo_toolchain import ROOT, HOST, LLVM, clean_env


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
def test_real_llvm_tabs_targets_and_relocation(tmp_path, monkeypatch, arch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pgo_asm

    # Assemble actual instructions; LLVM chooses its own tabs and opcode aliases.
    arm = ["b", "bl", "cbz x0,", "cbnz x0,", "tbz x0, #0,", "tbnz x0, #1,"]
    arm += [
        "b." + cc
        for cc in (
            "eq",
            "ne",
            "hs",
            "lo",
            "mi",
            "pl",
            "vs",
            "vc",
            "hi",
            "ls",
            "ge",
            "lt",
            "gt",
            "le",
            "al",
            "nv",
        )
    ]
    x86 = ["jmp", "call", "jrcxz", "jecxz", "loop", "loope", "loopne"]
    x86 += [
        "j" + cc
        for cc in (
            "o",
            "no",
            "b",
            "ae",
            "e",
            "ne",
            "be",
            "a",
            "s",
            "ns",
            "p",
            "np",
            "l",
            "ge",
            "le",
            "g",
        )
    ]
    branches = arm if arch == "aarch64" else x86
    indirect = (
        ["br x8", "blr x8"]
        if arch == "aarch64"
        else ["jmp *%rax", "call *%rax", "jmp *0x10(%rip)", "call *0x20(%rip)"]
    )
    normalized = []
    for padding in (0, 32):
        source = tmp_path / f"probe{padding}.s"
        source.write_text(
            ".text\n"
            + ("nop\n" * padding)
            + ".globl _execute\n_execute:\n"
            + "\n".join(b + " 1f" for b in branches)
            + "\n"
            + "\n".join(indirect)
            + "\n1: ret\n"
        )
        obj = source.with_suffix(".o")
        subprocess.run(
            [
                "clang",
                "-target",
                arch + "-linux-gnu",
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            check=True,
            capture_output=True,
        )
        text = subprocess.check_output(
            [str(LLVM / "llvm-objdump"), "-d", "--no-show-raw-insn", str(obj)],
            text=True,
        )
        (tmp_path / f"objdump{padding}.txt").write_text(text)
        # Symbol spelling is the sole adapter; instruction text is untouched.
        text = text.replace("<_execute>:", "<volas_core::frame::execute>:")
        assert "\t" in text
        site = tmp_path / "site"
        site.mkdir(exist_ok=True)
        shutil.copy2(obj, site / "volas_rs.so")
        report = tmp_path / f"report{padding}"
        real_output = pgo_asm.output

        def output(cmd):
            return text if "--disassemble" in cmd else real_output(cmd)

        with monkeypatch.context() as patch:
            patch.setattr(pgo_asm, "output", output)
            patch.setattr(pgo_asm, "manifest", lambda _: {"mode": "normal"})
            patch.setattr(
                sys, "argv", ["pgo_asm", "--site", str(site), "--output", str(report)]
            )
            with pytest.raises(ValueError, match="missing inventory symbols"):
                pgo_asm.main()
        f = json.loads((report / "report.json").read_text())["functions"][
            "volas_core::frame::execute"
        ]
        assert len(f["cfg_edges"]) == len(branches), f["cfg_edges"]
        assert f["cfg_edges"] == [
            [n, len(branches) + len(indirect)] for n in range(len(branches))
        ]
        assert len(f["calls"]) == (2 if arch == "aarch64" else 3)
        normalized.append(f["normalized"])
    assert normalized[0] == normalized[1]


@pytest.mark.parametrize("gate", ["asm_diff.sh", "hot_path_asm.sh"])
def test_real_cargo_redirect_with_stale_probes(tmp_path, gate):
    # A tiny real Cargo workspace avoids rebuilding the product, but invokes the
    # unmodified shell entry, Cargo config resolution, compiler and disassembler.
    scripts = tmp_path / "scripts"
    shutil.copytree(ROOT / "scripts", scripts)
    for filename in ("rust-toolchain.toml", "pyproject.toml"):
        shutil.copy2(ROOT / filename, tmp_path / filename)
    (scripts / "asm_baseline.txt").write_text(
        f"# arch: {platform.machine()})\nprobe_ma exact 1\nprobe_ema exact 1\nprobe_str_eq_scan max 1\n"
    )
    (scripts / "hot_asm_inventory.txt").write_text(
        "volas-core|execute|max|regression\n"
    )
    (scripts / "hot_asm_baseline.txt").write_text(
        f"# arch: {platform.machine()})\nexecute max 1\n"
    )
    (tmp_path / "Cargo.toml").write_text(
        '[workspace]\nmembers=["volas-core","volas-compute"]\nresolver="2"\n'
    )
    for crate, example in [
        ("volas-core", "asm_probe_string"),
        ("volas-compute", "asm_probe_numeric"),
    ]:
        p = tmp_path / crate
        (p / "src").mkdir(parents=True)
        (p / "examples").mkdir()
        (p / "Cargo.toml").write_text(
            f'[package]\nname="{crate}"\nversion="0.0.0"\nedition="2021"\n'
        )
        (p / "src/lib.rs").write_text(
            '#[inline(never)]\npub fn execute() { unsafe { core::arch::asm!("nop", "nop"); } }\n'
        )
        prefix = "_" if sys.platform == "darwin" else ""
        assembly = ".text\n" + "".join(
            f".globl {prefix}{name}\n{prefix}{name}:\nnop\nret\n"
            for name in ("probe_ma", "probe_ema", "probe_str_eq_scan")
        )
        (p / f"examples/{example}.rs").write_text(
            "core::arch::global_asm!(" + json.dumps(assembly) + ");"
            ' extern "C" { fn probe_ma(); fn probe_ema(); fn probe_str_eq_scan(); }'
            " fn main() { unsafe { probe_ma(); probe_ema(); probe_str_eq_scan(); } }\n"
        )
    (tmp_path / ".cargo").mkdir()
    redirect = tmp_path / "cargo redirected"
    (tmp_path / ".cargo/config.toml").write_text(
        "[build]\ntarget-dir=" + json.dumps(str(redirect)) + "\n"
    )
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--no-deps", "--format-version=1"],
            cwd=tmp_path,
            env=clean_env(),
            text=True,
        )
    )
    assert Path(metadata["target_directory"]) == redirect
    # Compile stale probes that pass the baseline into the old default location.
    source = tmp_path / "stale.s"
    prefix = "_" if sys.platform == "darwin" else ""
    source.write_text(
        ".text\n"
        + "".join(
            f".globl {prefix}{name}\n{prefix}{name}:\nret\n"
            for name in ("probe_ma", "probe_ema", "probe_str_eq_scan")
        )
    )
    stale = tmp_path / "target/release/examples"
    stale.mkdir(parents=True)
    for name in ("asm_probe_numeric", "asm_probe_string"):
        subprocess.run(
            [
                "clang",
                "-target",
                HOST,
                "-c",
                str(source),
                "-o",
                str(stale / name),
            ],
            check=True,
            capture_output=True,
        )
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    # git is only used to locate the shell's source root; Cargo is never replaced.
    (bin_ / "git").write_text('#!/bin/sh\nprintf "%s\\n" ' + str(tmp_path) + "\n")
    (bin_ / "git").chmod(0o755)
    env = clean_env(
        PATH=str(bin_) + os.pathsep + os.environ["PATH"],
        VOLAS_PYTHON=sys.executable,
        OBJDUMP=str(LLVM / "llvm-objdump"),
    )
    env.pop("CARGO_TARGET_DIR", None)
    result = subprocess.run(
        ["bash", str(scripts / gate), "check"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0, "stale probe incorrectly passed: " + result.stdout
    assert "CHANGED" in result.stdout or "REGRESSED" in result.stdout, (
        result.stdout + result.stderr
    )
