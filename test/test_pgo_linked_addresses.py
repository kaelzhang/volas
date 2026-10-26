"""Resolved operands from real final linked libraries retain target identities."""

import subprocess
import sys

import pytest

from test.pgo_toolchain import HOST, LLVM
from test.test_pgo_final_asm import final_report


@pytest.mark.parametrize(
    "target",
    list(
        dict.fromkeys(
            [HOST, "x86_64-apple-darwin"] if sys.platform == "darwin" else [HOST]
        )
    ),
)
@pytest.mark.parametrize(
    "change", ["within-page", "page-start", "relative", "target", "immediate"]
)
def test_linked_address_operands(tmp_path, monkeypatch, target, change):
    arm, apple = target.startswith("aarch64"), "apple" in target
    prefix = "_" if apple else ""
    symbol = prefix + "volas_core::frame::execute"
    reports, base = [], None
    for variant in (0, 1):
        p = tmp_path / str(variant)
        p.mkdir()
        selected = "other" if variant and change == "target" else "named"
        data_pad = 32 if variant and change == "within-page" else 16
        if change == "page-start":
            data_pad = 32 if variant else 0
        text_pad = 32 if variant and change == "relative" else 0
        immediate = 9 if variant and change == "immediate" else 7
        operand = prefix + selected
        if arm:
            page = operand + "@PAGE" if apple else operand
            off = operand + "@PAGEOFF" if apple else ":lo12:" + operand
            body = (
                f"adrp x0, {page}\nadd x1, x0, {off}\n"
                f"ldr x2, [x0, {off}]\nadd x3, x3, #{immediate}\n"
            )
        else:
            body = f"leaq {operand}(%rip), %rax\nmovq {operand}+8(%rip), %rcx\naddq ${immediate}, %rdx\n"
        data = ""
        for name in ("named", "other"):
            name = prefix + name
            data += (
                f".globl {name}\n"
                + (f".private_extern {name}\n" if apple else f".hidden {name}\n")
                + f"{name}:\n.quad 42,43\n"
            )
        source = p / "probe.s"
        source.write_text(
            f'.text\n.space {text_pad}\n.globl "{symbol}"\n"{symbol}":\n'
            + body
            + f"ret\n.data\n.space {data_pad}\n"
            + data
            + ("" if apple else '.section .note.GNU-stack,"",@progbits\n')
        )
        library = p / "linked.so"
        flags = ["-Wl,-no_fixup_chains,-install_name,linked.dylib"] if apple else []
        subprocess.run(
            [
                "clang",
                "-target",
                target,
                "-shared",
                str(source),
                *flags,
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
        )
        text = subprocess.check_output(
            [
                str(LLVM / "llvm-objdump"),
                "-d",
                "--no-show-raw-insn",
                "--section-headers",
                "--private-headers",
                "--syms",
                "--reloc",
                str(library),
            ],
            text=True,
        )
        (p / "objdump.txt").write_text(text)
        directory, report = final_report(
            p, monkeypatch, text, base, inventory=("execute",)
        )
        reports.append(report)
        base = directory
    assert reports[1]["changed"] == (
        [symbol] if change in ("target", "immediate") else []
    )
    normalized = reports[1]["functions"][symbol]["normalized"]
    assert any("@symbol(" + prefix + selected + ",0)" in i for i in normalized)
    if not arm:
        assert any("@symbol(" + prefix + selected + ",8)" in i for i in normalized)
