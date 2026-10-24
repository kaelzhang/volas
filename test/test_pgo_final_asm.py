"""Final-library entry regressions replay unedited LLVM text, including Mach-O."""

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from test.pgo_toolchain import ROOT, HOST, LLVM

FIXTURES = Path(__file__).with_name("fixtures") / "pgo-asm"
EXECUTE = "_volas_core::frame::execute"
CLOSURE = "_volas_core::frame::dispatch::closure"


def final_report(
    tmp_path,
    monkeypatch,
    text,
    base=None,
    empty=False,
    inventory=("execute", "dispatch"),
    missing=False,
):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pgo_asm

    site = tmp_path / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "volas_rs.so").touch()
    report = tmp_path / "report"
    real_output = pgo_asm.output

    def output(cmd):
        return text if str(LLVM / "llvm-objdump") == cmd[0] else real_output(cmd)

    with monkeypatch.context() as patch:
        patch.setattr(pgo_asm, "output", output)
        patch.setattr(pgo_asm, "manifest", lambda _: {"mode": "normal"})
        (tmp_path / "hot_asm_inventory.txt").write_text(
            "".join(f"core|{name}|max|test\n" for name in inventory)
        )
        patch.setattr(pgo_asm, "__file__", str(tmp_path / "pgo_asm.py"))
        argv = ["pgo_asm", "--site", str(site), "--output", str(report)]
        if base:
            argv += ["--base-report", str(base)]
        patch.setattr(sys, "argv", argv)
        if empty:
            with pytest.raises(
                ValueError, match="empty.*instruction|instruction.*empty"
            ):
                pgo_asm.main()
        elif missing:
            with pytest.raises(ValueError, match="missing inventory symbols"):
                pgo_asm.main()
        else:
            pgo_asm.main()
    return report, json.loads(
        (report / "report.json").read_text()
    ) if not empty else None


def test_real_macho_equal_relocation_is_unchanged(tmp_path, monkeypatch):
    # Captured from two genuinely linked libraries, image bases 0 and 0x4000.
    # llvm-objdump --disassemble --demangle --no-show-raw-insn
    #              --section-headers --syms --reloc linked.dylib
    texts = [(FIXTURES / f"macho-{origin}.txt").read_text() for origin in (0, 16384)]
    assert all("\tadrp\t" in text and "dyld_stub_binder+0x" in text for text in texts)
    base, b = final_report(tmp_path / "base", monkeypatch, texts[0])
    _, h = final_report(tmp_path / "head", monkeypatch, texts[1], base)
    assert (
        h["functions"][EXECUTE]["address"] - b["functions"][EXECUTE]["address"] == 16384
    )
    assert h["changed"] == []
    assert not any(
        re.search(r"<[^>]*\+0x", i)
        for f in h["functions"].values()
        for i in f["normalized"]
    )
    assert any("section" in i for i in h["functions"][EXECUTE]["normalized"])


def test_final_entry_rejects_one_empty_selected_body(tmp_path, monkeypatch):
    text = (FIXTURES / "macho-0.txt").read_text()
    # Delete only one selected body's actual instruction lines; other bodies stay.
    text = re.sub(
        r"(<" + re.escape(EXECUTE) + r">:\n).*?(?=\n[0-9a-f]+ <)",
        r"\1",
        text,
        count=1,
        flags=re.S,
    )
    final_report(tmp_path, monkeypatch, text, empty=True)


def test_final_entry_reports_removed_functions(tmp_path, monkeypatch):
    text = (FIXTURES / "macho-0.txt").read_text()
    base, _ = final_report(tmp_path / "base", monkeypatch, text)
    text = re.sub(
        r"^[0-9a-f]+ <" + re.escape(CLOSURE) + r">:\n.*?(?=Disassembly)",
        "",
        text,
        count=1,
        flags=re.S | re.M,
    )
    _, h = final_report(tmp_path / "head", monkeypatch, text, base, missing=True)
    assert h["changed"] == [CLOSURE]
    assert h["removed"] == [CLOSURE]
    assert h["added"] == [] and h["modified"] == []


@pytest.mark.parametrize(
    "target", list(dict.fromkeys([HOST, "aarch64-apple-darwin", "x86_64-linux-gnu"]))
)
def test_real_object_relocations_are_stable(tmp_path, monkeypatch, target):
    # Compile for the running platform's arch/format. LLVM retains relocation
    # records and the exact tabs/operands; no instruction spelling is synthesized.
    arm = target.startswith("aarch64")
    macho = "apple" in target
    prefix = "_" if macho else ""
    symbol = prefix + "volas_core::frame::execute"
    operand = "_payload" if macho else "payload"
    body = (
        (
            f"adrp x0, {operand}@PAGE\nadd x0, x0, {operand}@PAGEOFF\n"
            if macho
            else f"adrp x0, {operand}\nadd x0, x0, :lo12:{operand}\n"
        )
        if arm
        else (f"leaq {operand}(%rip), %rax\n")
    )
    reports = []
    for padding in (0, 4096):
        p = tmp_path / str(padding)
        p.mkdir()
        source = p / "probe.s"
        source.write_text(
            ".text\n"
            + ".space "
            + str(padding)
            + '\n.globl "'
            + symbol
            + '"\n"'
            + symbol
            + '":\n'
            + body
            + "ret\n.data\n"
            + operand
            + ":\n.quad 42\n"
        )
        obj = p / "probe.o"
        subprocess.run(
            ["clang", "-target", target, "-c", str(source), "-o", str(obj)],
            check=True,
            capture_output=True,
        )
        text = subprocess.check_output(
            [
                str(LLVM / "llvm-objdump"),
                "--disassemble",
                "--no-show-raw-insn",
                "--section-headers",
                "--private-headers",
                "--syms",
                "--reloc",
                str(obj),
            ],
            text=True,
        )
        (p / "objdump.txt").write_text(text)
        # Mach-O's assembler inserts ltmp0 at the same address. Bind its label to
        # the real public symbol when LLVM chooses that alias in disassembly.
        text = text.replace("<ltmp0>:", "<" + symbol + ">:") if not padding else text
        assert "RELOC" in text or "R_X86_64" in text or "R_AARCH64" in text
        _, report = final_report(p, monkeypatch, text, inventory=("execute",))
        reports.append(report["functions"][symbol]["normalized"])
    assert reports[0] == reports[1]
    assert any("relocation" in line for line in reports[0])


def test_real_native_linked_equal_relocation(tmp_path, monkeypatch):
    # Rebuild two actual final libraries on the runner too. Linux uses its native
    # ELF linker and ABI; macOS uses its native Mach-O linker and ABI.
    macho = sys.platform == "darwin"
    source = tmp_path / "probe.s"
    source.write_text(
        (FIXTURES / "macho.s").read_text()
        if macho
        else """.text
.globl "volas_core::frame::execute"
"volas_core::frame::execute":
leaq named(%rip), %rax
leaq hidden(%rip), %rcx
call foreign@PLT
ret
.globl "volas_core::frame::dispatch::closure"
"volas_core::frame::dispatch::closure":
ret
.data
.globl named
.hidden named
named:
.quad 42
.space 0x5000
hidden:
.quad 43
.section .note.GNU-stack,"",@progbits
"""
    )
    symbol = EXECUTE if macho else EXECUTE[1:]
    base = None
    reports = []
    for origin in (0, 16384):
        library = tmp_path / f"linked-{origin}.so"
        flags = (
            [
                "-Wl,-undefined,dynamic_lookup,-no_fixup_chains,-install_name,linked.dylib,-image_base,"
                + hex(origin)
            ]
            if macho
            else ["-Wl,-Ttext-segment=" + hex(origin)]
        )
        subprocess.run(
            ["clang", "-shared", str(source), *flags, "-o", str(library)],
            check=True,
            capture_output=True,
        )
        subprocess.run(["strip", "-x", str(library)], check=True, capture_output=True)
        text = subprocess.check_output(
            [
                str(LLVM / "llvm-objdump"),
                "--disassemble",
                "--demangle",
                "--no-show-raw-insn",
                "--syms",
                "--section-headers",
                "--private-headers",
                "--reloc",
                str(library),
            ],
            text=True,
        )
        directory, report = final_report(
            tmp_path / str(origin), monkeypatch, text, base
        )
        reports.append(report)
        base = directory
    assert (
        reports[1]["functions"][symbol]["address"]
        - reports[0]["functions"][symbol]["address"]
        == 16384
    )
    assert reports[1]["changed"] == []


def test_real_macho_same_named_sections_are_distinct_targets(tmp_path, monkeypatch):
    texts = [
        (FIXTURES / f"macho-section-{target}.txt").read_text()
        for target in ("one", "two")
    ]
    # Both targets are anonymous after strip, at offset zero in a __const section.
    assert all(text.count(" __const ") == 2 for text in texts)
    base, _ = final_report(
        tmp_path / "base", monkeypatch, texts[0], inventory=("execute",)
    )
    _, head = final_report(
        tmp_path / "head", monkeypatch, texts[1], base, inventory=("execute",)
    )
    assert head["changed"] == [EXECUTE]
    assert "__DATA_CONST,__const" in head["functions"][EXECUTE]["normalized"][0]
