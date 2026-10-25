"""Build real import-slot variants; capture foreign formats for portable replay."""

from pathlib import Path
import subprocess
import sys

from test.pgo_toolchain import LLVM, HOST

SYMBOL = "volas_core::frame::execute"
FIXTURES = Path(__file__).with_name("fixtures") / "pgo-imports"
TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "aarch64-linux-gnu",
    "x86_64-linux-gnu",
)


def linked_dump(directory, target, variant="base", metadata=True, chains=False, page_boundary=False):
    directory.mkdir(parents=True, exist_ok=True)
    apple, arm = "apple" in target, target.startswith("aarch64")
    prefix = "_" if apple else ""
    imported = prefix + ("changed" if variant == "target" else "foreign")
    call = ("bl " if arm else "call ") + imported
    if not apple and not arm:
        call += "@PLT"
    if arm:
        page = imported + "@GOTPAGE" if apple else ":got:" + imported
        off = imported + "@GOTPAGEOFF" if apple else ":got_lo12:" + imported
        body = f"{call}\nadrp x0, {page}\nldr x0, [x0, {off}]\nblr x0\n"
    else:
        body = f"{call}\nmovq {imported}@GOTPCREL(%rip), %rax\ncall *%rax\n"
    if page_boundary:
        assert arm
        body = f"adrp x0, {page}\nb 1f\n1:\nldr x0, [x0, {off}]\nblr x0\n"
    anchor = ("bl " if arm else "call ") + prefix + "aaa\n" if variant == "moved" else ""
    if page_boundary:
        anchor = "bl " + imported + "\n"
    source = (
        f'.text\n.globl {prefix}anchor\n{prefix}anchor:\n{anchor}ret\n'
        f'.globl "{SYMBOL}"\n"{SYMBOL}":\n{body}ret\n'
        + ("" if apple else '.section .note.GNU-stack,"",@progbits\n')
    )
    (directory / "probe.s").write_text(source)
    subprocess.run(
        ["clang", "-target", target, "-c", "probe.s", "-o", "probe.o"],
        cwd=directory, check=True, capture_output=True,
    )
    if apple:
        cmd = ["clang", "-target", target, "-shared", "probe.o",
               "-Wl,-undefined,dynamic_lookup,-install_name,linked.dylib"]
        if not chains:
            cmd += ["-Wl,-no_fixup_chains"]
    elif sys.platform == "darwin":
        cmd = ["ld.lld", "-shared", "probe.o"]
    else:
        cmd = ["clang", "-shared", "probe.o", "-nostdlib"]
    subprocess.run([*cmd, "-o", "linked.so"], cwd=directory, check=True, capture_output=True)
    flags = ["-d", "--demangle", "--no-show-raw-insn", "--section-headers",
             "--private-headers", "--syms", "--reloc"]
    if not apple and metadata:
        flags += ["--dynamic-reloc"]
    text = subprocess.check_output(
        [str(LLVM / "llvm-objdump"), *flags, "linked.so"], cwd=directory, text=True,
    )
    if apple and metadata:
        text += "\n" + subprocess.check_output(
            [str(LLVM / "llvm-objdump"), "--macho", "--indirect-symbols", "--bind",
             "--lazy-bind", "--weak-bind", "--dyld-info", "linked.so"],
            cwd=directory, text=True,
        )
    # Keep decoded instructions/tables intact while making captures diff-clean.
    text = "\n".join(line.rstrip(" \t") for line in text.splitlines()) + "\n"
    (directory / "objdump.txt").write_text(text)
    return text


if __name__ == "__main__":
    # Explicit maintainer command; cross-ELF capture on macOS requires ld.lld.
    scratch = Path(sys.argv[1]).resolve()
    FIXTURES.mkdir(exist_ok=True)
    for target in TARGETS:
        for variant in ("base", "target", "moved"):
            text = linked_dump(scratch / target / variant, target, variant)
            (FIXTURES / f"{target}-{variant}.txt").write_text(text)
        if target.startswith("aarch64"):
            for variant in ("base", "target"):
                text = linked_dump(scratch / ("page-" + target) / variant, target, variant, page_boundary=True)
                (FIXTURES / f"page-{target}-{variant}.txt").write_text(text)
    print("captured linked imports with", HOST)
