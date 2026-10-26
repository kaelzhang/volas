"""Link provider-qualified imports, including non-GOT data pointers."""

import json
import subprocess
import sys

from test.pgo_linked_imports import SYMBOL, FIXTURES, TARGETS
from test.pgo_toolchain import LLVM


def provider_dump(
    directory, target, variant="base", chains=False, version=None, custom_stub=False
):
    directory.mkdir(parents=True)
    apple, arm = "apple" in target, target.startswith("aarch64")
    prefix = "_" if apple else ""
    provider = "changedx" if variant == "provider" else "provider"
    install = f"@rpath/{provider}.dylib" if apple else f"{provider}.so"
    (directory / "provider.s").write_text(
        f".text\n.p2align 2\n.globl {prefix}foreign\n{prefix}foreign:\nret\n"
        f".globl {prefix}aaa\n{prefix}aaa:\nret\n"
    )

    def run(cmd):
        subprocess.run(cmd, cwd=directory, check=True, capture_output=True)

    run(["clang", "-target", target, "-c", "provider.s", "-o", "provider.o"])
    if apple:
        run(
            [
                "clang",
                "-target",
                target,
                "-shared",
                "provider.o",
                "-o",
                "provider.dylib",
                "-Wl,-install_name," + install,
            ]
        )
    else:
        (directory / "versions.map").write_text(
            f"{version or 'V1'} {{ global: foreign; aaa; }};\n"
        )
        run(
            [
                "ld.lld" if sys.platform == "darwin" else "ld",
                "-shared",
                "provider.o",
                "-soname",
                install,
                "--version-script=versions.map",
                "-o",
                "provider.so",
            ]
        )
    foreign = prefix + "foreign"
    moved = variant == "moved"
    if arm:
        page = foreign + "@GOTPAGE" if apple else ":got:" + foreign
        off = foreign + "@GOTPAGEOFF" if apple else ":got_lo12:" + foreign
        body = f"bl {foreign}\nadrp x0, {page}\nldr x0, [x0, {off}]\nblr x0\n"
        page = "slot@PAGE" if apple else "slot"
        off = "slot@PAGEOFF" if apple else ":lo12:slot"
        body += f"adrp x1, {page}\nldr x1, [x1, {off}]\n"
    else:
        body = f"call {foreign}{'' if apple else '@PLT'}\nmovq {foreign}@GOTPCREL(%rip), %rax\ncall *%rax\nmovq slot(%rip), %rax\n"
    extra = ""
    if custom_stub:
        if not apple or not arm:
            raise ValueError("custom stub fixture requires ARM64 Mach-O")
        body = body.replace("bl " + foreign, "bl custom_stub")
        extra = (
            ".section __TEXT,__stubs,symbol_stubs,pure_instructions,12\n"
            + f".indirect_symbol {foreign}\ncustom_stub:\n"
            + "adrp x16, slot@PAGE\nldr x16, [x16, slot@PAGEOFF]\nbr x16\n"
        )
    (directory / "probe.s").write_text(
        f".text\n.p2align 2\n.globl {prefix}anchor\n{prefix}anchor:\n"
        + (f"{'bl' if arm else 'call'} {prefix}aaa\n" if moved else "")
        + "ret\n"
        + f'.globl "{SYMBOL}"\n"{SYMBOL}":\n{body}ret\n.data\n.p2align 3\n'
        + (".space 32\n" if moved else "")
        + f"slot:\n.quad {foreign}+8\n"
        + extra
    )
    run(["clang", "-target", target, "-c", "probe.s", "-o", "probe.o"])
    if apple:
        cmd = [
            "clang",
            "-target",
            target,
            "-shared",
            "probe.o",
            "provider.dylib",
            "-Wl,-install_name,linked.so",
        ] + ([] if chains else ["-Wl,-no_fixup_chains"])
    else:
        cmd = [
            "ld.lld" if sys.platform == "darwin" else "ld",
            "-shared",
            "probe.o",
            "provider.so",
        ]
    if custom_stub:
        # The modern Apple linker asserts on object-defined symbol stubs.
        # Classic rewrites them: traditional uses a literal lazy-pointer load,
        # chained uses an ADRP/GOT pair. Both actually bind foreign with +0.
        cmd += ["-Wl,-ld_classic"]
        if chains:
            cmd += ["-Wl,-fixup_chains"]
    run([*cmd, "-o", "linked.so"])
    sys.path.insert(0, str(FIXTURES.parents[2] / "scripts"))
    from asm_references import disassemble

    text = disassemble(
        LLVM / "llvm-objdump",
        "linked.so",
        lambda cmd: subprocess.check_output(cmd, cwd=directory, text=True),
    )
    (directory / "objdump.txt").write_text(text)
    return text


if __name__ == "__main__":
    from pathlib import Path

    scratch = Path(sys.argv[1]).resolve()
    captures = {}
    for target in TARGETS:
        for chains in [False, True] if "apple" in target else [False]:
            for variant in ("base", "provider", "moved"):
                key = f"{target}-{chains}-{variant}"
                captures[key] = provider_dump(scratch / key, target, variant, chains)
        if target == "aarch64-apple-darwin":
            for chains in (False, True):
                key = f"{target}-{chains}-custom"
                captures[key] = provider_dump(
                    scratch / key, target, chains=chains, custom_stub=True
                )
        if "linux" in target:
            key = f"{target}-version-V2"
            captures[key] = provider_dump(scratch / key, target, version="V2")
    (FIXTURES / "bindings.json").write_text(json.dumps(captures, indent=2) + "\n")
