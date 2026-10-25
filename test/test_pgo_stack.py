"""Stack summaries classify real linked ARM64 instructions through one owner."""

import subprocess
import sys

from test.pgo_linked_imports import FIXTURES, SYMBOL
from test.pgo_toolchain import LLVM, ROOT
from test.test_pgo_final_asm import final_report


def test_linked_arm64_stack_accesses(tmp_path, monkeypatch):
    text = (FIXTURES / "stack.txt").read_text()
    if sys.platform == "darwin":
        # Rebuild the linked ARM64 code on macOS; foreign runners replay the exact
        # captured linked output (no invented instruction or binding records).
        lib = tmp_path / "stack.so"
        subprocess.run(
            ["clang", "-target", "aarch64-apple-darwin", "-shared",
             str(FIXTURES / "stack.s"), "-o", str(lib)], check=True, capture_output=True,
        )
        text = subprocess.check_output(
            [str(LLVM / "llvm-objdump"), "-d", "--no-show-raw-insn",
             "--section-headers", "--syms", str(lib)], text=True,
        )
    _, report = final_report(tmp_path / "report", monkeypatch, text, inventory=("execute", "dispatch", "merge", "indirect"))
    function = report["functions"][SYMBOL]
    stack = function["stack_accesses"]
    assert len(stack) == 18, stack
    assert any(i.startswith("ldur\tq0") for i in stack)
    assert any(i.startswith("stur\t") and "[x29" in i for i in stack)
    assert not any(i.startswith("prfm") or "x19]" in i for i in stack)
    assert not any(i.startswith("ldur\tx0") for i in stack)
    assert report["functions"]["volas_core::frame::dispatch"]["stack_accesses"] == []
    assert report["functions"]["volas_core::frame::merge"]["stack_accesses"] == ["str\tx0, [sp]"]
    assert report["functions"]["volas_core::frame::indirect"]["stack_accesses"] == [
        "ldur\tx1, [x29, #-0x8]", "str\tx1, [sp]", "ldp\tx29, x30, [sp]",
    ]
    # Memory accesses alone cannot prove register pressure caused a spill.
    assert function["confirmed_spills"] == []
    sys.path.insert(0, str(ROOT / "scripts"))
    from asm_symbols import parse, stack_access_indices
    parsed = parse(text)[SYMBOL]
    assert stack == [function["normalized"][i] for i in stack_access_indices(parsed)]
