"""The assembly gate must count its exact function and fail closed if absent."""
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _count(tmp_path, assembly, function="fold_forming_row"):
    fixture = tmp_path / "functions.s"
    fixture.write_text(assembly)
    source = (ROOT / "scripts/hot_path_asm.sh").read_text()
    definitions = source[source.index("count_fn() {"):source.index("\nbaseline_count()")]
    script = definitions + '\nasm_of() { printf "%s\\n" "$ASM_FIXTURE"; }\ncount_fn nokeep core "$COUNT_FUNCTION" max\n'
    result = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True,
                            env={**os.environ, "ASM_FIXTURE": str(fixture), "COUNT_FUNCTION": function})
    return int(result.stdout.strip())


def _body(name, instructions):
    return f"__ZN1x{len(name)}{name}17h1234567890abcdefE:\n" + "".join(f"\t{op}\n" for op in instructions) + "\t.cfi_endproc\n"


def test_hot_gate_selects_the_exact_function_instead_of_a_suffix(tmp_path):
    assembly = _body("refold_forming_row", ["nop", "nop", "nop", "ret"])
    assembly += _body("fold_forming_row", ["nop", "ret"])
    assert _count(tmp_path, assembly) == 2


def test_hot_gate_missing_function_cannot_pass_as_zero_instructions(tmp_path):
    assembly = _body("refold_forming_row", ["nop", "ret"])
    assert _count(tmp_path, assembly) == 999999


def test_hot_gate_empty_assembly_fails_closed(tmp_path):
    assert _count(tmp_path, "") == 999999
