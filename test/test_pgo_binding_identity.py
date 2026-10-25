"""Binding equality includes the provider namespace, version and addend."""

import json
import re

import pytest

from test.pgo_binding_libraries import provider_dump
from test.pgo_linked_imports import FIXTURES, SYMBOL, TARGETS
from test.pgo_toolchain import HOST
from test.test_pgo_imports import compare


@pytest.mark.parametrize("chains", [False, True])
@pytest.mark.parametrize("variant", ["provider", "moved"])
def test_linked_provider_identity(tmp_path, monkeypatch, chains, variant):
    texts = [provider_dump(tmp_path / v, HOST, v, chains) for v in ("base", variant)]
    check_binding(tmp_path / "reports", monkeypatch, texts, variant, HOST)


def check_binding(directory, monkeypatch, texts, variant, target):
    pattern = (
        r"^__\S+\s+\S+\s+(0x\S+).*foreign$"
        if "apple" in target
        else r"^([0-9a-f]+)\s+R_\S+\s+foreign(?:\+0x8)?$"
    )
    rows = [re.findall(pattern, t, re.M) for t in texts]
    assert len(rows[0]) >= 2 and len(rows[0]) == len(rows[1])
    assert (rows[0] != rows[1]) == (variant == "moved")
    _, head = compare(directory, monkeypatch, texts)
    assert head["changed"] == ([SYMBOL] if variant == "provider" else [])
    assert head["comparison_complete"]
    refs = [i for i in head["functions"][SYMBOL]["normalized"] if "@import(" in i]
    assert len(refs) >= (4 if target.startswith("aarch64") else 3)
    assert all(("changedx" if variant == "provider" else "provider") in i for i in refs)
    assert any("+8" in i for i in refs)


@pytest.mark.parametrize(
    "target,chains",
    [(t, c) for t in TARGETS for c in ([False, True] if "apple" in t else [False])],
)
@pytest.mark.parametrize("variant", ["provider", "moved"])
def test_captured_provider_identity(tmp_path, monkeypatch, target, chains, variant):
    captures = json.loads((FIXTURES / "bindings.json").read_text())
    texts = [captures[f"{target}-{chains}-{v}"] for v in ("base", variant)]
    check_binding(tmp_path, monkeypatch, texts, variant, target)


@pytest.mark.parametrize("target", [t for t in TARGETS if "linux" in t])
@pytest.mark.parametrize("version", ["V1", "V2"])
def test_linked_elf_symbol_version(tmp_path, monkeypatch, target, version):
    captures = json.loads((FIXTURES / "bindings.json").read_text())
    texts = [
        captures[f"{target}-False-base"],
        captures[f"{target}-version-V2"]
        if version == "V2"
        else captures[f"{target}-False-base"],
    ]
    if HOST == target.replace("linux", "unknown-linux"):
        texts = [
            provider_dump(tmp_path / str(n), target, version=v)
            for n, v in enumerate(("V1", version))
        ]
    _, head = compare(tmp_path / "reports", monkeypatch, texts)
    assert head["changed"] == ([SYMBOL] if version == "V2" else [])
    assert head["comparison_complete"]
    refs = [i for i in head["functions"][SYMBOL]["normalized"] if "@import(" in i]
    assert refs and all(version in i for i in refs)


def test_ambiguous_provider_is_unresolved(tmp_path, monkeypatch):
    text = (FIXTURES / "chained-8.txt").read_text()
    # Two valid load commands whose install names yield LLVM's same short name.
    load = text[text.index("Load command 12\n") : text.index("Load command 13\n")]
    text += "\n" + load.replace("@rpath/provider.dylib", "/other/provider.dylib")
    _, head = compare(tmp_path, monkeypatch, [text, text])
    assert head["changed"] == [SYMBOL]
    assert not head["comparison_complete"]


def test_binding_components_cannot_collide_in_normalized_text():
    import sys
    from test.pgo_toolchain import ROOT

    sys.path.insert(0, str(ROOT / "scripts"))
    from asm_references import References

    references = References("", {})
    bindings = [
        ("two-level:1:@rpath/y.dylib", "foreign,two-level:1:@rpath/x.dylib", 0),
        ("two-level:1:@rpath/x.dylib,two-level:1:@rpath/y.dylib", "foreign", 0),
        ("two-level:1:@rpath/y.dylib", "foreign%2Ctwo-level:1:@rpath/x.dylib", 0),
    ]
    references.imports = {n: {b} for n, b in enumerate(bindings)}
    normalized = [references.reference(n, {}) for n in range(len(bindings))]
    assert len(set(normalized)) == len(bindings)
    references.imports[1000] = references.imports[0]
    assert references.reference(1000, {}) == normalized[0]


@pytest.mark.parametrize("target_unknown", [False, True])
def test_unknown_binding_cannot_be_hidden_by_another_slot(target_unknown):
    import sys
    from test.pgo_toolchain import ROOT

    sys.path.insert(0, str(ROOT / "scripts"))
    from asm_references import References
    from asm_symbols import parse

    text = (FIXTURES / "chained-8.txt").read_text()
    # The stub reaches a pointer with unknown provider. A separate pointer for
    # the same symbol must not supply its known provider instead.
    text = re.sub(
        r"(__DATA\s+__data\s+0x8000\s+\S+\s+bind)\s+0x8\s+provider",
        r"\1 0x0 unknown-provider",
        text,
    )
    if target_unknown:
        text = text.replace("adrp\tx16, 0x4000", "adrp\tx16, 0x8000")
    references = References(text, parse(text))
    call = references.normalize(
        parse(text)[SYMBOL]["instructions"][0], parse(text)[SYMBOL], {}
    )
    assert ("@unresolved(" in call) == target_unknown


@pytest.mark.parametrize("chains", [False, True])
def test_linked_stub_uses_its_reached_pointer(tmp_path, monkeypatch, chains):
    target = "aarch64-apple-darwin"
    text = json.loads((FIXTURES / "bindings.json").read_text())[
        f"{target}-{chains}-custom"
    ]
    if HOST == target:
        text = provider_dump(
            tmp_path / "linked", target, chains=chains, custom_stub=True
        )
    _, report = compare(tmp_path / "reports", monkeypatch, [text, text])
    assert report["comparison_complete"]
    assert report["changed"] == []
    assert report["functions"][SYMBOL]["calls"][0].endswith(
        "@import(_foreign,two-level:1:@rpath/provider.dylib,+0)"
    )


@pytest.mark.parametrize("addend", [0, 8])
def test_literal_stub_keeps_its_actual_pointer_binding(addend):
    import sys
    from test.pgo_toolchain import ROOT

    sys.path.insert(0, str(ROOT / "scripts"))
    from asm_references import References
    from asm_symbols import parse

    text = json.loads((FIXTURES / "bindings.json").read_text())[
        "aarch64-apple-darwin-False-custom"
    ]
    # The real linker rewrites the supplied stub to a literal load of its lazy
    # pointer, not the +8 data pointer in the source. The nonzero case changes
    # only decoded metadata to check that binding addends are never inferred.
    if addend:
        text, changed = re.subn(
            r"(__DATA\s+__la_symbol_ptr\s+0x0*8000)\s+provider\s+_foreign",
            r"\1 pointer 8 provider _foreign",
            text.replace("Lazy bind table:", "Bind table:"),
        )
        assert changed == 1
    functions = parse(text)
    references = References(text, functions)
    function = functions[SYMBOL]
    call = references.normalize(function["instructions"][0], function, {})
    assert call.endswith(
        f"@import(_foreign,two-level:1:@rpath/provider.dylib,+{addend})"
    )
