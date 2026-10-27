#!/usr/bin/env python3
"""Report final-wheel assembly separately; never rewrite normal baselines."""

import argparse
import json
import re
from pathlib import Path

from build_pair import manifest
from build_release import output, source_hash


from asm_symbols import is_branch, is_call, parse, stack_access_indices, target_operand
from asm_references import References, disassemble



def inline_carriers(source, functions):
    """Locate source-reviewed containers, never count an absent body as zero.

    Linked release libraries omit wrapper/inlined symbols after LTO. These
    mappings retain the entire actual container for review; its instruction
    count is not an isolated count for the missing inventory function.
    Source provenance is checked by the entry point before using mappings.
    """
    mappings = {
        "fold_forming_row": (
            "crates/volas-python/src/frame_tf.rs",
            "inner.fold_forming_row(inner.height() - 1, bar, 0, ops)",
            r"volas_rs::frame_tf::.*::fold_bars(?:::h[0-9a-f]{16})?$",
            "source-callsite-container",
        ),
        "execute": (
            "crates/volas-directive/src/exec.rs",
            "execute_node(df, node.node())",
            r"volas_directive::exec::execute_node(?:::h[0-9a-f]{16})?$",
            "forwarding-wrapper-target",
        ),
    }
    evidence = {}
    for name, (path, statement, pattern, kind) in mappings.items():
        lines = (source / path).read_text().splitlines()
        locations = [n + 1 for n, line in enumerate(lines) if statement in line]
        hits = [symbol for symbol in functions if re.search(pattern, symbol)]
        if len(locations) == 1 and hits:
            evidence[name] = {
                "kind": kind,
                "source": {"path": path, "line": locations[0], "statement": statement},
                "containers": hits,
                "isolated_instruction_count": None,
            }
    return evidence


def inventory_complete(functions, missing):
    return not missing and not any(
        "@unresolved(" in i for f in functions.values() for i in f["normalized"]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base-report", type=Path)
    ap.add_argument("--source", type=Path, help="matching source for reviewed LTO containers")
    args = ap.parse_args()
    build = manifest(args.site)
    if args.source and source_hash(args.source) != build["identity"]["source"]:
        raise ValueError("inline report source provenance mismatch")
    host = next(
        line.split(": ")[1]
        for line in output(["rustc", "-Vv"]).splitlines()
        if line.startswith("host:")
    )
    objdump = (
        Path(output(["rustc", "--print", "sysroot"]))
        / "lib/rustlib"
        / host
        / "bin/llvm-objdump"
    )
    lib = next(args.site.rglob("volas_rs*.so"))
    text = disassemble(objdump, lib, output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "final-library.asm").write_text(text + "\n")
    functions = parse(text)
    references = References(text, functions)
    inventory = [
        line.split("|")[1]
        for line in Path(__file__)
        .with_name("hot_asm_inventory.txt")
        .read_text()
        .splitlines()
        if line
    ]
    carriers = inline_carriers(args.source, functions) if args.source else {}
    selected, missing, standalone_missing, inline_sites = {}, [], [], {}
    for name in inventory:
        hits = {
            s: f
            for s, f in functions.items()
            if re.search(r"::" + name + r"(?:\(|::|$)", s)
        }
        if not hits:
            standalone_missing.append(name)
            if name in carriers:
                inline_sites[name] = carriers[name]
                hits = {s: functions[s] for s in carriers[name]["containers"]}
            else:
                missing.append(name)
        for symbol, f in hits.items():
            ins = f["instructions"]
            if not ins:
                raise ValueError(f"empty instruction body in final library: {symbol}")
            addresses = {i["address"]: n for n, i in enumerate(ins)}
            normalized, edges, calls = [], [], []
            for n, i in enumerate(ins):
                body = i["text"]
                target = target_operand(body)
                if (
                    target
                    and int(target[1], 16) in addresses
                    and not references.instruction_relocations(i, f)
                ):
                    dest = addresses[int(target[1], 16)]
                    if is_branch(body):
                        edges.append([n, dest])
                body = references.normalize(i, f, addresses)
                normalized.append(body)
                if is_call(body):
                    calls.append(body)
            selected[symbol] = dict(
                address=f["address"],
                mod32=f["address"] % 32,
                mod64=f["address"] % 64,
                normalized=normalized,
                cfg_edges=edges,
                calls=calls,
                stack_accesses=[normalized[i] for i in stack_access_indices(f)],
                # A load/store does not establish why the compiler used memory.
                confirmed_spills=[],
                unresolved_references=[i for i in normalized if "@unresolved(" in i],
                nop=sum(i == "nop" for i in normalized),
                non_nop=sum(i != "nop" for i in normalized),
            )
    report = {
        "build": build,
        "functions": selected,
        "missing_symbols": missing,
        "standalone_missing_symbols": standalone_missing,
        "inline_sites": inline_sites,
        "status": "review-required",
        "comparison_complete": inventory_complete(selected, missing),
    }
    base_missing = []
    if args.base_report:
        base_report = json.loads((args.base_report / "report.json").read_text())
        base = base_report["functions"]
        base_missing = base_report.get("missing_symbols", [])
        report["base_missing_symbols"] = base_missing
        report["added"] = sorted(selected.keys() - base.keys())
        report["removed"] = sorted(base.keys() - selected.keys())
        report["modified"] = sorted(
            s
            for s in selected.keys() & base.keys()
            if selected[s]["normalized"] != base[s]["normalized"]
            or any("@unresolved(" in i for i in selected[s]["normalized"] + base[s]["normalized"])
        )
        report["comparison_complete"] &= inventory_complete(base, base_missing)
        report["changed"] = sorted(
            report["added"] + report["removed"] + report["modified"]
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if missing:
        raise ValueError(
            f"final library missing inventory symbols (inlining must be reviewed): {missing}"
        )
    if base_missing:
        raise ValueError(f"base report missing inventory symbols: {base_missing}")
    print("PGO final-library report: review required; normal baseline unchanged")


if __name__ == "__main__":
    main()
