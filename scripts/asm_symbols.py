#!/usr/bin/env python3
"""Fail-closed parsing shared by probe counts and final-wheel assembly reports."""

import argparse
import os
import re
import subprocess
import sys


def parse(text):
    functions = {}
    current = None
    for line in text.splitlines():
        header = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if header:
            address, symbol = header.groups()
            current = {"address": int(address, 16), "instructions": []}
            functions[symbol] = current
        elif current is not None:
            inst = re.match(r"^\s*([0-9a-f]+):\s+(.*)$", line)
            if inst:
                current["instructions"].append(
                    {"address": int(inst[1], 16), "text": inst[2]}
                )
    if not functions or not any(f["instructions"] for f in functions.values()):
        raise ValueError("empty disassembly")
    return functions


def require_symbol(functions, symbol):
    for name in (symbol, "_" + symbol):
        if name in functions and functions[name]["instructions"]:
            return functions[name]
    raise ValueError(f"missing assembly symbol: {symbol}")


def opcode(body):
    """LLVM separates mnemonics and operands with spaces or tabs."""
    return body.split(None, 1)[0]


def is_branch(body):
    return (
        re.fullmatch(
            r"b(?:\.[a-z]+)?|bc\.[a-z]+|bl|cbn?z|tbn?z|j[a-z]+|call[qwl]?|loop(?:e|ne|z|nz)?",
            opcode(body),
        )
        is not None
    )


def is_call(body):
    return (
        re.fullmatch(r"bl|blr|blraa|blrab|blraaz|blrabz|call[qwl]?", opcode(body))
        is not None
    )


def target_operand(body):
    """Return an original-text match, preserving operand character positions."""
    fields = re.match(r"(\S+)\s+(.*)", body)
    if fields is None:
        return None
    op, operands = fields.groups()
    position = (
        2
        if op in ("tbz", "tbnz")
        else 1
        if op in ("cbz", "cbnz", "adr", "adrp")
        else 0
        if is_branch(body)
        else None
    )
    if position is None:
        return None
    parts = list(re.finditer(r"[^,]+", operands))
    if position >= len(parts):
        raise ValueError(f"invalid target operands: {body}")
    part = parts[position]
    # x86 *address and register/memory operands are indirect, even with hex text.
    if part[0].lstrip().startswith("*"):
        return None
    start, end = fields.start(2) + part.start(), fields.start(2) + part.end()
    return re.compile(r"0x([0-9a-f]+)").search(body, start, end)


def source_instructions(text):
    instructions = []
    for line in text.splitlines():
        match = re.match(r"^\s+([a-z][a-z0-9.]*)\b(?:\s+.*)?$", line)
        if match:
            instructions.append(line.strip())
    if not instructions:
        raise ValueError("empty assembly body: no valid instructions")
    return instructions


def count_source(text, comparisons=None):
    instructions = source_instructions(text)
    if comparisons is None:
        return len(instructions)
    label, count, simd = None, 0, 0
    for line in text.splitlines():
        if re.fullmatch(r"[a-zA-Z0-9_]+:", line):
            label, count, simd = line[:-1], 0, 0
        elif line.strip() in instructions:
            opcode = line.strip().split()[0]
            count += 1
            simd += opcode.startswith("facgt")
            if opcode == "cbnz" and line.split()[-1] == label and simd == comparisons:
                return count
    raise ValueError("missing reviewed SIMD scan loop")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("binary", nargs="?")
    ap.add_argument("--source-count", action="store_true")
    ap.add_argument("--comparisons", type=int)
    ap.add_argument("symbol", nargs="?")
    args = ap.parse_args()
    if args.source_count:
        print(count_source(sys.stdin.read(), args.comparisons))
        sys.exit(0)
    text = subprocess.check_output(
        [os.environ.get("OBJDUMP", "objdump"), "-d", args.binary], text=True
    )
    print(len(require_symbol(parse(text), args.symbol)["instructions"]))
