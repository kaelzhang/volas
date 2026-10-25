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
    section = None
    for line in text.splitlines():
        heading = re.match(r"Disassembly of section (.+):", line)
        if heading:
            section = heading[1].split(",")[-1]
            current = None
        header = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if header:
            address, symbol = header.groups()
            current = {
                "address": int(address, 16),
                "section": section,
                "instructions": [],
            }
            functions[symbol] = current
        elif current is not None:
            inst = re.match(r"^\s*([0-9a-f]+):\s+(.*)$", line)
            if inst and not re.match(r"(?:R_|\w+_RELOC_)", inst[2]):
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


def written_registers(body, *, writeback=True):
    """ARM64 register definitions needed by reference/stack fact lifetimes."""
    op = opcode(body)
    registers = set()
    dest = re.match(r"\S+\s+([xw]\d+)\b", body)
    if dest and not (
        op.startswith("st")
        or op in ("cmp", "cmn", "tst", "ccmp", "ccmn")
        or is_branch(body)
        or is_call(body)
        or op in ("ret", "br", "braa", "brab")
    ):
        registers.add("x" + dest[1][1:])
    if op.startswith(("ldp", "ldnp")):
        second = re.match(r"\S+\s+\w+,\s*[xw](\d+)", body)
        if second:
            registers.add("x" + second[1])
    base = re.search(r"\[(x\d+)[^]]*\](?:!|,)", body)
    if writeback and base:
        registers.add(base[1])
    return registers


def stack_access_indices(function):
    """Load/store accesses via SP or an established ARM64 frame pointer.

    A forward must-analysis keeps x29 only where every incoming CFG path has
    established it from SP without overwriting it. Pre/post-index operands use
    the incoming fact before writeback. Accesses do not imply confirmed spills.
    """
    instructions = function["instructions"]
    if not instructions:
        return []
    addresses = {i["address"]: n for n, i in enumerate(instructions)}

    def establishes(body):
        return re.fullmatch(
            r"(?:mov\s+x29,\s*sp|(?:add|sub)\s+x29,\s*sp,\s*#(?:0x[0-9a-f]+|\d+))",
            body,
        ) is not None

    def restores_at_exit(index):
        body = instructions[index]["text"]
        if not opcode(body).startswith(("ldp", "ldnp")) or "[sp" not in body:
            return False
        for following in instructions[index + 1:]:
            body = following["text"]
            if opcode(body) in ("ret", "retaa", "retab"):
                return True
            target = target_operand(body)
            if opcode(body) == "b" and target and int(target[1], 16) not in addresses:
                return True  # frame restored before a tail call outside this body
            if is_branch(body) or is_call(body) or "x29" in body:
                return False
        return False

    # A normal prologue's invariant also covers jump-table / exception blocks
    # whose incoming edges are absent from numeric direct-branch disassembly.
    # Use it only if all later x29 definitions establish a frame or restore it
    # immediately along an exit path; arbitrary clobbers disable this inference.
    prologue = []
    for instruction in instructions:
        body = instruction["text"]
        if is_branch(body) or is_call(body) or opcode(body) in ("ret", "br"):
            break
        prologue.append(body)
    invariant = any(establishes(body) for body in prologue) and all(
        establishes(i["text"]) or restores_at_exit(n)
        for n, i in enumerate(instructions)
        if "x29" in written_registers(i["text"], writeback=False)
    )
    incoming = [None] * len(instructions)
    incoming[0] = False
    pending = [0]
    while pending:
        index = pending.pop()
        body = instructions[index]["text"]
        op = opcode(body)
        state = True if establishes(body) else (
            False if "x29" in written_registers(body, writeback=False) else incoming[index]
        )
        successors = []
        if is_branch(body) and not is_call(body):
            target = target_operand(body)
            if target and int(target[1], 16) in addresses:
                successors.append(addresses[int(target[1], 16)])
        if op not in ("b", "br", "braa", "brab", "ret", "retaa", "retab"):
            if index + 1 < len(instructions):
                successors.append(index + 1)
        for dest in successors:
            merged = state if incoming[dest] is None else incoming[dest] and state
            if incoming[dest] != merged:
                incoming[dest] = merged
                pending.append(dest)
    return [
        n for n, instruction in enumerate(instructions)
        if opcode(instruction["text"]).startswith(("ld", "st"))
        and (memory := re.search(r"\[(sp|x29)\b", instruction["text"]))
        and (memory[1] == "sp" or incoming[n] is True or (incoming[n] is None and invariant))
    ]


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
