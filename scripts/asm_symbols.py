#!/usr/bin/env python3
"""Fail-closed parsing shared by probe counts and final-wheel assembly reports."""

import argparse
import os
import re
import subprocess


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("binary")
    ap.add_argument("symbol")
    args = ap.parse_args()
    text = subprocess.check_output(
        [os.environ.get("OBJDUMP", "objdump"), "-d", args.binary], text=True
    )
    print(len(require_symbol(parse(text), args.symbol)["instructions"]))
