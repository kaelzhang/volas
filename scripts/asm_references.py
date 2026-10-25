"""Stable references from LLVM symbols, sections and instruction relocations."""

from collections import Counter
import re

from asm_symbols import is_branch, is_call, opcode, target_operand, written_registers


def disassemble(objdump, library, output):
    """LLVM owns object-format decoding; retain its numeric CFG and binding facts.

    Generic --symbolize-operands removes numeric branch targets and still uses
    nearest-symbol guesses for Mach-O imports. Its indirect/bind/dyld tables and
    ELF dynamic relocations/PLT labels supply the actual import identities.
    """
    text = output([
        str(objdump), "--disassemble", "--demangle", "--no-show-raw-insn",
        "--section-headers", "--private-headers", "--syms", "--reloc",
        "--dynamic-reloc", str(library),
    ])
    if "file format mach-o" in text:
        text += "\n" + output([
            str(objdump), "--macho", "--indirect-symbols", "--bind",
            "--lazy-bind", "--weak-bind", "--dyld-info", str(library),
        ])
    return text


class References:
    def __init__(self, text, functions):
        self.sections = []
        self.symbols = []
        self.relocations = {}
        section = None
        for line in text.splitlines():
            row = re.match(r"^\s*\d+\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+", line)
            if row:
                name, size, address = row.groups()
                self.sections.append((name, int(address, 16), int(size, 16)))
            symbol = re.match(r"^([0-9a-f]+)\s+", line)
            if symbol:
                sec = next(
                    (
                        m
                        for m in re.finditer(r"\S+", line[symbol.end() :])
                        if m[0].split(",")[-1] in {s[0] for s in self.sections}
                    ),
                    None,
                )
                if sec:
                    name = line[symbol.end() + sec.end() :].strip()
                    size = re.match(r"([0-9a-f]+)\s+(.+)$", name)
                    self.symbols.append(
                        (
                            sec[0].split(",")[-1],
                            int(symbol[1], 16),
                            int(size[1], 16) if size else 0,
                            (size[2] if size else name).removeprefix(".hidden "),
                        )
                    )
            heading = re.match(r"Disassembly of section (.+):", line)
            if heading:
                section = heading[1].split(",")[-1]
            relocation = re.match(
                r"^\s+([0-9a-f]+):\s+((?:R_|\w+_RELOC_)\S+)\s+(.+)$", line
            )
            if relocation:
                address, kind, value = relocation.groups()
                self.relocations.setdefault((section, int(address, 16)), []).append(
                    (kind, value)
                )
        qualified = {
            (name, int(address, 16), int(size, 16)): segment + "," + name
            for name, segment, address, size in re.findall(
                r"Section\n\s+sectname (\S+)\n\s+segname (\S+)\n"
                r"\s+addr 0x([0-9a-f]+)\n\s+size 0x([0-9a-f]+)",
                text,
            )
        }
        counts = Counter(name for name, _, _ in self.sections)
        occurrences = Counter()
        self.labels = {}
        for section in self.sections:
            name = section[0]
            self.labels[section] = qualified.get(
                section, name if counts[name] == 1 else f"{name}[{occurrences[name]}]"
            )
            occurrences[name] += 1
        # Disassembly also supplies useful function symbols when no symbol table
        # was supplied (probe consumers), but undefined symbols never become bases.
        known = {(a, n) for _, a, _, n in self.symbols}
        self.ends = {}
        self.next_addresses = {}
        for name, function in functions.items():
            address = function["address"]
            sec = self.section(address, function.get("section"))
            instructions = function["instructions"]
            for index, instruction in enumerate(instructions):
                if index + 1 < len(instructions):
                    self.next_addresses[
                        (function.get("section"), instruction["address"])
                    ] = instructions[index + 1]["address"]
                self.ends[(function.get("section"), instruction["address"])] = (
                    instructions[index + 1]["address"]
                    if index + 1 < len(instructions)
                    else instruction["address"] + 16
                )
            if sec and (address, name) not in known:
                self.symbols.append((sec[0], address, 0, name))

        self.imports = {}
        self.import_sections = {
            "__stubs", "__auth_stubs", "__got", "__auth_got", "__auth_ptr",
            "__la_symbol_ptr", "__nl_symbol_ptr", ".plt", ".plt.sec", ".plt.got",
            ".iplt", ".got", ".got.plt", ".igot.plt",
        }
        # LLVM decodes Mach-O section types too, including renamed stub/pointer
        # sections. Conventional names above also identify incomplete captures.
        for block in text.split("Section\n")[1:]:
            name = re.match(r"\s+sectname (\S+)", block)
            if name and re.search(
                r"^\s+type S_(?:SYMBOL_STUBS|(?:NON_LAZY|LAZY|LAZY_DYLIB)_SYMBOL_POINTERS)\b",
                block, re.M,
            ):
                self.import_sections.add(name[1])
        self.read_imports(text, functions)
        self.resolved = {}
        for function in functions.values():
            self.resolve_pages(function)

    def read_imports(self, text, functions):
        """Consume LLVM's decoded identities, never infer an import from a slot.

        Keep conflicting facts unresolved. LOCAL/ABSENT indirect entries are not
        symbols. Bind/dyld records also cover pointers outside named GOT sections.
        """
        indirect, table = False, None
        bindings = {}
        headings = {
            "dyld information:": "dyld",
            "Bind table:": "bind",
            "Lazy bind table:": "lazy",
            "Weak bind table:": "weak",
        }
        for line in text.splitlines():
            if line in headings:
                table = headings[line]
            heading = re.match(r"Indirect symbols for \([^,]+,([^)]*)\)", line)
            if heading:
                indirect = True
                self.import_sections.add(heading[1])
                continue
            slot = re.fullmatch(r"0x([0-9a-fA-F]+)\s+\d+\s+(\S+)", line)
            if indirect and slot:
                self.imports.setdefault(int(slot[1], 16), set()).add(slot[2])
            elif line and not line.startswith(("address", "0x")):
                indirect = False
            # Decoded dynamic relocations, not section-relative instruction
            # relocations (whose offset has a colon). RELATIVE has no import.
            dynamic = re.fullmatch(r"([0-9a-fA-F]+)\s+(R_\S+)\s+(\S+)", line)
            if dynamic and not dynamic[2].endswith("RELATIVE"):
                self.imports.setdefault(int(dynamic[1], 16), set()).add(dynamic[3])
            bind = re.match(r"^\S+\s+\S+\s+0x([0-9a-fA-F]+)\s+(.+)$", line)
            if not table or not bind:
                continue
            fields = bind[2].split()
            if table == "dyld":
                if len(fields) < 5 or fields[1] != "bind":
                    continue  # rebases describe internal pointers, not imports
                addend = int(fields[2], 0)
            elif table == "lazy":
                if len(fields) != 2:
                    continue
                addend = 0
            else:
                if len(fields) < 3 or fields[0] != "pointer":
                    continue
                addend = int(fields[1], 0)
            bindings.setdefault(int(bind[1], 16), set()).add((fields[-1], addend))
        for address, values in bindings.items():
            # Indirect symbols name the symbol, bindings additionally supply its
            # addend. Matching records refine that identity; disagreements remain
            # a conflict instead of silently selecting a different import.
            names = self.imports.setdefault(address, set())
            names.difference_update(name for name, _ in values)
            names.update(name + (f",{addend:+d}" if addend else "") for name, addend in values)
        # LLVM synthesizes ELF PLT labels from dynamic relocations. Unlike a
        # nearest-symbol annotation, the label names the exact stub entry.
        for name, function in functions.items():
            if name.endswith("@plt") and function.get("section") in self.import_sections:
                self.imports.setdefault(function["address"], set()).add(name[:-4])

    def resolve_pages(self, function):
        """Track ADRP definitions to their ADD/LDR consumers within a basic block.

        The page alone has no stable symbol identity. All consumers name what
        they actually reach, and the defining ADRP names that ordered target set.
        Register writes and control transfers end a definition's lifetime.
        """
        pages = {}
        block_starts = set()
        for instruction in function["instructions"]:
            target = target_operand(instruction["text"])
            if is_branch(instruction["text"]) and target:
                block_starts.add(int(target[1], 16))
        for instruction in function["instructions"]:
            if instruction["address"] in block_starts:
                pages.clear()
            body = instruction["text"]
            key = (function.get("section"), instruction["address"])
            op = opcode(body)
            page = re.fullmatch(r"adrp\s+(x\d+),\s*(0x[0-9a-f]+)(?: <.*>)?", body)
            if page and not self.instruction_relocations(instruction, function):
                pages[page[1]] = (key, int(page[2], 16), body, [])
                continue
            add = re.fullmatch(r"add\s+x\d+,\s*(x\d+),\s*#(0x[0-9a-f]+|[0-9]+)", body)
            load = re.fullmatch(
                r"(?:ldr|ldrsw|ldrb|ldrh)\s+\w+,\s*\[(x\d+)(?:,\s*#(0x[0-9a-f]+|[0-9]+))?\]",
                body,
            )
            consumer = add or load
            if (
                consumer
                and consumer[1] in pages
                and not self.instruction_relocations(instruction, function)
            ):
                definition, address, original, refs = pages[consumer[1]]
                ref = self.reference(address + int(consumer[2] or "0", 0), function)
                refs.append(ref)
                self.resolved[key] = (
                    body[: consumer.start(2)] + ref + body[consumer.end(2) :]
                    if consumer[2] is not None
                    else body[: consumer.end(1)] + ", #" + ref + body[consumer.end(1) :]
                )
                target = target_operand(original)
                self.resolved[definition] = (
                    original[: target.start()] + "@page(" + ",".join(refs) + ")"
                )
            for register in written_registers(body):
                pages.pop(register, None)
            if is_branch(body) or is_call(body) or op in ("ret", "br", "braa", "brab"):
                pages.clear()

    def section(self, address, name=None):
        return next(
            (
                s
                for s in self.sections
                if (name is None or s[0] == name) and s[1] <= address < s[1] + s[2]
            ),
            None,
        )

    def reference(self, address, function, page=False):
        # An unmatched ADRP names a page, not the import at its first slot. Even
        # if LLVM binds that slot, a later unpaired load can reach another one.
        # Paired consumers have already replaced their definition in resolved.
        if page and any(
            name in self.import_sections and start & ~4095 <= address < start + size
            for name, start, size in self.sections
        ):
            return "@unresolved(import-page)"
        imports = self.imports.get(address)
        if imports:
            if len(imports) == 1:
                return f"@import({next(iter(imports))})"
            return "@unresolved(import-conflict)"
        section = self.section(address)
        if section:
            name, start, size = section
            if name in self.import_sections:
                return f"@unresolved(import,{self.labels[section]},{address - start})"
            candidates = [
                s
                for s in self.symbols
                if s[0] == name
                and start <= s[1] <= address
                and not s[3].startswith(("ltmp", "$"))
            ]
            if candidates:
                symbol = max(candidates, key=lambda s: s[1])
                _, base, length, label = symbol
                # Do not invent symbol bounds from a nearest-symbol annotation.
                end = min(
                    (
                        a
                        for sec, a, _, _ in self.symbols
                        if sec == name and base < a < start + size
                    ),
                    default=start + size,
                )
                if address < (base + length if length else end):
                    return f"@symbol({label},{address - base})"
            return f"@section({self.labels[section]},{address - start})"
        if page:
            section = next(
                (s for s in self.sections if s[1] & ~4095 <= address < s[1] + s[2]),
                None,
            )
            if section:
                return f"@section-page({self.labels[section]},{address - (section[1] & ~4095)})"
        return f"@unresolved(function-relative,{address - function['address']})"

    def instruction_relocations(self, instruction, function):
        address = instruction["address"]
        sec = function.get("section")
        end = self.ends.get((sec, address), address + 16)
        return [
            (offset - address, kind, value)
            for (section, offset), values in self.relocations.items()
            if (sec is None or section == sec) and address <= offset < end
            for kind, value in values
        ]

    def normalize(self, instruction, function, addresses):
        body = instruction["text"]
        relocs = self.instruction_relocations(instruction, function)
        target = target_operand(body)
        # A relocation owns the encoded operand; its placeholder address is not a
        # CFG target. Preserve the ordered relocation kind, addend and byte offset.
        if relocs:
            if target:
                end = target.end()
                annotation = re.match(r" <.+>", body[end:])
                if annotation:
                    end += annotation.end()
                body = body[: target.start()] + "@relocated" + body[end:]
            elif "%rip" in body:
                body = re.sub(r"0x[0-9a-f]+(?=\(%rip\))", "@relocated", body)
                body = re.sub(r"0x[0-9a-f]+ <.+>", "@relocated", body)
            else:
                # ARM pageoff relocations own the immediate operand, not register
                # numbers or unrelated immediates (e.g. tbz's bit number).
                body = re.sub(r"#(?:0x[0-9a-f]+|[0-9]+)\s*$", "#@relocated", body)
            # ELF x86 encodes a PC-relative displacement without the 0x prefix.
            body = re.sub(r"-?\d+(?=\(%rip\))", "@relocated", body)
            return body + " @relocation" + repr(relocs)
        key = (function.get("section"), instruction["address"])
        if key in self.resolved:
            return self.resolved[key]
        rip = re.search(r"(?:-?0x[0-9a-f]+|-?[0-9]+)?(?=\(%rip\))", body)
        if rip:
            annotation = re.search(r"#+\s*0x([0-9a-f]+)(?: <.*>)?", body)
            end = self.next_addresses.get(key)
            # LLVM gives an exact resolved address in its comment, even for the
            # last instruction. Otherwise use the next instruction's address.
            if annotation or end is not None:
                dest = (
                    int(annotation[1], 16)
                    if annotation
                    else end + int(rip[0] or "0", 0)
                )
                ref = self.reference(dest, function)
                body = body[: rip.start()] + ref + body[rip.end() :]
        if target:
            dest = int(target[1], 16)
            ref = (
                f"@instruction{addresses[dest]}"
                if dest in addresses
                else self.reference(dest, function, opcode(body) == "adrp")
            )
            end = target.end()
            annotation = re.match(r" <.+>", body[end:])
            if annotation:
                end += annotation.end()
            body = body[: target.start()] + ref + body[end:]
        # Includes x86 RIP-relative comments and ARM literal loads, neither of
        # which is necessarily a branch/adr operand.
        body = re.sub(
            r"0x([0-9a-f]+) <.+>",
            lambda m: self.reference(int(m[1], 16), function),
            body,
        )
        body = re.sub(r"<[^>]*\+0x[0-9a-f]+>", "@unresolved(annotation)", body)
        return body
