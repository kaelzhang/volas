"""Stable references from LLVM symbols, sections and instruction relocations."""

from collections import Counter
from bisect import bisect_left
import re
from urllib.parse import quote

from asm_symbols import is_branch, is_call, opcode, target_operand, written_registers


def disassemble(objdump, library, output):
    """LLVM owns object-format decoding; retain its numeric CFG and binding facts.

    Generic --symbolize-operands removes numeric branch targets and still uses
    nearest-symbol guesses for Mach-O imports. Its indirect/bind/dyld tables and
    ELF dynamic relocations/PLT labels supply the actual import identities.
    """
    text = output(
        [
            str(objdump),
            "--disassemble",
            "--demangle",
            "--no-show-raw-insn",
            "--section-headers",
            "--private-headers",
            "--syms",
            "--reloc",
            "--dynamic-reloc",
            str(library),
        ]
    )
    if "file format mach-o" in text:
        text += "\n" + output(
            [
                str(objdump),
                "--macho",
                "--indirect-symbols",
                "--bind",
                "--lazy-bind",
                "--weak-bind",
                "--dyld-info",
                str(library),
            ]
        )
    elif "file format elf" in text:
        text += "\n" + output([str(objdump), "--dynamic-syms", str(library)])
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
            "__stubs",
            "__auth_stubs",
            "__got",
            "__auth_got",
            "__auth_ptr",
            "__la_symbol_ptr",
            "__nl_symbol_ptr",
            ".plt",
            ".plt.sec",
            ".plt.got",
            ".iplt",
            ".got",
            ".got.plt",
            ".igot.plt",
        }
        self.stub_sections = {"__stubs", "__auth_stubs"}
        # LLVM decodes Mach-O section types too, including renamed stub/pointer
        # sections. Conventional names above also identify incomplete captures.
        for block in text.split("Section\n")[1:]:
            name = re.match(r"\s+sectname (\S+)", block)
            if name and re.search(
                r"^\s+type S_(?:SYMBOL_STUBS|(?:NON_LAZY|LAZY|LAZY_DYLIB)_SYMBOL_POINTERS)\b",
                block,
                re.M,
            ):
                self.import_sections.add(name[1])
                if re.search(r"^\s+type S_SYMBOL_STUBS\b", block, re.M):
                    self.stub_sections.add(name[1])
        self.read_imports(text, functions)
        self.resolved = {}
        self.memory_targets = {}
        for function in functions.values():
            if function.get("section") in self.stub_sections:
                self.resolve_pages(function)
        self.resolve_stubs(functions)
        # Re-evaluate consumers that reach a now-resolved stub address.
        self.resolved.clear()
        self.memory_targets.clear()
        for function in functions.values():
            self.resolve_pages(function)

    def read_imports(self, text, functions):
        """Bind slots to complete loader identities; incomplete facts never equal.

        LLVM's Mach-O tables abbreviate dylibs. Resolve a short name only against
        a unique load command, retaining its two-level ordinal and install name.
        Indirect entries are names, never complete bindings. Stub identities come
        from their actual pointer targets. ELF dynamic symbols supply versions
        missing in relocation displays; PLT labels inherit the qualified binding.
        """
        macho = "file format mach-o" in text
        libraries = []
        for block in re.split(r"Load command \d+\n", text)[1:]:
            if re.search(
                r"cmd LC_(?:LOAD|LOAD_WEAK|REEXPORT|LAZY_LOAD|LOAD_UPWARD)_DYLIB\b",
                block,
            ):
                name = re.search(r"^\s+name (.+) \(offset \d+\)", block, re.M)
                if name:
                    libraries.append(name[1])

        def namespace(short):
            if short in (
                "flat-namespace",
                "main-executable",
                "weak-lookup",
                "this-image",
            ):
                return short
            candidates = [
                (n, name)
                for n, name in enumerate(libraries, 1)
                if re.sub(r"(?:\.[^.]+)?\.dylib$", "", name.rsplit("/", 1)[-1]) == short
            ]
            if len(candidates) != 1 or "TWOLEVEL" not in text:
                return None
            ordinal, install = candidates[0]
            return f"two-level:{ordinal}:{install}"

        versions = {}
        dynamic_symbols = False
        versioned = bool(re.search(r"\bVERSYM\b|\.gnu.version\b", text))
        for line in text.splitlines():
            if line == "DYNAMIC SYMBOL TABLE:":
                dynamic_symbols = True
                continue
            if not line:
                dynamic_symbols = False
            if dynamic_symbols and (
                row := re.match(
                    r"^[0-9a-f]+\s+.*?\*UND\*\s+[0-9a-f]+\s+(?:\(([^)]+)\)\s+)?(\S+)$",
                    line,
                )
            ):
                version, name = row.groups()
                versions.setdefault(name, set()).add(version or "")
        # Bind the ELF version's dependency as well when LLVM supplies VERNEED.
        version_libraries, provider = {}, None
        for line in text.splitlines():
            required = re.match(r"\s+required from (.+):$", line)
            if required:
                provider = required[1]
            elif provider and (
                row := re.match(r"\s+0x[0-9a-f]+ 0x[0-9a-f]+ \S+ (\S+)$", line)
            ):
                version_libraries.setdefault(row[1], set()).add(provider)

        def elf_identity(value):
            match = re.fullmatch(r"(.+?)([+-]0x[0-9a-f]+)?", value)
            name, addend = match.groups()
            choices = versions.get(name, {""} if not versioned else set())
            if len(choices) != 1:
                return None
            version = next(iter(choices))
            providers = version_libraries.get(version, set())
            if version and len(providers) != 1:
                return None
            scope = "elf-global" + (":" + next(iter(providers)) if version else "")
            symbol = name + ("@" + version if version else "")
            return (scope, symbol, int(addend, 0) if addend else 0)

        indirect, table = False, None
        bindings, names = {}, {}
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
                names.setdefault(int(slot[1], 16), set()).add(slot[2])
            elif line and not line.startswith(("address", "0x")):
                indirect = False
            dynamic = re.fullmatch(r"([0-9a-fA-F]+)\s+(R_\S+)\s+(\S+)", line)
            if dynamic and not dynamic[2].endswith("RELATIVE"):
                bindings.setdefault(int(dynamic[1], 16), set()).add(
                    elf_identity(dynamic[3])
                )
            bind = re.match(r"^\S+\s+\S+\s+0x([0-9a-fA-F]+)\s+(.+)$", line)
            if not table or not bind:
                continue
            fields = bind[2].split()
            if table == "dyld":
                if len(fields) != 5 or fields[1] != "bind":
                    continue
                addend, provider = int(fields[2], 0), fields[3]
            elif table == "lazy":
                if len(fields) != 2:
                    continue
                addend, provider = 0, fields[0]
            elif table == "weak":
                if len(fields) != 3 or fields[0] != "pointer":
                    continue
                addend, provider = int(fields[1], 0), "weak-lookup"
            else:
                if len(fields) != 4 or fields[0] != "pointer":
                    continue
                addend, provider = int(fields[1], 0), fields[2]
            scope = namespace(provider)
            bindings.setdefault(int(bind[1], 16), set()).add(
                (scope, fields[-1], addend) if scope else None
            )
        self.indirect_stubs = {}
        self.imports.update(bindings)
        for address, symbols in names.items():
            section = self.section(address)
            if section and section[0] in self.stub_sections:
                self.indirect_stubs[address] = symbols
            if address in bindings:
                if any(b is None or b[1] not in symbols for b in bindings[address]):
                    self.imports[address].add(None)
            else:
                self.imports[address] = {None}
        for name, function in functions.items():
            if (
                name.endswith("@plt")
                and function.get("section") in self.import_sections
            ):
                binding = elf_identity(name[:-4]) if not macho else None
                self.imports.setdefault(function["address"], set()).add(binding)

    def rip_target(self, instruction, function):
        """One owner for LLVM's x86 RIP comments / next-instruction arithmetic."""
        body = instruction["text"]
        rip = re.search(r"(?:-?0x[0-9a-f]+|-?[0-9]+)?(?=\(%rip\))", body)
        if not rip:
            return None
        annotation = re.search(r"#+\s*0x([0-9a-f]+)(?: <.*>)?", body)
        end = self.next_addresses.get((function.get("section"), instruction["address"]))
        if annotation or end is not None:
            dest = int(annotation[1], 16) if annotation else end + int(rip[0] or "0", 0)
            return rip, dest
        return None

    def literal_operand(self, body):
        """LLVM's ARM literal load names the reached address directly."""
        return re.fullmatch(
            r"(?:ldr|ldrsw)\s+\w+,\s*(0x[0-9a-f]+)(?: <[^>]+>)?", body
        )

    def resolve_stubs(self, functions):
        """Use decoded entry boundaries and existing address resolution, not names.

        LLVM's indirect table supplies exact stub entry addresses. Its load must
        reach a complete matching binding; multiple or unknown targets stay
        unresolved. This also retains nonzero addends in custom linked stubs.
        """
        locations = {}
        for function in functions.values():
            if function.get("section") not in self.stub_sections:
                continue
            for instruction in function["instructions"]:
                locations.setdefault(function.get("section"), []).append(
                    (instruction["address"], function, instruction)
                )
        for rows in locations.values():
            rows.sort(key=lambda row: row[0])
        addresses = {
            section: [row[0] for row in rows] for section, rows in locations.items()
        }
        grouped = {}
        for address in self.indirect_stubs:
            grouped.setdefault(self.section(address), []).append(address)
        ends = {}
        for section, entries in grouped.items():
            entries.sort()
            ends.update(zip(entries, [*entries[1:], section[1] + section[2]]))
        for address, names in self.indirect_stubs.items():
            section = self.section(address)
            end = ends[address]
            candidates = set()
            rows = locations.get(section[0], [])
            positions = addresses.get(section[0], [])
            for _, function, instruction in rows[
                bisect_left(positions, address) : bisect_left(positions, end)
            ]:
                key = (function.get("section"), instruction["address"])
                target = self.memory_targets.get(key)
                rip = self.rip_target(instruction, function)
                if rip:
                    target = rip[1]
                literal = self.literal_operand(instruction["text"])
                if literal:
                    target = int(literal[1], 16)
                if target in self.imports:
                    candidates.update(self.imports[target])
            if not candidates or any(
                b is None or b[1] not in names for b in candidates
            ):
                candidates.add(None)
            self.imports[address] = candidates

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
                reached = address + int(consumer[2] or "0", 0)
                if load:
                    self.memory_targets[key] = reached
                ref = self.reference(reached, function)
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
            if len(imports) == 1 and None not in imports:
                scope, symbol, addend = next(iter(imports))
                symbol, scope = (quote(v, safe="/:@+$._-") for v in (symbol, scope))
                return f"@import({symbol},{scope},{addend:+d})"
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
        literal = self.literal_operand(body)
        if literal:
            return body[: literal.start(1)] + self.reference(
                int(literal[1], 16), function
            )
        rip = self.rip_target(instruction, function)
        if rip:
            operand, dest = rip
            ref = self.reference(dest, function)
            body = body[: operand.start()] + ref + body[operand.end() :]
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
