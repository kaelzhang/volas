These are unedited dumps of real linked ARM64/x86_64 Mach-O and ELF libraries,
using Rust 1.96.0's LLVM 22.1.2 objdump. They include numeric instructions,
section/symbol tables, relocations, ELF dynamic relocations and Mach-O
indirect/bind/dyld information. No binary or instruction text was patched.

`test/pgo_linked_imports.py` contains the assembly source generator and exact
compiler, linker and objdump commands. `base` imports `foreign`, `target` imports
`changed` at the same slots, and `moved` adds an `aaa` import before `foreign`.
Tests independently verify the slot addresses before comparing the reports.
Native tests rebuild the libraries and exercise the actual report entry too;
captured foreign formats are replayed on every host. Capture all four formats
on macOS with clang and ld.lld installed:

```sh
python -m test.pgo_linked_imports /tmp/linked-import-captures
```

`stack.s` is a linked ARM64 instruction fixture, including SP and established
frame-pointer accesses, unscaled loads/stores, pre/post-index writeback, vector
loads/stores, and negative controls (a prefetch, another base, an overwritten
frame pointer, an unestablished frame pointer and a CFG join that loses the fact).
To reproduce `stack.txt` on macOS arm64 in a scratch directory:

```sh
clang -shared stack.s -o stack.so
"$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/llvm-objdump" -d --no-show-raw-insn --section-headers --syms stack.so > stack.txt
```

`chained.s` and `provider.s` test dyld chained bindings with an import whose name
has no leading underscore, and a data pointer addend not present in the indirect
symbol table. Copy both to a scratch directory, replace `ADDEND` with 8 or 16,
and repeat:

```sh
clang -shared provider.s -Wl,-install_name,@rpath/provider.dylib -o provider.dylib
clang -shared chained.s provider.dylib -Wl,-fixup_chains -o linked.so
```

Dump with `asm_references.disassemble()` (numeric generic dump plus Mach-O
tables), as the capture helper does, to reproduce `chained-8.txt` and
`chained-16.txt`. Empty `confirmed_spills` is intentional: these instructions
prove stack accesses, not the compiler's reason for using memory.

The `page-*` captures use the same generator with `page_boundary=True`. The
selected function has a bare ADRP, an unconditional branch, then a GOT load; an
unselected anchor emits the import stub. The import changes at the same slot.
Without a reaching page fact the page must stay unresolved, including when its
first slot is itself bound (for example to dyld_stub_binder). It cannot stand in
for the import reached by a later load. The capture command above produces these
ARM64 Mach-O/ELF cases too.

`bindings.json` preserves complete LLVM dumps as JSON strings, including original
whitespace. Regenerate it on macOS with LLVM and `ld.lld` available:

```sh
python -m test.pgo_binding_libraries /absolute/fresh/captures
```

The checked-in generator links each architecture against a provider library,
changes only its install name, or moves the same bindings (stub, GOT and data).
ELF variants also change the required symbol version. Portable tests replay all
captures; native variants are linked again on the running host.

The ARM64 `custom` variants supply a symbol stub in the input object. Apple
classic ld rewrites it to a literal lazy-pointer load (traditional bindings) or
an ADRP/GOT pair (chained bindings). Both reach a zero-addend binding, while the
separate data pointer retains +8. The literal-load test also varies decoded
binding metadata in memory to verify a nonzero addend; that supplemental case
is explicitly synthetic, not a separately linked binary.
