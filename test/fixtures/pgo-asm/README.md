These are unedited LLVM 22.1.2 (Rust 1.96.0) objdump outputs from two real
arm64 Mach-O dylibs. They retain LLVM's tabs, section table, defined/undefined
symbols and nearest-symbol annotations. Only their load addresses differ.
The fixture is replayed on ELF hosts too; tests also assemble native and cross
ELF/Mach-O objects with the running toolchain and consume their relocations.

To reproduce on macOS arm64, in a temporary directory containing `macho.s`:

```sh
clang -c macho.s -o probe.o
clang -shared probe.o -Wl,-undefined,dynamic_lookup,-no_fixup_chains,-image_base,0 -o linked.dylib
strip -x linked.dylib
"$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/llvm-objdump" --disassemble --demangle --no-show-raw-insn --section-headers --private-headers --syms --reloc --dynamic-reloc linked.dylib > macho-0.txt
"$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/llvm-objdump" --macho --indirect-symbols --bind --lazy-bind --weak-bind --dyld-info linked.dylib >> macho-0.txt
clang -shared probe.o -Wl,-undefined,dynamic_lookup,-no_fixup_chains,-image_base,0x4000 -o linked.dylib
strip -x linked.dylib
"$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/llvm-objdump" --disassemble --demangle --no-show-raw-insn --section-headers --private-headers --syms --reloc --dynamic-reloc linked.dylib > macho-16384.txt
"$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/llvm-objdump" --macho --indirect-symbols --bind --lazy-bind --weak-bind --dyld-info linked.dylib >> macho-16384.txt
```

The `macho-section-one` / `macho-section-two` sources and dumps test a real
Mach-O format ambiguity: `__TEXT,__const` and `__DATA_CONST,__const` both appear
as `__const` in LLVM's section summary. Mach-O private headers retain their
segment-qualified identities. The only instruction change is which
anonymous section is referenced. Reproduce each source in a temporary directory:

```sh
clang -shared macho-section-one.s -Wl,-no_fixup_chains -o sections.dylib
strip -x sections.dylib
"$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/llvm-objdump" --disassemble --demangle --no-show-raw-insn --section-headers --private-headers --syms --reloc sections.dylib > macho-section-one.txt
```

Repeat with `macho-section-two.s` to produce `macho-section-two.txt`.
