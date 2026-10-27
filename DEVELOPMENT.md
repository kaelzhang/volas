# Developer Guide

## Design notes & non-goals

- **Not a general-purpose DataFrame.** volas models exactly what OHLCV
  quant workflows need; it deliberately omits multi-level indexes,
  heterogeneous per-cell storage, joins, and general reshaping.
- **pandas-independent at runtime.** pandas and TA-Lib are used only as
  test oracles and benchmark comparators, never imported at runtime.
- **External API cleanliness first.** The Python surface is kept clean
  and pandas-shaped; internal layering is secondary to per-bar latency.

## Development

Requires Python >= 3.11 and rustup. The repository's `rust-toolchain.toml`
selects Rust 1.96.0 and its `llvm-tools-preview` component for every Cargo and
rustc invocation, including debug builds, coverage and CI. `pyproject.toml`'s
`build-system.requires` is the single maturin pin (1.13.3); install targets and
CI read it through `scripts/check_toolchain.py`. Use your chosen virtual
environment's Python throughout:

```sh
make install PYTHON=/path/to/venv/bin/python  # pinned Rust + maturin + dev deps
make build          # build the Rust extension, install the package in-place
make test           # full correctness suite and 100% union coverage
make coverage       # true cargo-test union pytest line coverage
make benchmark      # multi-library benchmark
make build-pkg      # release wheel + sdist in BUILD_ARTIFACTS
```

`make install-rust` installs the repository's exact Rust version and LLVM
component and checks the active selection. `make install` also installs the
exact maturin requirement with `$(PYTHON) -m pip`, uses it for the editable
installation (`--no-build-isolation`), and runs the full prerequisite check.
Neither target updates a floating stable toolchain. Continue passing the same
`PYTHON` to each make command, or export `VOLAS_PYTHON` once.

The release build owner reads runtime requirements from `pyproject.toml` and
installs them into a fresh artifact-local directory before the final native
computation smoke. The smoke runs with `-S` against that directory and the staged
wheel, so it also works in a virtual environment containing only build tools.
Final wheel installation resolves its runtime dependencies through pip first,
then reinstalls only the wheel itself, preserving compatible dependency versions. The
benchmark target prepares its dev and comparison libraries after this build.
The perf and Pages workflows provision and check the same pinned tools before
building or considering a skippable BASE comparison.

`make coverage` delegates to `scripts/coverage.sh`. `make benchmark`
compares pandas, stock-pandas, polars, TA-Lib, and volas where those
benchmark-only dependencies are installed.

### Dependency groups

- **`dev`** (`pip install -e .[dev]`) — everything the test suite needs;
  this is all CI installs. It includes pandas because the parity tests
  use it as an oracle. pandas is test-time only; volas has no pandas
  runtime dependency.
- **`benchmark`** (`pip install -e .[benchmark]`) — extra comparison
  libraries used only by the benchmark. `make benchmark` installs
  `.[dev,benchmark]`; a library that is only needed to benchmark, never
  to test, belongs here so CI test runs stay lean.

### Timezone data

Named zones read the tzdb release that `chrono-tz` bundles. Its compiled tables hold
explicit transitions through 2099; from 2100 a zone follows its TZif footer rule (a
POSIX TZ string, as Python's `zoneinfo` does), which
`crates/volas-core/src/tz/footers.rs` records for every zone. After upgrading
`chrono-tz`, regenerate that file from the new release (needs `zic`):

```sh
python scripts/gen_tz_footers.py
```

The file's header records the tzdb release and the `zic` that compiled it (the current
file: tzdb 2025b, `zic (tzcode) 2022g`, the one macOS ships). Any `zic` that compiles the
release will do: the `footers_*` tests in `volas-core` check that the file comes from the
bundled release and that each footer continues the table exactly, so they fail until the
file is regenerated after an upgrade.

### Releasing

Two artifacts ship from one version: the Python wheels (PyPI) and the Rust
crates (crates.io). Both are driven by a git tag.

```sh
make bump TYPE={major|minor|patch}   # bump the workspace version, commit, tag, push
```

Pushing the tag triggers `.github/workflows/release.yml`, which:

- verifies the tag matches the `Cargo.toml` version;
- builds + smoke-tests the wheels and the sdist, then publishes to **PyPI**
  (Trusted Publishing / OIDC);
- publishes the **Rust crates** to crates.io in dependency order (the
  `publish-crates` job runs `make cargo-publish`). `volas-python` is
  `publish = false`; the `volas` facade goes last.

crates.io publishing needs a `CARGO_REGISTRY_TOKEN` repo secret (an API token
with publish scope). To publish the crates by hand instead:

```sh
make cargo-publish          # real publish (run `cargo login` first)
make cargo-publish DRY=1    # dry-run: build + package every crate, upload nothing
```

### Benchmark & web report

`make benchmark` times every candidate on batch indicator computation,
the incremental append-one-bar path, and the full volas-vs-TA-Lib
coverage rows. To optimize one indicator, pass `INDICATOR=<directive>`;
that scoped run prints only that indicator's coverage rows and never
writes the web report:

```sh
make benchmark INDICATOR=roc:10
make benchmark                  # full run; always writes ./benchmark-report.html
make benchmark INDICATOR=bop    # ONE indicator: compact normalized report to
                                # stdout (volas vs talib, batch + after-append +
                                # append overhead); a working probe, NOT archived
make perf-ab BASE=HEAD~1 [INDICATOR=bop]
                                # the standard A/B: isolated release BASE + HEAD
                                # (incl. uncommitted changes), six balanced AB/BA
                                # pairs on this machine; full suite ->
                                # perf_gate verdict, single indicator -> compact
                                # HEAD/base report. The CI gate's exact method.
# Use the tracked benchmark entrypoints and the paired CPU follow-up below;
# keep one measurement protocol instead of ad-hoc timing scripts.
# Every full run is also compared against the all-time best run (the
# "performance leader", .benchmarks/LEADER.json) and a leader-report.md is
# written into the run's archive dir. Scoring uses WITHIN-RUN normalized
# ratios (volas/talib measured in the same session), so runs from different
# machines/thermal states compare meaningfully; a run that beats the leader
# becomes the new leader. LEADER.json also carries the forward work queues:
# the top-10 slowest-vs-talib indicators (batch + after-append) and the
# after-append-overhead degrader list.
```

The full gate includes every volas benchmark section, aggregates sub-5µs costs
by section, and enforces the original cost-ratio bounds on release builds.
Correctness runs keep the same operations and value assertions without timing
bounds on instrumented code. See [gate details](benchmarks/README.md).

Wall protocol: long-history `tails` and `middle` selections use pytest-benchmark
`pedantic(iterations=1)`, ten warmups and fixed single-call samples (500 rounds,
spread across A/B pairs: 84 rounds per worker for six pairs). The previous result
stays alive through the next call's end clock and is replaced outside the timed
region; the last result is released after the batch.

CPU protocol: `scripts/perf_cpu.py` uses fresh BASE/HEAD workers for each of two runs,
with 200 pairs per item per run in alternating AB/BA order. It reports the median
and inclusive quartiles of per-pair HEAD/BASE ratios from `time.process_time_ns`;
a regression requires a median above 1.02 in both runs. Each pair measures the
manifest's repetitions (or the driver's workload default), not the wall round count.
`scripts/perf_runtime.py` owns captured call preparation and warmup: pedantic calls
keep the declared `warmup_rounds` (ten for long selections), while automatic calls
use five warmups. Warmup runs once when a worker captures an item, before its first
pair. Setup and warmup stay outside the measurement clocks. For single-call samples,
the previous result stays alive through the next call's wall and CPU end clocks,
then is replaced outside both windows; the final result is released after the batch.
This follows the real wall runner's result lifetime.

Both drivers use `scripts/perf_runtime.py` to configure worker environments. On macOS,
`MallocMaxMagazines=1` and `MallocMaxMediumMagazines=1` limit magazine counts for the
affected allocation classes, reducing CPU migration differences in local reuse state.
They do not put all allocation classes or large allocations in one magazine. The
long-selection item measures the controlled, warmed creation cost of a selection;
cold-page and reclamation costs remain real costs outside this measurement's scope.
These worker settings do not change release builds. Saved `runtime.json` files record
them; CPU outputs retain them as `<output>.runtime.json`.

The locally generated `./benchmark-report.html` (and the always-current
published copy at <https://volas.ost.ai>, deployed by the
`pages` workflow) keeps the append and
batch sections as charts, then summarizes full coverage as one row per
TA-Lib indicator. Extra length fixtures and cached append-refresh
comparisons appear as additional `volas vs TA-Lib` columns instead of
duplicate indicator rows.

### Instruction-level review of the hot paths

`make perf-ab` answers *"did this change move wall-clock?"*; the **assembly
gates** answer the finer *"did a hot function's generated code get heavier?"* —
the regression a 5µs-floored benchmark cannot see. They make a by-hand
instruction-level review a permanent, reproducible gate.

Two complementary gates use arm64 baselines (other architectures report unsupported,
and CI runs them on an Apple-silicon runner):

- **`make asm-diff`** — the tiny, stable numeric/string kernels (`ma`, `ema`, the
  string compare scan). Each is wrapped by a `#[no_mangle] #[inline(never)]` probe
  in `crates/*/examples/asm_probe_*.rs` so it inlines into a standalone symbol;
  `scripts/asm_diff.sh` disassembles and holds the count **exact** (`ma`/`ema`)
  or non-increasing (string) against `scripts/asm_baseline.txt`.
- **`make hot-asm-check`** — the **broad hot-path functions** that are too large to
  inline into a wrapper. `scripts/hot_path_asm.sh` reads each function's *own* symbol
  straight from `rustc --emit asm` and gates its instruction count as `max`
  (must-not-increase) against `scripts/hot_asm_baseline.txt`.

Refresh a baseline only after a *reviewed, justified* change:
`make asm-diff-update` / `make hot-asm-update`, then commit the baseline.

The hot-path inventory (the single source of truth, in `scripts/hot_path_asm.sh`):

| function | crate | hot path |
| --- | --- | --- |
| `fold_forming_row` | volas-core | per-bar: tf-fold combines the forming row in place |
| `combine_at` | volas-core | per-bar: the fold cell-combine (per column, per bar) |
| `rolling_max` / `rolling_min` | volas-compute | van-Herk window extrema kernels |
| `execute` / `dispatch` | volas-directive | per-eval directive evaluation (dump only — `dispatch` grows with the command set) |

To **discover** (not just gate), `make hot-asm` dumps every inventory function's
disassembly to `target/hot-asm/<fn>.s` with a count and a `VS-BASE` delta
(`make hot-asm FN=combine_at` for one). Read those to spot what the optimizer
actually emitted on the hot path: per-bar heap allocations (`bl _..._alloc`),
un-elided bounds-check panics (`b.hs …; bl …panic`), broken inlining, register
spills, or missed SIMD.

The optimization loop, end to end: **`make hot-asm`** (read the hot path) →
hypothesize → implement → **`make perf-ab`** (prove wall-clock) +
**`make asm-diff` / `make hot-asm-check`** (prove no instruction regression
elsewhere) → `*-update` the baseline once the win is reviewed. A periodic broad
sweep across every hot path (an adversarially-verified, assembly-level review)
seeds new inventory entries and optimization candidates.

### Opt-in profile-guided release builds

`make build`, `make release`, `make build-pkg` and both sides of `make perf-ab`
use `scripts/build_release.py`. The default remains `BUILD_OPT=normal`.
The complete local PGO loop is validated on macOS arm64; other release targets
still need independent validation and CI integration.
Run `make install` to provision the repository pins before building. Have the
hash-locked wheels in `scripts/pgo-requirements.txt` available beforehand. The
current training lock covers CPython 3.12 on macOS arm64. The owner installs these
wheels into a fresh isolated training directory with `pip --require-hashes`, records
the selected wheel SHA-256 and verifies the installed payload before each stage.
For offline training, pass `--dependency-wheels <wheelhouse>` to the build owner.

```sh
make build BUILD_OPT=pgo BUILD_ARTIFACTS=/absolute/fresh/artifacts
python scripts/build_release.py --mode generate --artifacts /absolute/fresh/generate
python scripts/build_release.py --mode use --profile /absolute/profile.profdata \
    --artifacts /absolute/fresh/use
```

`pgo` builds an instrumented wheel, installs it in an isolated site, runs the
fixed-operation training workload, merges raw profiles, then builds and smoke-tests
the final use wheel. It retains logs, hashes, stage timings, raw profiles and
provenance. `use` requires the matching JSON sidecar. Source, ABI, target,
toolchain, configuration, training code and dependency mismatches fail closed.
Release acceptance rejects an optimized owner interpreter (`-O`, `-OO`, or
inherited nonzero `PYTHONOPTIMIZE`) before building. The owner and training
coordinator explicitly launch children with `PYTHONOPTIMIZE=0`. All training
acceptance checks and the final smoke use explicit checks that raise, so even
a directly invoked optimized interpreter still validates operations, inventory,
shapes, masks and values. The smoke executes `ma:2` and checks its warm-up mask
and numerical results before publishing any final identity.
Every invocation needs fresh stage directories. No profile is shared across sources.
A completed instrumented stage can be supplied with `--generated <generate-stage>`
to `--mode pgo`: its native hash, actual compiler arguments and compilation identity
are verified against the complete current identity, including training inputs.
Training always uses a new empty raw directory. A changed training identity requires
a fresh generate build.
Schema 2 manifests have one effective configuration identity: ordered Cargo-resolved
flags (including duplicates), the release profile, resolved features, actual compiler
and Cargo executables, linker/SDK identities and the deployment target. Profile and
pair checks compare that identity independently of the source hash. The build owner
constructs every child build environment from an explicit allowlist. Ambient compiler
and linker settings (including `LDFLAGS`, `CFLAGS`, `CPPFLAGS`, `RUSTFLAGS`, wrappers,
per-target variants and unknown build-script inputs) are removed before configuration
resolution or compilation. Their names are recorded under `environment.removed` in
build manifests; retained inputs and owner-supplied flags/tools are recorded separately.
Removed settings do not enter the build identity. Cargo registry authentication
remains available to the existing publish command; records retain only its SHA-256
fingerprint, never the token. `MACOSX_DEPLOYMENT_TARGET` remains
bound to that identity; explicit SDK/developer-directory overrides, Python optimization,
and unrepresentable Cargo configuration remain checked and rejected as before.
The same environment owner serves coverage, audit builds, assembly gates and Make's
debug/check/lint/publish builds through `scripts/build_env.py`; their command and
environment records live in `target/build-environments/`. Coverage and checked audit
supply `--rust-checks coverage|checked`; the owner constructs their fixed flags.
Shell entries initialize their own argument arrays before selecting a mode.
The shell boundary's `--set` accepts only the fixed allocator checks and an
absolute coverage profile path ending in `-%m-%p.profraw` with coverage selected;
arbitrary compiler settings and unknown inputs fail before a child is started.
Shell expansion cannot prove assignment provenance, so it cannot promote a
variable's arbitrary value to an owned compiler input.
Coverage also supplies its allocator guards (`MallocPreScribble=1` and
`MALLOC_PERTURB_=170`) as owned inputs, so Rust tests retain the same checks as
Python tests. Explicit owner inputs are preserved by provenance, irrespective of
their names; ambient values never become owned merely by matching those names.
All subprocess calls in `scripts/` use the environment owner to record the actual
command and child inputs before execution, including tool probes, pip dependency
installs, training stages and Cargo-to-rustc handoff. Runtime-only tools explicitly
preserve their runtime controls at this boundary. The Cargo resolver
records the complete target-info query and refuses an unfamiliar query grammar.
Both resolved flags and real rustc argv must match the requested mode and absolute
profile path. One argument parser accepts `-Cx`, `-C x`, `--codegen=x`, and
`--codegen x`, normalizing underscores in option names only. The same parser
checks mode, duplicate feedback options, paths, and forbidden overrides at
configuration preflight and on recorded compiler invocations. Host crates cannot
receive feedback flags. Coverage instrumentation is rejected in every release mode.
Rustc `@response-file` arguments are rejected by the same
validator at configuration preflight, resolved flags and actual argv; response-file
contents are not supported as build inputs. Cargo resolves ISA
flags; the owner forwards those resolved flags to maturin so macOS link arguments
cannot mask FMA or PGO instrumentation. Fat LTO and the release profile stay in Cargo.

Training covers the public indicator inventory and the existing live, first-cache,
range and interop factories. Its versioned operation budget is independent of
elapsed time (four fixed passes). Every stage checks state changes, lengths,
missing masks and values against operation expectations, and feeds verified results
into its digest. First-cache validation reads the requested result with
`get_column`, which cannot compute a missing cache; a factory caching a different
indicator fails before validation can materialize the requested one. The merged
profile retains function counters. A missing-profile
diagnostic for an executed training function fails the use build; untrained cold
misses are listed individually with counter evidence and reasons in
`profile-warnings.json`. Unrecognized diagnostic formats and CFG/hash mismatches
also fail. Hold-out inputs are declared in `scripts/pgo_train.py` and implemented in
`test/bench_holdout.py` (batch, 5m-to-1h folding/resends, and sliced/chunked/null
Arrow plus strided NumPy inputs); they never contribute to training. Training has a ten-minute hard cap. A successful build is
not a performance adoption verdict.

`make test` first runs the functional suite against an installed owner-built wheel,
then measures coverage with separate instrumentation and restores that exact wheel.
Build stages retain intermediate records in `build.json`. Only final normal/use
sites receive a consumable `volas-build.json`, after provenance, smoke and
uninstrumented checks all succeed; generate sites have no consumable identity.
Failed final checks leave a site that both performance entry points reject.
The installation identity owner registers `volas-build.json` in the installed
distribution RECORD, so replacing a release with `make build-ext` or direct
`maturin develop` removes the old identity. Coverage validates and saves its own
copy before installation changes, then restores the wheel and its registered
identity together. A debug install has no release identity; its full test run
measures coverage and restores a normal release through the build owner.
Keep coverage and performance builds sequential.

For code regressions, `make perf-ab BASE=<ref> BUILD_OPT=normal` compares normal
builds, and `BUILD_OPT=pgo` trains each source separately. The current driver also
builds historical archives. `PERF_COMPARE=optimization` requires identical source
hashes and compares normal against PGO. A reused `BASE_SITE` must have a valid
`volas-build.json` and matching native-library hash. Both the wall driver and
`scripts/perf_cpu.py <base-site> <head-site> <output.json> <manifest.json>` reject
mismatched modes or provenance before measurement. The CPU driver retains 200
AB/BA process-CPU pairs per item; launch it twice with fresh workers for two runs.
With `PERF_AB_JSON_DIR`, perf-ab also retains each side's build manifests, logs,
raw/indexed profiles, wheels and installed sites. Cargo target caches are excluded.

Normal assembly baselines remain unchanged. Missing symbols fail gates and other
architectures report unsupported. To inspect a final wheel separately, run
`python scripts/pgo_asm.py --site <use/site> --output <report-dir>` (optionally
`--base-report <normal-report-dir>`), or set `PGO_SITE` and `PGO_ASM_OUT` when
running either assembly gate. Reports contain final addresses, normalized
instructions, branch edges, calls, stack accesses and NOP counts. Every selected
function must contain instructions. Address references use defined symbols,
section-relative locations (distinguishing same-named sections) and relocation records; unresolved references are
explicitly marked, rather than trusting LLVM nearest-symbol annotations. Comparison
reports list added, removed and modified functions across both inventories.
The single reference owner consumes LLVM's numeric disassembly, Mach-O
indirect-symbol/bind/dyld tables, ELF dynamic relocations and LLVM's PLT labels.
Stub and GOT/PLT operands name their actual imports, including binding addends;
slot addresses are never import identities. Bindings include the namespace,
Mach-O two-level library ordinal and full install name, symbol, ELF symbol
version and its required provider where present, and addend. Components are escaped
before rendering, so delimiters and literal escapes cannot collapse distinct bindings.
Mach-O short dylib names must map to a unique load command. Indirect stub
entries resolve through their actual loaded pointer, including ARM page pairs,
literal loads and x86 RIP-relative loads; symbol names alone cannot supply a
binding from another slot. ELF dynamic-symbol versions qualify both PLT and data
relocations. Absent or ambiguous provider/version metadata stays unresolved. An unpaired ADRP that overlaps an import page
remains unresolved, even if LLVM binds the page's first slot. Missing or conflicting identities
remain explicit unresolved references. A comparison with any unresolved reference
has `comparison_complete=false` and marks the affected function as changed even
when its slot is identical. LLVM's generic `--symbolize-operands` alone retains
nearest-symbol guesses for Mach-O stubs and removes numeric CFG targets, so it
cannot replace these tables. The owner retains basic-block ARM64 ADRP/ADD/LDR and
x86 RIP operand normalization where LLVM does not provide a complete operand.
Address-only moves compare equal; target changes and non-address immediates
remain visible.
Stack accesses come from the shared instruction classifier: ARM64 load/store
forms (including unscaled and pre/post-index forms) through SP or x29 established
from SP. Frame facts propagate through CFG joins and end on register overwrites;
index writeback preserves a frame-derived address. A prologue invariant covers
indirect-jump and exception blocks when all later frame-pointer definitions
establish the frame or restore it along an exit path. `stack_accesses` lists memory
accesses, while `confirmed_spills` requires separate compiler evidence and is
currently empty; this does not assert that no spills occurred.
The existing CI jobs read the repository pins through the shared reader and
select that compiler and its LLVM components. Python build jobs install the exact
maturin requirement. Their prerequisite checks verify the active compiler, Cargo,
maturin and sysroot LLVM tools before running tests or gates. Both normal assembly
gates use the same Rust prerequisite checker.

`make build-pkg` writes `delivery.json` inside its isolated `BUILD_ARTIFACTS`
directory. `make publish` uploads only that manifest's hash-verified wheel and source
distribution. To upload a previously built package, run
`make upload BUILD_ARTIFACTS=<that-build-directory>`. Other files in `dist/` are
untouched and never selected implicitly.

Normal assembly gates remove ambient target-triple redirection and reject it in Cargo
config. Both gates pass the same explicit `--target-dir` to Cargo and read outputs
from that directory (`CARGO_TARGET_DIR`, or the repository's `target` directory),
so a configured build-directory redirect cannot select stale probes.
The hot-path gate emits each crate to its own invocation's explicit assembly file,
and rejects a symbol body without instructions. Final-library branch normalization
splits opcodes on any whitespace and uses opcode-specific address operands,
preserving bit-test immediates and separating indirect calls and address references
from CFG edges. ARM64 direct branches/calls and x86 conditional/unconditional
jumps, calls, and loop branches use the same classification owner.
