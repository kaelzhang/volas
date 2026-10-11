"""The directive inventory the generators draw from — derived, never hand-listed.

* Commands: the extension's own registry (``volas_rs._directive_commands``).
* Sub-command / alias tokens: every token the spec's sub-command matches name
  (``canon_sub`` and ``command_spec`` in ``spec.rs``) plus the candlestick registry
  (``CANDLE_PATTERNS``); which ``(command, token)`` pairs exist is decided by the
  directive validator itself (``volas.directive_stringify``).
* Argument arity and domains: probed through the validator — the "accepts at most
  N argument(s)" / "argument #i is required" / "argument #i must be ..." messages
  are the spec's ``ArgBound`` rendered, so the generators stay in sync with it.
* Binary operators: the parser's operator table (``peek_binop`` in ``parser.rs``).
* Executability: each line is run once on a small fixture frame; a line that needs
  an explicit second series (``correl``, ``beta``, ``mavp``) gets one.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import volas
import volas_rs

ROOT = Path(__file__).resolve().parents[2]
SPEC_RS = ROOT / 'crates/volas-directive/src/spec.rs'
PARSER_RS = ROOT / 'crates/volas-directive/src/parser.rs'
COMPUTE_SRC = ROOT / 'crates/volas-compute/src'

BASE = ('open', 'high', 'low', 'close', 'volume')


@dataclass(frozen=True)
class ArgDomain:
    """One positional argument's legal domain, as the validator states it."""

    kind: str  # 'int' | 'float' | 'choice' | 'str'
    lo: float = 0.0
    hi: float = 0.0
    choices: tuple = ()
    required: bool = False


@dataclass
class Line:
    """One output line (``ma``, ``macd.signal``, ``style.doji``) and its arguments."""

    head: str  # the spelling before ':' (command or command.sub)
    canonical: str  # directive_stringify of the default form
    aliases: tuple[str, ...]  # alternative heads that canonicalize the same
    args: tuple[ArgDomain, ...]
    series_suffix: str = ''  # '@...' needed for the line to execute ('' if none)
    executes: bool = True
    kind: str = 'num'  # the output's value kind on the fixture: 'num' | 'bool'
    form: str = ''  # a spelling that executes (the default form when it does)


def _const_names(path: Path, const: str) -> list[str]:
    src = path.read_text()
    m = re.search(rf'pub const {const}\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\];', src, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def commands() -> list[str]:
    registry = getattr(volas_rs, '_directive_commands', None)
    return list(registry) if registry else _const_names(SPEC_RS, 'COMMANDS')


def _sub_tokens() -> list[str]:
    src = SPEC_RS.read_text()
    tokens: set[str] = set()
    for group in re.findall(r'Some\(([^()]*)\)', src):
        tokens.update(re.findall(r'"([a-z_0-9]+)"', group))
    return sorted(tokens)


def binary_operators() -> list[str]:
    """Every binary operator spelling in the parser's ``peek_binop`` table."""
    src = PARSER_RS.read_text()
    body = src[src.index('fn peek_binop'):]
    body = body[:body.index('\n    }\n')]
    ops = []
    for a, b in re.findall(r"\(Some\(b'(\\\\|.)'\),\s*(?:Some\(b'(\\\\|.)'\)|_)\)\s*=>", body):
        tok = (a + b).replace('\\\\', '\\')
        ops.append(tok)
    return ops


def _stringify(form: str) -> str | None:
    return None if _error(form) else volas.directive_stringify(form)


def _error(form: str) -> str:
    # `directive_lookback` binds a bare command exactly as written (a missing
    # required argument is an error), where `directive_stringify` lets a bare name
    # pass as a possible column reference.
    try:
        volas.directive_stringify(form)
        volas.directive_lookback(form)
    except (volas.DirectiveError, ValueError) as e:
        return str(e)
    return ''


_BOUND_PATTERNS = [
    (re.compile(r'must be an integer >= (-?\d+)'), lambda m: ArgDomain('int', int(m[1]), int(m[1]) + 30)),
    (re.compile(r'must be an integer in \[(\d+), (\d+)\]'), lambda m: ArgDomain('int', int(m[1]), int(m[2]))),
    (re.compile(r'must be one of (.*?), got'), lambda m: ArgDomain(
        'choice', choices=tuple(int(x) for x in m[1].split(' / ')))),
    (re.compile(r'must be a finite number >= (-?[\d.]+)'), lambda m: ArgDomain('float', float(m[1]), float(m[1]) + 5)),
    (re.compile(r'must be a finite number > (-?[\d.]+)'), lambda m: ArgDomain('float', float(m[1]) + 1e-3, float(m[1]) + 5)),
    (re.compile(r'must be a number in \[(-?[\d.]+), (-?[\d.]+)\]'), lambda m: ArgDomain('float', float(m[1]), float(m[2]))),
    (re.compile(r'must be a finite number'), lambda m: ArgDomain('float', -5.0, 5.0)),
]


def _parse_domain(err: str, i: int) -> ArgDomain:
    if f'argument #{i} ' in err:
        for pattern, build in _BOUND_PATTERNS:
            m = pattern.search(err)
            if m:
                return build(m)
    # The validator accepted a free token: a string argument (a time frame for `hv`).
    return ArgDomain('str', choices=tuple(str(getattr(volas.TimeFrame, a)) for a in ('m1', 'm5', 'H1', 'D1')))


def _arity(head: str) -> int:
    err = _error(f"{head}:{','.join(['1'] * 12)}")
    m = re.search(r'accepts at most (\d+) argument', err)
    return int(m[1]) if m else 0


def _args(head: str) -> tuple[ArgDomain, ...]:
    """Each slot's domain and whether it is required, probed left to right with the
    earlier required slots filled (the validator reports the first bad slot)."""
    arity = _arity(head)
    filled = [''] * arity
    out = []
    for i in range(arity):
        probe = filled[:i] + ['zz']
        domain = _parse_domain(_error(f"{head}:{','.join(probe)}"), i)
        given = f"{head}:{','.join(filled[:i])}" if any(filled[:i]) else head
        required = f'argument #{i} is required' in _error(given)
        if required:
            filled[i] = sample_value(domain, 5)
        out.append(ArgDomain(domain.kind, domain.lo, domain.hi, domain.choices, required))
    return tuple(out)


def default_form(head: str, args: tuple[ArgDomain, ...]) -> str:
    """``head`` with every required argument filled in (``ma`` -> ``ma:5``)."""
    last = max((i for i, a in enumerate(args) if a.required), default=-1)
    if last < 0:
        return head
    vals = [sample_value(a, 5) if a.required else '' for a in args[:last + 1]]
    return f"{head}:{','.join(vals)}"


def sample_value(domain: ArgDomain, hint: int) -> str:
    """A deterministic in-domain token for ``domain`` steered by ``hint``."""
    if domain.kind == 'int':
        lo, hi = int(domain.lo), int(domain.hi)
        return str(min(hi, max(lo, hint)))
    if domain.kind == 'choice':
        return str(domain.choices[hint % len(domain.choices)])
    if domain.kind == 'str':
        return domain.choices[hint % len(domain.choices)]
    lo, hi = domain.lo, domain.hi
    return repr(round(lo + (hi - lo) * ((hint % 7) / 7.0), 4))


def _fixture() -> volas.DataFrame:
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.standard_normal(80))
    o = c + 0.4 * rng.standard_normal(80)
    return volas.DataFrame({'open': o, 'high': np.maximum(o, c) + rng.uniform(0.1, 1.5, 80),
                            'low': np.minimum(o, c) - rng.uniform(0.1, 1.5, 80), 'close': c,
                            'volume': rng.uniform(1, 100, 80)})


@functools.cache
def lines() -> tuple[Line, ...]:
    """Every output line of the directive language, with its argument domains."""
    tokens = _sub_tokens()
    # The candlestick registry, wherever the compute crate keeps it.
    patterns = next((names for path in sorted(COMPUTE_SRC.rglob('*.rs'))
                     if (names := _const_names(path, 'CANDLE_PATTERNS'))), [])
    heads: dict[str, list[str]] = {}
    for cmd in commands():
        candidates = [cmd] + [f'{cmd}.{t}' for t in tokens]
        if cmd == 'style':
            candidates += [f'style.{p}' for p in patterns]
        for head in candidates:
            canon = _stringify(default_form(head, _args(head)))
            if canon is None:
                continue
            key = canon.split(':')[0].split('@')[0]
            heads.setdefault(key, []).append(head)
    fixture = _fixture()
    out = []
    for key, spellings in sorted(heads.items()):
        head = key
        args = _args(head)
        form = default_form(head, args)
        line = Line(head=head, canonical=_stringify(form) or form,
                    aliases=tuple(sorted(set(spellings) - {head})), args=args, form=form)
        # The default spelling, else the first argument given explicitly (a revision
        # may not resolve a bare default form), each with the extra series it needs.
        explicit = f'{head}:{sample_value(args[0], 5)}' if args else form
        for spelling, suffix in [(f, s) for f in dict.fromkeys((form, explicit))
                                 for s in ('', '@close,open', '@close,volume')]:
            try:
                values = fixture.exec(spelling + suffix)
            except Exception:
                continue
            line.form, line.series_suffix = spelling, suffix
            line.kind = 'bool' if values.dtype == np.bool_ else 'num'
            break
        else:
            line.executes = False
        out.append(line)
    return tuple(out)


def executable_lines() -> tuple[Line, ...]:
    return tuple(line for line in lines() if line.executes)


@functools.cache
def unbounded_memory() -> frozenset[str]:
    """Commands with a line whose output still depends on a row more than its
    lookback behind (recursive / cumulative / positional): a large jump early in the
    fixture changes their output after the warm-up of a cut at that row."""
    base = _fixture()
    cols = {c: np.asarray(base[c], float).copy() for c in BASE}
    for c in ('open', 'high', 'low', 'close'):
        cols[c][3] *= 1.5
    cols['volume'][3] *= 7
    jumped = volas.DataFrame(cols)
    out = set()
    for line in executable_lines():
        form = line.form + line.series_suffix
        try:
            start = 4 + 2 * volas.directive_lookback(form) + 2
            a = np.asarray(base.exec(form), float)[start:]
            b = np.asarray(jumped.exec(form), float)[start:]
        except Exception:
            continue
        # A relative 1e-9 slack: running-sum kernels keep a rounding residue of a row
        # that left their window (the open last-ULP drift), which is not memory.
        scale = np.nanmax(np.abs(a), initial=1.0) if np.isfinite(a).any() else 1.0
        if start >= len(base) or not np.allclose(a, b, rtol=0, atol=1e-9 * scale, equal_nan=True):
            out.add(line.head.split('.')[0])
    return frozenset(out)


@functools.cache
def position_valued() -> frozenset[str]:
    """Heads whose output is a row position (it shifts by exactly the dropped rows
    when the same data is computed without its first rows)."""
    fixture = _fixture()
    tail = fixture.iloc[20:]
    out = set()
    for line in executable_lines():
        form = line.form + line.series_suffix
        try:
            full = np.asarray(fixture.exec(form), float)[40:]
            cut = np.asarray(tail.exec(form), float)[20:]
        except Exception:
            continue
        ok = np.isfinite(full) & np.isfinite(cut)
        if ok.any() and np.array_equal(full[ok] - cut[ok], np.full(ok.sum(), 20.0)):
            out.add(line.head)
    return frozenset(out)
