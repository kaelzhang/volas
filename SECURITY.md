# Security Policy

## Supported versions

volas is pre-1.x in API-stability terms (currently published as Beta). Security
fixes target the latest released version on PyPI.

## Reporting a vulnerability

Please report security issues **privately**, not in a public issue:

- Use GitHub's [private vulnerability reporting](https://github.com/kaelzhang/volas/security/advisories/new), or
- email **i+pypi@kael.me** with the details.

Include a description, affected version(s), and a minimal reproduction if you
can. You can expect an initial acknowledgement within a few days. Please give a
reasonable window to release a fix before any public disclosure.

## Scope

volas computes indicators/features over OHLCV data; it executes no network
calls and gives no trading advice. The most relevant classes of issue are
memory-safety bugs in the Rust kernels (e.g. an indicator path that reads
uninitialized or out-of-bounds memory) and panics reachable from valid Python
input. Reports in those areas are especially valuable.
