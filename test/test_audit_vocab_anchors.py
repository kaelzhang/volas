"""Systematic audit — external vocabulary anchors (SPEC §6.3 / G0).

The audit's vocabularies are GENERATED from machine-readable sources, never
hand-written (P3's lesson):
  - D (dtypes)        == the Rust `DType` enum, exported as `volas_rs._dtypes`.
  - T11 (commands)    == the spec.rs registry, exported as `_directive_commands`.
  - G0: every subject the manifest marks `covered` has at least one audit module.
"""

from __future__ import annotations

import pathlib

import pytest

import volas
import volas_rs

from . import audit_dims as A


def test_dtype_vocabulary_is_rust_sourced():
    """audit_dims.DTYPES == the Rust DType enum: a new Rust dtype auto-extends
    the audit matrix (or trips here until the fixture factory learns it)."""
    rust = set(volas_rs._dtypes)
    audit = {A._DTYPE_STR[d] for d in A.DTYPES}
    assert audit == rust, f"audit D vocabulary drifted from the Rust enum: {audit ^ rust}"


def test_directive_vocabulary_is_registry_sourced():
    """T11's command vocabulary is the spec.rs registry. Total-vocabulary check:
    the parser RECOGNISES every registered command (a missing-arg error is fine;
    'unknown command' is not), and recognises nothing the registry doesn't claim."""
    commands = volas_rs._directive_commands
    assert len(commands) > 100                      # the registry, not a sample
    for cmd in commands:
        try:
            volas.directive_lookback(cmd)           # bare form: ok or needs args
        except volas.DirectiveError as e:
            assert "unknown command" not in str(e), f"registry command {cmd!r} unparsed"
    # (the unknown-command rejection itself is guarded in test_audit_t11_directive;
    # bare directive_lookback treats an unknown name as a plain column reference.)


_SUBJECT_MODULES = {
    "T0": ["test_audit_t0_meta.py"],
    "T1": ["test_audit_t1_unary.py", "test_audit_param_census.py",
           "test_audit_frame_series_features.py"],
    "T2": ["test_audit_t2_reduce.py"],
    "T3": ["test_audit_t3_binop.py", "test_audit_irep.py"],
    "T4": ["test_audit_t4_fill.py"],
    "T5": ["test_audit_t5_namask.py"],
    "T6": ["test_audit_t6_order.py"],
    "T7": ["test_audit_t7_index.py", "test_audit_index_axis.py"],
    "T8": ["test_audit_t8_astype.py", "test_audit_t8_export.py", "test_audit_csv.py"],
    "T9": ["test_audit_t9_construct.py", "test_audit_datetime.py"],
    "T10": ["test_audit_state.py"],
    "T11": ["test_audit_t11_directive.py"],
    "T12": ["test_audit_t12_scalar.py", "test_audit_timestamp_methods.py"],
    "T13": ["test_audit_t13_tz.py"],
    "T15": ["test_audit_t15_dt.py"],
    "Row": ["test_audit_t0_meta.py"],
}


def test_g0_every_covered_subject_has_audit_modules():
    """G0: a `covered` manifest disposition is only honest if the subject has at
    least one existing audit module — 'declared' must imply 'exercised'."""
    from .test_audit_manifest import CLASSIFICATION
    here = pathlib.Path(__file__).parent
    used_subjects = {
        v for mapping in CLASSIFICATION.values() for v in mapping.values()
        if isinstance(v, str) and (v.startswith("T") or v == "Row")
    }
    for subj in sorted(used_subjects):
        mods = _SUBJECT_MODULES.get(subj)
        assert mods, f"subject {subj} has no registered audit modules (G0)"
        for m in mods:
            assert (here / m).exists(), f"{subj}: audit module {m} missing"
