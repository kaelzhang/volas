"""The single place that states every contract the differential harness checks.

Each switch below is the one knob a later contract decision flips. Every check in
the harness reads its expectation from here, never from a literal of its own, so a
decided contract changes the whole harness at once.

``OPEN_DEFECTS`` lists the reproduced defects the property machines must step
around until they are fixed: an op or an observation that would trip one of them
is not generated (see :func:`avoids`), and a dedicated ``xfail(strict=True)``
reproduction in ``test_harness_defects.py`` pins it instead. Fixing a defect means
deleting its id here (the machines then hunt it again) and removing that xfail.
"""

from __future__ import annotations

# --- defects the machines step around (each has a strict-xfail reproduction) -----

OPEN_DEFECTS: frozenset[str] = frozenset({
    # A recursive cached column without a carried state (a composite / explicit
    # series) is not continued across a slice that drops its history's start: it
    # stays as data (it cannot be continued exactly, so it is never re-warmed).
    'slice-recursive-rewarm',
    # A slice holding just a column's lookback keeps it cached, but a re-sent last row
    # reads the lookback rows before it, which the slice does not hold: the recompute
    # gives a missing value where the full history has one.
    'slice-resend-short',
})

# Directive families whose incremental refresh is known to diverge from the full
# compute (each pinned by a strict-xfail reproduction and by KNOWN_DIVERGENT in
# crates/volas-directive/tests/resume_matrix.rs); the machines do not cache them
# while listed.
DIVERGENT_FAMILIES: frozenset[str] = frozenset({
})


def avoids(defect: str) -> bool:
    """Whether the machines must not generate the trigger of ``defect`` (still open)."""
    return defect in OPEN_DEFECTS


# --- naming (README "directive_stringify") -------------------------------------------

# A bare name that is both a data column and a built-in command usable bare raises at
# bind time (REQUIREMENTS R17); a data column named like a bare command that a cached
# directive reads cannot be added (R17). A new directive binds data columns only.
AMBIGUOUS_BARE_NAME_RAISES = True
COLUMN_UNDER_A_CACHED_COMMAND_RAISES = True

# `directive_stringify(d)` is "the actual column name volas caches it under".
COLUMN_NAME_IS_STRINGIFY = True

# --- freshness of read surfaces (README "df.fulfill()" / "df.exec") ----------------

# `df.exec(d)` is a pure evaluation that reads no cache: its value equals a fresh
# compute even while the frame is stale.
EXEC_IS_FRESH = True
# A bulk / positional read (to_numpy, iloc, equals, to_pandas, ...) of a stale frame
# raises instead of returning stale values.
BULK_READ_STALE_RAISES = True
# `get_column` of a stale computed column refreshes it, like `df[name]`: it is a
# column read (CONTRACT #12), never a computation of a missing column.
GET_COLUMN_REFRESHES = True
# A comparison whose right-hand frame is stale raises like a stale left-hand side.
COMPARE_STALE_RHS_RAISES = True
# Appending a stale source frame raises (the source is read in bulk).
APPEND_STALE_SOURCE_RAISES = True
# Appended rows carrying one of the target's cached directive columns raise, in every
# append form (REQUIREMENTS R20): select the data columns first.
APPEND_CACHED_COLUMNS_RAISES = True

# --- append (I12) ---------------------------------------------------------------------

# Rows are keyed by their labels (docs/dataframe-api.md "df.append"): a row labelled
# like the last row replaces it; a DatetimeIndex only increases (a row labelled before
# the last raises); an integer index stays unique (a row labelled like an earlier one
# raises); a bare RangeIndex bar takes the next labels. A rejected append leaves the
# frame unchanged. See `Ledger.accepts` / `Ledger.append`.
APPEND_KEYED_BY_LABEL = True
# A bar whose index kind differs from the frame's (RangeIndex onto DatetimeIndex),
# a naive bar onto an aware frame (or the reverse), and a NaT-labelled bar raise.
APPEND_FOREIGN_INDEX_RAISES = True

# --- history of derived frames (I11) ---------------------------------------------------

# `iloc[a:]` keeps every already-cached column on its full-history continuation
# (values and carried state), while a directive first cached on the slice is
# computed over the retained rows only.
SLICE_CACHED_COLUMNS_CONTINUE = True

# --- tf fold (README "Live cumulation") --------------------------------------------------

# Aggregation of a missing fine value (docs/cumulation.md, "A missing input is never
# skipped"; a documented pandas divergence): a period's first / last / max / min /
# sum is missing as soon as any of its values is — for every dtype, identical in the
# live fold and the batch cumulate, never ±inf. The reference ledger implements it.
# A frame built with `time_frame=` from closed bars: the behaviour of a fine bar
# that falls in the LAST seeded period is undocumented, so it is not generated.
TF_SEED_LAST_PERIOD_GENERATED = False

# --- numeric exactness (I1) ---------------------------------------------------------------

# Incremental results are bit-exact with the oracle unless the directive's family
# is listed here as {family: relative tolerance}. Starts empty.
NUMERIC_EXCEPTIONS: dict[str, float] = {}
# While 'window-recompute-ulp' is open, finite-window results are compared to this
# relative tolerance and a comparison whose operands tie within it is not judged.
ULP_DRIFT_TOLERANCE = 1e-9

# --- bounded windows (docs "Bounded rolling window") -------------------------------------

# A windowed frame computes a directive requested after construction only when it is of
# history class 1 (a trailing window, no running accumulator, no recursion) and the margin
# holds its lookback; classes 2 and 3 must be declared in max_lookback, so a later request
# raises ValueError — as does one reading more rows than the margin. A declared or
# admitted column equals the unbounded history bit for bit, writes included.
WINDOWED_LATE_RECURSION_RAISES = True
