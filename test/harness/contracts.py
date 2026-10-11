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
    # A failed append leaves earlier columns grown (non-atomic append / fold).
    'append-atomicity',
    # A composite cached while a bare-named cached column is stale reads the stale
    # placeholder; a plain column named like a command rebinds cached composites.
    'stale-name-binding',
    # `exec` of a bare cached name and `get_column` of a stale computed column
    # return the stale placeholder instead of raising / recomputing.
    'unguarded-stale-read',
    # Append coerces a foreign index kind / timezone / NaT instead of raising.
    'append-index-coercion',
    # The window-recompute refresh differs from the full kernels in the last ULPs
    # (and so can flip a comparison sitting exactly on the threshold).
    'window-recompute-ulp',
    # Windowed frames: recursive columns are not bit-exact with the unbounded
    # history, and a base write restarts carried state on the retained rows only.
    'windowed-drift',
    # A comparison consumes a stale right-hand frame without raising.
    'stale-compare-rhs',
    # Integer folds read the NA placeholder as a value.
    'int-fold-na',
    # tz_localize drops the sub-second part of every timestamp.
    'localize-subsecond',
    # directive_stringify keeps the argument tokens as written ('ma:05', 'sar:0.020')
    # while df[d] names the column from the bound values ('ma:5', 'sar').
    'stringify-column-name',
    # An integral literal beyond i64 saturates in the cached name, so a refresh
    # re-parsing the name computes with a different constant.
    'literal-saturation',
    # The cached name of a command whose series is a parenthesized single primary
    # drops the parentheses (`ema:5@(rsi:5)` -> `ema:5@rsi:5`, `bbi@(0)` -> `bbi@0`)
    # and does not re-parse.
    'nested-series-name',
    # append reads a source frame without a freshness check (stale placeholders are
    # appended as data).
    'stale-append-source',
    # `macd` with fast > slow panics (a Rust slice-index panic) instead of erroring.
    'macd-fast-above-slow-panic',
    # A windowed frame checks freshness on its visible slice, where a cached column
    # longer than the window is no longer computed: a stale read then succeeds.
    'windowed-stale-logical-read',
    # An all-NaN period folds high / low to -inf / +inf instead of NaN (already
    # tracked); the tf reference reproduces that result while this is open.
    'all-nan-fold-extreme',
    # A finite but huge input (1e300) that has left a sliding window still skews the
    # full kernels' running sums (linearreg & co.): extreme magnitudes are only
    # generated in frames that hold nothing else.
    'sliding-sum-magnitude',
})

# Directive families whose incremental refresh is known to diverge from the full
# compute (each pinned by a strict-xfail reproduction and by KNOWN_DIVERGENT in
# crates/volas-directive/tests/resume_matrix.rs); the machines do not cache them
# while listed.
DIVERGENT_FAMILIES: frozenset[str] = frozenset({
    # a state captured inside the warm-up resumes with values where the full
    # compute is still NaN
    'adosc', 'ht_dcperiod', 'ht_dcphase', 'ht_phasor', 'ht_sine', 'ht_trendline', 'ht_trendmode',
    'mama',
    # the resume warm-up gate ignores a leading / interior missing-row shift
    'dema', 'tema', 't3', 'kdj', 'cmo',
    # the resume divides by a zero prior where the full kernel yields 0
    'trix',
    # last-ULP drift between the resume and the full kernel (already tracked)
    'stochrsi',
    # cancellation in the running co-moment sums: a window recompute and the full
    # kernel differ by more than 1e-9 relative, and |correl| exceeds 1
    'correl', 'beta',
    # cancellation in vr's running up / down volume sums: a window with no down
    # volume yields a huge ratio of either sign (the full kernel and a window
    # recompute disagree by orders of magnitude)
    'vr',
    # a NaN input keeps suppressing a candlestick pattern far beyond its lookback in
    # the full kernel, so any later window recompute disagrees with it
    'style', 'cdl',
})


def avoids(defect: str) -> bool:
    """Whether the machines must not generate the trigger of ``defect`` (still open)."""
    return defect in OPEN_DEFECTS


# --- naming (README "directive_stringify") -------------------------------------------

# `directive_stringify(d)` is "the actual column name volas caches it under".
COLUMN_NAME_IS_STRINGIFY = True

# --- freshness of read surfaces (README "df.fulfill()" / "df.exec") ----------------

# `df.exec(d)` is a pure evaluation that reads no cache: its value equals a fresh
# compute even while the frame is stale.
EXEC_IS_FRESH = True
# A bulk / positional read (to_numpy, iloc, equals, to_pandas, ...) of a stale frame
# raises instead of returning stale values.
BULK_READ_STALE_RAISES = True
# `get_column` of a stale computed column raises (it is one of the "every other
# read" surfaces the README lists as raising while stale).
GET_COLUMN_STALE_RAISES = True
# A comparison whose right-hand frame is stale raises like a stale left-hand side.
COMPARE_STALE_RHS_RAISES = True
# Appending a stale source frame raises (the source is read in bulk).
APPEND_STALE_SOURCE_RAISES = True

# --- append (I12) ---------------------------------------------------------------------

# A plain (non-tf) DatetimeIndex frame accepts a bar earlier than its last row
# (no documented ordering rule for plain frames; the tf fold raises instead).
PLAIN_APPEND_OUT_OF_ORDER_ACCEPTED = True
# A bar whose index kind differs from the frame's (RangeIndex onto DatetimeIndex),
# a naive bar onto an aware frame (or the reverse), and a NaT-labelled bar raise.
APPEND_FOREIGN_INDEX_RAISES = True

# --- history of derived frames (I11) ---------------------------------------------------

# `iloc[a:]` keeps every already-cached column on its full-history continuation
# (values and carried state), while a directive first cached on the slice is
# computed over the retained rows only.
SLICE_CACHED_COLUMNS_CONTINUE = True

# --- tf fold (README "Live cumulation") --------------------------------------------------

# Float aggregation of a NaN fine value: first / last keep it as it is, max / min
# skip it, sum propagates it. (Fold and batch cumulate agree on this today; pandas
# resample would skip NaN in every aggregator — an open contract question.)
TF_FLOAT_FIRST_LAST_KEEP_NAN = True
TF_FLOAT_MAX_MIN_SKIP_NAN = True
TF_FLOAT_SUM_PROPAGATES_NAN = True
# Integer aggregation skips NA cells (pandas semantics); an all-NA bucket is NA.
TF_INT_AGG_SKIPS_NA = True
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
# Windowed frames (README "Bounded rolling window"): finite-window indicators match
# the unbounded history to ~1e-13 relative; recursive ones are bit-exact.
WINDOWED_FINITE_TOLERANCE = 1e-12
