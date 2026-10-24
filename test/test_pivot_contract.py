"""A pivot becomes knowable on confirmation; published guidance must agree."""
from pathlib import Path

import numpy as np
import pytest
import volas

from .exact import assert_exact


@pytest.mark.parametrize('kind,sign', [('pivothigh', 1), ('pivotlow', -1)])
def test_pivot_confirmation_never_backfills_the_pivot_bar(kind, sign):
    values = sign * np.array([1., 2., 5., 3., 2., 1.])
    directive = f'{kind}:2,2@close'
    full = volas.DataFrame({'close': values}).exec(directive)
    assert_exact(full, [np.nan, np.nan, np.nan, np.nan, sign * 5., np.nan])
    live = volas.DataFrame({'close': values[:3]})
    live[directive]
    for end in range(4, len(values) + 1):
        live.append(volas.DataFrame({'close': values[end - 1:end]}))
        got = live[directive].to_numpy()
        assert_exact(got, full[:end])
        assert np.isnan(got[2])


def test_pivot_documentation_describes_a_causal_confirmation():
    root = Path(__file__).resolve().parents[1]
    doc = (root / 'docs/indicators-tradingview.md').read_text()
    pivot = doc.split('Fractal swing-point detection.', 1)[1].split('## ', 1)[0]
    comment = (root / 'crates/volas-compute/src/raw/indicators/oscillators.rs').read_text()
    comment = comment.split('/// Fractal pivot detection', 1)[1].split('pub fn pivot', 1)[0]
    for description in (pivot, comment):
        assert 'non-causal' not in description
        assert 'causal' in description
        assert 'back-fill' in description
