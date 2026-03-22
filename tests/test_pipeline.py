from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tilecal_ai.data.pipeline import TileCalPreprocessor


def test_prepare_normalizes_to_unit_scale() -> None:
    prep = TileCalPreprocessor(pedestal=1.0)
    signal = np.array([1, 2, 3, 4, 5, 4, 3], dtype=np.float32)
    out = prep.prepare(signal)
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_pileup_emulation_adds_components() -> None:
    prep = TileCalPreprocessor()
    signal = np.ones(7, dtype=np.float32)
    pileup = np.full(7, 2.0, dtype=np.float32)
    merged = prep.emulate_pileup(signal, pileup, alpha=0.5)
    np.testing.assert_allclose(merged, np.full(7, 2.0, dtype=np.float32))
