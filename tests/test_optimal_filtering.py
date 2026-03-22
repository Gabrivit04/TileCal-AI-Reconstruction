from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tilecal_ai.baselines.optimal_filtering import OptimalFiltering2


def test_optimal_filtering_inference_shapes() -> None:
    w_a = np.ones(7, dtype=np.float32) / 7.0
    w_t = np.linspace(-1.0, 1.0, 7, dtype=np.float32)
    of2 = OptimalFiltering2(w_a, w_t)

    adc = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    out = of2.infer(adc)

    assert isinstance(out.amplitude, float)
    assert isinstance(out.timing, float)
