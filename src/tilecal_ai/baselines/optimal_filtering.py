from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class OF2Result:
    amplitude: float
    timing: float


class OptimalFiltering2:
    """Minimal OF2 baseline implementation for TileCal pulse windows."""

    def __init__(self, amplitude_weights: np.ndarray, timing_weights: np.ndarray) -> None:
        self.amplitude_weights = np.asarray(amplitude_weights, dtype=np.float32)
        self.timing_weights = np.asarray(timing_weights, dtype=np.float32)

    def infer(self, adc: np.ndarray) -> OF2Result:
        x = np.asarray(adc, dtype=np.float32)
        if x.shape[-1] != self.amplitude_weights.shape[-1]:
            msg = "Input window and OF2 weights must have identical length."
            raise ValueError(msg)
        amplitude = float(np.dot(self.amplitude_weights, x))
        timing = float(np.dot(self.timing_weights, x) / max(abs(amplitude), 1e-6))
        return OF2Result(amplitude=amplitude, timing=timing)
