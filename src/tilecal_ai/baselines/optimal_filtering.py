"""Classical Optimal Filtering baseline for amplitude/timing estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OF2Coefficients:
    amplitude_weights: np.ndarray
    timing_weights: np.ndarray


class OptimalFiltering2:
    """Simple OF2 implementation using precomputed linear weights."""

    def __init__(self, coeffs: OF2Coefficients) -> None:
        self.coeffs = coeffs

    def predict(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        amp = samples @ self.coeffs.amplitude_weights
        tau = samples @ self.coeffs.timing_weights
        return amp, tau



def default_of2(window_size: int = 7) -> OptimalFiltering2:
    center = window_size // 2
    amp_w = np.zeros(window_size, dtype=np.float32)
    amp_w[center] = 1.0

    t = np.arange(window_size, dtype=np.float32)
    t = t - t.mean()
    t_norm = t / (np.linalg.norm(t) + 1e-6)
    return OptimalFiltering2(OF2Coefficients(amplitude_weights=amp_w, timing_weights=t_norm))
