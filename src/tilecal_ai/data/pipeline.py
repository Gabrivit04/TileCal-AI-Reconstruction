from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PulseSample:
    """Container for one TileCal pulse window."""

    adc: np.ndarray
    amplitude: float
    timing: float
    pileup_mu: float


class TileCalPreprocessor:
    """Fast NumPy-only preprocessing ready for PyTorch conversion.

    This class is deliberately framework-agnostic so it can be re-used in unit tests,
    notebook prototyping, and production data loading.
    """

    def __init__(self, pedestal: float = 0.0, eps: float = 1e-8) -> None:
        self.pedestal = pedestal
        self.eps = eps

    def subtract_pedestal(self, adc: np.ndarray) -> np.ndarray:
        return np.asarray(adc, dtype=np.float32) - self.pedestal

    def normalize(self, adc: np.ndarray) -> np.ndarray:
        arr = np.asarray(adc, dtype=np.float32)
        scale = np.max(np.abs(arr), axis=-1, keepdims=True)
        return arr / np.maximum(scale, self.eps)

    def augment(
        self,
        adc: np.ndarray,
        noise_std: float = 0.01,
        phase_shift_max: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        generator = rng if rng is not None else np.random.default_rng()
        arr = np.asarray(adc, dtype=np.float32).copy()

        if phase_shift_max > 0:
            shift = int(generator.integers(-phase_shift_max, phase_shift_max + 1))
            arr = np.roll(arr, shift, axis=-1)

        noise = generator.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
        return arr + noise

    def emulate_pileup(self, signal: np.ndarray, pileup: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        signal_arr = np.asarray(signal, dtype=np.float32)
        pileup_arr = np.asarray(pileup, dtype=np.float32)
        return signal_arr + alpha * pileup_arr

    def prepare(self, signal: np.ndarray, pileup: np.ndarray | None = None) -> np.ndarray:
        arr = self.subtract_pedestal(signal)
        if pileup is not None:
            arr = self.emulate_pileup(arr, pileup)
        return self.normalize(arr)
