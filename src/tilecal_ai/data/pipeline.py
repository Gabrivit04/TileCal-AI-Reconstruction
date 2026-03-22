"""Data ingestion and preprocessing pipeline for TileCal pulses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PipelineConfig:
    window_size: int = 7
    normalize: bool = True
    pedestal_subtraction: bool = True
    noise_std: float = 0.0
    phase_shift_max: int = 0


class TileCalPulseDataset(Dataset):
    """PyTorch-ready dataset with optional on-the-fly augmentation."""

    def __init__(
        self,
        pulses: np.ndarray,
        amplitude: np.ndarray,
        timing: np.ndarray,
        config: PipelineConfig | None = None,
        train: bool = True,
    ) -> None:
        self.config = config or PipelineConfig(window_size=pulses.shape[-1])
        self.train = train

        x = pulses.astype(np.float32)
        if self.config.pedestal_subtraction:
            pedestal = x[:, :1]
            x = x - pedestal

        if self.config.normalize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mean) / std

        self.x = x
        self.y_amp = amplitude.astype(np.float32)
        self.y_time = timing.astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.config.noise_std > 0:
            x = x + np.random.normal(0.0, self.config.noise_std, size=x.shape)
        if self.config.phase_shift_max > 0:
            shift = np.random.randint(-self.config.phase_shift_max, self.config.phase_shift_max + 1)
            x = np.roll(x, shift)
        return x

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.x[idx]
        if self.train:
            x = self._augment(x)
        return {
            "samples": torch.from_numpy(x).float(),
            "amplitude": torch.tensor(self.y_amp[idx]).float(),
            "timing": torch.tensor(self.y_time[idx]).float(),
        }


def make_dummy_dataset(n_events: int = 1024, window_size: int = 7) -> TileCalPulseDataset:
    """Generate synthetic pulse-like samples for development and CI checks."""
    t = np.linspace(0, 1, window_size, dtype=np.float32)
    amp = np.random.uniform(0.5, 10.0, size=n_events).astype(np.float32)
    tau = np.random.uniform(-0.2, 0.2, size=n_events).astype(np.float32)

    pulses = []
    for a, shift in zip(amp, tau, strict=False):
        pulse = a * np.exp(-((t - (0.4 + shift)) ** 2) / (2 * 0.06**2))
        pulses.append(pulse)
    pulses = np.stack(pulses)

    return TileCalPulseDataset(
        pulses=pulses,
        amplitude=amp,
        timing=tau,
        config=PipelineConfig(window_size=window_size, noise_std=0.01, phase_shift_max=1),
        train=True,
    )
