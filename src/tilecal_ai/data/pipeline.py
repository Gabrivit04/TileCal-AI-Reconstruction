from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PulseSample:
    adc: np.ndarray
    amplitude: float
    time: float


class TileCalPulseDataset(Dataset):
    """Torch-ready dataset with optional synthetic pile-up/noise augmentation."""

    def __init__(self, pulses: np.ndarray, targets: np.ndarray, noise_std: float = 0.0):
        if pulses.ndim != 2:
            msg = "pulses must be [n_events, n_samples]"
            raise ValueError(msg)
        if targets.shape[-1] != 2:
            msg = "targets must contain [amplitude, time]"
            raise ValueError(msg)
        self.pulses = pulses.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.noise_std = noise_std

    def __len__(self) -> int:
        return int(self.pulses.shape[0])

    def __getitem__(self, idx: int):
        x = self.pulses[idx].copy()
        if self.noise_std > 0:
            x += np.random.normal(loc=0.0, scale=self.noise_std, size=x.shape).astype(np.float32)
        y = self.targets[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def pedestal_subtract_and_normalize(pulses: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pedestal = pulses[:, :2].mean(axis=1, keepdims=True)
    centered = pulses - pedestal
    scale = np.maximum(np.abs(centered).max(axis=1, keepdims=True), eps)
    return centered / scale
