"""Ultra-compact neural models for TileCal pulse regression."""

from __future__ import annotations

import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, input_size: int = 7, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, input_size: int = 7, channels: int = 16) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1),
            nn.Conv1d(1, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.head(self.feature(x))


class TinyGRU(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        y, _ = self.gru(x)
        return self.out(y[:, -1, :])
