from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TileCalCNN1D(nn.Module):
    def __init__(self, input_len: int = 7, channels: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DepthwiseSeparableConvBlock(channels),
            DepthwiseSeparableConvBlock(channels),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * input_len, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.head(self.stem(x))
