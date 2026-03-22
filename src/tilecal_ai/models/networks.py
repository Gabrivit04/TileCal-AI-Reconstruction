from __future__ import annotations

import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self, n_samples: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_samples, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyCNN1D(nn.Module):
    def __init__(self, n_samples: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1, groups=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, 2)
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), 1, self.n_samples)
        x = self.features(x).squeeze(-1)
        return self.head(x)
