from __future__ import annotations

import torch
from torch import nn


class AmplitudeTimeLoss(nn.Module):
    """Multi-task loss for amplitude and timing regression."""

    def __init__(self, time_weight: float = 2.0, delta: float = 0.5):
        super().__init__()
        self.amp_loss = nn.MSELoss()
        self.time_loss = nn.HuberLoss(delta=delta)
        self.time_weight = time_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        amp = self.amp_loss(pred[:, 0], target[:, 0])
        t = self.time_loss(pred[:, 1], target[:, 1])
        return amp + self.time_weight * t
