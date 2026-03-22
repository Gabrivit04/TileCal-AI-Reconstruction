from __future__ import annotations

import torch
from torch import nn


class AmplitudeTimingLoss(nn.Module):
    """Multi-task loss with Huber timing term for pile-up robustness."""

    def __init__(self, timing_weight: float = 0.5, huber_delta: float = 1.0) -> None:
        super().__init__()
        self.timing_weight = timing_weight
        self.amp_loss = nn.MSELoss()
        self.time_loss = nn.HuberLoss(delta=huber_delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        amp = self.amp_loss(pred[:, 0], target[:, 0])
        tim = self.time_loss(pred[:, 1], target[:, 1])
        return amp + self.timing_weight * tim
