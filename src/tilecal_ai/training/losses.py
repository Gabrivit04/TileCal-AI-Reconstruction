"""Loss functions for joint amplitude/timing regression."""

from __future__ import annotations

import torch
from torch import nn


class MultiTaskTileCalLoss(nn.Module):
    def __init__(self, amp_weight: float = 1.0, time_weight: float = 1.0, use_huber: bool = True) -> None:
        super().__init__()
        self.amp_weight = amp_weight
        self.time_weight = time_weight
        self.amp_loss = nn.HuberLoss(delta=0.5) if use_huber else nn.MSELoss()
        self.time_loss = nn.HuberLoss(delta=0.2) if use_huber else nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_amp: torch.Tensor, target_time: torch.Tensor) -> torch.Tensor:
        amp_pred, time_pred = pred[:, 0], pred[:, 1]
        l_amp = self.amp_loss(amp_pred, target_amp)
        l_time = self.time_loss(time_pred, target_time)
        return self.amp_weight * l_amp + self.time_weight * l_time
