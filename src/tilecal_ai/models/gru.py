from __future__ import annotations

import torch
from torch import nn


class TileCalGRU(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))
