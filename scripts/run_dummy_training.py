from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tilecal_ai.models.cnn1d import TileCalCNN1D
from tilecal_ai.training.losses import AmplitudeTimingLoss
from tilecal_ai.training.trainer import train_one_epoch


def main() -> None:
    rng = np.random.default_rng(42)
    x = torch.tensor(rng.normal(size=(128, 7)), dtype=torch.float32)
    y = torch.tensor(rng.normal(size=(128, 2)), dtype=torch.float32)

    dl = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model = TileCalCNN1D(input_len=7)
    criterion = AmplitudeTimingLoss(timing_weight=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(model, dl, criterion, optimizer)
    print(f"dummy_epoch_loss={loss:.6f}")


if __name__ == "__main__":
    main()
