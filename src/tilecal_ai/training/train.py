"""Reference training loop for quick local experiments and CI sanity checks."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from tilecal_ai.data.pipeline import make_dummy_dataset
from tilecal_ai.models.networks import DepthwiseSeparableCNN
from tilecal_ai.training.losses import MultiTaskTileCalLoss


def run_sanity_training(epochs: int = 1, batch_size: int = 64) -> float:
    dataset = make_dummy_dataset(n_events=512, window_size=7)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DepthwiseSeparableCNN(input_size=7, channels=8)
    loss_fn = MultiTaskTileCalLoss(amp_weight=1.0, time_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    epoch_loss = 0.0
    for _ in range(epochs):
        for batch in loader:
            pred = model(batch["samples"])
            loss = loss_fn(pred, batch["amplitude"], batch["timing"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
    return epoch_loss / max(len(loader) * epochs, 1)


if __name__ == "__main__":
    avg_loss = run_sanity_training(epochs=1)
    print(f"sanity training loss: {avg_loss:.4f}")
