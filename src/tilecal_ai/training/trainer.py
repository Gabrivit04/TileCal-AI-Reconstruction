from __future__ import annotations

import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device: str = "cpu") -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(features)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(1, n_batches)
