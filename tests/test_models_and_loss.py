from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from tilecal_ai.hardware.qat import fake_quantize_tensor
from tilecal_ai.models.cnn1d import TileCalCNN1D
from tilecal_ai.models.gru import TileCalGRU
from tilecal_ai.models.mlp import TileCalMLP
from tilecal_ai.training.losses import AmplitudeTimingLoss


def test_models_output_two_targets() -> None:
    x = torch.randn(4, 7)
    assert TileCalMLP()(x).shape == (4, 2)
    assert TileCalCNN1D(input_len=7)(x).shape == (4, 2)
    assert TileCalGRU()(x).shape == (4, 2)


def test_multitask_loss_and_qat() -> None:
    criterion = AmplitudeTimingLoss(timing_weight=0.7)
    pred = torch.randn(8, 2)
    target = torch.randn(8, 2)
    loss = criterion(pred, target)
    assert loss.item() > 0

    quant = fake_quantize_tensor(pred, num_bits=8)
    assert quant.shape == pred.shape
