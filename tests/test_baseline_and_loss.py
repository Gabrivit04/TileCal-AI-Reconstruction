import numpy as np
import torch

from tilecal_ai.baseline.of2 import optimal_filter_reconstruction
from tilecal_ai.training.losses import AmplitudeTimeLoss


def test_optimal_filter_dimensions():
    s = np.ones((4, 7), dtype=np.float32)
    wa = np.ones(7, dtype=np.float32)
    wt = np.zeros(7, dtype=np.float32)
    amp, time = optimal_filter_reconstruction(s, wa, wt)
    assert amp.shape == (4,)
    assert time.shape == (4,)


def test_multitask_loss_runs():
    pred = torch.tensor([[1.0, 0.2], [0.5, -0.1]])
    target = torch.tensor([[0.8, 0.3], [0.4, 0.0]])
    loss = AmplitudeTimeLoss()(pred, target)
    assert loss.item() > 0
