import numpy as np
import torch

from tilecal_ai.baselines.optimal_filtering import default_of2
from tilecal_ai.data.pipeline import PipelineConfig, TileCalPulseDataset
from tilecal_ai.hls.export import HLSConfig, write_hls4ml_config
from tilecal_ai.models.networks import DepthwiseSeparableCNN, MLPRegressor, TinyGRU
from tilecal_ai.training.losses import MultiTaskTileCalLoss


def test_dataset_shapes_and_types() -> None:
    pulses = np.random.randn(16, 7).astype(np.float32)
    amp = np.ones(16, dtype=np.float32)
    tau = np.zeros(16, dtype=np.float32)
    ds = TileCalPulseDataset(pulses, amp, tau, PipelineConfig(window_size=7), train=False)
    batch = ds[0]
    assert batch["samples"].shape == (7,)
    assert batch["samples"].dtype == torch.float32


def test_models_forward_shape() -> None:
    x = torch.randn(8, 7)
    for model in [MLPRegressor(7), DepthwiseSeparableCNN(7), TinyGRU()]:
        y = model(x)
        assert y.shape == (8, 2)


def test_multitask_loss_backward() -> None:
    pred = torch.randn(4, 2, requires_grad=True)
    amp = torch.randn(4)
    tau = torch.randn(4)
    loss = MultiTaskTileCalLoss()(pred, amp, tau)
    loss.backward()
    assert pred.grad is not None


def test_optimal_filtering() -> None:
    of2 = default_of2(window_size=7)
    samples = np.random.randn(10, 7).astype(np.float32)
    amp, tau = of2.predict(samples)
    assert amp.shape == (10,)
    assert tau.shape == (10,)


def test_hls_config_export(tmp_path) -> None:
    out = write_hls4ml_config(tmp_path / "hls4ml.yaml", HLSConfig())
    assert out.exists()
    assert "ProjectName" in out.read_text(encoding="utf-8")
