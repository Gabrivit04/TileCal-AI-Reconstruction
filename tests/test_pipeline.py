import numpy as np

from tilecal_ai.data.pipeline import TileCalPulseDataset, pedestal_subtract_and_normalize


def test_pedestal_normalize_shape_and_scale():
    pulses = np.array([[1, 1, 3, 5, 3, 1, 1]], dtype=np.float32)
    out = pedestal_subtract_and_normalize(pulses)
    assert out.shape == pulses.shape
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_dataset_getitem_shapes():
    x = np.random.randn(10, 7).astype(np.float32)
    y = np.random.randn(10, 2).astype(np.float32)
    ds = TileCalPulseDataset(x, y, noise_std=0.01)
    pulse, target = ds[0]
    assert pulse.shape == (7,)
    assert target.shape == (2,)
