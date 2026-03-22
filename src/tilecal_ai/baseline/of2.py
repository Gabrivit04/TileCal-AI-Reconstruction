from __future__ import annotations

import numpy as np


def optimal_filter_reconstruction(samples: np.ndarray, weights_amp: np.ndarray, weights_time: np.ndarray):
    """Classical OF-style linear estimator for amplitude and time."""
    if samples.ndim != 2:
        raise ValueError("samples must be [n_events, n_samples]")
    amp = samples @ weights_amp
    time = samples @ weights_time
    return amp, time
