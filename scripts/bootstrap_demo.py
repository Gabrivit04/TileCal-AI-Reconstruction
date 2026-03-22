#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from tilecal_ai.baseline.of2 import optimal_filter_reconstruction
from tilecal_ai.data.pipeline import pedestal_subtract_and_normalize


def main() -> None:
    rng = np.random.default_rng(7)
    pulses = rng.normal(size=(100, 7)).astype(np.float32)
    pulses = pedestal_subtract_and_normalize(pulses)

    w_a = np.array([0.1, 0.1, 0.15, 0.3, 0.2, 0.1, 0.05], dtype=np.float32)
    w_t = np.array([-0.1, -0.1, -0.05, 0.0, 0.05, 0.1, 0.1], dtype=np.float32)
    amp, time = optimal_filter_reconstruction(pulses, w_a, w_t)

    print(f"events={len(amp)}")
    print(f"mean_amp={amp.mean():.4f}")
    print(f"mean_time={time.mean():.4f}")


if __name__ == "__main__":
    main()
