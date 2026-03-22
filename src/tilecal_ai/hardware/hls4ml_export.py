from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HLSConfig:
    backend: str = "Vitis"
    clock_period_ns: float = 5.0
    precision: str = "ap_fixed<8,2>"
    reuse_factor: int = 1


def build_hls4ml_config(model_name: str, input_len: int, cfg: HLSConfig | None = None) -> dict:
    config = cfg or HLSConfig()
    return {
        "Model": {
            "Name": model_name,
            "InputLength": input_len,
            "Backend": config.backend,
            "ClockPeriod": config.clock_period_ns,
            "Precision": config.precision,
            "ReuseFactor": config.reuse_factor,
        },
        "Metrics": ["latency_cycles", "dsp", "lut", "ff", "bram"],
    }
