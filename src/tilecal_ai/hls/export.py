"""hls4ml integration placeholders and config generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class HLSConfig:
    project_name: str = "tilecal_ai_hls"
    clock_period_ns: float = 5.0
    io_type: str = "io_parallel"
    precision: str = "ap_fixed<8,2>"


def write_hls4ml_config(output: str | Path, cfg: HLSConfig) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ProjectName": cfg.project_name,
        "ClockPeriod": cfg.clock_period_ns,
        "IOType": cfg.io_type,
        "DefaultPrecision": cfg.precision,
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path
