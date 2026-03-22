from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedPointSpec:
    total_bits: int
    int_bits: int

    @property
    def hls_type(self) -> str:
        return f"ap_fixed<{self.total_bits}, {self.int_bits}>"


def recommended_precision(profile: str = "balanced") -> FixedPointSpec:
    options = {
        "max-accuracy": FixedPointSpec(16, 6),
        "balanced": FixedPointSpec(8, 3),
        "ultra-low-latency": FixedPointSpec(6, 2),
    }
    return options[profile]
