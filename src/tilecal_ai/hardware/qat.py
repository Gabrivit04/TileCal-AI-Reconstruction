from __future__ import annotations

import torch


def fake_quantize_tensor(x: torch.Tensor, num_bits: int = 8, clip_value: float = 2.0) -> torch.Tensor:
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = max(clip_value / qmax, 1e-8)
    x_clipped = torch.clamp(x, -clip_value, clip_value)
    q = torch.round(x_clipped / scale).clamp(qmin, qmax)
    return q * scale


def quantization_aware_forward(model, x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    xq = fake_quantize_tensor(x, num_bits=num_bits)
    out = model(xq)
    return fake_quantize_tensor(out, num_bits=num_bits)
