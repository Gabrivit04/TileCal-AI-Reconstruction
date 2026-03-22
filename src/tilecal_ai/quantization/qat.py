"""Minimal QAT helper utilities for PyTorch models."""

from __future__ import annotations

import torch
from torch import nn


def prepare_qat(model: nn.Module) -> nn.Module:
    model.train()
    backend = "fbgemm"
    torch.backends.quantized.engine = backend
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    return torch.ao.quantization.prepare_qat(model)


def convert_int8(model: nn.Module) -> nn.Module:
    model.eval()
    return torch.ao.quantization.convert(model)
