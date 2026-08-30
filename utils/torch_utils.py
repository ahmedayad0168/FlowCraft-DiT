"""Small shared helpers: device/dtype resolution, seeding, and weight EMA."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn


def default_device() -> torch.device:
    """CUDA when available, then Apple MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    """Maps a precision name to a dtype, falling back to fp32 where unsupported.

    fp16 is deliberately never returned for autocast-free math on CPU, and bf16
    is downgraded to fp32 on devices without bf16 support.
    """
    precision = precision.lower()
    if precision in ("fp32", "float32", "32"):
        return torch.float32
    if precision in ("bf16", "bfloat16"):
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cpu":
            return torch.bfloat16
        return torch.float32
    if precision in ("fp16", "float16", "16"):
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    raise ValueError(f"Unknown precision '{precision}' (expected fp32, bf16 or fp16).")


def module_dtype(module: nn.Module) -> torch.dtype:
    """dtype of the first parameter/buffer of a module (fp32 if it has none)."""
    for tensor in module.parameters():
        return tensor.dtype
    for tensor in module.buffers():
        return tensor.dtype
    return torch.float32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of model weights, kept in float32 on CPU/GPU."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.shadow.parameters(), model.parameters()):
            ema_param.lerp_(param.detach(), 1.0 - self.decay)
        for ema_buf, buf in zip(self.shadow.buffers(), model.buffers()):
            ema_buf.copy_(buf)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.shadow.parameters()

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "EMA":
        self.shadow.to(device=device, dtype=dtype)
        return self
