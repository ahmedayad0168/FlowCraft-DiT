import math
from typing import Tuple

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Maps continuous timesteps t in [0, 1] to a conditioning vector."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        if frequency_dim % 2 != 0:
            raise ValueError(f"frequency_dim must be even, got {frequency_dim}.")
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """Sinusoidal timestep encodings, always computed in float32 for stability."""
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )
        args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] (scalar timesteps). Returns [B, hidden_dim].
        t_freq = self.sinusoidal_embedding(t, self.frequency_dim)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class AdaLNZero(nn.Module):
    """AdaLN-Zero conditioning: LayerNorm without affine + zero-init modulation."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, 6 * dim)
        self.zero_init()

    def zero_init(self) -> None:
        """Re-applies the zero initialization (call after any global init sweep)."""
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """cond: [B, D] global conditioning. Returns 6 tensors of shape [B, 1, D]."""
        return self.linear(cond).unsqueeze(1).chunk(6, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies x * (1 + scale) + shift."""
    return x * (1.0 + scale) + shift
