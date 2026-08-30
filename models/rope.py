from typing import Dict, Tuple

import torch
import torch.nn as nn


class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for spatial latent patches.

    Splits the head dimension in half: the first half rotates with the height
    coordinate, the second half with the width coordinate. cos/sin grids are
    cached per (h, w, device, dtype).
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be a multiple of 4 for 2D RoPE.")
        self.head_dim = head_dim
        self.theta = theta
        self._cache: Dict[Tuple[int, int, str, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_1d_freqs(self, seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
        half_dim = dim // 2
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
        )
        seq = torch.arange(seq_len, dtype=torch.float32, device=device)
        return torch.outer(seq, freqs)  # [seq_len, dim / 2]

    def forward(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin), each [h * w, head_dim]."""
        key = (h, w, str(device), dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        dim_per_axis = self.head_dim // 2
        angles_h = self._compute_1d_freqs(h, dim_per_axis, device)  # [H, head_dim/4]
        angles_w = self._compute_1d_freqs(w, dim_per_axis, device)  # [W, head_dim/4]

        angles_h = angles_h.unsqueeze(1).expand(h, w, -1)
        angles_w = angles_w.unsqueeze(0).expand(h, w, -1)

        # [H * W, head_dim / 2] -> duplicated to [H * W, head_dim] for rotate_half
        angles_2d = torch.cat([angles_h, angles_w], dim=-1).reshape(h * w, -1)
        angles_full = torch.cat([angles_2d, angles_2d], dim=-1)

        cos = angles_full.cos().to(dtype=dtype)
        sin = angles_full.sin().to(dtype=dtype)

        self._cache[key] = (cos, sin)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_2d(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q, k: [B, Heads, N_img, Head_Dim]; cos, sin: [N_img, Head_Dim]."""
    if cos.shape[-1] != q.shape[-1] or cos.shape[0] != q.shape[-2]:
        raise ValueError(
            f"RoPE grid {tuple(cos.shape)} does not match query shape {tuple(q.shape)}."
        )
    cos = cos.to(q.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(q.dtype).unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
