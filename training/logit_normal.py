from typing import Optional

import torch
import torch.nn as nn


class LogitNormalSampler(nn.Module):
    """Logit-normal timestep sampler, t = sigmoid(N(mean, std^2)).

    Concentrates training timesteps around mid-trajectory (t ~ 0.5), where
    velocity prediction is hardest, instead of wasting capacity on the
    near-deterministic endpoints.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-5):
        super().__init__()
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}.")
        self.mean = mean
        self.std = std
        self.eps = eps

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        normal_samples = (
            torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
            * self.std
            + self.mean
        )
        t = torch.sigmoid(normal_samples)
        return torch.clamp(t, min=self.eps, max=1.0 - self.eps).to(dtype)

    def forward(self, batch_size: int, **kwargs) -> torch.Tensor:
        return self.sample(batch_size, **kwargs)
