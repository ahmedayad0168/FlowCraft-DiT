from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalFlowMatcher:
    """Rectified-flow conditional flow matching with standard linear interpolation.

    Path: x_t = (1 - t) * x0 + t * x1, t in [0,1]
    Target velocity: u_t = x1 - x0
    """

    def __init__(self, sigma_min: float = 0.0):
        # sigma_min kept for API compatibility but effectively unused.
        self.sigma_min = sigma_min

    def sample_location_and_target(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_t, target_velocity)."""
        if t.ndim != 1 or t.shape[0] != x1.shape[0]:
            raise ValueError(f"t must have shape [{x1.shape[0]}], got {tuple(t.shape)}.")
        # Broadcast t to [B,1,1,1]
        t_expand = t.to(x1.dtype).view(-1, *([1] * (x1.ndim - 1)))
        xt = (1.0 - t_expand) * x0 + t_expand * x1
        target_velocity = x1 - x0
        return xt, target_velocity

    def compute_loss(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        text_embeds: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_prediction: bool = False,
    ):
        B = x1.shape[0]
        if x0 is None:
            x0 = torch.randn_like(x1)
        if t is None:
            t = torch.rand(B, device=x1.device, dtype=x1.dtype)

        xt, target_velocity = self.sample_location_and_target(x0, x1, t)

        model_kwargs = model_kwargs or {}
        pred_velocity = model(xt, t, text_embeds, **model_kwargs)
        if isinstance(pred_velocity, tuple):
            pred_velocity, extra = pred_velocity
        else:
            extra = None

        # Compute loss in float32 for numerical stability
        pred_vel_f = pred_velocity.float()
        target_f = target_velocity.float()
        if loss_weight is not None:
            per_sample = (pred_vel_f - target_f).pow(2).flatten(1).mean(1)
            loss = (per_sample * loss_weight.float().to(per_sample.device)).mean()
        else:
            loss = F.mse_loss(pred_vel_f, target_f)

        if return_prediction:
            return loss, pred_velocity, extra
        return loss