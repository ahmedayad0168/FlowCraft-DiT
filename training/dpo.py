from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow.cfm import ConditionalFlowMatcher


class FlowDPOLoss(nn.Module):
    """Direct Preference Optimization for flow matching (Diffusion/Flow-DPO).

    Winner and loser latents share the same noise and timestep, so the implicit
    reward difference only reflects the preference, not sampling variance.
    """

    def __init__(self, beta: float = 2000.0, sigma_min: float = 1e-4):
        super().__init__()
        self.beta = beta
        self.cfm = ConditionalFlowMatcher(sigma_min=sigma_min)

    def forward(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        x_w: torch.Tensor,
        x_l: torch.Tensor,
        text_embeds: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_metrics: bool = False,
    ):
        """
        x_w / x_l: preferred / dispreferred latents [B, C, H, W]
        text_embeds: [B, N_txt, txt_dim] shared prompt conditioning
        """
        if x_w.shape != x_l.shape:
            raise ValueError(
                "Winner/loser latents must have the same shape, got "
                f"{tuple(x_w.shape)} and {tuple(x_l.shape)}."
            )
        model_kwargs = model_kwargs or {}
        B = x_w.shape[0]

        x0 = torch.randn_like(x_w)
        if t is None:
            t = torch.rand(B, device=x_w.device, dtype=x_w.dtype)

        xt_w, target_v_w = self.cfm.sample_location_and_target(x0, x_w, t)
        xt_l, target_v_l = self.cfm.sample_location_and_target(x0, x_l, t)

        v_policy_w = policy_model(xt_w, t, text_embeds, **model_kwargs)
        v_policy_l = policy_model(xt_l, t, text_embeds, **model_kwargs)

        was_training = ref_model.training
        ref_model.eval()
        try:
            with torch.no_grad():
                v_ref_w = ref_model(xt_w, t, text_embeds, **model_kwargs)
                v_ref_l = ref_model(xt_l, t, text_embeds, **model_kwargs)
        finally:
            ref_model.train(was_training)

        def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (pred.float() - target.float()).pow(2).flatten(1).mean(1)

        err_policy_w = per_sample_mse(v_policy_w, target_v_w)
        err_policy_l = per_sample_mse(v_policy_l, target_v_l)
        err_ref_w = per_sample_mse(v_ref_w, target_v_w)
        err_ref_l = per_sample_mse(v_ref_l, target_v_l)

        # Implicit rewards: lower error than the reference model = higher reward.
        reward_w = -self.beta * (err_policy_w - err_ref_w)
        reward_l = -self.beta * (err_policy_l - err_ref_l)
        margin = reward_w - reward_l

        loss = -F.logsigmoid(margin).mean()
        if return_metrics:
            metrics = {
                "loss": loss.detach(),
                "reward_margin": margin.mean().detach(),
                "accuracy": (margin > 0).float().mean().detach(),
                "err_policy_w": err_policy_w.mean().detach(),
                "err_policy_l": err_policy_l.mean().detach(),
            }
            return loss, metrics
        return loss
