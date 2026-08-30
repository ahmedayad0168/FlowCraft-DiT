from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow.cfm import ConditionalFlowMatcher
from flow.euler import EulerSampler
from utils.torch_utils import default_device, module_dtype


class ReflowPipeline:
    """2-Rectified Flow (reflow) pair generation and student training loss.

    The teacher ODE is integrated from a noise sample x0 that is generated here
    and handed to the sampler, so the returned (x0, x1_pred) really is a
    trajectory pair: reusing independent noise would train the student on
    mismatched endpoints.
    """

    def __init__(self, teacher_model: nn.Module, teacher_steps: int = 50, sigma_min: float = 1e-4):
        self.teacher_model = teacher_model
        self.sampler = EulerSampler(num_steps=teacher_steps)
        self.cfm = ConditionalFlowMatcher(sigma_min=sigma_min)

    @torch.no_grad()
    def generate_reflow_pair(
        self,
        x_shape: Tuple[int, ...],
        text_embeds: torch.Tensor,
        null_text_embeds: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        null_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or getattr(text_embeds, "device", None) or default_device()
        dtype = dtype or module_dtype(self.teacher_model)

        x0 = torch.randn(x_shape, device=device, dtype=dtype, generator=generator)
        x1_pred = self.sampler.sample(
            model=self.teacher_model,
            text_embeds=text_embeds,
            null_text_embeds=null_text_embeds,
            cfg_scale=cfg_scale,
            device=device,
            dtype=dtype,
            x_init=x0,
            model_kwargs=model_kwargs,
            null_model_kwargs=null_model_kwargs,
        )
        return x0, x1_pred

    def compute_reflow_loss(
        self,
        student_model: nn.Module,
        x0: torch.Tensor,
        x1_pred: torch.Tensor,
        text_embeds: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Trains the student toward the straightened velocity target x1_pred - x0."""
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        xt, target_velocity = self.cfm.sample_location_and_target(x0, x1_pred, t)
        pred_velocity = student_model(xt, t, text_embeds, **(model_kwargs or {}))
        return F.mse_loss(pred_velocity.float(), target_velocity.float())
