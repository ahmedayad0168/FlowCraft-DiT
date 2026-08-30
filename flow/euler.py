from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utils.torch_utils import default_device, module_dtype


class EulerSampler:
    """First-order ODE solver for flow matching, with classifier-free guidance.

    Integrates t: 0 -> 1, i.e. noise -> clean latent.
    """

    def __init__(self, num_steps: int = 50):
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
        self.num_steps = num_steps

    def sample(
        self,
        model: nn.Module,
        x_shape: Optional[Tuple[int, ...]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        null_text_embeds: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        x_init: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        differentiable: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        null_model_kwargs: Optional[Dict[str, Any]] = None,
        return_trajectory: bool = False,
    ):
        """Integrates the learned velocity field from pure noise to a clean latent.

        x_init: optional starting noise. Pass it (instead of letting the sampler
            draw its own) whenever the caller needs the exact (x0, x1) pair, as
            reflow does.
        differentiable: keeps the autograd graph through every ODE step, which
            LADD needs so the adversarial loss reaches the student generator.
        """
        if text_embeds is None:
            raise ValueError("text_embeds is required.")
        if x_init is None and x_shape is None:
            raise ValueError("Provide either x_shape or x_init.")

        device = device or (x_init.device if x_init is not None else default_device())
        dtype = dtype or (x_init.dtype if x_init is not None else module_dtype(model))

        was_training = model.training
        if not differentiable:
            model.eval()
        grad_ctx = nullcontext() if differentiable else torch.no_grad()
        try:
            with grad_ctx:
                out = self._integrate(
                    model=model,
                    x_shape=x_shape,
                    text_embeds=text_embeds,
                    null_text_embeds=null_text_embeds,
                    cfg_scale=cfg_scale,
                    device=device,
                    dtype=dtype,
                    x_init=x_init,
                    generator=generator,
                    model_kwargs=model_kwargs or {},
                    null_model_kwargs=null_model_kwargs,
                    return_trajectory=return_trajectory,
                )
        finally:
            model.train(was_training)
        return out

    @staticmethod
    def _cat_kwargs(
        model_kwargs: Dict[str, Any], null_model_kwargs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Concatenates conditional/unconditional extra tensors for the CFG batch."""
        null_model_kwargs = null_model_kwargs if null_model_kwargs is not None else model_kwargs
        merged: Dict[str, Any] = {}
        for key, value in model_kwargs.items():
            null_value = null_model_kwargs.get(key, value)
            if torch.is_tensor(value) and torch.is_tensor(null_value):
                merged[key] = torch.cat([value, null_value], dim=0)
            else:
                merged[key] = value
        return merged

    def _integrate(
        self,
        model: nn.Module,
        x_shape: Optional[Tuple[int, ...]],
        text_embeds: torch.Tensor,
        null_text_embeds: Optional[torch.Tensor],
        cfg_scale: float,
        device: torch.device,
        dtype: torch.dtype,
        x_init: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        model_kwargs: Dict[str, Any],
        null_model_kwargs: Optional[Dict[str, Any]],
        return_trajectory: bool,
    ):
        if x_init is not None:
            x = x_init.to(device=device, dtype=dtype)
        else:
            x = torch.randn(x_shape, device=device, dtype=dtype, generator=generator)

        use_cfg = cfg_scale != 1.0 and null_text_embeds is not None
        if use_cfg:
            text_in_full = torch.cat([text_embeds, null_text_embeds], dim=0)
            cfg_kwargs = self._cat_kwargs(model_kwargs, null_model_kwargs)

        dt = 1.0 / self.num_steps
        timesteps = torch.linspace(0.0, 1.0 - dt, self.num_steps, device=device, dtype=torch.float32)
        trajectory: List[torch.Tensor] = [x]

        for t_val in timesteps:
            t = torch.full((x.shape[0],), float(t_val), device=device, dtype=dtype)

            if use_cfg:
                v_pred = model(
                    torch.cat([x, x], dim=0),
                    torch.cat([t, t], dim=0),
                    text_in_full,
                    **cfg_kwargs,
                )
                v_cond, v_uncond = v_pred.chunk(2, dim=0)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t, text_embeds, **model_kwargs)

            x = x + dt * v
            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return x, trajectory
        return x
