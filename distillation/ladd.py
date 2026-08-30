from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow.euler import EulerSampler


class LatentDiscriminator(nn.Module):
    """Latent-space discriminator for LADD."""

    def __init__(self, in_channels: int = 16, hidden_dim: int = 512, num_groups: int = 32):
        super().__init__()
        groups = min(num_groups, hidden_dim)
        while hidden_dim % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LADDDistiller(nn.Module):
    """Latent Adversarial Diffusion Distillation: 50-step teacher -> 1-4 step student.

    `feature_matching_weight` defaults to 0 because the L1 term only makes sense
    when `real_latents` is the teacher's output for the *same* prompt and noise
    (i.e. a paired reflow batch); on an unpaired real batch it pulls every
    sample toward the batch mean.
    """

    def __init__(self, student_steps: int = 4, feature_matching_weight: float = 0.0):
        super().__init__()
        self.student_steps = student_steps
        self.feature_matching_weight = feature_matching_weight
        self.sampler = EulerSampler(num_steps=student_steps)

    def generate(
        self,
        student_model: nn.Module,
        x_shape: Tuple[int, ...],
        text_embeds: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        x_init: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return self.sampler.sample(
            model=student_model,
            x_shape=x_shape,
            text_embeds=text_embeds,
            cfg_scale=1.0,
            device=device,
            dtype=dtype,
            x_init=x_init,
            model_kwargs=model_kwargs,
            differentiable=True,  # gradients must reach the student generator
        )

    def compute_generator_loss(
        self,
        student_model: nn.Module,
        discriminator: LatentDiscriminator,
        x_shape: Tuple[int, ...],
        text_embeds: torch.Tensor,
        real_latents: Optional[torch.Tensor] = None,
        x_init: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reference = real_latents if real_latents is not None else text_embeds
        fake_latents = self.generate(
            student_model=student_model,
            x_shape=x_shape,
            text_embeds=text_embeds,
            device=reference.device,
            dtype=reference.dtype,
            x_init=x_init,
            model_kwargs=model_kwargs,
        )

        g_loss = -discriminator(fake_latents).mean()
        if self.feature_matching_weight > 0.0:
            if real_latents is None:
                raise ValueError("feature_matching_weight > 0 requires paired real_latents.")
            g_loss = g_loss + self.feature_matching_weight * F.l1_loss(fake_latents, real_latents)

        return g_loss, fake_latents.detach()

    def compute_discriminator_loss(
        self,
        discriminator: LatentDiscriminator,
        real_latents: torch.Tensor,
        fake_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge loss on real vs. student-generated latents."""
        real_logits = discriminator(real_latents)
        fake_logits = discriminator(fake_latents.detach())

        loss_real = F.relu(1.0 - real_logits).mean()
        loss_fake = F.relu(1.0 + fake_logits).mean()
        return (loss_real + loss_fake) * 0.5
