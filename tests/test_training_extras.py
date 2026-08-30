"""CPU tests for DPO, LADD and REPA (no network downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distillation.ladd import LADDDistiller, LatentDiscriminator
from models.mm_dit import FlowCraftMMDiT, MMDiTConfig
from training.dpo import FlowDPOLoss
from training.repa import REPALoss

TXT_DIM = 32


def tiny_model() -> FlowCraftMMDiT:
    return FlowCraftMMDiT(
        MMDiTConfig(in_channels=4, patch_size=2, hidden_dim=32, num_heads=4, depth=2, txt_dim=TXT_DIM)
    )


# --------------------------------------------------------------------------- DPO


def test_dpo_loss_is_finite_and_reports_metrics():
    policy, reference = tiny_model(), tiny_model()
    x_w, x_l = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    text = torch.randn(2, 5, TXT_DIM)

    loss, metrics = FlowDPOLoss(beta=10.0)(
        policy, reference, x_w, x_l, text, return_metrics=True
    )
    assert torch.isfinite(loss)
    assert set(metrics) >= {"loss", "reward_margin", "accuracy"}
    loss.backward()
    assert policy.img_in.weight.grad is not None
    assert reference.img_in.weight.grad is None  # reference stays frozen


def test_dpo_restores_reference_training_mode_and_validates_shapes():
    policy, reference = tiny_model(), tiny_model().train()
    text = torch.randn(2, 5, TXT_DIM)
    FlowDPOLoss()(policy, reference, torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8), text)
    assert reference.training
    with pytest.raises(ValueError):
        FlowDPOLoss()(policy, reference, torch.randn(2, 4, 8, 8), torch.randn(2, 4, 4, 4), text)


# -------------------------------------------------------------------------- LADD


def test_ladd_generator_loss_reaches_the_student():
    student = tiny_model().train()
    discriminator = LatentDiscriminator(in_channels=4, hidden_dim=16)
    text = torch.randn(2, 5, TXT_DIM)

    g_loss, fake = LADDDistiller(student_steps=2).compute_generator_loss(
        student_model=student,
        discriminator=discriminator,
        x_shape=(2, 4, 8, 8),
        text_embeds=text,
        real_latents=torch.randn(2, 4, 8, 8),
    )
    assert torch.isfinite(g_loss) and not fake.requires_grad
    g_loss.backward()
    # The differentiable sampler is what lets the adversarial signal get here.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_ladd_discriminator_loss_ignores_generator_graph():
    student = tiny_model().train()
    discriminator = LatentDiscriminator(in_channels=4, hidden_dim=16)
    distiller = LADDDistiller(student_steps=1)
    fake = distiller.generate(
        student, (2, 4, 8, 8), torch.randn(2, 5, TXT_DIM), torch.device("cpu"), torch.float32
    )
    d_loss = distiller.compute_discriminator_loss(discriminator, torch.randn(2, 4, 8, 8), fake)
    d_loss.backward()
    assert all(p.grad is None for p in student.parameters())


def test_ladd_feature_matching_requires_paired_latents():
    distiller = LADDDistiller(student_steps=1, feature_matching_weight=1.0)
    with pytest.raises(ValueError):
        distiller.compute_generator_loss(
            student_model=tiny_model(),
            discriminator=LatentDiscriminator(in_channels=4, hidden_dim=16),
            x_shape=(2, 4, 8, 8),
            text_embeds=torch.randn(2, 5, TXT_DIM),
        )


def test_latent_discriminator_group_count_is_robust():
    # hidden_dim smaller than the default 32 groups used to crash GroupNorm.
    assert LatentDiscriminator(in_channels=4, hidden_dim=12)(torch.randn(1, 4, 8, 8)).ndim == 4


# -------------------------------------------------------------------------- REPA


class DummyDINO(nn.Module):
    """Stands in for DINOv2: 16x16 patch tokens of width `embed_dim`."""

    def __init__(self, embed_dim: int = 24, grid: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid = grid
        self.proj = nn.Linear(3, embed_dim)

    def forward_features(self, x: torch.Tensor) -> dict:
        pooled = nn.functional.adaptive_avg_pool2d(x, (self.grid, self.grid))
        tokens = self.proj(pooled.flatten(2).transpose(1, 2))
        return {"x_norm_patchtokens": tokens}


def test_repa_resamples_mismatched_token_grids():
    repa = REPALoss(dit_dim=32, dino_module=DummyDINO(grid=16))
    dit_features = torch.randn(2, 64, 32)          # 8x8 DiT grid vs 16x16 DINO grid
    images = torch.rand(2, 3, 64, 64) * 2 - 1
    loss = repa(dit_features, images)
    assert torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert repa.projector[0].weight.grad is not None


def test_repa_keeps_dino_frozen_and_in_eval():
    repa = REPALoss(dit_dim=32, dino_module=DummyDINO()).train()
    assert not repa.dinov2.training
    assert all(not p.requires_grad for p in repa.dinov2.parameters())


def test_repa_validates_input_range():
    with pytest.raises(ValueError):
        REPALoss(dit_dim=32, dino_module=DummyDINO(), input_range="whatever")
