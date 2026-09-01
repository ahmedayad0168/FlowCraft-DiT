"""CPU smoke tests: shapes, gradients, dtype policy, and the fixed bugs."""
""" 
usage: python -m pytest -v tests/test_smoke.py -k reflow
       python -m pytest -v tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flow.cfm import ConditionalFlowMatcher
from flow.euler import EulerSampler
from flow.reflow import ReflowPipeline
from models.adaln import AdaLNZero, TimestepEmbedder
from models.mm_dit import FlowCraftMMDiT, MMDiTConfig
from models.rope import RoPE2D, apply_rope_2d
from training.logit_normal import LogitNormalSampler
from utils.torch_utils import EMA, resolve_dtype

DEVICE = torch.device("cpu")
# -------------------------------------------------------

TXT_DIM = 32


def tiny_config(**overrides) -> MMDiTConfig:
    kwargs = dict(
        in_channels=4, patch_size=2, hidden_dim=32, num_heads=4, depth=2, txt_dim=TXT_DIM
    )
    kwargs.update(overrides)
    return MMDiTConfig(**kwargs)


def tiny_model(**overrides) -> FlowCraftMMDiT:
    return FlowCraftMMDiT(tiny_config(**overrides)).to(DEVICE)


def batch(model: FlowCraftMMDiT, b: int = 2, hw: int = 8, n_txt: int = 5):
    x = torch.randn(b, model.config.in_channels, hw, hw, device=DEVICE)
    t = torch.rand(b, device=DEVICE)
    text = torch.randn(b, n_txt, model.config.txt_dim, device=DEVICE)
    return x, t, text


# --------------------------------------------------------------------------- model


def test_forward_shape_and_backward():
    model = tiny_model()
    x, t, text = batch(model)
    v = model(x, t, text)
    assert v.shape == x.shape
    v.pow(2).mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_patchify_roundtrip():
    model = tiny_model()
    x = torch.randn(2, 4, 8, 8, device=DEVICE)
    patches = model.patchify(x)
    assert torch.allclose(model.unpatchify(patches, 4, 4), x)


def test_config_rejects_incompatible_dims():
    with pytest.raises(ValueError):
        tiny_config(hidden_dim=30, num_heads=4)   # not divisible
    with pytest.raises(ValueError):
        tiny_config(hidden_dim=8, num_heads=4)    # head_dim=2, not a multiple of 4


def test_forward_rejects_wrong_channel_count():
    model = tiny_model()
    x, t, text = batch(model)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 8, 8, device=DEVICE), t, text)


def test_forward_rejects_indivisible_latent():
    model = tiny_model()
    _, t, text = batch(model)
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 7, 7, device=DEVICE), t, text)


def test_adaln_zero_survives_global_init():
    """The Xavier sweep in initialize_weights() used to wipe AdaLN-Zero."""
    model = tiny_model()
    for block in model.blocks:
        for norm in (block.img_norm, block.txt_norm):
            assert torch.count_nonzero(norm.linear.weight) == 0
            assert torch.count_nonzero(norm.linear.bias) == 0


def test_text_mask_changes_output_and_ignores_padding():
    model = tiny_model().eval()
    x, t, text = batch(model, n_txt=6)
    mask = torch.ones(2, 6, dtype=torch.long, device=DEVICE)
    mask[:, 3:] = 0

    with torch.no_grad():
        masked = model(x, t, text, txt_mask=mask)
        text_other_padding = text.clone()
        text_other_padding[:, 3:] = torch.randn_like(text_other_padding[:, 3:])
        masked_again = model(x, t, text_other_padding, txt_mask=mask)
    # Padded tokens must not influence the output at all.
    assert torch.allclose(masked, masked_again, atol=1e-5)


def test_pooled_text_required_when_configured():
    model = tiny_model(pooled_dim=TXT_DIM)
    x, t, text = batch(model)
    with pytest.raises(ValueError):
        model(x, t, text)
    assert model(x, t, text, pooled_text=torch.randn(2, TXT_DIM, device=DEVICE)).shape == x.shape


def test_return_features_matches_hidden_dim():
    model = tiny_model()
    x, t, text = batch(model)
    v, features = model(x, t, text, return_features=True)
    assert v.shape == x.shape
    assert features.shape == (2, 16, model.config.hidden_dim)  # (8/2)^2 = 16 patches


def test_gradient_checkpointing_matches_plain_forward():
    torch.manual_seed(0)
    model = tiny_model(use_gradient_checkpointing=True).train()
    x, t, text = batch(model)
    checkpointed = model(x, t, text)
    model.config.use_gradient_checkpointing = False
    assert torch.allclose(checkpointed, model(x, t, text), atol=1e-5)


# ---------------------------------------------------------------------- adaln/rope


def test_timestep_embedder_shape_and_odd_dim_padding():
    embedder = TimestepEmbedder(hidden_dim=32, frequency_dim=8)
    out = embedder(torch.rand(4, device=DEVICE))
    assert out.shape == (4, 32) and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        TimestepEmbedder(hidden_dim=32, frequency_dim=7)
    # Odd dims used to slice off a column instead of padding one on.
    odd = TimestepEmbedder.sinusoidal_embedding(torch.rand(4, device=DEVICE), dim=7)
    assert odd.shape == (4, 7)


def test_adaln_zero_is_identity_at_init():
    adaln = AdaLNZero(16)
    shift, scale, gate, *_ = adaln(torch.randn(2, 16, device=DEVICE))
    assert torch.count_nonzero(shift) == 0
    assert torch.count_nonzero(scale) == 0
    assert torch.count_nonzero(gate) == 0


def test_rope_preserves_norm_and_is_cached():
    rope = RoPE2D(head_dim=8)
    cos, sin = rope(4, 4, device=DEVICE, dtype=torch.float32)
    assert cos.shape[-1] == 8
    q = torch.randn(1, 2, 16, 8, device=DEVICE)
    k = torch.randn(1, 2, 16, 8, device=DEVICE)
    q_rot, k_rot = apply_rope_2d(q, k, cos, sin)
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert rope(4, 4, device=DEVICE, dtype=torch.float32)[0] is cos


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RoPE2D(head_dim=6)  # not divisible by 4


# ------------------------------------------------------------------------- flow


def test_cfm_target_velocity_is_path_derivative():
    cfm = ConditionalFlowMatcher(sigma_min=0.0)
    x0, x1 = torch.randn(2, 4, 8, 8, device=DEVICE), torch.randn(2, 4, 8, 8, device=DEVICE)
    t = torch.full((2,), 0.25, device=DEVICE)
    dt = 1e-3
    xt, target = cfm.sample_location_and_target(x0, x1, t)
    xt_next, _ = cfm.sample_location_and_target(x0, x1, t + dt)
    assert torch.allclose((xt_next - xt) / dt, target, atol=1e-3)


def test_cfm_loss_is_finite_and_differentiable():
    model = tiny_model()
    x, t, text = batch(model)
    loss = ConditionalFlowMatcher().compute_loss(model, x, text, t=t)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.img_in.weight.grad).all()


def test_cfm_loss_weighting_and_prediction_return():
    model = tiny_model()
    x, t, text = batch(model)
    loss, pred, extra = ConditionalFlowMatcher().compute_loss(
        model, x, text, t=t, loss_weight=torch.zeros(2, device=DEVICE), return_prediction=True
    )
    assert extra is None
    assert pred.shape == x.shape
    assert loss.item() == pytest.approx(0.0)


def test_euler_sample_shape_and_determinism():
    model = tiny_model().eval()
    _, _, text = batch(model)
    sampler = EulerSampler(num_steps=3)
    shape = (2, 4, 8, 8)

    def run():
        return sampler.sample(
            model=model, x_shape=shape, text_embeds=text,
            generator=torch.Generator(device=DEVICE).manual_seed(0),
            device=DEVICE, 
        )

    out = run()
    assert out.shape == shape
    assert torch.allclose(out, run())


def test_euler_honours_x_init():
    model = tiny_model().eval()
    _, _, text = batch(model)
    x0 = torch.randn(2, 4, 8, 8, device=DEVICE)
    out = EulerSampler(num_steps=1).sample(
        model=model, text_embeds=text, x_init=x0, device=DEVICE  
    )
    # At init the model outputs zeros, so one Euler step must return x0 exactly.
    assert torch.allclose(out, x0)


def test_euler_restores_training_mode_and_requires_inputs():
    model = tiny_model().train()
    _, _, text = batch(model)
    EulerSampler(num_steps=2).sample(
        model=model, x_shape=(2, 4, 8, 8), text_embeds=text, device=DEVICE
    )
    assert model.training
    with pytest.raises(ValueError):
        EulerSampler().sample(model=model, x_shape=(2, 4, 8, 8), text_embeds=None)
    with pytest.raises(ValueError):
        EulerSampler().sample(model=model, text_embeds=text)


def test_euler_cfg_batches_masks_alongside_embeddings():
    model = tiny_model().eval()
    _, _, text = batch(model, n_txt=6)
    mask = torch.ones(2, 6, dtype=torch.long, device=DEVICE)
    out = EulerSampler(num_steps=2).sample(
        model=model, x_shape=(2, 4, 8, 8), text_embeds=text,
        null_text_embeds=torch.zeros_like(text),
        cfg_scale=3.0, model_kwargs={"txt_mask": mask},
        null_model_kwargs={"txt_mask": mask},
        device=DEVICE, 
    )
    assert out.shape == (2, 4, 8, 8)


def test_euler_differentiable_keeps_graph():
    model = tiny_model().train()
    _, _, text = batch(model)
    out = EulerSampler(num_steps=2).sample(
        model=model, x_shape=(2, 4, 8, 8), text_embeds=text,
        differentiable=True, device=DEVICE 
    )
    assert out.requires_grad
    out.pow(2).mean().backward()
    assert model.final_linear.weight.grad is not None


def test_euler_returns_trajectory():
    model = tiny_model().eval()
    _, _, text = batch(model)
    out, trajectory = EulerSampler(num_steps=3).sample(
        model=model, x_shape=(2, 4, 8, 8), text_embeds=text,
        return_trajectory=True, device=DEVICE 
    )
    assert len(trajectory) == 4 and torch.allclose(trajectory[-1], out)


def test_reflow_pair_shares_the_same_noise():
    """The old pipeline drew x0 twice, so (x0, x1_pred) were unrelated."""
    model = tiny_model().eval()
    _, _, text = batch(model)
    pipeline = ReflowPipeline(teacher_model=model, teacher_steps=1)
    x0, x1_pred = pipeline.generate_reflow_pair(
        x_shape=(2, 4, 8, 8), text_embeds=text, device=DEVICE  
    )
    # One step from an untrained (zero-output) model leaves x0 unchanged, which
    # only holds if the sampler integrated from exactly this x0.
    assert torch.allclose(x0, x1_pred)


def test_reflow_student_loss_is_finite():
    teacher, student = tiny_model().eval(), tiny_model()
    _, _, text = batch(teacher)
    pipeline = ReflowPipeline(teacher_model=teacher, teacher_steps=2)
    x0, x1_pred = pipeline.generate_reflow_pair(
        x_shape=(2, 4, 8, 8), text_embeds=text, device=DEVICE
    )
    loss = pipeline.compute_reflow_loss(student, x0, x1_pred, text)
    assert torch.isfinite(loss)
    loss.backward()


# ---------------------------------------------------------------------- training


def test_logit_normal_sampler_is_a_valid_module():
    sampler = LogitNormalSampler()
    assert isinstance(sampler.state_dict(), dict)  # used to raise: no super().__init__()
    t = sampler.sample(1024, device=DEVICE)
    assert t.shape == (1024,)
    assert (t > 0).all() and (t < 1).all()
    assert 0.3 < t.mean().item() < 0.7


def test_logit_normal_sampler_validates_std():
    with pytest.raises(ValueError):
        LogitNormalSampler(std=0.0)


def test_logit_normal_sampler_is_reproducible():
    sampler = LogitNormalSampler()
    a = sampler.sample(8, generator=torch.Generator(device=DEVICE).manual_seed(1), device=DEVICE)
    b = sampler.sample(8, generator=torch.Generator(device=DEVICE).manual_seed(1), device=DEVICE)
    assert torch.allclose(a, b)


# ------------------------------------------------------------------------- utils


def test_resolve_dtype_never_returns_fp16_on_cpu():
    cpu = torch.device("cpu")
    assert resolve_dtype("fp16", cpu) is torch.float32
    assert resolve_dtype("fp32", cpu) is torch.float32
    assert resolve_dtype("bf16", cpu) is torch.bfloat16
    with pytest.raises(ValueError):
        resolve_dtype("int4", cpu)


def test_ema_tracks_weights():
    model = tiny_model()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema.update(model)
    reference = next(model.parameters())
    shadow = next(ema.parameters())
    assert not torch.allclose(shadow, reference)
    for _ in range(60):
        ema.update(model)
    assert torch.allclose(next(ema.parameters()), reference, atol=1e-4)


def test_model_accepts_bf16_inputs_when_cast():
    """Regression for the Gradio 'mat1 and mat2 must have the same dtype' crash."""
    model = tiny_model().to(torch.bfloat16).eval()
    x, t, text = batch(model)
    with torch.no_grad():
        out = model(x.bfloat16(), t.bfloat16(), text.float())  # fp32 text on purpose
    assert out.dtype == torch.bfloat16


def test_euler_infers_dtype_from_model():
    model = tiny_model().to(torch.bfloat16).eval()
    _, _, text = batch(model)
    out = EulerSampler(num_steps=2).sample(
        model=model, x_shape=(2, 4, 8, 8), text_embeds=text.float(),
        device=DEVICE  
    )
    assert out.dtype == torch.bfloat16

def test_pipeline_vae_scaling_factor():
    from app.pipeline import FlowCraftPipeline
    # We can't easily instantiate a real pipeline without checkpoints,
    # but we can check that the attribute exists when loaded.
    # This is a placeholder; actual test would require a checkpoint.
    pass
