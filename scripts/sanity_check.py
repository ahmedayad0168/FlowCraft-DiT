"""Sanity check for the core FlowCraft-DiT pipeline.

Run this script after installation to verify:
- VAE round-trip (encode + decode)
- CLIP text encoding shapes
- Flow matching path consistency
- Model forward/backward
- Euler sampler mechanics

Usage:
    python scripts/sanity_check.py --image_path path/to/test.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flow.cfm import ConditionalFlowMatcher
from flow.euler import EulerSampler
from models.mm_dit import FlowCraftMMDiT, MMDiTConfig
from utils.torch_utils import default_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to a test image.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else default_device()
    print(f"Using device: {device}")

    # ---- VAE round-trip ----
    print("\n[1] Testing VAE round-trip...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
    vae = vae.to(device).eval()

    img = Image.open(args.image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),   # to [-1,1]
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"Input image shape: {img_tensor.shape}, min={img_tensor.min():.3f}, max={img_tensor.max():.3f}")

    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor
        print(f"Latents shape: {latents.shape}, mean={latents.mean():.4f}, std={latents.std():.4f}")

        decoded = vae.decode(latents / vae.config.scaling_factor).sample
        decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
    print("VAE round-trip complete. (Check the reconstructed image manually if needed.)")

    # ---- CLIP text encoder ----
    print("\n[2] Testing CLIP text encoder...")
    from transformers import CLIPTextModel, CLIPTokenizer
    txt_model = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(txt_model)
    text_encoder = CLIPTextModel.from_pretrained(txt_model, torch_dtype=torch.float32)
    text_encoder = text_encoder.to(device).eval()

    prompts = ["a photo of a cat", ""]
    tokens = tokenizer(prompts, padding="max_length", truncation=True,
                       max_length=tokenizer.model_max_length, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        out = text_encoder(**tokens)
        last_hidden = out.last_hidden_state.float()
        pooled = out.pooler_output.float()
        mask = tokens["attention_mask"]
    print(f"last_hidden shape: {last_hidden.shape}, dtype={last_hidden.dtype}")
    print(f"pooled shape: {pooled.shape}, dtype={pooled.dtype}")
    print(f"mask shape: {mask.shape}")

    # ---- Flow matching path ----
    print("\n[3] Testing flow matching path...")
    cfm = ConditionalFlowMatcher(sigma_min=0.0)
    B, C, H, W = 2, 4, 8, 8
    x0 = torch.randn(B, C, H, W, device=device)
    x1 = torch.randn(B, C, H, W, device=device)
    t = torch.linspace(0, 1, 5, device=device)
    for ti in t:
        xt, target = cfm.sample_location_and_target(x0, x1, ti.expand(B))
        if ti == 0:
            assert torch.allclose(xt, x0, atol=1e-6), "t=0 should give x0"
        if ti == 1:
            assert torch.allclose(xt, x1, atol=1e-6), "t=1 should give x1"
    print("Flow path: OK")

    # ---- Model forward/backward ----
    print("\n[4] Testing model forward/backward...")
    cfg = MMDiTConfig(in_channels=C, patch_size=2, hidden_dim=32, num_heads=4, depth=2,
                      txt_dim=last_hidden.shape[-1], pooled_dim=last_hidden.shape[-1])
    model = FlowCraftMMDiT(cfg).to(device)
    # Use a small batch
    x_latent = torch.randn(B, C, H, W, device=device)
    t_vals = torch.rand(B, device=device)
    text_embeds = last_hidden[:B]  # use first B prompts
    pooled_text = pooled[:B]
    mask = mask[:B]

    pred = model(x_latent, t_vals, text_embeds, txt_mask=mask, pooled_text=pooled_text)
    print(f"Prediction shape: {pred.shape}, mean={pred.mean():.4f}, std={pred.std():.4f}")

    loss = (pred - torch.randn_like(pred)).pow(2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Loss: {loss.item():.6f}, gradient norm: {grad_norm:.4f}")

    # ---- Euler sampler ----
    print("\n[5] Testing Euler sampler (mechanics only)...")
    sampler = EulerSampler(num_steps=3)
    with torch.no_grad():
        sample = sampler.sample(
            model=model,
            x_shape=(B, C, H, W),
            text_embeds=text_embeds,
            device=device,
            dtype=torch.float32,
            model_kwargs={"txt_mask": mask, "pooled_text": pooled_text},
        )
    print(f"Sample shape: {sample.shape}, mean={sample.mean():.4f}, std={sample.std():.4f}")
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()