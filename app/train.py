"""FlowCraft-DiT training entry point.

Wires together:
    coco_10k/captions.csv + coco_10k/images/   (data, scripts/download_coco_10k.py)
    CLIP text encoder                          (frozen, transformers)
    Stable-Diffusion VAE                       (frozen, diffusers)
    FlowCraftMMDiT                             (trainable, models/mm_dit.py)
    ConditionalFlowMatcher                     (flow/cfm.py)
    LogitNormalSampler                         (training/logit_normal.py)

Usage:
    python app/train.py --data_dir data/coco_30k --out_dir checkpoints --resolution 256 --batch_size 4 --grad_accum 4 --steps 50000 --lr 1e-4 --hidden_dim 768 --depth 12 --num_heads 12 --patch_size 2 --cond_dropout 0.1 --ema_decay 0.999 --precision bf16 --log_every 50 --save_every 2000 --preview_every 500 --preview_prompt "a photo of a dog on a beach" --preview_cfg 5.0 --preview_steps 28
    python app/train.py --data_dir data/coco_10k --out_dir checkpoints --resolution 128 --batch_size 4 --grad_accum 1 --steps 2000 --lr 1e-4 --hidden_dim 256 --depth 4 --num_heads 4 --patch_size 2 --cond_dropout 0.1 --ema_decay 0.999 --precision bf16 --log_every 100 --save_every 500 --preview_every 500 --preview_prompt "a photo of a dog" --preview_cfg 3.0 --preview_steps 15    

reuse: python app/train.py --resume checkpoints/flowcraft_step5000.pt --hidden_dim 256 --depth 4 --num_heads 4 --patch_size 2 --steps 20000
    """

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from flow.cfm import ConditionalFlowMatcher
from flow.euler import EulerSampler
from models.mm_dit import FlowCraftMMDiT, MMDiTConfig
from training.logit_normal import LogitNormalSampler
from utils.torch_utils import EMA, default_device, resolve_dtype, seed_everything

DEFAULT_TEXT_ENCODER_ID = "openai/clip-vit-large-patch14"
DEFAULT_VAE_ID = "stabilityai/sd-vae-ft-mse"


class CocoCaptionDataset(Dataset):
    """Reads coco_10k/captions.csv (image_id, file_name, caption)."""

    def __init__(self, data_dir: str, resolution: int = 256, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        captions_csv = self.data_dir / "captions.csv"
        if not captions_csv.is_file():
            raise FileNotFoundError(
                f"{captions_csv} not found. Run scripts/download_coco_10k.py first."
            )

        available = {p.name for p in self.image_dir.glob("*") if p.is_file()}
        rows, missing = [], 0
        with open(captions_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["file_name"] in available:
                    rows.append((row["file_name"], row["caption"]))
                else:
                    missing += 1
        if not rows:
            raise RuntimeError(
                f"No caption row in {captions_csv} matches a file in {self.image_dir}."
            )
        if missing:
            print(f"[data] skipped {missing} caption rows whose image file is missing")
        
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError(f"max_samples must be positive when set, got {max_samples}.")
            
            rows = rows[:min(max_samples, len(rows))]
            print(f"[data] limiting training to {len(rows)} caption rows for an overfit/debug run")
        self.rows = rows

        self.transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # VAE expects [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        file_name, caption = self.rows[idx]
        image = Image.open(self.image_dir / file_name).convert("RGB")
        return self.transform(image), caption


def load_frozen_encoders(
    device: torch.device, encoder_dtype: torch.dtype, text_encoder_id: str, vae_id: str
):
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(text_encoder_id)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_id, torch_dtype=encoder_dtype)
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=encoder_dtype)

    text_encoder = text_encoder.to(device).eval().requires_grad_(False)
    vae = vae.to(device).eval().requires_grad_(False)
    return tokenizer, text_encoder, vae


@torch.no_grad()
def encode_text(tokenizer, text_encoder, captions: List[str], device: torch.device):
    tokens = tokenizer(
        list(captions),
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    out = text_encoder(**tokens)
    return (
        out.last_hidden_state.float(),          # [B, N_txt, txt_dim]
        out.pooler_output.float(),              # [B, txt_dim]
        tokens["attention_mask"],               # [B, N_txt]
    )


@torch.no_grad()
def encode_images(vae, images: torch.Tensor, device: torch.device, dtype: torch.dtype):
    images = images.to(device=device, dtype=dtype)
    latents = vae.encode(images).latent_dist.sample()
    return (latents * vae.config.scaling_factor).float()  # [B, C, H/8, W/8]


def lr_lambda(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def drop_captions(captions: List[str], p: float) -> List[str]:
    """Randomly blanks captions so the model also learns the unconditional field (CFG)."""
    if p <= 0.0:
        return list(captions)
    return ["" if random.random() < p else c for c in captions]


@torch.no_grad()
def save_preview(
    model, vae, tokenizer, text_encoder, args, device, out_path: Path, prompt: str
) -> None:
    sampler = EulerSampler(num_steps=args.preview_steps)
    generator = torch.Generator(device=device).manual_seed(args.preview_seed)
    text_embeds, pooled, mask = encode_text(tokenizer, text_encoder, [prompt], device)
    null_embeds, null_pooled, null_mask = encode_text(tokenizer, text_encoder, [""], device)

    latent_hw = args.resolution // 8
    x = sampler.sample(
        model=model,
        x_shape=(1, model.config.in_channels, latent_hw, latent_hw),
        text_embeds=text_embeds,
        null_text_embeds=null_embeds,
        cfg_scale=args.preview_cfg,
        device=device,
        generator=generator,
        model_kwargs={"txt_mask": mask, "pooled_text": pooled},
        null_model_kwargs={"txt_mask": null_mask, "pooled_text": null_pooled},
    )
    latents = (x / vae.config.scaling_factor).to(next(vae.parameters()).dtype)
    decoded = vae.decode(latents).sample
    decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
    array = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(array).save(out_path)
    print(f"[preview saved] {out_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FlowCraft-DiT on COCO captions.")
    parser.add_argument("--data_dir", type=str, default="data/coco_10k")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--cond_dropout", type=float, default=0.1,
                        help="Probability of blanking a caption, needed for CFG at inference.")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="0 disables EMA.")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--preview_every", type=int, default=0, help="0 disables previews.")
    parser.add_argument("--preview_prompt", type=str, default="a photo of a dog on a beach")
    parser.add_argument("--preview_steps", type=int, default=28)
    parser.add_argument("--preview_cfg", type=float, default=5.0)
    parser.add_argument(
        "--preview_seed",
        type=int,
        default=1234,
        help="Fixed noise seed for comparable previews across training checkpoints.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit caption rows for a deliberate tiny-set overfit/debug experiment.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Fraction of data to use for validation (0 disables)."
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--text_encoder_id", type=str, default=DEFAULT_TEXT_ENCODER_ID)
    parser.add_argument("--vae_id", type=str, default=DEFAULT_VAE_ID)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else default_device()
    autocast_dtype = resolve_dtype(args.precision, device)
    use_autocast = autocast_dtype != torch.float32 and device.type in ("cuda", "cpu")
    encoder_dtype = torch.float32 if autocast_dtype == torch.float16 else autocast_dtype

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | autocast: {autocast_dtype} | encoders: {encoder_dtype}")
    print("Loading frozen VAE + CLIP text encoder ...")
    tokenizer, text_encoder, vae = load_frozen_encoders(
        device, encoder_dtype, args.text_encoder_id, args.vae_id
    )
    txt_dim = text_encoder.config.hidden_size
    in_channels = vae.config.latent_channels

    if args.resolution % (8 * args.patch_size) != 0:
        raise ValueError(
            f"--resolution {args.resolution} must be divisible by 8 * patch_size "
            f"({8 * args.patch_size}) so the latent grid divides into patches."
        )

    cfg = MMDiTConfig(
        in_channels=in_channels,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        txt_dim=txt_dim,
        pooled_dim=txt_dim,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    model = FlowCraftMMDiT(cfg).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, args.warmup_steps, args.steps, args.min_lr_ratio)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=autocast_dtype == torch.float16)
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    cfm = ConditionalFlowMatcher(sigma_min=0.0)
    t_sampler = LogitNormalSampler()

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        if ema is not None and ckpt.get("ema_state"):
            ema.load_state_dict(ckpt["ema_state"])
        start_step = ckpt["step"]
        print(f"Resumed from {args.resume} at step {start_step}")

    full_dataset = CocoCaptionDataset(
        args.data_dir, resolution=args.resolution, max_samples=args.max_samples
    )
    if args.val_split > 0:
        val_len = int(len(full_dataset) * args.val_split)
        train_len = len(full_dataset) - val_len
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
        print(f"Train samples: {train_len}, Val samples: {val_len}")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"Train samples: {len(train_dataset)}")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    def infinite(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    data_iter = infinite(loader)

    def save_checkpoint(step: int) -> None:
        ckpt_path = out_dir / f"flowcraft_step{step}.pt"
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": asdict(cfg),
            "text_encoder_id": args.text_encoder_id,
            "vae_id": args.vae_id,
            "args": vars(args),
            "step": step,
        }
        if ema is not None:
            payload["ema_state"] = ema.state_dict()
        torch.save(payload, ckpt_path)
        print(f"[checkpoint saved] {ckpt_path}")

    model.train()
    running_loss, logged_steps, skipped = 0.0, 0, 0
    step = start_step
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss, accum_ok = 0.0, 0

        for _ in range(args.grad_accum):
            images, captions = next(data_iter)
            captions = drop_captions(list(captions), args.cond_dropout)

            with torch.no_grad():
                latents = encode_images(vae, images, device, encoder_dtype)
                text_embeds, pooled, txt_mask = encode_text(
                    tokenizer, text_encoder, captions, device
                )
            if not torch.isfinite(latents).all():
                skipped += 1
                print(f"[warn] non-finite VAE latents at step {step}; batch skipped")
                continue

            t = t_sampler.sample(latents.shape[0], device=device, dtype=torch.float32)

            with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
                loss = cfm.compute_loss(
                    model,
                    latents,
                    text_embeds,
                    t=t,
                    model_kwargs={"txt_mask": txt_mask, "pooled_text": pooled},
                )
            if not torch.isfinite(loss):
                skipped += 1
                print(f"[warn] non-finite loss at step {step}; batch skipped")
                continue

            scaler.scale(loss / args.grad_accum).backward()
            accum_loss += loss.item() / args.grad_accum
            accum_ok += 1

        if accum_ok:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

        scheduler.step()
        step += 1
        running_loss += accum_loss
        logged_steps += 1

        if step % args.log_every == 0:
            avg = running_loss / max(1, logged_steps)
            print(
                f"step {step:>7d}/{args.steps} | loss {avg:.4f} "
                f"| lr {scheduler.get_last_lr()[0]:.2e} | skipped {skipped}"
            )
            running_loss, logged_steps = 0.0, 0

        if val_dataset is not None and step % args.log_every == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_imgs, val_caps in DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0):
                    val_caps = list(val_caps)
                    val_latents = encode_images(vae, val_imgs, device, encoder_dtype)
                    val_text, val_pooled, val_mask = encode_text(tokenizer, text_encoder, val_caps, device)
                    t_val = t_sampler.sample(val_latents.shape[0], device=device, dtype=torch.float32)
                    with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
                        loss_val = cfm.compute_loss(
                            model, val_latents, val_text, t=t_val,
                            model_kwargs={"txt_mask": val_mask, "pooled_text": val_pooled}
                        )
                    val_loss += loss_val.item()
                    val_steps += 1
                    if val_steps >= 10:  # limit to 10 batches for speed
                        break
            val_loss /= max(1, val_steps)
            print(f"step {step} | val loss {val_loss:.4f}")
            model.train()

        if args.preview_every and step % args.preview_every == 0:
            try:
                preview_model = ema.shadow if ema is not None else model
                save_preview(
                    preview_model, vae, tokenizer, text_encoder, args, device,
                    out_dir / f"preview_step{step}.png", args.preview_prompt,
                )
                model.train()
            except Exception as exc:
                print(f"[warn] preview failed at step {step}: {exc}")

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(step)

    print("Training complete.")


if __name__ == "__main__":
    main()
