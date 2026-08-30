from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class REPALoss(nn.Module):
    """Representation alignment against frozen DINOv2 patch features.

    The DiT token grid (latent_size / patch_size) and the DINOv2 token grid
    (image_size / 14) almost never match, so DINOv2 features are bilinearly
    resampled onto the DiT grid before the cosine loss instead of being
    silently broadcast/erroring out.
    """

    def __init__(
        self,
        dit_dim: int = 1536,
        dinov2_model_name: str = "dinov2_vits14",
        input_range: str = "tanh",
        dino_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        if input_range not in ("tanh", "unit"):
            raise ValueError("input_range must be 'tanh' ([-1, 1]) or 'unit' ([0, 1]).")
        self.input_range = input_range

        self.dinov2 = dino_module if dino_module is not None else torch.hub.load(
            "facebookresearch/dinov2", dinov2_model_name
        )
        self.dinov2.eval()
        for param in self.dinov2.parameters():
            param.requires_grad_(False)

        dino_dim = getattr(self.dinov2, "embed_dim", None)
        if dino_dim is None:
            raise AttributeError("DINOv2 backbone is missing 'embed_dim'.")

        self.projector = nn.Sequential(
            nn.Linear(dit_dim, dino_dim),
            nn.GELU(),
            nn.Linear(dino_dim, dino_dim),
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> "REPALoss":
        super().train(mode)
        self.dinov2.eval()  # the frozen backbone must never leave eval mode
        return self

    def _normalize(self, pixel_images: torch.Tensor) -> torch.Tensor:
        if self.input_range == "tanh":
            pixel_images = (pixel_images + 1.0) / 2.0
        pixel_images = pixel_images.clamp(0.0, 1.0)
        return (pixel_images - self.mean) / self.std

    @torch.no_grad()
    def extract_dino_features(self, pixel_images: torch.Tensor) -> torch.Tensor:
        """Returns DINOv2 patch tokens [B, N_patches, dino_dim] (CLS excluded)."""
        features = self.dinov2.forward_features(self._normalize(pixel_images))
        return features["x_norm_patchtokens"]

    @staticmethod
    def _resample_tokens(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        """Bilinearly resamples a square token grid to `target_len` tokens."""
        B, N, D = tokens.shape
        if N == target_len:
            return tokens
        src = int(round(N**0.5))
        dst = int(round(target_len**0.5))
        if src * src != N or dst * dst != target_len:
            raise ValueError(
                f"REPA needs square token grids to resample; got {N} DINO tokens "
                f"and {target_len} DiT tokens."
            )
        grid = tokens.transpose(1, 2).reshape(B, D, src, src)
        grid = F.interpolate(grid.float(), size=(dst, dst), mode="bilinear", align_corners=False)
        return grid.flatten(2).transpose(1, 2).to(tokens.dtype)

    def forward(self, dit_hidden_states: torch.Tensor, pixel_images: torch.Tensor) -> torch.Tensor:
        """
        dit_hidden_states: [B, N_img, dit_dim] image-stream features from MM-DiT
        pixel_images: [B, 3, H, W] the real images behind the latent batch
        """
        with torch.no_grad():
            dino_feats = self.extract_dino_features(pixel_images.to(self.mean.dtype))
            dino_feats = self._resample_tokens(dino_feats, dit_hidden_states.shape[1])
            dino_feats = F.normalize(dino_feats.float(), dim=-1)

        proj_dit_feats = self.projector(dit_hidden_states)
        proj_dit_feats = F.normalize(proj_dit_feats.float(), dim=-1)

        return 1.0 - (proj_dit_feats * dino_feats).sum(dim=-1).mean()
