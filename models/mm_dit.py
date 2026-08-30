from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.adaln import AdaLNZero, TimestepEmbedder, modulate
from models.rope import RoPE2D, apply_rope_2d


@dataclass
class MMDiTConfig:
    """Configuration schema for FlowCraft MM-DiT."""

    in_channels: int = 16            # Latent channels (e.g. SD3/FLUX VAE)
    patch_size: int = 2              # Spatial patch resolution
    hidden_dim: int = 1536           # Hidden transformer dimension
    num_heads: int = 24              # Attention heads
    depth: int = 24                  # Transformer block depth
    txt_dim: int = 4096              # Text encoder dimension (e.g. T5-XXL)
    pooled_dim: int = 0              # Pooled text vector dim (0 disables it)
    mlp_ratio: float = 4.0           # MLP expansion factor
    dropout: float = 0.0             # Dropout rate
    rope_theta: float = 10000.0      # RoPE base frequency
    repa_block_idx: int = -1         # Block whose image features REPA aligns
    use_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        head_dim = self.hidden_dim // self.num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be a multiple of 4 for 2D RoPE; "
                "adjust hidden_dim or num_heads."
            )


class JointAttention(nn.Module):
    """Joint attention over the concatenated [text, image] sequence.

    2D RoPE is applied to image tokens only; text tokens keep their (already
    positional) encoder embeddings.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv_img = nn.Linear(dim, 3 * dim)
        self.qkv_txt = nn.Linear(dim, 3 * dim)

        self.proj_img = nn.Linear(dim, dim)
        self.proj_txt = nn.Linear(dim, dim)

    def forward(
        self,
        norm_img: torch.Tensor,
        norm_txt: torch.Tensor,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_img, D = norm_img.shape
        _, N_txt, _ = norm_txt.shape

        def to_heads(qkv: torch.Tensor, n: int) -> torch.Tensor:
            return qkv.reshape(B, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        qkv_i = to_heads(self.qkv_img(norm_img), N_img)
        qkv_t = to_heads(self.qkv_txt(norm_txt), N_txt)

        q_i, k_i, v_i = qkv_i[0], qkv_i[1], qkv_i[2]  # [B, H, N_img, head_dim]
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]  # [B, H, N_txt, head_dim]

        cos, sin = rope_cos_sin
        q_i, k_i = apply_rope_2d(q_i, k_i, cos, sin)

        q = torch.cat([q_t, q_i], dim=2)
        k = torch.cat([k_t, k_i], dim=2)
        v = torch.cat([v_t, v_i], dim=2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N_txt + N_img, D)

        attn_txt, attn_img = attn_out.split([N_txt, N_img], dim=1)
        return self.proj_img(attn_img), self.proj_txt(attn_txt)


class MMDiTBlock(nn.Module):
    """Double-stream MM-DiT block with independent image/text MLPs and AdaLN."""

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        dim = config.hidden_dim
        mlp_hidden_dim = int(dim * config.mlp_ratio)

        self.img_norm = AdaLNZero(dim)
        self.txt_norm = AdaLNZero(dim)

        self.attn = JointAttention(dim, config.num_heads, config.dropout)

        self.mlp_img = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden_dim, dim),
        )
        self.mlp_txt = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        cond: torch.Tensor,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_msa_i, scale_i, g_msa_i, s_mlp_i, scale_mlp_i, g_mlp_i = self.img_norm(cond)
        s_msa_t, scale_t, g_msa_t, s_mlp_t, scale_mlp_t, g_mlp_t = self.txt_norm(cond)

        norm_img = modulate(self.img_norm.norm(x_img), s_msa_i, scale_i)
        norm_txt = modulate(self.txt_norm.norm(x_txt), s_msa_t, scale_t)

        attn_img, attn_txt = self.attn(norm_img, norm_txt, rope_cos_sin, attn_mask)

        x_img = x_img + g_msa_i * attn_img
        x_txt = x_txt + g_msa_t * attn_txt

        norm_mlp_i = modulate(self.img_norm.norm(x_img), s_mlp_i, scale_mlp_i)
        norm_mlp_t = modulate(self.txt_norm.norm(x_txt), s_mlp_t, scale_mlp_t)

        x_img = x_img + g_mlp_i * self.mlp_img(norm_mlp_i)
        x_txt = x_txt + g_mlp_t * self.mlp_txt(norm_mlp_t)

        return x_img, x_txt


class FlowCraftMMDiT(nn.Module):
    """Predicts the velocity field v_theta(x_t, t, text) for rectified flow."""

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.config = config
        p = config.patch_size

        self.img_in = nn.Linear(config.in_channels * p * p, config.hidden_dim)
        self.txt_in = nn.Linear(config.txt_dim, config.hidden_dim)
        self.time_embed = TimestepEmbedder(config.hidden_dim)
        self.pooled_in = (
            nn.Linear(config.pooled_dim, config.hidden_dim) if config.pooled_dim > 0 else None
        )

        head_dim = config.hidden_dim // config.num_heads
        self.rope2d = RoPE2D(head_dim=head_dim, theta=config.rope_theta)

        self.blocks = nn.ModuleList([MMDiTBlock(config) for _ in range(config.depth)])

        self.final_norm = nn.LayerNorm(config.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(config.hidden_dim, p * p * config.in_channels)
        self.final_adaLN = nn.Linear(config.hidden_dim, 2 * config.hidden_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

        # The global sweep above would otherwise destroy AdaLN-Zero, whose whole
        # point is that every block starts as an identity function.
        for block in self.blocks:
            block.img_norm.zero_init()
            block.txt_norm.zero_init()

        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, (H/p)*(W/p), p*p*C]."""
        p = self.config.patch_size
        B, C, H, W = x.shape
        if H % p != 0 or W % p != 0:
            raise ValueError(f"Latent size ({H}, {W}) must be divisible by patch_size {p}.")
        x = x.reshape(B, C, H // p, p, W // p, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p) * (W // p), p * p * C)

    def unpatchify(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """[B, N, p*p*C] -> [B, C, H, W]."""
        p = self.config.patch_size
        c = self.config.in_channels
        B = x.shape[0]
        x = x.reshape(B, h_patches, w_patches, p, p, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(B, c, h_patches * p, w_patches * p)

    def _build_attn_mask(
        self, txt_mask: torch.Tensor, n_img: int, num_heads: int
    ) -> torch.Tensor:
        """Blocks attention to padded text keys. Returns [B, 1, 1, N_txt + N_img] bool."""
        img_keep = txt_mask.new_ones((txt_mask.shape[0], n_img), dtype=torch.bool)
        keep = torch.cat([txt_mask.bool(), img_keep], dim=1)
        return keep[:, None, None, :]

    def forward(
        self,
        x_latent: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor,
        txt_mask: Optional[torch.Tensor] = None,
        pooled_text: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        """
        x_latent: [B, C, H, W] noisy latent at timestep t
        t: [B] continuous timesteps in [0, 1]
        text_embeds: [B, N_txt, txt_dim]
        txt_mask: [B, N_txt] 1 for real tokens, 0 for padding (optional)
        pooled_text: [B, pooled_dim] pooled text vector (optional, see config)
        return_features: also return the intermediate image features REPA uses
        """
        B, C, H, W = x_latent.shape
        if C != self.config.in_channels:
            raise ValueError(
                f"Latent has {C} channels but the model was configured for {self.config.in_channels}."
            )
        p = self.config.patch_size
        h_patches, w_patches = H // p, W // p

        x_img = self.img_in(self.patchify(x_latent))
        x_txt = self.txt_in(text_embeds.to(x_img.dtype))

        cond = self.time_embed(t)
        if self.pooled_in is not None:
            if pooled_text is None:
                raise ValueError("Model was configured with pooled_dim > 0 but pooled_text is None.")
            cond = cond + self.pooled_in(pooled_text.to(cond.dtype))

        rope_cos_sin = self.rope2d(
            h_patches, w_patches, device=x_latent.device, dtype=x_img.dtype
        )
        attn_mask = (
            self._build_attn_mask(txt_mask, x_img.shape[1], self.config.num_heads)
            if txt_mask is not None
            else None
        )

        repa_idx = self.config.repa_block_idx % len(self.blocks) if self.blocks else -1
        features = None
        for idx, block in enumerate(self.blocks):
            if self.config.use_gradient_checkpointing and self.training:
                x_img, x_txt = checkpoint(
                    block, x_img, x_txt, cond, rope_cos_sin, attn_mask, use_reentrant=False
                )
            else:
                x_img, x_txt = block(x_img, x_txt, cond, rope_cos_sin, attn_mask)
            if return_features and idx == repa_idx:
                features = x_img

        shift, scale = self.final_adaLN(cond).unsqueeze(1).chunk(2, dim=-1)
        x_img = modulate(self.final_norm(x_img), shift, scale)
        v_patches = self.final_linear(x_img)
        v_latent = self.unpatchify(v_patches, h_patches, w_patches)

        if return_features:
            return v_latent, features
        return v_latent
