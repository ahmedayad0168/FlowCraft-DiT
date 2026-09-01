"""
Production-ready FlowCraft-DiT pipeline - can be used independently of UI framework.

This module contains the core generation logic without UI dependencies,
making it suitable for testing, API integration, and batch processing.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import torch
from PIL import Image

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flow.euler import EulerSampler
from models.mm_dit import FlowCraftMMDiT, MMDiTConfig
from utils.torch_utils import default_device


# ============================================================
# Configuration & Constants
# ============================================================

DEFAULT_TEXT_ENCODER_ID = "openai/clip-vit-large-patch14"
DEFAULT_VAE_ID = "stabilityai/sd-vae-ft-mse"
VAE_SCALE_FACTOR = 8

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class GenerationConfig:
    """Configuration for a single generation."""
    prompt: str
    negative_prompt: str = ""
    steps: int = 28
    cfg_scale: float = 5.0
    resolution: int = 128
    seed: int = -1


@dataclass 
class GenerationResult:
    """Result of a generation."""
    image: Image.Image
    config: GenerationConfig
    timestamp: datetime = field(default_factory=datetime.now)
    generation_time: float = 0.0
    status: str = "success"


# ============================================================
# Pipeline Class
# ============================================================

class FlowCraftPipeline:
    """Production-ready FlowCraft-DiT text-to-image pipeline."""

    def __init__(
        self,
        checkpoint: Optional[str],
        device: Optional[str] = None,
        use_ema: bool = True,
        dtype: Optional[str] = None,
    ):
        self.device = torch.device(device) if device else default_device()
        
        # Determine dtype
        if dtype:
            self.dtype = self._get_dtype(dtype)
        else:
            # Use BF16 on supported CUDA devices, otherwise FP32
            if (
                self.device.type == "cuda"
                and torch.cuda.is_bf16_supported()
            ):
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32

        self.demo_mode = checkpoint is None
        self.ckpt = None
        self.training_resolution: Optional[int] = None
        self.vae_scaling_factor: Optional[float] = None  
    
        
        # Load checkpoint
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)
        
        # Determine model IDs
        text_encoder_id = (
            self.ckpt.get("text_encoder_id", DEFAULT_TEXT_ENCODER_ID)
            if self.ckpt
            else DEFAULT_TEXT_ENCODER_ID
        )
        
        vae_id = (
            self.ckpt.get("vae_id", DEFAULT_VAE_ID)
            if self.ckpt
            else DEFAULT_VAE_ID
        )
        
        # Load encoders
        self._load_encoders(text_encoder_id, vae_id)
        
        # Build and load model
        self._build_model()
        
        # Cache empty-prompt embeddings
        self._null_cache: Optional[tuple] = None
        
        # Log status
        if self.demo_mode:
            logger.warning("No checkpoint loaded. Model has random weights.")
        else:
            logger.info(f"FlowCraft-DiT ready | device={self.device} | dtype={self.dtype}")

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype_str = dtype_str.lower()
        if dtype_str not in dtype_map:
            raise ValueError(f"Invalid dtype: {dtype_str}. Must be one of {list(dtype_map.keys())}")
        return dtype_map[dtype_str]

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint with error handling."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            self.ckpt = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            saved_args = self.ckpt.get("args", {})
            saved_resolution = saved_args.get("resolution")
            if saved_resolution is not None:
                self.training_resolution = int(saved_resolution)
                logger.info(f"Training resolution: {self.training_resolution}px")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _load_encoders(self, text_encoder_id: str, vae_id: str):
        """Load CLIP text encoder and VAE with error handling."""
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import AutoencoderKL
        
        try:
            logger.info(f"Loading text encoder: {text_encoder_id}")
            self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_id)
            self.text_encoder = (
                CLIPTextModel.from_pretrained(text_encoder_id, torch_dtype=self.dtype)
                .to(self.device)
                .eval()
                .requires_grad_(False)
            )
            
            logger.info(f"Loading VAE: {vae_id}")
            self.vae = (
                AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float32)
                .to(self.device)
                .eval()
                .requires_grad_(False)
            )
            self.vae_scaling_factor = self.vae.config.scaling_factor
            logger.info(f"VAE scaling factor: {self.vae_scaling_factor}")
        except Exception as e:
            logger.error(f"Failed to load encoders: {e}")
            raise

    def _build_model(self):
        """Build and load the MM-DiT model."""
        if self.ckpt:
            if "config" not in self.ckpt:
                raise KeyError("Checkpoint does not contain a model config.")

            config = MMDiTConfig(**self.ckpt["config"])
        else:
            config = MMDiTConfig(
                in_channels=self.vae.config.latent_channels,
                hidden_dim=256,
                depth=4,
                num_heads=4,
                txt_dim=self.text_encoder.config.hidden_size,
                pooled_dim=self.text_encoder.config.hidden_size,
            )

        logger.info("Creating FlowCraft MM-DiT model on CPU...")
        self.model = FlowCraftMMDiT(config)

        # --------------------------------------------------------
        # IMPORTANT: verify that the model is not on META device
        # --------------------------------------------------------
        meta_params = [
            name for name, param in self.model.named_parameters()
            if param.device.type == "meta"
        ]

        if meta_params:
            raise RuntimeError(
                "FlowCraft model was created with META parameters before "
                f"checkpoint loading. First parameters: {meta_params[:10]}"
            )

        # --------------------------------------------------------
        # Load checkpoint weights
        # --------------------------------------------------------
        if self.ckpt:
            if use_ema := self.ckpt.get("ema_state"):
                weights = use_ema
                logger.info("Using EMA weights")
            else:
                weights = self.ckpt["model_state"]
                logger.info("Using raw model weights")

            # Checkpoint should contain real tensors
            meta_weights = [
                name for name, value in weights.items()
                if isinstance(value, torch.Tensor) and value.device.type == "meta"
            ]

            if meta_weights:
                raise RuntimeError(
                    "Checkpoint contains META tensors. "
                    f"First affected keys: {meta_weights[:10]}"
                )

            missing, unexpected = self.model.load_state_dict(
                weights,
                strict=False,
            )

            if missing:
                logger.warning(
                    f"Missing model keys ({len(missing)}): {missing[:10]}"
                )

            if unexpected:
                logger.warning(
                    f"Unexpected model keys ({len(unexpected)}): {unexpected[:10]}"
                )

        # --------------------------------------------------------
        # Move AFTER loading weights
        # --------------------------------------------------------
        self.model = self.model.to(
            device=self.device,
            dtype=self.dtype,
        )

        self.model.eval()
        self.model.requires_grad_(False)

        self.cfg = config

        logger.info(
            f"Model loaded successfully | "
            f"device={self.device} | dtype={self.dtype}"
        )

    @torch.no_grad()
    def _encode_text(self, prompts: List[str]) -> tuple:
        """Encode text prompts to embeddings."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        output = self.text_encoder(**tokens)
        
        return (
            output.last_hidden_state.to(self.dtype),
            output.pooler_output.to(self.dtype),
            tokens["attention_mask"],
        )

    @torch.no_grad()
    def _null_embeds(self) -> tuple:
        """Get embeddings for empty prompt (cached)."""
        if self._null_cache is None:
            self._null_cache = self._encode_text([""])
        return self._null_cache

    @torch.no_grad()
    def generate(self, config: GenerationConfig) -> GenerationResult:
        """Generate image from configuration with timing and error handling."""
        start_time = datetime.now()
        
        try:
            # Validate prompt
            if not config.prompt.strip():
                raise ValueError("Prompt cannot be empty")
            
            # Encode prompts
            text_embeds, pooled, mask = self._encode_text([config.prompt])
            
            # Handle negative prompt
            if config.negative_prompt.strip():
                null_embeds, null_pooled, null_mask = self._encode_text([config.negative_prompt])
            else:
                null_embeds, null_pooled, null_mask = self._null_embeds()
            
            # Validate resolution
            align = VAE_SCALE_FACTOR * self.cfg.patch_size
            resolution = max(align, (int(config.resolution) // align) * align)
            
            if self.training_resolution is not None and resolution != self.training_resolution:
                logger.warning(
                    f"Checkpoint trained at {self.training_resolution}px, but requested {resolution}px. "
                    "This may cause quality degradation."
                )
            
            latent_size = resolution // VAE_SCALE_FACTOR
            
            # Set up random seed
            generator = torch.Generator(device=self.device)
            if config.seed >= 0:
                generator.manual_seed(config.seed)
            else:
                generator.seed()
            
            # Euler sampling
            latents = EulerSampler(num_steps=config.steps).sample(
                model=self.model,
                x_shape=(1, self.cfg.in_channels, latent_size, latent_size),
                text_embeds=text_embeds,
                null_text_embeds=null_embeds,
                cfg_scale=config.cfg_scale,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
                model_kwargs={"txt_mask": mask, "pooled_text": pooled},
                null_model_kwargs={"txt_mask": null_mask, "pooled_text": null_pooled},
            )
            
            # VAE decode
            latents = latents.float()
            scaling_factor = self.vae_scaling_factor or self.vae.config.scaling_factor
            latents = latents / scaling_factor
            decoded = self.vae.decode(latents).sample
            
            # Denormalize and convert to PIL
            decoded = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
            image_array = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            image = Image.fromarray(image_array)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return GenerationResult(
                image=image,
                config=config,
                generation_time=generation_time,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Return error result
            return GenerationResult(
                image=Image.new("RGB", (512, 512), color="gray"),
                config=config,
                generation_time=generation_time,
                status=f"Error: {str(e)}"
            )
