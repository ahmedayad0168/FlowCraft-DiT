"""
FastAPI REST API for FlowCraft-DiT text-to-image generation.

This provides a production-ready REST API for programmatic access
to the generation pipeline, suitable for integration into larger systems.

Usage:
    python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import uuid
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import torch

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import FlowCraftPipeline, GenerationConfig, GenerationResult
from app.config_manager import get_config

# ============================================================
# Configuration
# ============================================================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[FlowCraftPipeline] = None

# Store for generation results (in production, use a database)
generation_store: dict = {}

# Load configuration
config = get_config()

# Pydantic Models
class GenerationRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., description="Text prompt for image generation", min_length=1)
    negative_prompt: str = Field("", description="Negative prompt (what to avoid)")
    steps: int = Field(config.generation.default_steps, ge=1, le=config.generation.max_steps, description="Number of inference steps")
    cfg_scale: float = Field(config.generation.default_cfg_scale, ge=1.0, le=config.generation.max_cfg_scale, description="Classifier-free guidance scale")
    resolution: int = Field(config.generation.default_resolution, ge=config.generation.min_resolution, le=config.generation.max_resolution, description="Output image resolution")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    return_image: bool = Field(True, description="Return image in response")

class GenerationResponse(BaseModel):
    """Response model for image generation."""
    id: str
    status: str
    message: str
    generation_time: float
    config: dict
    image_url: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: Optional[str] = None
    dtype: Optional[str] = None
    training_resolution: Optional[int] = None

class BatchGenerationRequest(BaseModel):
    """Request model for batch generation."""
    prompts: List[str] = Field(..., description="List of text prompts", min_items=1, max_items=10)
    negative_prompt: str = Field("", description="Negative prompt (what to avoid)")
    steps: int = Field(config.generation.default_steps, ge=1, le=config.generation.max_steps, description="Number of inference steps")
    cfg_scale: float = Field(config.generation.default_cfg_scale, ge=1.0, le=config.generation.max_cfg_scale, description="Classifier-free guidance scale")
    resolution: int = Field(config.generation.default_resolution, ge=config.generation.min_resolution, le=config.generation.max_resolution, description="Output image resolution")
    seed: int = Field(-1, description="Random seed (-1 for random)")

# FastAPI App
app = FastAPI(title="FlowCraft-DiT API",
               description="Production-ready REST API for FlowCraft-DiT text-to-image generation", version="1.0.0")

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(429, _rate_limit_exceeded_handler)
    RATE_LIMIT_ENABLED = True
except ImportError:
    RATE_LIMIT_ENABLED = False
    logger.warning("slowapi not installed; rate limiting disabled")

# Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup using configuration."""
    global pipeline

    checkpoint_path = config.model.checkpoint_path
    device = config.model.device if config.model.device != "auto" else None
    dtype = config.model.dtype

    try:
        logger.info(f"Loading pipeline from {checkpoint_path}")
        pipeline = FlowCraftPipeline(
            checkpoint=checkpoint_path,
            device=device,
            use_ema=config.model.use_ema,
            dtype=dtype
        )
        logger.info("Pipeline loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Checkpoint not found: {e}")
        pipeline = None
        # Keep API running in degraded state
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        pipeline = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global pipeline
    logger.info("Shutting down API")
    pipeline = None

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        return HealthResponse(status="degraded", model_loaded=False)

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=str(pipeline.device),
        dtype=str(pipeline.dtype),
        training_resolution=pipeline.training_resolution
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate a single image from text prompt."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    # Generate unique ID
    gen_id = str(uuid.uuid4())

    try:
        # Create generation config
        config_gen = GenerationConfig(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            resolution=request.resolution,
            seed=request.seed
        )

        logger.info(f"Generating image for prompt: {request.prompt[:50]}...")

        # Generate image
        result = pipeline.generate(config_gen)

        # Save image
        output_dir = Path(config.storage.api_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_path = output_dir / f"{gen_id}.png"
        result.image.save(image_path)

        # Store result
        generation_store[gen_id] = {
            "result": result,
            "image_path": str(image_path),
            "timestamp": datetime.now()
        }

        logger.info(f"Generation {gen_id} completed in {result.generation_time:.2f}s")

        return GenerationResponse(
            id=gen_id,
            status=result.status,
            message="Generation completed successfully",
            generation_time=result.generation_time,
            config={
                "prompt": result.config.prompt,
                "negative_prompt": result.config.negative_prompt,
                "steps": result.config.steps,
                "cfg_scale": result.config.cfg_scale,
                "resolution": result.config.resolution,
                "seed": result.config.seed
            },
            image_url=f"/image/{gen_id}" if request.return_image else None
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch", response_model=List[GenerationResponse])
async def batch_generate(request: BatchGenerationRequest):
    """Generate multiple images from text prompts."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    results = []

    for i, prompt in enumerate(request.prompts):
        try:
            gen_id = str(uuid.uuid4())

            config_gen = GenerationConfig(
                prompt=prompt,
                negative_prompt=request.negative_prompt,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                resolution=request.resolution,
                seed=request.seed if request.seed >= 0 else -1
            )

            logger.info(f"Batch generation {i+1}/{len(request.prompts)}: {prompt[:50]}...")

            result = pipeline.generate(config_gen)

            # Save image
            output_dir = Path(config.storage.api_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            image_path = output_dir / f"{gen_id}.png"
            result.image.save(image_path)

            # Store result
            generation_store[gen_id] = {
                "result": result,
                "image_path": str(image_path),
                "timestamp": datetime.now()
            }

            results.append(GenerationResponse(
                id=gen_id,
                status=result.status,
                message="Generation completed successfully",
                generation_time=result.generation_time,
                config={
                    "prompt": result.config.prompt,
                    "negative_prompt": result.config.negative_prompt,
                    "steps": result.config.steps,
                    "cfg_scale": result.config.cfg_scale,
                    "resolution": result.config.resolution,
                    "seed": result.config.seed
                },
                image_url=f"/image/{gen_id}"
            ))

        except Exception as e:
            logger.error(f"Batch generation {i+1} failed: {e}")
            results.append(GenerationResponse(
                id=str(uuid.uuid4()),
                status="error",
                message="Generation failed",
                generation_time=0.0,
                config={"prompt": prompt},
                error=str(e)
            ))

    return results

@app.get("/image/{gen_id}")
async def get_image(gen_id: str):
    """Retrieve a generated image by ID."""
    if gen_id not in generation_store:
        raise HTTPException(status_code=404, detail="Generation not found")

    image_path = generation_store[gen_id]["image_path"]

    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(image_path, media_type="image/png")

@app.get("/generations")
async def list_generations(limit: int = 10):
    """List recent generations."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    # Get recent generations
    recent = list(generation_store.items())[-limit:]

    return {
        "total": len(generation_store),
        "recent": [
            {
                "id": gen_id,
                "prompt": data["result"].config.prompt,
                "timestamp": data["timestamp"].isoformat(),
                "status": data["result"].status,
                "generation_time": data["result"].generation_time
            }
            for gen_id, data in recent
        ]
    }

@app.delete("/generations/{gen_id}")
async def delete_generation(gen_id: str):
    """Delete a generation."""
    if gen_id not in generation_store:
        raise HTTPException(status_code=404, detail="Generation not found")

    # Delete image file
    image_path = generation_store[gen_id]["image_path"]
    if Path(image_path).exists():
        Path(image_path).unlink()

    # Remove from store
    del generation_store[gen_id]

    return {"message": "Generation deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port)
