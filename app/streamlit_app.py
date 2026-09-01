"""
Production-ready Streamlit app for FlowCraft-DiT text-to-image generation.

Usage:
    streamlit run app/streamlit_app.py -- --checkpoint checkpoints/flowcraft_step10000.pt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import FlowCraftPipeline, GenerationConfig, GenerationResult
from app.config_manager import get_config


# ============================================================
# Streamlit Configuration
# ============================================================

# Page configuration
st.set_page_config(
    page_title="FlowCraft-DiT",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .info-box {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    .success-box {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
    }

    .warning-box {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
    }

    .generation-info {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Management
# ============================================================

def init_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []
    if "show_history" not in st.session_state:
        st.session_state.show_history = False

@st.cache_resource
def load_pipeline(checkpoint_path, device, use_ema):
    """Load the pipeline with caching."""
    try:
        return FlowCraftPipeline(
            checkpoint=checkpoint_path if checkpoint_path else None,
            device=device if device != "auto" else None,
            use_ema=use_ema
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ============================================================
# UI Components
# ============================================================

def render_header():
    """Render application header."""
    st.markdown('<h1 class="main-header">FlowCraft-DiT</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Flow Matching · MM-DiT · CLIP · Stable Diffusion VAE</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_sidebar():
    """Render sidebar with configuration and status."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Checkpoint loading
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="checkpoints/flowcraft_step10000.pt",
            help="Path to trained model checkpoint"
        )

        device_option = st.selectbox(
            "Device",
            ["auto", "cuda", "cpu", "mps"],
            index=0,
            help="Device to run inference on"
        )

        use_ema = st.checkbox("Use EMA Weights", value=True, help="Use exponential moving average weights")

        # Load/unload button
        col1, col2 = st.columns(2)
        with col1:
            load_btn = st.button("Load Model", type="primary")
        with col2:
            unload_btn = st.button("Unload Model")

        if load_btn:
            device = None if device_option == "auto" else device_option
            with st.spinner("Loading model..."):
                pipeline = load_pipeline(checkpoint_path, device, use_ema)
                if pipeline is not None:
                    st.session_state.pipeline = pipeline
                    st.success("Model loaded successfully!")
                else:
                    st.session_state.pipeline = None

        if unload_btn:
            st.session_state.pipeline = None
            st.session_state.generation_history = []
            st.success("Model unloaded")

        st.markdown("---")

        # Pipeline status
        st.header("📊 Status")
        if st.session_state.pipeline is not None:
            st.info(f"✅ Model loaded")
            st.info(f"Device: {st.session_state.pipeline.device}")
            st.info(f"Dtype: {st.session_state.pipeline.dtype}")
            if st.session_state.pipeline.training_resolution:
                st.info(f"Training resolution: {st.session_state.pipeline.training_resolution}px")
            if st.session_state.pipeline.demo_mode:
                st.warning("⚠️ Demo mode: random weights")
        else:
            st.warning("⚠️ No model loaded")

        st.markdown("---")

        # Generation history toggle
        st.header("📜 History")
        show_history = st.checkbox("Show Generation History", value=False)
        st.session_state.show_history = show_history

        if show_history and st.session_state.generation_history:
            st.write(f"Total generations: {len(st.session_state.generation_history)}")
            # Show last 10 generations in reverse order
            recent_history = st.session_state.generation_history[-10:]
            recent_history.reverse()
            for i, result in enumerate(recent_history):
                gen_num = len(st.session_state.generation_history) - i
                with st.expander(f"Generation {gen_num}"):
                    st.image(result.image, caption=result.config.prompt)
                    st.json({
                        "prompt": result.config.prompt,
                        "steps": result.config.steps,
                        "cfg": result.config.cfg_scale,
                        "resolution": result.config.resolution,
                        "seed": result.config.seed,
                        "time": f"{result.generation_time:.2f}s",
                        "status": result.status
                    })


def render_generation_interface():
    """Render main generation interface."""
    if st.session_state.pipeline is None:
        st.warning("⚠️ Please load a model in the sidebar first")
        return

    # Generation mode selection
    mode = st.radio(
        "Generation Mode",
        ["Single Image", "Batch Generation"],
        horizontal=True
    )

    if mode == "Single Image":
        render_single_generation()
    else:
        render_batch_generation()


def render_single_generation():
    """Render single image generation interface."""
    # Main generation card
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎨 Generation Settings")

        # Prompt input
        prompt = st.text_area(
            "Prompt",
            placeholder="A cinematic photo of a futuristic city at sunset, highly detailed...",
            height=100,
            help="Describe the image you want to generate"
        )

        negative_prompt = st.text_area(
            "Negative Prompt (optional)",
            placeholder="blurry, distorted, low quality...",
            height=60,
            help="What to avoid in the generated image"
        )

        # Advanced settings in expander
        with st.expander("Advanced Settings"):
            col_a, col_b = st.columns(2)
            with col_a:
                steps = st.slider(
                    "Inference Steps",
                    min_value=1,
                    max_value=50,
                    value=28,
                    step=1,
                    help="Number of Euler integration steps"
                )
                cfg_scale = st.slider(
                    "CFG Scale",
                    min_value=1.0,
                    max_value=15.0,
                    value=5.0,
                    step=0.5,
                    help="Classifier-free guidance strength"
                )
            with col_b:
                default_res = st.session_state.pipeline.training_resolution or 128
                resolution = st.selectbox(
                    "Resolution",
                    options=[default_res, 256, 512] if default_res not in [256, 512] else [128, 256, 512],
                    index=0,
                    help="Output image resolution"
                )
                seed = st.number_input(
                    "Seed (-1 for random)",
                    value=-1,
                    help="Random seed for reproducibility"
                )

        # Generate button
        generate_btn = st.button("🚀 Generate Image", type="primary", use_container_width=True)

    with col2:
        st.subheader("🖼️ Output")

        # Output placeholder
        output_container = st.container()

        if generate_btn:
            with st.spinner("Generating image..."):
                config = GenerationConfig(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    resolution=resolution,
                    seed=seed
                )

                result = st.session_state.pipeline.generate(config)

                # Add to history
                st.session_state.generation_history.append(result)

                # Display result
                with output_container:
                    if result.status == "success":
                        st.image(result.image, caption=config.prompt, use_container_width=True)

                        # Generation info
                        st.markdown(f"""
                        <div class="generation-info">
                        <strong>Generation Info:</strong><br>
                        Resolution: {result.config.resolution}×{result.config.resolution} |
                        Steps: {result.config.steps} |
                        CFG: {result.config.cfg_scale:.1f} |
                        Seed: {result.config.seed} |
                        Time: {result.generation_time:.2f}s
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Generation failed: {result.status}")

                        # Show error placeholder
                        error_placeholder = Image.new("RGB", (512, 512), color="lightgray")
                        st.image(error_placeholder, caption="Generation failed")


def render_batch_generation():
    """Render batch generation interface."""
    st.subheader("🎨 Batch Generation")

    # Prompt input (one per line)
    prompts_text = st.text_area(
        "Prompts (one per line)",
        placeholder="a photo of a dog on a beach\na painting of a mountain at sunset\na futuristic city skyline...",
        height=150,
        help="Enter multiple prompts, one per line"
    )

    negative_prompt = st.text_area(
        "Negative Prompt (applied to all)",
        placeholder="blurry, distorted, low quality...",
        height=60,
        help="What to avoid in all generated images"
    )

    # Advanced settings
    with st.expander("Advanced Settings"):
        col_a, col_b = st.columns(2)
        with col_a:
            steps = st.slider(
                "Inference Steps",
                min_value=1,
                max_value=50,
                value=28,
                step=1
            )
            cfg_scale = st.slider(
                "CFG Scale",
                min_value=1.0,
                max_value=15.0,
                value=5.0,
                step=0.5
            )
        with col_b:
            default_res = st.session_state.pipeline.training_resolution or 128
            resolution = st.selectbox(
                "Resolution",
                options=[default_res, 256, 512] if default_res not in [256, 512] else [128, 256, 512],
                index=0
            )
            seed = st.number_input(
                "Seed (-1 for random)",
                value=-1
            )

    # Generate button
    generate_btn = st.button("🚀 Generate Batch", type="primary", use_container_width=True)

    if generate_btn:
        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]

        if not prompts:
            st.error("Please enter at least one prompt")
            return

        with st.spinner(f"Generating {len(prompts)} images..."):
            results = []
            for i, prompt in enumerate(prompts):
                config = GenerationConfig(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    resolution=resolution,
                    seed=seed if seed >= 0 else -1
                )

                result = st.session_state.pipeline.generate(config)
                results.append(result)
                st.session_state.generation_history.append(result)

                st.progress((i + 1) / len(prompts))

            # Display results in grid
            st.subheader(f"🖼️ Generated Images ({len(results)})")

            cols = st.columns(min(4, len(results)))
            for i, result in enumerate(results):
                with cols[i % len(cols)]:
                    st.image(result.image, caption=result.config.prompt[:30] + "...")
                    st.caption(f"{result.generation_time:.2f}s | {result.status}")


def render_model_info():
    """Render detailed model information."""
    if st.session_state.pipeline is None:
        return

    with st.expander("📋 Model Information"):
        config = st.session_state.pipeline.cfg

        st.json({
            "model_config": {
                "hidden_dim": config.hidden_dim,
                "depth": config.depth,
                "num_heads": config.num_heads,
                "patch_size": config.patch_size,
                "in_channels": config.in_channels,
                "txt_dim": config.txt_dim,
                "pooled_dim": config.pooled_dim,
            },
            "training_info": {
                "resolution": st.session_state.pipeline.training_resolution,
                "device": str(st.session_state.pipeline.device),
                "dtype": str(st.session_state.pipeline.dtype),
                "demo_mode": st.session_state.pipeline.demo_mode,
            }
        })


# ============================================================
# Main Application
# ============================================================

def main():
    """Main Streamlit application."""
    # Initialize session state
    init_session_state()

    # Render UI
    render_header()
    render_sidebar()
    render_generation_interface()
    render_model_info()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>
        FlowCraft-DiT · Research / Educational Project · Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
