# app/quick_demo.py
"""Streamlit front‑end for Wan2.1 / HunyuanVideo image→video pipelines.
Adds support for the lightweight **Wan 2.1 I2V‑1.3B** checkpoint so that
users on 24–32 GB GPUs can fall back when the full 14 B model does not fit.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# 0)  Streamlit page settings
# -----------------------------------------------------------------------------
st.set_page_config(page_title="🎬 AI Image→Video Generator", layout="centered")
st.title("🎬 AI Image→Video Generator")

# -----------------------------------------------------------------------------
# 1)  Input widgets (main area)
# -----------------------------------------------------------------------------
uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    prompt: str = st.text_input(
        "Prompt",
        value="A young girl bravely and beautifully swings a sword.",
        placeholder="Describe what should appear in the video…",
    )
with col2:
    model: str = st.selectbox(
        "Model engine",
        [
            "Wan2.1 (I2V‑14B)",
            "Wan2.1 (I2V‑1.3B)",  # NEW: lightweight variant
            "HunyuanVideo‑I2V",
        ],
    )

# -----------------------------------------------------------------------------
# 2)  Advanced parameters (sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Advanced settings")

    # --- Common --------------------------------------------------------------
    frame_num: int = st.number_input(
        "Frame count (4n+1)", min_value=9, max_value=81, step=4, value=49,
        help="Total number of frames to synthesise. Wan2.1 expects 4n+1 frames.",
    )

    sample_steps: int = st.slider(
        "Sampling steps", min_value=10, max_value=40, value=20,
        help="Diffusion iterations (quality vs. speed).",
    )

    guide_scale: float = st.slider(
        "CFG scale", min_value=1.0, max_value=10.0, step=0.5, value=5.0,
        help="Classifier‑free guidance scale (higher → stronger prompt adherence).",
    )

    # --- Wan‑specific --------------------------------------------------------
    if model.startswith("Wan2.1"):
        resolution = st.selectbox(
            "Output resolution", [
                "1280*720", "960*540", "832*480", "720*1280"
            ],
            index=0,
        )
        t5_cpu: bool = st.checkbox(
            "Off‑load T5 encoder to CPU (save VRAM)", value=True,
        )
        shift: float = st.slider(
            "Noise schedule shift", min_value=1.0, max_value=6.0,
            step=0.5, value=5.0,
            help="Lower values (≈3) recommended for 480p output.",
        )

# -----------------------------------------------------------------------------
# 3)  Launch button
# -----------------------------------------------------------------------------
run_btn = st.button("🚀 Generate video", disabled=uploaded_img is None)

if not run_btn:
    st.stop()

# -----------------------------------------------------------------------------
# 4)  Prepare temporary paths
# -----------------------------------------------------------------------------
with st.status("⏳ Setting up…"):
    tmp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_img_file.write(uploaded_img.read())
    tmp_img_file.flush()

    output_dir = Path(tempfile.mkdtemp())

# -----------------------------------------------------------------------------
# 5)  Build command line
# -----------------------------------------------------------------------------
if model.startswith("Wan2.1"):
    # Heuristic: choose landscape / portrait preset when user kept default
    if resolution in {"1280*720", "720*1280"}:
        w, h = Image.open(tmp_img_file.name).size
        if h > w:
            resolution = "720*1280"
        else:
            resolution = "1280*720"

    # Select task & checkpoint according to the variant
    if "1.3B" in model:
        task = "i2v-1.3B"
        ckpt_dir = "external/Wan2.1/Wan2.1-I2V-1.3B-720P"  # <-- ensure weights here
    else:
        task = "i2v-14B"
        ckpt_dir = "external/Wan2.1/Wan2.1-I2V-14B-720P"

    cmd = [
        "python", "external/Wan2.1/generate.py",
        "--task", task,
        "--size", resolution,
        "--frame_num", str(frame_num),
        "--sample_steps", str(sample_steps),
        "--sample_guide_scale", str(guide_scale),
        "--sample_shift", str(shift),
        "--ckpt_dir", ckpt_dir,
        "--image", tmp_img_file.name,
        "--prompt", prompt,
        "--save_file", str(output_dir / "wan2.1.mp4"),
        "--offload_model", "True",
    ]
    if t5_cpu:
        cmd.append("--t5_cpu")

    extra_env = {"PYTHONPATH": os.path.abspath("external/Wan2.1")}

else:  # HunyuanVideo‑I2V
    cmd = [
        "python", "external/HunyuanVideo-I2V/sample_image2video.py",
        "--i2v-mode",
        "--i2v-image-path", tmp_img_file.name,
        "--model", "HYVideo-T/2",
        "--prompt", prompt,
        "--ckpts", "external/HunyuanVideo-I2V/ckpts",
        "--save-path", str(output_dir / "hunyuan.mp4"),
        "--frame-num", str(frame_num),
        "--sample-steps", str(sample_steps),
        "--cfg-scale", str(guide_scale),
    ]
    extra_env = {}

st.code("$ " + " ".join(cmd), language="bash")

# -----------------------------------------------------------------------------
# 6)  Run subprocess and stream logs
# -----------------------------------------------------------------------------
with st.status("🖥️  Running model… this can take a few minutes."):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **extra_env},
    )

if proc.returncode != 0:
    st.error("**Generation failed**. See logs below:")
    st.expander("STDOUT").write(proc.stdout or "<empty>")
    st.expander("STDERR").write(proc.stderr or "<empty>")
    st.stop()

# -----------------------------------------------------------------------------
# 7)  Display result
# -----------------------------------------------------------------------------
video_files = list(output_dir.glob("*.mp4"))
if not video_files:
    st.error("No video file produced.")
    st.stop()

video_path = video_files[0]
st.video(str(video_path))
with video_path.open("rb") as f:
    st.download_button("Download video", f.read(), file_name=video_path.name, mime="video/mp4")
