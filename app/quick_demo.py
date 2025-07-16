# app/quick_demo.py
"""
Streamlit front‑end for Wan2.1 / HunyuanVideo image→video pipelines.
This demo currently targets the official **Wan 2.1 I2V‑14B** checkpoint.
Support for the smaller 1.3 B variant was planned but the upstream
`generate.py` script does not recognise the ``i2v-1.3B`` task.  Until the
upstream tool adds this option we default to the 14 B model only.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import streamlit as st
from PIL import Image

WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external/weights"))

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
            "HunyuanVideo‑I2V",
            "LTX-Video (LTXV-2B)",
            "CogVideoX-2B",
            "Stable Video Diffusion",
            "SkyReels GGUF",
            "Wan2.1 GGUF",
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
    if model.startswith("LTX-Video"):
        # LTX-Video用: 必ずRGBで保存
        img = Image.open(uploaded_img).convert("RGB")
        img.save(tmp_img_file, format="JPEG")
    else:
        # 他モデルはそのまま保存
        tmp_img_file.write(uploaded_img.read())
    tmp_img_file.flush()

    output_dir = Path(tempfile.mkdtemp())

# -----------------------------------------------------------------------------
# 5)  Build command line
# -----------------------------------------------------------------------------
run_subprocess = True

if model.startswith("Wan2.1"):
    task = "i2v-14B"
    ckpt_dir = os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-720P")
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
        "--t5_cpu",
    ]

elif model.startswith("HunyuanVideo"):
    ckpt_dir = os.path.join(WEIGHTS_DIR, "HunyuanVideo-I2V")
    cmd = [
        "python", "external/HunyuanVideo-I2V/inference.py",
        "--ckpt_dir", ckpt_dir,
        "--input_media_path", tmp_img_file.name,  # 正しい引数名に修正
        "--prompt", prompt,
        "--output_path", str(output_dir / "hunyuan.mp4"),  # 正しい引数名に修正
        # 必要に応じて他の引数も追加
    ]

elif model.startswith("LTX-Video"):
    config_path = "external/LTX-Video/configs/ltxv-2b-0.9.1.yaml"
    cmd = [
        "python", "external/LTX-Video/inference.py",
        "--pipeline_config", config_path,
        "--input_media_path", tmp_img_file.name,
        "--prompt", prompt,
        "--output_path", str(output_dir / "ltx.mp4"),
        # 必要に応じて他の引数も追加
    ]

elif model.startswith("CogVideoX"):
    ckpt_dir = os.path.join(WEIGHTS_DIR, "CogVideoX-2B")
    cmd = [
        "python", "external/CogVideo/inference.py",
        "--ckpt_dir", ckpt_dir,
        "--input_path", tmp_img_file.name,  # 正しい引数名に修正
        "--prompt", prompt,
        "--output_path", str(output_dir / "cogvideo.mp4"),  # 正しい引数名に修正
        # 必要に応じて他の引数も追加
    ]

elif model.startswith("SkyReels"):
    ckpt_dir = os.path.join(WEIGHTS_DIR, "SkyReels-V2-14B")
    cmd = [
        "python", "external/SkyReels-V2/inference.py",
        "--ckpt_dir", ckpt_dir,
        "--input_path", tmp_img_file.name,  # 正しい引数名に修正
        "--prompt", prompt,
        "--output_path", str(output_dir / "skyreels.mp4"),  # 正しい引数名に修正
        # 必要に応じて他の引数も追加
    ]

st.code("$ " + " ".join(cmd), language="bash")

# -----------------------------------------------------------------------------
# 6)  Run subprocess and stream logs
# -----------------------------------------------------------------------------
extra_env = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

with st.status("🖥️  Running model… this can take a few minutes."):
    if run_subprocess:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **extra_env},
        )
    else:
        # The pipeline was executed directly in Python
        pass

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
