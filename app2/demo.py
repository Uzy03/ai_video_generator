# app2/demo.py
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
import requests
import re
import socket

WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external/weights"))

# -----------------------------------------------------------------------------
# 0)  Streamlit page settings
# -----------------------------------------------------------------------------
st.set_page_config(page_title="🎬 AI Image→Video Generator (Multi-GPU)", layout="centered")
st.title("🎬 AI Image→Video Generator (Multi-GPU)")

# MLLB APIサーバが起動していなければ自動起動
API_HOST = "localhost"
API_PORT = 8000

def is_api_running(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False

if not is_api_running(API_HOST, API_PORT):
    subprocess.Popen(["bash", "API/run_api.sh"])

# -----------------------------------------------------------------------------
# 1)  Input widgets (main area)
# -----------------------------------------------------------------------------
uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

# 画像サイズのデフォルト値を初期化
img_default_width = 768
img_default_height = 512
img_info_text = ""
if uploaded_img is not None:
    try:
        img = Image.open(uploaded_img)
        w, h = img.size
        img_default_width = min(w, 1280)
        img_default_height = min(h, 720)
        img_info_text = f"入力画像サイズ: {w} x {h}"
    except Exception:
        img_info_text = "画像サイズの取得に失敗しました"

col1, col2 = st.columns(2)
with col1:
    prompt_en: str = st.text_input(
        "Prompt (Japanese OK)",
        value="A young girl bravely and beautifully swings a sword.",
        placeholder="Enter your prompt in English or Japanese…",
    )
    # 日本語判定用正規表現
    def contains_japanese(text):
        return re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text) is not None
    # ローカルAPIで日本語→英語翻訳
    def translate_to_en_local(text):
        try:
            res = requests.post(
                "http://localhost:8000/translate",
                json={"text": text},
                timeout=10
            )
            if res.status_code == 200:
                return res.json().get("translatedText", text)
            else:
                return text
        except Exception:
            return text
    if prompt_en.strip():
        if contains_japanese(prompt_en):
            prompt_translated = translate_to_en_local(prompt_en)
        else:
            prompt_translated = prompt_en
    else:
        prompt_translated = ""
    st.caption(f"英語訳: {prompt_translated}")
    prompt = prompt_translated
with col2:
    model: str = st.selectbox(
        "Model engine",
        [
            "LTX-Video (LTXV-2B)",
        ],
    )

# -----------------------------------------------------------------------------
# 2)  Advanced parameters (sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Advanced settings")

    if img_info_text:
        st.info(img_info_text)

    # --- 共通パラメータ ---
    frame_num: int = st.number_input(
        "Frame count (8n+1)", min_value=9, max_value=161, step=8, value=17,
        help="Total number of frames to synthesise. LTX-Videoは9, 17, 25, ...など8n+1を推奨。",
    )
    fps: int = st.number_input(
        "Output FPS", min_value=1, max_value=60, step=1, value=1,
        help="生成される動画のフレームレート（fps）"
    )
    offload_to_cpu: bool = st.checkbox(
        "Offload to CPU (省メモリ)", value=True,
        help="一部計算をCPUに逃がしてVRAM消費を抑制"
    )

    # --- LTX-Videoや解像度指定モデルのみ ---
    if model.startswith("LTX-Video"):
        # 32の倍数リストを生成（256〜1280, 256〜720）
        width_options = [i for i in range(256, 1281, 32)]
        height_options = [i for i in range(256, 721, 32)]
        # 推奨解像度ボタン
        if 'set_recommended' not in st.session_state:
            st.session_state['set_recommended'] = False
        if st.button("推奨解像度(1216x704)に設定"):
            st.session_state['out_width'] = 1216
            st.session_state['out_height'] = 704
            st.session_state['set_recommended'] = True
        # デフォルト値の決定
        default_width = st.session_state.get('out_width', (img_default_width // 32) * 32)
        default_height = st.session_state.get('out_height', (img_default_height // 32) * 32)
        # セレクトボックス
        out_width = st.selectbox("Output width (32の倍数)", width_options, index=width_options.index(default_width) if default_width in width_options else 0)
        out_height = st.selectbox("Output height (32の倍数)", height_options, index=height_options.index(default_height) if default_height in height_options else 0)
        # 選択値をセッションに保存
        st.session_state['out_width'] = out_width
        st.session_state['out_height'] = out_height
        st.info(f"出力サイズ: {out_width} x {out_height}")

    # --- Wan‑specific --------------------------------------------------------
    if model.startswith("Wan2.1"):
        resolution = st.selectbox(
            "Output resolution", [
                "1280*720", "960*540", "832*480", "720*1280"
            ],
            index=0,
        )
        sample_steps: int = st.number_input(
            "Sample steps", min_value=1, max_value=100, step=1, value=16,
            help="サンプリングステップ数"
        )
        guide_scale: float = st.slider(
            "Guidance scale", min_value=1.0, max_value=20.0, step=0.5, value=7.5,
            help="Classifier-free guidance scale"
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

# 生成結果・エラーをセッションで管理
if 'video_path' not in st.session_state:
    st.session_state['video_path'] = None
if 'video_bytes' not in st.session_state:
    st.session_state['video_bytes'] = None
if 'gen_error' not in st.session_state:
    st.session_state['gen_error'] = None

if run_btn:
    st.session_state['gen_error'] = None
    try:
        # 4)  Prepare temporary paths
        with st.status("⏳ Setting up…"):
            tmp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            if model.startswith("LTX-Video"):
                img = Image.open(uploaded_img).convert("RGB")
                img.save(tmp_img_file, format="JPEG")
            else:
                tmp_img_file.write(uploaded_img.read())
            tmp_img_file.flush()

            output_dir = Path(tempfile.mkdtemp())

        # 5)  Build command line
        run_subprocess = True
        if model.startswith("LTX-Video"):
            config_path = "external/LTX-Video/configs/ltxv-2b-0.9.6-distilled.yaml"
            cmd = [
                "python", "external/LTX-Video/inference.py",
                "--prompt", prompt,
                "--conditioning_media_paths", tmp_img_file.name,
                "--conditioning_start_frames", "0",
                "--height", str(out_height),
                "--width", str(out_width),
                "--num_frames", str(frame_num),
                "--frame_rate", str(fps),
                "--output_path", str(output_dir / "ltx.mp4/output.mp4"),
                "--pipeline_config", config_path,
            ]
            if offload_to_cpu:
                cmd += ["--offload_to_cpu", "True"]
            extra_env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        elif model.startswith("Wan2.1"):
            task = "i2v-14B"
            ckpt_dir = os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-720P")
            cmd = [
                "python", "external/Wan2.1/generate.py",
                "--task", task,
                "--size", resolution,  # resolutionのみを使う
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
            extra_env = {}
        elif model.startswith("HunyuanVideo"):
            ckpt_dir = os.path.join(WEIGHTS_DIR, "HunyuanVideo-I2V")
            cmd = [
                "python", "external/HunyuanVideo-I2V/inference.py",
                "--ckpt_dir", ckpt_dir,
                "--input_media_path", tmp_img_file.name,
                "--prompt", prompt,
                "--output_path", str(output_dir / "hunyuan.mp4"),
            ]
            extra_env = {}
        elif model.startswith("CogVideoX"):
            ckpt_dir = os.path.join(WEIGHTS_DIR, "CogVideoX-2B")
            cmd = [
                "python", "external/CogVideo/inference.py",
                "--ckpt_dir", ckpt_dir,
                "--input_path", tmp_img_file.name,
                "--prompt", prompt,
                "--output_path", str(output_dir / "cogvideo.mp4"),
            ]
            extra_env = {}
        elif model.startswith("SkyReels"):
            ckpt_dir = os.path.join(WEIGHTS_DIR, "SkyReels-V2-14B")
            cmd = [
                "python", "external/SkyReels-V2/inference.py",
                "--ckpt_dir", ckpt_dir,
                "--input_path", tmp_img_file.name,
                "--prompt", prompt,
                "--output_path", str(output_dir / "skyreels.mp4"),
            ]
            extra_env = {}

        st.code("$ " + " ".join(cmd), language="bash")
        with st.status("🖥️  Running model… this can take a few minutes."):
            if run_subprocess:
                # ここでCUDA_VISIBLE_DEVICES=0,1を追加
                env_multi_gpu = {**os.environ, **extra_env, "CUDA_VISIBLE_DEVICES": "0,1"}
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env_multi_gpu,
                )
            else:
                proc = None
        if proc is not None and proc.returncode != 0:
            st.session_state['gen_error'] = "**Generation failed**. See logs below:"
            st.session_state['video_path'] = None
            st.session_state['video_bytes'] = None
            st.session_state['gen_stdout'] = proc.stdout or "<empty>"
            st.session_state['gen_stderr'] = proc.stderr or "<empty>"
        else:
            # 7)  Display result
            video_files = [f for f in output_dir.rglob("*.mp4") if f.is_file()]
            if not video_files:
                st.session_state['gen_error'] = "No video file produced."
                st.session_state['video_path'] = None
                st.session_state['video_bytes'] = None
            else:
                video_path = video_files[0]
                with video_path.open("rb") as f:
                    video_bytes = f.read()
                st.session_state['video_path'] = str(video_path)
                st.session_state['video_bytes'] = video_bytes
    except Exception as e:
        st.session_state['gen_error'] = f"エラー: {e}"
        st.session_state['video_path'] = None
        st.session_state['video_bytes'] = None

# 生成結果・エラーの表示
if st.session_state.get('gen_error'):
    st.error(st.session_state['gen_error'])
    if 'gen_stdout' in st.session_state:
        st.expander("STDOUT").write(st.session_state['gen_stdout'])
    if 'gen_stderr' in st.session_state:
        st.expander("STDERR").write(st.session_state['gen_stderr'])

if st.session_state.get('video_path') and st.session_state.get('video_bytes'):
    st.video(st.session_state['video_path'])
    st.download_button("Download video", st.session_state['video_bytes'], file_name=Path(st.session_state['video_path']).name, mime="video/mp4")
