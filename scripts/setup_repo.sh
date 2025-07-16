#!/usr/bin/env bash
set -euo pipefail

# --- paths ---
ROOT=$(pwd)
EXT=$ROOT/external
WEIGHTS=$EXT/weights

mkdir -p "$EXT" "$WEIGHTS"
cd  "$EXT"

# ---------- Git repos ----------
git clone https://github.com/Wan-Video/Wan2.1.git            || true
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V.git || true
git clone https://github.com/Lightricks/LTX-Video.git        || true
git clone https://github.com/THUDM/CogVideo.git              || true
git clone https://github.com/SkyworkAI/SkyReels-V2.git       || true

# ---------- HF weights ----------
# ※ 事前に:  huggingface-cli login  （トークン）＋各モデルページで「Agree and access」クリック

# Wan 2.1 (14B, FP16 safetensors)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir "$WEIGHTS/Wan2.1-I2V-14B-720P" --local-dir-use-symlinks False

# HunyuanVideo-I2V
huggingface-cli download tencent/HunyuanVideo-I2V \
    --local-dir "$WEIGHTS/HunyuanVideo-I2V" --local-dir-use-symlinks False

# LTX-Video distilled 2B
#huggingface-cli download Lightricks/LTX-Video-0.9.6-distilled \
#    --local-dir "$WEIGHTS/LTX-Video-2B" --local-dir-use-symlinks False

# --- LTX-Video distilled 2B (0.9.6) ---
#huggingface-cli download Lightricks/LTX-Video-2B-0.9.6-Distilled-04-25 \
#  --resume-download \
#  --local-dir "$WEIGHTS/LTX-Video-2B" \
#  --local-dir-use-symlinks False

cd weights
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Lightricks/LTX-Video LTX-Video-2B
cd LTX-Video-2B
git lfs pull --include "ltxv-2b-0.9.6-distilled-04-25.safetensors"
cd ..

# CogVideoX-2B (I2V)
huggingface-cli download THUDM/CogVideoX-2b \
    --local-dir "$WEIGHTS/CogVideoX-2B" --local-dir-use-symlinks False

# SkyReels-V2 I2V 14B 720P
huggingface-cli download Skywork/SkyReels-V2-I2V-14B-720P \
    --local-dir "$WEIGHTS/SkyReels-V2-14B" --local-dir-use-symlinks False

cd "$ROOT"

# ---------- Python deps ----------
pip install -r requirements.txt
pip install -r external/HunyuanVideo-I2V/requirements.txt
grep -v '^flash_attn' external/Wan2.1/requirements.txt | pip install -r /dev/stdin

echo "✅ All repos and weights fetched."
