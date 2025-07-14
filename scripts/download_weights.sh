#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Download model weights for all supported models
# ------------------------------------------------------------
mkdir -p weights
cd weights


# ----------------------------
# LTX-Video (LTXV-2B)
# ----------------------------
if [ ! -d "LTX-Video" ]; then
  echo ">>> Downloading LTX-Video weights"
  huggingface-cli download ltxai/ltxv-2b-0.9.6-distilled --local-dir LTX-Video --local-dir-use-symlinks False
fi

# ----------------------------
# CogVideoX-2B (I2V mode)
# ----------------------------
if [ ! -d "CogVideoX-2B" ]; then
  echo ">>> Downloading CogVideoX-2B weights"
  huggingface-cli download THUDM/CogVideoX-2B-I2V --local-dir CogVideoX-2B --local-dir-use-symlinks False
fi

# ----------------------------
# Stable Video Diffusion
# ----------------------------
if [ ! -d "stable-video-diffusion" ]; then
  echo ">>> Downloading Stable Video Diffusion weights"
  huggingface-cli download stabilityai/stable-video-diffusion-img2vid --local-dir stable-video-diffusion --local-dir-use-symlinks False
fi

# ----------------------------
# SkyReels V2 GGUF
# ----------------------------
if [ ! -d "SkyReels" ]; then
  echo ">>> Downloading SkyReels GGUF weights"
  huggingface-cli download skyworkshops/skyreels-v2-gguf --local-dir SkyReels --local-dir-use-symlinks False
fi

# ----------------------------
# Wan2.1 GGUF version
# ----------------------------
if [ ! -d "Wan2.1-GGUF" ]; then
  echo ">>> Downloading Wan2.1 GGUF weights"
  huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-gguf --local-dir Wan2.1-GGUF --local-dir-use-symlinks False
fi

cd ..

echo ">>> All weights downloaded."