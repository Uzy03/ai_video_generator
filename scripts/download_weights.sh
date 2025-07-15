#!/usr/bin/env bash
set -euo pipefail

mkdir -p weights && cd weights

# LTX-Video
[ ! -d "LTX-Video" ] && \
  huggingface-cli download Lightricks/LTX-Video \
    --local-dir LTX-Video --local-dir-use-symlinks False

# CogVideoX-2B
[ ! -d "CogVideoX-2B" ] && \
  huggingface-cli download THUDM/CogVideoX-2b \
    --local-dir CogVideoX-2B --local-dir-use-symlinks False

# Stable Video Diffusion（要 HF login & ライセンス同意）
[ ! -d "stable-video-diffusion" ] && \
  huggingface-cli download stabilityai/stable-video-diffusion-img2vid \
    --local-dir stable-video-diffusion --local-dir-use-symlinks False

# SkyReels GGUF
[ ! -d "SkyReels" ] && \
  huggingface-cli download wsbagnsv1/SkyReels-V2-I2V-14B-540P-GGUF \
    --local-dir SkyReels --local-dir-use-symlinks False

# Wan 2.1 GGUF（コミュニティ版）
[ ! -d "Wan2.1-GGUF" ] && \
  huggingface-cli download city96/Wan2.1-I2V-14B-720P-gguf \
    --local-dir Wan2.1-GGUF --local-dir-use-symlinks False

echo ">>> All weights downloaded."
