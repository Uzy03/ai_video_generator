#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# 保存先ディレクトリ（既定 ckpt_dir）
# ------------------------------------------------------------
DEST="external/Wan2.1/Wan2.1-I2V-14B-720P"
mkdir -p "$DEST"

echo ">>> Downloading Wan2.1 core weights (VAE / UNet shards / T5 / CLIP)…"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
  --local-dir "$DEST" --local-dir-use-symlinks False \
  --include \
    "Wan2.1_VAE.pth" \
    "models_t5_umt5-xxl-enc-bf16.pth" \
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "diffusion_pytorch_model-*.safetensors"

echo ">>> Downloading CLIP tokenizer (xlm-roberta-large)…"
huggingface-cli download xlm-roberta-large \
  --local-dir "$DEST/xlm-roberta-large" --local-dir-use-symlinks False

echo ">>> Downloading Wan-custom T5 tokenizer (extra_ids = 0)…"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
  "wan_t5_tokenizer" \
  --local-dir "$DEST/wan_t5_tokenizer" \
  --local-dir-use-symlinks False

echo ">>> Creating google/umt5-xxl → wan_t5_tokenizer symlink…"
mkdir -p "$DEST/google"
ln -sfn ../wan_t5_tokenizer "$DEST/google/umt5-xxl"

echo ">>> All Wan2.1 weights & tokenizers are ready:"
tree -L 2 "$DEST" | head -20


#追加
DEST="external/Wan2.1/Wan2.1-I2V-14B-720P/google/umt5-xxl"
mkdir -p "$DEST"

huggingface-cli download google/umt5-xxl \
  --local-dir "$DEST" \
  --local-dir-use-symlinks False

#追加
DEST="external/Wan2.1/Wan2.1-I2V-14B-720P"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
  --local-dir "$DEST" --local-dir-use-symlinks False \
  --include "config.json" "diffusion_pytorch_model.safetensors.index.json"
