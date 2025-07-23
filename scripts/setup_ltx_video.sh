#!/usr/bin/env bash
set -euo pipefail

# --- paths ---
ROOT=$(pwd)
EXT=$ROOT/external
WEIGHTS=$EXT/weights

mkdir -p "$EXT" "$WEIGHTS"
cd  "$EXT"

# ---------- Git repos ----------
# LTX-Videoのみクローン
if [ ! -d "LTX-Video" ]; then
  git clone https://github.com/Lightricks/LTX-Video.git
fi

# ---------- HF weights ----------
cd "$WEIGHTS"
# LTX-Video distilled 2B (0.9.6) のみダウンロード
if [ ! -d "LTX-Video-2B" ]; then
  git lfs install
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Lightricks/LTX-Video LTX-Video-2B
  cd LTX-Video-2B
  git lfs pull --include "ltxv-2b-0.9.6-distilled-04-25.safetensors"
  cd ..
fi

# ---------- MLLB (NLLB) モデルのダウンロード ----------
bash "$ROOT/API/download_model.sh"

cd "$ROOT"

# ---------- Python deps ----------
pip install -r "$ROOT/requirements.txt"
# LTX-Videoの追加依存
pip install -r "$ROOT/external/LTX-Video/pyproject.toml" || true

# 完了メッセージ
echo "✅ LTX-Videoのみセットアップ完了。" 