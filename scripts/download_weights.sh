#!/usr/bin/env bash
set -e

# weights フォルダを作成
mkdir -p ../weights

# 例）モデル重みのダウンロード
# gdown などで Google Drive / Hugging Face 上の重みを落とす
gdown --id YOUR_MODEL_FILE_ID -O ../weights/model.pth
