#!/bin/bash
# facebook/nllb-200-distilled-600M モデルをHuggingFaceからダウンロード

MODEL_NAME="facebook/nllb-200-distilled-600M"
MODEL_DIR="$(dirname "$0")/nllb-200-distilled-600M"

# transformers>=4.21.0 ならhuggingface-cliが使える
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cliが見つかりません。pip install huggingface_hub を実行してください。"
    exit 1
fi

mkdir -p "$MODEL_DIR"
huggingface-cli download $MODEL_NAME --local-dir "$MODEL_DIR" --local-dir-use-symlinks False

echo "モデルのダウンロードが完了しました: $MODEL_DIR" 