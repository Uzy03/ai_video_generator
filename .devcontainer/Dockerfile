#####################################################################
# .devcontainer/Dockerfile  　★最終版
#####################################################################
# ❶ CUDA ランタイムが入った公式 Python イメージを 1 行で指定
ARG BASE_IMAGE=mcr.microsoft.com/devcontainers/python:3.11-bookworm
FROM ${BASE_IMAGE}

#####################################################################
# ❷ CUDA 開発ツールキット (nvcc) を追加      ← flash-attn ビルド用
#####################################################################
#RUN apt-get update && \
#    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#        cuda-toolkit-12-2 \
#        ffmpeg libgl1-mesa-glx build-essential git && \
#    rm -rf /var/lib/apt/lists/*

#####################################################################
# ❸ ワークスペースに移動
#####################################################################
WORKDIR /workspace

#####################################################################
# ❹ 主要ライブラリ（torch / torchvision）は CUDA 12.6 ビルドが入っているので不要
#    追加パッケージ・flash-attn などは requirements 実行時にビルドされる
#####################################################################

#####################################################################
# ❺ 任意: clone しておきたい外部リポジトリ（時間短縮用）
#####################################################################
RUN git clone https://github.com/Wan-Video/Wan2.1.git && \
    git clone https://github.com/tencent/HunyuanVideo-I2V.git

#####################################################################
# ❻ pip 依存のインストール
#####################################################################
#COPY requirements.txt ./
#RUN pip install --no-cache-dir -r requirements.txt && \
#    pip install --no-cache-dir flash-attn==2.8.1 --no-build-isolation
