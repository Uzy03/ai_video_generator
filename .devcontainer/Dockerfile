# Python 3.10（arm64 / amd64 自動切替）
FROM mcr.microsoft.com/devcontainers/python:0-3.10

# OS 依存ライブラリだけ apt で入れる
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libgl1 build-essential git && \
    rm -rf /var/lib/apt/lists/*

# ... 既存のベースイメージ指定など
WORKDIR /workspace

# 1) まず torch と torchvision を先に入れる
RUN pip install --no-cache-dir torch torchvision

# .devcontainer/Dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel


# リポジトリをクローン済みとする
RUN git clone https://github.com/Wan-Video/Wan2.1.git \
    && git clone https://github.com/tencent/HunyuanVideo-I2V.git

# flash-attn を含むセットアップの前に packaging を入れる
RUN pip install --no-cache-dir packaging

# その後で各リポジトリの requirements をまとめて入れる
RUN pip install --no-cache-dir -r Wan2.1/requirements.txt \
    && pip install --no-cache-dir -r HunyuanVideo-I2V/requirements.txt
