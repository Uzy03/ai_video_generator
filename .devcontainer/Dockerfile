#####################################################################
# .devcontainer/Dockerfile  ― 最小構成 (nvcc も不要になります)
#####################################################################
ARG BASE_IMAGE=mcr.microsoft.com/devcontainers/python:3.11-bookworm
FROM ${BASE_IMAGE}

# OS 依存 (nvcc は要らない)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libgl1-mesa-glx build-essential git ninja-build && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ai_video_generator

# *** flash-attn を入れない ***
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# LTX-Video専用セットアップスクリプトを追加
COPY scripts/setup_ltx_video.sh scripts/setup_ltx_video.sh
RUN bash scripts/setup_ltx_video.sh
