# Python 3.10（arm64 / amd64 自動切替）
FROM mcr.microsoft.com/devcontainers/python:0-3.10

# OS 依存ライブラリだけ apt で入れる
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libgl1 build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    pip install --no-cache-dir -r requirements.txt
