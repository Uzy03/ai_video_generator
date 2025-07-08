# app/quick_demo.py

import os, sys
# スクリプト（app/quick_demo.py）の１つ上、つまりリポジトリルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np, cv2
from processing.pipeline import process_frame  # 既存のパイプライン関数

class Transformer(VideoTransformerBase):
    def transform(self, frame):
        # 1) VideoFrame → NumPy
        img = frame.to_ndarray(format="bgr24")
        # 2) JPEG にエンコードしてバイト列取得
        _, buf = cv2.imencode('.jpg', img)
        # 3) 既存関数を呼び出し（bytes → bytes）
        out_bytes = process_frame(buf.tobytes())
        # 4) JPEG bytes → NumPy
        arr = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)
        # 5) NumPy → VideoFrame に戻して返却
        return frame.from_ndarray(arr, format="bgr24")

st.title("🎬 AI Video Generator — Real-Time Demo")
webrtc_streamer(
    key="video",
    mode="SENDRECV",
    video_transformer_factory=Transformer,
    media_stream_constraints={"video": True, "audio": False},
)
