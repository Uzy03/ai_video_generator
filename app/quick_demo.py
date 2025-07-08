# app/quick_demo.py
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from processing.pipeline import process_frame
import av
import cv2
import numpy as np

class Transformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # フレームを JPEG バイト列に変換
        img = frame.to_ndarray(format="bgr24")
        _, buf = cv2.imencode('.jpg', img)
        # 非同期 pipeline を同期呼び出し
        processed_bytes = st.experimental_sync(process_frame)(buf.tobytes())
        # 戻り値を VideoFrame に変換して返す
        img2 = cv2.imdecode(
            np.frombuffer(processed_bytes, np.uint8), cv2.IMREAD_COLOR)
        return av.VideoFrame.from_ndarray(img2, format="bgr24")

st.title("🎬 AI Video Generator — Real-Time Demo")
webrtc_streamer(
    key="example",
    mode="SENDRECV",
    video_transformer_factory=Transformer,
    media_stream_constraints={"video": True, "audio": False},
)
