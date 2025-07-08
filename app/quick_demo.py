# app/quick_demo.py
import sys, os
# app/quick_demo.py の親ディレクトリ（プロジェクトルート）を path に追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from processing.pipeline import process_frame  # 既存のフレーム処理関数
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(
    page_title="🎨 AI Video Generator — Image Upload Demo",
    layout="centered"
)
st.title("🎨 AI Video Generator — Upload & Process")

# 画像アップロード UI
uploaded = st.file_uploader(
    label="Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    # 1) アップロードファイル → バイト列
    img_bytes = uploaded.read()

    # 2) 既存の処理関数に渡す（bytes → bytes）
    try:
        out_bytes = process_frame(img_bytes)
    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
        st.stop()

    # 3) 返ってきた JPEG bytes → NumPy (BGR)
    arr_bgr = cv2.imdecode(
        np.frombuffer(out_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    # 4) BGR → RGB に変換し、PIL Image にする
    arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(arr_rgb)

    # 5) 結果表示
    st.image(result_img, caption="Processed Image", use_column_width=True)

    # 6) ダウンロードボタン（任意）
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    st.download_button(
        label="Download Processed Image",
        data=buf,
        file_name="processed.jpg",
        mime="image/jpeg"
    )
else:
    st.info("上のボタンから画像（jpg/png）をアップロードしてください。")
