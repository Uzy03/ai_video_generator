# app/quick_demo.py

import streamlit as st
import subprocess, tempfile, os

st.set_page_config("Image→Video Demo", layout="centered")
st.title("🎬 AI Image→Video Generator")

# 1) 画像アップロード と プロンプト入力
uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
prompt   = st.text_input("Prompt", "A young girl bravely and beautifully swings a sword.")
model    = st.selectbox("Model", ["Wan2.1 (I2V-14B)", "HunyuanVideo-I2V"])

if st.button("Generate Video") and uploaded:
    # 一時ファイルに保存
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_img.write(uploaded.read())
    tmp_img.flush()

    # 出力先ディレクトリ
    out_dir = tempfile.mkdtemp()
    st.info("Generating… この処理には数分かかる場合があります")

    # 2) モデルごとに CLI コマンドを構築
    if model.startswith("Wan2.1"):
        cmd = [
            "python", "external/Wan2.1/generate.py",
            "--task",     "i2v-14B",
            "--size",     "1280*720",
            "--ckpt_dir", "external/Wan2.1/Wan2.1-I2V-14B-720P",
            "--image",    tmp_img.name,
            "--prompt",   prompt,
            "--save-path", os.path.join(out_dir, "wan2.1.mp4")
        ]
    else:
        cmd = [
            "python", "external/HunyuanVideo-I2V/sample_image2video.py",
            "--i2v-mode",
            "--i2v-image-path", tmp_img.name,
            "--model",          "HYVideo-T/2",
            "--prompt",         prompt,
            "--ckpts",          "external/HunyuanVideo-I2V/ckpts",
            "--save-path",      os.path.join(out_dir, "hunyuan.mp4"),
        ]

    # 3) サブプロセス実行
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"生成中にエラー発生: {e}")
        st.stop()

    # 4) 出力動画を表示＆ダウンロード
    video_files = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
    if video_files:
        path = os.path.join(out_dir, video_files[0])
        st.video(path)
        with open(path, "rb") as f:
            st.download_button("Download Video", f.read(), file_name=video_files[0], mime="video/mp4")
    else:
        st.error("動画ファイルが見つかりませんでした。")
