import streamlit as st
from websocket import create_connection
import cv2

ws = create_connection("ws://localhost:8000/ws")
cap = cv2.VideoCapture(0)

st.title("リアルタイム画像処理デモ")
img_placeholder = st.empty()

while True:
    ret, frame = cap.read()
    _, buf = cv2.imencode('.jpg', frame)
    ws.send(buf.tobytes())
    result = ws.recv()
    img = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
    img_placeholder.image(img, channels="BGR")