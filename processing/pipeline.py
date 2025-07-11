import cv2
import numpy as np
def process_frame(frame_bytes: bytes) -> bytes:
    # バイト列を NumPy 配列に変換
    arr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # OpenCV 処理実行 (例: グレースケール)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, buf = cv2.imencode('.jpg', gray)
    return buf.tobytes()