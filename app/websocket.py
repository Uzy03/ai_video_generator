from fastapi import APIRouter, WebSocket
from processing.pipeline import process_frame
import asyncio

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_bytes()
        # 後述の pipeline.process_frame を呼び出す
        processed = await asyncio.to_thread(process_frame, data)
        await ws.send_bytes(processed)