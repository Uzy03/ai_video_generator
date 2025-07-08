from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.websocket import router as ws_router

from pathlib import Path                                   # ★追加
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI()

app.mount(
    "/", StaticFiles(directory=BASE_DIR / "fronted", html=True), name="static"
)

@app.get("/", response_class=HTMLResponse)                # ★追加
async def root():
    return "<h3>AI Video Generator backend is running 🎬</h3>"

app.include_router(ws_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)