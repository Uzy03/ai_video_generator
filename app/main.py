from fastapi import FastAPI
from app.websocket import router as ws_router

app = FastAPI()
app.include_router(ws_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)