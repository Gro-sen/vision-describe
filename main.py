from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import threading
from config import broadcast_queue
from camera import capture_rtsp
from stream import generate_frames

app = FastAPI(title="实时安防视频分析系统")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===================== WebSocket 管理 =====================
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for ws in list(self.active_connections):
            try:
                await ws.send_text(json.dumps(message, ensure_ascii=False))
            except Exception:
                self.disconnect(ws)

manager = ConnectionManager()

# ===================== Web =====================
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

async def broadcast_worker():
    import json
    import asyncio
    while True:
        try:
            result = broadcast_queue.get(timeout=1)
            await manager.broadcast(result)
            broadcast_queue.task_done()
        except Exception:
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup():
    import threading
    import asyncio
    
    # 启动摄像头捕获线程
    camera_thread = threading.Thread(
        target=capture_rtsp, 
        daemon=True,
        name="RTSP-Capture-Thread"
    )
    camera_thread.start()
    print("【INFO】摄像头线程已启动")
    
    # 启动 WebSocket 广播任务
    asyncio.create_task(broadcast_worker())
    
    # 等待摄像头初始化
    await asyncio.sleep(2)
    print("【INFO】系统启动完成，等待 RTSP 连接...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
