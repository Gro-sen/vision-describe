from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import threading
import time
from datetime import datetime

# 正确导入 config 模块中的变量
import config

# 从 config 中导入需要的变量
from config import (
    broadcast_queue, 
    recognition_results,
    RTSP_URL, 
    INFER_INTERVAL,
    latest_frame,
    latest_frame_lock,
    inference_lock,
    last_infer_time,
    sound_lock
)

from camera import capture_rtsp
from stream import generate_frames

# 尝试导入知识库API，如果失败则提供替代方案
try:
    from kb.api import router as kb_router
    KB_AVAILABLE = True
except ImportError as e:
    print(f"【WARN】知识库API导入失败: {e}")
    KB_AVAILABLE = False
    
    # 创建简单的替代路由
    from fastapi import APIRouter
    kb_router = APIRouter(prefix="/kb", tags=["knowledge-base"])
    
    @kb_router.get("/stats")
    async def get_kb_stats():
        return {"status": "knowledge base not available"}

# ===================== FastAPI应用创建 =====================
app = FastAPI(title="智能安防视频分析系统")

# ===================== WebSocket 管理 =====================
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"【WS】客户端连接，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"【WS】客户端断开，剩余连接数: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for ws in list(self.active_connections):
            try:
                await ws.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                print(f"【WS】发送失败: {e}")
                disconnected.append(ws)
        
        for ws in disconnected:
            self.disconnect(ws)

manager = ConnectionManager()

# ===================== WebSocket路由 =====================
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    print("【WS】收到连接请求")
    await manager.connect(websocket)
    try:
        while True:
            # 接收消息（可以是心跳或其他数据）
            data = await websocket.receive_text()
            print(f"【WS】收到消息: {data}")
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        print("【WS】客户端断开连接")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"【WS】连接异常: {e}")
        manager.disconnect(websocket)

# ===================== Web路由 =====================
# 挂载静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ===================== API 端点 =====================
@app.get("/api/alarms/history")
async def get_alarm_history(limit: int = 50):
    """获取历史报警记录"""
    from config import recognition_results
    return recognition_results[-limit:]

@app.get("/api/system/status")
async def get_system_status():
    """获取系统状态"""
    import psutil
    import time
    from kb import kb
    
    status = {
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "uptime": time.time() - psutil.boot_time(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "knowledge_base": kb.get_statistics(),
        "alarms_today": len([r for r in recognition_results 
                           if r.get("is_alarm") == "是"])
    }
    return status

@app.get("/api/models/info")
async def get_model_info():
    """获取模型信息"""
    from config import model_config
    from reasoning_model import reasoning_model
    
    info = {
        "vision_model": model_config.VISION_MODEL,
        "reasoning_model": reasoning_model.model_name,
        "kb_retrieval_top_k": model_config.KB_RETRIEVAL_TOP_K,
        "kb_similarity_threshold": model_config.KB_SIMILARITY_THRESHOLD
    }
    return info

# ===================== 广播工作者 =====================
async def broadcast_worker():
    """WebSocket广播任务"""
    print("【WS】广播工作者已启动")
    while True:
        try:
            # 从队列获取结果（阻塞，最多1秒）
            result = broadcast_queue.get(timeout=1)
            print(f"【WS】准备广播: {result.get('alarm_level', '无')}报警")
            
            # 广播给所有连接的客户端
            await manager.broadcast(result)
            
            # 标记任务完成
            broadcast_queue.task_done()
            print(f"【WS】广播完成")
            
        except Exception as e:
            # 队列为空时正常等待
            await asyncio.sleep(0.1)

# ===================== 系统启动 =====================
@app.on_event("startup")
async def startup():
    """系统启动事件"""
    print("="*50)
    print("系统启动中...")
    
    # 打印所有路由
    print("\n注册的路由:")
    for route in app.routes:
        if hasattr(route, "path"):
            print(f"  - {route.path}")
    
    print("="*50)
    
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
    print("【INFO】系统启动完成")

# ===================== 注册知识库API路由 =====================
# 注意：这部分放在最后，避免影响其他路由
if KB_AVAILABLE:
    app.include_router(kb_router)
    print("✓ 知识库API已启用")
else:
    print("⚠ 知识库API未启用")

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("启动FastAPI服务器...")
    print(f"访问地址: http://localhost:8000")
    print(f"WebSocket地址: ws://localhost:8000/ws")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)