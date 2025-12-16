from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from kb.auto_writer import write_alarm_case_to_kb
import cv2
import base64
import asyncio
import json
import ollama
from datetime import datetime
from typing import List
from kb import indexing as kb_indexing
from kb import retriever as kb_retriever
import threading
import numpy as np
import time
import queue
import os
app = FastAPI(title="实时安防视频分析系统")

# ===================== 基础配置 =====================
rtsp_url = "rtsp://admin:147258369GS@192.168.1.111:554/stream1"
latest_frame = None
latest_frame_lock = threading.Lock()
VALID_LEVELS = ["一般", "严重", "紧急"]
inference_lock = threading.Lock()
last_infer_time = 0.0
INFER_INTERVAL = 2.0  # 秒
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_DIR = os.path.join(BASE_DIR, "alarms")
os.makedirs(ALARM_DIR, exist_ok=True)
recognition_results = []

broadcast_queue = queue.Queue()

ALARM_IMAGE_DIR = "alarms"
os.makedirs(ALARM_IMAGE_DIR, exist_ok=True)

# ===================== Web =====================
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ===================== WebSocket 管理 =====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_text(json.dumps(message, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ===================== 工具函数 =====================
def frame_to_base64(frame):
    frame = cv2.resize(frame, (640, 360))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def kb_query_local(text, top_k=3):
    try:
        return kb_retriever.query(
            text,
            top_k,
            "kb/index/faiss.index",
            "kb/index/docs.pkl"
        )
    except Exception:
        return []

def save_alarm_image(frame, alert_level):
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        print("【ERROR】非法 frame，跳过保存")
        return

    LEVEL_MAP = {
        "无": "none",
        "一般": "normal",
        "严重": "severe",
        "紧急": "critical"
    }

    safe_level = LEVEL_MAP.get(alert_level, "unknown")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{safe_level}.jpg"
    path = os.path.join(ALARM_DIR, filename)

    ok = cv2.imwrite(path, frame)
    if ok:
        print(f"【DEBUG】告警图片真实写入成功: {path}")
    else:
        print(f"【ERROR】告警图片写入失败: {path}")


# ===================== 核心：模型推理 =====================
def send_to_model(frame):
    global recognition_results

    image_b64 = frame_to_base64(frame)

    # ========= 第一阶段：事实判断（允许模型啰嗦） =========
    vision_prompt = """
你是一名【公司内部安防监控系统】视觉分析模块。

只做事实判断，不要下最终结论。

请判断：
1. 是否有人
2. 人员是否佩戴工牌（明确 / 未佩戴 / 无法确认）
3. 是否进入禁区
4. 是否存在明火、烟雾、触电风险

如无法确认，必须明确写“无法确认”。
"""

    vision_resp = ollama.chat(
        model="qwen2.5vl:7b",
        messages=[{
            "role": "user",
            "content": vision_prompt,
            "images": [image_b64]
        }]
    )

    vision_text = vision_resp["message"]["content"]

    # ========= 第二阶段：格式裁判（只允许一行） =========
    judge_prompt = f"""
你是安防系统【裁判模块】。

根据【事实描述】严格生成【一行输出】，禁止换行，禁止解释。

【事实描述】
{vision_text}

【强制规则】
- 只要出现“未佩戴工牌”或“无法确认是否佩戴工牌” → 必须告警
- 未佩戴工牌也未踏入禁区 → 一般
- 已佩戴工牌踏入禁区 → 严重
- 未佩戴工牌踏入禁区 → 紧急
- 出现明火、烟雾、触电风险 → 紧急
- 其他情况一律不告警

【唯一允许的输出格式（一行）】
【检测对象】人/物/环境 【是否告警】是/否 【告警等级】无/一般/严重/紧急 【告警原因】XXX 【画面简述】XXX
"""

    final_resp = ollama.chat(
        model="qwen2.5vl:7b",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    base_output = final_resp["message"]["content"].strip()

    # ========= 代码级兜底（关键） =========
    import re

    # 1️⃣ 只要不是标准一行格式 → 强制告警
    if "\n" in base_output or "【是否告警】" not in base_output:
        alert_level = "一般"
        save_alarm_image(frame, alert_level)
    else:
        # 2️⃣ 正常解析
        m_level = re.search(r"告警等级】\s*([^\s【]+)", base_output)
        m_alarm = re.search(r"是否告警】\s*(是|否)", base_output)

        alert_level = m_level.group(1) if m_level else "一般"
        is_alarm = m_alarm.group(1) if m_alarm else "是"

        # 3️⃣ 任何“是” → 存图
        if is_alarm == "是" and alert_level in ["一般", "严重", "紧急"]:
            save_alarm_image(frame, alert_level)

    # ========= RAG（辅助） =========
    kb_hits = kb_query_local(base_output)
    rag_used = bool(kb_hits)

    result = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": base_output,
        "rag_used": rag_used
    }

    recognition_results.append(result)
    if len(recognition_results) > 50:
        recognition_results.pop(0)

    broadcast_queue.put(result)

  

def try_infer(frame):
    global last_infer_time
    if not inference_lock.acquire(blocking=False):
        return

    last_infer_time = time.time()

    def _run():
        try:
            send_to_model(frame)
        except Exception as e:
            print("推理异常:", e)
        finally:
            inference_lock.release()

    threading.Thread(target=_run, daemon=True).start()


# ===================== RTSP 捕获 =====================
def capture_rtsp():
    global latest_frame

    while True:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("RTSP 打开失败，5 秒重试")
            time.sleep(5)
            continue

        print("RTSP 已连接")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("RTSP 断开，重连中")
                break

            with latest_frame_lock:
                latest_frame = frame.copy()

            if time.time() - last_infer_time >= INFER_INTERVAL:
                with latest_frame_lock:
                    snap = latest_frame.copy()
                try_infer(snap)

        cap.release()
        time.sleep(5)


# ===================== 视频流 =====================
def generate_frames():
    while True:
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            
            try:
                frame = cv2.zeros((270, 480, 3), dtype=np.uint8)
            except Exception:
                frame = np.zeros((270, 480, 3), dtype=np.uint8)

            cv2.putText(
                frame,
                "Waiting for RTSP...",
                (80, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        frame = cv2.resize(frame, (480, 270))
        _, buf = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

async def broadcast_worker():
    print("【DEBUG】broadcast_worker 启动")
    while True:
        try:
            result = broadcast_queue.get(timeout=1)
            print("【DEBUG】broadcast_worker 取到结果，广播中")
            await manager.broadcast(result)
            broadcast_queue.task_done()
        except queue.Empty:
            await asyncio.sleep(0.1)
        except Exception as e:
            print("【broadcast_worker 异常】", e)
            await asyncio.sleep(0.5)

# ===================== 路由 =====================
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


@app.post("/kb/index")
async def kb_index():
    return await asyncio.to_thread(
        kb_indexing.index_documents,
        "kb/source",
        "kb/index/faiss.index",
        "kb/index/docs.pkl"
    )

@app.post("/alarm/confirm")
async def confirm_alarm(case: dict):
    """
    人工确认告警 → 自动写入 KB
    """
    kb_path = write_alarm_case_to_kb(case)

    return {
        "status": "ok",
        "kb_written": kb_path
    }

# ===================== 启动 =====================
@app.on_event("startup")
async def startup():
    print("【DEBUG】FastAPI startup 执行")
    threading.Thread(target=capture_rtsp, daemon=True).start()
    asyncio.create_task(broadcast_worker())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
