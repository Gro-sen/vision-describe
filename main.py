from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import StreamingResponse

import cv2
import base64
import asyncio
import json
import ollama
from datetime import datetime
import threading
import numpy as np
import time
import queue
import os
from playsound import playsound
from threading import Lock

# ===================== 基础配置 =====================
app = FastAPI(title="实时安防视频分析系统")

rtsp_url = "rtsp://admin:147258369GS@192.168.1.111:554/stream1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_DIR = os.path.join(BASE_DIR, "alarms")
os.makedirs(ALARM_DIR, exist_ok=True)

INFER_INTERVAL = 2.0  # 推理间隔（秒）

latest_frame = None
latest_frame_lock = threading.Lock()

inference_lock = threading.Lock()
last_infer_time = 0.0

broadcast_queue = queue.Queue()
recognition_results = []

# ===================== 告警声音 =====================
SOUND_DIR = os.path.join(BASE_DIR, "sounds")

ALARM_SOUNDS = {
    "一般": os.path.join(SOUND_DIR, "normal.mp3"),
    "严重": os.path.join(SOUND_DIR, "severe.mp3"),
    "紧急": os.path.join(SOUND_DIR, "critical.mp3"),
}

sound_lock = Lock()

# ===================== Web =====================
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


# ===================== 工具函数 =====================
def frame_to_base64(frame):
    frame = cv2.resize(frame, (640, 360))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def save_alarm_image(frame, alert_level):
    if frame is None or frame.size == 0:
        return

    level_map = {
        "一般": "normal",
        "严重": "severe",
        "紧急": "critical"
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{level_map.get(alert_level, 'unknown')}.jpg"
    path = os.path.join(ALARM_DIR, filename)

    cv2.imwrite(path, frame)
    print(f"【ALARM】图片已保存：{path}")

def play_alarm_sound(level: str):
    """
    根据告警等级播放声音（异步，不阻塞）
    """
    sound_path = ALARM_SOUNDS.get(level)
    if not sound_path or not os.path.exists(sound_path):
        return

    def _play():
        with sound_lock:   # 防止多个声音同时播
            try:
                playsound(sound_path)
            except Exception as e:
                print(f"【WARN】告警声音播放失败: {e}")

    threading.Thread(target=_play, daemon=True).start()

# ===================== 规则引擎（核心） =====================
def decide_alarm(facts: dict):
    """
    只在这里写规则！
    后续这里可以接 KB / JSON / 数据库
    """

 # ===== ① 环境级风险：与是否有人无关 =====
    if facts.get("has_fire_or_smoke") or facts.get("has_electric_risk"):
        return "是", "紧急", "检测到环境安全事故风险"

    # ===== ② 无人员，且无环境风险 =====
    if not facts.get("has_person"):
        return "否", "无", "画面中未检测到人员"

    # ===== ③ 人员进入禁区 =====
    if facts.get("enter_restricted_area"):
        if facts.get("badge_status") == "未佩戴":
            return "是", "紧急", "未佩戴工牌进入禁区"
        else:
            return "是", "严重", "人员进入禁区"

    # ===== ④ 工牌异常 =====
    if facts.get("badge_status") in ["未佩戴", "无法确认"]:
        return "是", "一般", "人员未佩戴或无法确认工牌"

    # ===== ⑤ 正常 =====
    return "否", "无", "未发现安防异常"


# ===================== 核心：模型推理 =====================
def send_to_model(frame):
    global recognition_results

    image_b64 = frame_to_base64(frame)

    # ====== ① 视觉感知（只输出事实） ======
    vision_prompt = """
你是公司内部安防系统的【视觉感知模块】。

只输出 JSON，不要解释，不要多余文字。

格式如下：
{
  "has_person": true/false,
  "badge_status": "佩戴" / "未佩戴" / "无法确认" / "不适用",
  "enter_restricted_area": true/false,
  "has_fire_or_smoke": true/false,
  "has_electric_risk": true/false,
  "scene_summary": "一句话描述画面"
}
"""

    resp = ollama.chat(
        model="qwen2.5vl:7b",
        messages=[{
            "role": "user",
            "content": vision_prompt,
            "images": [image_b64]
        }]
    )

    raw_text = resp["message"]["content"]

    try:
        facts = json.loads(raw_text)
    except Exception:
        print("【WARN】模型输出无法解析，跳过")
        return

    # ====== ② 规则判定（完全由代码控制） ======
    is_alarm, level, reason = decide_alarm(facts)

    # ====== ③ 统一输出（永远标准） ======
    output = (
        f"【检测对象】人/环境 "
        f"【是否告警】{is_alarm} "
        f"【告警等级】{level} "
        f"【告警原因】{reason} "
        f"【画面简述】{facts.get('scene_summary', '')}"
    )

    # ====== ④ 告警才存图 ======
    if is_alarm == "是" and level != "无":
        save_alarm_image(frame, level)
        play_alarm_sound(level)
    result = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": output
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
        finally:
            inference_lock.release()

    threading.Thread(target=_run, daemon=True).start()


# ===================== RTSP 捕获 =====================
def capture_rtsp():
    global latest_frame

    while True:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            time.sleep(5)
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with latest_frame_lock:
                latest_frame = frame.copy()

            if time.time() - last_infer_time >= INFER_INTERVAL:
                try_infer(frame.copy())

        cap.release()
        time.sleep(3)


# ===================== 视频流 =====================
def generate_frames():
    while True:
        with latest_frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            frame = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for RTSP...", (80, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame = cv2.resize(frame, (480, 270))
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")


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
    while True:
        try:
            result = broadcast_queue.get(timeout=1)
            await manager.broadcast(result)
            broadcast_queue.task_done()
        except queue.Empty:
            await asyncio.sleep(0.1)


# ===================== 启动 =====================
@app.on_event("startup")
async def startup():
    threading.Thread(target=capture_rtsp, daemon=True).start()
    asyncio.create_task(broadcast_worker())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
