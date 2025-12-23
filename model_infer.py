import cv2
import base64
import json
import threading
import time
from datetime import datetime
import ollama

from config import latest_frame_lock, broadcast_queue, recognition_results, inference_lock, last_infer_time
from rules import decide_alarm
from sound import play_alarm_sound
from config import ALARM_DIR
import os

def frame_to_base64(frame):
    frame = cv2.resize(frame, (640, 360))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()

def save_alarm_image(frame, alert_level):
    if frame is None or frame.size == 0:
        return
    level_map = {"一般": "normal", "严重": "severe", "紧急": "critical"}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{level_map.get(alert_level, 'unknown')}.jpg"
    path = os.path.join(ALARM_DIR, filename)
    cv2.imwrite(path, frame)
    print(f"【ALARM】图片已保存：{path}")

def send_to_model(frame):
    global recognition_results

    image_b64 = frame_to_base64(frame)

    # ====== 视觉大模型 ======
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
        messages=[{"role": "user", "content": vision_prompt, "images": [image_b64]}]
    )

    raw_text = resp["message"]["content"]
    try:
        facts = json.loads(raw_text)
    except Exception:
        print("【WARN】模型输出无法解析，跳过")
        return

    # ====== 规则判定 ======
    is_alarm, level, reason = decide_alarm(facts)

    output = (
        f"【检测对象】人/环境 "
        f"【是否告警】{is_alarm} "
        f"【告警等级】{level} "
        f"【告警原因】{reason} "
        f"【画面简述】{facts.get('scene_summary', '')}"
    )

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

def try_infer(frame, last_infer_time_ref):
    """
    frame: 当前帧
    last_infer_time_ref: [last_infer_time] 形式的列表，确保线程能更新
    """
    from threading import Lock
    from config import inference_lock

    if not inference_lock.acquire(blocking=False):
        return

    last_infer_time_ref[0] = time.time()  # 更新全局推理时间

    def _run():
        try:
            send_to_model(frame)
        finally:
            inference_lock.release()

    threading.Thread(target=_run, daemon=True).start()

