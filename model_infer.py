import cv2
import base64
import json
import threading
import time
from datetime import datetime
import ollama
from config import latest_frame_lock, broadcast_queue, recognition_results, inference_lock, last_infer_time
from sound import play_alarm_sound
from config import ALARM_DIR
import os

# 导入新模块
from reasoning_model import reasoning_model
from kb import kb

def frame_to_base64(frame):
    """将帧转换为Base64编码"""
    frame = cv2.resize(frame, (640, 360))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()

def save_alarm_image(frame, alert_level, case_id=None):
    """保存报警图片"""
    if frame is None or frame.size == 0:
        return
    
    level_map = {"一般": "normal", "严重": "severe", "紧急": "critical"}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if case_id:
        filename = f"{ts}_{level_map.get(alert_level, 'unknown')}_{case_id}.jpg"
    else:
        filename = f"{ts}_{level_map.get(alert_level, 'unknown')}.jpg"
    
    path = os.path.join(ALARM_DIR, filename)
    cv2.imwrite(path, frame)
    print(f"【ALARM】图片已保存：{path}")
    return path

def vision_model_analysis(frame):
    """视觉大模型分析（第一阶段）"""
    image_b64 = frame_to_base64(frame)
    
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
  "scene_summary": "一句话描述画面",
  "object_details": {
    "person_count": 数量,
    "person_positions": ["位置描述"],
    "environment_status": "环境状态描述"
  }
}
"""
    resp = ollama.chat(
        model="qwen2.5vl:7b",
        messages=[{"role": "user", "content": vision_prompt, "images": [image_b64]}]
    )
    
    raw_text = resp["message"]["content"]
    try:
        facts = json.loads(raw_text)
    except Exception as e:
        print(f"【WARN】视觉模型输出无法解析: {e}")
        return None
    
    return facts

def send_to_model(frame):
    """完整的模型推理流程（双模型）"""
    global recognition_results
    
    # ====== 第一阶段：视觉模型分析 ======
    print("【INFO】第一阶段：视觉模型分析中...")
    vision_facts = vision_model_analysis(frame)
    
    if vision_facts is None:
        print("【ERROR】视觉分析失败，跳过本次推理")
        return
    
    # 记录视觉分析结果
    print(f"【DEBUG】视觉分析结果: {json.dumps(vision_facts, ensure_ascii=False)}")
    
    # ====== 第二阶段：推理模型分析 ======
    print("【INFO】第二阶段：推理模型分析中...")
    try:
        reasoning_result = reasoning_model.infer(vision_facts)
        
        # 记录推理结果
        with open("reasoning_debug.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"时间: {datetime.now().isoformat()}\n")
            f.write(f"视觉事实: {json.dumps(vision_facts, ensure_ascii=False)}\n")
            f.write(f"推理结果: {json.dumps(reasoning_result, ensure_ascii=False, indent=2)}\n")
            
    except Exception as e:
        print(f"【ERROR】推理模型异常: {e}")
        import traceback
        traceback.print_exc()
        # 使用后备决策
        reasoning_result = reasoning_model.get_fallback_decision(vision_facts)
    
    final_decision = reasoning_result.get("final_decision", {})
    analysis = reasoning_result.get("analysis", {})
    metadata = reasoning_result.get("metadata", {})
    
    # ====== 报警处理 ======
    is_alarm = final_decision.get("is_alarm", "否")
    alarm_level = final_decision.get("alarm_level", "无")
    alarm_reason = final_decision.get("alarm_reason", "")
    
    case_id = None
    
    if is_alarm == "是" and alarm_level != "无":
        try:
            # 保存报警图片
            image_path = save_alarm_image(frame, alarm_level)
            
            # 准备案例数据
            case_data = {
                "final_label": f"{alarm_level}_alarm",
                "scene": vision_facts.get('scene_summary', ''),
                "alarm_level": alarm_level,
                "alarm_reason": alarm_reason,
                "model_output": str(vision_facts),
                "human_confirm": "自动生成",
                # 添加额外信息
                "vision_facts": vision_facts,
                "final_decision": final_decision,
                "analysis": analysis,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "model_used": metadata.get("model", "unknown")
            }
            
            # 保存到知识库
            case_id = kb.add_case(case_data)
            print(f"【INFO】案例已保存到知识库：{case_id}")
            
        except Exception as e:
            print(f"【WARN】保存案例到知识库失败: {e}")
            case_id = None
        
        # 播放报警声音
        play_alarm_sound(alarm_level)
        
        print(f"【ALARM】{alarm_level}级报警：{alarm_reason}")
    
    # ====== 构建输出 ======
    output = {
        "vision_analysis": vision_facts.get("scene_summary", ""),
        "is_alarm": is_alarm,
        "alarm_level": alarm_level,
        "alarm_reason": alarm_reason,
        "confidence": final_decision.get("confidence", 0.0),
        "risk_assessment": analysis.get("risk_assessment", ""),
        "recommendation": analysis.get("recommendation", ""),
        "kb_cases_used": metadata.get("kb_cases_used", 0),
        "case_id": case_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": metadata.get("model", "unknown")
    }
    
    # 保存到历史结果
    recognition_results.append(output)
    if len(recognition_results) > 100:
        recognition_results.pop(0)
    
    # 广播到WebSocket
    broadcast_queue.put(output)
    
    # 打印结果
    print(f"【RESULT】报警决策: {is_alarm} ({alarm_level}) - {alarm_reason}")
    print(f"【RESULT】置信度: {final_decision.get('confidence', 0.0):.2f}")
    print(f"【RESULT】使用模型: {metadata.get('model', 'unknown')}")
    kb_used = output.get("kb_cases_used", 0)
    if kb_used > 0:
        print(f"【RESULT】参考了 {kb_used} 个历史案例")

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

