def decide_alarm(facts: dict):
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
