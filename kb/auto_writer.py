import os
from datetime import datetime

KB_SOURCE_DIR = "kb/source"

def write_alarm_case_to_kb(case: dict):
    """
    将人工确认的告警样本写入知识库（Markdown）
    """
    os.makedirs(KB_SOURCE_DIR, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{date_str}_{case['final_label']}.md"
    path = os.path.join(KB_SOURCE_DIR, filename)

    content = f"""# 安防告警案例：{case['final_label']}

## 场景
{case.get("scene", "公司内部")}

## 告警等级
{case.get("alarm_level", "一般")}

## 触发原因
{case.get("alarm_reason", "无")}

## 画面特征
- {case.get("model_output", "").strip()}

## 人工确认
{case.get("human_confirm", "已确认")}

## 结论
该场景在公司内部安防监控中应判定为【{case['alarm_level']}告警】。
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path
