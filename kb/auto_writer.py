import os
from datetime import datetime

KB_SOURCE_DIR = "kb/source"

def write_alarm_case_to_kb(case: dict):
    """
    将人工确认的告警样本写入知识库（Markdown）
    """
    os.makedirs(KB_SOURCE_DIR, exist_ok=True)
    
    # 修复：处理缺少 final_label 的情况
    if 'final_label' not in case:
        # 尝试从其他字段生成 final_label
        if 'alarm_level' in case:
            final_label = case['alarm_level']
        elif 'final_decision' in case and 'alarm_level' in case['final_decision']:
            final_label = case['final_decision']['alarm_level']
        else:
            final_label = "unknown"
        
        case['final_label'] = final_label
    
    # 修复：处理缺少其他必需字段的情况
    case.setdefault('scene', '公司内部')
    case.setdefault('alarm_level', '一般')
    case.setdefault('alarm_reason', '无')
    case.setdefault('model_output', '')
    case.setdefault('human_confirm', '自动生成')
    
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
该场景在公司内部安防监控中应判定为【{case.get('alarm_level', '一般')}告警】。
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"【KB】知识库案例已保存：{path}")
    return path