import os
import sys
import time
from datetime import datetime

def initialize_system():
    """系统初始化"""
    print("=" * 60)
    print("智能安防系统初始化")
    print("=" * 60)
    
    # 检查目录结构
    required_dirs = [
        "alarms",
        "sounds",
        "kb/source",
        "kb/index",
        "kb/cases",
        "static/css",
        "static/js"
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 确保目录存在: {dir_path}")
    
    # 检查声音文件
    sound_files = ["normal.mp3", "severe.mp3", "critical.mp3"]
    for sound_file in sound_files:
        path = os.path.join("sounds", sound_file)
        if not os.path.exists(path):
            print(f"⚠ 警告: 声音文件不存在 - {path}")
            # 创建空文件作为占位符
            with open(path, 'wb') as f:
                f.write(b'')
    
    # 检查知识库索引
    index_file = "kb/index/faiss.index"
    if os.path.exists(index_file):
        print("✓ 知识库索引已存在")
    else:
        print("ℹ 知识库索引不存在，将在第一次查询时创建")
    
    # 检查模型可用性
    print("\n检查模型可用性...")
    try:
        import ollama
        models = ollama.list()
        installed_models = [m['name'] for m in models['models']]
        
        required_models = ["qwen2.5vl:7b", "deepseek-r1:1.5b"]
        for model in required_models:
            if model in installed_models:
                print(f"✓ 模型已安装: {model}")
            else:
                print(f"⚠ 模型未安装: {model}")
                print(f"  请运行: ollama pull {model}")
    except Exception as e:
        print(f"⚠ 检查模型时出错: {e}")
    
    # 初始化知识库索引
    try:
        from kb import kb
        stats = kb.get_statistics()
        if stats["index_exists"]:
            print(f"✓ 知识库索引状态正常")
        else:
            print("ℹ 知识库索引需要重建")
            if input("是否现在重建索引? (y/n): ").lower() == 'y':
                result = kb.update_index()
                print(f"重建结果: {result}")
    except Exception as e:
        print(f"⚠ 初始化知识库时出错: {e}")
    
    print("\n" + "=" * 60)
    print("系统初始化完成")
    print("启动命令: python main.py")
    print("访问地址: http://localhost:8000")
    print("=" * 60)

if __name__ == "__main__":
    initialize_system()