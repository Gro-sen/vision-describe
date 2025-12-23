# config.py
import os
import threading
import queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建必要的目录
ALARM_DIR = os.path.join(BASE_DIR, "alarms")
os.makedirs(ALARM_DIR, exist_ok=True)

SOUND_DIR = os.path.join(BASE_DIR, "sounds")
os.makedirs(SOUND_DIR, exist_ok=True)

# 报警声音配置
ALARM_SOUNDS = {
    "一般": os.path.join(SOUND_DIR, "normal.mp3"),
    "严重": os.path.join(SOUND_DIR, "severe.mp3"),
    "紧急": os.path.join(SOUND_DIR, "critical.mp3"),
}

# RTSP 流地址
RTSP_URL = "rtsp://admin:147258369GS@192.168.1.111:554/stream1"

# 推理参数
INFER_INTERVAL = 2.0  # 秒

# 全局状态变量 - 使用一个类来确保引用一致性
class GlobalState:
    def __init__(self):
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.last_infer_time = 0.0
        self.broadcast_queue = queue.Queue()
        self.recognition_results = []
        self.sound_lock = threading.Lock()

class DoubleBuffer:
    """双缓冲类，减少锁竞争"""
    def __init__(self):
        self.front_buffer = None  # 前端缓冲（读取）
        self.back_buffer = None   # 后端缓冲（写入）
        self.buffer_lock = threading.Lock()
    
    def write(self, frame):
        """写入新帧到后端缓冲"""
        with self.buffer_lock:
            self.back_buffer = frame
    
    def read(self):
        """读取前端缓冲的帧，并交换缓冲区"""
        with self.buffer_lock:
            if self.front_buffer is not None:
                return self.front_buffer.copy()
            return None
    
    def swap(self):
        """交换前后缓冲区（由摄像头线程调用）"""
        with self.buffer_lock:
            self.front_buffer = self.back_buffer

# 创建全局状态实例
state = GlobalState()
frame_buffer = DoubleBuffer()
# 为了方便，也导出所有属性
latest_frame = state.latest_frame
latest_frame_lock = state.latest_frame_lock
inference_lock = state.inference_lock
last_infer_time = state.last_infer_time
broadcast_queue = state.broadcast_queue
recognition_results = state.recognition_results
sound_lock = state.sound_lock