# camera.py
import cv2
import time
import threading
import config  # 改为导入整个模块
from model_infer import try_infer

def capture_rtsp():
    global config
    
    last_infer_time_ref = [config.last_infer_time]
    frame_skip = 0  # 帧跳过计数器
    
    while True:
        cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            time.sleep(5)
            continue

        print("【SUCCESS】RTSP 连接成功")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ==== 优化2：帧跳过 ====
            # 每2帧处理1帧，减少处理负担
            frame_skip = (frame_skip + 1) % 2
            if frame_skip != 0:
                continue
            
            # 降低分辨率
            resized_frame = cv2.resize(frame, (640, 360))
            
            # 写入到双缓冲的后端
            config.frame_buffer.write(resized_frame.copy())
            
            # 交换缓冲区
            config.frame_buffer.swap()
            
            # 更新旧版兼容变量
            with config.latest_frame_lock:
                config.latest_frame = resized_frame.copy()
            
            # 推理
            current_time = time.time()
            if current_time - last_infer_time_ref[0] >= config.INFER_INTERVAL:
                try_infer(resized_frame.copy(), last_infer_time_ref)

        cap.release()