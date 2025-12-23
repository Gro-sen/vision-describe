# stream.py
import cv2
import numpy as np
import time
import config  # 改为导入整个模块

def generate_frames():
    from config import frame_buffer
    
    target_fps = 20
    target_frame_time = 1.0 / target_fps
    
    while True:
        start_time = time.time()
        
        # 从双缓冲读取（无锁读取）
        frame_copy = frame_buffer.read()
        
        if frame_copy is None:
            # 尝试从旧版变量读取
            with config.latest_frame_lock:
                frame_copy = config.latest_frame.copy() if config.latest_frame is not None else None
        
        if frame_copy is None:
            frame = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting...", (80, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            frame = cv2.resize(frame_copy, (480, 270))
        
        # 编码
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + 
               buf.tobytes() + b"\r\n")
        
        # 控制帧率
        processing_time = time.time() - start_time
        if processing_time < target_frame_time:
            time.sleep(target_frame_time - processing_time)