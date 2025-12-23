from playsound import playsound
import threading
from config import ALARM_SOUNDS, sound_lock
import os

def play_alarm_sound(level: str):
    sound_path = ALARM_SOUNDS.get(level)
    if not sound_path or not os.path.exists(sound_path):
        return

    def _play():
        with sound_lock:
            try:
                playsound(sound_path)
            except Exception as e:
                print(f"【WARN】告警声音播放失败: {e}")

    threading.Thread(target=_play, daemon=True).start()
