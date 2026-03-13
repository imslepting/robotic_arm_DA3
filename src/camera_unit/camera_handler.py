import cv2
import numpy as np

class DualCameraHandler:
    def __init__(self, config):
        self.indices = config['camera']['indices']
        self.width = config['camera']['width']
        self.height = config['camera']['height']
        
        # 根據設定選擇 API (預設使用 DSHOW)
        api = getattr(cv2, config['camera'].get('api_preference', 'CAP_ANY'))
        
        self.caps = [cv2.VideoCapture(idx, api) for idx in self.indices]
        self._setup_cameras()

    def _setup_cameras(self):
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def is_ready(self):
        return all(cap.isOpened() for cap in self.caps)

    def get_combined_frame(self):
        frames = []
        for cap in self.caps:
            ret, frame = cap.read()
            if not ret:
                return False, None
            frames.append(frame)
        
        # 水平拼接所有畫面
        combined = np.hstack(frames)
        return True, combined

    def release(self):
        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()