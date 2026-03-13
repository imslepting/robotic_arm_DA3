import cv2
import numpy as np

# 替換為你的 Windows 實體 IP 或 localhost
win_ip = "localhost" 
stream_url = f"http://{win_ip}:5000/video_combined"

cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取合併串流")
        break

    # 取得寬度並切成兩半
    height, width = frame.shape[:2]
    mid = width // 2
    
    frame_mxbrio = frame[:, :mid]      # 左半邊：MX Brio
    frame_realsense = frame[:, mid:]    # 右半邊：RealSense

    # --- 在此處加入 Depth-Anything-3 處理 ---
    # 例如：result = model.infer(frame_mxbrio)
    
    # 顯示
    cv2.imshow('Combined Stream (Left: MX Brio | Right: RealSense)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()