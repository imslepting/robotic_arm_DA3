import cv2

# 先試試看 MX Brio (通常是 0)
camera_index = 0 
cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

# 強制設定格式，避免 WSL2 timeout
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
if ret:
    print(f"成功！從 /dev/video{camera_index} 取得影像，解析度為: {frame.shape}")
    # 如果你有安裝 GUI 轉發，可以試試 cv2.imshow('test', frame)
else:
    print(f"失敗，請更換 camera_index 試試（例如 2）")

cap.release()