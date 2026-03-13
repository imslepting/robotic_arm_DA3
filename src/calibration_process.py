import cv2
import time
import yaml
import numpy as np
import subprocess
from camera_unit.calibrator import StereoCalibrator

def load_config():
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_windows_host_ip():
    """獲取 WSL2 的 Windows 主機 IP"""
    return "localhost" # 如果失敗，請填入你手動查到的 IP

def main():
    config = load_config()
    calibrator = StereoCalibrator(config)
    
    # --- 調整為 Flask 串流模式 ---
    win_ip = get_windows_host_ip()
    stream_url = f"http://{win_ip}:5000/video_combined"
    print(f"正在連線至串流伺服器: {stream_url}")
    
    cap = cv2.VideoCapture(stream_url)
    # 增加緩衝區大小以減少延遲（可選）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

    if not cap.isOpened():
        print("錯誤：無法連線至 Flask 伺服器，請確保 Windows 端的 win_cam.py 已執行")
        return

    num_required = config['calibration']['num_samples_required']
    captured_count = 0
    
    print(f"準備開始校正。需要拍攝 {num_required} 組成功的影像。")
    print("操作說明: [S] 捕捉影像 | [Q] 退出並開始計算參數")

    try:
        while captured_count < num_required:
            ret, frame = cap.read()
            if not ret:
                print("等待串流數據...")
                time.sleep(0.1)
                continue

            # 取得 Flask 傳來的合併畫面 (Side-by-Side)
            h, w, _ = frame.shape
            # 切分左右畫面
            img_l = frame[:, :w//2]
            img_r = frame[:, w//2:]

            # 顯示與 UI 提示
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Captured: {captured_count}/{num_required}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration Process", display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            # 手動觸發捕捉邏輯
            if key == ord('s'):
                print(f"正在分析第 {captured_count + 1} 張...")
                # 這裡調用你原本的 calibrator 邏輯
                found, corners = calibrator.add_corners(img_l, img_r)
                
                if found:
                    captured_count += 1
                    print("成功偵測到角點！")
                    # 畫面閃爍提示
                    cv2.rectangle(display_frame, (0,0), (w,h), (0,255,0), 20)
                    cv2.imshow("Calibration Process", display_frame)
                    cv2.waitKey(500)
                else:
                    print("失敗：兩台相機必須同時看清楚棋盤格。")

            elif key == ord('q'):
                break

        # 收集完畢後執行計算
        if captured_count >= 1: # 即使沒滿，只要有數據按下 q 也可以嘗試計算
            print("\n數據收集完畢，開始計算內外參數...")
            # 取得單個相機的原始解析度
            img_shape = (config['camera']['width'], config['camera']['height'])
            results = calibrator.calibrate(img_shape)
            
            # 儲存結果
            calibrator.save_results(results, config['calibration']['save_path'])
            print(f"校正完成！結果已儲存至 {config['calibration']['save_path']}")
            print(f"兩相機間的距離 (Baseline): {results.get('T', 'N/A')} mm")
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()