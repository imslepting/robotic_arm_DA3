import yaml
import cv2
from camera_unit.camera_handler import DualCameraHandler

def load_config(path="config/config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. 載入設定
    config = load_config()
    
    # 2. 初始化相機控制項
    handler = DualCameraHandler(config)
    
    if not handler.is_ready():
        print("錯誤：無法開啟指定的相機，請檢查索引值與連接。")
        return

    print("雙相機系統啟動成功！")
    win_name = config['window']['name']
    quit_key = config['window']['quit_key']

    try:
        while True:
            success, combined_frame = handler.get_combined_frame()
            
            if success:
                cv2.imshow(win_name, combined_frame)
            else:
                print("讀取影格失敗。")
                break

            # 按下指定鍵退出
            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                break
    finally:
        # 3. 確保資源釋放
        handler.release()
        print("系統已關閉。")

if __name__ == "__main__":
    main()