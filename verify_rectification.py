"""
Stereo Rectification Verification Script
=========================================
顯示 [上方] 原始左右影像 / [下方] Rectification 後的左右影像
並在兩列畫面上繪製等間距水平對極線，方便人工比較對齊狀況。

使用方式：
    python verify_rectification.py [--config config/pipeline_config.yaml]
    python verify_rectification.py --calib config/calibration_params.yml \
                                   --stream http://localhost:5000/video_combined \
                                   --width 640 --height 480

操作說明：
    [S]      - 從串流抓一幀並顯示比較圖
    [Space]  - 連續抓幀（暫停後按 Space 繼續）
    [Q/ESC]  - 離開
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Allow importing project modules
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pipeline.stereo_rectifier import StereoRectifier


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def draw_epipolar_lines(img: np.ndarray, n_lines: int = 15, color=(0, 255, 0)) -> np.ndarray:
    """在影像上繪製等間距水平對極線 (用於檢查 rectification 是否對齊)。"""
    out = img.copy()
    h, w = out.shape[:2]
    for i in range(1, n_lines + 1):
        y = int(i * h / (n_lines + 1))
        cv2.line(out, (0, y), (w, y), color, 1)
    return out


def build_comparison(img_l_raw, img_r_raw, img_l_rect, img_r_rect) -> np.ndarray:
    """
    組合比較畫面：
        上列：原始左 | 原始右      ← 紅色對極線
        下列：校正左 | 校正右      ← 綠色對極線
    兩列之間加一條白色分隔線。
    """
    # 繪製對極線
    raw_l  = draw_epipolar_lines(img_l_raw,  color=(0, 80, 255))   # 紅色（原始）
    raw_r  = draw_epipolar_lines(img_r_raw,  color=(0, 80, 255))
    rect_l = draw_epipolar_lines(img_l_rect, color=(0, 200, 60))    # 綠色（校正後）
    rect_r = draw_epipolar_lines(img_r_rect, color=(0, 200, 60))

    # 加標籤
    def label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    raw_l  = label(raw_l,  "Original LEFT")
    raw_r  = label(raw_r,  "Original RIGHT")
    rect_l = label(rect_l, "Rectified LEFT")
    rect_r = label(rect_r, "Rectified RIGHT")

    top_row    = np.hstack([raw_l,  raw_r])
    bottom_row = np.hstack([rect_l, rect_r])

    # 白色分隔線
    sep = np.full((6, top_row.shape[1], 3), 220, dtype=np.uint8)
    canvas = np.vstack([top_row, sep, bottom_row])
    return canvas


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stereo Rectification Verifier")
    parser.add_argument("--config", default="config/pipeline_config.yaml",
                        help="Pipeline YAML config (preferred)")
    parser.add_argument("--calib",  default=None,
                        help="Override: path to calibration YAML")
    parser.add_argument("--stream", default=None,
                        help="Override: video stream URL or device index")
    parser.add_argument("--width",  type=int, default=None, help="Override: frame width")
    parser.add_argument("--height", type=int, default=None, help="Override: frame height")
    parser.add_argument("--lines",  type=int, default=15,   help="Number of epipolar lines")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────
    cfg = {}
    if Path(args.config).exists():
        cfg = load_config(args.config)
        print(f"[verify] 已載入設定檔: {args.config}")
    else:
        print(f"[verify] 找不到設定檔 ({args.config})，使用命令列參數")

    calib_path  = args.calib  or cfg.get("camera", {}).get("calibration_path",
                               "config/calibration_params.yml")
    stream_url  = args.stream or cfg.get("stream", {}).get("url",
                               "http://localhost:5000/video_combined")
    width       = args.width  or cfg.get("camera", {}).get("width",  640)
    height      = args.height or cfg.get("camera", {}).get("height", 480)

    print(f"[verify] 校正檔案  : {calib_path}")
    print(f"[verify] 串流來源  : {stream_url}")
    print(f"[verify] 影像尺寸  : {width}x{height}")

    # ── Init Rectifier ────────────────────────────────
    rectifier = StereoRectifier(calib_path, image_size=(width, height))

    # ── Open stream ──────────────────────────────────
    # Support numeric device index as string ("0", "1", …)
    src = int(stream_url) if str(stream_url).isdigit() else stream_url
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[verify] ✗ 無法開啟串流: {stream_url}")
        sys.exit(1)

    print("\n操作說明:")
    print("  [S]      - 抓取當前幀並顯示比較圖")
    print("  [Space]  - 暫停 / 繼續連續更新")
    print("  [Q/ESC]  - 結束\n")

    paused = False
    canvas = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[verify] 串流中斷，等待中...")
                cv2.waitKey(500)
                continue

            h, w = frame.shape[:2]
            img_l = frame[:, : w // 2]
            img_r = frame[:, w // 2 :]

            # Resize to calibrated resolution if needed
            if img_l.shape[1] != width or img_l.shape[0] != height:
                img_l = cv2.resize(img_l, (width, height))
                img_r = cv2.resize(img_r, (width, height))

            img_l_rect, img_r_rect = rectifier.rectify(img_l, img_r)
            canvas = build_comparison(img_l, img_r, img_l_rect, img_r_rect)

        if canvas is not None:
            win_name = "Rectification Verification  [S=snap | Space=pause | Q=quit]"
            cv2.imshow(win_name, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):           # Q or ESC
            break
        elif key == ord(' '):               # Space → toggle pause
            paused = not paused
            print("[verify]", "暫停" if paused else "繼續")
        elif key == ord('s') and canvas is not None:
            save_path = "rectification_snapshot.png"
            cv2.imwrite(save_path, canvas)
            print(f"[verify] ✓ 已儲存比較圖: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[verify] 結束")


if __name__ == "__main__":
    main()
