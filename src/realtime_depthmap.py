#!/usr/bin/env python3
"""
DA3 Real-time Depth Map Preview.
Reads pipeline_config.yaml, initializes the streaming and DA3 pipeline,
and opens an OpenCV window displaying the original RGB stream and 
its corresponding depth map side-by-side.
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline.frame_buffer import CircularFrameBuffer
from pipeline.stereo_rectifier import StereoRectifier
from pipeline.pose_manager import PoseManager
from pipeline.inference_engine import InferenceEngine
from pipeline.depth_decoder import DepthDecoder

# ─── Global shutdown flag ─────────────────────────────────────
_shutdown = threading.Event()


def signal_handler(sig, frame):
    print("\n[Pipeline] 收到中斷信號，正在關閉...")
    _shutdown.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─── Thread 1: Capture ────────────────────────────────────────
def capture_thread(
    stream_url: str,
    rectifier,
    frame_buffer: CircularFrameBuffer,
    buffer_size: int = 1,
    enable_rectification: bool = True,
):
    """Read frames from stream, rectify if enabled, and push to buffer."""
    print(f"[Capture] 正在連線至串流: {stream_url}")
    if enable_rectification:
        print("[Capture] 立體校正模式: 啟用 ✅")
    else:
        print("[Capture] 立體校正模式: 停用 ⚠️  (使用原始幀)")

    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    if not cap.isOpened():
        print("[Capture] ❌ 無法連線至 Flask 伺服器")
        _shutdown.set()
        return

    print("[Capture] ✅ 串流連線成功")

    while not _shutdown.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        mid = w // 2
        img_left = frame[:, :mid]
        img_right = frame[:, mid:]

        if enable_rectification and rectifier is not None:
            img_left, img_right = rectifier.rectify(img_left, img_right)

        frame_buffer.push(img_left, img_right)

    cap.release()
    print("[Capture] 執行緒結束")


# ─── Thread 2: Inference ──────────────────────────────────────
def inference_thread(
    frame_buffer: CircularFrameBuffer,
    inference_engine: InferenceEngine,
    depth_decoder: DepthDecoder,
    pose_manager: PoseManager,
    result_holder: dict,
    result_lock: threading.Lock,
    camera_params_mode: str = "auto",
    temporal_frames: int = 3,
):
    """Pull batches from buffer, infer depth, and store for preview."""
    print("[Inference] 等待足夠的時序幀...")

    if camera_params_mode == "provided":
        batch_extrinsics = pose_manager.get_batch_extrinsics()
        batch_intrinsics = pose_manager.get_batch_intrinsics()
        print("[Inference] 使用校正後的相機內外參 (provided mode)")
    else:
        batch_extrinsics = None
        batch_intrinsics = None
        print("[Inference] 不傳入相機內外參，由模型自行估計 (auto mode)")

    while not _shutdown.is_set():
        if not frame_buffer.is_ready():
            time.sleep(0.01)
            continue

        batch = frame_buffer.get_temporal_batch()
        if batch is None:
            time.sleep(0.01)
            continue

        try:
            raw_result = inference_engine.infer(
                batch,
                extrinsics=batch_extrinsics,
                intrinsics=batch_intrinsics,
                infer_gs=False,  # We just need depth maps
            )
            decoded = depth_decoder.decode(raw_result)

            # Keep the current left RGB frame for side-by-side display
            idx_lt = temporal_frames - 1
            decoded["color_image_left"] = batch[idx_lt]

            with result_lock:
                result_holder["latest"] = decoded

        except Exception as e:
            print(f"[Inference] ❌ 推論錯誤: {e}")
            time.sleep(0.1)

    print("[Inference] 執行緒結束")


# ─── Thread 3: Preview ─────────────────────────────────────────
def preview_thread(
    result_holder: dict,
    result_lock: threading.Lock,
):
    """擷取解碼後的深度圖，並以 Magma/Inferno 偽彩色與原圖並排顯示。"""
    print("[Preview] 等待推論結果...")
    
    cv2.namedWindow("DA3 Depth Preview", cv2.WINDOW_NORMAL)

    while not _shutdown.is_set():
        with result_lock:
            decoded = result_holder.get("latest")
            if decoded is not None:
                result_holder["latest"] = None

        if decoded is None:
            time.sleep(0.01)
            continue
            
        try:
            depth_left = decoded["depth_left"]
            color_left = decoded["color_image_left"]
            time_ms = decoded.get("time_ms", 0)
            fps = 1000.0 / max(time_ms, 1)
            target_h, target_w = color_left.shape[:2]

            # --- 關鍵修正：在浮點數階段就進行平滑縮放 ---
            # 使用 INTER_LANCZOS4 (比 CUBIC 更平滑) 直接縮放到原圖尺寸
            depth_smooth = cv2.resize(depth_left, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            # --- 新增：輕微的高斯模糊，消弭 ViT 的邊界感 ---
            # ksize=(5,5) 可以過濾掉 Patch 邊緣的跳變
            depth_smooth = cv2.GaussianBlur(depth_smooth, (5, 5), 0)

            # --- 歸一化處理 ---
            valid_mask = (depth_smooth > 0) & np.isfinite(depth_smooth)
            if np.any(valid_mask):
                # 使用 5% 到 95% 避免極端點拉大對比
                d_min, d_max = np.percentile(depth_smooth[valid_mask], [5, 95])
            else:
                d_min, d_max = 0, 1

            depth_clipped = np.clip(depth_smooth, d_min, d_max)
            depth_norm = (depth_clipped - d_min) / (max(d_max - d_min, 1e-5))
            
            # 轉為 8-bit
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # 套用調色盤 (此時畫面已經是平滑的)
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

            # 拼接並顯示
            combined = cv2.hconcat([color_left, depth_colored])
            cv2.imshow("DA3 Depth Preview", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                _shutdown.set()
                break

        except Exception as e:
            print(f"[Preview] ❌ 預覽錯誤: {e}")
            time.sleep(0.1)

    cv2.destroyAllWindows()
    print("[Preview] 執行緒結束")


# ─── Main ─────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DA3 即時深度圖預覽管道")
    parser.add_argument(
        "--config", type=str, default="config/pipeline_config.yaml",
        help="Pipeline 設定檔路徑",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print("=" * 60)
    print("  DA3 即時深度圖預覽 (RGB + Depth)")
    print("=" * 60)

    cam_cfg = config["camera"]
    inf_cfg = config["inference"]
    buf_cfg = config["buffer"]
    stream_cfg = config["stream"]

    image_size = (cam_cfg["width"], cam_cfg["height"])
    enable_rectification = cam_cfg.get("enable_rectification", True)

    if enable_rectification:
        print("\n[Init] 初始化立體校正器...")
        rectifier = StereoRectifier(
            calibration_path=cam_cfg["calibration_path"],
            image_size=image_size,
        )
    else:
        print("\n[Init] 跳過立體校正器初始化 (enable_rectification: false)")
        rectifier = None

    temporal_frames = buf_cfg["temporal_frames"]

    print("[Init] 初始化位姿管理器...")
    pose_manager = PoseManager(
        calibration_path=cam_cfg["calibration_path"],
        image_size=image_size,
        temporal_frames=temporal_frames,
    )

    print(f"[Init] 初始化幀環形緩衝區 (temporal_frames={temporal_frames})...")
    frame_buffer = CircularFrameBuffer(capacity=temporal_frames)

    print("[Init] 初始化推論引擎...")
    inference_engine = InferenceEngine(
        model_name=inf_cfg["model_name"],
        trt_engine_path=inf_cfg.get("trt_engine_path"),
        weights_path=inf_cfg.get("weights_path"),
        config_path=inf_cfg.get("config_path"),
        use_fp16=inf_cfg["use_fp16"],
        use_async=inf_cfg["use_async"],
        process_res=inf_cfg["process_res"],
    )

    print("[Init] 初始化深度解碼器...")
    # Note: confidence_threshold is required but mainly impacts Gaussian rendering.
    # We pass it but it won't be heavily used since we color map valid depth instead.
    gs_cfg = config.get("gaussian", {})
    depth_decoder = DepthDecoder(
        confidence_threshold=gs_cfg.get("confidence_threshold", 0.5),
        temporal_frames=temporal_frames,
    )

    # ── Shared state ──
    result_holder = {}
    result_lock = threading.Lock()

    # ── Start threads ──
    threads = []

    t_capture = threading.Thread(
        target=capture_thread,
        kwargs={
            "stream_url": stream_cfg["url"],
            "rectifier": rectifier,
            "frame_buffer": frame_buffer,
            "buffer_size": stream_cfg["buffer_size"],
            "enable_rectification": enable_rectification,
        },
        daemon=True,
    )
    threads.append(t_capture)

    camera_params_mode = inf_cfg.get("camera_params_mode", "auto")
    print(f"[Init] 相機參數模式: {camera_params_mode}")

    t_inference = threading.Thread(
        target=inference_thread,
        kwargs={
            "frame_buffer": frame_buffer,
            "inference_engine": inference_engine,
            "depth_decoder": depth_decoder,
            "pose_manager": pose_manager,
            "result_holder": result_holder,
            "result_lock": result_lock,
            "camera_params_mode": camera_params_mode,
            "temporal_frames": temporal_frames,
        },
        daemon=True,
    )
    threads.append(t_inference)

    t_preview = threading.Thread(
        target=preview_thread,
        args=(result_holder, result_lock),
        daemon=True,
    )
    threads.append(t_preview)

    print("\n[Pipeline] 啟動所有執行緒...")
    for t in threads:
        t.start()

    print(f"[Pipeline] ✅ 即時深度預覽已啟動！")
    print("  在 OpenCV 視窗按 'ESC' 或 'q' 退出\n")

    try:
        while not _shutdown.is_set():
            _shutdown.wait(timeout=1.0)
    except KeyboardInterrupt:
        pass

    _shutdown.set()
    print("\n[Pipeline] 等待執行緒結束...")

    for t in threads:
        t.join(timeout=3.0)

    print("[Pipeline] ✅ 已完全關閉")


if __name__ == "__main__":
    main()
