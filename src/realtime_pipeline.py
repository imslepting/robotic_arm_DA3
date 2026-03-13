#!/usr/bin/env python3
"""
DA3 Real-time 3DGS Pipeline — Main Entry Point.

Multi-threaded pipeline:
  Thread 1 (Capture):   Flask stream → stereo rectification → circular buffer
  Thread 2 (Inference): buffer → batch assembly → DA3 async inference → depth
  Thread 3 (Render):    depth → Gaussian projection → Viser 3D viewer

Usage:
    python src/realtime_pipeline.py --config config/pipeline_config.yaml

Open the Viser viewer at http://<host>:<port> in a browser.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.frame_buffer import CircularFrameBuffer
from pipeline.stereo_rectifier import StereoRectifier
from pipeline.pose_manager import PoseManager
from pipeline.inference_engine import InferenceEngine
from pipeline.depth_decoder import DepthDecoder
from pipeline.gaussian_projector import GaussianProjector
from pipeline.confidence_filter import ConfidenceFilter
from pipeline.viser_renderer import ViserRenderer


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
    rectifier: StereoRectifier,
    frame_buffer: CircularFrameBuffer,
    buffer_size: int = 1,
):
    """Read frames from Flask stream, rectify, and push to buffer."""
    print(f"[Capture] 正在連線至串流: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    if not cap.isOpened():
        print("[Capture] ❌ 無法連線至 Flask 伺服器")
        _shutdown.set()
        return

    print("[Capture] ✅ 串流連線成功")
    frame_count = 0

    while not _shutdown.is_set():
        ret, frame = cap.read()
        if not ret:
            # Brief wait and retry
            time.sleep(0.01)
            continue

        # Split side-by-side frame into left and right
        h, w = frame.shape[:2]
        mid = w // 2
        img_left = frame[:, :mid]
        img_right = frame[:, mid:]

        # Stereo rectification (fast cv2.remap)
        rect_left, rect_right = rectifier.rectify(img_left, img_right)

        # Push to circular buffer
        frame_buffer.push(rect_left, rect_right)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"[Capture] 已接收 {frame_count} 幀")

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
):
    """Pull temporal batches from buffer and run DA3 inference."""
    print("[Inference] 等待足夠的時序幀...")

    # Precomputed camera tensors (constant, already on GPU)
    batch_extrinsics = pose_manager.get_batch_extrinsics()  # (6, 4, 4)
    batch_intrinsics = pose_manager.get_batch_intrinsics()  # (6, 3, 3)

    while not _shutdown.is_set():
        # Wait until buffer has enough frames
        if not frame_buffer.is_ready():
            time.sleep(0.01)
            continue

        # Get temporal batch: [L_{t-2}, L_{t-1}, L_t, R_{t-2}, R_{t-1}, R_t]
        batch = frame_buffer.get_temporal_batch()
        if batch is None:
            time.sleep(0.01)
            continue

        # Run inference with camera parameters
        try:
            raw_result = inference_engine.infer(
                batch,
                extrinsics=batch_extrinsics,
                intrinsics=batch_intrinsics,
            )
            decoded = depth_decoder.decode(raw_result)

            # Also store the current left RGB frame for coloring
            decoded["color_image_left"] = batch[2]    # L_t
            decoded["color_image_right"] = batch[5]   # R_t

            with result_lock:
                result_holder["latest"] = decoded

            fps = 1000.0 / max(decoded["time_ms"], 1)
            print(f"[Inference] 深度推論完成: {decoded['time_ms']:.0f}ms ({fps:.1f} FPS)")

        except Exception as e:
            print(f"[Inference] ❌ 推論錯誤: {e}")
            time.sleep(0.1)

    print("[Inference] 執行緒結束")


# ─── Thread 3: Render ─────────────────────────────────────────
def render_thread(
    pose_manager: PoseManager,
    gaussian_projector: GaussianProjector,
    confidence_filter: ConfidenceFilter,
    viser_renderer: ViserRenderer,
    result_holder: dict,
    result_lock: threading.Lock,
):
    """Convert depth to Gaussians and render via Viser."""
    print("[Render] 等待推論結果...")

    last_render_time = time.time()

    while not _shutdown.is_set():
        # Get latest inference result
        with result_lock:
            decoded = result_holder.get("latest")
            if decoded is not None:
                result_holder["latest"] = None  # Consume it

        if decoded is None:
            time.sleep(0.01)
            continue

        try:
            t0 = time.perf_counter()

            # Get dynamic confidence threshold from UI
            conf_thresh = viser_renderer.get_confidence_threshold()

            # Get camera parameters
            K_l = pose_manager.get_left_intrinsic().cpu().numpy()
            ext_l = pose_manager.get_left_extrinsic().cpu().numpy()

            # Project left depth to Gaussians
            depth_left = decoded["depth_left"]
            conf_left = decoded["conf_left"]
            color_left = decoded["color_image_left"]

            # Apply confidence mask
            conf_mask = conf_left >= conf_thresh

            # Gaussian projection
            gaussians = gaussian_projector.project(
                depth=depth_left,
                color_image=color_left,
                intrinsic=K_l,
                extrinsic=ext_l,
                confidence=conf_left,
                mask=conf_mask,
            )

            # Update Viser scene
            if gaussians["num_points"] > 0:
                viser_renderer.update_point_cloud(
                    points=gaussians["means"],
                    colors=gaussians["colors"],
                    name="realtime_3dgs",
                )

            t1 = time.perf_counter()
            render_ms = (t1 - t0) * 1000

            # Show camera frustums periodically
            if viser_renderer.frame_count % 30 == 0:
                ext_batch = pose_manager.get_batch_extrinsics().cpu().numpy()[:2]  # L and R only
                ixt_batch = pose_manager.get_batch_intrinsics().cpu().numpy()[:2]
                viser_renderer.update_cameras(ext_batch, ixt_batch)

            if viser_renderer.frame_count % 10 == 0:
                print(
                    f"[Render] 幀 {viser_renderer.frame_count}: "
                    f"{gaussians['num_points']:,} 點, "
                    f"渲染 {render_ms:.0f}ms, "
                    f"FPS {viser_renderer.fps:.1f}"
                )

        except Exception as e:
            print(f"[Render] ❌ 渲染錯誤: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

    viser_renderer.shutdown()
    print("[Render] 執行緒結束")


# ─── Main ─────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DA3 即時 3DGS 管道")
    parser.add_argument(
        "--config", type=str, default="config/pipeline_config.yaml",
        help="Pipeline 設定檔路徑",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print("=" * 60)
    print("  DA3 即時 3D 高斯潑濺管道")
    print("=" * 60)

    # ── Initialize modules ──
    cam_cfg = config["camera"]
    inf_cfg = config["inference"]
    gs_cfg = config["gaussian"]
    vis_cfg = config["viser"]
    buf_cfg = config["buffer"]
    stream_cfg = config["stream"]

    image_size = (cam_cfg["width"], cam_cfg["height"])

    print("\n[Init] 初始化立體校正器...")
    rectifier = StereoRectifier(
        calibration_path=cam_cfg["calibration_path"],
        image_size=image_size,
    )

    print("[Init] 初始化位姿管理器...")
    pose_manager = PoseManager(
        calibration_path=cam_cfg["calibration_path"],
        image_size=image_size,
    )

    print("[Init] 初始化幀環形緩衝區...")
    frame_buffer = CircularFrameBuffer(capacity=buf_cfg["capacity"])

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
    depth_decoder = DepthDecoder(
        confidence_threshold=gs_cfg["confidence_threshold"],
    )

    print("[Init] 初始化高斯投影器...")
    gaussian_projector = GaussianProjector(
        scale_multiplier=gs_cfg["scale_multiplier"],
        default_opacity=gs_cfg["opacity_default"],
        max_points=gs_cfg["max_points"],
    )

    print("[Init] 初始化信賴度過濾器...")
    confidence_filter = ConfidenceFilter(
        threshold=gs_cfg["confidence_threshold"],
    )

    print("[Init] 初始化 Viser 3D 檢視器...")
    viser_renderer = ViserRenderer(
        host=vis_cfg["host"],
        port=vis_cfg["port"],
        point_size=vis_cfg["point_size"],
        background_color=tuple(vis_cfg["background_color"]),
    )

    # ── Shared state ──
    result_holder = {}
    result_lock = threading.Lock()

    # ── Start threads ──
    threads = []

    t_capture = threading.Thread(
        target=capture_thread,
        args=(stream_cfg["url"], rectifier, frame_buffer, stream_cfg["buffer_size"]),
        daemon=True,
        name="CaptureThread",
    )
    threads.append(t_capture)

    t_inference = threading.Thread(
        target=inference_thread,
        args=(frame_buffer, inference_engine, depth_decoder, pose_manager, result_holder, result_lock),
        daemon=True,
        name="InferenceThread",
    )
    threads.append(t_inference)

    t_render = threading.Thread(
        target=render_thread,
        args=(
            pose_manager, gaussian_projector, confidence_filter,
            viser_renderer, result_holder, result_lock,
        ),
        daemon=True,
        name="RenderThread",
    )
    threads.append(t_render)

    print("\n[Pipeline] 啟動所有執行緒...")
    for t in threads:
        t.start()

    print(f"[Pipeline] ✅ 管道已啟動！")
    print(f"  串流來源: {stream_cfg['url']}")
    print(f"  3D 檢視器: http://{vis_cfg['host']}:{vis_cfg['port']}")
    print(f"  推論後端: {inference_engine.backend_name}")
    print(f"  按 Ctrl+C 停止")

    # Wait for shutdown
    try:
        while not _shutdown.is_set():
            _shutdown.wait(timeout=1.0)
    except KeyboardInterrupt:
        pass

    _shutdown.set()
    print("\n[Pipeline] 等待執行緒結束...")

    for t in threads:
        t.join(timeout=5.0)

    print("[Pipeline] ✅ 已完全關閉")


if __name__ == "__main__":
    main()
