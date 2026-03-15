#!/usr/bin/env python3
"""
Debug script to analyze depth value distribution and identify flattening issues.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline.frame_buffer import CircularFrameBuffer
from pipeline.stereo_rectifier import StereoRectifier
from pipeline.pose_manager import PoseManager
from pipeline.inference_engine import InferenceEngine
from pipeline.depth_decoder import DepthDecoder
from pipeline.gaussian_projector import GaussianProjector


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_depth():
    """Analyze depth distribution from a live stream sample."""
    config = load_config("config/pipeline_config.yaml")
    
    cam_cfg = config["camera"]
    inf_cfg = config["inference"]
    gs_cfg = config["gaussian"]
    stream_cfg = config["stream"]
    
    image_size = (cam_cfg["width"], cam_cfg["height"])

    print("=" * 70)
    print("  深度分析診斷工具")
    print("=" * 70)

    # Initialize modules
    print("\n[Init] 初始化模組...")
    rectifier = StereoRectifier(
        calibration_path=cam_cfg["calibration_path"],
        image_size=image_size,
    )

    pose_manager = PoseManager(
        calibration_path=cam_cfg["calibration_path"],
        image_size=image_size,
    )

    frame_buffer = CircularFrameBuffer(capacity=3)

    inference_engine = InferenceEngine(
        model_name=inf_cfg["model_name"],
        trt_engine_path=inf_cfg.get("trt_engine_path"),
        weights_path=inf_cfg.get("weights_path"),
        config_path=inf_cfg.get("config_path"),
        use_fp16=inf_cfg["use_fp16"],
        use_async=inf_cfg["use_async"],
        process_res=inf_cfg["process_res"],
    )

    depth_decoder = DepthDecoder(
        confidence_threshold=gs_cfg["confidence_threshold"],
    )

    gaussian_projector = GaussianProjector(
        scale_multiplier=gs_cfg["scale_multiplier"],
        default_opacity=gs_cfg["opacity_default"],
        max_points=gs_cfg["max_points"],
    )

    # Get camera parameters
    K_l = pose_manager.get_left_intrinsic().cpu().numpy()
    print(f"\n[摄像头] 左摄像头内参:\n{K_l}")
    print(f"  fx={K_l[0,0]:.1f}, fy={K_l[1,1]:.1f}, cx={K_l[0,2]:.1f}, cy={K_l[1,2]:.1f}")

    # Connect to stream
    print(f"\n[流媒体] 连接至: {stream_cfg['url']}")
    cap = cv2.VideoCapture(stream_cfg['url'])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, stream_cfg['buffer_size'])

    if not cap.isOpened():
        print("[流媒体] ❌ 无法连接到 Flask 服务器")
        return

    print("[流媒体] ✅ 连接成功，开始采集样本...")

    # Collect samples
    batch_extrinsics = pose_manager.get_batch_extrinsics()
    batch_intrinsics = pose_manager.get_batch_intrinsics()

    sample_count = 0
    max_samples = 5

    while sample_count < max_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        # Split stereo
        h, w = frame.shape[:2]
        mid = w // 2
        img_left = frame[:, :mid]
        img_right = frame[:, mid:]

        # Rectify and buffer
        rect_left, rect_right = rectifier.rectify(img_left, img_right)
        frame_buffer.push(rect_left, rect_right)

        if not frame_buffer.is_ready():
            continue

        # Get batch and run inference
        batch = frame_buffer.get_temporal_batch()
        if batch is None:
            continue

        print(f"\n[样本 {sample_count + 1}/{max_samples}] 运行推理...")
        raw_result = inference_engine.infer(
            batch,
            extrinsics=batch_extrinsics,
            intrinsics=batch_intrinsics,
        )

        decoded = depth_decoder.decode(raw_result)

        # Analyze depth
        depth_left = decoded["depth_left"]
        conf_left = decoded["conf_left"]

        print(f"  推理时间: {decoded['time_ms']:.1f}ms")
        print(f"  深度范围: [{depth_left.min():.4f}, {depth_left.max():.4f}] m")
        print(f"  深度平均值: {depth_left[depth_left > 0].mean():.4f} m")
        print(f"  深度标准差: {depth_left[depth_left > 0].std():.4f} m")
        print(f"  信心度范围: [{conf_left.min():.4f}, {conf_left.max():.4f}]")
        
        # Analyze Gaussian scales
        K_l = pose_manager.get_left_intrinsic().cpu().numpy()
        ext_l = pose_manager.get_left_extrinsic().cpu().numpy()
        
        # Simulate gaussian projection scale
        fx = K_l[0, 0]
        fy = K_l[1, 1]
        
        # Calculate what scales would be
        scale_x_min = depth_left.min() * gs_cfg["scale_multiplier"] / fx
        scale_x_max = depth_left.max() * gs_cfg["scale_multiplier"] / fx
        scale_x_mean = depth_left[depth_left > 0].mean() * gs_cfg["scale_multiplier"] / fx
        
        print(f"\n  高斯缩放因子 (scale_x):")
        print(f"    最小: {scale_x_min:.6f} m")
        print(f"    最大: {scale_x_max:.6f} m")
        print(f"    平均: {scale_x_mean:.6f} m")
        print(f"  配置缩放系数: {gs_cfg['scale_multiplier']}")
        print(f"  焦距: fx={fx:.1f}, fy={fy:.1f}")
        
        # Estimate z-scale
        scale_z_mean = scale_x_mean * 0.5
        print(f"    z缩放平均: {scale_z_mean:.6f} m")

        # Check depth range histogram
        valid_depth = depth_left[depth_left > 0.01]
        if len(valid_depth) > 0:
            hist, bins = np.histogram(valid_depth, bins=10)
            print(f"\n  深度直方图 (有效像素总数: {len(valid_depth)}):")
            for i in range(len(hist)):
                bar = "█" * int(hist[i] / hist.max() * 30) if hist.max() > 0 else ""
                print(f"    [{bins[i]:.2f}-{bins[i+1]:.2f}): {int(hist[i]):6d} {bar}")

        sample_count += 1

    cap.release()

    # Recommendations
    print("\n" + "=" * 70)
    print("  诊断建议")
    print("=" * 70)
    print("""
如果深度看起来很平，可能的原因和解决方案：

1. **深度值范围太小（都在0.1-0.5m之间）**
   → 检查摄像头焦距是否正确（在calibration_params.yml中）
   → 如果焦距过大，会导致保投影点集中在很小的范围
   → 解决: 验证标定参数或调整 scale_multiplier

2. **深度推理本身的效果不好**
   → 检查model_name是否正确 (da3-small/large/giant)
   → 尝试更大的模型以获得更好的细节
   → 解决: 在config中改用 da3-large 或 da3-giant

3. **scale_multiplier 太小**
   → 当前值: 0.01 (相对于焦距)
   → 如果焦距很大（如5000+），最终scale会很小
   → 建议值:
     * 如果深度在0-10m: 尝试 0.05-0.1
     * 如果深度在0-50m: 尝试 0.02-0.05
   → 解决: 在 gaussian.scale_multiplier 中增加此值

4. **confidence_threshold 太高**
   → 当前值: 0.8 (只显示高信心度点)
   → 这会过度过滤掉有纹理细节的区域
   → 解决: 降低到 0.5-0.7 以保留更多细节

5. **深度范围真的被压扁了**
   → DA3可能在这个特定场景输出的深度分布不均匀
   → 解决: 
     * 尝试不同的 process_res (504/768/1024)
     * 检查batch_size配置是否正确
     * 验证相机外参（世界坐标系设置）

建议的调试步骤：
1. 运行此诊断脚本,观察深度范围和高斯缩放
2. 根据结果调整 scale_multiplier
3. 如果深度范围太小,检查calibration_params.yml中的焦距
4. 尝试降低 confidence_threshold 到 0.5-0.6
5. 考虑升级到 da3-large 模型以获得更好的细节
    """)


if __name__ == "__main__":
    analyze_depth()
