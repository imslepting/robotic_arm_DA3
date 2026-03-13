#!/usr/bin/env python3
"""
Export DA3-Small model to ONNX format for TensorRT compilation.

Usage:
    python scripts/export_onnx.py \
        --config weights/config.json \
        --weights weights/model.safetensors \
        --output models/da3_small.onnx \
        --batch-size 6 \
        --height 480 \
        --width 640
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

# Ensure the project root is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from depth_anything_3.api import DepthAnything3


def parse_args():
    parser = argparse.ArgumentParser(description="Export DA3 model to ONNX")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config.json")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model.safetensors")
    parser.add_argument("--output", type=str, default="models/da3_small.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="Batch size (6 = 3 temporal × 2 stereo)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model config from {args.config}...")
    with open(args.config) as f:
        config = json.load(f)

    print("Building model...")
    model = DepthAnything3(**config)

    print(f"Loading weights from {args.weights}...")
    state_dict = load_file(args.weights)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.cuda()

    # Create dummy input: (B=1, N=batch_size, 3, H, W)
    # DA3 forward expects (B, N, 3, H, W)
    dummy_input = torch.randn(
        1, args.batch_size, 3, args.height, args.width,
        dtype=torch.float32, device="cuda"
    )

    print(f"Exporting to ONNX (opset {args.opset})...")
    print(f"  Input shape: {dummy_input.shape}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model.model,  # Export the inner model, not the wrapper
            (dummy_input, None, None),  # (images, extrinsics, intrinsics)
            str(output_path),
            opset_version=args.opset,
            input_names=["images"],
            output_names=["depth", "depth_conf"],
            dynamic_axes={
                "images": {0: "batch", 1: "num_views"},
                "depth": {0: "batch", 1: "num_views"},
                "depth_conf": {0: "batch", 1: "num_views"},
            },
            do_constant_folding=True,
        )
        print(f"✅ ONNX model saved to {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        print("   The model may contain operations not supported by ONNX.")
        print("   The pipeline will use PyTorch FP16 fallback instead.")
        sys.exit(1)


if __name__ == "__main__":
    main()
