#!/usr/bin/env bash
# ============================================
# Build TensorRT Engine from ONNX Model
# ============================================
#
# Usage:
#   bash scripts/build_trt_engine.sh [onnx_path] [engine_path]
#
# Defaults:
#   onnx_path   = models/da3_small.onnx
#   engine_path = models/da3_small_fp16.engine
#
# Requirements:
#   - NVIDIA TensorRT (trtexec) installed (comes with JetPack on Orin)
#   - ONNX model exported via scripts/export_onnx.py

set -euo pipefail

ONNX_PATH="${1:-models/da3_small.onnx}"
ENGINE_PATH="${2:-models/da3_small_fp16.engine}"

echo "============================================"
echo "  TensorRT Engine Builder"
echo "============================================"
echo "  ONNX input:  ${ONNX_PATH}"
echo "  Engine out:   ${ENGINE_PATH}"
echo ""

# Check trtexec availability
if ! command -v trtexec &> /dev/null; then
    echo "❌ trtexec not found. Install TensorRT or use JetPack."
    echo "   The pipeline will fall back to PyTorch FP16 inference."
    exit 1
fi

# Check ONNX file exists
if [ ! -f "${ONNX_PATH}" ]; then
    echo "❌ ONNX file not found: ${ONNX_PATH}"
    echo "   Run 'python scripts/export_onnx.py' first."
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "${ENGINE_PATH}")"

echo "Building TensorRT engine (FP16, explicitBatch=6)..."
echo ""

trtexec \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --fp16 \
    --optShapes=images:1x6x3x480x640 \
    --minShapes=images:1x1x3x480x640 \
    --maxShapes=images:1x6x3x480x640 \
    --workspace=4096 \
    --verbose \
    2>&1 | tee "$(dirname "${ENGINE_PATH}")/trtexec_build.log"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TensorRT engine saved to ${ENGINE_PATH}"
    echo "   Size: $(du -sh "${ENGINE_PATH}" | cut -f1)"
else
    echo ""
    echo "❌ TensorRT build failed. Check trtexec_build.log for details."
    echo "   The pipeline will fall back to PyTorch FP16 inference."
    exit 1
fi
