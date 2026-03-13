#!/bin/bash
# 相機診斷腳本

echo "=========================================="
echo "相機診斷"
echo "=========================================="

echo ""
echo "[1] 檢查設備"
v4l2-ctl --list-devices

echo ""
echo "[2] 檢查權限"
ls -la /dev/video0 /dev/video2 2>/dev/null || echo "設備不存在"
echo "當前用戶: $(whoami)"
echo "所屬群組: $(groups)"

echo ""
echo "[3] 測試 video0"
timeout 3 v4l2-ctl -d /dev/video0 --get-fmt-video 2>&1 | head -5 || echo "video0 無法訪問"

echo ""
echo "[4] 測試使用 ffmpeg"
which ffmpeg && echo "ffmpeg 已安裝" || echo "ffmpeg 未安裝"

echo ""
echo "=========================================="
