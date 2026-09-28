#!/bin/bash
# ============================================================
# RK3588 多人行为识别 演示脚本（权威工程版）
#
# 用法:
#   bash run_demo.sh                               # 默认视频，窗口显示
#   bash run_demo.sh videos_multi/m1_stand_walk.mp4
#   bash run_demo.sh rtsp://user:pass@ip:554/ch01
#   bash run_demo.sh data/videos/standing/stand_002.mp4 --headless
#
# 说明:
#   - 锁定演示版稳定配置（环境变量均可覆盖）
#   - --headless 无头模式，输出 output/behavior_result.mp4
# ============================================================
set -e
cd "$(dirname "$0")"

# ---- 自动探测可用的 Python 解释器（板端无 `python` 命令，见 find_python.sh）----
source "$(dirname "$0")/find_python.sh"

SOURCE="${1:-}"
if [ "$2" = "--headless" ]; then
  HEADLESS=1
fi

# ---- 演示版稳定配置（默认值，可被环境变量覆盖）----
export RK_POSE_BACKEND="${RK_POSE_BACKEND:-mediapipe}"
export RK_POSE_FULLFRAME="${RK_POSE_FULLFRAME:-1}"    # MediaPipe 自动不启用
export RK_FRAME_WIDTH="${RK_FRAME_WIDTH:-640}"
export RK_FRAME_HEIGHT="${RK_FRAME_HEIGHT:-360}"
export RK_CROP_PAD="${RK_CROP_PAD:-0.12}"
export RK_TRACKER_SMOOTH_ALPHA="${RK_TRACKER_SMOOTH_ALPHA:-0.65}"
export RK_OBJECT_SCORE="${RK_OBJECT_SCORE:-0.35}"
if [ -n "$HEADLESS" ]; then
  export RK_HEADLESS=1
fi
if [ -n "$SOURCE" ]; then
  export RK_SOURCE="$SOURCE"
fi

echo "=== RK3588 Behavior Demo (authoritative) ==="
echo "SOURCE      : ${RK_SOURCE:-<default>}"
echo "POSE_BACKEND: $RK_POSE_BACKEND"
echo "HEADLESS    : ${HEADLESS:-0}"
echo "PYTHON      : $RK_PYTHON_BIN"
echo "============================"

exec "$RK_PYTHON_BIN" app/main_realtime.py
