# -*- coding: utf-8 -*-
"""
安防版实时入口 —— 在权威工程主链路之上叠加安防事件与 MQTT 上报

与 app/main_realtime.py 的关系：
    完全复用其全部组件（YOLO 检测 + 姿态 + 行为识别 + 显示），
    **仅追加** SecurityThread（安防判定 + 上报），不改动原有任何一行逻辑。
    因此行为识别精度与性能不受影响（安防线程是独立线程、独立限频）。

启动：
    source security_mqtt.env          # 载入 RK_MQTT_TOKEN 等
    python3 app/main_security.py

环境变量：
    RK_MQTT_TOKEN      设备 AccessToken（必需，否则只判定不上报）
    RK_MQTT_BROKER     默认 192.168.1.8
    RK_MQTT_PORT       默认 1883
    RK_SECURITY_ZONES  区域 JSON（可选，缺省用内置示例区域）
    RK_FIRE_SMOKE_MODEL 烟火模型路径（可选，缺省关闭烟火）
    RK_SECURITY_ENABLED "0" 可整体关闭安防（退化为原 main_realtime.py）
"""

import json
import os
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from engine.ai_thread import AIThread
from engine.engine import PoseEngine
from engine.shared import SharedData
from engine.security_monitor import SecurityMonitor, SecurityThread
from engine.mqtt_reporter import MqttReporter
from npu.pose_rknn import PoseRKNN
from npu.yolo_rknn_v2 import YOLO_RKNN
from npu.mediapipe_pose import MediaPipePose
from rtsp.stream import RTSPThread
from display.display import DisplayThread


PROJECT_ROOT = BASE_DIR

YOLO_MODEL = os.environ.get("RK_YOLO_MODEL", os.path.join(PROJECT_ROOT, "models", "yolov5s-640-640.rknn"))
# 注意：models/model.rknn 是检测模型副本（md5 与 yolov5s-640-640.rknn 相同，见 P0-1），
# 真正的姿态模型是 yolov8n-pose.rknn —— 与 run_demo.sh 的权威默认值保持一致。
POSE_MODEL = os.environ.get(
    "RK_POSE_MODEL", os.path.join(PROJECT_ROOT, "models", "yolov8n-pose.rknn")
)
BEHAVIOR_MODEL = os.environ.get("RK_BEHAVIOR_MODEL", os.path.join(PROJECT_ROOT, "models", "behavior_tcn.rknn"))
SOURCE = os.environ.get("RK_SOURCE", os.path.join(PROJECT_ROOT, "data", "videos", "walking", "walk_001.mp4"))
# 与 run_demo.sh 一致：板端默认走 NPU（rknn），而非 CPU 的 mediapipe
POSE_BACKEND = os.environ.get("RK_POSE_BACKEND", "rknn").strip().lower()

FIRE_SMOKE_MODEL = os.environ.get("RK_FIRE_SMOKE_MODEL", "").strip()
FALL_CONFIDENCE = float(os.environ.get("RK_FALL_CONFIDENCE", "0.55"))
# 误报抑制（方案 A）：连续帧确认
FALL_CONFIRM_FRAMES = int(os.environ.get("RK_FALL_CONFIRM_FRAMES", "3"))
FIRE_SMOKE_CONFIRM_FRAMES = int(os.environ.get("RK_FIRE_SMOKE_CONFIRM_FRAMES", "3"))
SECURITY_ENABLED = os.environ.get("RK_SECURITY_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")

# 缺省示例区域（与 _8.4 默认一致；实际部署按现场画定）
DEFAULT_ZONES = [
    {
        "id": "intrusion_zone",
        "name": "Restricted Area",
        "type": "intrusion",
        "rect": [360, 40, 620, 340],
        "color": (0, 200, 255),
    },
    {
        "id": "loiter_zone",
        "name": "Stay Alert Area",
        "type": "loiter",
        "rect": [80, 40, 260, 340],
        "color": (255, 180, 0),
        "loiter_seconds": 8.0,
    },
]


def build_zones():
    raw = os.environ.get("RK_SECURITY_ZONES", "").strip()
    if not raw:
        return DEFAULT_ZONES
    try:
        zones = json.loads(raw)
        if isinstance(zones, list):
            print(f"[Security] 已从 RK_SECURITY_ZONES 载入 {len(zones)} 个区域")
            return zones
        print("[Security] RK_SECURITY_ZONES 不是列表，改用默认区域")
    except Exception as exc:
        print(f"[Security] RK_SECURITY_ZONES 解析失败（{exc}），改用默认区域")
    return DEFAULT_ZONES


def build_fire_smoke_detector():
    from engine.fire_smoke_detector import NullFireSmokeDetector

    if not FIRE_SMOKE_MODEL:
        print("[FireSmoke] 未配置 RK_FIRE_SMOKE_MODEL，烟火检测关闭")
        return NullFireSmokeDetector()
    try:
        from npu.fire_smoke_rknn import FireSmokeRKNN

        detector = FireSmokeRKNN(FIRE_SMOKE_MODEL)
        print(f"[FireSmoke] 已启用: {FIRE_SMOKE_MODEL}")
        return detector
    except Exception as exc:
        print(f"[FireSmoke] 初始化失败（{exc}），降级为关闭")
        return NullFireSmokeDetector()


def monitor(shared, security_thread=None):
    print("[MONITOR] started")

    while shared.running:
        time.sleep(5.0)

        detections = shared.get_detections()
        behaviors = shared.get_behaviors()
        fps = shared.get_fps()

        print("\n===================================")
        print("[SYSTEM STATUS]")
        print(f"FPS: {fps:.2f}")
        print(f"Detections: {len(detections)}")
        print(f"Behaviors: {len(behaviors)}")

        for track_id, behavior in behaviors.items():
            action = behavior.get("action", "unknown")
            score = float(behavior.get("confidence", 0.0))
            print(f"ID={track_id} | action={action} | score={score:.2f}")

        if security_thread is not None:
            stats = security_thread.monitor.stats
            print(
                "[SECURITY] 累计事件: "
                f"跌倒={stats.get('FALL_DETECTED', 0)} "
                f"入侵={stats.get('INTRUSION_DETECTED', 0)} "
                f"逗留={stats.get('LOITERING_DETECTED', 0)} "
                f"火焰={stats.get('FIRE_DETECTED', 0)} "
                f"烟雾={stats.get('SMOKE_DETECTED', 0)}"
            )
            if security_thread.mqtt_reporter is not None:
                print(f"[MQTT] {security_thread.mqtt_reporter.stats()}")

        print("===================================\n")


def main():
    print("\n==============================")
    print(" RK3588 SECURITY MODE START ")
    print("==============================\n")
    print("RUN PATH:", PROJECT_ROOT)

    yolo = YOLO_RKNN(YOLO_MODEL)
    pose = MediaPipePose() if POSE_BACKEND == "mediapipe" else PoseRKNN(POSE_MODEL)
    engine = PoseEngine(yolo, pose, behavior_model_path=BEHAVIOR_MODEL)

    shared = SharedData()

    # ---- 安防 + 上报（仅为追加，不影响原链路） ----
    security_thread = None
    zones = build_zones()
    security_monitor = SecurityMonitor(
        zones=zones,
        fall_confidence=FALL_CONFIDENCE,
        fall_confirm_frames=FALL_CONFIRM_FRAMES,
        fire_smoke_confirm_frames=FIRE_SMOKE_CONFIRM_FRAMES,
    )
    print(f"[Security] 已配置 {len(zones)} 个区域")
    print(
        f"[Security] 误报抑制：跌倒需连续 {FALL_CONFIRM_FRAMES} 帧确认"
        f"（阈值 {FALL_CONFIDENCE}），烟火需连续 {FIRE_SMOKE_CONFIRM_FRAMES} 帧确认"
    )

    if SECURITY_ENABLED:
        reporter = MqttReporter.from_env()
        if reporter.enabled:
            reporter.start()
        else:
            print("[Security] 未提供 RK_MQTT_TOKEN，仅本地判定不上报")

        security_thread = SecurityThread(
            shared,
            monitor=security_monitor,
            mqtt_reporter=reporter if reporter.enabled else None,
            fire_smoke_detector=build_fire_smoke_detector(),
        )
    else:
        print("[Security] 已按 RK_SECURITY_ENABLED=0 关闭安防")

    # ---- 原有主链路（与 main_realtime.py 完全一致） ----
    rtsp_thread = RTSPThread(shared, SOURCE)
    ai_thread = AIThread(engine, shared)
    display_thread = DisplayThread(shared)

    rtsp_thread.start()
    threading.Thread(target=ai_thread.run, daemon=True).start()
    threading.Thread(target=display_thread.run, daemon=True).start()
    threading.Thread(target=monitor, args=(shared, security_thread), daemon=True).start()

    if security_thread is not None:
        security_thread.start()

    try:
        print("\n[SYSTEM] RUNNING (security mode)...\n")
        while shared.running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[SYSTEM] STOPPING...")
        shared.running = False
        time.sleep(1.0)
        if security_thread is not None and security_thread.mqtt_reporter is not None:
            security_thread.mqtt_reporter.stop()
        print("[SYSTEM] STOPPED")


if __name__ == "__main__":
    main()
