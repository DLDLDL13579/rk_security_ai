import os
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from engine.ai_thread import AIThread
from engine.engine import PoseEngine
from engine.shared import SharedData
from npu.pose_rknn import PoseRKNN
from npu.yolo_rknn_v2 import YOLO_RKNN
from npu.mediapipe_pose import MediaPipePose
from rtsp.stream import RTSPThread
from display.display import DisplayThread


PROJECT_ROOT = BASE_DIR

YOLO_MODEL = os.environ.get(
    "RK_YOLO_MODEL",
    os.path.join(PROJECT_ROOT, "models", "yolov5s-640-640.rknn"),
)
POSE_MODEL = os.environ.get(
    "RK_POSE_MODEL",
    os.path.join(PROJECT_ROOT, "models", "yolov8n-pose.rknn"),
)
BEHAVIOR_MODEL = os.environ.get(
    "RK_BEHAVIOR_MODEL",
    os.path.join(PROJECT_ROOT, "models", "behavior_tcn.rknn"),
)
SOURCE = os.environ.get(
    "RK_SOURCE",
    os.path.join(PROJECT_ROOT, "data", "videos", "walking", "walk_001.mp4"),
)
POSE_BACKEND = os.environ.get(
    "RK_POSE_BACKEND",
    "rknn",
).strip().lower()


def monitor(shared):
    print("[MONITOR] started")

    while shared.running:
        time.sleep(2.0)

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
            state = behavior.get("state", "valid")
            print(f"ID={track_id} | action={action} | score={score:.2f} | state={state}")

        print("===================================\n")


def main():
    print("\n==============================")
    print(" RK3588 DEBUG MODE START ")
    print("==============================\n")
    print("RUN PATH:", PROJECT_ROOT)

    yolo = YOLO_RKNN(YOLO_MODEL)
    if POSE_BACKEND == "mediapipe":
        pose = MediaPipePose()
    else:
        pose = PoseRKNN(POSE_MODEL)
    engine = PoseEngine(yolo, pose, behavior_model_path=BEHAVIOR_MODEL)

    shared = SharedData()

    rtsp_thread = RTSPThread(shared, SOURCE)
    ai_thread = AIThread(engine, shared)
    display_thread = DisplayThread(shared)

    rtsp_thread.start()
    threading.Thread(target=ai_thread.run, daemon=True).start()
    threading.Thread(target=display_thread.run, daemon=True).start()
    threading.Thread(target=monitor, args=(shared,), daemon=True).start()

    try:
        print("\n[SYSTEM] RUNNING...\n")
        while shared.running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[SYSTEM] STOPPING...")
        shared.running = False
        time.sleep(1.0)
        print("[SYSTEM] STOPPED")


if __name__ == "__main__":
    main()
