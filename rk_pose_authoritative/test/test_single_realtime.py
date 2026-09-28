# -*- coding: utf-8 -*-
"""
============================================================
test_single_realtime.py — 实时单人行为测试（双通道稳定版）

流程:
    视频/RTSP/摄像头
        → YOLO 检测(单人)
        → MediaPipe 姿态(裁剪)
        → 46D 特征 → TCN
        → 双通道判定(Locomotion / Special)
        → 窗口显示 主目标行为 + 变更日志 + FPS

用法:
    python test_single_realtime.py
    python test_single_realtime.py --source videos/walking/walk_001.mp4
    python test_single_realtime.py --source rtsp://user:pass@ip:554/ch01
    python test_single_realtime.py --source videos/standing/stand_002.mp4 --headless

退出:
    窗口模式下按 ESC；任何模式下 Ctrl+C
============================================================
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _letterbox(frame, target_size):
    """保持宽高比缩放到 target_size (w, h)，不足处补黑边。"""
    tw, th = target_size
    h, w = frame.shape[:2]
    scale = min(tw / float(w), th / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def build_engine():
    from npu.yolo_rknn_v2 import YOLO_RKNN
    from npu.mediapipe_pose import MediaPipePose
    from engine.engine import PoseEngine

    yolo = YOLO_RKNN(
        os.path.join(PROJECT_ROOT, "models", "yolov5s-640-640.rknn")
    )
    pose = MediaPipePose()
    engine = PoseEngine(
        yolo=yolo,
        pose=pose,
        behavior_model_path=os.path.join(
            PROJECT_ROOT, "models", "behavior_tcn.rknn"
        ),
    )
    return engine


class Shared:
    """线程间共享：最新帧 + 最新结果"""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.result = {}
        self.running = True
        self.action_log = {}   # track_id -> last action


def capture_worker(shared, source, width, height):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[CAP] cannot open:", source)
        shared.running = False
        return

    while shared.running:
        ret, frame = cap.read()
        if not ret:
            # 视频文件循环播放
            if isinstance(source, str) and source.lower().endswith(
                (".mp4", ".avi", ".mkv")
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        if width > 0 and height > 0 and frame.shape[:2][::-1] != (width, height):
            frame = _letterbox(frame, (width, height))

        with shared.lock:
            shared.frame = frame
            shared.frame_id += 1

        time.sleep(0.005)

    cap.release()


def ai_worker(shared, engine, interval):
    last_run = 0.0
    while shared.running:
        with shared.lock:
            frame = shared.frame
            frame_id = shared.frame_id

        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        if now - last_run < interval:
            time.sleep(0.005)
            continue
        last_run = now

        try:
            det_result = engine.process_detections(frame)
            beh_result = engine.process_behaviors(
                frame,
                frame_id=frame_id,
                tracks=det_result.get("detections", []),
            )

            behaviors = beh_result.get("behaviors", [])
            # 主目标：面积最大的人
            main = None
            if behaviors:
                main = max(
                    behaviors,
                    key=lambda b: (
                        (b.get("bbox")[2] - b.get("bbox")[0])
                        * (b.get("bbox")[3] - b.get("bbox")[1])
                        if b.get("bbox")
                        else 0
                    ),
                )

            with shared.lock:
                shared.result = {
                    "frame_id": frame_id,
                    "main": main,
                    "all": behaviors,
                    "ai_fps": 1.0 / max(1e-6, time.time() - now),
                }

            # 动作变更日志
            if main is not None:
                tid = main.get("track_id", -1)
                action = main.get("action", "unknown")
                state = main.get("state", "")
                conf = float(main.get("confidence", 0.0))
                old = shared.action_log.get(tid)
                if old != action:
                    shared.action_log[tid] = action
                    print(
                        "[ACTION] ID%d %s conf=%.2f state=%s"
                        % (tid, action, conf, state)
                    )

        except Exception as e:
            print("[AI ERROR]", e)

        time.sleep(0.005)


def display_worker(shared, headless, out_path):
    writer = None
    last = time.time()
    fps = 0.0

    while shared.running:
        with shared.lock:
            frame = shared.frame
            result = dict(shared.result)

        if frame is None:
            time.sleep(0.01)
            continue

        show = cv2.resize(frame, (960, 540))

        main = result.get("main")
        if main is not None:
            bbox = main.get("bbox")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                sx = 960 / float(frame.shape[1])
                sy = 540 / float(frame.shape[0])
                x1, y1, x2, y2 = (
                    int(x1 * sx), int(y1 * sy),
                    int(x2 * sx), int(y2 * sy),
                )
                action = main.get("action", "unknown")
                conf = float(main.get("confidence", 0.0))
                state = main.get("state", "")
                color = (0, 0, 255) if action == "fall_down" else (0, 255, 0)
                if state in ("partial", "low_quality", "ambiguous", "contaminated"):
                    color = (0, 165, 255)
                cv2.rectangle(show, (x1, y1), (x2, y2), color, 2)
                label = "ID%d %s %.2f [%s]" % (
                    main.get("track_id", -1), action, conf, state
                )
                cv2.putText(
                    show, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )

        now = time.time()
        if now - last > 0.5:
            fps = 1.0 / max(1e-6, now - last)
            last = now
        cv2.putText(
            show, "FPS %.1f" % fps, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2,
        )

        if headless:
            if writer is None:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    15.0, (960, 540),
                )
            writer.write(show)
            time.sleep(0.02)
        else:
            cv2.imshow("Single Person Behavior Demo", show)
            key = cv2.waitKey(1)
            if key == 27:
                shared.running = False
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="single person realtime behavior test")
    parser.add_argument(
        "--source", nargs="?", default=None,
        help="video file / rtsp url (default: RK_SOURCE or videos/standing/stand_002.mp4)",
    )
    parser.add_argument("--headless", action="store_true",
                        help="no window, write annotated video")
    parser.add_argument(
        "--interval", type=float, default=0.05,
        help="AI 处理间隔秒（默认 0.05）",
    )
    parser.add_argument(
        "--out", default=os.path.join(PROJECT_ROOT, "output", "single_realtime_result.mp4")
    )
    args = parser.parse_args()

    source = args.source or os.environ.get("RK_SOURCE") or os.path.join(
        PROJECT_ROOT, "data", "videos", "standing", "stand_002.mp4"
    )
    if not os.path.isabs(source) and "://" not in source:
        source = os.path.join(PROJECT_ROOT, source)

    width = int(os.environ.get("RK_FRAME_WIDTH", "640"))
    height = int(os.environ.get("RK_FRAME_HEIGHT", "360"))

    print("=" * 56)
    print("  Single Person Realtime Test")
    print("=" * 56)
    print("SOURCE:", source)
    print("HEADLESS:", args.headless)
    print("BUILDING ENGINE ...")

    engine = build_engine()
    shared = Shared()

    threads = [
        threading.Thread(target=capture_worker, args=(shared, source, width, height), daemon=True),
        threading.Thread(target=ai_worker, args=(shared, engine, args.interval), daemon=True),
        threading.Thread(target=display_worker, args=(shared, args.headless, args.out), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while shared.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP]")
        shared.running = False

    time.sleep(0.3)
    print("EXIT")


if __name__ == "__main__":
    main()
