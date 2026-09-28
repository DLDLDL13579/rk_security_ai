import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from engine.behavior_filter import BehaviorFilter
from engine.feature import PoseFeatureExtractor
from engine.security_event_manager import SecurityEventManager
from engine.sequence_buffer import PoseSequenceBuffer
from engine.zone_manager import ZoneManager
from npu.behavior_rknn import BehaviorRKNN
from pose.mediapipe_pose import MediaPipePose


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(ROOT_DIR, "videos", "walking", "walk_001.mp4")
BEHAVIOR_MODEL = os.path.join(ROOT_DIR, "npu", "onnx-rknn", "behavior_tcn.rknn")
RECORD_DIR = os.path.join(ROOT_DIR, "recordings")
WINDOW_TITLE = "Single Person Security Demo"

PROCESS_SIZE = (640, 360)
DISPLAY_SIZE = (960, 540)
AI_INTERVAL = 0.12
FALL_CONFIDENCE = 0.55
EVENT_HOLD_SECONDS = 4.0
LOOP_VIDEO = False

SECURITY_ZONES = [
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

ACTION_LABELS = {
    "standing": "Standing",
    "walking": "Walking",
    "squat": "Squat",
    "bend": "Bend",
    "fall_down": "Fall",
    "warming_up": "WarmingUp",
    "unknown": "Unknown",
}

EVENT_LABELS = {
    "FALL_DETECTED": "Fall Detected",
    "INTRUSION_DETECTED": "Area Intrusion",
    "LOITERING_DETECTED": "Loitering",
}

CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (15, 17),
    (17, 19),
    (19, 21),
    (16, 18),
    (18, 20),
    (20, 22),
    (11, 23),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (12, 24),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
]


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.result = {}
        self.running = True
        self.source_fps = 0.0
        self.display_fps = 0.0


class PoseSmooth:
    def __init__(self, alpha=0.65):
        self.last = None
        self.alpha = float(alpha)

    def update(self, kpts):
        arr = np.asarray(kpts, dtype=np.float32)
        if arr.shape[0] == 0 or arr.ndim != 2 or arr.shape[1] < 3:
            return None
        arr = arr[:, :3].copy()
        if self.last is None:
            self.last = arr
        else:
            self.last = self.alpha * self.last + (1.0 - self.alpha) * arr
        return self.last


shared = SharedState()


def to_action_label(action):
    return ACTION_LABELS.get(str(action or "").strip().lower(), str(action or "Unknown"))


def to_event_label(event_type):
    return EVENT_LABELS.get(str(event_type or "").strip().upper(), str(event_type or "Event"))


def check_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def draw_skeleton(img, pts):
    if pts is None:
        return

    height, width = img.shape[:2]
    xy = []
    for point in pts:
        x = int(point[0] * width)
        y = int(point[1] * height)
        xy.append((x, y))
        if point[2] > 0.3:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    for a, b in CONNECTIONS:
        if a >= len(xy) or b >= len(xy):
            continue
        cv2.line(img, xy[a], xy[b], (255, 200, 0), 1, cv2.LINE_AA)


def bbox_from_pose(pts, width, height, visibility_thresh=0.3, pad=20):
    if pts is None:
        return None

    xs = []
    ys = []
    for point in pts:
        if point[2] > visibility_thresh:
            xs.append(int(point[0] * width))
            ys.append(int(point[1] * height))

    if len(xs) < 5:
        return None

    return (
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(width, max(xs) + pad),
        min(height, max(ys) + pad),
    )


def merge_alert_state(base_state, alert_name, zone_name=""):
    state = dict(base_state or {})
    alerts = list(state.get("alerts", []))
    zones = list(state.get("zones", []))
    if alert_name and alert_name not in alerts:
        alerts.append(alert_name)
    if zone_name and zone_name not in zones:
        zones.append(zone_name)
    state["alerts"] = alerts
    state["zones"] = zones
    return state


def make_default_result():
    return {
        "pose": None,
        "bbox": None,
        "raw_action": "unknown",
        "action": "warming_up",
        "score": 0.0,
        "ready": False,
        "buffer_len": 0,
        "security_state": {},
        "events": [],
        "zones": SECURITY_ZONES,
    }


def log_behavior(last_state, raw_action, stable_action, score, ready):
    state = (str(raw_action), str(stable_action), round(float(score), 3), bool(ready))
    if state == last_state[0]:
        return
    last_state[0] = state
    print(
        f"[Behavior] raw={raw_action} stable={stable_action} "
        f"score={float(score):.3f} ready={bool(ready)}"
    )


def log_event(last_state, event):
    signature = (
        event.get("event_type"),
        event.get("person_id"),
        event.get("zone_id"),
    )
    if signature == last_state[0]:
        return
    last_state[0] = signature
    message = to_event_label(event.get("event_type", "EVENT"))
    zone_name = str(event.get("zone_name", "")).strip()
    duration = event.get("duration")
    if zone_name:
        message += f" | {zone_name}"
    if duration is not None:
        message += f" | {float(duration):.1f}s"
    print(f"[Security] {message}")


def video_worker():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[Video ERROR] cannot open source: {VIDEO_SOURCE}")
        shared.running = False
        return

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1.0 or source_fps > 240.0:
        source_fps = 25.0
    frame_interval = 1.0 / source_fps if source_fps > 0 else 0.03
    shared.source_fps = source_fps
    print(f"[Video] source={VIDEO_SOURCE}")
    print(f"[Video] fps={source_fps:.2f} size={PROCESS_SIZE[0]}x{PROCESS_SIZE[1]}")

    next_tick = time.time()
    try:
        while shared.running:
            ret, frame = cap.read()
            if not ret:
                if LOOP_VIDEO:
                    cap.release()
                    time.sleep(0.2)
                    cap = cv2.VideoCapture(VIDEO_SOURCE)
                    next_tick = time.time()
                    continue
                shared.running = False
                break

            frame = cv2.resize(frame, PROCESS_SIZE)
            with shared.lock:
                shared.frame = frame

            next_tick += frame_interval
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.time()
    finally:
        cap.release()
        print("[Video] stopped")


def ai_worker():
    pose = MediaPipePose()
    feature = PoseFeatureExtractor()
    buffer = PoseSequenceBuffer(max_len=16)
    model = BehaviorRKNN(BEHAVIOR_MODEL)
    smooth = PoseSmooth()
    behavior_filter = BehaviorFilter(window=6, conf_thresh=0.35)
    zone_manager = ZoneManager(SECURITY_ZONES)
    event_manager = SecurityEventManager(default_cooldown=2.0)

    last_infer_at = 0.0
    last_ready = None
    recent_events = []
    behavior_log_state = [None]
    event_log_state = [None]

    try:
        while shared.running:
            with shared.lock:
                frame = None if shared.frame is None else shared.frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - last_infer_at < AI_INTERVAL:
                time.sleep(0.01)
                continue
            last_infer_at = now

            result = make_default_result()

            kpts = pose.detect(frame)
            if kpts is None:
                with shared.lock:
                    result["events"] = list(recent_events)
                    shared.result = result
                continue

            smooth_pts = smooth.update(kpts)
            if smooth_pts is None:
                continue

            feat = feature.extract(kpts)
            if feat is None:
                continue

            seq = buffer.update(0, feat)
            bbox = bbox_from_pose(
                smooth_pts,
                frame.shape[1],
                frame.shape[0],
            )

            raw_action = "unknown"
            stable_action = "warming_up"
            score = 0.0
            ready = False
            buffer_len = buffer.length(0)

            if seq is not None:
                raw_action, score = model.forward(seq)
                score = float(score)
                stable_action = behavior_filter.update(0, raw_action, score)
                ready = True
                last_ready = {
                    "raw_action": str(raw_action),
                    "action": str(stable_action),
                    "score": score,
                }
            elif last_ready is not None and buffer_len >= buffer.max_len:
                raw_action = last_ready["raw_action"]
                stable_action = last_ready["action"]
                score = float(last_ready["score"])
                ready = True

            log_behavior(behavior_log_state, raw_action, stable_action, score, ready)

            security_state = {}
            event_output = []
            if bbox is not None:
                persons = {
                    0: {
                        "bbox": list(bbox),
                        "score": score,
                    }
                }
                states, zone_events = zone_manager.evaluate(persons, now=now)
                security_state = states.get(0, {})
                for item in zone_events:
                    event = event_manager.emit(item, now=now, cooldown=2.0)
                    if event is not None:
                        event_output.append(event)

                if ready and stable_action == "fall_down" and score >= FALL_CONFIDENCE:
                    security_state = merge_alert_state(security_state, "FALL")
                    event = event_manager.emit(
                        {
                            "event_type": "FALL_DETECTED",
                            "source_key": "fall:0",
                            "person_id": 0,
                            "bbox": list(bbox),
                            "score": score,
                        },
                        now=now,
                        cooldown=2.0,
                    )
                    if event is not None:
                        event_output.append(event)

            for item in event_output:
                log_event(event_log_state, item)

            recent_events.extend(event_output)
            recent_events = [
                item
                for item in recent_events
                if now - float(item.get("ts", 0.0)) <= EVENT_HOLD_SECONDS
            ][-6:]

            result = {
                "pose": smooth_pts,
                "bbox": bbox,
                "raw_action": str(raw_action),
                "action": str(stable_action),
                "score": float(score),
                "ready": bool(ready),
                "buffer_len": int(buffer_len),
                "security_state": dict(security_state),
                "events": list(recent_events),
                "zones": SECURITY_ZONES,
            }

            with shared.lock:
                shared.result = result
    finally:
        pose.release()
        try:
            model.release()
        except Exception:
            pass
        print("[AI] stopped")


def ensure_writer(state, writer_state, frame):
    if writer_state["writer"] is not None:
        return writer_state["writer"]

    os.makedirs(RECORD_DIR, exist_ok=True)
    target_fps = shared.source_fps if shared.source_fps > 1.0 else 20.0
    path = os.path.join(
        RECORD_DIR,
        datetime.now().strftime("security_session_%Y%m%d_%H%M%S.mp4"),
    )
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, target_fps, (width, height))
    if not writer.isOpened():
        print(f"[Record ERROR] cannot open writer: {path}")
        return None
    writer_state["writer"] = writer
    writer_state["path"] = path
    print(f"[Record] {path}")
    return writer


def draw_zones(img, zones):
    for zone in zones or []:
        rect = zone.get("rect", [])
        if len(rect) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in rect]
        color = tuple(zone.get("color", (0, 200, 255)))
        label = f"{zone.get('name', 'Zone')} | {zone.get('type', 'zone')}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def draw_person_panel(img, result):
    bbox = result.get("bbox")
    if bbox is None:
        return

    x1, y1, x2, y2 = [int(v) for v in bbox]
    security_state = result.get("security_state", {})
    alerts = list(security_state.get("alerts", []))
    zones = list(security_state.get("zones", []))
    color = (0, 255, 0)
    if "FALL" in alerts:
        color = (0, 0, 255)
    elif alerts:
        color = (0, 200, 255)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    lines = [
        f"person | {result.get('score', 0.0):.2f}",
        f"raw: {to_action_label(result.get('raw_action', 'unknown'))}",
    ]
    if result.get("ready", False):
        lines.append(f"stable: {to_action_label(result.get('action', 'unknown'))}")
    else:
        lines.append(f"stable: WarmingUp ({result.get('buffer_len', 0)}/16)")
    if alerts:
        lines.append(f"alert: {'/'.join(alerts)}")
    if zones:
        lines.append(f"zone: {'/'.join(zones)}")

    panel_y = max(18, y1 - 18 * len(lines) - 8)
    for idx, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (x1, panel_y + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )


def draw_event_panel(img, events):
    if not events:
        return

    cv2.rectangle(img, (8, 54), (420, 54 + 24 * (len(events[:5]) + 1)), (0, 0, 0), -1)
    cv2.putText(
        img,
        "Security Events",
        (16, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )
    for idx, item in enumerate(events[:5]):
        message = to_event_label(item.get("event_type", "EVENT"))
        zone_name = str(item.get("zone_name", "")).strip()
        if zone_name:
            message += f" | {zone_name}"
        duration = item.get("duration")
        if duration is not None:
            message += f" | {float(duration):.1f}s"
        cv2.putText(
            img,
            message,
            (16, 98 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


def display_worker():
    last_tick = time.time()
    writer_state = {"writer": None, "path": ""}
    try:
        while shared.running:
            with shared.lock:
                frame = None if shared.frame is None else shared.frame.copy()
                result = dict(shared.result)

            if frame is None:
                time.sleep(0.01)
                continue

            image = frame.copy()
            draw_zones(image, result.get("zones", SECURITY_ZONES))
            draw_skeleton(image, result.get("pose"))
            draw_person_panel(image, result)
            draw_event_panel(image, result.get("events", []))

            now = time.time()
            dt = now - last_tick
            if dt > 0:
                shared.display_fps = 1.0 / dt
            last_tick = now

            cv2.putText(
                image,
                f"Display FPS: {shared.display_fps:.1f}",
                (18, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 200, 0),
                2,
            )
            cv2.putText(
                image,
                f"Source FPS: {shared.source_fps:.2f}",
                (18, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 0),
                1,
            )

            writer = ensure_writer(shared, writer_state, image)
            if writer is not None:
                writer.write(image)

            cv2.imshow(WINDOW_TITLE, cv2.resize(image, DISPLAY_SIZE))
            if cv2.waitKey(1) & 0xFF == 27:
                shared.running = False
                break
    finally:
        writer = writer_state.get("writer")
        if writer is not None:
            writer.release()
            print(f"[Record] saved -> {writer_state.get('path', '')}")
        cv2.destroyAllWindows()
        print("[Display] stopped")


def main():
    print("=" * 60)
    print(" Single Person Security Demo")
    print(" Pose -> Behavior -> Fall / Intrusion / Loitering")
    print("=" * 60)

    check_file(VIDEO_SOURCE, "video source")
    check_file(BEHAVIOR_MODEL, "behavior model")

    workers = [
        threading.Thread(target=video_worker, name="VideoWorker", daemon=True),
        threading.Thread(target=ai_worker, name="AIWorker", daemon=True),
        threading.Thread(target=display_worker, name="DisplayWorker", daemon=True),
    ]

    for worker in workers:
        worker.start()

    try:
        while shared.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        shared.running = False
    finally:
        shared.running = False
        for worker in workers:
            worker.join(timeout=2.0)
        print("[MAIN] stopped")


if __name__ == "__main__":
    main()
