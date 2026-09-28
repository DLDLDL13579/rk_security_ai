import threading
import time

from engine.behavior_filter import BehaviorFilter
from engine.feature import PoseFeatureExtractor
from engine.sequence_buffer import PoseSequenceBuffer
from npu.mediapipe_pose import MediaPipePose


CONF_THRESH = 0.35
MIN_BOX_SIZE = 24


class BehaviorThread(threading.Thread):
    def __init__(self, shared, behavior_model):
        super().__init__(name="BehaviorThread")
        self.shared = shared
        self.behavior = behavior_model
        self.pose = MediaPipePose(model_complexity=0)
        self.feature = PoseFeatureExtractor()
        self.buffer = PoseSequenceBuffer(max_len=16)
        self.filter = BehaviorFilter(window=8, conf_thresh=CONF_THRESH)
        self.last_behavior = {}
        self.last_ready_output = {}
        self.last_frame_id = -1
        print("[Behavior Thread] initialized")

    def log_behavior(self, pid, raw_action, stable_action, score, ready):
        state = (raw_action, stable_action, round(float(score), 3), bool(ready))
        old = self.last_behavior.get(pid)
        if old == state:
            return
        self.last_behavior[pid] = state
        print(
            f"[Behavior] id={pid} raw={raw_action} stable={stable_action} "
            f"score={float(score):.3f} ready={bool(ready)}"
        )

    def _cleanup_inactive(self, active_ids):
        stale_ids = [pid for pid in list(self.last_behavior.keys()) if pid not in active_ids]
        for pid in stale_ids:
            self.last_behavior.pop(pid, None)
            self.last_ready_output.pop(pid, None)
            self.buffer.clear(pid)
            self.filter.clear(pid)

    def _default_behavior(self, pid, bbox):
        return {
            "id": pid,
            "bbox": list(bbox),
            "raw_action": "unknown",
            "raw_confidence": 0.0,
            "stable_action": "warming_up",
            "stable_confidence": 0.0,
            "action": "warming_up",
            "confidence": 0.0,
            "ready": False,
            "buffer_len": self.buffer.length(pid),
        }

    def _clip_bbox(self, bbox, frame_shape):
        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame_shape[:2]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        return x1, y1, x2, y2

    def run(self):
        print("[Behavior Thread] started")
        while self.shared.running:
            frame, frame_id = self.shared.get_frame_packet()
            if frame is None:
                time.sleep(0.01)
                continue
            if frame_id == self.last_frame_id:
                time.sleep(0.005)
                continue
            self.last_frame_id = frame_id

            detections = self.shared.get_detections()
            if not isinstance(detections, dict):
                time.sleep(0.01)
                continue

            active_ids = set(detections.keys())
            behavior_output = {}

            for pid, obj in detections.items():
                bbox = obj.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue

                behavior_output[pid] = self._default_behavior(pid, bbox)

                try:
                    x1, y1, x2, y2 = self._clip_bbox(bbox, frame.shape)
                    if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    kpts = self.pose.detect(crop)
                    if kpts is None:
                        continue

                    feature = self.feature.extract(kpts, track_id=pid)
                    if feature is None:
                        continue

                    seq = self.buffer.update(pid, feature)
                    current_len = self.buffer.length(pid)
                    behavior_output[pid]["buffer_len"] = current_len
                    if seq is None:
                        cached = self.last_ready_output.get(pid)
                        if cached is not None and current_len >= self.buffer.max_len:
                            obj["behavior"] = cached["action"]
                            obj["behavior_score"] = cached["confidence"]
                            behavior_output[pid] = dict(cached)
                            behavior_output[pid]["bbox"] = [x1, y1, x2, y2]
                            behavior_output[pid]["buffer_len"] = current_len
                        else:
                            obj["behavior"] = "warming_up"
                            obj["behavior_score"] = 0.0
                            self.log_behavior(pid, "unknown", "warming_up", 0.0, False)
                        continue

                    raw_action, score = self.behavior.forward(seq)
                    score = float(score)
                    stable_action = self.filter.update(pid, raw_action, score)

                    obj["behavior"] = stable_action
                    obj["behavior_score"] = score
                    behavior_output[pid] = {
                        "id": pid,
                        "bbox": [x1, y1, x2, y2],
                        "raw_action": str(raw_action),
                        "raw_confidence": score,
                        "stable_action": str(stable_action),
                        "stable_confidence": score,
                        "action": str(stable_action),
                        "confidence": score,
                        "ready": True,
                        "buffer_len": current_len,
                    }
                    self.last_ready_output[pid] = dict(behavior_output[pid])
                    self.log_behavior(pid, raw_action, stable_action, score, True)
                except Exception as exc:
                    print(f"[Behavior ERROR] id={pid} {exc}")

            self._cleanup_inactive(active_ids)
            self.shared.set_detections(detections)
            self.shared.set_behaviors(behavior_output)
            time.sleep(0.02)

    def stop(self):
        self.shared.running = False
        self.buffer.clear()
        self.filter.clear()
        self.pose.release()
        try:
            self.behavior.release()
        except Exception:
            pass
