import copy
import threading


class SharedData:
    def __init__(self):
        self.lock = threading.Lock()
        self._events = []
        self._security_states = {}
        self._security_detections = []
        self._zone_overlays = []

        self._frame = None
        self._frame_id = -1

        self._detection_frame = None
        self._detection_frame_id = -1
        self._detections = {}

        self._behaviors = {}
        self._behavior_frame_id = -1

        self.running = True
        self._fps = 0.0

    def set_frame(self, frame):
        with self.lock:
            self._frame = frame
            self._frame_id += 1

    def get_frame(self):
        with self.lock:
            return self._frame

    def get_frame_packet(self, copy_frame=False):
        with self.lock:
            if self._frame is None:
                return None, -1
            frame = self._frame.copy() if copy_frame else self._frame
            return frame, self._frame_id

    def publish_detection_packet(self, frame_id, frame, detections):
        with self.lock:
            self._detection_frame_id = frame_id
            self._detection_frame = None if frame is None else frame.copy()
            self._detections = copy.deepcopy(detections)

    def get_detection_packet(self):
        with self.lock:
            return {
                "frame_id": self._detection_frame_id,
                "frame": None if self._detection_frame is None else self._detection_frame.copy(),
                "detections": copy.deepcopy(self._detections),
            }

    def set_results(self, results):
        self.publish_detection_packet(self._frame_id, self._frame, results)

    def set_detections(self, results):
        self.set_results(results)

    def get_results(self):
        return self.get_detections()

    def get_detections(self):
        with self.lock:
            return copy.deepcopy(self._detections)

    def publish_behaviors(self, frame_id, behaviors):
        with self.lock:
            self._behavior_frame_id = frame_id
            self._behaviors = copy.deepcopy(behaviors)

    def set_behaviors(self, behaviors):
        self.publish_behaviors(self._detection_frame_id, behaviors)

    def get_behaviors(self):
        with self.lock:
            return copy.deepcopy(self._behaviors)

    def get_behavior_frame_id(self):
        with self.lock:
            return self._behavior_frame_id

    def get_result_packet(self):
        with self.lock:
            return {
                "frame_id": self._detection_frame_id,
                "frame": None if self._detection_frame is None else self._detection_frame.copy(),
                "detections": copy.deepcopy(self._detections),
                "behaviors": copy.deepcopy(self._behaviors),
                "behavior_frame_id": self._behavior_frame_id,
            }

    def set_fps(self, fps):
        with self.lock:
            self._fps = fps

    def get_fps(self):
        with self.lock:
            return self._fps

    def snapshot(self):
        with self.lock:
            return {
                "frame_id": self._frame_id,
                "frame": None if self._frame is None else self._frame.copy(),
                "detection_frame_id": self._detection_frame_id,
                "detection_frame": None if self._detection_frame is None else self._detection_frame.copy(),
                "behavior_frame_id": self._behavior_frame_id,
                "detections": copy.deepcopy(self._detections),
                "behaviors": copy.deepcopy(self._behaviors),
                "fps": self._fps,
            }

    # === SECURITY PATCH (security_monitor) ===
    # 以下方法由 security_patch 增量追加，用于安防事件/区域状态共享。
    # 原有方法未被改动。

    def set_events(self, events):
        with self.lock:
            self._events = copy.deepcopy(events)

    def get_events(self):
        with self.lock:
            return copy.deepcopy(self._events)

    def set_security_states(self, security_states):
        with self.lock:
            self._security_states = copy.deepcopy(security_states)

    def get_security_states(self):
        with self.lock:
            return copy.deepcopy(self._security_states)

    def set_security_detections(self, detections):
        with self.lock:
            self._security_detections = copy.deepcopy(detections)

    def get_security_detections(self):
        with self.lock:
            return copy.deepcopy(self._security_detections)

    def set_zone_overlays(self, zones):
        with self.lock:
            self._zone_overlays = copy.deepcopy(zones)

    def get_zone_overlays(self):
        with self.lock:
            return copy.deepcopy(self._zone_overlays)
