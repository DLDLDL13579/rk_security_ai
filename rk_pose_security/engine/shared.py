import copy
import threading


class SharedData:
    def __init__(self):
        self.lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._detections = {}
        self._behaviors = {}
        self._events = []
        self._security_states = {}
        self._security_detections = []
        self._zone_overlays = []
        self.running = True
        self._fps = 0.0
        self._source_fps = 0.0
        self._source_is_file = False
        self._source_size = (0, 0)

    def set_frame(self, frame):
        with self.lock:
            self._frame = frame
            self._frame_id += 1

    def get_frame(self):
        with self.lock:
            return self._frame

    def get_frame_packet(self):
        with self.lock:
            return self._frame, self._frame_id

    def set_results(self, results):
        with self.lock:
            self._detections = copy.deepcopy(results)

    def set_detections(self, results):
        self.set_results(results)

    def get_results(self):
        return self.get_detections()

    def get_detections(self):
        with self.lock:
            return copy.deepcopy(self._detections)

    def set_behaviors(self, behaviors):
        with self.lock:
            self._behaviors = copy.deepcopy(behaviors)

    def get_behaviors(self):
        with self.lock:
            return copy.deepcopy(self._behaviors)

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

    def set_fps(self, fps):
        with self.lock:
            self._fps = fps

    def get_fps(self):
        with self.lock:
            return self._fps

    def set_source_info(self, fps=0.0, is_file=False, size=None):
        with self.lock:
            self._source_fps = float(fps or 0.0)
            self._source_is_file = bool(is_file)
            if isinstance(size, (list, tuple)) and len(size) == 2:
                self._source_size = (int(size[0]), int(size[1]))

    def get_source_fps(self):
        with self.lock:
            return self._source_fps

    def is_file_source(self):
        with self.lock:
            return self._source_is_file

    def get_source_size(self):
        with self.lock:
            return self._source_size

    def snapshot(self):
        with self.lock:
            return {
                "frame": None if self._frame is None else self._frame.copy(),
                "detections": copy.deepcopy(self._detections),
                "behaviors": copy.deepcopy(self._behaviors),
                "events": copy.deepcopy(self._events),
                "security_states": copy.deepcopy(self._security_states),
                "security_detections": copy.deepcopy(self._security_detections),
                "zone_overlays": copy.deepcopy(self._zone_overlays),
                "fps": self._fps,
                "source_fps": self._source_fps,
                "source_is_file": self._source_is_file,
                "source_size": self._source_size,
            }
