import threading
import time

from engine.tracker import SimpleTracker


PERSON_CONF_THRESH = 0.35


class YOLOThread(threading.Thread):
    def __init__(self, shared, detector):
        super().__init__(name="YOLOThread")
        self.shared = shared
        self.detector = detector
        self.tracker = SimpleTracker(max_lost=15, iou_threshold=0.35)
        self.last_log_time = 0.0
        self.last_frame_id = -1
        print("[YOLO Thread] initialized")

    def print_status(self, persons):
        now = time.time()
        if now - self.last_log_time < 5.0:
            return
        self.last_log_time = now
        print(f"[YOLO] persons={len(persons)}")
        for pid, obj in persons.items():
            print(f"[YOLO] id={pid} score={obj['score']:.3f} bbox={obj['bbox']}")

    def run(self):
        print("[YOLO Thread] started")
        while self.shared.running:
            frame, frame_id = self.shared.get_frame_packet()
            if frame is None:
                time.sleep(0.01)
                continue
            if frame_id == self.last_frame_id:
                time.sleep(0.005)
                continue
            self.last_frame_id = frame_id

            try:
                results = self.detector.detect(frame) or []
                person_detections = []
                for obj in results:
                    if obj.get("name", "") != "person":
                        continue
                    score = float(obj.get("score", 0.0))
                    if score < PERSON_CONF_THRESH:
                        continue
                    person_detections.append(
                        {
                            "bbox": [int(v) for v in obj.get("bbox", [0, 0, 0, 0])],
                            "score": score,
                            "name": "person",
                        }
                    )

                tracked = self.tracker.update(person_detections)
                persons = {}
                for det in tracked:
                    pid = int(det.get("track_id", det.get("id", -1)))
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    if pid < 0 or len(bbox) != 4:
                        continue
                    persons[pid] = {
                        "id": pid,
                        "track_id": pid,
                        "bbox": [int(v) for v in bbox],
                        "score": float(det.get("score", 0.0)),
                        "name": "person",
                        "behavior": "",
                    }

                if persons:
                    self.print_status(persons)

                self.shared.set_detections(persons)
            except Exception as exc:
                print("[YOLO ERROR]", exc)

            time.sleep(0.01)

    def stop(self):
        self.shared.running = False
