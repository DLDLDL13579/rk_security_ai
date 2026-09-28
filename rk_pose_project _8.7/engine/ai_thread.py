import os
import time
import traceback


class AIThread:
    def __init__(self, engine, shared):
        self.engine = engine
        self.shared = shared

        self.detect_interval = float(
            os.environ.get(
                "RK_DETECT_INTERVAL",
                "0.015",
            )
        )
        self.behavior_interval = float(
            os.environ.get(
                "RK_BEHAVIOR_LOOP_INTERVAL",
                str(self.engine.behavior_interval),
            )
        )

        self.last_detect_time = 0.0
        self.last_behavior_time = 0.0

        print("[AI Thread] initialized")

    def run(self):
        print("[AI Thread] started")

        while self.shared.running:
            frame, frame_id = self.shared.get_frame_packet(copy_frame=False)

            if frame is None:
                time.sleep(0.005)
                continue

            now = time.time()
            did_work = False

            try:
                if now - self.last_detect_time >= self.detect_interval:
                    detection_result = self.engine.process_detections(frame)
                    det_output = {}

                    for det in detection_result.get("detections", []):
                        if not hasattr(det, "track_id"):
                            continue

                        tid = det.track_id
                        det_output[tid] = {
                            "id": tid,
                            "bbox": det.bbox,
                            "name": det.cls,
                            "score": float(det.score),
                        }

                    self.shared.publish_detection_packet(frame_id, frame, det_output)
                    self.last_detect_time = now
                    did_work = True

                if now - self.last_behavior_time >= self.behavior_interval:
                    behavior_result = self.engine.process_behaviors(frame, frame_id=frame_id)
                    behavior_output = {}

                    for b in behavior_result.get("behaviors", []):
                        tid = b.get("track_id", -1)
                        behavior_output[tid] = {
                            "id": tid,
                            "bbox": b.get("bbox"),
                            "action": b.get("action", "unknown"),
                            "confidence": float(b.get("confidence", 0.0)),
                            "state": b.get("state", "valid"),
                            "proximity_state": b.get("proximity_state", "normal"),
                            "pose_quality": float(b.get("pose_quality", 0.0)),
                        }

                    self.shared.publish_behaviors(frame_id, behavior_output)
                    self.last_behavior_time = now
                    did_work = True

                if did_work:
                    print(
                        "[AI]",
                        "person:",
                        len(self.shared.get_detections()),
                        "behavior:",
                        len(self.shared.get_behaviors()),
                    )

            except Exception:
                print("[AI ERROR]")
                traceback.print_exc()

            time.sleep(0.001 if did_work else 0.005)

    def stop(self):
        self.shared.running = False
