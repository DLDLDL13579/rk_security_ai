import os
import cv2
import time


class DisplayThread:
    def __init__(self, shared, width=960, height=540):
        self.shared = shared
        self.width = width
        self.height = height

        self.last_time = time.time()
        self.fps = 0.0
        self.track_motion = {}
        self.predict_lag_frames = max(
            0.0,
            float(os.environ.get("RK_DISPLAY_PREDICT_LAG", "2.6")),
        )
        self.predict_alpha = min(
            0.95,
            max(0.0, float(os.environ.get("RK_DISPLAY_PREDICT_ALPHA", "0.48"))),
        )
        self.headless = os.environ.get(
            "RK_HEADLESS", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_path = os.environ.get(
            "RK_OUTPUT_VIDEO",
            os.path.join(project_root, "output", "behavior_result.mp4"),
        )
        self.video_writer = None

    def _ensure_writer(self, frame_shape):
        if self.video_writer is not None:
            return
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (self.width, self.height),
        )
        print("[Display] headless recording ->", self.output_path)

    def update_fps(self):
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now

    def clip_bbox(self, bbox, width, height):
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width - 1))
        y2 = max(0, min(int(round(y2)), height - 1))
        return [x1, y1, x2, y2]

    def scale_bbox(self, bbox, src_width, src_height):
        if src_width <= 0 or src_height <= 0:
            return bbox

        scale_x = self.width / float(src_width)
        scale_y = self.height / float(src_height)
        x1, y1, x2, y2 = bbox
        return [
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        ]

    def predict_bbox(self, pid, bbox, result_frame_id, current_frame_id):
        x1, y1, x2, y2 = map(float, bbox)
        lag_frames = max(0, current_frame_id - result_frame_id)
        mem = self.track_motion.get(pid)

        if mem is None:
            self.track_motion[pid] = {
                "bbox": [x1, y1, x2, y2],
                "frame_id": result_frame_id,
                "vx": 0.0,
                "vy": 0.0,
            }
            return [x1, y1, x2, y2]

        last_bbox = mem["bbox"]
        last_frame_id = mem["frame_id"]
        dt = max(1, result_frame_id - last_frame_id)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        last_cx = (last_bbox[0] + last_bbox[2]) * 0.5
        last_cy = (last_bbox[1] + last_bbox[3]) * 0.5

        vx = (cx - last_cx) / dt
        vy = (cy - last_cy) / dt

        mem["bbox"] = [x1, y1, x2, y2]
        mem["frame_id"] = result_frame_id
        mem["vx"] = (1.0 - self.predict_alpha) * mem["vx"] + self.predict_alpha * vx
        mem["vy"] = (1.0 - self.predict_alpha) * mem["vy"] + self.predict_alpha * vy

        lag = min(self.predict_lag_frames, float(lag_frames) + 0.6)
        dx = mem["vx"] * lag
        dy = mem["vy"] * lag

        return [
            x1 + dx,
            y1 + dy,
            x2 + dx,
            y2 + dy,
        ]

    def draw_person(self, img, pid, det, behavior, result_frame_id, current_frame_id, source_shape):
        bbox = det.get("bbox")
        if bbox is None:
            return

        predicted_bbox = self.predict_bbox(
            pid,
            bbox,
            result_frame_id,
            current_frame_id,
        )
        scaled_bbox = self.scale_bbox(
            predicted_bbox,
            source_shape[1],
            source_shape[0],
        )
        x1, y1, x2, y2 = self.clip_bbox(scaled_bbox, self.width, self.height)

        if x2 <= x1 or y2 <= y1:
            return

        score = float(det.get("score", 0.0))
        name = det.get("name", "person")
        action = ""
        action_score = 0.0
        status = ""

        if behavior is not None:
            action = behavior.get("action", "")
            action_score = float(behavior.get("confidence", 0.0))
            status = behavior.get("state", "")

        color = (0, 255, 0)
        if action == "fall_down":
            color = (0, 0, 255)
        elif status in ("ambiguous", "contaminated", "partial", "low_quality"):
            color = (0, 165, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{pid} {name} {score:.2f}"
        if action:
            label += f" | {action} {action_score:.2f}"
        if status not in ("", "valid", "warming"):
            label += f" [{status}]"

        font_scale = 0.62
        thickness = 2
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        label_top = max(0, y1 - th - 12)
        label_bottom = min(self.height - 1, y1)
        label_right = min(self.width - 1, x1 + tw + 10)

        cv2.rectangle(
            img,
            (x1, label_top),
            (label_right, label_bottom),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1 + 5, label_bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
        )

    def run(self):
        print("[Display] Thread Started")

        while self.shared.running:
            detection_packet = self.shared.get_detection_packet()
            frame, current_frame_id = self.shared.get_frame_packet(copy_frame=True)

            if frame is None:
                frame = detection_packet.get("frame")
                current_frame_id = detection_packet.get("frame_id", -1)

            result_frame_id = detection_packet.get("frame_id", -1)
            frame = frame if frame is not None else detection_packet.get("frame")

            if frame is None:
                time.sleep(0.005)
                continue

            source_shape = frame.shape[:2]
            show = cv2.resize(
                frame,
                (self.width, self.height),
            )

            detections = detection_packet.get("detections", {})
            behaviors = self.shared.get_behaviors()

            if isinstance(detections, dict):
                for pid, det in detections.items():
                    behavior = behaviors.get(pid) if isinstance(behaviors, dict) else None
                    self.draw_person(
                        show,
                        pid,
                        det,
                        behavior,
                        result_frame_id,
                        current_frame_id,
                        source_shape,
                    )

            self.update_fps()
            self.shared.set_fps(self.fps)

            cv2.putText(
                show,
                f"FPS:{self.fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

            if self.headless:
                self._ensure_writer(show.shape)
                if self.video_writer is not None:
                    self.video_writer.write(show)
                time.sleep(0.005)
            else:
                cv2.imshow("RK3588 Smart Security", show)
                key = cv2.waitKey(1)
                if key == 27:
                    self.shared.running = False
                    break

        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("[Display] stopped")
