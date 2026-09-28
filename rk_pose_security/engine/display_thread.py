import os
import time

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


CHINESE_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

ACTION_MAP = {
    "standing": "Standing",
    "walking": "Walking",
    "squat": "Squat",
    "bend": "Bend",
    "fall_down": "Fall",
    "warming_up": "WarmingUp",
    "unknown": "Unknown",
}

ALERT_MAP = {
    "FALL": "FALL",
    "INTRUSION": "INTRUSION",
    "LOITERING": "LOITERING",
}

ZONE_TYPE_MAP = {
    "intrusion": "Intrusion",
    "loiter": "Loiter",
    "both": "Intrusion+Loiter",
}

EVENT_MAP = {
    "FALL_DETECTED": "Fall Detected",
    "INTRUSION_DETECTED": "Area Intrusion",
    "LOITERING_DETECTED": "Loitering",
    "FIRE_DETECTED": "Fire Detected",
    "SMOKE_DETECTED": "Smoke Detected",
}

LABEL_MAP = {
    "person": "Person",
    "fire": "Fire",
    "smoke": "Smoke",
    "flame": "Flame",
    "alert": "Alert",
}


class DisplayThread:
    def __init__(
        self,
        shared,
        width=960,
        height=540,
        save_video=False,
        save_path="",
        record_fps=0.0,
        window_title="RK3588 Security Demo",
    ):
        self.shared = shared
        self.width = int(width)
        self.height = int(height)
        self.last_time = time.time()
        self.fps = 0.0
        self.save_video = bool(save_video)
        self.save_path = str(save_path or "").strip()
        self.record_fps = float(record_fps or 0.0)
        self.window_title = str(window_title)
        self.writer = None
        self._font_cache = {}
        self._text_backend_logged = False
        self.last_frame_id = -1

    def _load_font(self, size):
        if size in self._font_cache:
            return self._font_cache[size]

        font = None
        if ImageFont is not None:
            for path in CHINESE_FONT_CANDIDATES:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, size=size)
                        break
                    except Exception:
                        continue
            if font is None:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

        self._font_cache[size] = font
        return font

    def _to_action(self, action):
        return ACTION_MAP.get(str(action or "").strip().lower(), str(action or "Unknown"))

    def _to_alert(self, alert):
        return ALERT_MAP.get(str(alert or "").strip().upper(), str(alert or ""))

    def _to_zone_type(self, zone_type):
        return ZONE_TYPE_MAP.get(str(zone_type or "").strip().lower(), str(zone_type or "Zone"))

    def _to_label(self, label):
        return LABEL_MAP.get(str(label or "").strip().lower(), str(label or "Object"))

    def _to_event(self, event_type):
        return EVENT_MAP.get(str(event_type or "").strip().upper(), str(event_type or "Event"))

    def _create_text_canvas(self, img):
        if Image is None or ImageDraw is None:
            if not self._text_backend_logged:
                print("[Display WARN] Pillow not found, text rendering falls back to OpenCV")
                self._text_backend_logged = True
            return None, None
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return pil_image, ImageDraw.Draw(pil_image)

    def _finalize_text_canvas(self, pil_image):
        if pil_image is None:
            return None
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    def _measure_text(self, draw, text, size=24):
        text = str(text or "")
        if draw is None:
            return max(1, int(len(text) * size * 0.58)), max(1, int(size * 1.35))
        font = self._load_font(size)
        if font is None:
            return max(1, int(len(text) * size * 0.58)), max(1, int(size * 1.35))
        box = draw.textbbox((0, 0), text, font=font)
        return max(1, box[2] - box[0]), max(1, box[3] - box[1])

    def _put_text(self, img, draw, text, org, color, size=24):
        x, y = int(org[0]), int(org[1])
        text = str(text or "")
        if draw is not None:
            font = self._load_font(size)
            if font is not None:
                draw.text((x, y), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
                return
        cv2.putText(
            img,
            text,
            (x, y + size),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, size / 34.0),
            color,
            2,
        )

    def _draw_rect(self, img, draw, pt1, pt2, color, thickness=2, fill=False):
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        if fill:
            cv2.rectangle(img, pt1, pt2, color, -1)
        else:
            cv2.rectangle(img, pt1, pt2, color, int(thickness))

        if draw is not None:
            rgb = (int(color[2]), int(color[1]), int(color[0]))
            if fill:
                draw.rectangle([pt1, pt2], fill=rgb)
            else:
                for offset in range(max(1, int(thickness))):
                    draw.rectangle(
                        [
                            (pt1[0] - offset, pt1[1] - offset),
                            (pt2[0] + offset, pt2[1] + offset),
                        ],
                        outline=rgb,
                    )

    def _ensure_writer(self, frame):
        if not self.save_video or self.writer is not None or not self.save_path:
            return

        height, width = frame.shape[:2]
        save_dir = os.path.dirname(self.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        target_fps = self.record_fps
        source_fps = float(self.shared.get_source_fps() or 0.0)
        if source_fps > 1.0:
            target_fps = source_fps
        if target_fps <= 0.0:
            target_fps = 20.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.save_path, fourcc, target_fps, (width, height))
        if not self.writer.isOpened():
            self.writer = None
            print(f"[Display ERROR] cannot open video writer: {self.save_path}")
            return
        print(f"[Display] recording -> {self.save_path} fps={target_fps:.2f}")

    def _write_frame(self, frame):
        if not self.save_video:
            return
        self._ensure_writer(frame)
        if self.writer is not None:
            self.writer.write(frame)

    def _release_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"[Display] video saved: {self.save_path}")

    def update_fps(self):
        now = time.time()
        delta = now - self.last_time
        if delta > 0:
            self.fps = 1.0 / delta
        self.last_time = now
        self.shared.set_fps(self.fps)

    def draw_zone(self, img, draw, zone):
        rect = zone.get("rect", [])
        if len(rect) != 4:
            return
        x1, y1, x2, y2 = [int(v) for v in rect]
        color = tuple(zone.get("color", (0, 200, 255)))
        name = str(zone.get("name", zone.get("id", "Zone")))
        zone_type = self._to_zone_type(zone.get("type", "intrusion"))
        label = f"{name} | {zone_type}"
        self._draw_rect(img, draw, (x1, y1), (x2, y2), color, thickness=2)
        tw, th = self._measure_text(draw, label, size=22)
        box_y1 = max(0, y1 - th - 12)
        self._draw_rect(img, draw, (x1, box_y1), (x1 + tw + 12, y1), (0, 0, 0), fill=True)
        self._put_text(img, draw, label, (x1 + 6, box_y1 + 3), color, size=22)

    def draw_security_object(self, img, draw, item):
        bbox = item.get("bbox", [])
        if len(bbox) != 4:
            return
        label = self._to_label(item.get("label", "alert"))
        score = float(item.get("score", 0.0))
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 0, 255) if label.lower() == "fire" else (128, 128, 128)
        self._draw_rect(img, draw, (x1, y1), (x2, y2), color, thickness=2)
        text = f"{label} {score:.2f}"
        tw, th = self._measure_text(draw, text, size=22)
        box_y1 = max(0, y1 - th - 12)
        self._draw_rect(img, draw, (x1, box_y1), (x1 + tw + 12, y1), (0, 0, 0), fill=True)
        self._put_text(img, draw, text, (x1 + 6, box_y1 + 3), color, size=22)

    def draw_person(self, img, draw, pid, det, behavior, security_state):
        bbox = det.get("bbox")
        if bbox is None or len(bbox) != 4:
            return

        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return

        score = float(det.get("score", 0.0))
        name = self._to_label(det.get("name", "person"))
        raw_action = "Unknown"
        raw_score = 0.0
        stable_action = "WarmingUp"
        stable_score = 0.0
        ready = False
        buffer_len = 0

        if isinstance(behavior, dict):
            raw_action = self._to_action(behavior.get("raw_action", "unknown"))
            raw_score = float(behavior.get("raw_confidence", 0.0))
            stable_action = self._to_action(
                behavior.get("stable_action", behavior.get("action", "unknown"))
            )
            stable_score = float(behavior.get("stable_confidence", behavior.get("confidence", 0.0)))
            ready = bool(behavior.get("ready", False))
            buffer_len = int(behavior.get("buffer_len", 0))

        alerts = []
        zones = []
        if isinstance(security_state, dict):
            alerts = [self._to_alert(item) for item in security_state.get("alerts", []) if item]
            zones = [str(item) for item in security_state.get("zones", []) if item]

        color = (0, 255, 0)
        if "FALL" in alerts:
            color = (0, 0, 255)
        elif alerts:
            color = (0, 200, 255)

        self._draw_rect(img, draw, (x1, y1), (x2, y2), color, thickness=3)

        lines = [
            f"ID:{pid} {name} {score:.2f}",
            f"Raw:{raw_action} {raw_score:.2f}",
        ]
        if ready:
            lines.append(f"Stable:{stable_action} {stable_score:.2f}")
        else:
            lines.append(f"Stable:WarmingUp ({buffer_len}/16)")
        if alerts:
            lines.append(f"Alert:{'/'.join(alerts)}")
        if zones:
            lines.append(f"Zone:{'/'.join(zones)}")

        widths = []
        heights = []
        for text in lines:
            tw, th = self._measure_text(draw, text, size=20)
            widths.append(tw)
            heights.append(th)

        panel_w = max(widths) + 18 if widths else 180
        panel_h = sum(heights) + (len(lines) * 6) + 10
        panel_y1 = max(0, y1 - panel_h - 6)
        self._draw_rect(img, draw, (x1, panel_y1), (x1 + panel_w, y1), (0, 0, 0), fill=True)

        text_y = panel_y1 + 4
        for idx, text in enumerate(lines):
            self._put_text(img, draw, text, (x1 + 8, text_y), (0, 255, 255), size=20)
            text_y += heights[idx] + 6

    def draw_event_panel(self, img, draw, events):
        if not isinstance(events, list) or not events:
            return

        title = "Security Events"
        lines = []
        for item in events[:5]:
            event_type = self._to_event(item.get("event_type", "EVENT"))
            pid = item.get("person_id")
            zone_name = str(item.get("zone_name", "")).strip()
            label = self._to_label(item.get("label", "")).strip()
            duration = item.get("duration")
            score = float(item.get("score", 0.0))
            parts = [event_type]
            if pid not in (None, ""):
                parts.append(f"ID:{pid}")
            if zone_name:
                parts.append(zone_name)
            elif label:
                parts.append(label)
            if duration is not None:
                parts.append(f"{float(duration):.1f}s")
            if score > 0:
                parts.append(f"{score:.2f}")
            lines.append(" | ".join(parts))

        widths = []
        heights = []
        title_w, title_h = self._measure_text(draw, title, size=24)
        for text in lines:
            tw, th = self._measure_text(draw, text, size=20)
            widths.append(tw)
            heights.append(th)

        panel_x = 16
        panel_y = 58
        panel_w = max([title_w, 340] + widths) + 20
        panel_h = title_h + sum(heights) + (len(lines) * 6) + 20
        self._draw_rect(
            img,
            draw,
            (panel_x - 8, panel_y - 28),
            (panel_x + panel_w, panel_y + panel_h),
            (0, 0, 0),
            fill=True,
        )
        self._put_text(img, draw, title, (panel_x, panel_y - 22), (0, 0, 255), size=24)

        text_y = panel_y + 4
        for idx, text in enumerate(lines):
            self._put_text(img, draw, text, (panel_x, text_y), (255, 255, 255), size=20)
            text_y += heights[idx] + 6

    def draw_status_bar(self, img, draw):
        source_fps = float(self.shared.get_source_fps() or 0.0)
        lines = [
            f"Display FPS: {self.fps:.1f}",
            f"Source FPS: {source_fps:.2f}" if source_fps > 0 else "Source FPS: Unknown",
            "Exit: ESC",
        ]
        x = 20
        y = 18
        widths = []
        heights = []
        for text in lines:
            tw, th = self._measure_text(draw, text, size=22)
            widths.append(tw)
            heights.append(th)
        panel_w = max(widths) + 16
        panel_h = sum(heights) + (len(lines) * 6) + 12
        self._draw_rect(img, draw, (x - 8, y - 8), (x + panel_w, y + panel_h), (0, 0, 0), fill=True)
        text_y = y
        for idx, text in enumerate(lines):
            self._put_text(img, draw, text, (x, text_y), (255, 200, 0), size=22)
            text_y += heights[idx] + 6

    def run(self):
        print("[Display] started")
        try:
            while self.shared.running:
                frame, frame_id = self.shared.get_frame_packet()
                if frame is None:
                    time.sleep(0.005)
                    continue
                if frame_id == self.last_frame_id:
                    time.sleep(0.005)
                    continue
                self.last_frame_id = frame_id

                show = frame.copy()
                detections = self.shared.get_detections()
                behaviors = self.shared.get_behaviors()
                events = self.shared.get_events()
                security_states = self.shared.get_security_states()
                security_detections = self.shared.get_security_detections()
                zone_overlays = self.shared.get_zone_overlays()

                pil_image, draw = self._create_text_canvas(show)

                if isinstance(zone_overlays, list):
                    for zone in zone_overlays:
                        self.draw_zone(show, draw, zone)

                if isinstance(security_detections, list):
                    for item in security_detections:
                        self.draw_security_object(show, draw, item)

                if isinstance(detections, dict):
                    for pid, det in detections.items():
                        behavior = behaviors.get(pid) if isinstance(behaviors, dict) else None
                        security_state = security_states.get(pid) if isinstance(security_states, dict) else None
                        self.draw_person(show, draw, pid, det, behavior, security_state)

                self.update_fps()
                self.draw_status_bar(show, draw)
                self.draw_event_panel(show, draw, events)

                rendered = self._finalize_text_canvas(pil_image)
                if rendered is not None:
                    show = rendered

                self._write_frame(show)
                view = cv2.resize(show, (self.width, self.height))
                cv2.imshow(self.window_title, view)

                key = cv2.waitKey(1)
                if key == 27:
                    self.shared.running = False
                    break
        finally:
            self._release_writer()
            cv2.destroyAllWindows()
