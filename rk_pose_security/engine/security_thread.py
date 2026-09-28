import threading
import time

from engine.fire_smoke_detector import NullFireSmokeDetector
from engine.security_event_manager import SecurityEventManager
from engine.zone_manager import ZoneManager


class SecurityThread(threading.Thread):
    def __init__(
        self,
        shared,
        fire_smoke_detector=None,
        zone_manager=None,
        interval=0.05,
        fall_confidence=0.55,
    ):
        super().__init__(name="SecurityThread")
        self.shared = shared
        self.interval = float(interval)
        self.fall_confidence = float(fall_confidence)
        self.fire_smoke_detector = fire_smoke_detector or NullFireSmokeDetector()
        self.zone_manager = zone_manager or ZoneManager([])
        self.event_manager = SecurityEventManager(default_cooldown=3.0)
        self.recent_events = []
        self.event_hold_seconds = 4.0
        self.last_event_signature = None
        self.last_frame_id = -1
        print("[Security Thread] initialized")

    def _merge_state(self, base_state, pid, alert_name, zone_name=""):
        state = dict(base_state.get(pid, {}))
        alerts = list(state.get("alerts", []))
        if alert_name and alert_name not in alerts:
            alerts.append(alert_name)
        state["alerts"] = alerts
        zones = list(state.get("zones", []))
        if zone_name and zone_name not in zones:
            zones.append(zone_name)
        state["zones"] = zones
        base_state[pid] = state

    def _normalize_fire_smoke(self, objects):
        normalized = []
        for idx, item in enumerate(objects or []):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox", [])
            if len(bbox) != 4:
                continue
            label = str(item.get("label", item.get("name", ""))).strip().lower()
            if label not in ("fire", "smoke", "flame"):
                continue
            normalized.append(
                {
                    "id": str(item.get("id", f"fs_{idx}")),
                    "label": "fire" if label == "flame" else label,
                    "bbox": [int(v) for v in bbox],
                    "score": float(item.get("score", 0.0)),
                }
            )
        return normalized

    def _format_event_log(self, event):
        event_type = str(event.get("event_type", "EVENT"))
        pid = event.get("person_id")
        zone_name = str(event.get("zone_name", "")).strip()
        label = str(event.get("label", "")).strip()
        score = float(event.get("score", 0.0))
        duration = event.get("duration")
        parts = [event_type]
        if pid not in (None, ""):
            parts.append(f"id={pid}")
        if zone_name:
            parts.append(f"zone={zone_name}")
        if label:
            parts.append(f"label={label}")
        if duration is not None:
            parts.append(f"duration={float(duration):.1f}s")
        if score > 0:
            parts.append(f"score={score:.2f}")
        return " | ".join(parts)

    def _append_event(self, event_output, event):
        if event is None:
            return
        signature = (
            event.get("event_type"),
            event.get("person_id"),
            event.get("zone_id"),
            event.get("label"),
        )
        if signature != self.last_event_signature:
            print(f"[Security EVENT] {self._format_event_log(event)}")
            self.last_event_signature = signature
        event_output.append(event)

    def run(self):
        print("[Security Thread] started")
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
            behaviors = self.shared.get_behaviors()
            now = time.time()
            security_states, zone_events = self.zone_manager.evaluate(detections, now=now)
            fire_smoke_objects = self._normalize_fire_smoke(self.fire_smoke_detector.detect(frame))
            event_output = []

            for item in zone_events:
                self._append_event(
                    event_output,
                    self.event_manager.emit(item, now=now, cooldown=2.0),
                )

            if isinstance(behaviors, dict):
                for pid, behavior in behaviors.items():
                    if not isinstance(behavior, dict):
                        continue
                    action = str(behavior.get("action", behavior.get("behavior", ""))).strip()
                    score = float(
                        behavior.get(
                            "confidence",
                            behavior.get("score", behavior.get("behavior_score", 0.0)),
                        )
                    )
                    if action == "fall_down" and score >= self.fall_confidence:
                        self._merge_state(security_states, pid, "FALL")
                        bbox = behavior.get("bbox")
                        if bbox is None and isinstance(detections, dict):
                            bbox = (detections.get(pid) or {}).get("bbox")
                        self._append_event(
                            event_output,
                            self.event_manager.emit(
                                {
                                    "event_type": "FALL_DETECTED",
                                    "source_key": f"fall:{pid}",
                                    "person_id": pid,
                                    "bbox": list(bbox or []),
                                    "score": score,
                                },
                                now=now,
                                cooldown=2.0,
                            ),
                        )

            for item in fire_smoke_objects:
                label = item["label"]
                event_name = "FIRE_DETECTED" if label == "fire" else "SMOKE_DETECTED"
                self._append_event(
                    event_output,
                    self.event_manager.emit(
                        {
                            "event_type": event_name,
                            "source_key": f"{label}:{item['id']}",
                            "label": label,
                            "bbox": list(item["bbox"]),
                            "score": float(item["score"]),
                        },
                        now=now,
                        cooldown=1.5,
                    ),
                )

            self.recent_events.extend(event_output)
            self.recent_events = [
                item
                for item in self.recent_events
                if now - float(item.get("ts", 0.0)) <= self.event_hold_seconds
            ][-6:]

            self.shared.set_zone_overlays(self.zone_manager.get_zone_overlays())
            self.shared.set_security_states(security_states)
            self.shared.set_security_detections(fire_smoke_objects)
            self.shared.set_events(self.recent_events)
            time.sleep(self.interval)

    def stop(self):
        self.shared.running = False
