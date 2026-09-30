import copy
import time


class ZoneManager:
    def __init__(self, zones=None):
        self.zones = [self._normalize_zone(item, idx) for idx, item in enumerate(zones or [])]
        self.person_zone_state = {}

    def _normalize_zone(self, zone, idx):
        zone = dict(zone or {})
        rect = list(zone.get("rect", [0, 0, 0, 0]))
        if len(rect) != 4:
            rect = [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in rect]
        return {
            "id": str(zone.get("id", f"zone_{idx}")),
            "name": str(zone.get("name", f"zone_{idx}")),
            "type": str(zone.get("type", "intrusion")).strip().lower(),
            "rect": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
            "color": tuple(zone.get("color", (0, 200, 255))),
            "loiter_seconds": float(zone.get("loiter_seconds", 10.0)),
            "enabled": bool(zone.get("enabled", True)),
        }

    def get_zone_overlays(self):
        return copy.deepcopy(self.zones)

    def _bbox_anchor(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (float(x1) + float(x2)) * 0.5
        bottom_y = float(max(y1, y2))
        return (center_x, bottom_y)

    def _point_in_rect(self, point, rect):
        px, py = point
        x1, y1, x2, y2 = rect
        return x1 <= px <= x2 and y1 <= py <= y2

    def evaluate(self, persons, now=None):
        now = time.time() if now is None else float(now)
        persons = persons or {}
        events = []
        security_states = {}
        visible_keys = set()

        for pid, det in persons.items():
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            person_alerts = []
            zone_names = []
            anchor = self._bbox_anchor(bbox)

            for zone in self.zones:
                if not zone["enabled"]:
                    continue

                key = (pid, zone["id"])
                visible_keys.add(key)
                state = self.person_zone_state.setdefault(
                    key,
                    {
                        "inside": False,
                        "entered_at": 0.0,
                        "loiter_reported": False,
                    },
                )
                inside = self._point_in_rect(anchor, zone["rect"])

                if inside:
                    zone_names.append(zone["name"])
                    if not state["inside"]:
                        state["inside"] = True
                        state["entered_at"] = now
                        state["loiter_reported"] = False
                        if zone["type"] in ("intrusion", "both"):
                            person_alerts.append("INTRUSION")
                            events.append(
                                {
                                    "event_type": "INTRUSION_DETECTED",
                                    "source_key": f"intrusion:{zone['id']}:{pid}",
                                    "person_id": pid,
                                    "zone_id": zone["id"],
                                    "zone_name": zone["name"],
                                    "bbox": list(bbox),
                                    "anchor": [anchor[0], anchor[1]],
                                    "score": float(det.get("score", 0.0)),
                                }
                            )

                    if zone["type"] in ("loiter", "both"):
                        dwell = max(0.0, now - state["entered_at"])
                        if dwell >= zone["loiter_seconds"]:
                            person_alerts.append("LOITERING")
                            if not state["loiter_reported"]:
                                state["loiter_reported"] = True
                                events.append(
                                    {
                                        "event_type": "LOITERING_DETECTED",
                                        "source_key": f"loiter:{zone['id']}:{pid}",
                                        "person_id": pid,
                                        "zone_id": zone["id"],
                                        "zone_name": zone["name"],
                                        "bbox": list(bbox),
                                        "anchor": [anchor[0], anchor[1]],
                                        "duration": dwell,
                                        "score": float(det.get("score", 0.0)),
                                    }
                                )
                else:
                    state["inside"] = False
                    state["entered_at"] = 0.0
                    state["loiter_reported"] = False

            if person_alerts or zone_names:
                security_states[pid] = {
                    "alerts": sorted(set(person_alerts)),
                    "zones": zone_names,
                }

        stale_keys = [key for key in list(self.person_zone_state.keys()) if key not in visible_keys]
        for key in stale_keys:
            self.person_zone_state.pop(key, None)

        return security_states, events
