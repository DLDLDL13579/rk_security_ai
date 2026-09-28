import time


class SecurityEventManager:
    def __init__(self, default_cooldown=3.0):
        self.default_cooldown = float(default_cooldown)
        self.last_emit = {}

    def emit(self, event, now=None, cooldown=None):
        if not isinstance(event, dict):
            return None

        now = time.time() if now is None else float(now)
        cooldown = self.default_cooldown if cooldown is None else float(cooldown)
        event_type = str(event.get("event_type", "UNKNOWN"))
        source_key = str(
            event.get("source_key")
            or event.get("person_id")
            or event.get("zone_id")
            or event.get("label")
            or "global"
        )
        cache_key = f"{event_type}:{source_key}"
        last_ts = float(self.last_emit.get(cache_key, 0.0))
        if now - last_ts < cooldown:
            return None

        self.last_emit[cache_key] = now
        result = dict(event)
        result["ts"] = now
        return result
