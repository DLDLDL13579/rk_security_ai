import os
import sys


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.insert(0, ROOT)


from engine.security_event_manager import SecurityEventManager
from engine.zone_manager import ZoneManager


def main():
    zones = [
        {
            "id": "intrusion_zone",
            "name": "IntrusionZone",
            "type": "both",
            "rect": [100, 100, 300, 300],
            "loiter_seconds": 2.0,
        }
    ]
    manager = ZoneManager(zones)
    events = SecurityEventManager(default_cooldown=0.0)

    persons = {
        1: {
            "bbox": [150, 150, 250, 280],
            "score": 0.92,
        }
    }

    states_a, raw_events_a = manager.evaluate(persons, now=0.0)
    emitted_a = [events.emit(item, now=0.0) for item in raw_events_a]
    emitted_a = [item for item in emitted_a if item is not None]

    states_b, raw_events_b = manager.evaluate(persons, now=3.0)
    emitted_b = [events.emit(item, now=3.0) for item in raw_events_b]
    emitted_b = [item for item in emitted_b if item is not None]

    print("[states_a]", states_a)
    print("[events_a]", emitted_a)
    print("[states_b]", states_b)
    print("[events_b]", emitted_b)


if __name__ == "__main__":
    main()
