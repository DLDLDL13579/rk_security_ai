from collections import defaultdict, deque


class BehaviorFilter:
    def __init__(self, window=8, fall_fast=True, conf_thresh=0.0):
        print("[Behavior Filter V3] Init")
        self.window = int(window)
        self.fall_fast = bool(fall_fast)
        self.conf_thresh = float(conf_thresh)
        self.history = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, person_id, action, confidence):
        action = str(action or "unknown")
        confidence = float(confidence or 0.0)

        if action != "fall_down" and confidence < self.conf_thresh:
            action = "unknown"
            confidence = 0.0

        if self.fall_fast and action == "fall_down" and confidence > 0.45:
            self.history[person_id].append(action)
            return "fall_down"

        self.history[person_id].append((action, confidence))
        scores = {}

        for item in self.history[person_id]:
            if isinstance(item, tuple):
                act, conf = item
            else:
                act = item
                conf = 0.5
            scores[act] = scores.get(act, 0.0) + float(conf)

        if not scores:
            return "unknown"

        return max(scores, key=scores.get)

    def cleanup(self):
        self.history.clear()

    def clear(self, person_id=None):
        if person_id is None:
            self.cleanup()
            return
        self.history.pop(person_id, None)
