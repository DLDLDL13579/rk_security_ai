class BaseFireSmokeDetector:
    def detect(self, frame):
        raise NotImplementedError("fire/smoke detector must implement detect(frame)")


class NullFireSmokeDetector(BaseFireSmokeDetector):
    def detect(self, frame):
        return []


class CallableFireSmokeDetector(BaseFireSmokeDetector):
    def __init__(self, detect_fn):
        self.detect_fn = detect_fn

    def detect(self, frame):
        if self.detect_fn is None:
            return []
        result = self.detect_fn(frame)
        return result or []
