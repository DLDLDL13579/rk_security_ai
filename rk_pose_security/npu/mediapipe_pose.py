import cv2
import mediapipe as mp
import numpy as np


class MediaPipePose:
    def __init__(
        self,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    ):
        self.mp_pose = mp.solutions.pose
        self.requested_complexity = self._sanitize_complexity(model_complexity)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.smooth_landmarks = bool(smooth_landmarks)
        self.num_kpts = 33
        self.pose = None
        self.active_complexity = None
        self.last_error = ""
        self._fallback_complexities = self._build_complexity_order(self.requested_complexity)
        self._initialize_pose()

    def _sanitize_complexity(self, value):
        try:
            value = int(value)
        except Exception:
            value = 1
        if value not in (0, 1, 2):
            value = 1
        return value

    def _build_complexity_order(self, requested):
        order = []
        for item in (requested, 1, 0, 2):
            item = self._sanitize_complexity(item)
            if item not in order:
                order.append(item)
        return order

    def _create_pose(self, complexity):
        return self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(complexity),
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def _initialize_pose(self):
        self.release()
        errors = []
        for complexity in self._fallback_complexities:
            try:
                self.pose = self._create_pose(complexity)
                self.active_complexity = complexity
                self.last_error = ""
                print(f"[MediaPipe] ready complexity={complexity}")
                return
            except Exception as exc:
                errors.append(f"init c={complexity}: {exc}")
        self.pose = None
        self.active_complexity = None
        self.last_error = "; ".join(errors)
        print(f"[MediaPipe ERROR] init failed {self.last_error}")

    def _ordered_complexities(self):
        if self.active_complexity not in self._fallback_complexities:
            return list(self._fallback_complexities)
        current_index = self._fallback_complexities.index(self.active_complexity)
        return (
            self._fallback_complexities[current_index:]
            + self._fallback_complexities[:current_index]
        )

    def _process_with_recovery(self, rgb):
        errors = []
        for complexity in self._ordered_complexities():
            if self.pose is None or self.active_complexity != complexity:
                try:
                    self.release()
                    self.pose = self._create_pose(complexity)
                    self.active_complexity = complexity
                    print(f"[MediaPipe] fallback complexity={complexity}")
                except Exception as exc:
                    errors.append(f"init c={complexity}: {exc}")
                    self.pose = None
                    self.active_complexity = None
                    continue
            try:
                return self.pose.process(rgb)
            except Exception as exc:
                errors.append(f"process c={complexity}: {exc}")
                self.release()
                self.pose = None
                self.active_complexity = None
        self.last_error = "; ".join(errors)
        print(f"[MediaPipe ERROR] process failed {self.last_error}")
        return None

    def detect(self, img):
        if img is None or img.size == 0:
            return None

        if self.pose is None:
            self._initialize_pose()
            if self.pose is None:
                return None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self._process_with_recovery(rgb)
        if results is None or not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        if len(lm) != self.num_kpts:
            print("[POSE ERROR] landmarks:", len(lm))
            return None

        keypoints = []
        for point in lm:
            keypoints.append(
                [
                    float(point.x),
                    float(point.y),
                    float(point.z),
                    float(point.visibility),
                ]
            )

        kpts = np.asarray(keypoints, dtype=np.float32)
        if kpts.shape != (33, 4):
            print("[POSE FORMAT ERROR]", kpts.shape)
            return None
        return kpts

    def release(self):
        if self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
        self.pose = None
