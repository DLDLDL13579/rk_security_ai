import os
from types import SimpleNamespace

import numpy as np

from engine.feature import PoseFeatureExtractor
from engine.geometry import compute_geometry
from engine.locomotion_engine import LocomotionEngine
from engine.object_engine import ObjectEngine
from engine.sequence_buffer import PoseSequenceBuffer
from engine.special_action_engine import SpecialActionEngine
from engine.tracker import SimpleTracker
from npu.behavior_rknn import BehaviorRKNN


class PoseEngine:
    def __init__(self, yolo, pose, behavior_model_path=None):
        print("[ENGINE] Init")

        self.yolo = yolo
        self.pose = pose

        # 全帧姿态模式：姿态后端支持 detect_persons 且 RK_POSE_FULLFRAME 未关闭时，
        # 每帧只做一次全帧姿态推理，按 bbox 匹配到各轨迹（多人性能关键）
        fullframe_flag = os.environ.get(
            "RK_POSE_FULLFRAME", "1"
        ).strip().lower()
        self.full_frame_pose = (
            fullframe_flag in ("1", "true", "yes", "on")
            and hasattr(pose, "detect_persons")
        )
        self._fullframe_persons = []
        if self.full_frame_pose:
            print("[ENGINE] full-frame pose enabled (detect_persons)")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if behavior_model_path is None:
            behavior_model_path = os.path.join(
                project_root,
                "models",
                "behavior_tcn.rknn",
            )
        elif not os.path.isabs(behavior_model_path):
            behavior_model_path = os.path.join(project_root, behavior_model_path)

        self.feature = PoseFeatureExtractor()
        self.buffer = PoseSequenceBuffer(
            max_len=16,
            stride=max(1, int(os.environ.get("RK_BEHAVIOR_STRIDE", "2"))),
        )
        self.behavior = BehaviorRKNN(behavior_model_path)
        self.tracker = SimpleTracker()
        self.object_engine = ObjectEngine()
        self.locomotion_engine = LocomotionEngine()
        self.special_engine = SpecialActionEngine()

        self.frame_index = 0
        self.min_person_height = int(
            os.environ.get("RK_MIN_PERSON_HEIGHT", "72")
        )
        self.min_person_area = int(
            os.environ.get("RK_MIN_PERSON_AREA", str(72 * 48))
        )
        self.crop_pad = float(os.environ.get("RK_CROP_PAD", "0.12"))
        self.behavior_interval = float(os.environ.get("RK_BEHAVIOR_INTERVAL", "0.06"))
        self.max_behavior_tracks_per_frame = max(
            1,
            int(os.environ.get("RK_MAX_BEHAVIOR_TRACKS", "8")),
        )
        self.behavior_warmup_len = max(
            8,
            min(16, int(os.environ.get("RK_BEHAVIOR_WARMUP_LEN", "12"))),
        )
        self.pose_quality_threshold = float(
            os.environ.get("RK_POSE_QUALITY_THRESHOLD", "0.28")
        )
        self.pose_smooth_alpha = min(
            0.95,
            max(0.0, float(os.environ.get("RK_POSE_SMOOTH_ALPHA", "0.72"))),
        )
        self.net_gain = float(os.environ.get("RK_LOCO_NET_GAIN", "0.5"))

        self.behavior_cache = {}
        self.pose_smooth_state = {}
        self.pose_motion_state = {}
        self.track_motion = {}
        self.latest_tracks = []
        self.latest_detection_frame_id = -1

        print("[ENGINE] Ready")

    def check_feature(self, feat):
        if feat is None:
            return None

        feat = np.asarray(feat, dtype=np.float32)
        feat = np.clip(feat, -10.0, 10.0)
        return feat

    def process_detections(self, frame):
        self.frame_index += 1

        result = {
            "frame_id": self.frame_index,
            "detections": [],
        }

        if frame is None:
            return result

        yolo_out = self.yolo.detect(frame)
        detections = self.object_engine.process(yolo_out)

        persons = []
        for det in detections:
            if det.cls != "person":
                continue
            if not self._keep_person_bbox(det.bbox):
                continue
            persons.append(det)

        tracked = self.tracker.update(persons)

        self.latest_tracks = tracked
        self.latest_detection_frame_id = self.frame_index

        if self.tracker.last_removed_ids:
            self._cleanup_removed_tracks(self.tracker.last_removed_ids)

        result["detections"] = tracked
        return result

    def process_behaviors(self, frame, frame_id=None, tracks=None):
        if frame is None:
            return {
                "frame_id": self.latest_detection_frame_id if frame_id is None else frame_id,
                "behaviors": list(self.behavior_cache.values()),
                "events": [],
            }

        tracks = self.latest_tracks if tracks is None else tracks
        frame_id = self.latest_detection_frame_id if frame_id is None else frame_id

        # NPU 全帧姿态：每帧只推理一次，按 bbox 匹配到各轨迹
        if self.full_frame_pose:
            try:
                self._fullframe_persons = self.pose.detect_persons(frame) or []
            except Exception as e:
                print("[ENGINE] full-frame pose failed, fallback to per-crop:", e)
                self.full_frame_pose = False
                self._fullframe_persons = []
        else:
            self._fullframe_persons = []

        selected_tracks = self._select_behavior_tracks(tracks)
        for person in selected_tracks:
            behavior = self._infer_person_behavior(frame, frame_id, person)
            if behavior is None:
                continue
            self.behavior_cache[person.track_id] = behavior

        cached_behaviors = []
        for person in tracks:
            cached = self.behavior_cache.get(person.track_id)
            if cached is None:
                continue

            current = dict(cached)
            current["bbox"] = list(person.bbox)
            if cached.get("frame_id") != frame_id:
                current["state"] = "cached"
            cached_behaviors.append(current)

        return {
            "frame_id": frame_id,
            "behaviors": cached_behaviors,
            "events": [],
        }

    def _infer_person_behavior(self, frame, frame_id, person):
        raw_keypoints = None
        if self.full_frame_pose and self._fullframe_persons:
            matched = self._match_fullframe_person(
                person.bbox, self._fullframe_persons
            )
            if matched is not None:
                raw_keypoints = matched.get("keypoints")
        else:
            bbox = self._clip_bbox(person.bbox, frame.shape[:2])
            crop_box = self._build_crop_box(bbox, frame.shape[:2])
            x1, y1, x2, y2 = crop_box

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            try:
                raw_keypoints = self.pose.detect(crop)
            except Exception:
                raw_keypoints = None

        if raw_keypoints is None:
            return None

        keypoints = self._smooth_keypoints(person.track_id, raw_keypoints)
        pose_quality = self._estimate_pose_quality(keypoints)
        if pose_quality < self.pose_quality_threshold:
            return None

        motion = self._estimate_track_motion(person.track_id, person.bbox, frame_id)
        pose_energy = self._estimate_pose_energy(person.track_id, keypoints, frame_id)
        geometry = compute_geometry(keypoints)

        model_action = ""
        model_score = 0.0

        feature = self.feature.extract(keypoints, track_id=person.track_id)
        feature = self.check_feature(feature)
        if feature is not None:
            sequence = self.buffer.update(person.track_id, feature)
            if sequence is None:
                sequence = self.buffer.get_sequence(
                    person.track_id,
                    min_len=self.behavior_warmup_len,
                    pad_to_max=True,
                )

            if sequence is not None:
                model_action, model_score = self.behavior.forward(sequence)

        special_result = self.special_engine.resolve(
            track_id=person.track_id,
            action=model_action,
            score=float(model_score),
            pose_quality=pose_quality,
            bbox=person.bbox,
            keypoints=keypoints,
            geometry=geometry,
        )

        locomotion_result = self.locomotion_engine.resolve(
            track_id=person.track_id,
            motion=motion,
            pose_energy=pose_energy,
            pose_quality=pose_quality,
            keypoints=keypoints,
            model_action=model_action,
            model_score=float(model_score),
            geometry=geometry,
        )

        result = special_result if special_result is not None else locomotion_result
        result_state = result.get("state", "valid")
        if result.get("channel") == "special" and result_state == "valid":
            if self.buffer.length(person.track_id) < self.buffer.max_len:
                result_state = "warming"

        # 关键点转为简单格式供显示层画骨架（2026-09-28 新增）
        # 注：full-frame 路径（v8-pose）为整帧归一化坐标 0~1；
        #     裁剪路径（MediaPipe）为裁剪框内归一化，显示层按需换算。
        kpts_simple = [
            {
                "x": float(pt.x),
                "y": float(pt.y),
                "v": float(getattr(pt, "visibility", 1.0)),
            }
            for pt in keypoints
        ]

        return {
            "track_id": person.track_id,
            "bbox": list(person.bbox),
            "action": result.get("action", "standing"),
            "confidence": float(result.get("confidence", 0.0)),
            "state": result_state,
            "pose_quality": float(pose_quality),
            "motion": float(motion),
            "pose_energy": float(pose_energy),
            "gait_amp": float(result.get("amp", 0.0)),
            "channel": result.get("channel", "locomotion"),
            "frame_id": frame_id,
            "keypoints": kpts_simple,
            "keypoints_frame": "full_frame" if self.full_frame_pose else "crop",
            "crop_box": None if self.full_frame_pose else list(crop_box),
        }

    @staticmethod
    def _bbox_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def _match_fullframe_person(self, bbox, persons):
        best = None
        best_iou = 0.30
        for p in persons:
            pb = p.get("bbox")
            if not pb or len(pb) < 4:
                continue
            iou = self._bbox_iou(bbox, pb)
            if iou > best_iou:
                best_iou = iou
                best = p
        return best

    def _keep_person_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return height >= self.min_person_height and (width * height) >= self.min_person_area

    def _clip_bbox(self, bbox, image_shape):
        height, width = image_shape
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        return [x1, y1, x2, y2]

    def _build_crop_box(self, bbox, image_shape):
        x1, y1, x2, y2 = self._clip_bbox(bbox, image_shape)
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        pad_x = int(width * self.crop_pad)
        pad_y = int(height * self.crop_pad)

        crop_box = self._clip_bbox(
            [
                x1 - pad_x,
                y1 - pad_y,
                x2 + pad_x,
                y2 + pad_y,
            ],
            image_shape,
        )

        if crop_box[2] - crop_box[0] < 48 or crop_box[3] - crop_box[1] < 64:
            return [x1, y1, x2, y2]
        return crop_box

    def _estimate_pose_quality(self, keypoints):
        upper_ratio = self._visible_ratio(keypoints, (0, 11, 12, 23, 24))
        lower_ratio = self._visible_ratio(keypoints, (23, 24, 25, 26, 27, 28))
        valid_ratio = self._visible_ratio(keypoints, range(len(keypoints)))
        return float(
            max(
                0.0,
                min(
                    1.0,
                    valid_ratio * 0.4 + upper_ratio * 0.35 + lower_ratio * 0.25,
                ),
            )
        )

    def _visible_ratio(self, keypoints, indexes, threshold=0.3):
        visible = 0
        total = 0
        for index in indexes:
            if index >= len(keypoints):
                continue
            total += 1
            if getattr(keypoints[index], "visibility", 0.0) > threshold:
                visible += 1
        return visible / max(1, total)

    def _smooth_keypoints(self, track_id, keypoints):
        current = np.asarray(
            [
                [
                    float(point.x),
                    float(point.y),
                    float(getattr(point, "visibility", 0.0)),
                ]
                for point in keypoints
            ],
            dtype=np.float32,
        )

        last = self.pose_smooth_state.get(track_id)
        if last is None or last.shape != current.shape:
            smooth = current
        else:
            alpha = self.pose_smooth_alpha
            smooth = alpha * last + (1.0 - alpha) * current

        self.pose_smooth_state[track_id] = smooth.copy()
        return [
            SimpleNamespace(
                x=float(item[0]),
                y=float(item[1]),
                visibility=float(item[2]),
            )
            for item in smooth
        ]

    def _estimate_track_motion(self, track_id, bbox, frame_id):
        x1, y1, x2, y2 = map(float, bbox)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        person_height = max(1.0, y2 - y1)
        current = {
            "center_x": center_x,
            "center_y": center_y,
            "frame_id": frame_id,
            "speed": 0.0,
        }

        prev = self.track_motion.get(track_id)
        if prev is None:
            self.track_motion[track_id] = current
            return 0.0

        dt = max(1, frame_id - prev["frame_id"])
        dx = center_x - prev["center_x"]
        dy = center_y - prev["center_y"]
        speed = ((dx * dx + dy * dy) ** 0.5) / (person_height * dt)

        # 净位移累积：面向/背向镜头行走时瞬时位移小但持续同向，
        # 用窗口内净位移补足 motion，避免 walking 被误判为 standing
        net_x = prev.get("net_x", 0.0) + dx
        net_y = prev.get("net_y", 0.0) + dy
        net_disp = ((net_x * net_x + net_y * net_y) ** 0.5) / person_height
        net_ema = 0.85 * prev.get("net_ema", 0.0) + 0.15 * net_disp
        current["net_x"] = net_x * 0.80
        current["net_y"] = net_y * 0.80
        current["net_ema"] = net_ema
        speed = max(speed, net_ema * self.net_gain)

        current["speed"] = 0.65 * prev.get("speed", 0.0) + 0.35 * speed
        self.track_motion[track_id] = current
        return current["speed"]

    def _estimate_pose_energy(self, track_id, keypoints, frame_id):
        hip_center_x = (keypoints[23].x + keypoints[24].x) * 0.5
        hip_center_y = (keypoints[23].y + keypoints[24].y) * 0.5

        lower_body = np.asarray(
            [
                keypoints[25].x - hip_center_x,
                keypoints[25].y - hip_center_y,
                keypoints[26].x - hip_center_x,
                keypoints[26].y - hip_center_y,
                keypoints[27].x - hip_center_x,
                keypoints[27].y - hip_center_y,
                keypoints[28].x - hip_center_x,
                keypoints[28].y - hip_center_y,
            ],
            dtype=np.float32,
        )
        current = {
            "frame_id": frame_id,
            "lower_body": lower_body,
            "energy": 0.0,
        }

        prev = self.pose_motion_state.get(track_id)
        if prev is None:
            self.pose_motion_state[track_id] = current
            return 0.0

        dt = max(1, frame_id - prev["frame_id"])
        delta = np.abs(lower_body - prev["lower_body"]) / float(dt)
        ankle_energy = float(np.mean(delta[4:8]))
        knee_energy = float(np.mean(delta[0:4]))
        raw_energy = ankle_energy * 0.65 + knee_energy * 0.35
        current["energy"] = 0.68 * prev.get("energy", 0.0) + 0.32 * raw_energy
        self.pose_motion_state[track_id] = current
        return current["energy"]

    def _cleanup_removed_tracks(self, removed_ids):
        for track_id in removed_ids:
            self.buffer.clear(track_id)
            self.feature.clear_track(track_id)
            self.behavior_cache.pop(track_id, None)
            self.pose_smooth_state.pop(track_id, None)
            self.pose_motion_state.pop(track_id, None)
            self.track_motion.pop(track_id, None)
            self.locomotion_engine.clear_track(track_id)
            self.special_engine.clear_track(track_id)

    def _select_behavior_tracks(self, tracked):
        if len(tracked) <= self.max_behavior_tracks_per_frame:
            return list(tracked)

        return sorted(
            tracked,
            key=lambda person: (person.bbox[2] - person.bbox[0]) * (person.bbox[3] - person.bbox[1]),
            reverse=True,
        )[: self.max_behavior_tracks_per_frame]
