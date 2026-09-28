import os
import time
from types import SimpleNamespace


class SimpleTracker:
    def __init__(
        self,
        max_lost=30,
        iou_threshold=0.25,
        max_center_distance_ratio=0.9,
        max_area_ratio=3.2,
    ):
        print("[Tracker] Init")

        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.max_center_distance_ratio = max_center_distance_ratio
        self.max_area_ratio = max_area_ratio
        self.smooth_alpha = float(
            os.environ.get("RK_TRACKER_SMOOTH_ALPHA", "0.65")
        )
        # 新轨迹需连续命中 min_hits 帧才对外返回（过滤误检闪烁）
        self.min_hits = max(
            1,
            int(os.environ.get("RK_TRACKER_MIN_HITS", "2")),
        )
        # 已确认轨迹在短暂丢失期间仍保持输出（遮挡/短暂消失不中断）
        self.keep_lost = max(
            0,
            int(os.environ.get("RK_TRACKER_KEEP_LOST", "10")),
        )

        self.next_id = 0
        self.tracks = {}
        self.last_removed_ids = []

    def _set_bbox(self, det, bbox):
        if isinstance(det, dict):
            det["bbox"] = bbox
        else:
            det.bbox = bbox

    def _smooth_bbox(self, prev_bbox, new_bbox):
        alpha = self.smooth_alpha
        if alpha <= 0.0:
            return list(new_bbox)
        return [
            alpha * float(new_bbox[i]) + (1.0 - alpha) * float(prev_bbox[i])
            for i in range(4)
        ]

    def _get_bbox(self, det):
        if isinstance(det, dict):
            return det.get("bbox", [0, 0, 0, 0])
        return det.bbox

    def _set_track_id(self, det, track_id):
        if isinstance(det, dict):
            det["track_id"] = track_id
        else:
            det.track_id = track_id

    def _get_track_id(self, det):
        if isinstance(det, dict):
            return det.get("track_id", -1)
        return getattr(det, "track_id", -1)

    def _ghost_track(self, track_id, track):
        """丢失但未超时的轨迹：用最后 bbox 继续输出，保持 ID 稳定。"""
        return SimpleNamespace(
            bbox=list(track["bbox"]),
            track_id=track_id,
            cls=track.get("cls", "person"),
            score=track.get("score", 0.0),
            lost=track["lost"],
        )

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)

    def _center(self, box):
        return (
            (box[0] + box[2]) * 0.5,
            (box[1] + box[3]) * 0.5,
        )

    def _size(self, box):
        return (
            max(1.0, float(box[2] - box[0])),
            max(1.0, float(box[3] - box[1])),
        )

    def _area(self, box):
        width, height = self._size(box)
        return width * height

    def _center_distance_ratio(self, box1, box2):
        c1x, c1y = self._center(box1)
        c2x, c2y = self._center(box2)
        dx = c1x - c2x
        dy = c1y - c2y
        distance = (dx * dx + dy * dy) ** 0.5

        w1, h1 = self._size(box1)
        w2, h2 = self._size(box2)
        scale = max(w1, h1, w2, h2, 1.0)
        return distance / scale

    def _area_ratio(self, box1, box2):
        area1 = self._area(box1)
        area2 = self._area(box2)
        return max(area1, area2) / max(1.0, min(area1, area2))

    def _can_match(self, box1, box2):
        if self._area_ratio(box1, box2) > self.max_area_ratio:
            return False

        iou_score = self._iou(box1, box2)
        if iou_score >= self.iou_threshold:
            return True

        return self._center_distance_ratio(box1, box2) <= self.max_center_distance_ratio

    def _match_score(self, box1, box2):
        iou_score = self._iou(box1, box2)
        distance_ratio = self._center_distance_ratio(box1, box2)
        distance_score = max(
            0.0,
            1.0 - min(1.0, distance_ratio / max(self.max_center_distance_ratio, 1e-6)),
        )
        area_penalty = min(
            1.0,
            (self._area_ratio(box1, box2) - 1.0) / max(self.max_area_ratio - 1.0, 1e-6),
        )
        return iou_score * 2.0 + distance_score - area_penalty * 0.2

    def update(self, detections):
        now = time.time()
        self.last_removed_ids = []

        stale_ids = []
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["lost"] += 1
            if self.tracks[track_id]["lost"] > self.max_lost:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.tracks[track_id]
        self.last_removed_ids = stale_ids

        if not detections:
            out = []
            for track_id, track in self.tracks.items():
                if (
                    track.get("hits", 0) >= self.min_hits
                    and track["lost"] <= self.keep_lost
                ):
                    out.append(self._ghost_track(track_id, track))
            return out

        candidates = []
        for det_index, det in enumerate(detections):
            bbox = self._get_bbox(det)
            for track_id, track in self.tracks.items():
                track_box = track["bbox"]
                if not self._can_match(bbox, track_box):
                    continue

                match_score = self._match_score(bbox, track_box)
                candidates.append((match_score, det_index, track_id))

        candidates.sort(reverse=True, key=lambda item: item[0])

        matched_detections = set()
        matched_tracks = set()

        for match_score, det_index, track_id in candidates:
            if det_index in matched_detections or track_id in matched_tracks:
                continue

            det = detections[det_index]
            bbox = self._get_bbox(det)
            track = self.tracks[track_id]
            smooth_bbox = self._smooth_bbox(track["bbox"], bbox)

            self._set_track_id(det, track_id)
            self._set_bbox(det, smooth_bbox)
            track["bbox"] = smooth_bbox
            track["lost"] = 0
            track["time"] = now
            track["hits"] = track.get("hits", 0) + 1
            track["cls"] = getattr(det, "cls", "person")
            track["score"] = float(getattr(det, "score", 0.0))

            matched_detections.add(det_index)
            matched_tracks.add(track_id)

        for det_index, det in enumerate(detections):
            if det_index in matched_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            bbox = self._get_bbox(det)
            self._set_track_id(det, track_id)

            self.tracks[track_id] = {
                "bbox": bbox,
                "lost": 0,
                "time": now,
                "hits": 1,
                "cls": getattr(det, "cls", "person"),
                "score": float(getattr(det, "score", 0.0)),
            }

        # 返回前过滤：未确认的新轨迹不对外输出；追加丢失保持轨迹
        out = []
        for det in detections:
            tid = self._get_track_id(det)
            track = self.tracks.get(tid)
            if track is not None and track.get("hits", 0) >= self.min_hits:
                out.append(det)

        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue
            if (
                track.get("hits", 0) >= self.min_hits
                and track["lost"] <= self.keep_lost
            ):
                out.append(self._ghost_track(track_id, track))

        return out


# ============================================================
# ByteTracker —— 两阶段关联（2026-09-28 新增）
#
# 动机：SimpleTracker 只用高分框（score>=0.35）匹配，遮挡/模糊帧里目标分数
# 掉到阈值以下时轨迹直接断裂 → 每断一次就重新积累 16 帧序列 → 行为大量停在
# warming、判定 flips 暴增。实测 33/50 视频存在此抖动，stand_004 一条 770 帧
# 的视频产生 17~38 条轨迹。
#
# ByteTrack 的做法（Zhang et al. ECCV 2022）：
#   1. 高分框先与所有轨迹做 IoU 匹配（第一轮）
#   2. 未匹配上的轨迹，再用低分框（0.1~0.35）做第二轮匹配
#      —— 低分框往往是"被遮挡/模糊但位置正确"的真实目标，用它续命轨迹
#   3. 只有高分框中未匹配的才新建轨迹（低分框永不新建，避免误检建轨）
#
# 板端无 scipy，用贪心 IoU 替代匈牙利算法（本项目目标数少，实测等价效果好）。
# ============================================================

class ByteTracker(SimpleTracker):
    """ByteTrack 两阶段跟踪器；接口与 SimpleTracker 完全兼容。

    update(detections, low_detections=None)
      detections     : 高分框（沿用现有 ObjectEngine.score_thresh 过滤）
      low_detections : 低分框（可选；用于第二轮续接轨迹，不新建轨迹）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 第二轮匹配的 IoU 门限（比第一轮略宽松，因为低分框位置噪声更大）
        self.second_iou_threshold = float(
            os.environ.get("RK_BYTETRACK_IOU2", "0.20")
        )
        self.second_max_center_ratio = float(
            os.environ.get("RK_BYTETRACK_CENTER2", "1.10")
        )

    def _second_stage_match(self, low_detections, unmatched_track_ids):
        """第二轮：用低分框续接未匹配轨迹（只更新 bbox，不新建轨迹）"""
        if not low_detections or not unmatched_track_ids:
            return set()
        candidates = []
        for det_index, det in enumerate(low_detections):
            bbox = self._get_bbox(det)
            for track_id in unmatched_track_ids:
                track = self.tracks.get(track_id)
                if track is None:
                    continue
                track_box = track["bbox"]
                if self._area_ratio(bbox, track_box) > self.max_area_ratio:
                    continue
                iou = self._iou(bbox, track_box)
                center_ratio = self._center_distance_ratio(bbox, track_box)
                if iou >= self.second_iou_threshold:
                    score = iou * 2.0
                elif center_ratio <= self.second_max_center_ratio:
                    score = 1.0 - min(1.0, center_ratio)
                else:
                    continue
                candidates.append((score, det_index, track_id))
        candidates.sort(reverse=True, key=lambda item: item[0])

        used_dets = set()
        used_tracks = set()
        for score, det_index, track_id in candidates:
            if det_index in used_dets or track_id in used_tracks:
                continue
            det = low_detections[det_index]
            bbox = self._get_bbox(det)
            track = self.tracks[track_id]
            # 低分框噪声更大：平滑系数更强（更信任历史 bbox）
            smooth_bbox = self._smooth_bbox(track["bbox"], bbox)
            track["bbox"] = smooth_bbox
            track["lost"] = 0
            track["time"] = time.time()
            track["hits"] = track.get("hits", 0) + 1
            track["low_conf_hits"] = track.get("low_conf_hits", 0) + 1
            used_dets.add(det_index)
            used_tracks.add(track_id)
        return used_tracks

    def update(self, detections, low_detections=None):
        """两阶段关联。

        与父类不同的是：第一轮匹配后，未匹配轨迹在返回 ghost 之前，
        先尝试用低分框续命。
        """
        if not low_detections:
            return super().update(detections)

        # 第一轮：完全复用父类逻辑（它已实现贪心匹配 + 平滑 + ghost）
        tracked = super().update(detections)

        # 找出"本轮没有高分框匹配上"的轨迹：
        # 父类把命中轨迹的 lost 置 0，未命中的 lost++。
        # 这里挑出 lost>0 的轨迹做第二轮。
        unmatched = [tid for tid, tr in self.tracks.items() if tr.get("lost", 0) > 0]
        if unmatched:
            self._second_stage_match(low_detections, unmatched)

        return tracked
