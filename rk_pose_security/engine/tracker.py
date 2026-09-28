# engine/tracker.py

import time


class SimpleTracker:
    """
    Simple IoU Tracker

    支持：
        Detection对象
        dict对象

    自动分配 track_id
    """

    def __init__(self,
                 max_lost=30,
                 iou_threshold=0.3):

        print("[Tracker] Init")

        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

        self.next_id = 0

        # track_id -> info
        self.tracks = {}
    # ==========================================
    # bbox
    # ==========================================

    def _get_bbox(self, det):

        if isinstance(det, dict):
            return det.get("bbox", [0, 0, 0, 0])

        return det.bbox


    # ==========================================
    # score
    # ==========================================

    def _get_score(self, det):

        if isinstance(det, dict):
            return det.get("score", 0)

        return det.score


    # ==========================================
    # track id
    # ==========================================

    def _set_track_id(self, det, track_id):

        if isinstance(det, dict):
            det["track_id"] = track_id
        else:
            det.track_id = track_id


    def _get_track_id(self, det):

        if isinstance(det, dict):
            return det.get("track_id", -1)

        return getattr(det, "track_id", -1)
    # ==========================================
    # IOU
    # ==========================================

    def _iou(self, box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)
    # ==========================================
    # UPDATE
    # ==========================================

    def update(self, detections):

        now = time.time()

        # -------------------------
        # 清理失效轨迹
        # -------------------------

        remove_ids = []

        for tid in self.tracks:

            self.tracks[tid]["lost"] += 1

            if self.tracks[tid]["lost"] > self.max_lost:
                remove_ids.append(tid)

        for tid in remove_ids:
            del self.tracks[tid]

        if len(detections) == 0:
            return detections

        # -------------------------
        # IoU匹配
        # -------------------------

        for det in detections:

            bbox = self._get_bbox(det)

            best_iou = 0
            best_id = -1

            for tid, track in self.tracks.items():

                score = self._iou(
                    bbox,
                    track["bbox"]
                )

                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_iou > self.iou_threshold:

                self._set_track_id(det, best_id)

                self.tracks[best_id]["bbox"] = bbox
                self.tracks[best_id]["lost"] = 0
                self.tracks[best_id]["time"] = now

            else:

                new_id = self.next_id
                self.next_id += 1

                self._set_track_id(det, new_id)

                self.tracks[new_id] = {
                    "bbox": bbox,
                    "lost": 0,
                    "time": now
                }

        return detections
