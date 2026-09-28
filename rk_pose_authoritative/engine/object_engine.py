# =========================================================
# engine/object_engine.py
# RK3588 Smart Security AI
# Object Engine V2.5 Final
# =========================================================

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import os
import time
import math


# =========================================================
# COCO 类别
# =========================================================

CLASS_NAMES = {

    0:"person",
    1:"bicycle",
    2:"car",
    3:"motorcycle",
    4:"airplane",
    5:"bus",
    6:"train",
    7:"truck",
    8:"boat",

    9:"traffic light",
    10:"fire hydrant",
    11:"stop sign",
    12:"parking meter",
    13:"bench",

    14:"bird",
    15:"cat",
    16:"dog",
    17:"horse",
    18:"sheep",
    19:"cow",

    20:"elephant",
    21:"bear",
    22:"zebra",
    23:"giraffe",

    24:"backpack",
    25:"umbrella",
    26:"handbag",
    27:"tie",
    28:"suitcase",

    39:"bottle",

    56:"chair",

    62:"tv",
    63:"laptop",
    67:"cell phone",

    43:"knife"
}


# =========================================================
# 中文
# =========================================================

CLASS_NAMES_ZH = {

    "person":"人员",
    "bicycle":"自行车",
    "car":"汽车",
    "motorcycle":"摩托车",
    "airplane":"飞机",
    "bus":"公交车",
    "train":"火车",
    "truck":"卡车",
    "boat":"船",

    "traffic light":"红绿灯",
    "fire hydrant":"消防栓",
    "stop sign":"停止牌",
    "parking meter":"停车计时器",
    "bench":"长椅",

    "bird":"鸟",
    "cat":"猫",
    "dog":"狗",
    "horse":"马",
    "sheep":"羊",
    "cow":"牛",

    "elephant":"大象",
    "bear":"熊",
    "zebra":"斑马",
    "giraffe":"长颈鹿",

    "backpack":"背包",
    "umbrella":"雨伞",
    "handbag":"手提包",

    "bottle":"瓶子",

    "chair":"椅子",

    "tv":"电视",

    "laptop":"笔记本",

    "cell phone":"手机",

    "knife":"刀具"
}


# =========================================================
# Detection
# =========================================================
class Detection:
    """
    统一检测结果数据结构
    """

    def __init__(self,
                 cls_id,
                 cls_name,
                 score,
                 bbox,
                 track_id=-1,
                 timestamp=None,
                 extra=None,
                 low_conf=False):
        # ByteTrack 低分框标记：True 表示该框仅用于跟踪器第二轮续接轨迹，
        # 不参与行为识别（2026-09-28）
        self.low_conf = bool(low_conf)
        # YOLO类别ID
        self.cls_id = int(cls_id)

        # 英文类别
        self.cls = str(cls_name)

        # 中文类别
        self.cls_zh = CLASS_NAMES_ZH.get(self.cls, self.cls)

        # 置信度
        self.score = float(score)

        # 边框
        self.bbox = list(map(int, bbox))

        # Tracker编号
        self.track_id = int(track_id)

        # 中心点
        x1, y1, x2, y2 = self.bbox
        self.center = (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

        # 面积
        self.area = max(0, (x2 - x1) * (y2 - y1))

        # 时间戳（若未提供则使用当前时间）
        self.timestamp = timestamp if timestamp is not None else time.time()

        # 额外信息
        self.extra = extra if extra is not None else {}

    def to_dict(self):
        return {
            "cls_id": self.cls_id,
            "cls": self.cls,
            "cls_zh": self.cls_zh,
            "score": self.score,
            "bbox": self.bbox,
            "track_id": self.track_id,
            "center": self.center,
            "area": self.area,
            "timestamp": self.timestamp,
            "extra": self.extra
        }


# =========================================================
# Object Engine
# =========================================================

class ObjectEngine:

    def __init__(self):
        self.score_thresh = float(
            os.environ.get("RK_OBJECT_SCORE", "0.35")
        )
        # ByteTrack 低分框门限（2026-09-28）：低于 score_thresh 但高于此值的框
        # 不参与对外输出，只用于跟踪器第二轮续接轨迹。设为 0 可关闭该功能。
        self.low_score_thresh = float(
            os.environ.get("RK_OBJECT_LOW_SCORE", "0.10")
        )
        self.iou_thresh = 0.30
        print("[ObjectEngine] Ready")

    # =====================================================
    # bbox中心
    # =====================================================
    def calc_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return [
            (x1 + x2) / 2,
            (y1 + y2) / 2
        ]

    # =====================================================
    # bbox面积
    # =====================================================
    def calc_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    # =====================================================
    # YOLO结果 -> Detection
    # =====================================================
    def process(self, yolo_results):
        """
        将 YOLO 原始输出统一转换为 Detection 对象列表
        """
        detections = []

        if yolo_results is None:
            return detections

        for obj in yolo_results:
            try:
                # -----------------------------
                # class id
                # -----------------------------
                cls_id = obj.get("class", obj.get("cls_id", -1))

                # -----------------------------
                # class name
                # -----------------------------
                cls = obj.get("name")
                if cls is None:
                    cls = CLASS_NAMES.get(cls_id, "unknown")

                # -----------------------------
                # 中文类别（此处暂不处理，由 Detection 内部完成）
                # -----------------------------

                # -----------------------------
                # score
                # -----------------------------
                score = float(obj.get("score", 0.0))
                # 低于对外阈值但高于低分门限的框保留给跟踪器（ByteTrack 第二轮），
                # 作为 low_conf 标记；低于低分门限的直接丢弃（2026-09-28）
                is_low = score < self.score_thresh
                if is_low and score < self.low_score_thresh:
                    continue

                # -----------------------------
                # bbox
                # -----------------------------
                bbox = obj.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = [x1, y1, x2, y2]

                # -----------------------------
                # 创建 Detection 对象
                # -----------------------------
                det = Detection(
                    cls_id=cls_id,
                    cls_name=cls,
                    score=score,
                    bbox=bbox,
                    track_id=-1,
                    low_conf=is_low
                )
                detections.append(det)

            except Exception as e:
                print("[ObjectEngine ERROR]", e)
                continue

        return detections

    # =====================================================
    # Detection -> dict
    # =====================================================
    def to_dict(self, detections):
        result = []
        for det in detections:
            result.append(det.to_dict())
        return result

    # =====================================================
    # 统计类别数量
    # =====================================================
    def statistics(self, detections):
        stat = {}
        for det in detections:
            stat.setdefault(det.cls, 0)
            stat[det.cls] += 1
        return stat

    # =====================================================
    # 是否存在某目标
    # =====================================================
    def has_object(self, detections, cls_name):
        for det in detections:
            if det.cls == cls_name:
                return True
        return False

    # =====================================================
    # 获取某类别
    # =====================================================
    def get_objects(self, detections, cls_name):
        result = []
        for det in detections:
            if det.cls == cls_name:
                result.append(det)
        return result

    # =====================================================
    # Tracker绑定
    # =====================================================
    def bind_tracker(self, detections, trackers):
        if trackers is None:
            return detections

        for det in detections:
            best_iou = 0.0
            best_id = -1

            for trk in trackers:
                if "bbox" not in trk:
                    continue
                iou_score = self.iou(det.bbox, trk["bbox"])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_id = trk.get("track_id", trk.get("id", -1))

            if best_iou >= self.iou_thresh:
                det.track_id = best_id

        return detections

    # =====================================================
    # IOU
    # =====================================================
    def iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter <= 0:
            return 0.0

        areaA = (box1[2] - box1[0]) * (box1[3] - box1[1])
        areaB = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (areaA + areaB - inter + 1e-6)

    # =====================================================
    # Track分组
    # =====================================================
    def group_by_track(self, detections):
        result = {}
        for det in detections:
            result.setdefault(det.track_id, [])
            result[det.track_id].append(det)
        return result

    # =====================================================
    # 中文分类统计
    # =====================================================
    def statistics_zh(self, detections):
        result = {}
        for det in detections:
            result.setdefault(det.cls_zh, 0)
            result[det.cls_zh] += 1
        return result

    # =====================================================
    # MQTT JSON
    # =====================================================
    def to_mqtt_json(self, detections):
        payload = []
        for det in detections:
            payload.append({
                "track_id": det.track_id,
                "class_id": det.cls_id,
                "class": det.cls,
                "class_zh": det.cls_zh,
                "score": round(det.score, 3),
                "bbox": det.bbox,
                "center": det.center,
                "area": int(det.area),
                "timestamp": det.timestamp
            })
        return payload

    # =====================================================
    # Scene Engine接口
    # =====================================================
    def to_scene_objects(self, detections):
        scene = []
        for det in detections:
            scene.append({
                "id": det.track_id,
                "class": det.cls,
                "bbox": det.bbox,
                "center": det.center,
                "area": det.area
            })
        return scene

    # =====================================================
    # Debug
    # =====================================================
    def debug_print(self, detections):
        print("\n========== Detection ==========")
        for det in detections:
            print(
                f"[{det.track_id}] "
                f"{det.cls_zh}({det.cls}) "
                f"{det.score:.2f} "
                f"{det.bbox}"
            )
        print("===============================\n")
