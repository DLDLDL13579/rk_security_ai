# -*- coding: utf-8 -*-
"""
火焰/烟雾检测 —— RKNN 适配器（已实现完整前处理与后处理）

移植来源：历史工程 rk_pose_project _8.4/npu/fire_smoke_rknn.py（骨架，postprocess 为 TODO）
后处理参考：单人检测/test_smoke_fire.py（前人调试脚本，含 3 种预处理模式与自动解码）

模型事实（由 models/fire/best.onnx 元数据实测确认，2026-09-29）：
    输入  : images [1, 3, 640, 640]
    输出  : output0 [1, 6, 8400]        # YOLOv8 detect 头，无 objectness
    类别  : {0: 'smoke', 1: 'fire'}     # ← 注意顺序：smoke 在前，fire 在后
    imgsz : [640, 640]   stride: 32   task: detect   opset: 12

【重要修正】_8.4 骨架里 CLASS_NAMES = ["fire", "smoke"] 的顺序与模型实际相反，
会导致火焰与烟雾判定互换。本实现按模型元数据的真实顺序修正为 ["smoke", "fire"]。

输出格式（与 _8.4 SecurityThread._normalize_fire_smoke 约定的接口一致）：
    [{"bbox": [x1, y1, x2, y2], "score": 0.91, "label": "fire"}, ...]
"""

import numpy as np

try:
    import cv2
except ImportError:  # 允许在无 cv2 的环境导入本模块做纯后处理单测
    cv2 = None

try:
    from rknnlite.api import RKNNLite
except ImportError:  # 允许在非板端环境导入本模块（单测/离线验证用）
    RKNNLite = None


class FireSmokeRKNN:
    """火焰/烟雾 RKNN 检测器（YOLOv8 单输出头）"""

    # ← 按模型元数据 names={0:'smoke', 1:'fire'} 的真实顺序
    CLASS_NAMES = ["smoke", "fire"]
    INPUT_SIZE = 640
    CONF_THRESH = 0.25   # 与前人 test_smoke_fire.py 保持一致
    NMS_THRESH = 0.50

    def __init__(self, model_path, conf_thresh=None, nms_thresh=None):
        if RKNNLite is None:
            raise RuntimeError(
                "rknnlite 不可用：本类只能在 RK3588 板端（rk3588_clean 环境）实例化"
            )

        self.model_path = model_path
        if conf_thresh is not None:
            self.CONF_THRESH = float(conf_thresh)
        if nms_thresh is not None:
            self.NMS_THRESH = float(nms_thresh)

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

        print(f"[FireSmoke] model loaded: {model_path}")

    # ------------------------------------------------------------------
    # 前处理
    # ------------------------------------------------------------------
    def preprocess(self, image):
        """
        BGR uint8 原图 → [1, 3, 640, 640] RGB uint8

        与训练/导出管线一致（Ultralytics YOLOv8 导出、opset 12）：
        直接 resize 到 640×640（不做 letterbox —— 与 best.onnx 的导出方式一致），
        BGR→RGB，保持 uint8 值域交由 RKNN 内部量化处理。
        """
        if cv2 is None:
            raise RuntimeError("preprocess 需要 cv2（板端 rk3588_clean 环境自带）")

        src_h, src_w = image.shape[:2]
        resized = cv2.resize(image, (self.INPUT_SIZE, self.INPUT_SIZE))
        input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)
        return input_data, src_w, src_h

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def postprocess(self, outputs, src_w, src_h):
        """
        YOLOv8 输出 [1, 6, 8400] → [{"bbox", "score", "label"}, ...]

        属性布局（6 = 4 box + 2 class）：
            行 0..3 : cx, cy, w, h   （640 输入尺度，单位像素）
            行 4..5 : smoke, fire 类别分数（已过 sigmoid）
        """
        if not outputs:
            return []

        arr = np.asarray(outputs[0], dtype=np.float32)

        # 归一化到 [6, 8400]
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"unexpected fire/smoke output shape: {arr.shape}")
        if arr.shape[0] != 4 + len(self.CLASS_NAMES):
            # 兼容 (8400, 6) 布局
            if arr.shape[1] == 4 + len(self.CLASS_NAMES):
                arr = arr.T
            else:
                raise ValueError(
                    f"unexpected fire/smoke output layout: {arr.shape}, "
                    f"expected ({4 + len(self.CLASS_NAMES)}, N) or (N, {4 + len(self.CLASS_NAMES)})"
                )

        boxes_xywh = arr[:4, :]                       # (4, 8400)
        class_scores = arr[4:, :]                     # (2, 8400)

        # 分数可能已是 sigmoid 后的概率，也可能仍是原始 logits
        if float(class_scores.min()) < 0.0 or float(class_scores.max()) > 1.0:
            class_scores = self._sigmoid(class_scores)

        class_ids = np.argmax(class_scores, axis=0)
        scores = class_scores[class_ids, np.arange(class_scores.shape[1])]

        keep = scores >= self.CONF_THRESH
        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[:, keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # cxcywh(640尺度) → xyxy(原图尺度)
        cx, cy, bw, bh = boxes_xywh[0], boxes_xywh[1], boxes_xywh[2], boxes_xywh[3]
        sx = float(src_w) / float(self.INPUT_SIZE)
        sy = float(src_h) / float(self.INPUT_SIZE)

        x1 = (cx - bw / 2.0) * sx
        y1 = (cy - bh / 2.0) * sy
        x2 = (cx + bw / 2.0) * sx
        y2 = (cy + bh / 2.0) * sy

        np.clip(x1, 0, src_w, out=x1)
        np.clip(y1, 0, src_h, out=y1)
        np.clip(x2, 0, src_w, out=x2)
        np.clip(y2, 0, src_h, out=y2)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # 逐类别 NMS
        final_boxes = []
        final_scores = []
        final_classes = []
        for cls_id in range(len(self.CLASS_NAMES)):
            mask = class_ids == cls_id
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            kept = self._nms(boxes[idx], scores[idx], self.NMS_THRESH)
            for k in kept:
                final_boxes.append(boxes[idx][k])
                final_scores.append(float(scores[idx][k]))
                final_classes.append(int(cls_id))

        results = []
        for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
            results.append(
                {
                    "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "score": round(float(score), 3),
                    "label": self.CLASS_NAMES[cls_id],
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    @staticmethod
    def _nms(boxes, scores, thresh):
        """标准 IoU NMS（纯 numpy，向量化）"""
        if len(boxes) == 0:
            return []

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= thresh)[0]
            order = order[inds + 1]

        return keep

    # ------------------------------------------------------------------
    def detect(self, image):
        """检测入口：返回 [{"bbox", "score", "label"}, ...]"""
        if image is None or getattr(image, "size", 0) == 0:
            return []

        input_data, src_w, src_h = self.preprocess(image)
        outputs = self.rknn.inference(inputs=[input_data])
        outputs = [np.asarray(item, dtype=np.float32) for item in outputs]
        return self.postprocess(outputs, src_w, src_h)

    def release(self):
        try:
            self.rknn.release()
        except Exception:
            pass
