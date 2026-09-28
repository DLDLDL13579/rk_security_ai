# ============================================================
# npu/pose_rknn.py
#
# RK3588 Smart Security AI Platform
#
# YOLOv5-Pose RKNN Wrapper (Multi-Person Decode V2.0)
#
# Pipeline:
#     YOLOv5-Pose RKNN
#              |
#         RAW Output
#              |
#         Multi-Person Decode (3-scale anchors + NMS + keypoints)
#              |
#        17 Keypoints (per person, 640 网格坐标)
#              |
#        MediaPipe 33 Format (归一化 0~1)
#
# Interface:
#     detect(img)           单人裁剪接口（兼容旧引擎）
#     detect_persons(img)   多人接口：返回全部人的 bbox + 33 点
#
# 环境变量:
#     RK_POSE_CONF          0.35   目标置信度
#     RK_POSE_NMS           0.45   NMS IoU
#     RK_POSE_KPTS          17     关键点数量
#     RK_POSE_NUM_CLASSES   自动   类别数（C - 5 - 3*kpts 推导）
#     RK_POSE_KPT_SIGMOID   0      关键点 x/y 是否已含 sigmoid
# ============================================================

import os

import cv2
import numpy as np

from rknnlite.api import RKNNLite


ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
]


# ============================================================
# Landmark
# ============================================================

class Landmark:

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


# ============================================================
# Pose RKNN
# ============================================================

class PoseRKNN:

    def __init__(self, model_path):
        print("[Pose] Loading...")

        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError("load pose rknn failed")

        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("init pose runtime failed")

        self.input_size = 640

        self.conf_threshold = float(
            os.environ.get("RK_POSE_CONF", "0.35")
        )
        self.nms_threshold = float(
            os.environ.get("RK_POSE_NMS", "0.45")
        )
        self.num_kpts = int(
            os.environ.get("RK_POSE_KPTS", "17")
        )
        self.num_classes = int(
            os.environ.get("RK_POSE_NUM_CLASSES", "0")
        ) or None
        self.kpt_sigmoid = int(
            os.environ.get("RK_POSE_KPT_SIGMOID", "0")
        ) == 1

        # 支持全帧多人姿态（detect_persons），引擎可走全帧路径
        self.supports_full_frame = True

        self._shape_checked = False
        # 模型类型标志（v8 在首次 _check_outputs 时确定）
        self._model_is_v8 = False

        print("[Pose] OK")

    # ========================================================
    # preprocess
    # ========================================================

    def preprocess(self, img):
        h0, w0 = img.shape[:2]
        scale = min(
            self.input_size / float(w0),
            self.input_size / float(h0),
        )
        nw = max(1, int(round(w0 * scale)))
        nh = max(1, int(round(h0 * scale)))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8,
        )
        pad_x = (self.input_size - nw) // 2
        pad_y = (self.input_size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(canvas, 0)
        return inp, scale, pad_x, pad_y

    # ========================================================
    # sigmoid
    # ========================================================

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # ========================================================
    # check output shape（首次推理打印，用于验证模型）
    # ========================================================

    def _check_outputs(self, outputs):
        # v8-pose 结构（4 输出 / 65 通道）走独立校验，2026-09-28
        if self._is_v8_model(outputs):
            self._model_is_v8 = True
            self._check_outputs_v8(outputs)
            return
        self._model_is_v8 = False
        c = int(outputs[0].shape[1])
        if self.num_classes is None:
            self.num_classes = c - 5 - 3 * self.num_kpts
        base = 5 + self.num_classes + 3 * self.num_kpts
        if self.num_classes <= 0 or c != 3 * base:
            hint = ""
            if c == 255:
                hint = (
                    "（C=255 = 3×85，这是标准 YOLOv5 检测模型，"
                    "不含人体关键点）"
                )
            raise RuntimeError(
                f"模型输出通道 C={c} 与 YOLOv5-Pose 布局 "
                f"3×(5+nc+3×kpts)=3×{base}={3 * base} 不符{hint}。"
                "请提供真正的 YOLOv5-Pose rknn 模型"
                "（COCO-Pose 可设置 RK_POSE_NUM_CLASSES=80）"
            )
        if not self._shape_checked:
            print("================")
            print("POSE OUTPUT SHAPES")
            for i, o in enumerate(outputs):
                print(i, o.shape)
            print("num_classes:", self.num_classes)
            print("================")
            self._shape_checked = True

    # ========================================================
    # decode single layer
    # ========================================================

    # ========================================================
    # v8-pose 支持（anchor-free + DFL）—— 2026-09-28 新增
    #
    # 模型: yolov8n-pose.rknn
    # 输出: 4 个张量
    #   3 x (1, 65, H, W)   H/W = 80/40/20；65 = 4*16(DFL) + 1(conf)
    #   1 x (1, 17, 3, N)   关键点（N = 80*80+40*40+20*20 = 8400）
    # 与 v5-pose（anchor-based, C=3*(5+nc+3*kpts)）完全不同
    # ========================================================

    @staticmethod
    def _softmax(x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    def _is_v8_model(self, outputs):
        """判断是否为 v8-pose 结构（4 输出 + 65 通道）"""
        if len(outputs) != 4:
            return False
        c = int(outputs[0].shape[1])
        return c == 4 * 16 + 1        # 64 DFL + 1 conf

    def _check_outputs_v8(self, outputs):
        c = int(outputs[0].shape[1])
        if c != 65:
            raise RuntimeError(
                f"v8-pose 期望通道 C=65 (4*16 DFL + 1 conf)，实际 C={c}"
            )
        if not self._shape_checked:
            print("================")
            print("POSE OUTPUT SHAPES (v8-pose)")
            for i, o in enumerate(outputs):
                print(i, o.shape)
            print("================")
            self._shape_checked = True

    def decode_v8(self, outputs, strides=(8, 16, 32)):
        """v8-pose 解码：DFL 积分 + 关键点张量

        返回 (boxes, scores, kpts_list)，坐标均为 640 输入尺度
        kpts 每项为 (num_kpts, 3) 的 x/y/conf
        """
        boxes, scores, kpts_list = [], [], []
        kpt_tensor = outputs[3][0]          # (17, 3, N)

        offset = 0
        for o, st in zip(outputs[:3], strides):
            p = o[0]                        # (65, H, W)
            _, H, W = p.shape
            N = H * W

            # DFL: 4 边 × 16 bin → softmax 后按 bin 值加权求和
            xywh = p[:64].reshape(4, 16, N)
            conf = self.sigmoid(p[64]).reshape(-1)
            bins = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
            dist = (bins * self._softmax(xywh, axis=1)).sum(axis=1)   # (4,N)

            hit = np.nonzero(conf > self.conf_threshold)[0]
            if hit.size:
                gy = (hit // W).astype(np.float32)
                gx = (hit % W).astype(np.float32)
                l, t, r, b = dist[:, hit]
                x1 = (gx + 0.5 - l) * st
                y1 = (gy + 0.5 - t) * st
                x2 = (gx + 0.5 + r) * st
                y2 = (gy + 0.5 + b) * st
                for k, idx in enumerate(hit):
                    boxes.append([x1[k], y1[k], x2[k], y2[k]])
                    scores.append(float(conf[idx]))
                    # 关键点用全局候选索引（offset 为前几层的累计）
                    kpts_list.append(
                        kpt_tensor[:, :, offset + idx].astype(np.float32)
                    )
            offset += N

        return boxes, scores, kpts_list

    def decode_layer(self, pred, anchors, stride):
        # pred: (1, C, H, W), C = 3 * (5 + nc + 3*kpts)
        pred = np.squeeze(pred, 0)
        C, H, W = pred.shape
        base = 5 + self.num_classes + 3 * self.num_kpts
        pred = pred.reshape(3, base, H, W)

        boxes = []
        scores = []
        kpts_list = []
        kpt_start = 5 + self.num_classes

        for a in range(3):
            anchor_w, anchor_h = anchors[a]
            feat = pred[a]  # (base, H, W)
            for gy in range(H):
                for gx in range(W):
                    det = feat[:, gy, gx]
                    obj = self.sigmoid(det[4])
                    if obj < self.conf_threshold:
                        continue

                    if self.num_classes > 1:
                        cls_scores = det[5:5 + self.num_classes]
                        cls_id = int(np.argmax(cls_scores))
                        score = obj * float(cls_scores[cls_id])
                    else:
                        cls_id = 0
                        score = obj * float(det[5])

                    if score < self.conf_threshold:
                        continue

                    x = (det[0] * 2.0 - 0.5 + gx) * stride
                    y = (det[1] * 2.0 - 0.5 + gy) * stride
                    bw = (det[2] * 2.0) ** 2 * anchor_w
                    bh = (det[3] * 2.0) ** 2 * anchor_h
                    x1 = x - bw / 2.0
                    y1 = y - bh / 2.0
                    x2 = x + bw / 2.0
                    y2 = y + bh / 2.0

                    kpts = np.zeros((self.num_kpts, 3), dtype=np.float32)
                    for i in range(self.num_kpts):
                        px = det[kpt_start + i * 3 + 0]
                        py = det[kpt_start + i * 3 + 1]
                        pv = det[kpt_start + i * 3 + 2]
                        if self.kpt_sigmoid:
                            px = self.sigmoid(px)
                            py = self.sigmoid(py)
                        px = (px * 2.0 - 0.5 + gx) * stride
                        py = (py * 2.0 - 0.5 + gy) * stride
                        pv = self.sigmoid(pv)
                        kpts[i, 0] = px
                        kpts[i, 1] = py
                        kpts[i, 2] = pv

                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    kpts_list.append(kpts)

        return boxes, scores, kpts_list

    # ========================================================
    # NMS
    # ========================================================

    def nms(self, boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]
        return keep

    # ========================================================
    # COCO17 -> MediaPipe33
    # ========================================================

    def convert17to33(self, pts):
        result = [
            Landmark(0, 0, 0, 1.0)
            for _ in range(33)
        ]
        mapping = {
            0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
            5: 11, 6: 12, 7: 13, 8: 14,
            9: 15, 10: 16, 11: 23, 12: 24,
            13: 25, 14: 26, 15: 27, 16: 28,
        }
        for src, dst in mapping.items():
            result[dst] = pts[src]
        # 填充缺失点（沿用上一个可见点）
        for i in range(33):
            if result[i].visibility <= 0:
                result[i] = Landmark(
                    result[i - 1].x,
                    result[i - 1].y,
                    0,
                    1.0,
                )
        return result

    # ========================================================
    # detect_persons: 多人接口
    # ========================================================

    def detect_persons(self, img):
        if img is None:
            return []

        h0, w0 = img.shape[:2]
        inp, scale, pad_x, pad_y = self.preprocess(img)
        outputs = self.rknn.inference(inputs=[inp])
        if outputs is None:
            return []

        outputs = [np.array(x, dtype=np.float32) for x in outputs]
        self._check_outputs(outputs)

        boxes = []
        scores = []
        kpts_list = []

        # 按模型类型选择解码路径（2026-09-28：兼容 v5-pose 与 v8-pose）
        if self._model_is_v8:
            boxes, scores, kpts_list = self.decode_v8(outputs)
        else:
            strides = [8, 16, 32]
            for out, anchor, stride in zip(outputs, ANCHORS, strides):
                b, s, k = self.decode_layer(out, anchor, stride)
                boxes.extend(b)
                scores.extend(s)
                kpts_list.extend(k)

        if not boxes:
            return []

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        keep = self.nms(boxes, scores)

        inv = 1.0 / scale

        persons = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            kpts = kpts_list[i]
            bx1 = int(round((x1 - pad_x) * inv))
            by1 = int(round((y1 - pad_y) * inv))
            bx2 = int(round((x2 - pad_x) * inv))
            by2 = int(round((y2 - pad_y) * inv))
            bx1 = max(0, min(bx1, w0))
            by1 = max(0, min(by1, h0))
            bx2 = max(0, min(bx2, w0))
            by2 = max(0, min(by2, h0))
            norm_kpts = [
                Landmark(
                    float((kx - pad_x) * inv) / w0,
                    float((ky - pad_y) * inv) / h0,
                    0.0,
                    float(kv),
                )
                for kx, ky, kv in kpts
            ]
            persons.append({
                "bbox": [bx1, by1, bx2, by2],
                "score": float(scores[i]),
                "keypoints": self.convert17to33(norm_kpts),
            })

        return persons

    # ========================================================
    # detect: 兼容单人裁剪接口（返回 list[33] 或 None）
    # ========================================================

    def detect(self, img):
        persons = self.detect_persons(img)
        if not persons:
            return None
        return persons[0]["keypoints"]

    # ========================================================
    # release
    # ========================================================

    def release(self):
        self.rknn.release()


# ============================================================
# 板端验证入口: python npu/pose_rknn.py <image.jpg>
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python npu/pose_rknn.py <image.jpg>")
        sys.exit(1)

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pose = PoseRKNN(os.path.join(_root, "models", "model.rknn"))
    img = cv2.imread(sys.argv[1])
    persons = pose.detect_persons(img)
    print("persons:", len(persons))
    for p in persons:
        print("bbox:", p["bbox"], "score:", round(p["score"], 3))
    pose.release()
