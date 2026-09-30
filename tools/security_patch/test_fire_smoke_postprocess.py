# -*- coding: utf-8 -*-
"""
烟火检测后处理离线单测（不依赖板端、不依赖 rknnlite）

验证点：
  1. YOLOv8 单输出 [1, 6, 8400] 的解码正确性（cxcywh → xyxy、尺度换算）
  2. 类别顺序为模型真实顺序 ["smoke", "fire"]（修正 _8.4 骨架的顺序 bug）
  3. 置信度过滤与逐类别 NMS 生效
  4. 空输出/无命中时安全返回 []

用法：
    /usr/local/bin/python3 test_fire_smoke_postprocess.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入模块文件，绕过 npu/__init__ 的潜在依赖
import importlib.util

MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npu", "fire_smoke_rknn.py")
spec = importlib.util.spec_from_file_location("fire_smoke_rknn", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
FireSmokeRKNN = mod.FireSmokeRKNN


def make_output(dets, num_anchors=8400):
    """
    dets: [(cx, cy, w, h, smoke_score, fire_score), ...] 均为 640 输入尺度
    返回 [1, 6, 8400] 的 YOLOv8 风格输出
    """
    out = np.zeros((4 + 2, num_anchors), dtype=np.float32)
    for idx, det in enumerate(dets):
        out[0, idx] = det[0]
        out[1, idx] = det[1]
        out[2, idx] = det[2]
        out[3, idx] = det[3]
        out[4, idx] = det[4]
        out[5, idx] = det[5]
    return out.reshape(1, 4 + 2, num_anchors)


class _Detector(FireSmokeRKNN):
    """绕过 __init__ 的 RKNN 加载，只测后处理"""

    def __init__(self):
        self.CONF_THRESH = 0.25
        self.NMS_THRESH = 0.50


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    detector = _Detector()
    src_w, src_h = 1920, 1080
    all_pass = True

    print("=" * 62)
    print(" 烟火后处理离线单测")
    print("=" * 62)

    # ---- 用例 1：类别顺序 ----
    all_pass &= check(
        "类别顺序为模型真实顺序 ['smoke', 'fire']",
        detector.CLASS_NAMES == ["smoke", "fire"],
        f"实际={detector.CLASS_NAMES}",
    )

    # ---- 用例 2：单个 fire 正确解码 ----
    # 中心 (320,320) 尺寸 (320,160) → 640 尺度 xyxy = (160,240,480,400)
    # 原图 1920×1080：sx=3.0  sy=1.6875 → (480, 405, 1440, 675)
    out = make_output([(320.0, 320.0, 320.0, 160.0, 0.02, 0.95)])
    res = detector.postprocess([out], src_w, src_h)
    all_pass &= check("单个 fire 检出", len(res) == 1, f"检出数={len(res)}")
    if res:
        r = res[0]
        expect = [480, 405, 1440, 675]
        ok_box = all(abs(a - b) <= 2 for a, b in zip(r["bbox"], expect))
        all_pass &= check(
            "fire 框坐标换算正确（640尺度→原图）",
            ok_box,
            f"期望≈{expect} 实际={r['bbox']}",
        )
        all_pass &= check("fire 标签正确", r["label"] == "fire", f"实际={r['label']}")
        all_pass &= check("fire 分数正确", abs(r["score"] - 0.95) < 0.01, f"实际={r['score']}")

    # ---- 用例 3：smoke 与 fire 同时存在，标签不混淆 ----
    out = make_output(
        [
            (100.0, 100.0, 80.0, 80.0, 0.90, 0.01),   # smoke
            (500.0, 500.0, 100.0, 100.0, 0.01, 0.85),  # fire
        ]
    )
    res = detector.postprocess([out], src_w, src_h)
    labels = sorted(r["label"] for r in res)
    all_pass &= check(
        "smoke/fire 同时检出且标签不混淆",
        labels == ["fire", "smoke"],
        f"实际={labels}",
    )

    # ---- 用例 4：低置信度被过滤 ----
    out = make_output([(320.0, 320.0, 100.0, 100.0, 0.10, 0.20)])
    res = detector.postprocess([out], src_w, src_h)
    all_pass &= check("低于阈值(0.25)被过滤", len(res) == 0, f"检出数={len(res)}")

    # ---- 用例 5：NMS 抑制重叠框 ----
    out = make_output(
        [
            (320.0, 320.0, 200.0, 200.0, 0.0, 0.95),
            (325.0, 322.0, 200.0, 200.0, 0.0, 0.90),  # 与上框高度重叠
            (700.0, 200.0, 120.0, 120.0, 0.0, 0.80),  # 独立框，应保留
        ]
    )
    res = detector.postprocess([out], src_w, src_h)
    all_pass &= check(
        "NMS 抑制重叠框、保留独立框",
        len(res) == 2,
        f"检出数={len(res)}（期望 2）",
    )

    # ---- 用例 6：空输出安全返回 ----
    all_pass &= check("空输出返回 []", detector.postprocess([], src_w, src_h) == [])
    empty = np.zeros((1, 6, 8400), dtype=np.float32)
    all_pass &= check("全零输出返回 []", detector.postprocess([empty], src_w, src_h) == [])

    # ---- 用例 7：输出布局兼容 (1, 8400, 6) ----
    # 注意：必须用真正的转置构造（reshape 只是内存重解释，会打乱数据布局）
    base = make_output([(320.0, 320.0, 320.0, 160.0, 0.02, 0.95)]).reshape(6, 8400)
    out_t = base.T.reshape(1, 8400, 6)
    res = detector.postprocess([out_t], src_w, src_h)
    all_pass &= check(
        "兼容 (1, 8400, 6) 布局（真转置）",
        len(res) == 1 and res[0]["label"] == "fire",
        f"检出={[ (r['label'], r['score']) for r in res ]}",
    )

    # ---- 用例 8：不同分辨率尺度换算 ----
    out = make_output([(320.0, 320.0, 640.0, 640.0, 0.0, 0.99)])
    res = detector.postprocess([out], 1280, 720)
    ok = res and abs(res[0]["bbox"][0]) <= 2 and abs(res[0]["bbox"][2] - 1280) <= 2
    all_pass &= check(
        "1280×720 分辨率尺度换算正确",
        bool(ok),
        f"实际={res[0]['bbox'] if res else None}（期望≈[0,0,1280,720]）",
    )

    print("=" * 62)
    print(f" 结果：{'全部通过 ✅' if all_pass else '存在失败 ❌'}")
    print("=" * 62)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
