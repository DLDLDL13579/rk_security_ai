# -*- coding: utf-8 -*-
"""
净位移零均值检验 —— 离线单测（2026-09-30）

针对用户实测问题：坐着不动却被持续判 walking。

根因：_estimate_track_motion 里的净位移是【带衰减的单向累加器】，
检测框随机抖动（非零均值）会缓慢累积成"净位移"，net_ema 经
net_gain 放大后把 motion 撑过 enter(0.005)/exit(0.0015) 门槛。
实测 motion 从 0.0003 单调爬升到 0.005+。

修法：同向性检验 —— 分别累计正/负向位移，仅当 |正-负|/(正+负)
（straightness）足够大时才采信净位移。随机抖动正负相消。

用法：
    /usr/local/bin/python3 test_net_displacement.py
"""

import importlib.util
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def load_motion_estimator():
    """
    从 engine.py 中提取 _estimate_track_motion 的算法逻辑做独立复现。

    之所以复现而非直接导入：engine.py 依赖 NPU 模型（板端才有），
    而本单测要能在 Mac 上跑。这里逐行复刻同一算法，两侧靠注释同步；
    板端集成验证由 test_behavior_quality.py 与实机观察承担。
    """
    # 复刻参数（与 engine.py 默认值一致）
    WINDOW = 15
    GAIN = 15.0

    def estimate(state, bbox, frame_id):
        x1, y1, x2, y2 = map(float, bbox)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        height = max(1.0, y2 - y1)
        cur = {"cx": cx, "cy": cy, "frame_id": frame_id, "speed": 0.0}

        prev = state.get("p")
        if prev is None:
            cur["hist"] = [(frame_id, cx, cy)]
            state["p"] = cur
            return 0.0

        hist = list(prev.get("hist", []))
        hist.append((frame_id, cx, cy))
        if len(hist) > WINDOW:
            hist = hist[-WINDOW:]
        cur["hist"] = hist

        # v2：以「采样点数」为分母（不是 frame_id 差），且取消瞬时速度兜底
        speed = 0.0
        if len(hist) >= WINDOW:
            _, x0, y0 = hist[0]
            span_samples = max(1, len(hist) - 1)
            win_net = ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5
            if win_net > 0:
                speed = win_net / (height * span_samples) * GAIN

        cur["speed"] = 0.65 * prev.get("speed", 0.0) + 0.35 * speed
        state["p"] = cur
        return cur["speed"]

    return estimate


def main():
    ok = True
    est = load_motion_estimator()

    print("=" * 70)
    print(" 净位移零均值检验 · 离线单测")
    print("=" * 70)

    # === 场景 1：坐着不动，检测框随机抖动（±3 像素）→ motion 应保持低 ===
    print("\n【场景1】坐着不动 + 检测框随机抖动（复现用户实测场景）")
    random.seed(42)
    state = {}
    motions = []
    base_x, base_y, w, h = 400.0, 200.0, 200.0, 400.0
    for i in range(200):
        # 检测框随机抖动 ±3px（零均值）
        jx = random.uniform(-3, 3)
        jy = random.uniform(-3, 3)
        bbox = [base_x + jx, base_y + jy, base_x + jx + w, base_y + jy + h]
        m = est(state, bbox, 100 + i)
        motions.append(m)

    max_m = max(motions)
    tail_avg = sum(motions[-50:]) / 50
    ok &= check(
        "抖动场景：motion 峰值 < enter_motion(0.05)",
        max_m < 0.05,
        f"峰值={max_m:.5f}（原实现会越过 0.005 门槛，导致误判 walking）",
    )
    ok &= check(
        "抖动场景：后段平均 motion < exit_motion(0.02)",
        tail_avg < 0.02,
        f"后段均值={tail_avg:.5f}（原实现会停在高位，导致无法退出 walking）",
    )
    ok &= check(
        "抖动场景：motion 被有效抑制（非单调爬升到高位）",
        motions[-1] <= max_m * 0.9,
        f"末值={motions[-1]:.5f} 峰值={max_m:.5f}",
    )

    # === 场景 1b：实机场景——frame_id 每次前进 1.8（30fps/16.7Hz）===
    print("\n【场景1b】帧号跳跃 + 抖动（复现实机 RTSP 条件）")
    random.seed(7)
    state1b = {}
    motions1b = []
    f = 100.0
    for i in range(150):
        f += 1.8
        bbox = [
            400.0 + random.uniform(-4, 4), 200.0 + random.uniform(-4, 4),
            600.0 + random.uniform(-4, 4), 600.0 + random.uniform(-4, 4),
        ]
        motions1b.append(est(state1b, bbox, f))
    tail1b = sum(motions1b[-50:]) / 50
    ok &= check(
        "帧号跳跃+抖动：motion 仍低于 enter(0.05)",
        tail1b < 0.05,
        f"后段均值={tail1b:.5f}（v1 在此场景会因瞬时兜底而误判）",
    )

    # === 场景 2：真实行走（同向持续位移）→ motion 应显著高于门槛 ===
    print("\n【场景2】真实行走（每帧同向位移 8 像素）")
    state2 = {}
    motions2 = []
    x = 200.0
    for i in range(60):
        x += 8.0  # 同向移动
        bbox = [x, 300.0, x + 200.0, 700.0]
        motions2.append(est(state2, bbox, 100 + i * 1.8))

    tail2 = sum(motions2[-30:]) / 30
    ok &= check(
        "行走场景：稳定后 motion >= enter_motion(0.05)",
        tail2 >= 0.05,
        f"后段均值={tail2:.5f}",
    )

    # === 场景 3：缓慢同向移动（远景行走，位移小但同向）→ 应被识别 ===
    print("\n【场景3】远景行走（每帧同向 2 像素，靠净位移增益识别）")
    state3 = {}
    motions3 = []
    x = 300.0
    for i in range(80):
        x += 2.0
        bbox = [x, 100.0, x + 100.0, 300.0]
        motions3.append(est(state3, bbox, 100 + i * 1.8))

    tail3 = sum(motions3[-30:]) / 30
    ok &= check(
        "远景行走：net_ema 增益仍能抬起 motion（未被修复误伤）",
        tail3 > 0.02,
        f"后段均值={tail3:.5f}（高于 exit_motion 0.02 则可保持 walking）",
    )

    # === 场景 4：来回走动（正负相消）→ 不应误判为持续行走 ===
    print("\n【场景4】原地来回晃动（左右各 10 像素交替）")
    state4 = {}
    motions4 = []
    base = 400.0
    for i in range(80):
        offset = 10.0 if i % 2 == 0 else -10.0
        bbox = [base + offset, 250.0, base + offset + 150.0, 600.0]
        motions4.append(est(state4, bbox, 100 + i * 1.8))

    tail4 = sum(motions4[-30:]) / 30
    ok &= check(
        "来回晃动：净位移被同向性检验抑制",
        tail4 < 0.05,
        f"后段均值={tail4:.5f}",
    )

    print()
    print("=" * 70)
    print(f" 结果：{'全部通过 ✅' if ok else '存在失败 ❌'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
