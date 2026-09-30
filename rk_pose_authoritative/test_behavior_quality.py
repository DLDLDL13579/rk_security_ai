# -*- coding: utf-8 -*-
"""
行为判定质量重构 —— 离线单测（2026-09-30）

直接针对用户实测反馈的四个现象：
  ① 坐着（脚被遮挡）被判下蹲        → 膝角可见性掩码
  ② 识别不到脚时被判站立            → 位移主证据 + 掩码后无假步态
  ③ 跌倒经常误检                    → 后跌倒阶段（持续 3 秒才确认）
  ④ 抖腿被判行走                   → 行走必须以 bbox 位移为主证据

用法：
    /usr/local/bin/python3 test_behavior_quality.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


geo = load("engine/geometry.py", "engine.geometry")


# ------------------------------------------------------------------
# 造数据工具：Landmark 与关键点序列
# ------------------------------------------------------------------
class LM:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.visibility = v


def make_kpts(knee_v=1.0, ankle_v=1.0, torso_v=1.0, tilt="up", knee_deg_hint=170.0):
    """
    构造 33 点关键点。
    tilt: "up"(直立) / "wide"(横向倒地) / "bend"(前倾)
    knee_deg_hint 通过调整踝 y 相对髋/膝的位置来近似目标角度
    """
    kpts = [LM(0.5, 0.1, 0.0) for _ in range(33)]  # 默认不可见

    # 躯干四点（肩 11/12，髋 23/24）
    tv = torso_v
    if tilt == "up":
        sx, sy, hx, hy = 0.45, 0.30, 0.47, 0.55
    elif tilt == "bend":
        sx, sy, hx, hy = 0.45, 0.30, 0.55, 0.48
    else:  # wide 倒地：肩在左、髋在右，躯干接近水平（dx 大、dy 极小）
        sx, sy, hx, hy = 0.30, 0.40, 0.65, 0.42
    kpts[11] = LM(sx, sy, tv)
    kpts[12] = LM(sx + 0.02, sy + 0.08, tv)
    kpts[23] = LM(hx, hy, tv)
    kpts[24] = LM(hx + 0.02, hy + 0.08, tv)

    # 下肢（髋 23/24 已设）——膝 25/26、踝 27/28
    # 膝角近似：踝相对膝的竖直偏移决定角度
    ky = (hy + 0.55) / 2 + 0.05
    ay = ky + (hy - ky) * (knee_deg_hint / 180.0) + 0.1
    kpts[25] = LM(0.46, ky, knee_v)
    kpts[26] = LM(0.54, ky, knee_v)
    kpts[27] = LM(0.46, ay, ankle_v)
    kpts[28] = LM(0.54, ay, ankle_v)

    return kpts


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def main():
    ok = True
    print("=" * 70)
    print(" 行为判定质量重构 · 离线单测（四大现象）")
    print("=" * 70)

    # ================= ① 脚不可见 → 不再产生假膝角 =================
    print("\n【现象①】坐着但脚被遮挡 → 不应被判下蹲（膝角掩码）")
    kpts_nofoot = make_kpts(ankle_v=0.05, knee_deg_hint=90.0)  # 踝不可见，"坐姿"
    g = geo.compute_geometry(kpts_nofoot)
    ok &= check(
        "踝不可见 → foot_valid=False",
        g["foot_valid"] is False,
        f"foot_valid={g['foot_valid']}",
    )
    ok &= check(
        "踝不可见 → knee_valid=False（膝角不可信）",
        g["knee_valid"] is False,
        f"knee_valid={g['knee_valid']}（原实现会用假踝算出膝角≈{g['knee_mean']:.0f}° 并判下蹲）",
    )

    kpts_withfoot = make_kpts(ankle_v=1.0, knee_deg_hint=90.0)  # 正常坐姿
    g2 = geo.compute_geometry(kpts_withfoot)
    ok &= check(
        "踝可见 → knee_valid=True",
        g2["knee_valid"] is True,
        f"knee_valid={g2['knee_valid']}",
    )

    # 躯干被遮挡（肩髋不可见）→ torso 量不可信
    kpts_notorso = make_kpts(torso_v=0.05, tilt="wide")
    g3 = geo.compute_geometry(kpts_notorso)
    ok &= check(
        "躯干不可见 → torso_valid=False（跌倒几何证据失效）",
        g3["torso_valid"] is False,
        f"torso_valid={g3['torso_valid']}",
    )

    # ================= ② 脚不可见 → 步态证据应为空 =================
    print("\n【现象②】脚不可见时 → 不产生假步态（站立误判源之一）")
    seq_nofoot = [make_kpts(ankle_v=0.05) for _ in range(10)]
    amp, n = geo.ankle_amp(seq_nofoot)
    ok &= check(
        "全部帧踝不可见 → amp 样本数 < 3 → 无步态证据",
        n < 3 and amp == 0.0,
        f"n={n}, amp={amp}（原实现会算出假幅度）",
    )
    cad = geo.cadence(seq_nofoot)
    ok &= check("全部帧踝不可见 → cadence=0", cad == 0, f"cadence={cad}")

    # 踝可见且真实摆动 → 步态证据保留
    seq_walk = []
    for i in range(10):
        k = make_kpts(ankle_v=1.0)
        off = 0.02 if i % 2 == 0 else -0.02
        k[27] = LM(0.46, k[27].y + off, 1.0)
        seq_walk.append(k)
    amp_w, n_w = geo.ankle_amp(seq_walk)
    ok &= check(
        "踝可见且摆动 → amp 正常计算",
        n_w >= 3 and amp_w > 0.03,
        f"n={n_w}, amp={amp_w:.3f}",
    )

    # ================= ③ 跌倒须持续（后跌倒阶段）=================
    print("\n【现象③】跌倒误检 → 后跌倒阶段状态机（持续 3 秒才确认）")
    spec_eng = load(
        "engine/special_action_engine.py", "engine.special_action_engine"
    )
    SE = spec_eng.SpecialActionEngine
    import time as _time

    eng = SE()
    fall_bbox = [0.1, 0.4, 0.9, 0.6]  # 宽 > 高（倒地形态）
    fall_kpts = make_kpts(tilt="wide", knee_v=1.0, ankle_v=1.0)
    fall_geo = geo.compute_geometry(fall_kpts)

    # 单帧瞬态命中 → 不应确认 fall_down
    r = eng.resolve(
        track_id=1, action="", score=0.0, pose_quality=0.9,
        bbox=fall_bbox, keypoints=fall_kpts, geometry=fall_geo, frame_id=100,
    )
    ok &= check(
        "瞬态命中 1 帧 → 不判跌倒（原实现 5 帧 streak 即确认）",
        not (r and r.get("action") == "fall_down" and r.get("state") == "valid"),
        f"r={r}",
    )
    # 需要篡改 since 模拟时间流逝太复杂——直接看状态机字段
    pf = eng.post_fall.get(1)
    ok &= check(
        "状态机进入 candidate（开始计时）",
        pf is not None and pf["phase"] == "candidate",
        f"pf={pf}",
    )

    # 帧号推进到 ≥ fall_confirm_frames → 应确认（帧号计时，不依赖墙钟）
    # 这是 2026-09-30 全量评估暴露的修复：原用墙钟计时，评估全速播放时
    # 永远到不了 3 秒 → 跌倒 SE 从 80% 崩到 30%
    need = eng.fall_confirm_frames
    r2 = eng.resolve(
        track_id=1, action="", score=0.0, pose_quality=0.9,
        bbox=fall_bbox, keypoints=fall_kpts, geometry=fall_geo,
        frame_id=100 + need + 1,
    )
    ok &= check(
        f"持续躺地 ≥{need} 帧（=3 秒@25fps）→ 判跌倒",
        r2 is not None and r2.get("action") == "fall_down",
        f"r2={r2}",
    )
    ok &= check(
        "确认帧数按 3 秒 × 25fps 换算",
        need == 75,
        f"实际={need} 帧",
    )

    # 瞬态（坐下）：命中 1 帧后特征消失 → 状态机清零
    eng2 = SE()
    sit_bbox = [0.3, 0.5, 0.7, 0.9]
    sit_kpts = make_kpts(tilt="up", knee_deg_hint=90.0)
    sit_geo = geo.compute_geometry(sit_kpts)
    eng2.resolve(track_id=9, action="", score=0.0, pose_quality=0.9,
                 bbox=fall_bbox, keypoints=fall_kpts, geometry=fall_geo, frame_id=10)
    # 下一帧特征消失（人起身/坐正）
    eng2.resolve(track_id=9, action="", score=0.0, pose_quality=0.9,
                 bbox=sit_bbox, keypoints=sit_kpts, geometry=sit_geo, frame_id=11)
    pf9 = eng2.post_fall.get(9)
    ok &= check(
        "瞬态命中后特征消失 → 状态机回 idle（不是跌倒）",
        pf9 is None or pf9["phase"] == "idle",
        f"pf9={pf9}",
    )

    # === 崩溃回归测试（2026-09-30 全量评估时实测到 TypeError）===
    # 场景：track 首帧就被模型高置信直接报 fall_down，而几何特征未命中
    # → pf["since"] 仍为 None，旧代码 `_now() - pf["since"]` 抛 TypeError
    eng_crash = SE()
    try:
        eng_crash.resolve(
            track_id=77, action="fall_down", score=0.99, pose_quality=0.9,
            bbox=[0.2, 0.2, 0.8, 0.9],  # 窄高框（非倒地形态，几何不命中）
            keypoints=sit_kpts, geometry=sit_geo,
        )
        ok &= check("回归：模型直接报 fall_down 且 since=None 时不崩溃", True)
    except TypeError as exc:
        ok &= check(
            "回归：模型直接报 fall_down 且 since=None 时不崩溃",
            False,
            f"抛 TypeError: {exc}",
        )

    # ================= ④ 抖腿 ≠ 行走（位移主证据）=================
    print("\n【现象④】抖腿被判行走 → 行走必须以位移为主证据")
    spec_loco = load("engine/locomotion_engine.py", "engine.locomotion_engine")
    LE = spec_loco.LocomotionEngine

    # 抖腿：bbox 不动（motion 极小），踝摆大
    eng3 = LE()
    jiggle_kpts = []
    for i in range(12):
        k = make_kpts(ankle_v=1.0)
        off = 0.05 if i % 2 == 0 else -0.05  # 抖腿：踝大幅快速摆动
        k[27] = LM(0.46, k[27].y + off, 1.0)
        jiggle_kpts.append(k)
    # 先灌历史，再 resolve 一帧
    for k in jiggle_kpts[:-1]:
        eng3.resolve(track_id=5, motion=0.0005, pose_energy=0.3,
                     pose_quality=0.9, keypoints=k,
                     model_action="standing", model_score=0.99)
    r4 = eng3.resolve(
        track_id=5, motion=0.0005, pose_energy=0.3,  # bbox 几乎不动
        pose_quality=0.9, keypoints=jiggle_kpts[-1],
        model_action="standing", model_score=0.99,
    )
    ok &= check(
        "抖腿（位移≈0）+ 模型说 standing → 不判行走",
        r4["action"] == "standing",
        f"action={r4['action']}（原实现踝摆即可触发 walking）",
    )

    # 真实行走：bbox 持续位移 + 步态
    eng4 = LE()
    walk_kpts = [make_kpts(ankle_v=1.0) for _ in range(12)]
    for i, k in enumerate(walk_kpts[:-1]):
        eng4.resolve(track_id=6, motion=0.012, pose_energy=0.02,
                     pose_quality=0.9, keypoints=k,
                     model_action="walking", model_score=0.99)
    r5 = eng4.resolve(
        track_id=6, motion=0.012, pose_energy=0.02,  # 持续位移
        pose_quality=0.9, keypoints=walk_kpts[-1],
        model_action="walking", model_score=0.99,
    )
    ok &= check(
        "真实行走（位移持续）+ 模型说 walking → 判行走",
        r5["action"] == "walking",
        f"action={r5['action']}",
    )

    # 模型说 walking 但人在原地（模型抖动）→ 不采纳
    eng5 = LE()
    stand_kpts = [make_kpts(ankle_v=1.0) for _ in range(12)]
    for k in stand_kpts[:-1]:
        eng5.resolve(track_id=7, motion=0.0005, pose_energy=0.005,
                     pose_quality=0.9, keypoints=k,
                     model_action="standing", model_score=0.99)
    r6 = eng5.resolve(
        track_id=7, motion=0.0005, pose_energy=0.005,
        pose_quality=0.9, keypoints=stand_kpts[-1],
        model_action="walking", model_score=0.99,  # 模型偶发抖动
    )
    ok &= check(
        "模型抖动报 walking 但无位移 → 不判行走（原覆盖 bug 场景）",
        r6["action"] == "standing",
        f"action={r6['action']}",
    )

    print()
    print("=" * 70)
    print(f" 结果：{'全部通过 ✅' if ok else '存在失败 ❌'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
