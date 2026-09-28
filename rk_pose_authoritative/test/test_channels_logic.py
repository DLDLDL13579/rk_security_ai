# -*- coding: utf-8 -*-
"""
合成关键点单测：geometry / locomotion / special 通道逻辑
不依赖 RKNN / MediaPipe / numpy，可直接在本机运行：
    python test_core/test_channels_logic.py
"""

import os
import sys
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.geometry import ankle_amp, cadence, compute_geometry
from engine.locomotion_engine import LocomotionEngine
from engine.special_action_engine import SpecialActionEngine


def lm(x, y):
    return SimpleNamespace(x=float(x), y=float(y), visibility=1.0)


def standing_pose():
    k = [lm(0.0, 0.0) for _ in range(33)]
    k[11] = lm(0.35, 0.28)
    k[12] = lm(0.65, 0.28)
    k[23] = lm(0.38, 0.55)
    k[24] = lm(0.62, 0.55)
    k[25] = lm(0.36, 0.80)
    k[26] = lm(0.64, 0.80)
    k[27] = lm(0.34, 0.98)
    k[28] = lm(0.66, 0.98)
    return k


def walking_pose(frame):
    k = standing_pose()
    phase = 1.0 if frame % 2 == 0 else -1.0
    k[27].y = 0.95 + 0.018 * phase
    k[28].y = 0.95 - 0.018 * phase
    k[27].x = 0.34 + 0.010 * phase
    k[28].x = 0.66 - 0.010 * phase
    return k


def jitter_standing_pose(frame):
    """静止站立 + 踝部微小抖动（模拟关键点噪声/呼吸晃动）"""
    k = standing_pose()
    phase = 1.0 if frame % 2 == 0 else -1.0
    k[27].y = 0.98 + 0.010 * phase
    k[28].y = 0.98 - 0.010 * phase
    return k


def hallucinated_ankle_pose(frame):
    """踝关节不可见 + 幅度巨大（模拟小目标/遮挡下关键点失败）"""
    k = standing_pose()
    phase = 1.0 if frame % 2 == 0 else -1.0
    k[27].y = 0.5 + 1.0 * phase
    k[28].y = 0.5 - 1.0 * phase
    k[27].visibility = 0.05
    k[28].visibility = 0.05
    return k


def squat_pose():
    k = [lm(0.0, 0.0) for _ in range(33)]
    k[11] = lm(0.35, 0.25)
    k[12] = lm(0.65, 0.25)
    k[23] = lm(0.40, 0.50)
    k[24] = lm(0.60, 0.50)
    k[25] = lm(0.35, 0.70)
    k[26] = lm(0.65, 0.70)
    k[27] = lm(0.47, 0.73)
    k[28] = lm(0.53, 0.73)
    return k


def bend_pose():
    k = [lm(0.0, 0.0) for _ in range(33)]
    k[11] = lm(0.30, 0.32)
    k[12] = lm(0.55, 0.36)
    k[23] = lm(0.58, 0.62)
    k[24] = lm(0.82, 0.66)
    k[25] = lm(0.60, 0.86)
    k[26] = lm(0.90, 0.90)
    k[27] = lm(0.58, 0.98)
    k[28] = lm(0.92, 0.98)
    return k


def fall_pose():
    k = [lm(0.0, 0.0) for _ in range(33)]
    k[11] = lm(0.30, 0.40)
    k[12] = lm(0.48, 0.42)
    k[23] = lm(0.62, 0.52)
    k[24] = lm(0.80, 0.54)
    k[25] = lm(0.72, 0.76)
    k[26] = lm(0.92, 0.78)
    k[27] = lm(0.82, 0.98)
    k[28] = lm(1.00, 0.98)
    return k


def partial_squat_pose():
    """半匹配下蹲：膝盖弯曲不足（约120°），但躯干竖直（模拟遮挡/非标准动作）"""
    k = [lm(0.0, 0.0) for _ in range(33)]
    k[11] = lm(0.35, 0.25)
    k[12] = lm(0.65, 0.25)
    k[23] = lm(0.40, 0.50)
    k[24] = lm(0.60, 0.50)
    k[25] = lm(0.35, 0.70)
    k[26] = lm(0.65, 0.70)
    k[27] = lm(0.46, 0.80)
    k[28] = lm(0.54, 0.80)
    return k


FALL_BBOX = [25, 35, 85, 60]


def check(name, cond):
    if cond:
        print("[PASS]", name)
    else:
        print("[FAIL]", name)
        raise SystemExit(1)


def test_geometry():
    g_stand = compute_geometry(standing_pose())
    g_squat = compute_geometry(squat_pose())
    g_bend = compute_geometry(bend_pose())
    g_fall = compute_geometry(fall_pose())

    check("standing knee straight", g_stand["knee_mean"] > 150)
    check("standing tilt ~0", g_stand["torso_tilt"] < 0.2)
    check("squat knee flexed", g_squat["knee_mean"] <= 112)
    check("squat tilt small", g_squat["torso_tilt"] <= 0.65)
    check("bend tilt mid", 0.55 <= g_bend["torso_tilt"] <= 1.45)
    check("bend knee straight", g_bend["knee_mean"] >= 130)
    check("fall tilt large", g_fall["torso_tilt"] >= 0.80)
    check("fall hip drop small", g_fall["hip_drop"] <= 0.62)


def test_locomotion():
    os.environ["RK_LOCO_WARMUP_SAMPLES"] = "3"
    os.environ["RK_LOCO_SWITCH_STREAK"] = "2"
    eng = LocomotionEngine()
    tid = 1

    for i in range(12):
        r = eng.resolve(
            tid, motion=0.03, pose_energy=0.012, pose_quality=0.9,
            keypoints=walking_pose(i),
        )
    check("walking detected", r["action"] == "walking" and r["state"] == "valid")

    # 步态窗口(12)排空 + 2 帧切换确认
    for _ in range(14):
        r = eng.resolve(
            tid, motion=0.002, pose_energy=0.001, pose_quality=0.9,
            keypoints=standing_pose(),
        )
    check("standing detected", r["action"] == "standing" and r["state"] == "valid")


def test_locomotion_jitter_standing():
    """静止但有踝部抖动的场景必须保持 standing（回归：防误判 walking）"""
    os.environ["RK_LOCO_WARMUP_SAMPLES"] = "3"
    os.environ["RK_LOCO_SWITCH_STREAK"] = "2"
    eng = LocomotionEngine()
    tid = 7
    result = None
    for i in range(20):
        result = eng.resolve(
            tid, motion=0.003, pose_energy=0.20, pose_quality=0.9,
            keypoints=jitter_standing_pose(i),
            model_action="walking", model_score=0.95,
        )
    check("jitter standing stays standing",
          result["action"] == "standing" and result["state"] == "valid")


def test_locomotion_hallucinated_ankle():
    """踝点不可见时，幅度巨大的步态信号不得触发 walking"""
    os.environ["RK_LOCO_WARMUP_SAMPLES"] = "3"
    os.environ["RK_LOCO_SWITCH_STREAK"] = "2"
    eng = LocomotionEngine()
    tid = 8
    result = None
    for i in range(20):
        result = eng.resolve(
            tid, motion=0.003, pose_energy=0.20, pose_quality=0.9,
            keypoints=hallucinated_ankle_pose(i),
            model_action="walking", model_score=0.95,
        )
    check("hallucinated ankle stays standing",
          result["action"] == "standing" and result["state"] == "valid")


def test_special():
    os.environ["RK_SPECIAL_FALL_GEO_STREAK"] = "2"
    os.environ["RK_SPECIAL_SQUAT_GEO_STREAK"] = "2"
    os.environ["RK_SPECIAL_BEND_GEO_STREAK"] = "2"
    eng = SpecialActionEngine()

    for tid, pose, bbox, expect in [
        (10, squat_pose, [20, 20, 80, 90], "squat"),
        (11, bend_pose, [20, 30, 90, 95], "bend"),
        (12, fall_pose, FALL_BBOX, "fall_down"),
    ]:
        result = None
        for _ in range(5):
            result = eng.resolve(
                track_id=tid,
                action="",
                score=0.0,
                pose_quality=0.9,
                bbox=bbox,
                keypoints=pose(),
                geometry=compute_geometry(pose()),
            )
        check(expect + " valid", result is not None and result["action"] == expect
              and result["state"] == "valid")

    result = eng.resolve(
        track_id=13, action="", score=0.0, pose_quality=0.9,
        bbox=[20, 20, 80, 90], keypoints=standing_pose(),
        geometry=compute_geometry(standing_pose()),
    )
    check("standing no special", result is None)


def test_special_partial():
    """遮挡/非标准动作：几何半匹配应输出 squat [partial]，而不是回落到 standing"""
    os.environ["RK_SPECIAL_SQUAT_GEO_STREAK"] = "2"
    eng = SpecialActionEngine()
    g = compute_geometry(partial_squat_pose())
    assert 100 < g["knee_mean"] <= 130, g["knee_mean"]
    result = None
    for _ in range(6):
        result = eng.resolve(
            track_id=20, action="", score=0.0, pose_quality=0.9,
            bbox=[20, 20, 80, 90], keypoints=partial_squat_pose(),
            geometry=g,
        )
    check("partial squat confirmed",
          result is not None and result["action"] == "squat"
          and result["state"] == "partial")


def test_special_hold():
    """确认过的特殊动作在短暂遮挡期间保持输出（held），不闪回 standing/walking"""
    os.environ["RK_SPECIAL_SQUAT_GEO_STREAK"] = "2"
    os.environ["RK_SPECIAL_HOLD_FRAMES"] = "3"
    eng = SpecialActionEngine()
    g = compute_geometry(squat_pose())
    result = None
    for _ in range(5):
        result = eng.resolve(
            track_id=21, action="", score=0.0, pose_quality=0.9,
            bbox=[20, 20, 80, 90], keypoints=squat_pose(), geometry=g,
        )
    check("squat confirmed first",
          result is not None and result["action"] == "squat"
          and result["state"] == "valid")

    sg = compute_geometry(standing_pose())
    held_count = 0
    for _ in range(6):
        r = eng.resolve(
            track_id=21, action="", score=0.0, pose_quality=0.9,
            bbox=[20, 20, 80, 90], keypoints=standing_pose(), geometry=sg,
        )
        if r is not None and r["state"] == "held" and r["action"] == "squat":
            held_count += 1
    check("squat held after occlusion", 0 < held_count <= 3)

    final = eng.resolve(
        track_id=21, action="", score=0.0, pose_quality=0.9,
        bbox=[20, 20, 80, 90], keypoints=standing_pose(), geometry=sg,
    )
    check("hold expires", final is None)


def test_gait():
    hist = [walking_pose(i) for i in range(10)]
    check("ankle amp", ankle_amp(hist) >= 0.01)
    check("cadence", cadence(hist) >= 2)
    hist2 = [standing_pose() for _ in range(10)]
    check("standing amp ~0", ankle_amp(hist2) < 0.005)
    check("standing cadence 0", cadence(hist2) == 0)


if __name__ == "__main__":
    test_geometry()
    test_gait()
    test_locomotion()
    test_locomotion_jitter_standing()
    test_locomotion_hallucinated_ankle()
    test_special()
    test_special_partial()
    test_special_hold()
    print("\nALL CHANNEL LOGIC TESTS PASSED")
