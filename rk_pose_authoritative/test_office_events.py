# -*- coding: utf-8 -*-
"""
办公区五事件 —— 离线单测（2026-09-30）

覆盖用户明确的办公区事件清单：
    离岗（30 分钟）、睡岗（静止 + 姿态，B 方案）、区域入侵、跌倒、烟火

本单测聚焦新增的【离岗】与【睡岗】（其余三项已由
test_security_events.py 覆盖）。

用法：
    /usr/local/bin/python3 test_office_events.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AM = load("engine/attendance_monitor.py", "engine.attendance_monitor")
AttendanceMonitor = AM.AttendanceMonitor
EVENT_OFF_DUTY = AM.EVENT_OFF_DUTY
EVENT_SLEEPING = AM.EVENT_SLEEPING


class LM:
    def __init__(self, x, y, v=1.0):
        self.x = x
        self.y = y
        self.visibility = v


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return cond


# 工位区域（推理分辨率 640x360 空间）：左下工位
DESK_ZONE = {
    "id": "desk_a",
    "name": "工位A",
    "type": "desk",
    "rect": [40, 60, 300, 340],
    "enabled": True,
}

# 区域外的人（锚点在工位右侧）
OUT_BBOX = [400, 100, 500, 300]
# 区域内的人（锚点落在工位内）
IN_BBOX = [120, 100, 220, 300]


def person(bbox, motion=0.0):
    return {1: {"bbox": bbox, "motion": motion}}


def make_sleep_kpts(tilted=True):
    """构造 33 点：tilted=True 为伏案姿态（躯干前倾），False 为直立"""
    k = [LM(0.5, 0.5, 0.0) for _ in range(33)]
    if tilted:
        # 伏案姿态：髋位于肩的【后下方】——dx 明显 > 0
        # 肩中心 (0.40, 0.30)，髋中心 (0.46, 0.40) → dx=0.06 dy=0.10 → tilt=0.60
        # 落在判据区间 [0.18, 1.20] 内
        k[11] = LM(0.39, 0.30); k[12] = LM(0.41, 0.30)
        k[23] = LM(0.45, 0.40); k[24] = LM(0.47, 0.40)
    else:
        # 直立姿态：肩髋 x 对齐 → dx≈0 → tilt≈0（低于下界 0.18）
        k[11] = LM(0.39, 0.30); k[12] = LM(0.41, 0.30)
        k[23] = LM(0.39, 0.50); k[24] = LM(0.41, 0.50)
    return k


def main():
    ok = True
    t0 = 100000.0

    print("=" * 72)
    print(" 办公区五事件 · 离线单测（离岗 + 睡岗）")
    print("=" * 72)

    # ================== 离岗（30 分钟） ==================
    print("\n【离岗】区域内连续无人 30 分钟")
    m = AttendanceMonitor(off_duty_seconds=1800, off_duty_cooldown=600)

    # 1) 有人在岗 → 不报
    ev = m.evaluate_off_duty([DESK_ZONE], person(IN_BBOX), now=t0)
    ok &= check("有人在岗不报离岗", len(ev) == 0, f"事件={ev}")

    # 2) 人离开 29 分钟 → 仍不报（未达 30 分钟阈值）
    ev = m.evaluate_off_duty([DESK_ZONE], {}, now=t0 + 29 * 60)
    ok &= check("离开 29 分钟不报（阈值 30 分钟）", len(ev) == 0, f"事件={len(ev)} 个")

    # 3) 离开 31 分钟 → 报离岗
    ev = m.evaluate_off_duty([DESK_ZONE], {}, now=t0 + 31 * 60)
    ok &= check("离开 31 分钟 → 报离岗", len(ev) == 1, f"事件={len(ev)} 个")
    if ev:
        e = ev[0]
        ok &= check("离岗事件带时长信息", e.get("idle_minutes", 0) >= 30,
                    f"idle_minutes={e.get('idle_minutes')}")
        ok &= check("离岗事件带工位名", e.get("zone_name") == "工位A", str(e.get("zone_name")))

    # 4) 冷却期内重复触发 → 被抑制
    ev = m.evaluate_off_duty([DESK_ZONE], {}, now=t0 + 32 * 60)
    ok &= check("冷却期内不重复报离岗", len(ev) == 0, f"事件={len(ev)} 个")

    # 5) 人回到工位 → 计时清零
    m.evaluate_off_duty([DESK_ZONE], person(IN_BBOX), now=t0 + 40 * 60)
    ev = m.evaluate_off_duty([DESK_ZONE], {}, now=t0 + 45 * 60)
    ok &= check("人回来后再离开 → 重新计时（不立即报警）", len(ev) == 0, f"事件={len(ev)} 个")

    # 6) 首次观察就没人 → 不立即报警（以当前时刻为起点）
    m2 = AttendanceMonitor(off_duty_seconds=1800)
    ev = m2.evaluate_off_duty([DESK_ZONE], {}, now=t0)
    ok &= check("开机首次即无人 → 不误报（以当前时刻为起点）", len(ev) == 0, f"事件={len(ev)} 个")

    # 7) 非 desk 区域不参与离岗判定
    other_zone = dict(DESK_ZONE)
    other_zone["type"] = "intrusion"
    m3 = AttendanceMonitor(off_duty_seconds=60)
    m3.evaluate_off_duty([other_zone], person(IN_BBOX), now=t0)
    ev = m3.evaluate_off_duty([other_zone], {}, now=t0 + 120)
    ok &= check("intrusion 区域不参与离岗判定", len(ev) == 0, f"事件={len(ev)} 个")

    # ================== 睡岗（B 方案：静止 + 姿态） ==================
    print("\n【睡岗】B 方案 —— 静止 5 分钟 + 伏案/低头姿态")

    # 1) 静止但直立（正常坐着工作）→ 不报（这是 B 方案的关键）
    ms = AttendanceMonitor(sleep_still_seconds=300, sleeping_cooldown=300)
    kpts_upright = make_sleep_kpts(tilted=False)
    ev = []
    for i in range(20):
        ev = ms.evaluate_sleeping(
            person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
            keypoints_map={1: kpts_upright}, now=t0 + i * 30,
        )
    ev = ms.evaluate_sleeping(
        person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
        keypoints_map={1: kpts_upright}, now=t0 + 20 * 30 + 400,
    )
    ok &= check(
        "静止 5 分钟但姿态直立 → 不报睡岗（B 方案核心）",
        len(ev) == 0,
        f"事件={len(ev)} 个（A 方案会误报）",
    )

    # 2) 静止 + 伏案姿态 → 报睡岗
    ms2 = AttendanceMonitor(sleep_still_seconds=300, sleeping_cooldown=300)
    kpts_bent = make_sleep_kpts(tilted=True)
    ev = []
    for i in range(30):
        ev = ms2.evaluate_sleeping(
            person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
            keypoints_map={1: kpts_bent}, now=t0 + i * 30,
        )
        if ev:
            break
    ok &= check(
        "静止 + 伏案姿态 → 报睡岗",
        len(ev) == 1,
        f"事件={len(ev)} 个",
    )
    if ev:
        ok &= check("睡岗事件带静止时长", ev[0].get("still_minutes", 0) >= 5,
                    f"still_minutes={ev[0].get('still_minutes')}")
        ok &= check("睡岗事件带姿态说明", bool(ev[0].get("posture")),
                    f"posture={ev[0].get('posture')}")

    # 3) 姿态不可信（关键点遮挡）→ 不报（宁可漏判）
    ms3 = AttendanceMonitor(sleep_still_seconds=60)
    kpts_hidden = [LM(0.5, 0.5, 0.05) for _ in range(33)]  # 全部低可见性
    ev = []
    for i in range(20):
        ev = ms3.evaluate_sleeping(
            person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
            keypoints_map={1: kpts_hidden}, now=t0 + i * 30,
        )
    ok &= check(
        "姿态不可信（遮挡）→ 不报睡岗（防误报）",
        len(ev) == 0,
        f"事件={len(ev)} 个",
    )

    # 4) 期间有运动 → 重新计时（人动过不算睡岗）
    ms4 = AttendanceMonitor(sleep_still_seconds=300)
    ev = []
    for i in range(12):
        ev = ms4.evaluate_sleeping(
            person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
            keypoints_map={1: kpts_bent}, now=t0 + i * 30,
        )
    # 第 6 分钟动了
    ms4.evaluate_sleeping(
        person(IN_BBOX, motion=0.5), {1: {"motion": 0.5}},
        keypoints_map={1: kpts_bent}, now=t0 + 360,
    )
    ev = ms4.evaluate_sleeping(
        person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
        keypoints_map={1: kpts_bent}, now=t0 + 400,
    )
    ok &= check(
        "中途有运动 → 重新计时（不报）",
        len(ev) == 0,
        f"事件={len(ev)} 个",
    )

    # 5) 睡岗冷却去重：观察 500 秒（< 冷却 600 秒）应只报 1 次
    ms5 = AttendanceMonitor(sleep_still_seconds=60, sleeping_cooldown=600)
    n_events = 0
    times = []
    for i in range(17):  # 17 × 30 = 510 秒 < 600 秒冷却
        now = t0 + i * 30
        ev = ms5.evaluate_sleeping(
            person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
            keypoints_map={1: kpts_bent}, now=now,
        )
        n_events += len(ev)
        if ev:
            times.append(i * 30)
    ok &= check(
        "睡岗冷却生效（510 秒内只报 1 次，冷却 600 秒）",
        n_events == 1,
        f"实际 {n_events} 次，触发时刻={times}",
    )

    # 6) 冷却过后可再次报警（验证不是永久抑制）
    ev = ms5.evaluate_sleeping(
        person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
        keypoints_map={1: kpts_bent}, now=t0 + 17 * 30 + 700,
    )
    ok &= check(
        "冷却过后可再次报睡岗（非永久抑制）",
        len(ev) == 1,
        f"事件={len(ev)} 个",
    )

    # ================== 真机格式回归（2026-09-30） ==================
    # 真机暴露：engine.py 序列化后的 keypoints 是简单 dict {"x","y","v"}，
    # 而单测造的是 Landmark 对象。修复前每帧抛
    #   'dict' object has no attribute 'x'
    # 睡岗判据因此永远不生效。本组用 dict 格式复跑同一判据，锁死该 bug。
    print("\n【真机格式回归】keypoints 为 dict（engine.py 的 kpts_simple）")

    def make_sleep_kpts_dict(tilted=True):
        """与 make_sleep_kpts 相同姿态，但用真机的 dict 格式 {"x","y","v"}"""
        k = [{"x": 0.5, "y": 0.5, "v": 0.0} for _ in range(33)]
        if tilted:
            k[11] = {"x": 0.39, "y": 0.30, "v": 1.0}
            k[12] = {"x": 0.41, "y": 0.30, "v": 1.0}
            k[23] = {"x": 0.45, "y": 0.40, "v": 1.0}
            k[24] = {"x": 0.47, "y": 0.40, "v": 1.0}
        else:
            k[11] = {"x": 0.39, "y": 0.30, "v": 1.0}
            k[12] = {"x": 0.41, "y": 0.30, "v": 1.0}
            k[23] = {"x": 0.39, "y": 0.50, "v": 1.0}
            k[24] = {"x": 0.41, "y": 0.50, "v": 1.0}
        return k

    md = AttendanceMonitor(sleep_still_seconds=300, sleeping_cooldown=0)
    kd = make_sleep_kpts_dict(tilted=True)
    # 先灌够静止时长与姿态帧数
    for i in range(10):
        md.evaluate_sleeping(
            person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
            keypoints_map={1: kd}, now=2000.0 + i,
        )
    ev = md.evaluate_sleeping(
        person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
        keypoints_map={1: kd}, now=2000.0 + 400,
    )
    ok &= check(
        "dict 格式关键点：伏案姿态 + 静止 → 报睡岗（修复前会抛异常）",
        len(ev) == 1,
        f"事件={len(ev)} 个",
    )

    # 直立姿态不报（dict 格式下同样成立）
    mu = AttendanceMonitor(sleep_still_seconds=300, sleeping_cooldown=0)
    ku = make_sleep_kpts_dict(tilted=False)
    for i in range(12):
        mu.evaluate_sleeping(
            person(IN_BBOX, motion=0.0), {1: {"motion": 0.0}},
            keypoints_map={1: ku}, now=3000.0 + i * 40,
        )
    ok &= check(
        "dict 格式关键点：直立姿态静止 → 不报睡岗",
        mu.stats[EVENT_SLEEPING] == 0,
        f"误报 {mu.stats[EVENT_SLEEPING]} 次",
    )

    # 两种格式必须给出一致判定（防止适配层只修好一种）
    ok &= check(
        "Landmark 与 dict 两种格式判定一致（tilt 相同）",
        AttendanceMonitor._kpt_get(LM(0.39, 0.30, 1.0), "x")
        == AttendanceMonitor._kpt_get({"x": 0.39, "y": 0.30, "v": 1.0}, "x"),
        "取值适配结果不一致",
    )

    # ================== 配置与统计 ==================
    print("\n【配置与可观测性】")
    m6 = AttendanceMonitor()
    ok &= check("默认离岗阈值 = 1800 秒（30 分钟）", m6.off_duty_seconds == 1800,
                f"实际={m6.off_duty_seconds}")
    ok &= check("默认睡岗静止阈值 = 300 秒（5 分钟）", m6.sleep_still_seconds == 300,
                f"实际={m6.sleep_still_seconds}")
    st = m6.status()
    ok &= check("status 暴露配置与统计",
                st.get("off_duty_seconds") == 1800 and "stats" in st,
                f"keys={sorted(st.keys())}")

    print()
    print("=" * 72)
    print(f" 结果：{'全部通过 ✅' if ok else '存在失败 ❌'}")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
