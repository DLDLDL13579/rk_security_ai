# -*- coding: utf-8 -*-
"""
三类安防事件判定 —— 离线单测（不依赖板端、不依赖网络、不依赖模型）

覆盖用户需求的三类事件：
    ① 跌倒      FALL_DETECTED
    ② 区域入侵  INTRUSION_DETECTED
    ③ 长时间逗留 LOITERING_DETECTED

以及：事件去重冷却、状态清理、上报 payload 联动。

用法：
    /usr/local/bin/python3 test_security_events.py
"""

import importlib.util
import json
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


# 先注册包路径，让模块内的 `from engine.xxx import` 可用
os.makedirs(os.path.join(HERE, "engine"), exist_ok=True)
load("engine/fire_smoke_detector.py", "engine.fire_smoke_detector")
load("engine/security_event_manager.py", "engine.security_event_manager")
load("engine/zone_manager.py", "engine.zone_manager")
monitor_mod = load("engine/security_monitor.py", "engine.security_monitor")

SecurityMonitor = monitor_mod.SecurityMonitor
EVENT_FALL = monitor_mod.EVENT_FALL
EVENT_INTRUSION = monitor_mod.EVENT_INTRUSION
EVENT_LOITERING = monitor_mod.EVENT_LOITERING


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


# 入侵区：画面右侧 (x 360..620, y 40..340)；逗留区：左侧 (x 80..260)
ZONES = [
    {
        "id": "intrusion_zone",
        "name": "Restricted Area",
        "type": "intrusion",
        "rect": [360, 40, 620, 340],
    },
    {
        "id": "loiter_zone",
        "name": "Stay Alert Area",
        "type": "loiter",
        "rect": [80, 40, 260, 340],
        "loiter_seconds": 8.0,
    },
]

# 锚点取框底中心：bbox=[x1,y1,x2,y2] → ((x1+x2)/2, y2)
IN_ZONE_BBOX = [450, 100, 550, 300]      # 锚点 (500, 300) → 落在入侵区内
OUT_ZONE_BBOX = [700, 100, 800, 300]     # 锚点 (750, 300) → 区域外
IN_LOITER_BBOX = [130, 100, 230, 300]    # 锚点 (180, 300) → 落在逗留区内


def det(bbox, score=0.9):
    return {"id": 1, "bbox": bbox, "name": "person", "score": score}


def main():
    all_pass = True
    t0 = 1000.0

    print("=" * 66)
    print(" 三类安防事件判定 —— 离线单测")
    print("=" * 66)

    # ================= ① 区域入侵 =================
    m = SecurityMonitor(zones=ZONES, event_hold_seconds=10.0)

    # 人在区域外：不报警
    states, events = m.update({1: det(OUT_ZONE_BBOX)}, {}, now=t0)
    all_pass &= check("区域外不触发入侵", len(events) == 0, f"事件={events}")

    # 人进入区域：触发入侵
    states, events = m.update({1: det(IN_ZONE_BBOX)}, {}, now=t0 + 0.1)
    types = [e["event_type"] for e in events]
    all_pass &= check("进入区域触发 INTRUSION_DETECTED", EVENT_INTRUSION in types, f"事件={types}")
    if events:
        e = events[0]
        all_pass &= check("入侵事件带 person_id", e.get("person_id") == 1, f"实际={e.get('person_id')}")
        all_pass &= check("入侵事件带 zone_name", e.get("zone_name") == "Restricted Area", str(e.get("zone_name")))
        all_pass &= check("入侵事件带 bbox", len(e.get("bbox", [])) == 4, str(e.get("bbox")))
        all_pass &= check("入侵事件带 ts 时间戳", "ts" in e, str(e.get("ts")))
    all_pass &= check(
        "检测状态标记入侵告警",
        states.get(1, {}).get("alerts") == ["INTRUSION"],
        f"实际={states.get(1)}",
    )

    # 持续停留在区域内：冷却期内不重复报警
    _, events2 = m.update({1: det(IN_ZONE_BBOX)}, {}, now=t0 + 0.5)
    all_pass &= check("冷却期内不重复报入侵", len(events2) == 0, f"事件={events2}")

    # 离开后再进入：冷却过后可再次报警
    m.update({1: det(OUT_ZONE_BBOX)}, {}, now=t0 + 1.0)
    _, events3 = m.update({1: det(IN_ZONE_BBOX)}, {}, now=t0 + 5.0)
    all_pass &= check(
        "离开后重新进入可再次报警（冷却已过）",
        EVENT_INTRUSION in [e["event_type"] for e in events3],
        f"事件={[e['event_type'] for e in events3]}",
    )

    # ================= ② 长时间逗留 =================
    m = SecurityMonitor(zones=ZONES, event_hold_seconds=30.0)

    # 进入逗留区，未达时长：不报
    _, events = m.update({2: det(IN_LOITER_BBOX)}, {}, now=t0)
    all_pass &= check("逗留未达时长不报警", len(events) == 0, f"事件={[e['event_type'] for e in events]}")

    # 停留 5 秒（< 8 秒阈值）：仍不报
    _, events = m.update({2: det(IN_LOITER_BBOX)}, {}, now=t0 + 5.0)
    all_pass &= check("逗留 5 秒（阈值 8 秒）不报警", len(events) == 0, f"事件={[e['event_type'] for e in events]}")

    # 停留 8.5 秒：触发逗留
    _, events = m.update({2: det(IN_LOITER_BBOX)}, {}, now=t0 + 8.5)
    types = [e["event_type"] for e in events]
    all_pass &= check("逗留超阈值触发 LOITERING_DETECTED", EVENT_LOITERING in types, f"事件={types}")
    if events:
        e = events[0]
        all_pass &= check("逗留事件带持续时长", e.get("duration", 0) >= 8.0, f"duration={e.get('duration')}")
        all_pass &= check("逗留事件带 zone_id", e.get("zone_id") == "loiter_zone", str(e.get("zone_id")))

    # 继续停留：不重复报（loiter_reported 去重）
    _, events = m.update({2: det(IN_LOITER_BBOX)}, {}, now=t0 + 20.0)
    all_pass &= check(
        "持续逗留不重复报警（去重生效）",
        EVENT_LOITERING not in [e["event_type"] for e in events],
        f"事件={[e['event_type'] for e in events]}",
    )

    # ================= ③ 跌倒（含连续帧确认，方案 A）=================
    m = SecurityMonitor(zones=[], fall_confidence=0.55)

    # 低于置信度：不报
    _, events = m.update({3: det(IN_ZONE_BBOX)}, {3: {"action": "fall_down", "confidence": 0.40}}, now=t0)
    all_pass &= check("跌倒置信度不足不报警", len(events) == 0, f"事件={[e['event_type'] for e in events]}")

    # 其他行为：不报
    _, events = m.update({3: det(IN_ZONE_BBOX)}, {3: {"action": "walking", "confidence": 0.95}}, now=t0)
    all_pass &= check("非跌倒行为不报跌倒", EVENT_FALL not in [e["event_type"] for e in events])

    # === 误报抑制：单帧抖动不应触发（对应实测的 walking→fall_down 瞬时抖动）===
    m2 = SecurityMonitor(zones=[], fall_confidence=0.55, fall_confirm_frames=3)
    _, ev1 = m2.update({3: det(IN_ZONE_BBOX)}, {3: {"action": "fall_down", "confidence": 0.88}}, now=t0)
    all_pass &= check(
        "误报抑制：单帧 fall_down 不触发",
        len(ev1) == 0,
        f"事件={[e['event_type'] for e in ev1]}（应被连续帧过滤挡下）",
    )
    # 中间间隔过长（模拟抖动后恢复正常）→ 计数应清零
    m2.update({3: det(IN_ZONE_BBOX)}, {3: {"action": "walking", "confidence": 0.99}}, now=t0 + 0.1)
    _, ev2 = m2.update({3: det(IN_ZONE_BBOX)}, {3: {"action": "fall_down", "confidence": 0.88}}, now=t0 + 5.0)
    all_pass &= check(
        "误报抑制：间隔超阈值后计数清零（不累积）",
        len(ev2) == 0,
        f"事件={[e['event_type'] for e in ev2]}",
    )

    # === 真实跌倒：连续多帧应正常触发 ===
    m3 = SecurityMonitor(zones=[], fall_confidence=0.55, fall_confirm_frames=3)
    seq_events = []
    for i in range(3):  # 连续 3 帧 fall_down
        _, ev = m3.update(
            {3: det(IN_ZONE_BBOX)}, {3: {"action": "fall_down", "confidence": 0.88}}, now=t0 + i * 0.05
        )
        seq_events.extend(ev)
    types = [e["event_type"] for e in seq_events]
    all_pass &= check(
        "真实跌倒：连续 3 帧后正常触发",
        EVENT_FALL in types,
        f"事件={types}",
    )
    if seq_events:
        e = seq_events[0]
        all_pass &= check("跌倒事件带置信度", abs(e.get("score", 0) - 0.88) < 0.01, f"score={e.get('score')}")
        all_pass &= check("跌倒事件 bbox 从检测结果补齐", len(e.get("bbox", [])) == 4, str(e.get("bbox")))
        all_pass &= check("事件标注确认帧数", e.get("confirmed_frames") == 3, f"实际={e.get('confirmed_frames')}")
    all_pass &= check(
        "被抑制的帧有计数（可观测）",
        m3.stats_suppressed[EVENT_FALL] == 2,
        f"实际={m3.stats_suppressed[EVENT_FALL]}（前 2 帧被挡）",
    )

    # 跌倒状态写入 security_states（连续帧确认后）
    m4 = SecurityMonitor(zones=[], fall_confidence=0.55, fall_confirm_frames=2)
    for i in range(2):
        states, _ = m4.update(
            {4: det(OUT_ZONE_BBOX)}, {4: {"action": "fall_down", "confidence": 0.9}}, now=t0 + i * 0.05
        )
    all_pass &= check(
        "跌倒写入 status.alerts=FALL",
        states.get(4, {}).get("alerts") == ["FALL"],
        f"实际={states.get(4)}",
    )

    # ================= ④ 烟火（含连续帧确认）=================
    m = SecurityMonitor(zones=[], fire_smoke_confirm_frames=3)
    _, ev1 = m.update({}, {}, fire_smoke_objects=[{"bbox": [10, 10, 100, 100], "score": 0.9, "label": "fire"}], now=t0)
    all_pass &= check(
        "误报抑制：单帧火焰不触发",
        len(ev1) == 0,
        f"事件={[e['event_type'] for e in ev1]}",
    )
    seq = []
    for i in range(3):
        _, ev = m.update(
            {}, {},
            fire_smoke_objects=[{"bbox": [10, 10, 100, 100], "score": 0.9, "label": "fire"}],
            now=t0 + i * 0.05,
        )
        seq.extend(ev)
    all_pass &= check(
        "真实火焰：连续 3 帧触发 FIRE_DETECTED",
        "FIRE_DETECTED" in [e["event_type"] for e in seq],
        f"事件={[e['event_type'] for e in seq]}",
    )

    # _8.4 的 flame 别名归一化（连续帧）
    m = SecurityMonitor(zones=[], fire_smoke_confirm_frames=1)
    _, events = m.update({}, {}, fire_smoke_objects=[{"bbox": [10, 10, 100, 100], "score": 0.9, "label": "flame"}], now=t0)
    all_pass &= check(
        "flame 别名归一化为 fire",
        "FIRE_DETECTED" in [e["event_type"] for e in events],
        f"事件={[e['event_type'] for e in events]}",
    )

    # ================= ⑤ 状态清理与统计 =================
    m = SecurityMonitor(zones=ZONES)
    m.update({9: det(IN_ZONE_BBOX)}, {}, now=t0)
    all_pass &= check("目标在区域内有状态记录", (9, "intrusion_zone") in m.zone_manager.person_zone_state)
    # 目标消失（不再出现）：状态应被清理，避免内存泄漏（_8.4 的 display 泄漏教训）
    m.update({}, {}, now=t0 + 1.0)
    all_pass &= check(
        "目标消失后区域状态被清理（无内存泄漏）",
        (9, "intrusion_zone") not in m.zone_manager.person_zone_state,
        f"残留={list(m.zone_manager.person_zone_state.keys())}",
    )

    m = SecurityMonitor(zones=ZONES, fall_confirm_frames=1)
    m.update({1: det(IN_ZONE_BBOX)}, {1: {"action": "fall_down", "confidence": 0.9}}, now=t0)
    all_pass &= check(
        "事件计数统计正确",
        m.stats[EVENT_INTRUSION] == 1 and m.stats[EVENT_FALL] == 1,
        f"实际={m.stats}",
    )

    # ================= ⑥ 端到端：判定 → 上报 payload =================
    reporter_mod = load("engine/mqtt_reporter.py", "engine.mqtt_reporter")
    reporter = reporter_mod.MqttReporter(token="dummy", send_interval=0.0)

    captured = []

    class FakeClient:
        def publish(self, topic, payload):
            captured.append((topic, payload))

        def loop_stop(self):
            pass

        def disconnect(self):
            pass

    reporter._client = FakeClient()
    reporter._connected = True

    m = SecurityMonitor(zones=ZONES)
    dets = {1: det(IN_ZONE_BBOX)}
    _, events = m.update(dets, {}, now=t0)
    reporter.publish_objects(dets, events)

    all_pass &= check("端到端：判定结果成功上报", len(captured) == 1, f"发送次数={len(captured)}")
    if captured:
        payload = json.loads(captured[0][1])
        all_pass &= check(
            "上报 payload 含告警与目标",
            payload["alert_count"] == 1
            and payload["count"] == 1
            and payload["alerts"][0]["alert_type"] == EVENT_INTRUSION,
            f"alerts={payload['alerts']}",
        )
        all_pass &= check(
            "入侵事件字段在 payload 中完整",
            payload["alerts"][0]["alert_type"] == "INTRUSION_DETECTED"
            and len(payload["alerts"][0]["bbox"]) == 4,
            f"alert={payload['alerts'][0]}",
        )

    print("=" * 66)
    print(f" 结果：{'全部通过 ✅' if all_pass else '存在失败 ❌'}")
    print("=" * 66)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
