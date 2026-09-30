# -*- coding: utf-8 -*-
"""
============================================================
engine/attendance_monitor.py

办公区在岗事件判定：离岗、睡岗（2026-09-30 新增）

需求（用户 2026-09-30 明确）：
    办公区五个事件 —— 离岗、睡岗、区域入侵、跌倒、烟火
    其中 区域入侵/跌倒/烟火 已有实现；本模块补齐【离岗】与【睡岗】。

设计依据（行业做法调研）：
    安防系统关心的是【事件告警】而非【逐帧状态分类】。行业主流方案
    （海康、鲲云、华安泰等）输出的是"发现未戴安全帽""检测到离岗"这类
    事件，而不是"当前是站立/行走"。本模块即按事件式设计：
    只有满足完整判据（时长 + 条件）才产生一次事件，带冷却去重。

判据定义：
    ┌─ 离岗（OFF_DUTY）────────────────────────────────────────┐
    │  工位区域（type=desk）内【连续无人】达到 off_duty_seconds   │
    │  · 用户指定 30 分钟（1800 秒）                             │
    │  · 用墙钟计时（30 分钟是真实时间，与视频速度无关）          │
    │  · 恢复有人后立即清零；事件带冷却，避免重复刷屏             │
    └──────────────────────────────────────────────────────────┘
    ┌─ 睡岗（SLEEPING）────────────────────────────────────────┐
    │  【B 方案：静止 + 姿态】双判据，用户选定                    │
    │  ① 连续静止 ≥ sleep_still_seconds（默认 300 秒）           │
    │     静止 = 检测框位移低于阈值（复用 motion 语义）           │
    │  ② 姿态符合伏案/后仰：躯干倾斜或头部低垂                    │
    │     · torso_tilt 落在 [tilt_lo, tilt_hi]（前倾伏案）        │
    │     · 或 头部（鼻）相对肩部中心的竖直偏移超阈值（低头）      │
    │  ③ 姿态不可信（关键点遮挡）时不判 —— 宁可漏判不误报         │
    └──────────────────────────────────────────────────────────┘

环境变量：
    RK_OFFDUTY_SECONDS      离岗时长阈值（默认 1800 = 30 分钟）
    RK_SLEEP_STILL_SECONDS  睡岗静止时长（默认 300 = 5 分钟）
    RK_SLEEP_TILT_LO/HI     伏案躯干倾斜区间（默认 0.18 / 1.20）
    RK_SLEEP_HEAD_DROP      低头阈值：鼻相对肩中心下移量/身高（默认 0.06）
    RK_ATTENDANCE_ENABLED   置 0 关闭本模块
"""

import os
import time


EVENT_OFF_DUTY = "OFF_DUTY_DETECTED"
EVENT_SLEEPING = "SLEEPING_DETECTED"


class AttendanceMonitor:
    """工位在岗事件判定（离岗 + 睡岗）"""

    def __init__(
        self,
        off_duty_seconds=None,
        sleep_still_seconds=None,
        sleep_tilt_lo=None,
        sleep_tilt_hi=None,
        sleep_head_drop=None,
        off_duty_cooldown=None,
        sleeping_cooldown=None,
        enabled=None,
    ):
        # 离岗阈值：用户指定 30 分钟
        self.off_duty_seconds = float(
            os.environ.get("RK_OFFDUTY_SECONDS", "1800")
            if off_duty_seconds is None
            else off_duty_seconds
        )
        # 睡岗静止阈值
        self.sleep_still_seconds = float(
            os.environ.get("RK_SLEEP_STILL_SECONDS", "300")
            if sleep_still_seconds is None
            else sleep_still_seconds
        )
        # 伏案姿态区间（躯干水平/垂直比）
        self.sleep_tilt_lo = float(
            os.environ.get("RK_SLEEP_TILT_LO", "0.18")
            if sleep_tilt_lo is None
            else sleep_tilt_lo
        )
        self.sleep_tilt_hi = float(
            os.environ.get("RK_SLEEP_TILT_HI", "1.20")
            if sleep_tilt_hi is None
            else sleep_tilt_hi
        )
        # 低头阈值：鼻相对肩中心的下移量 / 身高
        self.sleep_head_drop = float(
            os.environ.get("RK_SLEEP_HEAD_DROP", "0.06")
            if sleep_head_drop is None
            else sleep_head_drop
        )
        self.enabled = (
            os.environ.get("RK_ATTENDANCE_ENABLED", "1").strip().lower()
            not in ("0", "false", "no", "off")
            if enabled is None
            else bool(enabled)
        )

        # 每个工位区域的"最后有人时间"：zone_id -> timestamp
        self.zone_last_seen = {}
        # 每个轨迹的静止起点与姿态累计：track_id -> {...}
        self.person_state = {}
        # 事件冷却（秒）：离岗 10 分钟、睡岗 5 分钟，避免重复刷屏
        self.off_duty_cooldown = float(
            os.environ.get("RK_OFFDUTY_COOLDOWN", "600")
            if off_duty_cooldown is None
            else off_duty_cooldown
        )
        self.sleeping_cooldown = float(
            os.environ.get("RK_SLEEPING_COOLDOWN", "300")
            if sleeping_cooldown is None
            else sleeping_cooldown
        )
        self.last_event_at = {}

        self.stats = {EVENT_OFF_DUTY: 0, EVENT_SLEEPING: 0}
        self.stats_suppressed = {EVENT_OFF_DUTY: 0, EVENT_SLEEPING: 0}

    # ------------------------------------------------------------------
    def _cooldown_ok(self, key, now, cooldown):
        last = self.last_event_at.get(key, 0.0)
        if now - last < cooldown:
            return False
        self.last_event_at[key] = now
        return True

    # ------------------------------------------------------------------
    def evaluate_off_duty(self, zones, detections, now=None):
        """
        离岗判定：工位区域内连续无人达到阈值。

        zones      : ZoneManager 的 zone dict 列表（取 type == "desk"）
        detections : {track_id: {"bbox": [...]}} 当前帧检测
        """
        now = time.time() if now is None else float(now)
        events = []

        for zone in zones or []:
            if not zone.get("enabled", True):
                continue
            ztype = str(zone.get("type", "")).strip().lower()
            if ztype not in ("desk", "attendance"):
                continue

            zid = zone["id"]
            # 判断区域内是否有人（用锚点：框底中心）
            occupied = False
            for pid, det in (detections or {}).items():
                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                px = (float(x1) + float(x2)) * 0.5
                py = float(max(y1, y2))
                zx1, zy1, zx2, zy2 = zone["rect"]
                if zx1 <= px <= zx2 and zy1 <= py <= zy2:
                    occupied = True
                    break

            if occupied:
                # 有人在岗：刷新时间戳，清零累计
                self.zone_last_seen[zid] = now
                continue

            # 区域内无人：从"最后一次有人"开始计时
            last = self.zone_last_seen.get(zid)
            if last is None:
                # 首次观察该区域就没人：以当前时刻为起点，避免开机即误报
                self.zone_last_seen[zid] = now
                continue

            idle = now - last
            if idle < self.off_duty_seconds:
                continue

            key = f"offduty:{zid}"
            if not self._cooldown_ok(key, now, self.off_duty_cooldown):
                self.stats_suppressed[EVENT_OFF_DUTY] += 1
                continue

            # 发出事件后重置计时起点：下次报警需重新累积满整个离岗时长
            self.zone_last_seen[zid] = now

            self.stats[EVENT_OFF_DUTY] += 1
            events.append(
                {
                    "event_type": EVENT_OFF_DUTY,
                    "source_key": key,
                    "zone_id": zid,
                    "zone_name": zone.get("name", zid),
                    "duration": idle,
                    "idle_minutes": round(idle / 60.0, 1),
                    "bbox": list(zone["rect"]),
                }
            )

        return events

    # ------------------------------------------------------------------
    def evaluate_sleeping(self, detections, behaviors, keypoints_map=None, now=None):
        """
        睡岗判定（B 方案：静止 + 姿态）。

        detections     : {track_id: {"bbox": [...]}}
        behaviors      : {track_id: {"motion": float, ...}}（motion 为新量纲）
        keypoints_map  : {track_id: [Landmark...]}（可选，用于姿态判据）
        """
        now = time.time() if now is None else float(now)
        events = []
        still_motion_max = float(os.environ.get("RK_SLEEP_MOTION_MAX", "0.05"))

        seen = set()
        for pid, det in (detections or {}).items():
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            seen.add(pid)

            beh = (behaviors or {}).get(pid) or {}
            motion = float(beh.get("motion", 0.0))
            kpts = (keypoints_map or {}).get(pid)
            posture_ok, posture_info = self._sleep_posture(kpts, bbox)

            st = self.person_state.setdefault(
                pid, {"still_since": None, "posture_frames": 0}
            )

            if motion <= still_motion_max:
                if st["still_since"] is None:
                    st["still_since"] = now
                # 姿态判据：连续满足若干次才认为姿态成立（抗单帧抖动）
                if posture_ok:
                    st["posture_frames"] += 1
                else:
                    st["posture_frames"] = max(0, st["posture_frames"] - 1)
            else:
                # 有运动 → 重新计时（人动过就不算睡岗）
                st["still_since"] = None
                st["posture_frames"] = 0

            still = (now - st["still_since"]) if st["still_since"] else 0.0
            if still < self.sleep_still_seconds:
                continue
            # B 方案：必须同时满足姿态判据（连续 3 次以上）
            if st["posture_frames"] < 3:
                self.stats_suppressed[EVENT_SLEEPING] += 1
                continue

            key = f"sleeping:{pid}"
            if not self._cooldown_ok(key, now, self.sleeping_cooldown):
                self.stats_suppressed[EVENT_SLEEPING] += 1
                continue

            # 冷却通过并发出事件后：重置静止计时起点，
            # 使下次报警需要【重新累积】整个静止时长（否则下一帧会再次满足
            # still >= 阈值 的条件，冷却一过就重复报警）
            st["still_since"] = now
            st["posture_frames"] = 0

            self.stats[EVENT_SLEEPING] += 1
            events.append(
                {
                    "event_type": EVENT_SLEEPING,
                    "source_key": key,
                    "person_id": pid,
                    "bbox": list(bbox),
                    "duration": still,
                    "still_minutes": round(still / 60.0, 1),
                    "posture": posture_info,
                }
            )

        # 清理消失轨迹，防内存泄漏
        for pid in list(self.person_state.keys()):
            if pid not in seen:
                self.person_state.pop(pid, None)

        return events

    # ------------------------------------------------------------------
    @staticmethod
    def _kpt_get(pt, field):
        """
        关键点取值适配（2026-09-30 真机修复）。

        真机上 behaviors 里的 keypoints 是 engine.py 序列化后的简单 dict
        {"x":..,"y":..,"v":..}（engine.py 的 kpts_simple，供显示层画骨架），
        而离线单测造的是带 .x/.y/.visibility 的 Landmark 对象。
        两种格式混用会抛 'dict' object has no attribute 'x'，
        导致每帧刷异常、睡岗判据永远不生效（真机实测到的正是这个）。

        统一为"按字段取值"，两种格式都能读：
            Landmark 对象 → pt.x / pt.visibility
            dict          → pt["x"] / pt["v"]（dict 的可见性字段名是 v）
        """
        if isinstance(pt, dict):
            if field == "visibility":
                return float(pt.get("v", pt.get("visibility", 1.0)))
            return float(pt.get(field, 0.0))
        if field == "visibility":
            return float(getattr(pt, "visibility", 1.0))
        return float(getattr(pt, field, 0.0))

    def _sleep_posture(self, kpts, bbox):
        """
        姿态判据（B 方案核心）：伏案 或 低头。

        返回 (是否满足, 说明字符串)。关键点不可信时返回 (False, "no_pose")
        —— 宁可漏判也不误报（这是用户选 B 方案的意图）。

        关键点同时支持 Landmark 对象与简单 dict 两种格式，见 _kpt_get。
        """
        if not kpts or len(kpts) < 29:
            return False, "no_pose"

        def gx(i):
            return self._kpt_get(kpts[i], "x")

        def gy(i):
            return self._kpt_get(kpts[i], "y")

        def vis(i):
            return self._kpt_get(kpts[i], "visibility") > 0.3

        # 躯干四点必须可见，否则姿态不可信
        if not all(vis(i) for i in (11, 12, 23, 24)):
            return False, "torso_hidden"

        sx = (gx(11) + gx(12)) * 0.5
        sy = (gy(11) + gy(12)) * 0.5
        hx = (gx(23) + gx(24)) * 0.5
        hy = (gy(23) + gy(24)) * 0.5

        dy = abs(hy - sy)
        tilt = abs(hx - sx) / max(dy, 1e-6)

        # 判据一：伏案（躯干前倾落在区间内）
        if self.sleep_tilt_lo <= tilt <= self.sleep_tilt_hi:
            return True, f"tilt={tilt:.2f}"

        # 判据二：低头（鼻相对肩中心下移超过阈值）
        if vis(0):
            nose_y = gy(0)
            height = max(1e-6, float(bbox[3] - bbox[1]))
            # 归一化到身高：鼻明显低于肩部中心
            drop = (nose_y - sy) / (height / 480.0) if height > 0 else 0.0
            if drop >= self.sleep_head_drop:
                return True, f"head_drop={drop:.3f}"

        return False, f"upright(tilt={tilt:.2f})"

    # ------------------------------------------------------------------
    def clear_track(self, track_id):
        self.person_state.pop(track_id, None)

    def status(self):
        return {
            "enabled": self.enabled,
            "off_duty_seconds": self.off_duty_seconds,
            "sleep_still_seconds": self.sleep_still_seconds,
            "zones_tracked": len(self.zone_last_seen),
            "stats": dict(self.stats),
            "suppressed": dict(self.stats_suppressed),
        }
