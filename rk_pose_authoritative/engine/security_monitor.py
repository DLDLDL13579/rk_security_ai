# -*- coding: utf-8 -*-
"""
安防事件监控核心（三类事件 + 烟火 + 上报）

移植来源：历史工程 rk_pose_project _8.4/engine/security_thread.py
移植改动：
    ① 把线程内的判定逻辑抽成本模块的纯函数核心 SecurityMonitor.update()，
       使其**可离线单测**（不依赖板端、不依赖线程、不依赖网络）
    ② 保持原有判定语义完全不变（区域入侵 / 长时间逗留 / 跌倒 / 烟火 +
       事件去重冷却），确保精度与历史行为一致
    ③ 通过 MqttReporter 向 IoTSharp 平台上报（前人在 v4.0 中已验证过链路）

三类安防事件（用户需求口径）：
    ① 跌倒      FALL_DETECTED      行为模型 fall_down 且置信度 ≥ fall_confidence
    ② 区域入侵  INTRUSION_DETECTED 人体锚点（框底中心）进入 intrusion 区域
    ③ 长时间逗留 LOITERING_DETECTED 在 loiter 区域停留 ≥ loiter_seconds
    （附带：烟火 FIRE_DETECTED / SMOKE_DETECTED，模型已接入见 fire_smoke_rknn.py）
"""

import threading
import time

from engine.fire_smoke_detector import NullFireSmokeDetector
from engine.security_event_manager import SecurityEventManager
from engine.zone_manager import ZoneManager

# 事件类型常量（与平台 alert_type 约定一致）
EVENT_FALL = "FALL_DETECTED"
EVENT_INTRUSION = "INTRUSION_DETECTED"
EVENT_LOITERING = "LOITERING_DETECTED"
EVENT_FIRE = "FIRE_DETECTED"
EVENT_SMOKE = "SMOKE_DETECTED"

# 三类安防事件 → 上报用的告警类型（与平台历史数据字段对齐）
SECURITY_EVENT_TYPES = (EVENT_FALL, EVENT_INTRUSION, EVENT_LOITERING)


class SecurityMonitor:
    """安防事件判定核心（无线程、无 I/O，可离线单测）"""

    def __init__(
        self,
        zones=None,
        fall_confidence=0.55,
        zone_cooldown=2.0,
        fall_cooldown=2.0,
        fire_smoke_cooldown=1.5,
        event_hold_seconds=4.0,
        max_recent_events=6,
        fall_confirm_frames=3,
        fire_smoke_confirm_frames=3,
        confirm_reset_gap=1.5,
    ):
        self.fall_confidence = float(fall_confidence)
        self.zone_cooldown = float(zone_cooldown)
        self.fall_cooldown = float(fall_cooldown)
        self.fire_smoke_cooldown = float(fire_smoke_cooldown)
        self.event_hold_seconds = float(event_hold_seconds)
        self.max_recent_events = int(max_recent_events)

        # === 误报抑制（方案 A，2026-09-30）===
        # 实测问题：60 秒内误报 4 次（跌倒 1、烟雾 2、火焰 1）。
        # 根因：单帧抖动即触发（日志见 walking 0.96→1.00 之后紧跟一次 fall_down 0.88）。
        # 策略：要求「连续 N 帧」命中才确认，且帧间间隔超过 confirm_reset_gap 则清零计数。
        # 真实事件通常持续多帧（跌倒至少数秒），瞬时抖动无法通过连续帧确认。
        self.fall_confirm_frames = max(1, int(fall_confirm_frames))
        self.fire_smoke_confirm_frames = max(1, int(fire_smoke_confirm_frames))
        self.confirm_reset_gap = float(confirm_reset_gap)

        # track_id -> 连续命中状态
        self._fall_streak = {}          # {"count": int, "last_ts": float, "max_score": float}
        self._fire_smoke_streak = {}    # (label, key) -> {"count","last_ts","max_score","item"}
        self.stats_suppressed = {       # 被连续性过滤挡掉的次数（可观测性）
            EVENT_FALL: 0,
            EVENT_FIRE: 0,
            EVENT_SMOKE: 0,
        }

        self.zone_manager = ZoneManager(zones or [])
        self.event_manager = SecurityEventManager(default_cooldown=3.0)

        self.recent_events = []
        self.last_event_signature = None
        self.stats = {
            EVENT_FALL: 0,
            EVENT_INTRUSION: 0,
            EVENT_LOITERING: 0,
            EVENT_FIRE: 0,
            EVENT_SMOKE: 0,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_fire_smoke(objects):
        """统一烟火检测输出（与 _8.4 SecurityThread._normalize_fire_smoke 一致）"""
        normalized = []
        for idx, item in enumerate(objects or []):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox", [])
            if len(bbox) != 4:
                continue
            label = str(item.get("label", item.get("name", ""))).strip().lower()
            if label not in ("fire", "smoke", "flame"):
                continue
            normalized.append(
                {
                    "id": str(item.get("id", f"fs_{idx}")),
                    "label": "fire" if label == "flame" else label,
                    "bbox": [int(v) for v in bbox],
                    "score": float(item.get("score", 0.0)),
                }
            )
        return normalized

    # ------------------------------------------------------------------
    def update(self, detections, behaviors, fire_smoke_objects=None, now=None):
        """
        单帧更新：返回 (security_states, events)

        detections : {track_id: {"bbox": [...], "score": ...}}（权威工程格式）
        behaviors  : {track_id: {"action", "confidence", "bbox", ...}}
        fire_smoke : [{"bbox", "score", "label"}, ...]（可空）
        """
        now = time.time() if now is None else float(now)
        detections = detections or {}
        behaviors = behaviors or {}

        security_states, zone_events = self.zone_manager.evaluate(detections, now=now)
        fire_smoke_objects = self.normalize_fire_smoke(fire_smoke_objects)
        event_output = []

        # ---- 区域入侵 / 长时间逗留 ----
        for item in zone_events:
            event = self.event_manager.emit(item, now=now, cooldown=self.zone_cooldown)
            self._append_event(event_output, event)

        # ---- 跌倒（需连续 N 帧确认，抑制瞬时抖动误报）----
        if isinstance(behaviors, dict):
            for pid, behavior in behaviors.items():
                if not isinstance(behavior, dict):
                    continue
                action = str(behavior.get("action", behavior.get("behavior", ""))).strip()
                score = float(
                    behavior.get(
                        "confidence",
                        behavior.get("score", behavior.get("behavior_score", 0.0)),
                    )
                )

                # 未命中：不计入连续帧
                if action != "fall_down" or score < self.fall_confidence:
                    continue

                if not self._confirm_fall(pid, score, now):
                    continue

                self._merge_state(security_states, pid, "FALL")
                bbox = behavior.get("bbox")
                if bbox is None and isinstance(detections, dict):
                    bbox = (detections.get(pid) or {}).get("bbox")

                event = self.event_manager.emit(
                    {
                        "event_type": EVENT_FALL,
                        "source_key": f"fall:{pid}",
                        "person_id": pid,
                        "bbox": list(bbox or []),
                        "score": score,
                        "confirmed_frames": self.fall_confirm_frames,
                    },
                    now=now,
                    cooldown=self.fall_cooldown,
                )
                self._append_event(event_output, event)

        # ---- 烟火（同样需连续 N 帧确认）----
        for item in fire_smoke_objects:
            label = item["label"]
            event_name = EVENT_FIRE if label == "fire" else EVENT_SMOKE

            if not self._confirm_fire_smoke(label, item, now):
                continue

            event = self.event_manager.emit(
                {
                    "event_type": event_name,
                    "source_key": f"{label}:{item['id']}",
                    "label": label,
                    "bbox": list(item["bbox"]),
                    "score": float(item["score"]),
                    "confirmed_frames": self.fire_smoke_confirm_frames,
                },
                now=now,
                cooldown=self.fire_smoke_cooldown,
            )
            self._append_event(event_output, event)

        # ---- 维护近期事件窗口（供显示/上报使用） ----
        self.recent_events.extend(event_output)
        self.recent_events = [
            item
            for item in self.recent_events
            if now - float(item.get("ts", 0.0)) <= self.event_hold_seconds
        ][-self.max_recent_events:]

        for event in event_output:
            etype = event.get("event_type")
            if etype in self.stats:
                self.stats[etype] += 1

        return security_states, event_output

    # ------------------------------------------------------------------
    # 误报抑制：连续帧确认（方案 A，2026-09-30）
    # ------------------------------------------------------------------
    def _confirm_fall(self, pid, score, now):
        """
        跌倒连续帧确认：同一 track 连续 N 帧判定为 fall_down 才放行。

        返回 True 表示本次可以发事件（同时清零计数，回到冷却周期语义）。
        """
        state = self._fall_streak.get(pid)
        if state is None or (now - state["last_ts"]) > self.confirm_reset_gap:
            state = {"count": 0, "last_ts": now, "max_score": 0.0}

        state["count"] += 1
        state["last_ts"] = now
        state["max_score"] = max(state["max_score"], float(score))
        self._fall_streak[pid] = state

        if state["count"] >= self.fall_confirm_frames:
            self._fall_streak.pop(pid, None)   # 已确认，重置计数
            return True

        self.stats_suppressed[EVENT_FALL] += 1
        return False

    def _confirm_fire_smoke(self, label, item, now):
        """烟火连续帧确认：同一目标连续 N 帧命中才放行。"""
        # 位置量化成网格作为目标标识，避免逐帧 bbox 微抖动导致无法累计
        bbox = item.get("bbox") or [0, 0, 0, 0]
        gx = int((float(bbox[0]) + float(bbox[2])) / 2.0 / 64)
        gy = int((float(bbox[1]) + float(bbox[3])) / 2.0 / 64)
        key = (label, gx, gy)

        state = self._fire_smoke_streak.get(key)
        if state is None or (now - state["last_ts"]) > self.confirm_reset_gap:
            state = {"count": 0, "last_ts": now, "max_score": 0.0, "item": None}

        state["count"] += 1
        state["last_ts"] = now
        state["item"] = item
        state["max_score"] = max(state["max_score"], float(item.get("score", 0.0)))
        self._fire_smoke_streak[key] = state

        # 清理过期 streak，防止长期运行内存增长
        if len(self._fire_smoke_streak) > 64:
            stale = [
                k for k, v in self._fire_smoke_streak.items()
                if (now - v["last_ts"]) > self.confirm_reset_gap * 4
            ]
            for k in stale:
                self._fire_smoke_streak.pop(k, None)

        if state["count"] >= self.fire_smoke_confirm_frames:
            self._fire_smoke_streak.pop(key, None)
            return True

        self.stats_suppressed[
            EVENT_FIRE if label == "fire" else EVENT_SMOKE
        ] += 1
        return False

    # ------------------------------------------------------------------
    def _merge_state(self, base_state, pid, alert_name, zone_name=""):
        state = dict(base_state.get(pid, {}))
        alerts = list(state.get("alerts", []))
        if alert_name and alert_name not in alerts:
            alerts.append(alert_name)
        state["alerts"] = alerts
        zones = list(state.get("zones", []))
        if zone_name and zone_name not in zones:
            zones.append(zone_name)
        state["zones"] = zones
        base_state[pid] = state

    def _append_event(self, event_output, event):
        if event is None:
            return
        signature = (
            event.get("event_type"),
            event.get("person_id"),
            event.get("zone_id"),
            event.get("label"),
        )
        if signature != self.last_event_signature:
            print(f"[Security EVENT] {self.format_event_log(event)}")
            self.last_event_signature = signature
        event_output.append(event)

    @staticmethod
    def format_event_log(event):
        event_type = str(event.get("event_type", "EVENT"))
        pid = event.get("person_id")
        zone_name = str(event.get("zone_name", "")).strip()
        label = str(event.get("label", "")).strip()
        score = float(event.get("score", 0.0))
        duration = event.get("duration")
        parts = [event_type]
        if pid not in (None, ""):
            parts.append(f"id={pid}")
        if zone_name:
            parts.append(f"zone={zone_name}")
        if label:
            parts.append(f"label={label}")
        if duration is not None:
            parts.append(f"duration={float(duration):.1f}s")
        if score > 0:
            parts.append(f"score={score:.2f}")
        return " | ".join(parts)

    def zone_overlays(self):
        return self.zone_manager.get_zone_overlays()


class SecurityThread(threading.Thread):
    """
    安防监控线程（薄封装：循环读共享数据 → SecurityMonitor.update → 上报）

    与 _8.4 SecurityThread 的差异：判定逻辑已移入 SecurityMonitor；
    共享数据回写改为 hasattr 探测，未打补丁的 SharedData 也能正常运行。
    上报间隔由 MqttReporter 内部限流控制。
    """

    def __init__(
        self,
        shared,
        monitor=None,
        mqtt_reporter=None,
        fire_smoke_detector=None,
        interval=0.05,
        report_interval=0.5,
    ):
        super().__init__(name="SecurityThread")
        self.shared = shared
        self.interval = float(interval)
        self.report_interval = float(report_interval)
        self.monitor = monitor or SecurityMonitor()
        self.mqtt_reporter = mqtt_reporter
        self.fire_smoke_detector = fire_smoke_detector or NullFireSmokeDetector()

        self.last_frame_id = -1
        self.last_report_at = 0.0
        self.daemon = True
        print("[Security Thread] initialized")

    def _publish_state(self, security_states):
        """把最新状态写回共享数据（可选：SharedData 已打补丁时生效）"""
        for setter, value in (
            ("set_security_states", security_states),
            ("set_events", self.monitor.recent_events),
            ("set_zone_overlays", self.monitor.zone_overlays()),
        ):
            fn = getattr(self.shared, setter, None)
            if callable(fn):
                try:
                    fn(value)
                except Exception as exc:
                    print(f"[Security Thread] {setter} 失败: {exc}")

    def run(self):
        print("[Security Thread] started")

        while getattr(self.shared, "running", False):
            frame, frame_id = self.shared.get_frame_packet()

            if frame is None:
                time.sleep(0.01)
                continue
            if frame_id == self.last_frame_id:
                time.sleep(0.005)
                continue
            self.last_frame_id = frame_id

            try:
                detections = self.shared.get_detections()
                behaviors = self.shared.get_behaviors()

                fire_smoke_objects = []
                try:
                    fire_smoke_objects = self.fire_smoke_detector.detect(frame)
                except Exception as exc:
                    print(f"[Security Thread] 烟火检测异常: {exc}")

                security_states, events = self.monitor.update(
                    detections, behaviors, fire_smoke_objects
                )
                self._publish_state(security_states)

                # ---- 上报（限流由 reporter 内部处理） ----
                if self.mqtt_reporter is not None:
                    now = time.time()
                    if now - self.last_report_at >= self.report_interval:
                        self.last_report_at = now
                        self.mqtt_reporter.publish_objects(detections, events)

            except Exception as exc:
                print(f"[Security Thread] 循环异常: {exc}")

            time.sleep(self.interval)

    def stop(self):
        self.shared.running = False
