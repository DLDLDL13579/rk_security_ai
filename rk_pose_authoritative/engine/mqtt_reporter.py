# -*- coding: utf-8 -*-
"""
安防事件 MQTT 上报模块 —— 移植自历史工程 v4.0/main_q.py（前人已跑通的实现）

移植说明：
    来源：rk_pose_project v4.0/main_q.py 的 publish_detections() 与 MQTT 客户端初始化段
    该实现历史上真实上报成功过（平台库中存在对应 objects/count/timestamp 记录，
    最后一次 2026-07-21）。

与来源的差异（本次移植的改进）：
    ① 抽成独立类，不再污染主程序全局变量
    ② 线程安全（内部锁），可从多线程调用
    ③ 断线自动重连（loop_start 的自动重连 + 发送失败静默降级，不阻塞主流程）
    ④ 端口默认 1883（修正板端 mqtt.py 误用 2927 —— 那是 Web 端口，非 MQTT）
    ⑤ 凭据从环境变量读取，不进 git

平台事实（2026-09-29 实测确认）：
    平台    : IoTSharp 192.168.1.8（Docker Compose v3.6.2）
    MQTT    : 1883（0.0.0.0 映射，实测 OPEN）
    Web     : 2927（管理界面）
    话题    : devices/me/telemetry      ← IoTSharp 直连设备格式
    认证    : 用户名 = 设备 AccessToken，密码留空
    上报格式: {"timestamp", "objects": [...], "count", "alerts": [...], "alert_count"}

用法：
    reporter = MqttReporter.from_env()
    reporter.start()
    reporter.publish_objects(objects, alerts)
    reporter.stop()
"""

import json
import os
import threading
import time

try:
    import paho.mqtt.client as mqtt
except ImportError:  # 允许在无 paho 的环境导入（离线单测）
    mqtt = None


class MqttReporter:
    """安防事件 MQTT 上报器（IoTSharp 直连设备协议）"""

    DEFAULT_BROKER = "192.168.1.8"
    DEFAULT_PORT = 1883
    DEFAULT_TOPIC = "devices/me/telemetry"
    DEFAULT_INTERVAL = 0.5   # 秒，最小发送间隔（与来源实现一致）

    # COCO 常见类别中文名（权威工程 object_engine.CLASS_NAMES_ZH 的常用子集）。
    # 上游 ai_thread 只给出英文 name，此处兜底补中文，避免平台上显示「未知」。
    CLASS_ZH_FALLBACK = {
        "person": "人员",
        "bicycle": "自行车",
        "car": "汽车",
        "motorcycle": "摩托车",
        "airplane": "飞机",
        "bus": "公交车",
        "train": "火车",
        "truck": "卡车",
        "boat": "船",
        "fire": "火焰",
        "smoke": "烟雾",
    }

    def __init__(
        self,
        token,
        broker=None,
        port=None,
        topic=None,
        send_interval=None,
        client_id=None,
        enabled=True,
    ):
        self.token = (token or "").strip()
        self.broker = broker or self.DEFAULT_BROKER
        self.port = int(port or self.DEFAULT_PORT)
        self.topic = topic or self.DEFAULT_TOPIC
        self.send_interval = float(
            self.DEFAULT_INTERVAL if send_interval is None else send_interval
        )
        self.enabled = bool(enabled) and bool(self.token)

        self._client = None
        self._lock = threading.Lock()
        self._last_send = 0.0
        self._connected = False
        self._sent_count = 0
        self._fail_count = 0
        self._skip_count = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, prefix="RK_MQTT_"):
        """
        从环境变量构造（凭据不进 git）：

            RK_MQTT_TOKEN     设备 AccessToken（必填，缺省则上报自动禁用）
            RK_MQTT_BROKER    broker 地址，默认 192.168.1.8
            RK_MQTT_PORT      端口，默认 1883
            RK_MQTT_TOPIC     话题，默认 devices/me/telemetry
            RK_MQTT_INTERVAL  最小发送间隔秒，默认 0.5
            RK_MQTT_ENABLED   "0"/"false" 可强制关闭
        """
        token = os.environ.get(f"{prefix}TOKEN", "").strip()
        enabled_raw = os.environ.get(f"{prefix}ENABLED", "1").strip().lower()
        enabled = enabled_raw not in ("0", "false", "no", "off")

        return cls(
            token=token,
            broker=os.environ.get(f"{prefix}BROKER", cls.DEFAULT_BROKER),
            port=os.environ.get(f"{prefix}PORT", cls.DEFAULT_PORT),
            topic=os.environ.get(f"{prefix}TOPIC", cls.DEFAULT_TOPIC),
            send_interval=os.environ.get(f"{prefix}INTERVAL", cls.DEFAULT_INTERVAL),
            enabled=enabled,
        )

    # ------------------------------------------------------------------
    def start(self):
        """建立连接（失败不抛异常，只打印并保持禁用状态）"""
        if not self.enabled:
            print("[MQTT] 上报未启用（缺少 RK_MQTT_TOKEN 或被显式关闭）")
            return False
        if mqtt is None:
            print("[MQTT] paho-mqtt 不可用，上报禁用")
            self.enabled = False
            return False

        try:
            # paho-mqtt 2.x 需要显式指定 CallbackAPIVersion
            try:
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            except AttributeError:
                client = mqtt.Client()  # paho 1.x 回退

            client.username_pw_set(self.token, "")
            client.connect(self.broker, self.port, keepalive=60)
            client.loop_start()
            self._client = client
            self._connected = True
            print(f"[MQTT] 已连接 {self.broker}:{self.port}，话题 {self.topic}")
            return True
        except Exception as exc:
            print(f"[MQTT] 连接失败（不影响识别主流程）: {exc}")
            self._client = None
            self._connected = False
            return False

    # ------------------------------------------------------------------
    @classmethod
    def _normalize_object(cls, track_id, obj):
        """把内部跟踪结果统一成平台约定字段（保持来源实现的字段名）"""
        if not isinstance(obj, dict):
            return None

        en_name = obj.get("name", obj.get("class", "unknown"))
        zh_name = obj.get("name_zh", obj.get("class_zh", ""))
        if not zh_name:
            # 上游只给英文名时补中文兜底（否则平台显示「未知」）
            zh_name = cls.CLASS_ZH_FALLBACK.get(str(en_name).strip().lower(), "未知")

        return {
            "track_id": track_id,
            "class": en_name,
            "class_zh": zh_name,
            "score": float(obj.get("score", 0.0)),
            "bbox": list(obj.get("bbox", [])),
            "behavior": obj.get("behavior", ""),
            "behavior_score": float(obj.get("behavior_score", 0.0)),
        }

    def _normalize_alerts(self, alerts):
        out = []
        for alert in alerts or []:
            if not isinstance(alert, dict):
                continue
            out.append(
                {
                    "track_id": alert.get("track_id", alert.get("person_id", -1)),
                    "alert_type": alert.get(
                        "alert_type", alert.get("event_type", "unknown")
                    ),
                    "bbox": list(alert.get("bbox", [])),
                    "timestamp": alert.get("timestamp", alert.get("ts", time.time())),
                }
            )
        return out

    # ------------------------------------------------------------------
    def publish(self, payload, force=False):
        """发送已构造好的 payload（带最小间隔限流）"""
        if not self.enabled or self._client is None:
            return False

        now = time.time()
        with self._lock:
            if not force and (now - self._last_send) < self.send_interval:
                self._skip_count += 1
                return False
            self._last_send = now

        try:
            self._client.publish(self.topic, json.dumps(payload))
            self._sent_count += 1
            return True
        except Exception as exc:
            self._fail_count += 1
            print(f"[MQTT] 发送失败: {exc}")
            return False

    def publish_objects(self, objects, alerts=None, timestamp=None):
        """
        上报目标与告警（平台约定格式）

        objects: {track_id: {...}} 或 [{...}, ...]
        alerts : 安防事件列表（跌倒/入侵/逗留/烟火）
        """
        if not self.enabled:
            return False

        # 统一 objects 为列表
        normalized = []
        if isinstance(objects, dict):
            for tid, obj in objects.items():
                if tid == "alerts":   # 兼容来源实现里的 alerts 键
                    continue
                item = self._normalize_object(tid, obj)
                if item is not None:
                    normalized.append(item)
        elif isinstance(objects, (list, tuple)):
            for idx, obj in enumerate(objects):
                tid = obj.get("track_id", idx) if isinstance(obj, dict) else idx
                item = self._normalize_object(tid, obj)
                if item is not None:
                    normalized.append(item)

        if isinstance(objects, dict) and not alerts:
            alerts = objects.get("alerts", [])

        alert_list = self._normalize_alerts(alerts)

        payload = {
            "timestamp": float(timestamp if timestamp is not None else time.time()),
            "objects": normalized,
            "count": len(normalized),
            "alerts": alert_list,
            "alert_count": len(alert_list),
        }
        return self.publish(payload)

    # ------------------------------------------------------------------
    def stats(self):
        return {
            "enabled": self.enabled,
            "connected": self._connected,
            "sent": self._sent_count,
            "failed": self._fail_count,
            "throttled": self._skip_count,
            "broker": f"{self.broker}:{self.port}",
            "topic": self.topic,
        }

    def stop(self):
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
        self._connected = False
