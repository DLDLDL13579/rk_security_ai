# -*- coding: utf-8 -*-
"""
MQTT 上报模块离线单测（不依赖板端、不依赖网络、不依赖 paho）

用假 client 验证上报数据结构与限流逻辑，确保与平台约定字段完全一致
（字段名与历史成功上报的记录对齐：timestamp/objects/count/alerts/alert_count）。

用法：
    /usr/local/bin/python3 test_mqtt_reporter.py
"""

import importlib.util
import json
import os
import sys
import time

MODULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "engine", "mqtt_reporter.py"
)
spec = importlib.util.spec_from_file_location("mqtt_reporter", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MqttReporter = mod.MqttReporter


class FakeClient:
    """记录 publish 调用的假客户端"""

    def __init__(self):
        self.published = []

    def publish(self, topic, payload):
        self.published.append((topic, payload))

    def loop_stop(self):
        pass

    def disconnect(self):
        pass


def attach_fake(reporter):
    """把假 client 注入 reporter（绕过真实网络连接）"""
    fake = FakeClient()
    reporter._client = fake
    reporter._connected = True
    return fake


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    all_pass = True

    print("=" * 64)
    print(" MQTT 上报模块离线单测")
    print("=" * 64)

    # ---- 用例 1：未配置 token 时自动禁用，不抛异常 ----
    r = MqttReporter(token="", enabled=True)
    all_pass &= check("无 token 时自动禁用", r.enabled is False)
    all_pass &= check("禁用时 publish 安全返回 False", r.publish({"a": 1}) is False)
    all_pass &= check(
        "禁用时 publish_objects 安全返回 False", r.publish_objects([{"score": 1.0}]) is False
    )

    # ---- 用例 2：payload 结构符合平台约定 ----
    r = MqttReporter(token="dummy_token", send_interval=0.0)
    fake = attach_fake(r)
    objects = {
        3: {
            "name": "person",
            "name_zh": "人员",
            "score": 0.88,
            "bbox": [10, 20, 110, 220],
            "behavior": "fall_down",
            "behavior_score": 0.77,
        }
    }
    alerts = [
        {
            "person_id": 3,
            "event_type": "FALL_DETECTED",
            "bbox": [10, 20, 110, 220],
            "ts": 1770000000.0,
        }
    ]
    ok = r.publish_objects(objects, alerts, timestamp=1770000000.0)
    all_pass &= check("publish_objects 返回 True", ok is True)
    all_pass &= check("确实调用了 publish", len(fake.published) == 1, f"次数={len(fake.published)}")

    if fake.published:
        topic, raw = fake.published[0]
        all_pass &= check("话题为 devices/me/telemetry", topic == "devices/me/telemetry", topic)
        payload = json.loads(raw)
        expect_keys = {"timestamp", "objects", "count", "alerts", "alert_count"}
        all_pass &= check(
            "payload 顶层字段与平台约定一致",
            set(payload.keys()) == expect_keys,
            f"实际={sorted(payload.keys())}",
        )
        all_pass &= check("count 正确", payload["count"] == 1, f"实际={payload['count']}")
        all_pass &= check(
            "alert_count 正确", payload["alert_count"] == 1, f"实际={payload['alert_count']}"
        )
        obj = payload["objects"][0]
        expect_obj_keys = {
            "track_id",
            "class",
            "class_zh",
            "score",
            "bbox",
            "behavior",
            "behavior_score",
        }
        all_pass &= check(
            "object 字段与历史成功上报一致",
            set(obj.keys()) == expect_obj_keys,
            f"实际={sorted(obj.keys())}",
        )
        all_pass &= check("track_id 保留", obj["track_id"] == 3, f"实际={obj['track_id']}")
        all_pass &= check("class_zh 中文透传", obj["class_zh"] == "人员", obj["class_zh"])
        alert = payload["alerts"][0]
        all_pass &= check(
            "alert 字段映射正确（person_id→track_id, event_type→alert_type）",
            alert["track_id"] == 3 and alert["alert_type"] == "FALL_DETECTED",
            f"实际={alert}",
        )

    # ---- 用例 3：限流生效 ----
    r = MqttReporter(token="dummy", send_interval=10.0)
    fake = attach_fake(r)
    first = r.publish({"n": 1})
    second = r.publish({"n": 2})
    all_pass &= check("限流：首次发送成功", first is True)
    all_pass &= check("限流：间隔内第二次被跳过", second is False)
    all_pass &= check("限流计数正确", r.stats()["throttled"] == 1, f"实际={r.stats()['throttled']}")
    forced = r.publish({"n": 3}, force=True)
    all_pass &= check("force=True 可绕过限流", forced is True)

    # ---- 用例 4：列表输入与字典输入等价 ----
    r = MqttReporter(token="dummy", send_interval=0.0)
    fake = attach_fake(r)
    r.publish_objects([{"track_id": 7, "name": "person", "score": 0.5}])
    payload = json.loads(fake.published[0][1])
    all_pass &= check(
        "列表输入也能正确归一化",
        payload["count"] == 1 and payload["objects"][0]["track_id"] == 7,
        f"实际={payload['objects'][0]['track_id']}",
    )

    # ---- 用例 5：空输入安全 ----
    r = MqttReporter(token="dummy", send_interval=0.0)
    fake = attach_fake(r)
    r.publish_objects({})
    payload = json.loads(fake.published[0][1])
    all_pass &= check(
        "空输入产生合法空 payload",
        payload["count"] == 0 and payload["objects"] == [] and payload["alerts"] == [],
        f"实际={payload}",
    )

    # ---- 用例 6：环境变量构造 ----
    os.environ["RK_MQTT_TOKEN"] = "env_token"
    os.environ["RK_MQTT_PORT"] = "1883"
    r = MqttReporter.from_env()
    all_pass &= check(
        "from_env 正确读取 token 与端口",
        r.token == "env_token" and r.port == 1883 and r.enabled is True,
        f"token={r.token} port={r.port}",
    )
    del os.environ["RK_MQTT_TOKEN"]
    del os.environ["RK_MQTT_PORT"]
    r = MqttReporter.from_env()
    all_pass &= check("from_env 无 token 时禁用", r.enabled is False)

    # ---- 用例 7：默认端口是 1883 而非 2927（修正板端 mqtt.py 的错误） ----
    all_pass &= check("默认端口为 1883（MQTT）", MqttReporter.DEFAULT_PORT == 1883)
    all_pass &= check("默认话题正确", MqttReporter.DEFAULT_TOPIC == "devices/me/telemetry")

    # ---- 用例 8：stats 可观测 ----
    r = MqttReporter(token="dummy", send_interval=0.0)
    attach_fake(r)
    r.publish_objects([{"track_id": 1, "name": "person"}])
    s = r.stats()
    all_pass &= check(
        "stats 统计发送数",
        s["sent"] == 1 and s["connected"] is True,
        f"实际={s}",
    )

    print("=" * 64)
    print(f" 结果：{'全部通过 ✅' if all_pass else '存在失败 ❌'}")
    print("=" * 64)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
