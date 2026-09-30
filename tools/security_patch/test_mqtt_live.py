# -*- coding: utf-8 -*-
"""
真实网络 MQTT 上报联调测试（在板端运行，需能访问平台）

作用：不启动摄像头与模型，直接构造一条安防事件并上报到 IoTSharp，
      用于验证「板端 → 平台」链路是否打通。这是端到端联调的第一步。

先决条件：
    /etc/hosts 或网络能解析 192.168.1.8
    环境变量 RK_MQTT_TOKEN 已设置（设备 AccessToken）
       source security_mqtt.env

用法：
    python3 test_mqtt_live.py
    python3 test_mqtt_live.py --object      # 额外发一条普通目标（无告警）
    RK_MQTT_BROKER=192.168.1.8 RK_MQTT_PORT=1883 python3 test_mqtt_live.py

预期结果：
    终端打印「发送成功」，随后可在平台侧确认：
      docker exec pgsql psql -U postgres -d IoTSharp -c \
        "SELECT \"KeyName\",\"Value_String\",\"DateTime\" FROM \"TelemetryData\" \
         WHERE \"DeviceId\"='<你的设备ID>' ORDER BY \"DateTime\" DESC LIMIT 5;"
"""

import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from engine.mqtt_reporter import MqttReporter


def main():
    ap = argparse.ArgumentParser(description="MQTT 真实上报联调测试")
    ap.add_argument("--object", action="store_true", help="额外发送一条普通目标数据")
    ap.add_argument("--repeat", type=int, default=1, help="重复发送次数")
    ap.add_argument("--interval", type=float, default=1.0, help="重复发送间隔秒")
    args = ap.parse_args()

    print("=" * 64)
    print(" MQTT 上报联调测试（板端 → IoTSharp）")
    print("=" * 64)

    reporter = MqttReporter.from_env()

    print(f"broker : {reporter.broker}:{reporter.port}")
    print(f"topic  : {reporter.topic}")
    print(f"token  : {'已设置(' + str(len(reporter.token)) + '位)' if reporter.token else '【未设置】'}")

    if not reporter.enabled:
        print("\n[ERROR] 未设置 RK_MQTT_TOKEN，无法测试。")
        print("        请先: source security_mqtt.env  或 export RK_MQTT_TOKEN=<设备令牌>")
        return 2

    print("\n[1] 建立连接 ...")
    if not reporter.start():
        print("[FAIL] 连接失败：请检查网络可达性与令牌有效性")
        print("       排查: ping 192.168.1.8 / nc -z 192.168.1.8 1883")
        return 3
    time.sleep(1.0)

    print("\n[2] 发送测试安防事件 ...")
    now = time.time()
    alerts = [
        {
            "person_id": 99,
            "event_type": "INTRUSION_DETECTED",
            "zone_name": "联调测试区",
            "bbox": [450, 100, 550, 300],
            "ts": now,
        },
        {
            "person_id": 99,
            "event_type": "FALL_DETECTED",
            "bbox": [450, 100, 550, 300],
            "score": 0.91,
            "ts": now + 0.01,
        },
        {
            "person_id": 99,
            "event_type": "LOITERING_DETECTED",
            "zone_name": "联调测试区",
            "duration": 12.5,
            "bbox": [450, 100, 550, 300],
            "ts": now + 0.02,
        },
    ]
    objects = {
        99: {
            "name": "person",
            "name_zh": "人员",
            "score": 0.93,
            "bbox": [450, 100, 550, 300],
            "behavior": "fall_down",
            "behavior_score": 0.91,
        }
    }

    if args.object:
        objects[100] = {
            "name": "person",
            "name_zh": "人员",
            "score": 0.88,
            "bbox": [700, 120, 800, 320],
            "behavior": "walking",
            "behavior_score": 0.85,
        }

    sent = 0
    for i in range(max(1, args.repeat)):
        ok = reporter.publish_objects(objects, alerts, timestamp=now + i * args.interval)
        if ok:
            sent += 1
            payload = {
                "timestamp": now + i * args.interval,
                "objects": list(objects.values()),
                "count": len(objects),
                "alerts": alerts,
                "alert_count": len(alerts),
            }
            print(f"  [{i + 1}] 发送成功 | count={payload['count']} alert_count={payload['alert_count']}")
            print(f"      payload 摘要: {json.dumps(payload, ensure_ascii=False)[:150]}...")
        else:
            print(f"  [{i + 1}] 发送失败")

        if i < args.repeat - 1:
            time.sleep(args.interval)

    print("\n[3] 结果 ...")
    stats = reporter.stats()
    print(f"  成功: {stats['sent']}  失败: {stats['failed']}  限流跳过: {stats['throttled']}")

    reporter.stop()

    if sent > 0:
        print("\n" + "=" * 64)
        print(" 链路已打通 ✅")
        print("=" * 64)
        return 0

    print("\n[FAIL] 未成功发送任何消息")
    return 4


if __name__ == "__main__":
    sys.exit(main())
