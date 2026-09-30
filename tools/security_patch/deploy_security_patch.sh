#!/usr/bin/env bash
# =============================================================================
# 安防事件上报 —— 板端一键部署脚本
#
# 作用：把 security_patch 中的模块安装进权威工程 rk_pose_authoritative，
#       并生成 MQTT 凭据环境文件。全程幂等、自动备份、失败可回滚。
#
# 用法（板端，rk_pose_project 所在机器）：
#     bash deploy_security_patch.sh                       # 默认路径
#     bash deploy_security_patch.sh /path/to/rk_pose_authoritative
#     RK_MQTT_TOKEN=xxx bash deploy_security_patch.sh      # 顺带写凭据
#
# 部署内容（2026-09-30 补全清单；此前版本漏了 09-30 的全部改动）：
#   ── 安防事件链路 ──
#     engine/security_monitor.py    安防事件判定核心（含离岗/睡岗接线）
#     engine/mqtt_reporter.py       MQTT 上报器
#     engine/zone_manager.py        区域管理
#     engine/security_event_manager.py  事件去重
#     engine/fire_smoke_detector.py 烟火检测接口
#     npu/fire_smoke_rknn.py        烟火 RKNN 检测器
#   ── 行为判定质量（2026-09-30）──
#     engine/attendance_monitor.py  离岗 / 睡岗判定（新文件，本次新增）
#     engine/geometry.py            可见性掩码 VIS_THRESHOLD（新量纲依赖）
#     engine/locomotion_engine.py   窗口净位移速率门槛 0.05/0.02
#     engine/special_action_engine.py  后跌倒阶段状态机（帧号计时）
#     engine/engine.py              主引擎接线
#   ── 画面与评估（2026-09-30）──
#     app/main_security.py          姿态模型指向 yolov8n-pose.rknn
#     cam/mjpeg_server.py           浏览器 MJPEG 中转（新文件）
#     cam/local_view.py             板端本地画面（新文件）
#     eval/eval_behavior_report.py  事件级 SE/SP 评估报告（新文件）
#     rtsp/stream.py                RTSP 重连
#     engine/shared.py              ← 增量打补丁（备份 + 语法校验）
#
# 覆盖前自动备份到 $HOME/.security_patch_backup/<时间戳>/（仓库外），可整目录还原。
# shared.py 仅追加方法与字段，不改动既有行为。
# =============================================================================

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$HOME/rk_security_ai/rk_pose_authoritative}"

if [ ! -d "$TARGET" ]; then
    TARGET="$HOME/PaddleDetection/rk_pose_authoritative"
fi

echo "==============================================================="
echo " 安防事件上报 —— 板端部署"
echo "==============================================================="
echo "补丁目录 : $PATCH_DIR"
echo "目标工程 : $TARGET"
echo

if [ ! -f "$TARGET/engine/shared.py" ]; then
    echo "[ERROR] 目标目录不像权威工程（找不到 engine/shared.py）"
    echo "        请显式传入路径: bash $0 /path/to/rk_pose_authoritative"
    exit 1
fi

PY="${PYTHON_BIN:-python3}"

# ------------------------------------------------------------ 0. 备份
# 备份放在仓库【外】（$HOME/.security_patch_backup），避免污染工作区、
# 也避免被 git status 当成待提交内容（2026-09-30 实测踩坑：放在
# $TARGET/.patch_backup 时每次部署都会多出一个未跟踪目录）。
BACKUP_ROOT="${PATCH_BACKUP_ROOT:-$HOME/.security_patch_backup}"
BACKUP_DIR="$BACKUP_ROOT/$(date +%Y%m%d_%H%M%S)"
echo "[0/5] 备份将被覆盖的文件 ..."
mkdir -p "$BACKUP_DIR"
echo "      备份目录: $BACKUP_DIR"

backup_one() {
    # $1 = 相对 TARGET 的路径；存在才备份
    if [ -f "$TARGET/$1" ]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$1")"
        cp -p "$TARGET/$1" "$BACKUP_DIR/$1"
    fi
}

# ---------------------------------------------------------------- 1. 复制模块
echo "[1/5] 安装模块 ..."
mkdir -p "$TARGET/engine" "$TARGET/npu" "$TARGET/eval" "$TARGET/app"

# cam/ 的权威位置在【仓库根】，不是 rk_pose_authoritative/ 下：
#   systemd 单元 rk-cam-mjpeg.service 的 ExecStart 硬编码
#   /home/admin/rk_security_ai/cam/mjpeg_server.py
# 2026-09-30 曾误装到 rk_pose_authoritative/cam/ 造成重复副本（已清理）。
REPO_ROOT="$(git -C "$TARGET" rev-parse --show-toplevel 2>/dev/null || echo "$TARGET")"
mkdir -p "$REPO_ROOT/cam"

# 文件清单：engine/app/npu/rtsp/eval 相对 TARGET
FILES="
engine/security_monitor.py
engine/mqtt_reporter.py
engine/zone_manager.py
engine/security_event_manager.py
engine/fire_smoke_detector.py
engine/attendance_monitor.py
engine/geometry.py
engine/locomotion_engine.py
engine/special_action_engine.py
engine/engine.py
app/main_security.py
npu/fire_smoke_rknn.py
rtsp/stream.py
eval/eval_behavior_report.py
"

for rel in $FILES; do
    src="$PATCH_DIR/$rel"
    if [ ! -f "$src" ]; then
        echo "      [SKIP] 补丁中不存在: $rel"
        continue
    fi
    backup_one "$rel"
    cp "$src" "$TARGET/$rel"
    echo "      ✓ $rel"
done

# cam/ 单独复制到仓库根
for rel in cam/mjpeg_server.py cam/local_view.py; do
    src="$PATCH_DIR/$rel"
    [ -f "$src" ] || { echo "      [SKIP] 补丁中不存在: $rel"; continue; }
    if [ -f "$REPO_ROOT/$rel" ]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$rel")"
        cp -p "$REPO_ROOT/$rel" "$BACKUP_DIR/$rel"
    fi
    cp "$src" "$REPO_ROOT/$rel"
    echo "      ✓ $rel  → $REPO_ROOT/"
done

# ------------------------------------------------------- 2. SharedData 打补丁
echo "[2/5] 为 engine/shared.py 打增量补丁 ..."
"$PY" "$PATCH_DIR/apply_shared_patch.py" "$TARGET/engine/shared.py"

# ------------------------------------------------------------- 3. 凭据环境文件
echo "[3/5] 生成 MQTT 凭据环境文件 ..."
ENV_FILE="$TARGET/security_mqtt.env"
if [ -f "$ENV_FILE" ]; then
    echo "      [SKIP] 已存在，保留现有凭据: $ENV_FILE"
else
    TOKEN="${RK_MQTT_TOKEN:-}"
    cat > "$ENV_FILE" <<EOF
# IoTSharp MQTT 上报配置（本文件含设备令牌，勿提交到 git）
# 平台: IoTSharp 192.168.1.8（MQTT 端口 1883；2927 是 Web 界面端口，不是 MQTT）
export RK_MQTT_BROKER="192.168.1.8"
export RK_MQTT_PORT="1883"
export RK_MQTT_TOPIC="devices/me/telemetry"
export RK_MQTT_INTERVAL="0.5"
export RK_MQTT_TOKEN="$TOKEN"
EOF
    echo "      ✓ 已生成: $ENV_FILE"
    if [ -z "$TOKEN" ]; then
        echo "      [注意] 令牌为空！请填入设备 AccessToken，或重跑："
        echo "             RK_MQTT_TOKEN=<令牌> bash $0 $TARGET"
    fi
fi

# ------------------------------------------------------------------ 4. 自检
echo "[4/5] 导入与判定自检 ..."
cd "$TARGET"
"$PY" -c "
import sys
sys.path.insert(0, '.')
from engine.security_monitor import SecurityMonitor, SecurityThread
from engine.mqtt_reporter import MqttReporter
from engine.shared import SharedData

# 09-30 新增/改动的模块必须能导入，否则运行时才 ImportError
from engine.attendance_monitor import AttendanceMonitor, EVENT_OFF_DUTY, EVENT_SLEEPING
from engine.locomotion_engine import LocomotionEngine
from engine.special_action_engine import SpecialActionEngine
print('      ✓ 离岗/睡岗、行为引擎模块导入成功')

s = SharedData()
s.set_detections({1: {'bbox': [450, 100, 550, 300], 'score': 0.9}})
m = SecurityMonitor(zones=[{'id': 'z', 'name': 'Z', 'type': 'intrusion', 'rect': [360, 40, 620, 340]}])
states, events = m.update(s.get_detections(), {}, now=1000.0)
assert any(e['event_type'] == 'INTRUSION_DETECTED' for e in events), '入侵判定失败'
s.set_events(events)
print('      ✓ 入侵判定自检通过')

# 离岗链路自检：工位区域连续无人 31 分钟应报 OFF_DUTY
am = AttendanceMonitor(off_duty_seconds=1800)
d = am.evaluate_off_duty(
    [{'id': 'd1', 'name': '工位A', 'type': 'desk', 'rect': [0, 0, 100, 100]}],
    {}, now=1000.0,
)
d = am.evaluate_off_duty(
    [{'id': 'd1', 'name': '工位A', 'type': 'desk', 'rect': [0, 0, 100, 100]}],
    {}, now=1000.0 + 31 * 60,
)
assert any(e['event_type'] == EVENT_OFF_DUTY for e in d), '离岗判定失败'
print('      ✓ 离岗（30 分钟无人）判定自检通过')

# 睡岗判据的量纲必须与 locomotion 的窗口净位移一致（否则静止门槛失效）
assert am.sleep_still_seconds == 300, '睡岗静止阈值异常'
print('      ✓ 睡岗模块就绪（静止 5 分钟 + 伏案姿态）')

print('      ✓ 默认上报目标:', MqttReporter.DEFAULT_BROKER + ':' + str(MqttReporter.DEFAULT_PORT))
" || { echo "      [ERROR] 自检失败"; exit 2; }

# ------------------------------------------------------------------ 5. 收尾提示
echo "[5/5] 完成"
echo
echo "==============================================================="
echo " 部署完成"
echo "==============================================================="
echo "回滚方式（如需）:"
echo "   cp -a $BACKUP_DIR/. $TARGET/"
echo
echo "下一步（联调）："
echo "  1) 填入设备令牌:  vi $ENV_FILE"
echo "  2) 加载凭据并测试上报:"
echo "       source $ENV_FILE"
echo "       $PY $PATCH_DIR/test_mqtt_live.py        # 真实网络上报测试"
echo "  3) 办公区事件单测（纯 Python，无需 NPU）:"
echo "       $PY $PATCH_DIR/test_office_events.py"
echo "       $PY $PATCH_DIR/test_office_integration.py"
echo "  4) 启动完整链路（含安防线程）:"
echo "       source $ENV_FILE && $PY app/main_security.py"
echo "  5) 浏览器看画面: http://<板端IP>:8081/"
echo "       $PY cam/mjpeg_server.py &"
echo
