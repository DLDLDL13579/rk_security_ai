# security_patch · 安防功能补丁包

> 本目录是 AI 安防系统**功能增量的源文件**，用于把新增/修改的模块部署进板端权威工程。
> 最后更新：2026-09-30

## 它是什么

板端权威工程 `~/rk_security_ai/rk_pose_authoritative` 里的安防相关模块，
在本地以「补丁」形式维护：**这里改 → 部署到板端 → 板端提交推送**。

**权威源是板端**（`DLDLDL13579/rk_security_ai`），本目录是便于编辑与回顾的工作副本。
两者若不一致，以板端为准。

## 怎么用

```bash
# 板端执行（脚本幂等、覆盖前自动备份到仓库外）
bash deploy_security_patch.sh                      # 默认路径
bash deploy_security_patch.sh /path/to/rk_pose_authoritative
```

脚本流程：`[0/5] 备份` → `[1/5] 安装模块` → `[2/5] shared.py 打增量补丁`
→ `[3/5] 生成 MQTT 凭据` → `[4/5] 导入与判定自检` → `[5/5] 完成`

**自检会实测离岗判定**（工位连续无人 31 分钟是否报事件），不只是导入检查——
这样模块缺失或格式不匹配会在部署时就暴露，而不是等到运行时静默失效。

## 目录内容

| 路径 | 说明 |
|---|---|
| `deploy_security_patch.sh` | 一键部署脚本（含备份与自检） |
| `apply_shared_patch.py` | 给 `engine/shared.py` 追加方法与字段（幂等） |
| `engine/` | 安防事件判定、上报、区域、行为引擎 |
| `npu/fire_smoke_rknn.py` | 烟火检测 RKNN 前后处理 |
| `rtsp/stream.py` | RTSP 拉流（含指数退避重连） |
| `cam/` | 浏览器 MJPEG 中转 + 板端本地画面 |
| `eval/eval_behavior_report.py` | 事件级 SE/SP 评估报告 |
| `app/main_security.py` | 主入口 |
| `wifi_switch.sh` | 网络切换辅助 |

### 关键模块

| 文件 | 职责 |
|---|---|
| `engine/security_monitor.py` | 安防事件判定核心 + 线程，接线离岗/睡岗 |
| `engine/attendance_monitor.py` | **离岗 / 睡岗判定**（2026-09-30 新增） |
| `engine/locomotion_engine.py` | 站立/行走判定（窗口净位移速率） |
| `engine/special_action_engine.py` | 跌倒判定（后跌倒阶段状态机，帧号计时） |
| `engine/geometry.py` | 几何量计算（含关键点可见性掩码） |
| `engine/zone_manager.py` | 区域管理（intrusion / loiter / desk） |
| `engine/mqtt_reporter.py` | MQTT 上报（IoTSharp） |

## 测试

纯 Python，**无需 NPU**，Mac 与板端均可跑（本机用 `/usr/local/bin/python3`）：

```bash
python3 test_office_events.py         # 离岗/睡岗判据（含 dict 格式回归）
python3 test_office_integration.py    # 五事件集成、互不串扰
python3 test_behavior_quality.py      # 四个误判现象的回归
python3 test_net_displacement.py      # 窗口净位移量纲
python3 test_security_events.py       # 入侵/逗留/跌倒事件
python3 test_fire_smoke_postprocess.py
python3 test_mqtt_reporter.py
python3 test_mqtt_live.py             # 需真实网络与令牌
```

## 部署时必须知道的几个坑

**① `cam/` 装到仓库根，不是 `rk_pose_authoritative/` 下。**
systemd 单元 `rk-cam-mjpeg.service` 的 `ExecStart` 硬编码
`/home/admin/rk_security_ai/cam/mjpeg_server.py`。曾误装到子目录造成重复副本。

**② 姿态模型必须显式指向 `models/yolov8n-pose.rknn`。**
`models/model.rknn` 是**检测模型副本**，用它会让关键点全部不达标、行为识别静默失效
（`Behaviors: 0`，不报错）。

**③ 离岗判定需要 `type=desk` 的区域才生效。**
当前板端区域配置只有 `type=both` 的 `z_all`，故离岗暂不会触发，需现场标定工位坐标。

**④ 关键点有"对象"与"dict"两种格式。**
`engine.py` 写入 `behaviors` 的是序列化 dict `{"x","y","v"}`（注意可见性是 **`v`**），
离线单测造的是 Landmark 对象（`.x` / `.visibility`）。新增判据必须两种都支持，
且**必须跑一次真机链路**——单测用理想格式，测不出格式不匹配导致的静默失效。

**⑤ 部署后需重启主进程才生效。** 用 `~/start_security.sh`（含单实例保护）。

**⑥ 摄像头凭据不再硬编码在源码里。**（2026-09-30 安全整改）
`cam/*.py` 原先明文写着摄像头账号密码并进了公开仓库，现已改为环境变量注入：

```bash
# 方式一：直接给完整地址
export CAM_RTSP="rtsp://<user>:<password>@<ip>:554/Streaming/Channels/102"
# 方式二：分开给三件套
export CAM_USER=<user> CAM_PASSWORD=<password> CAM_HOST=<ip>
```

板端由 `rk-cam-mjpeg.service` 的 `Environment=` 提供（单元文件权限 600）。
**缺凭据时会明确报错退出**，不再有硬编码兜底。`/status` 接口与启动横幅
也已脱敏（`rtsp://user:***@host/...`）。

> ⚠️ 历史提交 `aff2906` 中仍含旧密码。**用户已知情并决定暂不处置**（2026-09-30）——
> 摄像头在 `eth2` 专网、仅板端可达、未开 `ip_forward`，外部拿到凭据也路由不到该网段。
> 这是已做出的决策，不要再当作未决问题提出。将来若需彻底消除：① 摄像头侧改密
> （同步更新 `rk-cam-mjpeg.service` 的 `Environment=`）；② `git filter-repo` 重写历史。

## 目录归属

**本目录已纳入板端仓库**：`~/rk_security_ai/tools/security_patch/`（提交 `f6dd181`）。
板端是权威源，本地这份是便于编辑的工作副本；两者若不一致以板端为准。

## 板端重启方式

```bash
# 启动（脚本已固化全部 RK_* 环境变量，含 MQTT 令牌）
bash ~/start_security.sh

# 真实日志在这里（main_security.py 内部自己重定向）
tail -f ~/rk_security_ai/logs/security_ai.log
```
