# rk_pose_authoritative（权威工程）

RK3588 多人实时行为识别工程的**唯一权威工作目录**。代码基于 `最新工程/rk_pose_project _8.7`，按本目录结构调整路径后整理而成；原 8.7 工程保留作为对照。

## 目录结构

```
rk_pose_authoritative/
├── app/                    # 程序入口（main_realtime.py）
├── engine/                 # 核心引擎（检测/跟踪/序列/特征/双通道行为）
├── npu/                    # RKNN 封装（yolo_rknn_v2、pose_rknn、behavior_rknn、mediapipe_pose）
├── models/                 # 推理模型（yolov5s-640-640.rknn、model.rknn、behavior_tcn.rknn）
├── rtsp/                   # 视频/RTSP 采集（stream.py，类 RTSPThread）
├── display/                # 显示与录像（display.py，类 DisplayThread）
├── tools/                  # 工具（模型转换 onnx_rknn/、多人视频合成、RTSP 录制）
├── test/                   # 测试（逻辑单测、模型单测、实时测试、样例图片）
├── eval/                   # 多人行为评估（test_multi_behavior_eval.py）
├── data/
│   └── videos/             # 测试视频（standing/walking/squat/bend/fall_down）
├── docs/                   # 运行手册（demo_runbook.md）
├── core/                   # 预留抽象层（当前主流程未使用）
├── output/                 # 输出目录（评估 CSV、无头录像）
├── run_demo.sh             # 一键演示
├── run_eval_all.sh         # 批量评估
└── requirements.txt        # 板端依赖
```

## 快速开始

```bash
# 板端（RK3588）
pip install -r requirements.txt
bash run_demo.sh                                  # 默认视频
bash run_demo.sh data/videos/standing/stand_002.mp4 --headless
bash run_demo.sh rtsp://user:pass@ip:554/ch01

# 评估
bash run_eval_all.sh                              # data/videos 全部 -> output/eval_all_*.csv

# 合成多人测试视频
python tools/make_multi_person_video.py \
    --left data/videos/standing/stand_001.mp4 \
    --right data/videos/walking/walk_001.mp4 \
    --out videos_multi/m1_stand_walk.mp4 --flip-right
```

## 与 8.7 的结构对照

| 8.7 原位置 | 权威工程位置 | 调整 |
|---|---|---|
| `main.py` | `app/main_realtime.py` | 入口下沉到 app/，自动把工程根加入 sys.path |
| `engine/rtsp_thread.py` | `rtsp/stream.py` | 采集线程移入 rtsp/ 包 |
| `engine/display_thread.py` | `display/display.py` | 显示线程移入 display/ 包 |
| `pose/mediapipe_pose.py` | `npu/mediapipe_pose.py` | 姿态后端统一放 npu/ |
| `npu/*.rknn` | `models/*.rknn` | 模型集中到 models/，代码默认路径已同步 |
| `videos/` | `data/videos/` | 数据目录统一收进 data/ |
| `onnx_rknn/` | `tools/onnx_rknn/` | 转换工具归入 tools/ |
| `test_core/` + 根测试脚本 | `test/` | 测试统一归入 test/ |
| `test_multi_behavior_eval.py` | `eval/` | 评估工具独立成 eval/ |
| `make_multi_person_video.py` / `record_rtsp.py` | `tools/` | 工具脚本归入 tools/ |

## 运行说明

- 所有默认路径基于工程根目录自动拼接，`RK_*` 环境变量仍可覆盖（配置表见 `docs/demo_runbook.md` 与《8.7 工程使用与维护手册》）。
- 行为模型默认从 `models/behavior_tcn.rknn` 加载；重新转换后运行 `tools/onnx_rknn/convert_behavior_tcn.py` 会自动部署到该位置。
- `core/` 为预留抽象层（跟踪器升级、多源适配），当前主流程不使用。

详细的使用方法、文件夹与代码关联、现存问题与排查，见 [docs/使用与维护说明.md](docs/使用与维护说明.md)（同目录有 Word 版 `使用与维护说明.docx`）。
