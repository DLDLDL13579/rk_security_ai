# RK3588 多人行为识别 — 演示版运行手册（方案 C）

## 1. 演示配置（已固化）

```bash
RK_POSE_BACKEND=mediapipe        # 姿态后端：逐人裁剪（稳定版）
RK_POSE_FULLFRAME=1              # NPU 全帧路径（MediaPipe 自动不启用，代码就绪待 Pose 模型）
RK_FRAME_WIDTH/HEIGHT=640/360    # 输入分辨率
RK_CROP_PAD=0.12                 # 裁剪外扩 12%
RK_TRACKER_SMOOTH_ALPHA=0.65     # 轨迹 bbox 平滑
RK_OBJECT_SCORE=0.35             # 检测置信度阈值
```

Locomotion / Special 通道阈值已按实测数据收紧（详见 engine 默认值，均可
通过 `RK_LOCO_*` / `RK_SPECIAL_*` 覆盖）。

## 2. 一键演示

```bash
# 有显示器：窗口实时显示，ESC 退出
bash run_demo.sh

# 指定演示素材（多人合成视频 / RTSP）
bash run_demo.sh videos_multi/m1_stand_walk.mp4
bash run_demo.sh rtsp://user:pass@ip:554/ch01

# 无头模式：输出 output/behavior_result.mp4
bash run_demo.sh videos/standing/stand_002.mp4 --headless
```

## 3. 合成多人演示素材

```bash
python make_multi_person_video.py \
    --left videos/standing/stand_001.mp4 \
    --right videos/walking/walk_001.mp4 \
    --out videos_multi/m1_stand_walk.mp4 --flip-right --height 480
```

## 4. 评估与验收

```bash
python test_multi_behavior_eval.py \
    --source videos/standing/stand_002.mp4 --out output/eval_demo.csv
```

重点指标：

| 指标 | 说明 | 演示版参考值 |
| --- | --- | --- |
| max_concurrent | 同时检出人数（按检测框） | = 场景人数 |
| coverage | 行为帧/检测帧 | 100% |
| 站立者 standing 占比 | 稳定性的核心 | ≥95%，walking=0 |
| avg_flips | 每人行为翻转数 | 越低越好（演示版 ~5） |
| FPS | 实时性 | 单人 ~6，3 人 ~2.4（NPU 提速待 Pose 模型） |

## 5. 已知限制（演示时注意）

- 多人场景 FPS 约 2.4（MediaPipe 逐人裁剪）；NPU 全帧提速待真正的
  YOLOv5-Pose rknn 模型；
- 无 sit 类：坐立会判为 squat（合理近似）；
- 小目标/遮挡时 MediaPipe 下肢关键点不可靠，已由可见性门兜底。

## 6. 常见问题

- 画面空白：确认 `RK_SOURCE` 路径正确；RTSP 需要网络可达；
- 行为迟迟不出：行为需要约 1~2 秒序列积累（每人独立）；
- 想换姿态后端：`RK_POSE_BACKEND=rknn`（需 Pose 模型）。
